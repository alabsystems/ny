// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::DiffError;
use crate::onnx_proto;
use ndarray::ArrayD;
#[cfg(feature = "ort")]
use ndarray::IxDyn;
#[cfg(feature = "ort")]
use ort::session::Session;
use prost::Message;
use std::collections::HashMap;
use std::path::Path;
use tracing::debug;
#[cfg(feature = "ort")]
use tracing::{info, warn};

#[cfg(not(feature = "ort"))]
fn ort_unavailable<T>() -> Result<T, DiffError> {
    Err(DiffError::OrtUnavailable)
}

// --- Process-global ORT session cache (#ort-session-once) -------------------
//
// One `ny vnncomp` instance evaluates the SAME model through ONNX Runtime many
// times: the attack scorer, the ORT-guided PGD refinement, and every
// trusted-oracle witness re-confirmation. Committing a session runs the full
// ORT GraphTransformer optimization pipeline (tens of fusion passes) — pure,
// repeated waste, since the committed graph is IMMUTABLE and its numeric output
// depends only on (model bytes, optimization level, providers), all identical
// across those re-forwards. Caching the committed session keyed by the model
// file identity is therefore VALUE-IDENTICAL to rebuilding it: the same session
// on the same input yields bit-identical outputs (only the ORT input tensor and
// the run differ per call; the graph does not).
//
// Every caller here commits with the SAME configuration — `Session::builder()`
// with default optimization level and default providers — so the file identity
// (canonical path, size, mtime) fully keys the numeric behavior; the config is
// captured implicitly as a constant. Mirrors the VNN-LIB parse memo
// (`vnnlib/parser.rs`, #vnnlib-parse-once).
//
// `Session::run` takes `&mut self` in this `ort` version (2.0.0-rc.12), but
// `Session` is `Send + Sync`; the cached session is shared behind an
// `Arc<Mutex<Session>>` so the (serial, per-instance) re-forwards take the run
// lock in turn. Kill-switch: `NY_ORT_SESSION_CACHE=0` reverts to a fresh
// session per call (the A/B + escape hatch).
#[cfg(feature = "ort")]
type SessionCacheKey = (std::path::PathBuf, u64, Option<std::time::SystemTime>);

#[cfg(feature = "ort")]
static ORT_SESSION_CACHE: std::sync::OnceLock<
    std::sync::Mutex<HashMap<SessionCacheKey, std::sync::Arc<std::sync::Mutex<Session>>>>,
> = std::sync::OnceLock::new();

#[cfg(feature = "ort")]
fn session_cache_enabled() -> bool {
    !std::env::var("NY_ORT_SESSION_CACHE").is_ok_and(|v| v == "0")
}

/// File-identity cache key for `path`, or `None` (skip the cache) when the
/// metadata is unavailable. The canonical path unifies different spellings;
/// size + mtime invalidate on any content change the filesystem can observe.
#[cfg(feature = "ort")]
fn session_cache_key(path: &Path) -> Option<SessionCacheKey> {
    let meta = std::fs::metadata(path).ok()?;
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    Some((canon, meta.len(), meta.modified().ok()))
}

/// Commit a fresh ORT session from in-memory model bytes with the shared
/// default configuration (default optimization level + default providers). The
/// single build site every cached and uncached path routes through, so the
/// session configuration stays identical everywhere.
#[cfg(feature = "ort")]
fn build_session(model_bytes: &[u8]) -> Result<Session, DiffError> {
    Ok(Session::builder()?.commit_from_memory(model_bytes)?)
}

/// Return a shared committed session for the model at `path` (graph bytes
/// `model_bytes`, already gzip-decoded). Served from the process-global cache
/// when the file is unchanged; built and cached on a miss. Falls back to a
/// fresh (uncached) session when the cache is disabled or the file metadata is
/// unreadable. The build runs under the cache lock so a concurrent request for
/// the same key waits for and reuses the one build rather than duplicating it.
#[cfg(feature = "ort")]
fn cached_session(
    path: &Path,
    model_bytes: &[u8],
) -> Result<std::sync::Arc<std::sync::Mutex<Session>>, DiffError> {
    let key = if session_cache_enabled() {
        session_cache_key(path)
    } else {
        None
    };
    let Some(key) = key else {
        return Ok(std::sync::Arc::new(std::sync::Mutex::new(build_session(
            model_bytes,
        )?)));
    };
    let cache = ORT_SESSION_CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(existing) = guard.get(&key) {
        debug!(path = %path.display(), "Reusing cached ORT session (model unchanged)");
        return Ok(std::sync::Arc::clone(existing));
    }
    let session = std::sync::Arc::new(std::sync::Mutex::new(build_session(model_bytes)?));
    debug!(path = %path.display(), "Committed new ORT session (cache miss)");
    guard.insert(key, std::sync::Arc::clone(&session));
    Ok(session)
}

/// A reusable ONNX-Runtime forward pass: the session is built once and a flat input
/// is run repeatedly with cheap per-call overhead.
///
/// Used by the ORT-guided witness refinement in the VNN-COMP trusted-oracle gate,
/// which needs hundreds of forward evaluations during a bounded local search. The
/// input shape is derived from the ONNX graph (gzip-aware) via
/// [`read_input_shape_maybe_gzip`], so it is decoupled from ny's own graph loader —
/// the same trusted-oracle property the single-shot [`run_inference_bytes`] relies on.
#[cfg(feature = "ort")]
pub struct OrtForward {
    session: std::sync::Arc<std::sync::Mutex<Session>>,
    input_shape: Vec<usize>,
}

#[cfg(feature = "ort")]
impl OrtForward {
    /// Build a reusable forward from a model path (gzip-aware) and the expected flat
    /// input length, reading the input tensor shape straight from the ONNX protobuf.
    ///
    /// The committed session is served from the process-global session cache
    /// (#ort-session-once) keyed by the model file identity — value-identical to
    /// rebuilding it, and shared across every ORT re-forward of the same model in
    /// an instance. `NY_ORT_SESSION_CACHE=0` reverts to a fresh session per call.
    pub fn from_path(path: impl AsRef<Path>, expected_len: usize) -> Result<Self, DiffError> {
        let path = path.as_ref();
        let (model_bytes, input_shape) = read_input_shape_maybe_gzip(path, expected_len)?;
        let session = cached_session(path, &model_bytes)?;
        Ok(Self {
            session,
            input_shape,
        })
    }

    /// The flat input length this forward expects (product of the input shape).
    pub fn input_len(&self) -> usize {
        self.input_shape.iter().product()
    }

    /// Run a single forward on a flat input, returning the flattened first output.
    ///
    /// The input is reshaped to the declared `input_shape`; a length mismatch is a
    /// hard error rather than a silent reshape.
    pub fn run(&mut self, flat_input: &[f32]) -> Result<Vec<f32>, DiffError> {
        let expected: usize = self.input_shape.iter().product();
        if flat_input.len() != expected {
            return Err(DiffError::LoadError(format!(
                "OrtForward input length {} does not match declared shape {:?} ({} elements)",
                flat_input.len(),
                self.input_shape,
                expected
            )));
        }

        let input_tensor =
            ort::value::TensorRef::from_array_view((self.input_shape.as_slice(), flat_input))?;
        // `Session::run` needs `&mut self`; the cached session is shared behind a
        // Mutex, so take the run lock (uncontended in the serial per-instance
        // flow). The output data is copied out before the guard drops.
        let mut session = self.session.lock().unwrap_or_else(|p| p.into_inner());
        let outputs = session.run(ort::inputs![input_tensor])?;

        let (_name, output) = outputs
            .iter()
            .next()
            .ok_or_else(|| DiffError::LoadError("ONNX Runtime returned no outputs".to_string()))?;
        let (_shape, data) = output.try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }
}

/// Non-ort stub: `OrtForward` exists so callers compile without the feature, but every
/// constructor returns [`DiffError::OrtUnavailable`].
#[cfg(not(feature = "ort"))]
pub struct OrtForward {
    _input_shape: Vec<usize>,
}

#[cfg(not(feature = "ort"))]
impl OrtForward {
    pub fn from_path(_path: impl AsRef<Path>, _expected_len: usize) -> Result<Self, DiffError> {
        ort_unavailable()
    }

    pub fn input_len(&self) -> usize {
        self._input_shape.iter().product()
    }

    pub fn run(&mut self, _flat_input: &[f32]) -> Result<Vec<f32>, DiffError> {
        ort_unavailable()
    }
}

/// Run inference on a single model and get outputs.
///
/// This runs the model and returns the final output(s).
#[cfg(feature = "ort")]
pub fn run_inference(
    path: impl AsRef<Path>,
    input: &ArrayD<f32>,
) -> Result<Vec<ArrayD<f32>>, DiffError> {
    // Create session
    let mut session = Session::builder()?.commit_from_file(path.as_ref())?;

    // Convert input to ort tensor
    let input_shape: Vec<usize> = input.shape().to_vec();
    let input_data: Vec<f32> = input.iter().cloned().collect();

    let input_tensor =
        ort::value::TensorRef::from_array_view((input_shape.as_slice(), input_data.as_slice()))?;

    // Run inference
    let outputs = session.run(ort::inputs![input_tensor])?;

    // Extract outputs as ArrayD
    let mut result = Vec::new();
    for (_name, output) in outputs.iter() {
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        // #2983: ORT shapes are i64; use try_from to reject negative dimensions.
        let shape_vec: Vec<usize> = shape
            .iter()
            .map(|&d| usize::try_from(d).unwrap_or(1))
            .collect();
        let array = ArrayD::from_shape_vec(IxDyn(&shape_vec), data.to_vec())
            .map_err(|e| DiffError::LoadError(format!("Shape error: {}", e)))?;
        result.push(array);
    }

    Ok(result)
}

/// Run inference on an in-memory ONNX model and get outputs.
#[cfg(not(feature = "ort"))]
pub fn run_inference(
    _path: impl AsRef<Path>,
    _input: &ArrayD<f32>,
) -> Result<Vec<ArrayD<f32>>, DiffError> {
    ort_unavailable()
}

/// Run inference on an in-memory ONNX model and get outputs.
#[cfg(feature = "ort")]
pub fn run_inference_bytes(
    model_bytes: &[u8],
    input: &ArrayD<f32>,
) -> Result<Vec<ArrayD<f32>>, DiffError> {
    let mut session = Session::builder()?.commit_from_memory(model_bytes)?;

    let input_shape: Vec<usize> = input.shape().to_vec();
    let input_data: Vec<f32> = input.iter().cloned().collect();

    let input_tensor =
        ort::value::TensorRef::from_array_view((input_shape.as_slice(), input_data.as_slice()))?;

    let outputs = session.run(ort::inputs![input_tensor])?;

    let mut result = Vec::new();
    for (_name, output) in outputs.iter() {
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        // #2983: ORT shapes are i64; use try_from to reject negative dimensions.
        let shape_vec: Vec<usize> = shape
            .iter()
            .map(|&d| usize::try_from(d).unwrap_or(1))
            .collect();
        let array = ArrayD::from_shape_vec(IxDyn(&shape_vec), data.to_vec())
            .map_err(|e| DiffError::LoadError(format!("Shape error: {}", e)))?;
        result.push(array);
    }

    Ok(result)
}

/// Create a modified ONNX model with all intermediate tensors exposed as outputs.
#[cfg(not(feature = "ort"))]
pub fn run_inference_bytes(
    _model_bytes: &[u8],
    _input: &ArrayD<f32>,
) -> Result<Vec<ArrayD<f32>>, DiffError> {
    ort_unavailable()
}

/// Read ONNX model bytes (gzip-aware) and derive the graph's first activation-input
/// tensor shape directly from the protobuf.
///
/// Returns `(model_bytes, input_shape)`. The shape is read straight from the graph
/// input `ValueInfoProto` — NOT via the higher-level graph loader — so a trusted
/// ONNX-Runtime forward can be set up independently of any graph-loading/op bug in
/// the verifier's own model conversion (the cgan_2023 false-counterexample root
/// cause). Symbolic / dynamic / non-positive dims (e.g. a symbolic batch axis) are
/// resolved to `1`. When the declared element count disagrees with `expected_len`
/// (or no shape is declared), a flat `[expected_len]` shape is returned so the
/// witness can still be presented to ORT, which derives its tensor shape from the
/// `ArrayD` shape and will reject a genuinely incompatible layout.
fn declared_shape_element_count(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |elements, &dimension| {
        elements.checked_mul(dimension)
    })
}

pub fn read_input_shape_maybe_gzip(
    path: impl AsRef<Path>,
    expected_len: usize,
) -> Result<(Vec<u8>, Vec<usize>), DiffError> {
    let model_bytes = crate::io::read_bytes_maybe_gzip(path.as_ref())
        .map_err(|e| DiffError::LoadError(format!("{e}")))?;

    let model = onnx_proto::ModelProto::decode(model_bytes.as_slice())
        .map_err(|e| DiffError::LoadError(format!("Failed to parse ONNX protobuf: {e}")))?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| DiffError::LoadError("ONNX model has no graph".to_string()))?;

    // Initializers can appear in `graph.input` in older opsets; the activation input
    // is the first graph input that is not also an initializer.
    let initializer_names: std::collections::HashSet<&str> =
        graph.initializer.iter().map(|i| i.name.as_str()).collect();
    let value_input = graph
        .input
        .iter()
        .find(|i| !initializer_names.contains(i.name.as_str()))
        .or_else(|| graph.input.first());

    let shape: Vec<usize> = value_input
        .and_then(|vi| vi.r#type.as_ref())
        .and_then(|t| t.tensor_type.as_ref())
        .and_then(|tt| tt.shape.as_ref())
        .map(|s| {
            s.dim
                .iter()
                .map(|d| match d.value {
                    Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(v))
                        if v > 0 =>
                    {
                        v as usize
                    }
                    // Dynamic / symbolic / non-positive dim -> single example.
                    _ => 1usize,
                })
                .collect()
        })
        .unwrap_or_default();

    let declared = declared_shape_element_count(&shape);
    if shape.is_empty() || declared != Some(expected_len) {
        return Ok((model_bytes, vec![expected_len]));
    }
    Ok((model_bytes, shape))
}

#[cfg(test)]
mod input_shape_tests {
    use super::declared_shape_element_count;

    #[test]
    fn declared_element_count_overflow_falls_back() {
        assert_eq!(declared_shape_element_count(&[usize::MAX, 2]), None);
        assert_eq!(declared_shape_element_count(&[2, 3, 4]), Some(24));
    }
}

/// Create a modified ONNX model with all intermediate tensors exposed as outputs.
///
/// This function parses the ONNX protobuf, adds all node outputs to the graph outputs,
/// and returns the modified model bytes.
pub fn expose_intermediate_outputs(model_bytes: &[u8]) -> Result<Vec<u8>, DiffError> {
    // Parse the ONNX model
    let mut model = onnx_proto::ModelProto::decode(model_bytes)
        .map_err(|e| DiffError::LoadError(format!("Failed to parse ONNX protobuf: {}", e)))?;

    let graph = model
        .graph
        .as_mut()
        .ok_or_else(|| DiffError::LoadError("ONNX model has no graph".to_string()))?;

    // Collect existing output names to avoid duplicates
    let existing_outputs: std::collections::HashSet<String> =
        graph.output.iter().map(|o| o.name.clone()).collect();

    // Collect all input names (to skip adding them as outputs)
    let input_names: std::collections::HashSet<String> =
        graph.input.iter().map(|i| i.name.clone()).collect();

    // Collect initializer names (weights - skip these too)
    let initializer_names: std::collections::HashSet<String> =
        graph.initializer.iter().map(|i| i.name.clone()).collect();

    // Add all node outputs that aren't already graph outputs, inputs, or initializers
    let mut new_outputs = Vec::new();
    for node in &graph.node {
        for output_name in &node.output {
            if output_name.is_empty() {
                continue;
            }
            if existing_outputs.contains(output_name) {
                continue;
            }
            if input_names.contains(output_name) {
                continue;
            }
            if initializer_names.contains(output_name) {
                continue;
            }

            // Create a ValueInfoProto for this output
            // We don't set the type info - ONNX Runtime will infer it
            new_outputs.push(onnx_proto::ValueInfoProto {
                name: output_name.clone(),
                r#type: None,
            });
        }
    }

    debug!("Adding {} intermediate outputs to model", new_outputs.len());

    graph.output.extend(new_outputs);

    // Serialize back to bytes
    let mut buf = Vec::new();
    model.encode(&mut buf).map_err(|e| {
        DiffError::LoadError(format!("Failed to serialize modified ONNX model: {}", e))
    })?;

    Ok(buf)
}

/// Run inference and capture all intermediate outputs.
///
/// This modifies the ONNX graph to expose intermediate tensors as outputs,
/// runs inference, and returns all intermediate values keyed by tensor name.
#[cfg(feature = "ort")]
pub fn run_inference_with_intermediates(
    path: impl AsRef<Path>,
    input: &ArrayD<f32>,
) -> Result<HashMap<String, ArrayD<f32>>, DiffError> {
    // Read the original model bytes
    let model_bytes = crate::io::read_bytes_maybe_gzip(path.as_ref())
        .map_err(|e| DiffError::LoadError(format!("{e}")))?;

    // Modify graph to expose all intermediate outputs
    let modified_bytes = expose_intermediate_outputs(&model_bytes)?;

    // Create session from modified model bytes
    let mut session = Session::builder()?.commit_from_memory(&modified_bytes)?;

    let input_shape: Vec<usize> = input.shape().to_vec();
    let input_data: Vec<f32> = input.iter().cloned().collect();

    let input_tensor =
        ort::value::TensorRef::from_array_view((input_shape.as_slice(), input_data.as_slice()))?;

    let outputs = session.run(ort::inputs![input_tensor])?;

    let mut result = HashMap::new();

    // Extract all outputs with their names
    for (name, output) in outputs.iter() {
        match output.try_extract_tensor::<f32>() {
            Ok((shape, data)) => {
                // #2983: ORT shapes are i64; use try_from to reject negative dimensions.
                let shape_vec: Vec<usize> = shape
                    .iter()
                    .map(|&d| usize::try_from(d).unwrap_or(1))
                    .collect();
                match ArrayD::from_shape_vec(IxDyn(&shape_vec), data.to_vec()) {
                    Ok(array) => {
                        result.insert(name.to_string(), array);
                    }
                    Err(e) => {
                        warn!("Failed to convert tensor {} to array: {}", name, e);
                    }
                }
            }
            Err(e) => {
                // Some tensors may not be f32 (e.g., shape tensors are i64)
                debug!("Skipping non-f32 tensor {}: {}", name, e);
            }
        }
    }

    info!("Extracted {} intermediate outputs from model", result.len());

    Ok(result)
}

/// Run inference on in-memory ONNX bytes and capture intermediate outputs.
#[cfg(not(feature = "ort"))]
pub fn run_inference_with_intermediates(
    _path: impl AsRef<Path>,
    _input: &ArrayD<f32>,
) -> Result<HashMap<String, ArrayD<f32>>, DiffError> {
    ort_unavailable()
}

/// Run inference on in-memory ONNX bytes and capture intermediate outputs.
#[cfg(feature = "ort")]
pub fn run_inference_with_intermediates_bytes(
    model_bytes: &[u8],
    input: &ArrayD<f32>,
) -> Result<HashMap<String, ArrayD<f32>>, DiffError> {
    let modified_bytes = expose_intermediate_outputs(model_bytes)?;

    let mut session = Session::builder()?.commit_from_memory(&modified_bytes)?;

    let input_shape: Vec<usize> = input.shape().to_vec();
    let input_data: Vec<f32> = input.iter().cloned().collect();

    let input_tensor =
        ort::value::TensorRef::from_array_view((input_shape.as_slice(), input_data.as_slice()))?;

    let outputs = session.run(ort::inputs![input_tensor])?;

    let mut result = HashMap::new();

    for (name, output) in outputs.iter() {
        match output.try_extract_tensor::<f32>() {
            Ok((shape, data)) => {
                // #2983: ORT shapes are i64; use try_from to reject negative dimensions.
                let shape_vec: Vec<usize> = shape
                    .iter()
                    .map(|&d| usize::try_from(d).unwrap_or(1))
                    .collect();
                match ArrayD::from_shape_vec(IxDyn(&shape_vec), data.to_vec()) {
                    Ok(array) => {
                        result.insert(name.to_string(), array);
                    }
                    Err(e) => {
                        warn!("Failed to convert tensor {} to array: {}", name, e);
                    }
                }
            }
            Err(e) => {
                debug!("Skipping non-f32 tensor {}: {}", name, e);
            }
        }
    }

    info!(
        "Extracted {} intermediate outputs from in-memory model",
        result.len()
    );

    Ok(result)
}

#[cfg(not(feature = "ort"))]
pub fn run_inference_with_intermediates_bytes(
    _model_bytes: &[u8],
    _input: &ArrayD<f32>,
) -> Result<HashMap<String, ArrayD<f32>>, DiffError> {
    ort_unavailable()
}

// --- Session cache soundness tests (#ort-session-once) -----------------------
//
// A caching bug that returned a WRONG session (stale after a model change, or a
// different graph under the same key) could feed the trusted-oracle gate a
// wrong ORT output and let a false `sat` through, so the cache must be
// value-identical to per-call construction. These tests assert exactly that:
// bit-identical outputs, one shared session per file identity, and invalidation
// on a metadata change.
#[cfg(all(test, feature = "ort"))]
mod cache_tests {
    use super::*;
    use crate::test_fixtures::require_test_model;
    use std::sync::{Arc, Mutex};

    // `simple_mlp.onnx` declares a `[1, 2]` input.
    const INPUT_LEN: usize = 2;
    const INPUT: [f32; INPUT_LEN] = [0.371_25_f32, -0.913_5_f32];

    // The cache and the kill-switch env var are process-global; serialize the
    // cache tests so a parallel env flip can't perturb another test's lookup.
    static CACHE_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn model_bytes() -> (std::path::PathBuf, Vec<u8>, Vec<usize>) {
        let path = require_test_model("simple_mlp.onnx");
        let (bytes, shape) = read_input_shape_maybe_gzip(&path, INPUT_LEN).expect("read shape");
        (path, bytes, shape)
    }

    fn run_shared(session: Arc<Mutex<Session>>, shape: Vec<usize>) -> Vec<f32> {
        // Construct through the private OrtForward fields (this module is a
        // child of `inference`) so the run path is byte-for-byte the real one.
        let mut fwd = OrtForward {
            session,
            input_shape: shape,
        };
        fwd.run(&INPUT).expect("ort run")
    }

    fn ptr_of(a: &Arc<Mutex<Session>>) -> usize {
        Arc::as_ptr(a) as usize
    }

    #[test]
    fn cached_session_output_is_bit_identical_to_a_fresh_build() {
        let _g = CACHE_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (path, bytes, shape) = model_bytes();

        // Fresh, uncached session (the pre-cache behavior).
        let fresh = Arc::new(Mutex::new(build_session(&bytes).expect("fresh build")));
        let fresh_out = run_shared(Arc::clone(&fresh), shape.clone());

        // Cached session (default path).
        let cached = cached_session(&path, &bytes).expect("cached");
        let cached_out = run_shared(Arc::clone(&cached), shape.clone());
        // A second, independent forward that MUST reuse the same session.
        let cached_again = cached_session(&path, &bytes).expect("cached again");
        let cached_out2 = run_shared(cached_again, shape);

        assert!(!fresh_out.is_empty(), "model produced no output");
        assert_eq!(fresh_out.len(), cached_out.len());
        for (f, c) in fresh_out.iter().zip(cached_out.iter()) {
            assert_eq!(
                f.to_bits(),
                c.to_bits(),
                "cached session output diverged from a fresh build at the BIT level: {f} vs {c}"
            );
        }
        assert_eq!(
            cached_out, cached_out2,
            "reused session must be deterministic"
        );
    }

    #[test]
    fn same_file_identity_shares_one_session() {
        let _g = CACHE_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (path, bytes, _shape) = model_bytes();
        let a = cached_session(&path, &bytes).expect("a");
        let b = cached_session(&path, &bytes).expect("b");
        assert_eq!(
            ptr_of(&a),
            ptr_of(&b),
            "two lookups for the same unchanged model must return the SAME cached session"
        );
    }

    #[test]
    fn distinct_paths_get_distinct_sessions() {
        let _g = CACHE_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (_path, bytes, _shape) = model_bytes();
        let f1 = tempfile::NamedTempFile::new().expect("t1");
        let f2 = tempfile::NamedTempFile::new().expect("t2");
        std::fs::write(f1.path(), &bytes).expect("w1");
        std::fs::write(f2.path(), &bytes).expect("w2");
        let s1 = cached_session(f1.path(), &bytes).expect("s1");
        let s2 = cached_session(f2.path(), &bytes).expect("s2");
        assert_ne!(
            ptr_of(&s1),
            ptr_of(&s2),
            "distinct model paths must not collide on one cached session"
        );
    }

    #[test]
    fn metadata_change_invalidates_the_cached_session() {
        let _g = CACHE_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (_path, bytes, _shape) = model_bytes();
        let f = tempfile::NamedTempFile::new().expect("tmp");
        std::fs::write(f.path(), &bytes).expect("write v1");
        let before = cached_session(f.path(), &bytes).expect("before");

        // Change the file's observable identity (length). The key is (canonical
        // path, len, mtime); a length change alone must force a rebuild.
        let mut grown = bytes.clone();
        grown.extend_from_slice(&[0u8; 8]);
        std::fs::write(f.path(), &grown).expect("write v2");
        // Build from the ORIGINAL valid bytes so the commit still succeeds; the
        // point is that the CACHE KEY changed, not the graph.
        let after = cached_session(f.path(), &bytes).expect("after");

        assert_ne!(
            ptr_of(&before),
            ptr_of(&after),
            "a metadata (length) change must invalidate and rebuild the cached session"
        );
    }

    #[test]
    fn kill_switch_bypasses_the_cache() {
        let _g = CACHE_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (path, bytes, _shape) = model_bytes();
        // With the cache disabled, every call builds a brand-new session.
        // (Serialized + restored via the blessed env choke point.)
        let (a, b) =
            ny_test_utils::env::with_serialized_env_vars(&[("NY_ORT_SESSION_CACHE", "0")], || {
                let a = cached_session(&path, &bytes).expect("a");
                let b = cached_session(&path, &bytes).expect("b");
                (a, b)
            });
        assert_ne!(
            ptr_of(&a),
            ptr_of(&b),
            "NY_ORT_SESSION_CACHE=0 must build a fresh session per call"
        );
    }
}
