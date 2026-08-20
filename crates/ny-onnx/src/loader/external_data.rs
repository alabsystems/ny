// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Secure, model-origin-relative ONNX external tensor data loading.
//!
//! ONNX external locations are untrusted paths authored by the model. They are
//! resolved through a directory capability rooted at the model's parent, never
//! through ambient `File::open(base.join(location))`. Strict lexical checks and
//! per-component symlink rejection provide clear diagnostics; cap-std's
//! capability-relative open is the containment boundary if the directory is
//! concurrently modified.

use super::tensor::expected_raw_data_byte_len;
use crate::onnx_proto::{self, TensorProto};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptions, OpenOptionsFollowExt, OpenOptionsSyncExt};
use cap_std::ambient_authority;
use cap_std::fs::{Dir, File};
use flate2::read::GzDecoder;
use ny_core::{NyError, Result};
use sha1::{Digest, Sha1};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

const DATA_LOCATION_DEFAULT: i32 = 0;
const DATA_LOCATION_EXTERNAL: i32 = 1;
const HASH_BUFFER_BYTES: usize = 64 * 1024;

#[derive(Debug)]
struct ExternalDataSpec {
    location: PathBuf,
    offset: u64,
    length: Option<u64>,
    checksum: Option<[u8; 20]>,
}

struct CachedExternalFile {
    file: File,
    len: u64,
    sha1: Option<[u8; 20]>,
}

/// Resolver anchored to the directory containing one ONNX model file.
pub(super) struct ExternalDataResolver {
    base_dir: Dir,
    base_display: PathBuf,
    /// File name of the model INSIDE `base_dir`, after resolving the operator's
    /// path (which may be a symlink) to its real location.
    model_file_name: PathBuf,
    files: HashMap<PathBuf, CachedExternalFile>,
}

impl ExternalDataResolver {
    pub(super) fn for_model_path(model_path: &Path) -> Result<Self> {
        // Resolve the OPERATOR-supplied model path — including any symlink on
        // it — before anchoring the capability, then anchor at the directory
        // that really holds the file.
        //
        // The path of the model is trusted input: it comes from the CLI or a
        // shipped preset, never from model bytes. What 38a2fecf's capability
        // scope exists to contain is the untrusted `external_data` `location`
        // strings AUTHORED INSIDE the model, and that containment is unchanged
        // — every sidecar is still opened through this one `Dir`, with lexical
        // checks and per-component symlink rejection, and can still only name a
        // file in the model's own directory.
        //
        // Anchoring at the symlink's parent instead is what broke nn4sys:
        // `benchmarks/.../nn4sys/onnx/mscn_2048d.onnx` is a symlink to
        // `../../nn4sys_2023/onnx/mscn_2048d.onnx`, so cap-std refused the
        // model itself with "a path led outside of the filesystem" and 34 of
        // the category's 194 instances became MODEL-LOAD-FAILURE. Resolving
        // first also matches ONNX's own rule that an external `location` is
        // relative to the model file: relative to the REAL file is the only
        // reading under which the sidecars a model ships with are found.
        let resolved = std::fs::canonicalize(model_path).map_err(|err| {
            NyError::ModelLoad(format!(
                "failed to resolve ONNX model path {}: {err}",
                model_path.display()
            ))
        })?;
        let model_file_name = resolved
            .file_name()
            .ok_or_else(|| {
                NyError::ModelLoad(format!(
                    "ONNX model path {} does not name a file",
                    model_path.display()
                ))
            })?
            .to_owned();
        let parent = resolved.parent().unwrap_or_else(|| Path::new("."));
        let parent = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        let base_dir = Dir::open_ambient_dir(parent, ambient_authority()).map_err(|err| {
            NyError::ModelLoad(format!(
                "failed to open ONNX model directory {} for external data: {err}",
                parent.display()
            ))
        })?;
        Ok(Self {
            base_dir,
            base_display: parent.to_path_buf(),
            model_file_name: PathBuf::from(model_file_name),
            files: HashMap::new(),
        })
    }

    /// Read the model through the same retained directory capability that will
    /// later resolve its external data. This prevents a directory rename or a
    /// process-wide current-directory change from binding the protobuf bytes
    /// and sidecars to different origins.
    pub(super) fn read_model_bytes(&self, model_path: &Path) -> Result<Vec<u8>> {
        // The name resolved in `for_model_path`, i.e. the real file inside
        // `base_dir` — NOT `model_path.file_name()`, which may be a symlink
        // pointing out of that directory and would be refused by the
        // capability open.
        let file_name = self.model_file_name.as_path();
        let mut options = OpenOptions::new();
        options.read(true).nonblock(true);
        let file = self
            .base_dir
            .open_with(file_name, &options)
            .map_err(|err| {
                NyError::ModelLoad(format!(
                    "failed to open ONNX model {} through its retained directory capability: {err}",
                    model_path.display()
                ))
            })?;
        let metadata = file.metadata().map_err(|err| {
            NyError::ModelLoad(format!(
                "failed to inspect ONNX model {}: {err}",
                model_path.display()
            ))
        })?;
        if !metadata.is_file() {
            return Err(NyError::ModelLoad(format!(
                "ONNX model path {} is not a regular file",
                model_path.display()
            )));
        }

        let mut data = Vec::new();
        // Decide gzip from the RESOLVED name, which is the file actually being
        // read; a symlink is free to be spelled without the `.gz` suffix.
        if file_name
            .extension()
            .and_then(|extension| extension.to_str())
            == Some("gz")
        {
            GzDecoder::new(file)
                .read_to_end(&mut data)
                .map_err(|err| NyError::ModelLoad(format!("Failed to decode gzip: {err}")))?;
        } else {
            let mut file = file;
            file.read_to_end(&mut data)
                .map_err(|err| NyError::ModelLoad(format!("Failed to read file: {err}")))?;
        }
        Ok(data)
    }

    /// Validate every modeled TensorProto before any ONNX Runtime call.
    ///
    /// This is deliberately separate from `read_tensor`: validation prevents
    /// an untrusted external path from reaching ORT before ny has applied its
    /// own filesystem containment policy.
    pub(super) fn validate_model(&mut self, model: &onnx_proto::ModelProto) -> Result<bool> {
        let Some(graph) = model.graph.as_ref() else {
            return Ok(false);
        };
        reject_sparse_initializers(graph)?;
        let mut found_external = false;
        for tensor in &graph.initializer {
            found_external |= self.validate_tensor(tensor)?;
        }
        for node in &graph.node {
            for attribute in &node.attribute {
                reject_unmodeled_tensor_container(attribute, node)?;
                if let Some(tensor) = attribute.t.as_ref() {
                    found_external |= self.validate_tensor(tensor)?;
                }
            }
        }
        Ok(found_external)
    }

    /// Read one validated external tensor slice. Inline tensors return `None`.
    pub(super) fn read_tensor(&mut self, tensor: &TensorProto) -> Result<Option<Vec<u8>>> {
        let Some(spec) = parse_external_spec(tensor)? else {
            return Ok(None);
        };
        let expected_len = expected_raw_data_byte_len(tensor)?;
        let expected_len_u64 = u64::try_from(expected_len).map_err(|_| {
            NyError::ModelLoad(format!(
                "Tensor {} external payload length does not fit u64",
                tensor.name
            ))
        })?;
        let file = self.open_external_file(&spec.location, &tensor.name)?;
        validate_open_file_bounds(file.len, &spec, expected_len_u64, &tensor.name)?;
        if let Some(expected_checksum) = spec.checksum {
            verify_checksum(file, expected_checksum, &tensor.name, &spec.location)?;
        }

        file.file
            .seek(SeekFrom::Start(spec.offset))
            .map_err(|err| {
                NyError::ModelLoad(format!(
                    "Tensor {} failed to seek external data {} to offset {}: {err}",
                    tensor.name,
                    spec.location.display(),
                    spec.offset
                ))
            })?;
        let mut payload = Vec::new();
        payload.try_reserve_exact(expected_len).map_err(|err| {
            NyError::ModelLoad(format!(
                "Tensor {} cannot allocate {} external-data bytes: {err}",
                tensor.name, expected_len
            ))
        })?;
        payload.resize(expected_len, 0);
        file.file.read_exact(&mut payload).map_err(|err| {
            NyError::ModelLoad(format!(
                "Tensor {} failed to read {} bytes from external data {} at offset {}: {err}",
                tensor.name,
                expected_len,
                spec.location.display(),
                spec.offset
            ))
        })?;
        Ok(Some(payload))
    }

    /// Attribute tensors are consumed at several later constant-folding sites.
    /// Materialize only those uncommon payloads once; graph initializers use
    /// the streaming `read_tensor` path and never retain a second raw copy.
    pub(super) fn materialize_attribute_tensors(
        &mut self,
        nodes: &mut [onnx_proto::NodeProto],
    ) -> Result<()> {
        for node in nodes {
            for attribute in &mut node.attribute {
                let Some(tensor) = attribute.t.as_mut() else {
                    continue;
                };
                let Some(payload) = self.read_tensor(tensor)? else {
                    continue;
                };
                tensor.raw_data = payload;
                tensor.external_data.clear();
                tensor.data_location = DATA_LOCATION_DEFAULT;
            }
        }
        Ok(())
    }

    fn validate_tensor(&mut self, tensor: &TensorProto) -> Result<bool> {
        let Some(spec) = parse_external_spec(tensor)? else {
            return Ok(false);
        };
        let expected_len = u64::try_from(expected_raw_data_byte_len(tensor)?).map_err(|_| {
            NyError::ModelLoad(format!(
                "Tensor {} external payload length does not fit u64",
                tensor.name
            ))
        })?;
        let file = self.open_external_file(&spec.location, &tensor.name)?;
        validate_open_file_bounds(file.len, &spec, expected_len, &tensor.name)?;
        if let Some(expected_checksum) = spec.checksum {
            verify_checksum(file, expected_checksum, &tensor.name, &spec.location)?;
        }
        Ok(true)
    }

    fn open_external_file(
        &mut self,
        location: &Path,
        tensor_name: &str,
    ) -> Result<&mut CachedExternalFile> {
        if !self.files.contains_key(location) {
            let file =
                open_regular_file_no_symlinks(&self.base_dir, location, tensor_name).map_err(
                    |err| {
                        NyError::ModelLoad(format!(
                            "Tensor {tensor_name} failed to open external data {} relative to model directory {}: {err}",
                            location.display(),
                            self.base_display.display()
                        ))
                    },
                )?;
            let metadata = file.metadata().map_err(|err| {
                NyError::ModelLoad(format!(
                    "Tensor {tensor_name} failed to inspect external data {}: {err}",
                    location.display()
                ))
            })?;
            self.files.insert(
                location.to_path_buf(),
                CachedExternalFile {
                    file,
                    len: metadata.len(),
                    sha1: None,
                },
            );
        }
        self.files.get_mut(location).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "internal error caching external data {}",
                location.display()
            ))
        })
    }
}

/// In-memory ONNX APIs intentionally have no ambient filesystem authority.
pub(super) fn reject_external_data_without_origin(model: &onnx_proto::ModelProto) -> Result<()> {
    let Some(graph) = model.graph.as_ref() else {
        return Ok(());
    };
    reject_sparse_initializers(graph)?;
    for tensor in &graph.initializer {
        reject_tensor_without_origin(tensor)?;
    }
    for node in &graph.node {
        for attribute in &node.attribute {
            reject_unmodeled_tensor_container(attribute, node)?;
            if let Some(tensor) = attribute.t.as_ref() {
                reject_tensor_without_origin(tensor)?;
            }
        }
    }
    Ok(())
}

fn reject_sparse_initializers(graph: &onnx_proto::GraphProto) -> Result<()> {
    if graph.sparse_initializer.is_empty() {
        return Ok(());
    }
    Err(NyError::ModelLoad(format!(
        "ONNX graph contains {} sparse initializer(s), but NY does not support sparse tensor storage; refusing to discard model parameters",
        graph.sparse_initializer.len()
    )))
}

fn reject_unmodeled_tensor_container(
    attribute: &onnx_proto::AttributeProto,
    node: &onnx_proto::NodeProto,
) -> Result<()> {
    let has_tensor = attribute.t.is_some();
    if has_tensor != (attribute.r#type == onnx_proto::attribute_type::TENSOR) {
        return Err(NyError::ModelLoad(format!(
            "ONNX attribute '{}' on node '{}' has inconsistent tensor payload/type metadata",
            attribute.name, node.name
        )));
    }
    // AttributeProto types whose payload can recursively contain TensorProto
    // values are not represented by NY's deliberately small protobuf schema.
    // Reject them by their retained type tag instead of allowing prost to drop
    // a graph/repeated/sparse tensor payload before external-data validation.
    if matches!(attribute.r#type, 5 | 9..=12) {
        return Err(NyError::ModelLoad(format!(
            "ONNX attribute '{}' on node '{}' uses unsupported tensor-container type {}; nested, repeated, and sparse tensor containers are rejected before loading",
            attribute.name, node.name, attribute.r#type
        )));
    }
    Ok(())
}

fn reject_tensor_without_origin(tensor: &TensorProto) -> Result<()> {
    match tensor.data_location {
        DATA_LOCATION_DEFAULT => {
            if tensor.external_data.is_empty() {
                Ok(())
            } else {
                Err(NyError::ModelLoad(format!(
                    "Tensor {} has external_data metadata but data_location is DEFAULT",
                    tensor.name
                )))
            }
        }
        DATA_LOCATION_EXTERNAL => Err(NyError::ModelLoad(format!(
            "Tensor {} uses ONNX external data, but an in-memory byte load has no model-origin directory; use load_onnx or load_onnx_with_config with the model file path",
            tensor.name
        ))),
        other => Err(NyError::ModelLoad(format!(
            "Tensor {} has unknown ONNX data_location {}",
            tensor.name, other
        ))),
    }
}

fn parse_external_spec(tensor: &TensorProto) -> Result<Option<ExternalDataSpec>> {
    match tensor.data_location {
        DATA_LOCATION_DEFAULT => {
            if tensor.external_data.is_empty() {
                return Ok(None);
            }
            return Err(NyError::ModelLoad(format!(
                "Tensor {} has external_data metadata but data_location is DEFAULT",
                tensor.name
            )));
        }
        DATA_LOCATION_EXTERNAL => {}
        other => {
            return Err(NyError::ModelLoad(format!(
                "Tensor {} has unknown ONNX data_location {}",
                tensor.name, other
            )));
        }
    }

    let populated_inline_fields: Vec<&str> = [
        tensor.segment.is_some().then_some("segment"),
        (!tensor.raw_data.is_empty()).then_some("raw_data"),
        (!tensor.float_data.is_empty()).then_some("float_data"),
        (!tensor.int32_data.is_empty()).then_some("int32_data"),
        (!tensor.int64_data.is_empty()).then_some("int64_data"),
        (!tensor.double_data.is_empty()).then_some("double_data"),
        (!tensor.string_data.is_empty()).then_some("string_data"),
        (!tensor.uint64_data.is_empty()).then_some("uint64_data"),
    ]
    .into_iter()
    .flatten()
    .collect();
    if !populated_inline_fields.is_empty() {
        return Err(NyError::ModelLoad(format!(
            "Tensor {} is EXTERNAL but also populates inline field(s): {}",
            tensor.name,
            populated_inline_fields.join(", ")
        )));
    }

    let mut seen = HashSet::new();
    let mut location = None;
    let mut offset = None;
    let mut length = None;
    let mut checksum = None;
    for entry in &tensor.external_data {
        if !seen.insert(entry.key.as_str()) {
            return Err(NyError::ModelLoad(format!(
                "Tensor {} has duplicate external_data key '{}'",
                tensor.name, entry.key
            )));
        }
        match entry.key.as_str() {
            "location" => location = Some(validate_location(&entry.value, &tensor.name)?),
            "offset" => offset = Some(parse_decimal_u64("offset", &entry.value, &tensor.name)?),
            "length" => length = Some(parse_decimal_u64("length", &entry.value, &tensor.name)?),
            "checksum" => checksum = Some(parse_sha1(&entry.value, &tensor.name)?),
            // ONNX tooling may add basepath after loading. It is metadata only:
            // an authored value never overrides the actual model origin.
            "basepath" => {}
            other => {
                return Err(NyError::ModelLoad(format!(
                    "Tensor {} has unknown external_data key '{}'",
                    tensor.name, other
                )));
            }
        }
    }
    let location = location.ok_or_else(|| {
        NyError::ModelLoad(format!(
            "Tensor {} external_data is missing required 'location'",
            tensor.name
        ))
    })?;
    Ok(Some(ExternalDataSpec {
        location,
        offset: offset.unwrap_or(0),
        length,
        checksum,
    }))
}

fn validate_location(value: &str, tensor_name: &str) -> Result<PathBuf> {
    if value.is_empty() || value.as_bytes().contains(&0) {
        return Err(NyError::ModelLoad(format!(
            "Tensor {tensor_name} has an empty or NUL-containing external data location"
        )));
    }
    let bytes = value.as_bytes();
    let looks_like_windows_prefix = value.starts_with('\\')
        || (bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':');
    if value.contains('\\') || looks_like_windows_prefix {
        return Err(NyError::ModelLoad(format!(
            "Tensor {tensor_name} external data location '{value}' is not a portable relative POSIX path"
        )));
    }
    let path = Path::new(value);
    if path.is_absolute() {
        return Err(NyError::ModelLoad(format!(
            "Tensor {tensor_name} external data location '{value}' must be relative to the model directory"
        )));
    }

    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(NyError::ModelLoad(format!(
                    "Tensor {tensor_name} external data location '{value}' attempts to escape the model directory"
                )));
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        return Err(NyError::ModelLoad(format!(
            "Tensor {tensor_name} external data location '{value}' does not name a file"
        )));
    }
    Ok(normalized)
}

fn parse_decimal_u64(field: &str, value: &str, tensor_name: &str) -> Result<u64> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(NyError::ModelLoad(format!(
            "Tensor {tensor_name} external_data {field} '{value}' is not a non-negative decimal integer"
        )));
    }
    value.parse::<u64>().map_err(|err| {
        NyError::ModelLoad(format!(
            "Tensor {tensor_name} external_data {field} '{value}' is out of range: {err}"
        ))
    })
}

fn parse_sha1(value: &str, tensor_name: &str) -> Result<[u8; 20]> {
    if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(NyError::ModelLoad(format!(
            "Tensor {tensor_name} external_data checksum must be exactly 40 hexadecimal SHA1 characters"
        )));
    }
    let mut digest = [0u8; 20];
    // Keep `chunks_exact` for this hex-pair walk; the tippy `as_chunks`
    // rewrite reshapes the pair type for no clarity gain.
    #[allow(unknown_lints)] // stock 1.95 clippy (public pin) does not know the lint below
    #[allow(clippy::chunks_exact_to_as_chunks)]
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(pair).map_err(|err| {
            NyError::ModelLoad(format!(
                "Tensor {tensor_name} external_data checksum is not ASCII: {err}"
            ))
        })?;
        digest[index] = u8::from_str_radix(text, 16).map_err(|err| {
            NyError::ModelLoad(format!(
                "Tensor {tensor_name} external_data checksum is invalid: {err}"
            ))
        })?;
    }
    Ok(digest)
}

fn open_regular_file_no_symlinks(
    base_dir: &Dir,
    location: &Path,
    tensor_name: &str,
) -> Result<File> {
    let mut directory = base_dir.try_clone().map_err(|err| {
        NyError::ModelLoad(format!(
            "Tensor {tensor_name} failed to retain its model directory capability: {err}"
        ))
    })?;
    let mut components = location.components().peekable();
    while let Some(component) = components.next() {
        let Component::Normal(part) = component else {
            return Err(NyError::ModelLoad(format!(
                "Tensor {tensor_name} external data path {} was not normalized",
                location.display()
            )));
        };
        if components.peek().is_some() {
            directory = directory.open_dir_nofollow(part).map_err(|err| {
                NyError::ModelLoad(format!(
                    "external data path {} contains a symbolic link or non-directory component at {}: {err}",
                    location.display(),
                    part.to_string_lossy()
                ))
            })?;
            continue;
        }

        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No).nonblock(true);
        let file = directory.open_with(part, &options).map_err(|err| {
            NyError::ModelLoad(format!(
                "external data path {} contains a symbolic link, is missing, or cannot be opened: {err}",
                location.display()
            ))
        })?;
        let metadata = file.metadata().map_err(|err| {
            NyError::ModelLoad(format!(
                "failed to inspect external data {} after opening: {err}",
                location.display()
            ))
        })?;
        if !metadata.is_file() {
            return Err(NyError::ModelLoad(format!(
                "external data {} is not a regular file",
                location.display()
            )));
        }
        return Ok(file);
    }
    Err(NyError::ModelLoad(format!(
        "Tensor {tensor_name} external data path is empty"
    )))
}

fn validate_open_file_bounds(
    file_len: u64,
    spec: &ExternalDataSpec,
    expected_len: u64,
    tensor_name: &str,
) -> Result<()> {
    if spec.offset > file_len {
        return Err(NyError::ModelLoad(format!(
            "Tensor {tensor_name} external offset {} exceeds file size {} for {}",
            spec.offset,
            file_len,
            spec.location.display()
        )));
    }
    let available = file_len - spec.offset;
    let length = spec.length.unwrap_or(available);
    if length > available {
        return Err(NyError::ModelLoad(format!(
            "Tensor {tensor_name} external length {length} exceeds {available} available bytes at offset {} in {}",
            spec.offset,
            spec.location.display()
        )));
    }
    if length != expected_len {
        return Err(NyError::ModelLoad(format!(
            "Tensor {tensor_name} shape/type requires {expected_len} raw bytes, but external_data selects {length} bytes"
        )));
    }
    Ok(())
}

fn verify_checksum(
    file: &mut CachedExternalFile,
    expected: [u8; 20],
    tensor_name: &str,
    location: &Path,
) -> Result<()> {
    let actual = if let Some(cached) = file.sha1 {
        cached
    } else {
        file.file.seek(SeekFrom::Start(0)).map_err(|err| {
            NyError::ModelLoad(format!(
                "Tensor {tensor_name} failed to seek external data {} for checksum: {err}",
                location.display()
            ))
        })?;
        let mut hasher = Sha1::new();
        let mut buffer = vec![0u8; HASH_BUFFER_BYTES];
        // Hash exactly the size captured from the opened file descriptor.
        // A concurrent append must not turn checksum validation into an
        // unbounded EOF chase; an early EOF is a deterministic mutation error.
        let mut remaining = file.len;
        while remaining != 0 {
            let requested = usize::try_from(remaining.min(HASH_BUFFER_BYTES as u64))
                .expect("bounded by HASH_BUFFER_BYTES");
            let count = file.file.read(&mut buffer[..requested]).map_err(|err| {
                NyError::ModelLoad(format!(
                    "Tensor {tensor_name} failed to hash external data {}: {err}",
                    location.display()
                ))
            })?;
            if count == 0 {
                return Err(NyError::ModelLoad(format!(
                    "Tensor {tensor_name} external data {} was truncated during checksum validation",
                    location.display()
                )));
            }
            hasher.update(&buffer[..count]);
            remaining -= u64::try_from(count).expect("hash buffer length fits u64");
        }
        let digest: [u8; 20] = hasher.finalize().into();
        file.sha1 = Some(digest);
        digest
    };
    if actual != expected {
        return Err(NyError::ModelLoad(format!(
            "Tensor {tensor_name} external data {} failed SHA1 checksum validation (expected {}, got {})",
            location.display(),
            hex_sha1(&expected),
            hex_sha1(&actual)
        )));
    }
    Ok(())
}

fn hex_sha1(digest: &[u8; 20]) -> String {
    let mut out = String::with_capacity(40);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proto::{
        attribute_type, tensor_shape_proto, AttributeProto, GraphProto, ModelProto, NodeProto,
        OperatorSetIdProto, SparseTensorProto, StringStringEntryProto, TensorShapeProto,
        TensorTypeProto, TypeProto, ValueInfoProto,
    };
    use crate::{load_onnx, load_onnx_bytes};
    use prost::Message;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    fn entry(key: &str, value: impl ToString) -> StringStringEntryProto {
        StringStringEntryProto {
            key: key.to_string(),
            value: value.to_string(),
        }
    }

    fn external_tensor(
        name: &str,
        dims: &[i64],
        location: &str,
        offset: Option<u64>,
        length: Option<u64>,
        checksum: Option<&str>,
    ) -> TensorProto {
        let mut external_data = vec![entry("location", location)];
        if let Some(offset) = offset {
            external_data.push(entry("offset", offset));
        }
        if let Some(length) = length {
            external_data.push(entry("length", length));
        }
        if let Some(checksum) = checksum {
            external_data.push(entry("checksum", checksum));
        }
        TensorProto {
            dims: dims.to_vec(),
            data_type: 1,
            name: name.to_string(),
            external_data,
            data_location: DATA_LOCATION_EXTERNAL,
            ..Default::default()
        }
    }

    fn inline_tensor(name: &str, dims: &[i64], raw_data: Vec<u8>) -> TensorProto {
        TensorProto {
            dims: dims.to_vec(),
            data_type: 1,
            name: name.to_string(),
            raw_data,
            ..Default::default()
        }
    }

    fn value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: 1,
                    shape: Some(TensorShapeProto {
                        dim: dims
                            .iter()
                            .map(|dim| tensor_shape_proto::Dimension {
                                value: Some(tensor_shape_proto::dimension::Value::DimValue(*dim)),
                            })
                            .collect(),
                    }),
                }),
            }),
        }
    }

    fn model(
        initializers: Vec<TensorProto>,
        nodes: Vec<NodeProto>,
        outputs: &[(&str, &[i64])],
    ) -> ModelProto {
        ModelProto {
            ir_version: 9,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            graph: Some(GraphProto {
                name: "external-data-test".to_string(),
                initializer: initializers,
                node: nodes,
                output: outputs
                    .iter()
                    .map(|(name, dims)| value_info(name, dims))
                    .collect(),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn write_model(temp: &TempDir, filename: &str, model: &ModelProto) -> PathBuf {
        let path = temp.path().join(filename);
        fs::write(&path, model.encode_to_vec()).expect("write test ONNX");
        path
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn sha1_hex(bytes: &[u8]) -> String {
        let digest: [u8; 20] = Sha1::digest(bytes).into();
        hex_sha1(&digest)
    }

    #[test]
    fn protobuf_roundtrip_preserves_external_data_tag_13() {
        let tensor = external_tensor("weight", &[1], "weights.data", Some(3), Some(4), None);
        let decoded = TensorProto::decode(tensor.encode_to_vec().as_slice()).expect("decode");
        assert_eq!(decoded.data_location, DATA_LOCATION_EXTERNAL);
        assert_eq!(decoded.external_data, tensor.external_data);
    }

    #[test]
    fn shared_external_sidecar_matches_inline_weights_exactly() {
        let temp = TempDir::new().expect("tempdir");
        let a = f32_bytes(&[1.25, -2.5]);
        let b = f32_bytes(&[7.75]);
        let mut sidecar = vec![0xA5; 8];
        let a_offset = sidecar.len() as u64;
        sidecar.extend_from_slice(&a);
        sidecar.extend_from_slice(&[0x5A; 13]);
        let b_offset = sidecar.len() as u64;
        sidecar.extend_from_slice(&b);
        fs::write(temp.path().join("weights.data"), &sidecar).expect("write sidecar");

        let external = model(
            vec![
                external_tensor(
                    "a",
                    &[2],
                    "weights.data",
                    Some(a_offset),
                    Some(a.len() as u64),
                    None,
                ),
                external_tensor(
                    "b",
                    &[1],
                    "weights.data",
                    Some(b_offset),
                    Some(b.len() as u64),
                    None,
                ),
            ],
            Vec::new(),
            &[("a", &[2]), ("b", &[1])],
        );
        let external_path = write_model(&temp, "external.onnx", &external);

        let inline = model(
            vec![inline_tensor("a", &[2], a), inline_tensor("b", &[1], b)],
            Vec::new(),
            &[("a", &[2]), ("b", &[1])],
        );
        let external_loaded = load_onnx(external_path).expect("external model loads");
        let inline_loaded =
            load_onnx_bytes("inline.onnx", &inline.encode_to_vec()).expect("inline model loads");
        for name in ["a", "b"] {
            assert_eq!(
                external_loaded.weights.get(name).expect("external weight"),
                inline_loaded.weights.get(name).expect("inline weight")
            );
        }
    }

    #[test]
    fn missing_length_selects_from_offset_to_eof() {
        let temp = TempDir::new().expect("tempdir");
        let payload = f32_bytes(&[3.5]);
        let mut sidecar = vec![0u8; 11];
        sidecar.extend_from_slice(&payload);
        fs::write(temp.path().join("one.data"), &sidecar).expect("write sidecar");
        let model = model(
            vec![external_tensor(
                "weight",
                &[1],
                "one.data",
                Some(11),
                None,
                None,
            )],
            Vec::new(),
            &[("weight", &[1])],
        );
        let loaded = load_onnx(write_model(&temp, "model.onnx", &model)).expect("model loads");
        assert_eq!(
            loaded.weights.get("weight").expect("weight").as_slice(),
            Some(&[3.5][..])
        );
    }

    #[test]
    fn byte_loader_refuses_to_infer_filesystem_authority_from_name() {
        let model = model(
            vec![external_tensor(
                "weight",
                &[1],
                "weights.data",
                Some(0),
                Some(4),
                None,
            )],
            Vec::new(),
            &[("weight", &[1])],
        );
        let error = load_onnx_bytes("/tmp/apparently/path/model.onnx", &model.encode_to_vec())
            .expect_err("bytes API must fail closed");
        let message = error.to_string();
        assert!(message.contains("no model-origin directory"), "{message}");
        assert!(message.contains("load_onnx"), "{message}");
    }

    #[test]
    fn unmodeled_tensor_container_attributes_fail_closed() {
        for attribute_type in [5, 9, 10, 11, 12] {
            let node = NodeProto {
                name: "container".to_string(),
                op_type: "Custom".to_string(),
                attribute: vec![AttributeProto {
                    name: "payload".to_string(),
                    r#type: attribute_type,
                    ..Default::default()
                }],
                ..Default::default()
            };
            let model = model(Vec::new(), vec![node], &[]);
            let error = load_onnx_bytes("container.onnx", &model.encode_to_vec())
                .expect_err("tensor container must fail closed");
            let message = error.to_string();
            assert!(message.contains("tensor-container"), "{message}");
            assert!(message.contains(&attribute_type.to_string()), "{message}");
        }
    }

    #[test]
    fn graph_sparse_initializers_fail_closed_before_conversion() {
        let mut sparse_model = model(Vec::new(), Vec::new(), &[]);
        sparse_model
            .graph
            .as_mut()
            .expect("graph")
            .sparse_initializer
            .push(SparseTensorProto {
                values: Some(external_tensor(
                    "sparse_values",
                    &[1],
                    "sparse.data",
                    Some(0),
                    Some(4),
                    None,
                )),
                indices: Some(TensorProto {
                    dims: vec![1],
                    data_type: 7,
                    name: "sparse_indices".to_string(),
                    int64_data: vec![0],
                    ..Default::default()
                }),
                dims: vec![1],
            });

        let bytes = sparse_model.encode_to_vec();
        let decoded = ModelProto::decode(bytes.as_slice()).expect("decode sparse initializer");
        assert_eq!(
            decoded
                .graph
                .as_ref()
                .expect("graph")
                .sparse_initializer
                .len(),
            1
        );
        let error = load_onnx_bytes("sparse.onnx", &bytes)
            .expect_err("sparse initializer must fail closed");
        let message = error.to_string();
        assert!(message.contains("sparse initializer"), "{message}");
        assert!(message.contains("discard model parameters"), "{message}");

        let temp = TempDir::new().expect("tempdir");
        let error = load_onnx(write_model(&temp, "sparse.onnx", &sparse_model))
            .expect_err("file loader must reject sparse initializer before sidecar access");
        let message = error.to_string();
        assert!(message.contains("sparse initializer"), "{message}");
        assert!(message.contains("discard model parameters"), "{message}");
    }

    #[test]
    fn checksum_covers_the_entire_sidecar_and_is_enforced() {
        let temp = TempDir::new().expect("tempdir");
        let sidecar = f32_bytes(&[1.0, 2.0]);
        fs::write(temp.path().join("weights.data"), &sidecar).expect("write sidecar");
        let checksum = sha1_hex(&sidecar);
        let valid = model(
            vec![external_tensor(
                "weight",
                &[1],
                "weights.data",
                Some(4),
                Some(4),
                Some(&checksum),
            )],
            Vec::new(),
            &[("weight", &[1])],
        );
        load_onnx(write_model(&temp, "valid.onnx", &valid)).expect("valid checksum loads");

        let invalid = model(
            vec![external_tensor(
                "weight",
                &[1],
                "weights.data",
                Some(4),
                Some(4),
                Some("0000000000000000000000000000000000000000"),
            )],
            Vec::new(),
            &[("weight", &[1])],
        );
        let error = load_onnx(write_model(&temp, "invalid.onnx", &invalid))
            .expect_err("wrong checksum must fail");
        assert!(error.to_string().contains("SHA1 checksum"), "{error}");
    }

    #[test]
    fn metadata_and_file_bounds_fail_closed() {
        let malformed = [
            (vec![entry("offset", "0")], "missing required 'location'"),
            (vec![entry("location", "")], "empty"),
            (
                vec![entry("location", "weights.data"), entry("offset", "-1")],
                "non-negative decimal",
            ),
            (
                vec![
                    entry("location", "weights.data"),
                    entry("offset", "not-a-number"),
                ],
                "non-negative decimal",
            ),
            (
                vec![
                    entry("location", "weights.data"),
                    entry("offset", "18446744073709551616"),
                ],
                "out of range",
            ),
            (
                vec![
                    entry("location", "weights.data"),
                    entry("length", "not-a-number"),
                ],
                "non-negative decimal",
            ),
            (
                vec![
                    entry("location", "weights.data"),
                    entry("length", "18446744073709551616"),
                ],
                "out of range",
            ),
            (
                vec![
                    entry("location", "weights.data"),
                    entry("location", "other.data"),
                ],
                "duplicate",
            ),
            (
                vec![entry("location", "weights.data"), entry("evil", "x")],
                "unknown",
            ),
            (
                vec![entry("location", "weights.data"), entry("checksum", "abcd")],
                "40 hexadecimal",
            ),
        ];
        for (external_data, expected) in malformed {
            let tensor = TensorProto {
                dims: vec![1],
                data_type: 1,
                name: "weight".to_string(),
                external_data,
                data_location: DATA_LOCATION_EXTERNAL,
                ..Default::default()
            };
            let error = parse_external_spec(&tensor).expect_err("metadata must fail");
            assert!(error.to_string().contains(expected), "{error}");
        }

        let temp = TempDir::new().expect("tempdir");
        fs::write(temp.path().join("weights.data"), [0u8; 4]).expect("write sidecar");
        let out_of_bounds = model(
            vec![external_tensor(
                "weight",
                &[1],
                "weights.data",
                Some(2),
                Some(4),
                None,
            )],
            Vec::new(),
            &[("weight", &[1])],
        );
        let error = load_onnx(write_model(&temp, "bounds.onnx", &out_of_bounds))
            .expect_err("slice past EOF must fail");
        assert!(error.to_string().contains("available bytes"), "{error}");

        let wrong_shape_length = model(
            vec![external_tensor(
                "weight",
                &[2],
                "weights.data",
                Some(0),
                Some(4),
                None,
            )],
            Vec::new(),
            &[("weight", &[2])],
        );
        let error = load_onnx(write_model(&temp, "length.onnx", &wrong_shape_length))
            .expect_err("shape byte mismatch must fail");
        assert!(
            error.to_string().contains("requires 8 raw bytes"),
            "{error}"
        );
    }

    #[test]
    fn data_location_and_inline_external_ambiguities_fail_closed() {
        let default_with_metadata = TensorProto {
            dims: vec![1],
            data_type: 1,
            name: "default_with_metadata".to_string(),
            external_data: vec![entry("location", "weights.data")],
            ..Default::default()
        };
        let error =
            parse_external_spec(&default_with_metadata).expect_err("DEFAULT metadata must fail");
        assert!(error.to_string().contains("DEFAULT"), "{error}");

        let mut external_with_inline = external_tensor(
            "external_with_inline",
            &[1],
            "weights.data",
            None,
            Some(4),
            None,
        );
        external_with_inline.raw_data = f32_bytes(&[1.0]);
        let error =
            parse_external_spec(&external_with_inline).expect_err("mixed storage must fail");
        assert!(error.to_string().contains("inline field"), "{error}");

        let hidden_fields: [(&str, fn(&mut TensorProto)); 3] = [
            ("string_data", |tensor: &mut TensorProto| {
                tensor.string_data.push(b"hidden".to_vec())
            }),
            ("uint64_data", |tensor: &mut TensorProto| {
                tensor.uint64_data.push(7)
            }),
            ("segment", |tensor: &mut TensorProto| {
                tensor.segment = Some(onnx_proto::TensorSegmentProto { begin: 0, end: 1 })
            }),
        ];
        for (field, mutate) in hidden_fields {
            let mut tensor = external_tensor(field, &[1], "weights.data", None, Some(4), None);
            mutate(&mut tensor);
            let error = parse_external_spec(&tensor).expect_err("mixed storage must fail");
            assert!(error.to_string().contains(field), "field={field} {error}");
        }

        let unknown_location = TensorProto {
            dims: vec![1],
            data_type: 1,
            name: "unknown_location".to_string(),
            data_location: 2,
            ..Default::default()
        };
        let error = parse_external_spec(&unknown_location).expect_err("unknown enum must fail");
        assert!(
            error.to_string().contains("unknown ONNX data_location"),
            "{error}"
        );
    }

    #[test]
    fn missing_and_non_regular_sidecars_fail_closed() {
        // A directory named as a sidecar must FAIL CLOSED — that is the property
        // under test, and it holds on both platforms. The message differs by
        // where the refusal originates: Unix opens the directory and then hits
        // the explicit regular-file check, while the Windows capability refuses
        // to open a directory at all ("Access is denied", os error 5), one step
        // earlier. Accepting either wording keeps the assertion about the
        // rejection rather than about which layer produced it.
        const DIRECTORY_REJECTED: &[&str] = &["not a regular file", "cannot be opened"];
        const MISSING_REJECTED: &[&str] = &["is missing"];

        let temp = TempDir::new().expect("tempdir");
        fs::create_dir(temp.path().join("directory.data")).expect("sidecar directory");
        for (filename, location, expected) in [
            ("missing.onnx", "missing.data", MISSING_REJECTED),
            ("directory.onnx", "directory.data", DIRECTORY_REJECTED),
        ] {
            let model = model(
                vec![external_tensor(
                    "weight",
                    &[1],
                    location,
                    Some(0),
                    Some(4),
                    None,
                )],
                Vec::new(),
                &[("weight", &[1])],
            );
            let error = load_onnx(write_model(&temp, filename, &model))
                .expect_err("invalid sidecar must fail");
            let message = error.to_string();
            assert!(
                expected.iter().any(|needle| message.contains(needle)),
                "expected one of {expected:?}, got: {message}"
            );
        }
    }

    #[test]
    fn authored_basepath_is_accepted_but_never_overrides_model_origin() {
        let temp = TempDir::new().expect("tempdir");
        fs::write(temp.path().join("weights.data"), f32_bytes(&[6.25])).expect("sidecar");
        let mut tensor = external_tensor("weight", &[1], "weights.data", Some(0), Some(4), None);
        tensor
            .external_data
            .push(entry("basepath", "/attacker-controlled/ignored"));
        let model = model(vec![tensor], Vec::new(), &[("weight", &[1])]);
        let loaded = load_onnx(write_model(&temp, "basepath.onnx", &model)).expect("model loads");
        assert_eq!(
            loaded.weights.get("weight").expect("weight").as_slice(),
            Some(&[6.25][..])
        );
    }

    #[test]
    fn traversal_and_nonportable_locations_are_rejected() {
        for location in [
            "../secret.data",
            "nested/../../secret.data",
            "/etc/passwd",
            r"C:\secret.data",
            r"\\server\share\data",
        ] {
            let tensor = external_tensor("weight", &[1], location, Some(0), Some(4), None);
            let error = parse_external_spec(&tensor).expect_err("location must fail");
            let message = error.to_string();
            assert!(
                message.contains("model directory") || message.contains("portable relative"),
                "location={location} message={message}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn final_and_parent_symlink_components_are_rejected() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().expect("tempdir");
        let real = temp.path().join("real");
        fs::create_dir(&real).expect("real dir");
        fs::write(real.join("weights.data"), f32_bytes(&[1.0])).expect("sidecar");
        symlink(real.join("weights.data"), temp.path().join("leaf.data")).expect("leaf symlink");
        symlink(&real, temp.path().join("linked-dir")).expect("parent symlink");

        for (filename, location) in [
            ("leaf.onnx", "leaf.data"),
            ("parent.onnx", "linked-dir/weights.data"),
        ] {
            let model = model(
                vec![external_tensor(
                    "weight",
                    &[1],
                    location,
                    Some(0),
                    Some(4),
                    None,
                )],
                Vec::new(),
                &[("weight", &[1])],
            );
            let error =
                load_onnx(write_model(&temp, filename, &model)).expect_err("symlink must fail");
            assert!(error.to_string().contains("symbolic link"), "{error}");
        }
    }

    #[cfg(unix)]
    #[ntest::timeout(10000)]
    #[test]
    fn fifo_sidecar_is_rejected_without_blocking() {
        let temp = TempDir::new().expect("tempdir");
        let fifo = temp.path().join("weights.fifo");
        let status = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .expect("run POSIX mkfifo");
        assert!(status.success(), "mkfifo failed with {status}");
        let model = model(
            vec![external_tensor(
                "weight",
                &[1],
                "weights.fifo",
                Some(0),
                Some(4),
                None,
            )],
            Vec::new(),
            &[("weight", &[1])],
        );
        let error = load_onnx(write_model(&temp, "fifo.onnx", &model)).expect_err("FIFO must fail");
        assert!(error.to_string().contains("regular file"), "{error}");
    }

    #[test]
    fn regular_file_hardlink_is_valid_directory_authority() {
        let temp = TempDir::new().expect("tempdir");
        let original = temp.path().join("original.data");
        fs::write(&original, f32_bytes(&[2.75])).expect("write original");
        fs::hard_link(&original, temp.path().join("weights.data")).expect("create hard link");
        let model = model(
            vec![external_tensor(
                "weight",
                &[1],
                "weights.data",
                Some(0),
                Some(4),
                None,
            )],
            Vec::new(),
            &[("weight", &[1])],
        );
        let loaded = load_onnx(write_model(&temp, "hardlink.onnx", &model))
            .expect("a regular inode named inside the capability is authorized");
        assert_eq!(
            loaded.weights.get("weight").expect("weight").as_slice(),
            Some(&[2.75][..])
        );
    }

    #[test]
    fn sparse_sidecar_offset_above_two_gib_is_read_without_whole_file_allocation() {
        let temp = TempDir::new().expect("tempdir");
        let offset = (1u64 << 31) + 4096;
        let sidecar_path = temp.path().join("large.data");
        let mut sidecar = fs::File::create(&sidecar_path).expect("create sparse sidecar");
        sidecar.set_len(offset + 4).expect("extend sparse sidecar");
        sidecar
            .seek(SeekFrom::Start(offset))
            .expect("seek sparse sidecar");
        sidecar
            .write_all(&42.5f32.to_le_bytes())
            .expect("write tensor slice");
        drop(sidecar);

        let model = model(
            vec![external_tensor(
                "weight",
                &[1],
                "large.data",
                Some(offset),
                Some(4),
                None,
            )],
            Vec::new(),
            &[("weight", &[1])],
        );
        let loaded =
            load_onnx(write_model(&temp, "large.onnx", &model)).expect("sparse model loads");
        assert_eq!(
            loaded.weights.get("weight").expect("weight").as_slice(),
            Some(&[42.5][..])
        );
    }

    #[test]
    fn external_constant_attribute_is_materialized_before_folding() {
        let temp = TempDir::new().expect("tempdir");
        fs::write(temp.path().join("constant.data"), f32_bytes(&[9.25])).expect("sidecar");
        let constant = NodeProto {
            name: "constant".to_string(),
            op_type: "Constant".to_string(),
            output: vec!["constant_output".to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: attribute_type::TENSOR,
                t: Some(external_tensor(
                    "attribute_payload",
                    &[1],
                    "constant.data",
                    Some(0),
                    Some(4),
                    None,
                )),
                ..Default::default()
            }],
            ..Default::default()
        };
        let model = model(Vec::new(), vec![constant], &[("constant_output", &[1])]);
        let loaded =
            load_onnx(write_model(&temp, "constant.onnx", &model)).expect("constant loads");
        assert_eq!(
            loaded
                .weights
                .get("constant_output")
                .expect("folded constant")
                .as_slice(),
            Some(&[9.25][..])
        );
    }

    #[test]
    fn external_integer_shape_controls_are_available_to_qdq_preflight() {
        let temp = TempDir::new().expect("tempdir");
        let shape_payload = 3_i64.to_le_bytes();
        fs::write(temp.path().join("shape.data"), shape_payload).expect("shape sidecar");

        for constant_attribute in [false, true] {
            let mut external_shape = external_tensor(
                if constant_attribute {
                    "shape_attribute_payload"
                } else {
                    "shape"
                },
                &[1],
                "shape.data",
                Some(0),
                Some(shape_payload.len() as u64),
                None,
            );
            external_shape.data_type = 7; // INT64

            let mut initializers = vec![
                inline_tensor("x", &[1, 3], f32_bytes(&[0.0, 1.0, 2.0])),
                inline_tensor("scale_base", &[3], f32_bytes(&[0.5, 0.5, 0.5])),
                TensorProto {
                    dims: vec![3],
                    data_type: 2, // UINT8
                    name: "zero_point".to_string(),
                    int32_data: vec![0, 0, 0],
                    ..Default::default()
                },
            ];
            let mut nodes = Vec::new();
            if constant_attribute {
                nodes.push(NodeProto {
                    name: "shape_constant".to_string(),
                    op_type: "Constant".to_string(),
                    output: vec!["shape".to_string()],
                    attribute: vec![AttributeProto {
                        name: "value".to_string(),
                        r#type: attribute_type::TENSOR,
                        t: Some(external_shape),
                        ..Default::default()
                    }],
                    ..Default::default()
                });
            } else {
                initializers.push(external_shape);
            }
            nodes.extend([
                NodeProto {
                    name: "reshape_scale".to_string(),
                    op_type: "Reshape".to_string(),
                    input: vec!["scale_base".to_string(), "shape".to_string()],
                    output: vec!["scale".to_string()],
                    ..Default::default()
                },
                NodeProto {
                    name: "quantize".to_string(),
                    op_type: "QuantizeLinear".to_string(),
                    input: vec![
                        "x".to_string(),
                        "scale".to_string(),
                        "zero_point".to_string(),
                    ],
                    output: vec!["q".to_string()],
                    ..Default::default()
                },
                NodeProto {
                    name: "dequantize".to_string(),
                    op_type: "DequantizeLinear".to_string(),
                    input: vec![
                        "q".to_string(),
                        "scale".to_string(),
                        "zero_point".to_string(),
                    ],
                    output: vec!["y".to_string()],
                    ..Default::default()
                },
            ]);

            let model = model(initializers, nodes, &[("y", &[1, 3])]);
            let filename = if constant_attribute {
                "qdq-constant-shape.onnx"
            } else {
                "qdq-initializer-shape.onnx"
            };
            let loaded = load_onnx(write_model(&temp, filename, &model))
                .expect("external INT64 Q/DQ shape control loads");
            assert_eq!(
                loaded
                    .weights
                    .get("y")
                    .expect("folded Q/DQ output")
                    .as_slice(),
                Some(&[0.0, 1.0, 2.0][..])
            );
        }
    }
}
