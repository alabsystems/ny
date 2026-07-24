// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Subprocess execution backend for ONNX Runtime shape inference.
//!
//! ORT shape inference is a blocking FFI call into ONNX Runtime's C++ that can
//! panic, abort, or fault. The workspace release profile unwinds Rust panics,
//! but no Rust panic boundary can contain a native abort, segmentation fault,
//! or hung FFI call. Running inference in a short-lived child process makes
//! those failures observable as an ordinary exit status or timeout: the parent
//! reports a shape-inference error and the loaders degrade to the same graceful
//! no-inferred-shapes fallback used for any other inference failure.
//!
//! # Wire protocol (version 1)
//!
//! * Parent spawns `<exe> __shape-infer` with piped stdin/stdout/stderr.
//! * Parent streams the raw ONNX model bytes to the child's stdin, then closes
//!   it (EOF marks end of input).
//! * Child runs the in-process shape inference and writes a single JSON
//!   document to stdout: `{"version": 1, "shapes": {"tensor": [1, 3], ...}}`.
//! * Exit status 0 with a parseable, version-matching document is the ONLY
//!   success. Anything else — spawn failure, non-zero exit, signal/abort,
//!   deadline expiry, unparseable output, version mismatch — is an inference
//!   error. Shapes are never fabricated from a failed exchange.

use ny_core::{NyError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Hidden CLI subcommand that serves the shape-inference protocol.
///
/// The double-underscore prefix keeps it out of any real command namespace;
/// binaries that embed the server intercept this as the first argument before
/// normal CLI parsing.
pub const SHAPE_INFER_SUBCOMMAND: &str = "__shape-infer";

/// Version tag carried in every response; bump on any wire-format change so a
/// parent never misreads output from a mismatched child binary.
const SHAPE_INFER_PROTOCOL_VERSION: u32 = 1;

/// Parent-side deadline for the whole child exchange. The child enforces its
/// own 30s in-process ORT deadline (`ORT_SHAPE_INFERENCE_TIMEOUT`) and exits
/// cleanly when it fires, so this outer limit only backstops a child that is
/// itself wedged (e.g. ORT hanging process teardown). It must stay comfortably
/// above the child's internal deadline so the child's clean timeout path wins.
const SUBPROCESS_DEADLINE: Duration = Duration::from_secs(45);

/// Poll interval for reaping the child under the deadline loop.
const REAP_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Maximum bytes of child stderr echoed into error messages.
const STDERR_TAIL_BYTES: usize = 512;

/// Versioned response document written by the child to stdout.
#[derive(Debug, Serialize, Deserialize)]
struct ShapeInferResponse {
    version: u32,
    shapes: HashMap<String, Vec<i64>>,
}

/// Server side of the protocol: read ONNX model bytes from `input` until EOF,
/// run the in-process ORT shape inference, and write the versioned response
/// document to `output`.
///
/// Embed this behind a hidden CLI entry (see [`SHAPE_INFER_SUBCOMMAND`]) with
/// `input`/`output` wired to the process stdin/stdout, and exit non-zero on
/// `Err` so the parent observes failure through the exit status.
pub fn serve_shape_infer_request(input: &mut dyn Read, output: &mut dyn Write) -> Result<()> {
    let mut model_bytes = Vec::new();
    input.read_to_end(&mut model_bytes).map_err(|e| {
        NyError::ModelLoad(format!(
            "shape-infer server failed to read model bytes: {e}"
        ))
    })?;
    let shapes = super::infer_tensor_shapes_from_ort(&model_bytes)?;
    let response = ShapeInferResponse {
        version: SHAPE_INFER_PROTOCOL_VERSION,
        shapes,
    };
    let payload = serde_json::to_vec(&response).map_err(|e| {
        NyError::ModelLoad(format!("shape-infer server failed to encode response: {e}"))
    })?;
    output.write_all(&payload).map_err(|e| {
        NyError::ModelLoad(format!("shape-infer server failed to write response: {e}"))
    })?;
    output.flush().map_err(|e| {
        NyError::ModelLoad(format!("shape-infer server failed to flush response: {e}"))
    })?;
    Ok(())
}

/// Client side of the protocol: run ORT shape inference for `model_bytes` in a
/// child process serving [`SHAPE_INFER_SUBCOMMAND`].
///
/// Every failure mode returns `Err` (never panics, never fabricates shapes);
/// the loaders' existing `unwrap_or_else` fallback turns that into the sound
/// no-inferred-shapes path.
pub(crate) fn infer_tensor_shapes_via_subprocess(
    exe: &Path,
    model_bytes: &[u8],
) -> Result<HashMap<String, Vec<i64>>> {
    let mut child = Command::new(exe)
        .arg(SHAPE_INFER_SUBCOMMAND)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            NyError::ModelLoad(format!(
                "shape-infer subprocess {} failed to spawn: {e}",
                exe.display()
            ))
        })?;

    // Feed stdin from a helper thread so a child that never reads (or exits
    // early) cannot deadlock us against a full pipe. A write error here is
    // expected for an early-exiting child and surfaces via the exit status.
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| NyError::ModelLoad("shape-infer subprocess stdin unavailable".into()))?;
    let stdin_bytes = model_bytes.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&stdin_bytes);
        // Dropping stdin closes the pipe: EOF marks end of the request.
    });

    // Drain stdout/stderr concurrently for the same no-deadlock reason.
    let mut stdout_pipe = child
        .stdout
        .take()
        .ok_or_else(|| NyError::ModelLoad("shape-infer subprocess stdout unavailable".into()))?;
    let stdout_reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout_pipe.read_to_end(&mut buf);
        buf
    });
    let mut stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| NyError::ModelLoad("shape-infer subprocess stderr unavailable".into()))?;
    let stderr_reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr_pipe.read_to_end(&mut buf);
        buf
    });

    // Reap the child under the parent-side deadline. On expiry, kill it and
    // report a timeout; the pipe readers finish on their own once the pipes
    // close, so detaching them here leaks nothing.
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if start.elapsed() >= SUBPROCESS_DEADLINE {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(NyError::ModelLoad(format!(
                        "shape-infer subprocess timed out after {}s",
                        SUBPROCESS_DEADLINE.as_secs()
                    )));
                }
                std::thread::sleep(REAP_POLL_INTERVAL);
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(NyError::ModelLoad(format!(
                    "shape-infer subprocess wait failed: {e}"
                )));
            }
        }
    };

    let _ = writer.join();
    let stdout_bytes = stdout_reader.join().unwrap_or_default();
    let stderr_bytes = stderr_reader.join().unwrap_or_default();

    if !status.success() {
        return Err(NyError::ModelLoad(format!(
            "shape-infer subprocess exited with {status}: {}",
            stderr_tail(&stderr_bytes)
        )));
    }
    parse_shape_infer_response(&stdout_bytes)
}

/// Decode and version-check a response document.
pub(super) fn parse_shape_infer_response(bytes: &[u8]) -> Result<HashMap<String, Vec<i64>>> {
    let response: ShapeInferResponse = serde_json::from_slice(bytes).map_err(|e| {
        NyError::ModelLoad(format!(
            "shape-infer subprocess wrote an unparseable response: {e}"
        ))
    })?;
    if response.version != SHAPE_INFER_PROTOCOL_VERSION {
        return Err(NyError::ModelLoad(format!(
            "shape-infer subprocess protocol version {} does not match expected {}",
            response.version, SHAPE_INFER_PROTOCOL_VERSION
        )));
    }
    Ok(response.shapes)
}

/// Last `STDERR_TAIL_BYTES` of child stderr, lossily decoded for diagnostics.
fn stderr_tail(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "(no stderr)".to_string();
    }
    let tail_start = bytes.len().saturating_sub(STDERR_TAIL_BYTES);
    String::from_utf8_lossy(&bytes[tail_start..])
        .trim()
        .to_string()
}
