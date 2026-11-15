// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! End-to-end fail-closed behaviour of the `certify_onnx` binary on a model
//! whose declared dims ask for an input-chosen allocation.
//!
//! A Gemm weight declared `[2^21, 1]` used to make the DAG loader build a
//! `2^21 x 2^21` exact identity (2^42 rationals) before any size check ran:
//! an allocation abort, not a diagnostic. A first fix bounded each dimension
//! (2^20) but not their product, so `[2^20, 0]` (a 62-byte model, no weight
//! data, passing the `kin * kout` length check) still asked for a
//! `2^20 x 2^20` identity and was killed at 11 GB of RSS. An input-chosen
//! matrix shape is now bounded by its ENTRY count (`MAX_MATRIX_ENTRIES`,
//! 2^26), and the loader answers `MODEL_UNSUPPORTED` with exit status 2, the
//! status a malformed model already gets.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

/// A length-delimited protobuf field.
fn ld(field: u64, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    varint((field << 3) | 2, &mut out);
    varint(payload.len() as u64, &mut out);
    out.extend_from_slice(payload);
    out
}

fn tensor(name: &str, dims: &[u64], floats: &[f32]) -> Vec<u8> {
    let mut t = Vec::new();
    for &d in dims {
        varint(1 << 3, &mut t); // dims = 1, varint
        varint(d, &mut t);
    }
    let packed: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
    t.extend(ld(4, &packed));
    t.extend(ld(8, name.as_bytes()));
    t
}

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> Vec<u8> {
    let mut n = Vec::new();
    for i in inputs {
        n.extend(ld(1, i.as_bytes()));
    }
    for o in outputs {
        n.extend(ld(2, o.as_bytes()));
    }
    n.extend(ld(4, op.as_bytes()));
    n
}

/// `out = Gemm(x, W, C)` with `W` declared `[kin, kout]` and no weight data
/// (the loader must refuse the shape before it ever checks the data length).
fn wide_gemm_model(kin: u64, kout: u64) -> Vec<u8> {
    let bias = vec![0.0; usize::try_from(kout).expect("small kout")];
    let mut g = Vec::new();
    g.extend(ld(1, &node("Gemm", &["x", "W", "C"], &["out"])));
    g.extend(ld(5, &tensor("W", &[kin, kout], &[])));
    g.extend(ld(5, &tensor("C", &[kout], &bias)));
    g.extend(ld(11, &ld(1, b"x")));
    g.extend(ld(12, &ld(1, b"out")));
    ld(7, &g)
}

/// Run `certify_onnx` on `model` against a one-input box. A regression to an
/// input-chosen allocation would exhaust memory rather than exit, so the
/// child is killed after a deadline and that counts as a failure.
fn run_certify(model_bytes: &[u8]) -> (Option<i32>, String) {
    let dir = tempfile::tempdir().expect("scratch dir");
    let model = dir.path().join("wide.onnx");
    let prop = dir.path().join("prop.vnnlib");
    std::fs::write(&model, model_bytes).expect("write model");
    std::fs::write(
        &prop,
        "(assert (>= X_0 0.0))\n(assert (<= X_0 1.0))\n(assert (<= Y_0 0.0))\n",
    )
    .expect("write vnnlib");
    let mut child = Command::new(env!("CARGO_BIN_EXE_certify_onnx"))
        .arg(&model)
        .arg(&prop)
        .arg(dir.path().join("out"))
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run certify_onnx");
    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll certify_onnx") {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("certify_onnx did not exit within 60 s (input-chosen allocation?)");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let mut stderr = String::new();
    if let Some(mut e) = child.stderr.take() {
        e.read_to_string(&mut stderr).expect("read stderr");
    }
    (status.code(), stderr)
}

#[test]
fn oversized_gemm_width_exits_2_with_a_diagnostic() {
    let (code, stderr) = run_certify(&wide_gemm_model(1 << 21, 1));
    assert_eq!(code, Some(2), "stderr: {stderr}");
    assert!(
        stderr.contains(
            "MODEL_UNSUPPORTED: matrix 2097152x2097152 exceeds MAX_MATRIX_ENTRIES (67108864)"
        ),
        "stderr: {stderr}"
    );
}

#[test]
fn quadratic_identity_within_the_per_dimension_bound_exits_2() {
    // The review's reproduction: each dimension within 2^20, `kout = 0` so
    // zero weight floats pass the length check, and the identity over the
    // `kin` inputs alone is 2^40 entries.
    let (code, stderr) = run_certify(&wide_gemm_model(1 << 20, 0));
    assert_eq!(code, Some(2), "stderr: {stderr}");
    assert!(
        stderr.contains(
            "MODEL_UNSUPPORTED: matrix 1048576x1048576 exceeds MAX_MATRIX_ENTRIES (67108864)"
        ),
        "stderr: {stderr}"
    );
}

#[test]
fn identity_one_past_the_entry_cap_exits_2() {
    // 8193^2 = 2^26 + 16385: the smallest square identity past the cap.
    let (code, stderr) = run_certify(&wide_gemm_model(8193, 1));
    assert_eq!(code, Some(2), "stderr: {stderr}");
    assert!(
        stderr.contains("MODEL_UNSUPPORTED: matrix 8193x8193 exceeds MAX_MATRIX_ENTRIES"),
        "stderr: {stderr}"
    );
}

#[test]
fn small_shape_reaches_the_data_check_not_the_cap() {
    // Within the cap the loader proceeds and the missing weight data is what
    // fails (closed, by panic): the cap refuses only shapes past it.
    let (code, stderr) = run_certify(&wide_gemm_model(4, 1));
    assert_ne!(code, Some(2), "stderr: {stderr}");
    assert!(!stderr.contains("MODEL_UNSUPPORTED"), "stderr: {stderr}");
    assert!(
        stderr.contains("Gemm weight size mismatch"),
        "stderr: {stderr}"
    );
}
