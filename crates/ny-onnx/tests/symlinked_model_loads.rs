// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Regression pin for the nn4sys model-open gate.
//
// Commit 38a2fecf routed the model file itself through the cap-std directory
// capability that later resolves the model's untrusted `external_data`
// locations. The capability is anchored at the model path's parent, so a model
// that is a SYMLINK to another directory was refused with
//
//   NY-HARNESS: MODEL-LOAD-FAILURE -- failed to open ONNX model ... through its
//   retained directory capability: a path led outside of the filesystem
//
// `benchmarks/vnncomp2025/benchmarks/nn4sys/onnx/mscn_2048d{,_dual}.onnx` are
// exactly that (symlinks into `../../nn4sys_2023/onnx/`), which made 34 of the
// REGULAR-track category's 194 instances unloadable and zeroed two banked
// unsats. The fix resolves the operator-supplied model path FIRST and anchors
// the capability at the real file's directory, which leaves the thing that gate
// exists to contain — the untrusted `location` strings authored inside the
// model — confined exactly as before.
//
// This test pins both halves: a symlinked model LOADS, and an external-data
// `location` still cannot escape the (resolved) model directory.

use std::io::Write;
// `Path` is only named by the `#[cfg(unix)]` model-writer below.
#[cfg(unix)]
use std::path::Path;
use std::path::PathBuf;

use ny_onnx::load_onnx;

/// Minimal single-Relu ONNX model, built by hand so the test needs no fixture.
///
/// Only the symlink test above needs it, so it carries that test's `unix` gate.
#[cfg(unix)]
fn write_minimal_model(path: &Path) {
    // ModelProto { ir_version: 8, opset_import: [{domain:"", version:13}],
    //              graph: GraphProto { ... } }
    fn varint(mut value: u64, out: &mut Vec<u8>) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                break;
            }
            out.push(byte | 0x80);
        }
    }
    fn tag(field: u32, wire: u32, out: &mut Vec<u8>) {
        varint(u64::from((field << 3) | wire), out);
    }
    fn len_delim(field: u32, payload: &[u8], out: &mut Vec<u8>) {
        tag(field, 2, out);
        varint(payload.len() as u64, out);
        out.extend_from_slice(payload);
    }
    fn string_field(field: u32, value: &str, out: &mut Vec<u8>) {
        len_delim(field, value.as_bytes(), out);
    }
    fn varint_field(field: u32, value: u64, out: &mut Vec<u8>) {
        tag(field, 0, out);
        varint(value, out);
    }

    // TensorShapeProto::Dimension { dim_value: 1 }
    let mut dim = Vec::new();
    varint_field(1, 1, &mut dim);
    // TensorShapeProto { dim: [dim] }
    let mut shape = Vec::new();
    len_delim(1, &dim, &mut shape);
    // TypeProto::Tensor { elem_type: 1 (FLOAT), shape }
    let mut tensor_type = Vec::new();
    varint_field(1, 1, &mut tensor_type);
    len_delim(2, &shape, &mut tensor_type);
    // TypeProto { tensor_type }
    let mut type_proto = Vec::new();
    len_delim(1, &tensor_type, &mut type_proto);

    let value_info = |name: &str| {
        let mut buf = Vec::new();
        string_field(1, name, &mut buf);
        len_delim(2, &type_proto, &mut buf);
        buf
    };

    // NodeProto { input: ["x"], output: ["y"], name: "relu", op_type: "Relu" }
    let mut node = Vec::new();
    string_field(1, "x", &mut node);
    string_field(2, "y", &mut node);
    string_field(3, "relu", &mut node);
    string_field(4, "Relu", &mut node);

    // GraphProto { node: [node], name: "g", input: [x], output: [y] }
    let mut graph = Vec::new();
    len_delim(1, &node, &mut graph);
    string_field(2, "g", &mut graph);
    len_delim(11, &value_info("x"), &mut graph);
    len_delim(12, &value_info("y"), &mut graph);

    // OperatorSetIdProto { domain: "", version: 13 }
    let mut opset = Vec::new();
    string_field(1, "", &mut opset);
    varint_field(2, 13, &mut opset);

    let mut model = Vec::new();
    varint_field(1, 8, &mut model); // ir_version
    len_delim(7, &graph, &mut model);
    len_delim(8, &opset, &mut model);

    let mut file = std::fs::File::create(path).expect("create model");
    file.write_all(&model).expect("write model");
}

fn scratch_dir(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "ny-symlink-model-{}-{}-{}",
        name,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).expect("create scratch root");
    root
}

/// A model reached through a symlink that leaves its own directory must LOAD.
/// This is the nn4sys shape: `nn4sys/onnx/m.onnx -> ../../nn4sys_2023/onnx/m.onnx`.
///
/// Unix-only, and not merely because `std::os::unix::fs::symlink` is (naming it
/// unconditionally is what kept this crate from compiling on Windows at all).
/// The Windows equivalent, `std::os::windows::fs::symlink_file`, needs
/// `SeCreateSymbolicLinkPrivilege` — Developer Mode or an elevated shell — so
/// the test would fail on an ordinary Windows checkout for a reason unrelated
/// to the gate it pins. The scored platform is Linux, where this runs.
#[cfg(unix)]
#[test]
fn model_reached_through_an_escaping_symlink_loads() {
    let root = scratch_dir("load");
    let real_dir = root.join("real/onnx");
    let link_dir = root.join("linked/onnx");
    std::fs::create_dir_all(&real_dir).expect("real dir");
    std::fs::create_dir_all(&link_dir).expect("link dir");

    let real_model = real_dir.join("m.onnx");
    write_minimal_model(&real_model);

    let linked_model = link_dir.join("m.onnx");
    std::os::unix::fs::symlink("../../real/onnx/m.onnx", &linked_model).expect("symlink");

    let direct = load_onnx(&real_model).expect("the real path must load");
    let through_link = load_onnx(&linked_model).unwrap_or_else(|error| {
        panic!("a symlinked model must load, not fail closed: {error}");
    });
    assert_eq!(
        direct.network.layers.len(),
        through_link.network.layers.len(),
        "loading through the symlink must yield the same graph"
    );

    std::fs::remove_dir_all(&root).ok();
}

/// The containment 38a2fecf added is UNCHANGED: an `external_data` location
/// authored inside the model still cannot name a file outside the model's
/// (resolved) directory.
#[test]
fn external_data_location_still_cannot_escape_the_model_directory() {
    let root = scratch_dir("escape");
    let model_dir = root.join("onnx");
    std::fs::create_dir_all(&model_dir).expect("model dir");
    std::fs::write(root.join("secret.bin"), vec![0_u8; 4]).expect("secret");

    // A model whose single initializer claims external data at "../secret.bin".
    // Hand-rolled like above; only the initializer differs.
    let mut file = std::fs::File::create(model_dir.join("m.onnx")).expect("create");
    // TensorProto { dims:[1], data_type:1, name:"w", data_location:1,
    //               external_data:[{key:"location", value:"../secret.bin"}] }
    let mut kv = Vec::new();
    kv.extend_from_slice(&[0x0a, 0x08]);
    kv.extend_from_slice(b"location");
    kv.push(0x12);
    kv.push(13);
    kv.extend_from_slice(b"../secret.bin");
    let mut tensor = Vec::new();
    tensor.extend_from_slice(&[0x08, 0x01]); // dims: 1
    tensor.extend_from_slice(&[0x10, 0x01]); // data_type: FLOAT
    tensor.push(0x3a);
    tensor.push(1);
    tensor.extend_from_slice(b"w"); // name
    tensor.push(0x6a); // field 13 (external_data), wire 2
    tensor.push(kv.len() as u8);
    tensor.extend_from_slice(&kv);
    tensor.extend_from_slice(&[0x70, 0x01]); // data_location: EXTERNAL
    let mut graph = Vec::new();
    graph.push(0x2a); // field 5 (initializer), wire 2
    graph.push(tensor.len() as u8);
    graph.extend_from_slice(&tensor);
    graph.extend_from_slice(&[0x12, 0x01, b'g']); // name
    let mut model = Vec::new();
    model.extend_from_slice(&[0x08, 0x08]); // ir_version 8
    model.push(0x3a); // field 7 (graph), wire 2
    model.push(graph.len() as u8);
    model.extend_from_slice(&graph);
    file.write_all(&model).expect("write");
    drop(file);

    let error = load_onnx(model_dir.join("m.onnx"))
        .expect_err("an external location outside the model directory must be refused");
    let text = error.to_string();
    assert!(
        !text.contains("secret"),
        "the escaping location must not have been opened: {text}"
    );

    std::fs::remove_dir_all(&root).ok();
}
