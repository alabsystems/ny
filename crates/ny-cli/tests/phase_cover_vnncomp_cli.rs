// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Real scored entry point, original scalar VNN-LIB, actual AY batch helper,
//! private Source replay and ordinary G2/G3 publication on authored cycles.
#![cfg(feature = "mip")]

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use ny_onnx::onnx_proto::{
    tensor_shape_proto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use prost::Message;

// The original scalar-V1 adapter authenticates exactly 30 inputs. Seven
// coordinates carry the cycle; the other authored coordinates have zero
// weights. Keep that actual admission boundary in the scored-route control.
const INPUTS: usize = 30;

fn info(name: &str, width: i64) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: [1, width]
                        .into_iter()
                        .map(|n| tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(n)),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

fn tensor(name: &str, dims: &[i64], values: &[f32]) -> TensorProto {
    TensorProto {
        name: name.into(),
        dims: dims.to_vec(),
        data_type: 1,
        raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ..Default::default()
    }
}

fn model(bias: f32, positive_duplicates: bool, disjoint: bool) -> ModelProto {
    let heads = if positive_duplicates || disjoint {
        14
    } else {
        7
    };
    let mut first = vec![0.0f32; INPUTS * heads];
    let mut final_weight = vec![0.0f32; 2 * heads];
    for j in 0..7 {
        first[j * heads + j] = 1.0;
        first[((j + 1) % 7) * heads + j] = -1.0;
        final_weight[2 * j] = -1.0;
        if positive_duplicates {
            first[j * heads + 7 + j] = 1.0;
            first[((j + 1) % 7) * heads + 7 + j] = -1.0;
            final_weight[2 * (7 + j)] = 0.25;
        }
        if disjoint {
            first[(7 + j) * heads + 7 + j] = 1.0;
            first[(7 + (j + 1) % 7) * heads + 7 + j] = -1.0;
            final_weight[2 * (7 + j)] = -1.0;
        }
    }
    let mut graph = GraphProto {
        name: "authored_seven_cycle".into(),
        input: vec![info("x", INPUTS as i64)],
        output: vec![info("output", 2)],
        initializer: vec![
            tensor("w1", &[INPUTS as i64, heads as i64], &first),
            tensor("b1", &[heads as i64], &vec![0.0; heads]),
            tensor("w2", &[heads as i64, 2], &final_weight),
            tensor("b2", &[2], &[bias, 0.0]),
        ],
        ..Default::default()
    };
    for (op, input, output) in [
        ("MatMul", vec!["x", "w1"], "pre_bias"),
        ("Add", vec!["pre_bias", "b1"], "pre_relu"),
        ("Relu", vec!["pre_relu"], "relu"),
        ("MatMul", vec!["relu", "w2"], "final_product"),
        ("Add", vec!["final_product", "b2"], "output"),
    ] {
        graph.node.push(NodeProto {
            op_type: op.into(),
            input: input.into_iter().map(str::to_owned).collect(),
            output: vec![output.into()],
            ..Default::default()
        });
    }
    ModelProto {
        ir_version: 7,
        graph: Some(graph),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        ..Default::default()
    }
}

fn property() -> String {
    let mut text = String::new();
    for j in 0..INPUTS {
        text.push_str(&format!("(declare-const X_{j} Real)\n"));
    }
    text.push_str("(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n");
    for j in 0..INPUTS {
        text.push_str(&format!("(assert (>= X_{j} -1))\n(assert (<= X_{j} 1))\n"));
    }
    text.push_str("(assert (<= Y_0 Y_1))\n");
    text
}

fn run(root: &Path, bias: f32, positive_duplicates: bool) -> (String, String) {
    run_mode(root, bias, positive_duplicates, false, false)
}

fn run_mode(
    root: &Path,
    bias: f32,
    positive_duplicates: bool,
    disjoint: bool,
    default_directional: bool,
) -> (String, String) {
    let configs = root.join("configs/vnncomp99");
    fs::create_dir_all(&configs).unwrap();
    let policy = if default_directional {
        String::new()
    } else {
        "  head_groups: true\n  head_directional_cuts: false\n  head_phase_cover: 7\n".into()
    };
    fs::write(
        configs.join("phase_cover_cycle.yaml"),
        format!("margin_row:\n{policy}  gpu: false\n  gpu_batch: false\n  gpu_backward: false\n"),
    )
    .unwrap();
    let onnx = root.join("original.onnx");
    let vnnlib = root.join("original.vnnlib");
    let result = root.join("result.txt");
    fs::write(
        &onnx,
        model(bias, positive_duplicates, disjoint).encode_to_vec(),
    )
    .unwrap();
    fs::write(&vnnlib, property()).unwrap();
    // Regular output files avoid a full pipe blocking the child before the
    // watchdog can reap it. This is a functional test, not a timing claim.
    let mut command = Command::new(env!("CARGO_BIN_EXE_ny"));
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("NY_") {
            command.env_remove(key);
        }
    }
    let mut child = command
        .args(["vnncomp", "v1", "phase_cover_cycle"])
        .arg(&onnx)
        .arg(&vnnlib)
        .arg(&result)
        .args(["20", "--configs-dir"])
        .arg(root.join("configs"))
        .args(["--proof-profile", "competition"])
        .env("NY_VNNCOMP_EXTERNAL_WATCHDOG", "0")
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("RAYON_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .stdin(Stdio::null())
        .stdout(fs::File::create(root.join("stdout.log")).unwrap())
        .stderr(fs::File::create(root.join("stderr.log")).unwrap())
        .spawn()
        .expect("spawn actual VNN-COMP entry point");
    let deadline = Instant::now() + Duration::from_secs(45);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("VNN-COMP functional control exceeded watchdog");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let stderr = format!(
        "{}\nstdout:\n{}",
        fs::read_to_string(root.join("stderr.log")).unwrap(),
        fs::read_to_string(root.join("stdout.log")).unwrap()
    );
    assert!(status.success(), "{status}\n{stderr}");
    (fs::read_to_string(result).unwrap(), stderr)
}

#[test]
fn default_directional_intersection_retains_native_closure_and_original_g3() {
    let root = tempfile::tempdir().unwrap();
    let (result, stderr) = run_mode(root.path(), 12.5, false, true, true);
    assert!(result.starts_with("unsat"), "{result}\n{stderr}");
    assert!(
        stderr.contains("[g3] lane=margin_row_input_head"),
        "{stderr}"
    );
    let line = stderr
        .lines()
        .find(|line| line.starts_with("[input-head-directional]"))
        .unwrap_or_else(|| panic!("default directional path did not engage (VOID):\n{stderr}"));
    assert!(
        field(line, "calls").parse::<usize>().unwrap() <= 3,
        "{line}"
    );
    assert!(field(line, "cuts").parse::<usize>().unwrap() >= 2, "{line}");
    assert!(
        field(line, "native_replayed").parse::<usize>().unwrap() > 0,
        "{line}"
    );
    let before = f64::from_bits(u64::from_str_radix(field(line, "before_bits"), 16).unwrap());
    let after = f64::from_bits(u64::from_str_radix(field(line, "after_bits"), 16).unwrap());
    assert!(before < 0.0 && after > 0.49 && after <= 0.5, "{line}");
}

#[test]
fn default_directional_intersection_never_certifies_unsafe_two_cycles() {
    let root = tempfile::tempdir().unwrap();
    let (result, stderr) = run_mode(root.path(), 11.5, false, true, true);
    assert!(!result.starts_with("unsat"), "{result}\n{stderr}");
    assert!(
        !stderr.contains("[g3] lane=margin_row_input_head"),
        "{stderr}"
    );
    let line = stderr
        .lines()
        .find(|line| line.starts_with("[input-head-directional]"))
        .unwrap_or_else(|| panic!("unsafe directional path did not engage (VOID):\n{stderr}"));
    assert!(field(line, "cuts").parse::<usize>().unwrap() >= 2, "{line}");
    // The unsafe sibling stays open after native replay, so this scored path
    // must still exercise the actual bounded AY helper and Source dual replay.
    assert!(field(line, "calls").parse::<usize>().unwrap() > 0, "{line}");
    assert!(
        field(line, "ay_replayed").parse::<usize>().unwrap() > 0,
        "{line}"
    );
    let after = f64::from_bits(u64::from_str_radix(field(line, "after_bits"), 16).unwrap());
    assert!(after <= -0.5, "{line}");
}

fn field<'a>(line: &'a str, name: &str) -> &'a str {
    line.split_whitespace()
        .find_map(|word| word.strip_prefix(name).and_then(|s| s.strip_prefix('=')))
        .unwrap_or_else(|| panic!("missing {name}: {line}"))
}

#[test]
fn native_phase_cover_closes_seven_cycle_through_original_vnncomp_and_g3() {
    let root = tempfile::tempdir().unwrap();
    let (result, stderr) = run(root.path(), 6.5, false);
    assert!(result.starts_with("unsat"), "{result}\n{stderr}");
    assert!(
        stderr.contains("property_format=scalar_v1_real"),
        "{stderr}"
    );
    assert!(
        stderr.contains("[g3] lane=margin_row_input_head"),
        "{stderr}"
    );
    let phase = stderr
        .lines()
        .find(|line| line.starts_with("[input-head-phase-cover]"))
        .unwrap_or_else(|| panic!("phase cover did not engage (VOID):\n{stderr}"));
    assert_eq!(field(phase, "calls"), "0", "{phase}");
    assert_eq!(field(phase, "native_replayed"), "128", "{phase}");
    assert_eq!(field(phase, "ay_replayed"), "0", "{phase}");
    assert_eq!(field(phase, "leaves"), "128", "{phase}");
    assert_eq!(field(phase, "replayed"), "128", "{phase}");
    let delta = stderr
        .lines()
        .find(|line| line.starts_with("[input-head-phase-cover-delta]"))
        .unwrap_or(phase);
    let before = f64::from_bits(u64::from_str_radix(field(delta, "before_bits"), 16).unwrap());
    let after = f64::from_bits(u64::from_str_radix(field(delta, "after_bits"), 16).unwrap());
    assert!(before <= 0.0, "old portfolio already closed: {delta}");
    assert!(after > 0.0 && after <= 0.5, "{delta}");
}

#[test]
fn unsafe_seven_cycle_cannot_publish_unsat_on_actual_scored_entry_point() {
    // The alternating endpoint has sum ReLU(x_j-x_next)=6, hence the
    // original stored f32 model has margin exactly -0.5 at this real point.
    let vertex = [-1i32, 1, -1, 1, -1, 1, -1];
    let sum: i32 = (0..7)
        .map(|j| (vertex[j] - vertex[(j + 1) % 7]).max(0))
        .sum();
    assert_eq!(sum, 6);
    let root = tempfile::tempdir().unwrap();
    let (result, stderr) = run(root.path(), 5.5, false);
    assert!(
        !result.starts_with("unsat"),
        "unsafe source published unsat: {result}\n{stderr}"
    );
    assert!(
        !stderr.contains("[g3] lane=margin_row_input_head"),
        "{stderr}"
    );
}

#[test]
fn actual_ay_phase_cover_combines_selected_phases_with_unselected_positive_heads() {
    // Fourteen genuine heads: negative cycle plus positive duplicate ReLUs.
    // The latter cost +1/4 each. The exact original minimum is 4.75-6*3/4=.25.
    // Native negative-head caps ignore the positive coupling and leave some
    // leaves open. Actual AY hints must recover it through Source weak duals.
    let root = tempfile::tempdir().unwrap();
    let (result, stderr) = run(root.path(), 4.75, true);
    assert!(result.starts_with("unsat"), "{result}\n{stderr}");
    assert!(
        stderr.contains("[g3] lane=margin_row_input_head"),
        "{stderr}"
    );
    let phase = stderr
        .lines()
        .find(|line| line.starts_with("[input-head-phase-cover]"))
        .unwrap_or_else(|| panic!("phase cover did not engage (VOID):\n{stderr}"));
    assert_eq!(field(phase, "calls"), "1", "{phase}");
    assert_eq!(field(phase, "leaves"), "128", "{phase}");
    assert_eq!(field(phase, "replayed"), "128", "{phase}");
    assert!(
        field(phase, "ay_replayed").parse::<usize>().unwrap() > 0,
        "no actual AY leaf replay: {phase}"
    );
    let delta = stderr
        .lines()
        .find(|line| line.starts_with("[input-head-phase-cover-delta]"))
        .unwrap_or(phase);
    let before = f64::from_bits(u64::from_str_radix(field(delta, "before_bits"), 16).unwrap());
    let after = f64::from_bits(u64::from_str_radix(field(delta, "after_bits"), 16).unwrap());
    assert!(before <= 0.0, "old portfolio already closed: {delta}");
    assert!(after > 0.0 && after <= 0.25, "{delta}");
}
