// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #la-warm-frontier — ARRIVAL on the real scored entry point.
//!
//! `bab.la_warm_frontier` is a plain `BetaCrownConfig` field written by
//! `preset::apply`; nothing reads an environment variable for it. This test
//! proves the whole chain on `ny vnncomp v1` (the `router_shadow.rs` /
//! `flight_lever_receipt.rs` pattern: a temporary `configs/vnncomp99/<cat>.yaml`,
//! no `benchmarks/` dependency): a two-head RESIDUAL ReLU network (every hidden
//! ReLU sits under a skip, so the #warm-dag guard refuses every single-block
//! seed), a CONJUNCTIVE two-constraint unsafe region (`Y_0 <= t0` AND
//! `Y_1 <= t1`: `verify/graph.rs` hands >= 2 constraints of one clause to the
//! conjunctive multi-objective ReLU-split lane; a disjunction of single-
//! constraint clauses is decomposed upfront into per-clause SINGLE-objective
//! runs (`route=auto_disjunction`), a lane that carries no lA cache at all),
//! and
//!
//! * the ARMED yaml prints the lane's own observable
//!   `[la-warm-frontier] lane=multi_objective armed=1 captured=N …` with the
//!   root capture engaged (`captured>0`);
//! * the same yaml WITHOUT the key prints no `[la-warm-frontier]` line at all;
//! * the two published verdicts never contradict (one `sat` against one
//!   `unsat` would be a soundness emergency).
//!
//! The child inherits the test's environment unchanged, and this file reads
//! and writes no environment variable (so it adds no site to the
//! `source_frontier` ratchet). Run it from a clean environment
//! (`env -i PATH=… HOME=… RUSTUP_HOME=… CARGO_HOME=…`) for the zero-`NY_*`
//! arrival proof; the review runner does, adding back only the scored-fixture
//! process-hygiene switch that keeps no out-of-process watchdog `ny` alive
//! after the child exits. No environment variable routes to this key.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use ny_test_utils::workspace_root;

const ARMED_CATEGORY: &str = "la_warm_frontier_armed_2026";
const DARK_CATEGORY: &str = "la_warm_frontier_dark_2026";
const ARMED_YAML: &str = "bab:\n  la_warm_frontier: true\n  branching:\n    method: relu\n";
const DARK_YAML: &str = "bab:\n  branching:\n    method: relu\n";

/// Dyadic weights in `[-1, 1]` (multiples of 1/8, exact in f32).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let k = ((self.0 >> 33) % 17) as i32 - 8;
        k as f32 / 8.0
    }

    fn take(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// `x[3] -> z1 = W1 x + b1 (4) -> h1 = relu(z1) -> z2 = W2 h1 + b2 -> h2 = relu(z2)
///  -> a1 = h2 + z1 -> z3 = W3 a1 + b3 -> h3 = relu(z3) -> a2 = h3 + a1
///  -> y = Wo a2 + bo (2)`.
struct Net {
    layers: Vec<(Vec<f32>, Vec<f32>, usize, usize)>,
}

impl Net {
    fn new(seed: u64) -> Self {
        let mut rng = Lcg(seed);
        let mut layer = |rows: usize, cols: usize| {
            let w = rng.take(rows * cols);
            let b = rng.take(rows).into_iter().map(|v| v * 0.5).collect();
            (w, b, rows, cols)
        };
        Self {
            layers: vec![layer(4, 3), layer(4, 4), layer(4, 4), layer(2, 4)],
        }
    }

    fn affine(layer: &(Vec<f32>, Vec<f32>, usize, usize), x: &[f64]) -> Vec<f64> {
        let (w, b, rows, cols) = layer;
        (0..*rows)
            .map(|i| {
                f64::from(b[i])
                    + (0..*cols)
                        .map(|j| f64::from(w[i * cols + j]) * x[j])
                        .sum::<f64>()
            })
            .collect()
    }

    fn eval(&self, x: &[f64]) -> Vec<f64> {
        let relu = |v: &[f64]| v.iter().map(|z| z.max(0.0)).collect::<Vec<_>>();
        let add = |p: &[f64], q: &[f64]| p.iter().zip(q).map(|(a, b)| a + b).collect::<Vec<_>>();
        let z1 = Self::affine(&self.layers[0], x);
        let h2 = relu(&Self::affine(&self.layers[1], &relu(&z1)));
        let a1 = add(&h2, &z1);
        let h3 = relu(&Self::affine(&self.layers[2], &a1));
        let a2 = add(&h3, &a1);
        Self::affine(&self.layers[3], &a2)
    }

    /// Grid minimum of each head over `[-1, 1]^3` (an upper estimate of the
    /// true minimum).
    fn grid_min(&self) -> [f64; 2] {
        let n = 60;
        let mut min = [f64::INFINITY; 2];
        for i in 0..=n {
            for j in 0..=n {
                for k in 0..=n {
                    let x = [i, j, k].map(|t| -1.0 + 2.0 * f64::from(t) / f64::from(n));
                    let y = self.eval(&x);
                    min[0] = min[0].min(y[0]);
                    min[1] = min[1].min(y[1]);
                }
            }
        }
        min
    }

    fn onnx(&self) -> Vec<u8> {
        use ny_onnx::onnx_proto::tensor_shape_proto::{dimension, Dimension};
        use ny_onnx::onnx_proto::{
            AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
            TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
        };
        use prost::Message;
        let vi = |name: &str, dims: &[i64]| ValueInfoProto {
            name: name.to_owned(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: 1,
                    shape: Some(TensorShapeProto {
                        dim: dims
                            .iter()
                            .map(|&d| Dimension {
                                value: Some(dimension::Value::DimValue(d)),
                            })
                            .collect(),
                    }),
                }),
            }),
        };
        let tensor = |name: &str, dims: &[i64], v: &[f32]| TensorProto {
            name: name.to_owned(),
            dims: dims.to_vec(),
            data_type: 1,
            float_data: v.to_vec(),
            ..Default::default()
        };
        let gemm = |input: &str, index: usize, out: &str| NodeProto {
            op_type: "Gemm".to_owned(),
            input: vec![input.to_owned(), format!("W{index}"), format!("B{index}")],
            output: vec![out.to_owned()],
            attribute: vec![AttributeProto {
                name: "transB".to_owned(),
                i: Some(1),
                r#type: 2,
                ..Default::default()
            }],
            ..Default::default()
        };
        let unary = |op: &str, input: &str, out: &str| NodeProto {
            op_type: op.to_owned(),
            input: vec![input.to_owned()],
            output: vec![out.to_owned()],
            ..Default::default()
        };
        let add = |a: &str, b: &str, out: &str| NodeProto {
            op_type: "Add".to_owned(),
            input: vec![a.to_owned(), b.to_owned()],
            output: vec![out.to_owned()],
            ..Default::default()
        };
        let mut initializer = Vec::new();
        for (index, (w, b, rows, cols)) in self.layers.iter().enumerate() {
            initializer.push(tensor(
                &format!("W{index}"),
                &[*rows as i64, *cols as i64],
                w,
            ));
            initializer.push(tensor(&format!("B{index}"), &[*rows as i64], b));
        }
        let model = ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            graph: Some(GraphProto {
                name: "la_warm_frontier_residual".to_owned(),
                node: vec![
                    gemm("x", 0, "z1"),
                    unary("Relu", "z1", "h1"),
                    gemm("h1", 1, "z2"),
                    unary("Relu", "z2", "h2"),
                    add("h2", "z1", "a1"),
                    gemm("a1", 2, "z3"),
                    unary("Relu", "z3", "h3"),
                    add("h3", "a1", "a2"),
                    gemm("a2", 3, "y"),
                ],
                initializer,
                input: vec![vi("x", &[1, 3])],
                output: vec![vi("y", &[1, 2])],
                ..Default::default()
            }),
            ..Default::default()
        };
        model.encode_to_vec()
    }
}

/// The unsafe region `Y_0 <= t0 AND Y_1 <= t1` over `X in [-1, 1]^3` (one
/// clause, two constraints: the conjunctive multi-objective lane).
fn property(thresholds: [f64; 2]) -> String {
    let mut text = String::new();
    for i in 0..3 {
        text.push_str(&format!("(declare-const X_{i} Real)\n"));
    }
    for i in 0..2 {
        text.push_str(&format!("(declare-const Y_{i} Real)\n"));
    }
    for i in 0..3 {
        text.push_str(&format!(
            "(assert (>= X_{i} -1.0))\n(assert (<= X_{i} 1.0))\n"
        ));
    }
    text.push_str(&format!(
        "(assert (<= Y_0 {:.6}))\n(assert (<= Y_1 {:.6}))\n",
        thresholds[0], thresholds[1]
    ));
    text
}

struct Run {
    stderr: String,
    result: String,
}

fn run(configs: &Path, category: &str, model: &Path, property: &Path, root: &Path) -> Run {
    let result = root.join(format!("{category}.result"));
    let mut command = Command::new(env!("CARGO_BIN_EXE_ny"));
    command.current_dir(workspace_root()).args([
        "vnncomp",
        "v1",
        category,
        model.to_str().expect("model path UTF-8"),
        property.to_str().expect("property path UTF-8"),
        result.to_str().expect("result path UTF-8"),
        "20",
        "--configs-dir",
        configs.to_str().expect("configs path UTF-8"),
    ]);
    let output = command.output().expect("run the scored fixture");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "{category}: status={}\nstdout={}\nstderr={stderr}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
    );
    let result = fs::read_to_string(&result).expect("the result file is written");
    Run { stderr, result }
}

fn frontier_lines(run: &Run) -> Vec<&str> {
    run.stderr
        .lines()
        .filter(|line| line.starts_with("[la-warm-frontier]"))
        .collect()
}

fn field(line: &str, key: &str) -> Option<u64> {
    line.split_whitespace()
        .find_map(|token| token.strip_prefix(&format!("{key}=")))
        .and_then(|value| value.parse().ok())
}

fn verdict(result: &str) -> &str {
    result.lines().next().unwrap_or("").trim()
}

#[ntest::timeout(600000)]
#[test]
fn la_warm_frontier_yaml_key_arrives_on_the_scored_multi_objective_lane() {
    let root = tempfile::tempdir().expect("fixture root");
    let configs: PathBuf = root.path().join("configs");
    let year = configs.join("vnncomp99");
    fs::create_dir_all(&year).expect("config tree");
    fs::write(year.join(format!("{ARMED_CATEGORY}.yaml")), ARMED_YAML).expect("armed yaml");
    fs::write(year.join(format!("{DARK_CATEGORY}.yaml")), DARK_YAML).expect("dark yaml");

    let net = Net::new(0x1a2b_3c4d);
    let model = root.path().join("residual.onnx");
    fs::write(&model, net.onnx()).expect("write the model");
    // Just below each head's grid minimum: tight enough that the root bound
    // is not expected to close, so the lane captures (and may branch).
    let min = net.grid_min();
    let spread = 0.15;
    let prop = root.path().join("property.vnnlib");
    fs::write(&prop, property([min[0] - spread, min[1] - spread])).expect("property");

    let armed = run(&configs, ARMED_CATEGORY, &model, &prop, root.path());
    let dark = run(&configs, DARK_CATEGORY, &model, &prop, root.path());

    let armed_lines = frontier_lines(&armed);
    for line in &armed_lines {
        println!("[la-warm-frontier-arrival] armed: {line}");
    }
    println!(
        "[la-warm-frontier-arrival] verdicts armed={} dark={} grid_min={min:?}",
        verdict(&armed.result),
        verdict(&dark.result)
    );
    let lane = armed_lines
        .iter()
        .find(|line| line.starts_with("[la-warm-frontier] lane=multi_objective armed=1 "))
        .unwrap_or_else(|| {
            panic!(
                "the armed yaml never reached the multi-objective lane (no \
                 `[la-warm-frontier] lane=multi_objective armed=1` line):\n{}",
                armed.stderr
            )
        });
    let captured = field(lane, "captured").expect("captured= field");
    assert!(
        captured > 0,
        "the armed lane captured no frontier at the root: {lane}"
    );
    assert!(
        frontier_lines(&dark).is_empty(),
        "a yaml without the key printed the observable: {:?}",
        frontier_lines(&dark)
    );
    let (a, d) = (verdict(&armed.result), verdict(&dark.result));
    assert!(
        !((a == "sat" && d == "unsat") || (a == "unsat" && d == "sat")),
        "SOUNDNESS EMERGENCY: armed={a} dark={d}"
    );
    println!(
        "[la-warm-frontier-arrival] TEST_COMPLETE captured={captured} resumed={}",
        field(lane, "resumed").unwrap_or(0)
    );
}
