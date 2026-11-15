// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-lane — the lowdim lane on the SHIPPED `lsnc_relu` model (2026 corpus,
//! `1.0`), through its LANE-PRIVATE interval twin.
//!
//! The unit tests in `ny_propagate::lowdim::lsnc_tests` pin each lowering on
//! synthetic graphs; this file pins what only the real graph can:
//!
//! 1. the real graph, loaded exactly as `ny-cli` loads it, still carries its
//!    `OpaqueSkip` nodes and the lane DECLINES it; the twin admits it (four
//!    `IntervalConst` nodes, two McCormick products), and building the twin
//!    leaves the global interval carrier dark;
//! 2. SAMPLED CONTAINMENT: on 256 boxes (root, random sub-boxes of every
//!    width down to points, boxes around the ORT-confirmed dV counterexample),
//!    the lane's published bounds of every output, both directions, contain an
//!    INDEPENDENT exact-f64 forward of the real ONNX (a protobuf interpreter
//!    written here, constants folded in f64 from the stored f32 leaves — it
//!    shares no code with ny's loader or the lane) at >= 200,000 points;
//! 3. the row-2 ROOT upper bound of `Y_0 = dV` is at or above the maximum of
//!    ONNX Runtime samples of the ORIGINAL model over the row's box;
//! 4. #lsnc-lane stage 2, the JOINT QUADRATIC BOUND at the two
//!    `ReduceSum(Mul)` sites (`V` and `dV`): on >= 200,000 points over random
//!    sub-boxes of the real row-2 / row-79 box, both site intervals, both
//!    sites' joint PLANES (from the same pass) and the published bounds of
//!    `Y_0` / `Y_1` contain the independent exact forward; and the row-2 root
//!    `dV` upper bound is PINNED to the numpy prototype's `allmax` value;
//! 5. #lsnc-lane stage 3, the SEARCH stack (clip, `sbwr` split, parallel
//!    waves): every clip plane of a box contains the exact forward, the
//!    relaxed clip never cuts a sampled point that satisfies the whole
//!    conjunction, and the armed stack never proves a clause the exact
//!    forward attains (at the ORT-confirmed dV counterexample) while it does
//!    prove an impossible one. (The throughput benchmark is a separate,
//!    feature-gated target: `tests/lsnc_lane_bench.rs`.)
//!
//! Registered in the DEFAULT suite and SKIPS LOUDLY without the corpus: every
//! test prints the `LSNC_LANE_CORPUS_SKIPPED` banner and returns. A gate that
//! greps for the banner can tell VOID from PASS; one that does not, cannot.
//! Point the tree at the corpus with the 2026 benchmark-root override
//! (`common::BenchYear::Vnncomp2026`).

use std::collections::HashMap;
use std::path::PathBuf;

use ndarray::{ArrayD, IxDyn};
use ny_onnx::onnx_proto::{ModelProto, NodeProto, TensorProto};
use ny_onnx::{
    load_onnx_with_config, CompoundNodePolicy, GraphNetworkOptions, OnnxLoadConfig,
    ShapeInferBackend,
};
use ny_propagate::lowdim::{
    admit, bound_boxes, clip_report, joint_root_report, joint_site_boxes, lowered_point_values,
    prove_conjunction_admitted, LaneAdmission, LowDimConfig, LowDimOutcome, LowDimParallel,
    LowDimSplit,
};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;
use prost::Message;

mod common;
use common::{try_benchmark_root, BenchYear};

/// The official input box of every `lsnc_relu` row (both years).
const BOX_LO: [f32; 6] = [-0.7, -0.7, -1.0, -3.0, -3.0, -2.0];
const BOX_HI: [f32; 6] = [0.7, 0.7, 1.0, 3.0, 3.0, 2.0];

/// The ORT-confirmed dV counterexample on file (`lsnc_interval_carrier.rs`):
/// `Y_0 = 2.8978e-4` there.
const DV_WITNESS: [f32; 6] = [
    -0.337_032_47,
    -0.121_915_82,
    -0.251_726_03,
    -0.744_893_8,
    2.086_692,
    -0.230_242_04,
];

fn skip_loudly(test: &str, reason: &str) {
    use std::io::Write as _;
    let banner = format!(
        "\n=================== LSNC_LANE_CORPUS_SKIPPED ===================\n\
         test: {test}\nreason: {reason}\n\
         This soundness test DID NOT RUN. Its `ok` is VOID, not a pass.\n\
         Point the 2026 benchmark-root override at <ny>/benchmarks/vnncomp2026/benchmarks.\n\
         ================================================================\n"
    );
    // Written to the RAW stderr handle, not through `eprintln!`: libtest's
    // output capture intercepts only the `print!` family and DISCARDS the
    // captured text of a PASSING test, so an `eprintln!` banner is invisible to
    // any gate that runs without `--nocapture` / `--show-output` — the skipped
    // test would print a bare `... ok` and a VOID would read as a PASS, which
    // is the exact trap this banner exists to close. A direct handle write
    // bypasses the capture and always reaches the log.
    let mut err = std::io::stderr().lock();
    let _ = err.write_all(banner.as_bytes());
    let _ = err.flush();
}

/// The 2026 `1.0` corpus directory, or a loud skip.
fn corpus(test: &str) -> Option<PathBuf> {
    match try_benchmark_root(BenchYear::Vnncomp2026) {
        Ok(root) => {
            let dir = root.join("lsnc_relu/1.0");
            if dir.join("onnx/relu_quadrotor2d_state.onnx").is_file() {
                Some(dir)
            } else {
                skip_loudly(test, &format!("{} has no lsnc_relu model", dir.display()));
                None
            }
        }
        Err(reason) => {
            skip_loudly(test, &reason);
            None
        }
    }
}

fn load_config() -> OnnxLoadConfig {
    OnnxLoadConfig::default().with_shape_infer_backend(ShapeInferBackend::InProcess)
}

fn graph_options() -> GraphNetworkOptions {
    GraphNetworkOptions {
        compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
        ..GraphNetworkOptions::default()
    }
}

fn root_box() -> BoundedTensor {
    let lo = ArrayD::from_shape_vec(IxDyn(&[1, 6]), BOX_LO.to_vec()).expect("shape");
    let hi = ArrayD::from_shape_vec(IxDyn(&[1, 6]), BOX_HI.to_vec()).expect("shape");
    BoundedTensor::new(lo, hi).expect("box")
}

/// The real graph with its lane twin attached, and the lane's admission.
fn admitted(dir: &std::path::Path) -> (GraphNetwork, LaneAdmission) {
    let path = dir.join("onnx/relu_quadrotor2d_state.onnx");
    let model = load_onnx_with_config(&path, &load_config()).expect("the lsnc model loads");
    let mut graph = model
        .to_graph_network_with_options(graph_options())
        .expect("the lsnc model lowers to a graph");
    let twin = ny_onnx::lowdim_twin::build_lane_twin(&path, &load_config(), graph_options())
        .unwrap_or_else(|reason| panic!("the lsnc lane twin must build: {reason}"));
    graph.attach_lowdim_twin(twin);
    let admission = admit(&graph, &root_box())
        .unwrap_or_else(|reason| panic!("the twin must be admitted: {reason}"));
    (graph, admission)
}

// ---------------------------------------------------------------------------
// An INDEPENDENT exact-f64 ONNX forward (no ny loader, no lane code)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct T {
    shape: Vec<usize>,
    data: Vec<f64>,
}

fn decode(t: &TensorProto) -> T {
    let shape: Vec<usize> = t
        .dims
        .iter()
        .map(|&d| usize::try_from(d).expect("non-negative dim"))
        .collect();
    let raw = &t.raw_data;
    let data: Vec<f64> = match t.data_type {
        1 if !raw.is_empty() => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f64::from(f32::from_le_bytes(*c)))
            .collect(),
        1 => t.float_data.iter().map(|&v| f64::from(v)).collect(),
        7 if !raw.is_empty() => raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c) as f64)
            .collect(),
        7 => t.int64_data.iter().map(|&v| v as f64).collect(),
        11 if !raw.is_empty() => raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect(),
        11 => t.double_data.clone(),
        other => panic!("dtype {other} is not in the lsnc model"),
    };
    assert_eq!(
        data.len(),
        shape.iter().product::<usize>(),
        "tensor {}",
        t.name
    );
    T { shape, data }
}

fn attr<'a>(node: &'a NodeProto, name: &str) -> Option<&'a ny_onnx::onnx_proto::AttributeProto> {
    node.attribute.iter().find(|a| a.name == name)
}

fn attr_i(node: &NodeProto, name: &str, default: i64) -> i64 {
    attr(node, name).map_or(default, |a| a.i_value())
}

fn attr_f(node: &NodeProto, name: &str, default: f64) -> f64 {
    attr(node, name).map_or(default, |a| f64::from(a.f_value()))
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// numpy broadcasting of two shapes.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| {
            let da = if i + a.len() >= n {
                a[i + a.len() - n]
            } else {
                1
            };
            let db = if i + b.len() >= n {
                b[i + b.len() - n]
            } else {
                1
            };
            assert!(da == db || da == 1 || db == 1, "broadcast {a:?} vs {b:?}");
            if da == 1 {
                db
            } else {
                da
            }
        })
        .collect()
}

/// Strides of `shape` right-aligned into `out`, zero along broadcast axes.
fn broadcast_strides(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let off = out.len() - shape.len();
    let own = strides(shape);
    let mut st = vec![0usize; out.len()];
    for i in 0..shape.len() {
        st[off + i] = if shape[i] == 1 { 0 } else { own[i] };
    }
    st
}

fn binary(a: &T, b: &T, f: impl Fn(f64, f64) -> f64) -> T {
    let shape = broadcast_shape(&a.shape, &b.shape);
    let (sa, sb) = (
        broadcast_strides(&a.shape, &shape),
        broadcast_strides(&b.shape, &shape),
    );
    let total: usize = shape.iter().product();
    let mut idx = vec![0usize; shape.len()];
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        let ia: usize = idx.iter().zip(&sa).map(|(i, s)| i * s).sum();
        let ib: usize = idx.iter().zip(&sb).map(|(i, s)| i * s).sum();
        data.push(f(a.data[ia], b.data[ib]));
        for k in (0..shape.len()).rev() {
            idx[k] += 1;
            if idx[k] < shape[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    T { shape, data }
}

fn unary(a: &T, f: impl Fn(f64) -> f64) -> T {
    T {
        shape: a.shape.clone(),
        data: a.data.iter().map(|&v| f(v)).collect(),
    }
}

/// `alpha * op(A) @ op(B)` for 2-D `A`, `B`.
fn matmul2(a: &T, b: &T, ta: bool, tb: bool, alpha: f64) -> T {
    assert_eq!((a.shape.len(), b.shape.len()), (2, 2), "2-D matmul");
    let (m, k) = if ta {
        (a.shape[1], a.shape[0])
    } else {
        (a.shape[0], a.shape[1])
    };
    let (k2, n) = if tb {
        (b.shape[1], b.shape[0])
    } else {
        (b.shape[0], b.shape[1])
    };
    assert_eq!(k, k2, "inner dimensions");
    let at = |i: usize, p: usize| {
        if ta {
            a.data[p * a.shape[1] + i]
        } else {
            a.data[i * a.shape[1] + p]
        }
    };
    let bt = |p: usize, j: usize| {
        if tb {
            b.data[j * b.shape[1] + p]
        } else {
            b.data[p * b.shape[1] + j]
        }
    };
    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let s: f64 = (0..k).map(|p| at(i, p) * bt(p, j)).sum();
            data[i * n + j] = alpha * s;
        }
    }
    T {
        shape: vec![m, n],
        data,
    }
}

fn resolve_axis(axis: i64, rank: usize) -> usize {
    let r = i64::try_from(rank).expect("rank");
    usize::try_from(if axis < 0 { axis + r } else { axis }).expect("axis in range")
}

fn run_op(node: &NodeProto, ins: &[&T]) -> T {
    match node.op_type.as_str() {
        "Mul" => binary(ins[0], ins[1], |a, b| a * b),
        "Add" => binary(ins[0], ins[1], |a, b| a + b),
        "Sub" => binary(ins[0], ins[1], |a, b| a - b),
        "Div" => binary(ins[0], ins[1], |a, b| a / b),
        "Neg" => unary(ins[0], |a| -a),
        "Relu" => unary(ins[0], |a| a.max(0.0)),
        "Cast" => {
            assert_eq!(
                attr_i(node, "to", 1),
                1,
                "only a FLOAT cast is in the model"
            );
            ins[0].clone()
        }
        "Gemm" => {
            let y = matmul2(
                ins[0],
                ins[1],
                attr_i(node, "transA", 0) != 0,
                attr_i(node, "transB", 0) != 0,
                attr_f(node, "alpha", 1.0),
            );
            match ins.get(2) {
                Some(c) => {
                    let beta = attr_f(node, "beta", 1.0);
                    binary(&y, &unary(c, |v| beta * v), |a, b| a + b)
                }
                None => y,
            }
        }
        "MatMul" => {
            let (a, b) = (ins[0], ins[1]);
            if a.shape.len() == 1 {
                let a2 = T {
                    shape: vec![1, a.shape[0]],
                    data: a.data.clone(),
                };
                let y = matmul2(&a2, b, false, false, 1.0);
                T {
                    shape: vec![y.shape[1]],
                    data: y.data,
                }
            } else {
                matmul2(a, b, false, false, 1.0)
            }
        }
        "Slice" => {
            let d = ins[0];
            let rank = d.shape.len();
            let (mut lo, mut hi) = (vec![0usize; rank], d.shape.clone());
            for q in 0..ins[1].data.len() {
                let axis = ins.get(3).map_or(q as i64, |a| a.data[q] as i64);
                let ax = resolve_axis(axis, rank);
                let step = ins.get(4).map_or(1, |s| s.data[q] as i64);
                assert_eq!(step, 1, "unit-step slices only");
                let n = d.shape[ax] as i64;
                let clamp = |v: f64| -> usize {
                    let v = v as i64; // saturating: INT64_MAX ends clamp to n
                    let v = if v < 0 { v + n } else { v };
                    usize::try_from(v.clamp(0, n)).expect("clamped")
                };
                lo[ax] = clamp(ins[1].data[q]);
                hi[ax] = clamp(ins[2].data[q]);
            }
            let shape: Vec<usize> = (0..rank).map(|i| hi[i] - lo[i]).collect();
            let (src, total) = (strides(&d.shape), shape.iter().product::<usize>());
            let mut idx = vec![0usize; rank];
            let mut data = Vec::with_capacity(total);
            for _ in 0..total {
                let f: usize = (0..rank).map(|i| (idx[i] + lo[i]) * src[i]).sum();
                data.push(d.data[f]);
                for k in (0..rank).rev() {
                    idx[k] += 1;
                    if idx[k] < shape[k] {
                        break;
                    }
                    idx[k] = 0;
                }
            }
            T { shape, data }
        }
        "Gather" => {
            let (d, ix) = (ins[0], ins[1]);
            let ax = resolve_axis(attr_i(node, "axis", 0), d.shape.len());
            let outer: usize = d.shape[..ax].iter().product();
            let inner: usize = d.shape[ax + 1..].iter().product();
            let dim = d.shape[ax];
            let mut data = Vec::new();
            for o in 0..outer {
                for &j in &ix.data {
                    let j = if j < 0.0 { j + dim as f64 } else { j };
                    let j = j as usize;
                    for i in 0..inner {
                        data.push(d.data[(o * dim + j) * inner + i]);
                    }
                }
            }
            let mut shape = d.shape[..ax].to_vec();
            shape.extend_from_slice(&ix.shape);
            shape.extend_from_slice(&d.shape[ax + 1..]);
            T { shape, data }
        }
        "Concat" => {
            let rank = ins[0].shape.len();
            let ax = resolve_axis(attr_i(node, "axis", 0), rank);
            let outer: usize = ins[0].shape[..ax].iter().product();
            let mut shape = ins[0].shape.clone();
            shape[ax] = ins.iter().map(|t| t.shape[ax]).sum();
            let mut data = Vec::new();
            for o in 0..outer {
                for t in ins {
                    let block: usize = t.shape[ax..].iter().product();
                    data.extend_from_slice(&t.data[o * block..(o + 1) * block]);
                }
            }
            T { shape, data }
        }
        "ReduceSum" => {
            let d = ins[0];
            let rank = d.shape.len();
            let axes: Vec<usize> = ins
                .get(1)
                .map(|a| {
                    a.data
                        .iter()
                        .map(|&v| resolve_axis(v as i64, rank))
                        .collect()
                })
                .unwrap_or_else(|| (0..rank).collect());
            let keep = attr_i(node, "keepdims", 1) != 0;
            let mut kept = d.shape.clone();
            for &a in &axes {
                kept[a] = 1;
            }
            let (src, dst) = (strides(&d.shape), strides(&kept));
            let mut data = vec![0.0; kept.iter().product()];
            for (f, &v) in d.data.iter().enumerate() {
                let o: usize = (0..rank)
                    .map(|i| {
                        let c = f / src[i] % d.shape[i];
                        if axes.contains(&i) {
                            0
                        } else {
                            c * dst[i]
                        }
                    })
                    .sum();
                data[o] += v;
            }
            let shape = if keep {
                kept
            } else {
                (0..rank)
                    .filter(|i| !axes.contains(i))
                    .map(|i| d.shape[i])
                    .collect()
            };
            T { shape, data }
        }
        other => panic!("op {other} is not in the lsnc model"),
    }
}

/// The lsnc graph as an f64 program: constants folded once (every node whose
/// inputs are all constant), the rest evaluated per batch with the batch as
/// the ONNX leading dimension.
struct OnnxF64 {
    consts: HashMap<String, T>,
    live: Vec<NodeProto>,
    input: String,
    output: String,
}

impl OnnxF64 {
    fn load(path: &std::path::Path) -> Self {
        let bytes = ny_onnx::read_onnx_bytes_maybe_gzip(path).expect("model bytes");
        let model = ModelProto::decode(bytes.as_slice()).expect("protobuf");
        let graph = model.graph.expect("graph");
        let mut consts: HashMap<String, T> = graph
            .initializer
            .iter()
            .map(|t| (t.name.clone(), decode(t)))
            .collect();
        let input = graph
            .input
            .iter()
            .map(|v| v.name.clone())
            .find(|n| !consts.contains_key(n))
            .expect("one activation input");
        let output = graph.output[0].name.clone();
        let mut live = Vec::new();
        for node in graph.node {
            if node.op_type == "Constant" {
                let t = attr(&node, "value")
                    .and_then(|a| a.t.as_ref())
                    .expect("Constant carries a tensor value");
                consts.insert(node.output[0].clone(), decode(t));
                continue;
            }
            let inputs: Vec<&str> = node
                .input
                .iter()
                .filter(|i| !i.is_empty())
                .map(String::as_str)
                .collect();
            if inputs.iter().all(|i| consts.contains_key(*i)) {
                let refs: Vec<&T> = inputs.iter().map(|i| &consts[*i]).collect();
                let y = run_op(&node, &refs);
                consts.insert(node.output[0].clone(), y);
            } else {
                live.push(node);
            }
        }
        Self {
            consts,
            live,
            input,
            output,
        }
    }

    /// Outputs at `n` points (`xs` is `[n][6]`), `[n][8]`.
    fn eval(&self, xs: &[f64], n: usize) -> Vec<Vec<f64>> {
        let mut env: HashMap<&str, T> = HashMap::new();
        env.insert(
            self.input.as_str(),
            T {
                shape: vec![n, xs.len() / n],
                data: xs.to_vec(),
            },
        );
        for node in &self.live {
            let y = {
                let refs: Vec<&T> = node
                    .input
                    .iter()
                    .filter(|i| !i.is_empty())
                    .map(|i| {
                        env.get(i.as_str())
                            .or_else(|| self.consts.get(i))
                            .unwrap_or_else(|| panic!("`{i}` not produced"))
                    })
                    .collect();
                run_op(node, &refs)
            };
            env.insert(node.output[0].as_str(), y);
        }
        let out = &env[self.output.as_str()];
        assert_eq!(out.data.len(), n * 8, "output shape {:?}", out.shape);
        out.data.chunks(8).map(<[f64]>::to_vec).collect()
    }
}

/// Deterministic xorshift64* in `[0, 1)`.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn root_f64() -> (Vec<f64>, Vec<f64>) {
    (
        BOX_LO.iter().map(|&v| f64::from(v)).collect(),
        BOX_HI.iter().map(|&v| f64::from(v)).collect(),
    )
}

// ---------------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------------

#[test]
fn lsnc_the_twin_admits_the_real_net_and_the_real_graph_stays_dark() {
    let Some(dir) = corpus("lsnc_the_twin_admits_the_real_net_and_the_real_graph_stays_dark")
    else {
        return;
    };
    let path = dir.join("onnx/relu_quadrotor2d_state.onnx");
    let dark = load_onnx_with_config(&path, &load_config())
        .expect("loads")
        .to_graph_network_with_options(graph_options())
        .expect("graph");
    let opaque = dark
        .node_names()
        .iter()
        .filter(|n| {
            dark.node(n)
                .is_some_and(|node| matches!(node.layer(), ny_propagate::Layer::OpaqueSkip(_)))
        })
        .count();
    assert!(
        opaque > 0,
        "the premise: the exact load leaves OpaqueSkip nodes"
    );
    let declined = admit(&dark, &root_box());
    assert!(
        declined.is_err(),
        "without the twin the lane must decline the real graph: {declined:?}"
    );

    let (graph, adm) = admitted(&dir);
    assert_eq!(graph.lowdim_twin_state(), "bound");
    assert!(adm.from_twin());
    assert_eq!(
        adm.interval_consts(),
        4,
        "two IntervalLinear (the Q consumers) and two IntervalOffset (the equilibrium cones)"
    );
    assert_eq!(adm.bilinear_ops(), 2, "V and dV each take one product");
    assert_eq!((adm.input_dim(), adm.output_width()), (6, 8));
    eprintln!(
        "[lsnc-lane-test] lowered_ops={} interval_consts={} bilinear={} opaque_in_real_graph={opaque}",
        adm.lowered_ops(),
        adm.interval_consts(),
        adm.bilinear_ops()
    );
    assert!(
        (40..=60).contains(&adm.lowered_ops()),
        "the armed carrier graph has 48 nodes; got {}",
        adm.lowered_ops()
    );

    // The GLOBAL carrier stayed dark: an ordinary load on this same thread,
    // right after the lane-private one, creates no interval constant.
    let carrier_env = ny_levers::RawLeverInputs::capture(ny_levers::all())
        .get(&ny_levers::decls::dark_probes::CONST_MATMUL_INTERVAL)
        .is_some();
    if !carrier_env {
        let again = load_onnx_with_config(&path, &load_config()).expect("loads");
        assert_eq!(
            again.weights.interval_len(),
            0,
            "the lane-private scope leaked into an ordinary load"
        );
    }

    // The row-2 spec is the 13-clause x 3-row grouped shape the lane serves.
    let spec = ny_onnx::vnnlib::load_vnnlib(dir.join("vnnlib/quadrotor2d_state_2.vnnlib"))
        .expect("row 2 parses");
    assert_eq!(spec.output_constraint_clauses.len(), 13);
    assert!(spec.output_constraint_clauses.iter().all(|c| c.len() == 3));
}

/// THE soundness test: sampled containment of the lane's published bounds
/// against an independent exact-f64 forward of the real ONNX.
#[test]
fn lsnc_lane_bounds_contain_an_exact_f64_forward_of_the_real_onnx() {
    let Some(dir) = corpus("lsnc_lane_bounds_contain_an_exact_f64_forward_of_the_real_onnx") else {
        return;
    };
    let (_graph, adm) = admitted(&dir);
    let exact = OnnxF64::load(&dir.join("onnx/relu_quadrotor2d_state.onnx"));

    // The interpreter is a faithful forward: it reproduces the ORT output on
    // file at the dV counterexample (ORT is f32, so relative 1e-4).
    let w: Vec<f64> = DV_WITNESS.iter().map(|&v| f64::from(v)).collect();
    let yw = &exact.eval(&w, 1)[0];
    assert!(
        (yw[0] - 2.897_847_4e-4).abs() < 2e-6 && (yw[1] - 0.831_713_3).abs() < 1e-4,
        "the independent forward disagrees with ORT at the witness: {yw:?}"
    );

    let (lo, hi) = root_f64();
    let mut rng = Rng(0x6c73_6e63_6c61_6e65);
    const BOXES: usize = 256;
    const POINTS: usize = 800;
    let widths = [1.0, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4, 0.0];
    let (mut blo, mut bhi) = (Vec::with_capacity(BOXES * 6), Vec::with_capacity(BOXES * 6));
    for b in 0..BOXES {
        let centre: Vec<f64> = if b % 16 == 3 {
            w.clone()
        } else {
            (0..6)
                .map(|k| lo[k] + (hi[k] - lo[k]) * rng.unit())
                .collect()
        };
        let frac = if b == 0 {
            1.0
        } else {
            widths[b % widths.len()]
        };
        for k in 0..6 {
            let r = 0.5 * frac * (hi[k] - lo[k]);
            blo.push((centre[k] - r).max(lo[k]));
            bhi.push((centre[k] + r).min(hi[k]));
        }
    }
    let mut objectives: Vec<Vec<f64>> = Vec::new();
    for j in 0..8 {
        for sign in [1.0, -1.0] {
            let mut e = vec![0.0; 8];
            e[j] = sign;
            objectives.push(e);
        }
    }
    let half = BOXES / 2 * 6;
    let mut lbs = bound_boxes(
        &adm,
        &blo[..half],
        &bhi[..half],
        &objectives,
        &LowDimConfig::default(),
    )
    .expect("bounds (plain)");
    lbs.extend(
        bound_boxes(
            &adm,
            &blo[half..],
            &bhi[half..],
            &objectives,
            &LowDimConfig {
                refine_max_width: 8,
                ..LowDimConfig::default()
            },
        )
        .expect("bounds (refined)"),
    );
    assert_eq!(lbs.len(), BOXES);

    let mut checked = 0usize;
    for b in 0..BOXES {
        let (l, h) = (&blo[b * 6..(b + 1) * 6], &bhi[b * 6..(b + 1) * 6]);
        let lb = &lbs[b];
        // A vacuous bound would pass containment trivially: every published
        // bound must be finite.
        assert!(
            lb.iter().all(|v| v.is_finite()),
            "box {b}: a non-finite bound {lb:?}"
        );
        let mut xs: Vec<f64> = Vec::with_capacity((POINTS + 64) * 6);
        for _ in 0..POINTS {
            xs.extend((0..6).map(|k| l[k] + (h[k] - l[k]) * rng.unit()));
        }
        for corner in 0..64usize {
            xs.extend((0..6).map(|k| if corner >> k & 1 == 1 { h[k] } else { l[k] }));
        }
        let n = xs.len() / 6;
        let ys = exact.eval(&xs, n);
        for (p, y) in ys.iter().enumerate() {
            for j in 0..8 {
                let (lower, upper) = (lb[2 * j], -lb[2 * j + 1]);
                let tol = 1e-9 * (1.0 + y[j].abs());
                assert!(
                    lower <= y[j] + tol && y[j] <= upper + tol,
                    "box {b} point {p} {:?}: exact Y_{j} = {:e} escapes the lane's published \
                     [{lower:e}, {upper:e}] — a FALSE ENCLOSURE",
                    &xs[p * 6..(p + 1) * 6],
                    y[j]
                );
            }
            checked += 1;
        }
        // Non-vacuity on point boxes: the bound hugs the value (the interval
        // constants' radius is ~1e-7 relative, so 1e-2 is generous).
        if l == h {
            for j in 0..8 {
                let y = ys[0][j];
                assert!(
                    y - lb[2 * j] <= 1e-2 * (1.0 + y.abs()),
                    "point box {b}: Y_{j} lower bound {} is loose against {y}",
                    lb[2 * j]
                );
            }
        }
        // The lowered program's own point value agrees with the exact forward
        // (it evaluates the interval constants at their midpoints).
        for (p, y) in ys.iter().take(8).enumerate() {
            let mine = lowered_point_values(&adm, &xs[p * 6..(p + 1) * 6]).expect("point");
            for j in 0..8 {
                assert!(
                    (mine[j] - y[j]).abs() <= 1e-4 * (1.0 + y[j].abs()),
                    "box {b} point {p}: lowered Y_{j} = {} vs exact {}",
                    mine[j],
                    y[j]
                );
            }
        }
    }
    assert!(
        checked >= 200_000,
        "the containment sample must reach 200k points, got {checked}"
    );
    eprintln!("[lsnc-lane-test] containment checked={checked} boxes={BOXES}");
}

/// The row-2 ROOT upper bound of `Y_0 = dV` covers the ORT sample maximum of
/// the ORIGINAL model over the row's input box.
#[test]
fn lsnc_row2_root_dv_upper_bound_covers_the_ort_sample_max() {
    let Some(dir) = corpus("lsnc_row2_root_dv_upper_bound_covers_the_ort_sample_max") else {
        return;
    };
    let path = dir.join("onnx/relu_quadrotor2d_state.onnx");
    let spec = ny_onnx::vnnlib::load_vnnlib(dir.join("vnnlib/quadrotor2d_state_2.vnnlib"))
        .expect("row 2 parses");
    let (lo, hi) = root_f64();
    assert_eq!(spec.input_bounds.len(), 6);
    for (k, &(l, h)) in spec.input_bounds.iter().enumerate() {
        assert!(
            (l - lo[k]).abs() < 1e-6 && (h - hi[k]).abs() < 1e-6,
            "row 2's box differs from the shared lsnc box at X_{k}: [{l}, {h}]"
        );
    }
    let (_graph, adm) = admitted(&dir);
    let mut e = vec![0.0; 8];
    e[0] = -1.0;
    let lb = bound_boxes(&adm, &lo, &hi, &[e], &LowDimConfig::default()).expect("root bound")[0][0];
    let dv_upper = -lb;

    let mut ort = match ny_onnx::diff::OrtForward::from_path(&path, 6) {
        Ok(ort) => ort,
        Err(error) => {
            skip_loudly(
                "lsnc_row2_root_dv_upper_bound_covers_the_ort_sample_max",
                &format!("ONNX Runtime unavailable: {error}"),
            );
            return;
        }
    };
    let mut rng = Rng(0x726f_7732_6476_7570);
    let mut best = f64::NEG_INFINITY;
    let mut run = |x: &[f32]| -> f64 { f64::from(ort.run(x).expect("ORT forward")[0]) };
    for corner in 0..64usize {
        let x: Vec<f32> = (0..6)
            .map(|k| {
                if corner >> k & 1 == 1 {
                    BOX_HI[k]
                } else {
                    BOX_LO[k]
                }
            })
            .collect();
        best = best.max(run(x.as_slice()));
    }
    best = best.max(run(&DV_WITNESS[..]));
    for _ in 0..20_000 {
        let x: Vec<f32> = (0..6)
            .map(|k| (lo[k] + (hi[k] - lo[k]) * rng.unit()) as f32)
            .collect();
        best = best.max(run(x.as_slice()));
    }
    eprintln!(
        "[lsnc-lane-test] row2 root dV upper bound {dv_upper:.6} vs ORT sample max {best:.6} \
         (numpy plain-lane reference 28.0387; graph-lane joint reference 0.2753)"
    );
    assert!(
        dv_upper >= best,
        "the lane's root dV upper bound {dv_upper} is BELOW a value ORT reaches ({best}): \
         a FALSE ENCLOSURE"
    );
    assert!(
        dv_upper.is_finite() && dv_upper < 100.0,
        "the root dV bound {dv_upper} is vacuous (the plain lane's reference is ~28.04)"
    );
}

// ---------------------------------------------------------------------------
// #lsnc-lane stage 2: the joint quadratic bound on the real graph
// ---------------------------------------------------------------------------

/// The armed configuration the measurement yaml uses: joint on, every layer
/// (all <= 8 wide) refined.
fn joint_cfg() -> LowDimConfig {
    LowDimConfig {
        joint_quadratic: true,
        refine_max_width: 8,
        ..LowDimConfig::default()
    }
}

/// The two sites by name: `V` is the `/lyapunov/` reduction, `dV` the other.
fn site_indices(adm: &LaneAdmission) -> (usize, usize) {
    let names = adm.joint_site_names();
    assert_eq!(
        names.len(),
        2,
        "V and dV are the two joint sites: {names:?}"
    );
    let v = names
        .iter()
        .position(|n| n.contains("lyapunov"))
        .unwrap_or_else(|| panic!("no `/lyapunov/` site among {names:?}"));
    (v, 1 - v)
}

/// Both rows' official input boxes are the shared lsnc box (the bands differ,
/// the box does not), so sub-boxes of it are sub-boxes of the real rows.
fn assert_row_box(dir: &std::path::Path, row: usize) {
    let spec =
        ny_onnx::vnnlib::load_vnnlib(dir.join(format!("vnnlib/quadrotor2d_state_{row}.vnnlib")))
            .unwrap_or_else(|e| panic!("row {row} parses: {e}"));
    let (lo, hi) = root_f64();
    assert_eq!(spec.input_bounds.len(), 6);
    for (k, &(l, h)) in spec.input_bounds.iter().enumerate() {
        assert!(
            (l - lo[k]).abs() < 1e-6 && (h - hi[k]).abs() < 1e-6,
            "row {row}'s box differs from the shared lsnc box at X_{k}: [{l}, {h}]"
        );
    }
}

/// THE stage-2 soundness test: the joint sites' intervals, the joint planes
/// of the same pass, and the published bounds of `Y_0 = dV` and `Y_1 = V`
/// contain an INDEPENDENT exact-f64 forward of the real ONNX on >= 200,000
/// points over sub-boxes of every width (root, down to points, and boxes
/// around the ORT-confirmed dV counterexample).
#[test]
fn lsnc_joint_sites_planes_and_bounds_contain_the_exact_forward_on_row_subboxes() {
    let Some(dir) =
        corpus("lsnc_joint_sites_planes_and_bounds_contain_the_exact_forward_on_row_subboxes")
    else {
        return;
    };
    assert_row_box(&dir, 2);
    assert_row_box(&dir, 79);
    let (_graph, adm) = admitted(&dir);
    let (iv, idv) = site_indices(&adm);
    let exact = OnnxF64::load(&dir.join("onnx/relu_quadrotor2d_state.onnx"));
    let w: Vec<f64> = DV_WITNESS.iter().map(|&v| f64::from(v)).collect();
    let (lo, hi) = root_f64();
    let mut rng = Rng(0x6a6f_696e_7432_7362);
    const BOXES: usize = 128;
    const POINTS: usize = 1536;
    let widths = [1.0, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4, 0.0];
    let (mut blo, mut bhi) = (Vec::with_capacity(BOXES * 6), Vec::with_capacity(BOXES * 6));
    for b in 0..BOXES {
        let centre: Vec<f64> = if b % 16 == 5 {
            w.clone()
        } else {
            (0..6)
                .map(|k| lo[k] + (hi[k] - lo[k]) * rng.unit())
                .collect()
        };
        let frac = if b == 0 {
            1.0
        } else {
            widths[b % widths.len()]
        };
        for k in 0..6 {
            let r = 0.5 * frac * (hi[k] - lo[k]);
            blo.push((centre[k] - r).max(lo[k]));
            bhi.push((centre[k] + r).min(hi[k]));
        }
    }
    let cfg = joint_cfg();
    let sites = joint_site_boxes(&adm, &blo, &bhi, &cfg).expect("site boxes");
    let mut objectives = Vec::new();
    for j in [0usize, 1] {
        for sign in [1.0, -1.0] {
            let mut e = vec![0.0; 8];
            e[j] = sign;
            objectives.push(e);
        }
    }
    let lbs = bound_boxes(&adm, &blo, &bhi, &objectives, &cfg).expect("bounds");
    assert_eq!((sites.len(), lbs.len()), (BOXES, BOXES));
    let (mut checked, mut planes_seen) = (0usize, 0usize);
    for b in 0..BOXES {
        let (l, h) = (&blo[b * 6..(b + 1) * 6], &bhi[b * 6..(b + 1) * 6]);
        let lb = &lbs[b];
        assert!(
            lb.iter().all(|v| v.is_finite()),
            "box {b}: non-finite bound {lb:?}"
        );
        let (sv, sdv) = (&sites[b][iv], &sites[b][idv]);
        planes_seen += usize::from(sv.lower.is_some()) + usize::from(sdv.lower.is_some());
        let mut xs: Vec<f64> = Vec::with_capacity((POINTS + 64) * 6);
        for _ in 0..POINTS {
            xs.extend((0..6).map(|k| l[k] + (h[k] - l[k]) * rng.unit()));
        }
        for corner in 0..64usize {
            xs.extend((0..6).map(|k| if corner >> k & 1 == 1 { h[k] } else { l[k] }));
        }
        let n = xs.len() / 6;
        let ys = exact.eval(&xs, n);
        for (p, y) in ys.iter().enumerate() {
            let x = &xs[p * 6..(p + 1) * 6];
            for (site, value, what) in [(sv, y[1], "V = Y_1"), (sdv, y[0], "dV = Y_0")] {
                let tol = 1e-9 * (1.0 + value.abs());
                assert!(
                    site.lo <= value + tol && value <= site.hi + tol,
                    "box {b} point {p} {x:?}: exact {what} = {value:e} escapes the joint site \
                     interval [{:e}, {:e}] — a FALSE ENCLOSURE",
                    site.lo,
                    site.hi
                );
                if let (Some((gl, cl)), Some((gh, ch))) = (&site.lower, &site.upper) {
                    let at =
                        |g: &[f64], c: f64| g.iter().zip(x).map(|(a, t)| a * t).sum::<f64>() + c;
                    let (pl, ph) = (at(gl.as_slice(), *cl), at(gh.as_slice(), *ch));
                    assert!(
                        pl <= value + tol && value <= ph + tol,
                        "box {b} point {p} {x:?}: exact {what} = {value:e} escapes the joint \
                         PLANES [{pl:e}, {ph:e}] — a FALSE PLANE"
                    );
                }
            }
            for (j, value) in [(0usize, y[0]), (1, y[1])] {
                let (lower, upper) = (lb[2 * j], -lb[2 * j + 1]);
                let tol = 1e-9 * (1.0 + value.abs());
                assert!(
                    lower <= value + tol && value <= upper + tol,
                    "box {b} point {p} {x:?}: exact Y_{j} = {value:e} escapes the published \
                     [{lower:e}, {upper:e}] with the joint armed — a FALSE ENCLOSURE"
                );
            }
            checked += 1;
        }
    }
    assert!(
        checked >= 200_000,
        "the sample must reach 200k points, got {checked}"
    );
    assert!(
        planes_seen >= BOXES,
        "the armed joint produced planes on only {planes_seen} of {} site-boxes: VOID",
        2 * BOXES
    );
    eprintln!(
        "[lsnc-lane-test] joint containment checked={checked} boxes={BOXES} \
         site_boxes_with_planes={planes_seen}/{}",
        2 * BOXES
    );
}

/// The numpy prototype's published root upper bound of `Y_0 = dV` on row 2
/// (2026 `1.0` corpus) with the `allmax` stack — joint vertex bound, CROWN
/// operand certificates, refinement of every relaxed-op input, the interval
/// meet — from `scratchpad/lsncproto/gate_engine.py` `ClauseProblem.evaluate`
/// on 2026-09-25. Its refinement-only value (no joint) is 14.0653, its plain
/// value 28.0387; its `V` lower bound is -2.9e-10 with the joint and -30.088
/// without.
const PROTO_ROW2_ROOT_DV_UB_ALLMAX: f64 = 13.124_669_11;

/// REGRESSION PIN: the lane's published row-2 root `dV` upper bound with the
/// joint armed matches the prototype's `allmax` value (the Rust lane charges
/// rigorous rounding and the enclosure radius of the interval constants, so
/// it may sit a hair above, never meaningfully below), still covers every
/// exact sample, and the joint is ENGAGED: planes at both sites, and `V`'s
/// lower bound and `dV`'s upper bound tighter than the unarmed lane's.
#[test]
fn lsnc_row2_root_joint_dv_upper_bound_pins_the_numpy_allmax_value() {
    let Some(dir) = corpus("lsnc_row2_root_joint_dv_upper_bound_pins_the_numpy_allmax_value")
    else {
        return;
    };
    assert_row_box(&dir, 2);
    let (_graph, adm) = admitted(&dir);
    let report = joint_root_report(&adm, &joint_cfg());
    let plain = joint_root_report(
        &adm,
        &LowDimConfig {
            refine_max_width: 8,
            ..LowDimConfig::default()
        },
    );
    eprintln!(
        "[lsnc-lane-test] row2 root joint dv_ub={:.9} v_lb={:.3e} (prototype allmax {PROTO_ROW2_ROOT_DV_UB_ALLMAX}) \
         | unarmed dv_ub={:.9} v_lb={:.6} | sites {:?}",
        report.dv_ub,
        report.v_lb,
        plain.dv_ub,
        plain.v_lb,
        report
            .sites
            .iter()
            .map(|s| (s.name.as_str(), s.lo, s.hi, s.lower.is_some()))
            .collect::<Vec<_>>()
    );
    assert!(report.armed && !plain.armed);
    assert_eq!(report.sites.len(), 2);
    assert!(
        report
            .sites
            .iter()
            .all(|s| s.lower.is_some() && s.upper.is_some()),
        "the armed joint must produce planes at both root sites: VOID otherwise"
    );
    let pin = PROTO_ROW2_ROOT_DV_UB_ALLMAX;
    assert!(
        report.dv_ub >= pin * (1.0 - 1e-3) && report.dv_ub <= pin * (1.0 + 1e-2),
        "row-2 root dV upper bound {} drifted from the prototype's allmax {pin}",
        report.dv_ub
    );
    assert!(
        report.dv_ub < plain.dv_ub - 0.1,
        "the joint did not tighten the root dV bound ({} vs unarmed {})",
        report.dv_ub,
        plain.dv_ub
    );
    assert!(
        report.v_lb >= -1e-2 && plain.v_lb < -1.0,
        "the joint must lift V's root lower bound to ~0 (got {}, unarmed {})",
        report.v_lb,
        plain.v_lb
    );
    // Containment at the root against the independent exact forward.
    let exact = OnnxF64::load(&dir.join("onnx/relu_quadrotor2d_state.onnx"));
    let (lo, hi) = root_f64();
    let mut rng = Rng(0x726f_6f74_6a6f_696e);
    let mut xs: Vec<f64> = DV_WITNESS.iter().map(|&v| f64::from(v)).collect();
    for corner in 0..64usize {
        xs.extend((0..6).map(|k| if corner >> k & 1 == 1 { hi[k] } else { lo[k] }));
    }
    for _ in 0..100_000 {
        xs.extend((0..6).map(|k| lo[k] + (hi[k] - lo[k]) * rng.unit()));
    }
    let n = xs.len() / 6;
    let ys = exact.eval(&xs, n);
    let dv_max = ys.iter().map(|y| y[0]).fold(f64::NEG_INFINITY, f64::max);
    let v_min = ys.iter().map(|y| y[1]).fold(f64::INFINITY, f64::min);
    eprintln!("[lsnc-lane-test] row2 root exact sample: max dV {dv_max:.6} min V {v_min:.6}");
    assert!(
        report.dv_ub >= dv_max && report.v_lb <= v_min,
        "a FALSE ENCLOSURE at the root: dv_ub {} vs sampled max {dv_max}, v_lb {} vs sampled \
         min {v_min}",
        report.dv_ub,
        report.v_lb
    );
}

// ---------------------------------------------------------------------------
// #lsnc-lane stage 3: the search stack (clip, smart split, parallel waves)
// ---------------------------------------------------------------------------

/// The measurement arming of stage 3: joint, refinement, relaxed clip,
/// `sbwr` split, decisions inside the rayon tasks.
fn stack_cfg() -> LowDimConfig {
    LowDimConfig {
        joint_quadratic: true,
        refine_max_width: 8,
        clip: true,
        split: LowDimSplit::SbWr,
        parallel: LowDimParallel::Full,
        ..LowDimConfig::default()
    }
}

/// `sign * e_j` over the 8 outputs.
fn unit_objective(j: usize, sign: f64) -> Vec<f64> {
    let mut e = vec![0.0; 8];
    e[j] = sign;
    e
}

/// The `q`-quantile of `v` (nearest rank).
fn quantile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() - 1) as f64 * q).round() as usize]
}

/// CLIP SOUNDNESS ON THE REAL NET. On 40 sub-boxes of the shared lsnc box (every
/// width down to points; some around the ORT-confirmed dV counterexample), for a
/// clause-0-shaped conjunction (`V` in a band, `dV` above a threshold) with
/// SYNTHETIC thresholds at the exact forward's [0.2, 0.8] quantiles of `V` and
/// its median of `dV` over the box (so a real share of the samples satisfies
/// the whole conjunction):
/// (a) every clip plane the lane computes for the box — each backward pass's
///     own `a . x + c <= objective . y(x)` — holds at every sample of the INDEPENDENT
///     exact-f64 forward;
/// (b) every sample satisfying every conjunct lies inside the clipped box, and
///     an emptied box holds none.
/// Non-vacuity: >= 5,000 such points, planes on every box, and at least one
/// box actually shrunk.
#[test]
fn lsnc_clip_planes_contain_the_exact_forward_and_never_cut_a_sampled_counterexample() {
    let Some(dir) =
        corpus("lsnc_clip_planes_contain_the_exact_forward_and_never_cut_a_sampled_counterexample")
    else {
        return;
    };
    assert_row_box(&dir, 2);
    let (_graph, adm) = admitted(&dir);
    let exact = OnnxF64::load(&dir.join("onnx/relu_quadrotor2d_state.onnx"));
    let objectives = vec![
        unit_objective(1, -1.0),
        unit_objective(1, 1.0),
        unit_objective(0, -1.0),
    ];
    let quantiles = [0.8, 0.8, 0.5];
    let cfg = stack_cfg();
    let w: Vec<f64> = DV_WITNESS.iter().map(|&v| f64::from(v)).collect();
    let (lo, hi) = root_f64();
    let mut rng = Rng(0x636c_6970_7265_616c);
    let widths = [1.0, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4, 0.0];
    const BOXES: usize = 40;
    const POINTS: usize = 1024;
    let (mut tested, mut shrunk, mut planes) = (0usize, 0usize, 0usize);
    for b in 0..BOXES {
        let centre: Vec<f64> = if b % 8 == 3 {
            w.clone()
        } else {
            (0..6)
                .map(|k| lo[k] + (hi[k] - lo[k]) * rng.unit())
                .collect()
        };
        let frac = if b == 0 {
            1.0
        } else {
            widths[b % widths.len()]
        };
        let (l, h): (Vec<f64>, Vec<f64>) = (0..6)
            .map(|k| {
                let r = 0.5 * frac * (hi[k] - lo[k]);
                ((centre[k] - r).max(lo[k]), (centre[k] + r).min(hi[k]))
            })
            .unzip();
        let mut xs: Vec<f64> = Vec::with_capacity((POINTS + 64) * 6);
        for _ in 0..POINTS {
            xs.extend((0..6).map(|k| l[k] + (h[k] - l[k]) * rng.unit()));
        }
        for corner in 0..64usize {
            xs.extend((0..6).map(|k| if corner >> k & 1 == 1 { h[k] } else { l[k] }));
        }
        let n = xs.len() / 6;
        let ys = exact.eval(&xs, n);
        let values: Vec<Vec<f64>> = ys
            .iter()
            .map(|y| {
                objectives
                    .iter()
                    .map(|o| o.iter().zip(y).map(|(a, v)| a * v).sum())
                    .collect()
            })
            .collect();
        let thresholds: Vec<f64> = quantiles
            .iter()
            .enumerate()
            .map(|(i, &q)| {
                let col: Vec<f64> = values.iter().map(|v| v[i]).collect();
                quantile(&col, q)
            })
            .collect();
        let report = clip_report(&adm, &l, &h, &objectives, &thresholds, &cfg)
            .expect("clip report")
            .pop()
            .expect("one box");
        assert!(
            !report.planes.is_empty(),
            "box {b}: the lane produced no clip plane: VOID"
        );
        planes += report.planes.len();
        for (p, v) in values.iter().enumerate() {
            let x = &xs[p * 6..(p + 1) * 6];
            for (i, a, c) in &report.planes {
                let pv = a.iter().zip(x).map(|(u, t)| u * t).sum::<f64>() + c;
                let tol = 1e-9 * (1.0 + v[*i].abs());
                assert!(
                    pv <= v[*i] + tol,
                    "box {b} point {p} {x:?}: the clip plane of objective {i} gives {pv:e} above \
                     the exact value {:e} — a FALSE PLANE",
                    v[*i]
                );
            }
            if !v.iter().zip(&thresholds).all(|(val, t)| val <= t) {
                continue;
            }
            tested += 1;
            match &report.clipped {
                None => {
                    panic!("box {b}: the clip EMPTIED a box holding the exact counterexample {x:?}")
                }
                Some((cl, ch)) => assert!(
                    (0..6).all(|k| cl[k] <= x[k] && x[k] <= ch[k]),
                    "box {b}: the clip cut the exact counterexample {x:?} out of [{cl:?}, {ch:?}]"
                ),
            }
        }
        if let Some((cl, ch)) = &report.clipped {
            shrunk += usize::from((0..6).any(|k| cl[k] > l[k] || ch[k] < h[k]));
        }
    }
    eprintln!(
        "[lsnc-lane-test] clip on the real net: counterexamples_tested={tested} \
         boxes_shrunk={shrunk}/{BOXES} planes={planes}"
    );
    assert!(
        tested >= 5_000,
        "VOID: only {tested} counterexample points tested"
    );
    assert!(shrunk > 0, "VOID: the clip never shrank a box");
}

/// The smallest f32 at or above `t`: a lane proof of `objective . y > t32`
/// with `t32 >= t` also refutes the conjunct `objective . y <= t`.
fn f32_up(t: f64) -> f32 {
    let f = t as f32;
    if f64::from(f) < t {
        f.next_up()
    } else {
        f
    }
}

/// END TO END ON THE REAL NET, the armed stage-3 stack: a clause the exact
/// forward ATTAINS — clause 0's shape (`dV >= 1e-6`) with a `V` band placed
/// around the ORT-confirmed dV counterexample, thresholds rounded so the point
/// satisfies every conjunct — is NEVER proved (any other outcome is fine; a
/// candidate stop must hand back an in-box point); and an impossible clause
/// (`V <= -1`: `V = x'Px`, `P` positive semidefinite) IS proved.
#[test]
fn lsnc_the_full_stack_never_proves_a_clause_the_exact_forward_attains() {
    let Some(dir) = corpus("lsnc_the_full_stack_never_proves_a_clause_the_exact_forward_attains")
    else {
        return;
    };
    let (_graph, adm) = admitted(&dir);
    let exact = OnnxF64::load(&dir.join("onnx/relu_quadrotor2d_state.onnx"));
    let w: Vec<f64> = DV_WITNESS.iter().map(|&v| f64::from(v)).collect();
    let y = exact.eval(&w, 1).pop().expect("one point");
    let (dv, v) = (y[0], y[1]);
    assert!(
        dv > 1e-6,
        "the witness must satisfy dV >= 1e-6 (exact dV {dv:e})"
    );
    let objectives: Vec<Vec<f32>> = vec![
        vec![0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let thresholds = [f32_up(-(v - 0.01)), f32_up(v + 0.01), f32_up(-1e-6)];
    for (o, &t) in objectives.iter().zip(&thresholds) {
        let at: f64 = o.iter().zip(&y).map(|(a, b)| f64::from(*a) * b).sum();
        assert!(
            at <= f64::from(t),
            "the witness must satisfy every conjunct ({at} > {t}): the control is VOID"
        );
    }
    let cfg = stack_cfg();
    let mut witness = None;
    let out = prove_conjunction_admitted(
        &adm,
        &objectives,
        &thresholds,
        std::time::Instant::now() + std::time::Duration::from_secs(20),
        &cfg,
        &mut witness,
    );
    eprintln!("[lsnc-lane-test] attained clause (V={v:.6}, dV={dv:.3e}): {out:?}");
    assert!(
        !matches!(out, LowDimOutcome::Verified(_)),
        "the armed stack PROVED a clause the exact forward attains at the ORT-confirmed dV \
         counterexample — a FALSE PROOF: {out:?}"
    );
    if let Some(p) = &witness {
        let (lo, hi) = root_f64();
        assert!(
            p.len() == 6 && (0..6).all(|k| lo[k] <= p[k] && p[k] <= hi[k]),
            "a candidate outside the box: {p:?}"
        );
    }
    let mut none = None;
    let impossible = prove_conjunction_admitted(
        &adm,
        &[vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        &[-1.0],
        std::time::Instant::now() + std::time::Duration::from_secs(20),
        &cfg,
        &mut none,
    );
    eprintln!("[lsnc-lane-test] impossible clause V <= -1: {impossible:?}");
    assert!(
        matches!(impossible, LowDimOutcome::Verified(_)),
        "the armed stack failed to prove V <= -1 impossible: {impossible:?}"
    );
}
