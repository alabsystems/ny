// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared synthetic fixtures for the checker's tests: a minimal ONNX
//! protobuf WRITER (the checker's reader is tested against bytes this file
//! produces, never against a producer's), a VNN-LIB generator, and a
//! `ny-proof-bundle/v2` builder that computes the `end` digest.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use ny_cert::json::{self, Json, Map};
use ny_proofcheck::bundle::sha256_hex;
use ny_proofcheck::{check_bundle, CheckArgs, CheckOutcome};

// ---------------------------------------------------------------------------
// Protobuf writer
// ---------------------------------------------------------------------------

fn varint(out: &mut Vec<u8>, mut v: u64) {
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

fn key(out: &mut Vec<u8>, field: u64, wire: u64) {
    varint(out, (field << 3) | wire);
}

fn bytes_field(out: &mut Vec<u8>, field: u64, b: &[u8]) {
    key(out, field, 2);
    varint(out, b.len() as u64);
    out.extend_from_slice(b);
}

fn str_field(out: &mut Vec<u8>, field: u64, s: &str) {
    bytes_field(out, field, s.as_bytes());
}

fn int_field(out: &mut Vec<u8>, field: u64, v: i64) {
    key(out, field, 0);
    varint(out, v as u64);
}

fn packed_ints(out: &mut Vec<u8>, field: u64, vs: &[i64]) {
    let mut body = Vec::new();
    for &v in vs {
        varint(&mut body, v as u64);
    }
    bytes_field(out, field, &body);
}

/// An initializer / constant tensor.
#[derive(Debug, Clone)]
pub enum TData {
    /// FLOAT data.
    F32(Vec<f32>),
    /// INT64 data.
    I64(Vec<i64>),
}

/// A tensor to encode.
#[derive(Debug, Clone)]
pub struct T {
    /// Name.
    pub name: String,
    /// Dims.
    pub dims: Vec<i64>,
    /// Data.
    pub data: TData,
}

fn tensor_bytes(t: &T) -> Vec<u8> {
    let mut o = Vec::new();
    packed_ints(&mut o, 1, &t.dims);
    match &t.data {
        TData::F32(v) => {
            int_field(&mut o, 2, 1);
            str_field(&mut o, 8, &t.name);
            let mut raw = Vec::with_capacity(v.len() * 4);
            for x in v {
                raw.extend_from_slice(&x.to_le_bytes());
            }
            bytes_field(&mut o, 9, &raw);
        }
        TData::I64(v) => {
            int_field(&mut o, 2, 7);
            str_field(&mut o, 8, &t.name);
            let mut raw = Vec::with_capacity(v.len() * 8);
            for x in v {
                raw.extend_from_slice(&x.to_le_bytes());
            }
            bytes_field(&mut o, 9, &raw);
        }
    }
    o
}

/// A node attribute.
#[derive(Debug, Clone)]
pub enum A {
    /// Float.
    F(&'static str, f32),
    /// Int.
    I(&'static str, i64),
    /// Ints.
    Is(&'static str, Vec<i64>),
    /// String.
    S(&'static str, &'static str),
    /// Tensor.
    Tn(&'static str, T),
}

fn attr_bytes(a: &A) -> Vec<u8> {
    let mut o = Vec::new();
    match a {
        A::F(n, v) => {
            str_field(&mut o, 1, n);
            key(&mut o, 2, 5);
            o.extend_from_slice(&v.to_bits().to_le_bytes());
            int_field(&mut o, 20, 1);
        }
        A::I(n, v) => {
            str_field(&mut o, 1, n);
            int_field(&mut o, 3, *v);
            int_field(&mut o, 20, 2);
        }
        A::Is(n, v) => {
            str_field(&mut o, 1, n);
            packed_ints(&mut o, 8, v);
            int_field(&mut o, 20, 7);
        }
        A::S(n, v) => {
            str_field(&mut o, 1, n);
            bytes_field(&mut o, 4, v.as_bytes());
            int_field(&mut o, 20, 3);
        }
        A::Tn(n, t) => {
            str_field(&mut o, 1, n);
            bytes_field(&mut o, 5, &tensor_bytes(t));
            int_field(&mut o, 20, 4);
        }
    }
    o
}

/// A graph node.
#[derive(Debug, Clone)]
pub struct N {
    /// Op type.
    pub op: &'static str,
    /// Inputs.
    pub inputs: Vec<String>,
    /// Outputs.
    pub outputs: Vec<String>,
    /// Attributes.
    pub attrs: Vec<A>,
}

/// Shorthand for a node.
pub fn node(op: &'static str, inputs: &[&str], outputs: &[&str], attrs: Vec<A>) -> N {
    N {
        op,
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs,
    }
}

/// A FLOAT tensor.
pub fn tf(name: &str, dims: &[i64], v: &[f64]) -> T {
    T {
        name: name.to_string(),
        dims: dims.to_vec(),
        data: TData::F32(v.iter().map(|&x| x as f32).collect()),
    }
}

/// An INT64 tensor.
pub fn ti(name: &str, dims: &[i64], v: &[i64]) -> T {
    T {
        name: name.to_string(),
        dims: dims.to_vec(),
        data: TData::I64(v.to_vec()),
    }
}

/// A declared dim of a graph input or output.
#[derive(Debug, Clone, Copy)]
pub enum Dm {
    /// `dim_value`.
    V(i64),
    /// `dim_param` (a symbolic dim).
    P(&'static str),
}

fn value_info(name: &str, dims: &[i64]) -> Vec<u8> {
    let dims: Vec<Dm> = dims.iter().map(|&d| Dm::V(d)).collect();
    value_info_dims(name, &dims)
}

fn value_info_dims(name: &str, dims: &[Dm]) -> Vec<u8> {
    let mut shape = Vec::new();
    for d in dims {
        let mut dim = Vec::new();
        match *d {
            Dm::V(v) => int_field(&mut dim, 1, v),
            Dm::P(p) => str_field(&mut dim, 2, p),
        }
        bytes_field(&mut shape, 1, &dim);
    }
    let mut tt = Vec::new();
    int_field(&mut tt, 1, 1);
    bytes_field(&mut tt, 2, &shape);
    let mut ty = Vec::new();
    bytes_field(&mut ty, 1, &tt);
    let mut o = Vec::new();
    str_field(&mut o, 1, name);
    bytes_field(&mut o, 2, &ty);
    o
}

/// Encode a model: `input` of `in_dims`, `nodes`, `inits`, one `output`
/// (default-domain opset 13).
pub fn model(
    input: &str,
    in_dims: &[i64],
    nodes: &[N],
    inits: &[T],
    output: &str,
    out_dims: &[i64],
) -> Vec<u8> {
    let dims: Vec<Dm> = in_dims.iter().map(|&d| Dm::V(d)).collect();
    model_ext(input, &dims, nodes, inits, output, out_dims, 13)
}

/// [`model`] with symbolic input dims and an explicit default-domain opset.
pub fn model_ext(
    input: &str,
    in_dims: &[Dm],
    nodes: &[N],
    inits: &[T],
    output: &str,
    out_dims: &[i64],
    opset: i64,
) -> Vec<u8> {
    let mut g = Vec::new();
    for n in nodes {
        let mut nb = Vec::new();
        for i in &n.inputs {
            str_field(&mut nb, 1, i);
        }
        for o in &n.outputs {
            str_field(&mut nb, 2, o);
        }
        str_field(&mut nb, 4, n.op);
        for a in &n.attrs {
            bytes_field(&mut nb, 5, &attr_bytes(a));
        }
        bytes_field(&mut g, 1, &nb);
    }
    str_field(&mut g, 2, "g");
    for t in inits {
        bytes_field(&mut g, 5, &tensor_bytes(t));
    }
    bytes_field(&mut g, 11, &value_info_dims(input, in_dims));
    bytes_field(&mut g, 12, &value_info(output, out_dims));
    let mut m = Vec::new();
    int_field(&mut m, 1, 8);
    let mut op = Vec::new();
    str_field(&mut op, 1, "");
    int_field(&mut op, 2, opset);
    bytes_field(&mut m, 8, &op);
    bytes_field(&mut m, 7, &g);
    m
}

// ---------------------------------------------------------------------------
// VNN-LIB
// ---------------------------------------------------------------------------

/// A robustness property: `X_i in [lo_i, hi_i]`, unsafe iff some class in
/// `adv` scores at least the true class `t` (one disjunct per class).
pub fn robustness_vnnlib(lo: &[&str], hi: &[&str], n_y: usize, t: usize, adv: &[usize]) -> String {
    let mut s = String::new();
    for i in 0..lo.len() {
        s.push_str(&format!("(declare-const X_{i} Real)\n"));
    }
    for j in 0..n_y {
        s.push_str(&format!("(declare-const Y_{j} Real)\n"));
    }
    for i in 0..lo.len() {
        s.push_str(&format!(
            "(assert (>= X_{i} {}))\n(assert (<= X_{i} {}))\n",
            lo[i], hi[i]
        ));
    }
    s.push_str("(assert (or\n");
    for j in adv {
        s.push_str(&format!("    (and (>= Y_{j} Y_{t}))\n"));
    }
    s.push_str("))\n");
    s
}

// ---------------------------------------------------------------------------
// Bundle builder
// ---------------------------------------------------------------------------

/// A literal to emit.
#[derive(Debug, Clone, Copy)]
pub enum L {
    /// `[tensor id, index, side]`.
    R(u64, u64, bool),
    /// `["x", index, "<decimal>", side]`.
    X(u64, f64, bool),
}

fn lit_json(l: L) -> Json {
    match l {
        L::R(t, i, p) => Json::Arr(vec![Json::U64(t), Json::U64(i), Json::U64(u64::from(p))]),
        L::X(i, t, p) => Json::Arr(vec![
            Json::Str("x".into()),
            Json::U64(i),
            Json::Str(format!("{t:?}")),
            Json::U64(u64::from(p)),
        ]),
    }
}

fn obj(pairs: Vec<(&str, Json)>) -> Json {
    let mut m = Map::new();
    for (k, v) in pairs {
        m.insert(k.to_string(), v);
    }
    Json::Obj(m)
}

/// The header line.
pub fn header(onnx: &Path, onnx_bytes: &[u8], vnn: &Path, vnn_bytes: &[u8]) -> String {
    let fref = |p: &Path, b: &[u8]| {
        obj(vec![
            ("path", Json::Str(p.display().to_string())),
            ("sha256", Json::Str(sha256_hex(b))),
            ("bytes", Json::U64(b.len() as u64)),
        ])
    };
    json::to_string(&obj(vec![
        ("kind", Json::Str("header".into())),
        ("schema", Json::Str("ny-proof-bundle/v2".into())),
        ("build", Json::Str("test-fixture".into())),
        ("profile", Json::Str("proof".into())),
        ("profile_source", Json::Str("cli".into())),
        ("lane", Json::Str("margin_row".into())),
        ("site_space", Json::Str("onnx-relu-input/v1".into())),
        ("input_space", Json::Str("onnx-input0-flat/v1".into())),
        ("onnx", fref(onnx, onnx_bytes)),
        ("vnnlib", fref(vnn, vnn_bytes)),
    ]))
}

/// The sites line: `(id, name, size, op)`.
pub fn sites(entries: &[(u64, &str, u64, u64)]) -> String {
    json::to_string(&obj(vec![
        ("kind", Json::Str("sites".into())),
        (
            "tensors",
            Json::Arr(
                entries
                    .iter()
                    .map(|&(id, name, size, op)| {
                        obj(vec![
                            ("id", Json::U64(id)),
                            ("name", Json::Str(name.into())),
                            ("size", Json::U64(size)),
                            ("op", Json::U64(op)),
                        ])
                    })
                    .collect(),
            ),
        ),
    ]))
}

/// A clauses line.
pub fn clauses(region: u64, list: &[(u64, Vec<String>)]) -> String {
    json::to_string(&obj(vec![
        ("kind", Json::Str("clauses".into())),
        ("region", Json::U64(region)),
        (
            "list",
            Json::Arr(
                list.iter()
                    .map(|(id, atoms)| {
                        obj(vec![
                            ("id", Json::U64(*id)),
                            (
                                "atoms",
                                Json::Arr(atoms.iter().map(|a| Json::Str(a.clone())).collect()),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
    ]))
}

/// A node line. `dis = None` is `"*"`.
#[allow(clippy::too_many_arguments)]
pub fn node_line(
    id: u64,
    lits: &[L],
    close: &str,
    dis: Option<&[u64]>,
    beta: Option<&[f64]>,
    alpha: Option<u64>,
    lift: bool,
    margin: f64,
) -> String {
    let mut hint = vec![("lift", Json::Bool(lift)), ("margin", Json::F64(margin))];
    if let Some(b) = beta {
        hint.push(("beta", Json::Arr(b.iter().map(|x| Json::F64(*x)).collect())));
    }
    if let Some(a) = alpha {
        hint.push(("alpha", Json::U64(a)));
    }
    json::to_string(&obj(vec![
        ("kind", Json::Str("node".into())),
        ("id", Json::U64(id)),
        ("region", Json::U64(0)),
        (
            "lits",
            Json::Arr(lits.iter().map(|l| lit_json(*l)).collect()),
        ),
        ("close", Json::Str(close.into())),
        ("src", Json::Str("tree".into())),
        (
            "dis",
            match dis {
                None => Json::Str("*".into()),
                Some(d) => Json::Arr(d.iter().map(|x| Json::U64(*x)).collect()),
            },
        ),
        ("hint", obj(hint)),
    ]))
}

/// A `hint.alpha` line.
pub fn alpha_line(id: u64, tensor: u64, idx: &[u64], val: &[f64]) -> String {
    json::to_string(&obj(vec![
        ("kind", Json::Str("hint.alpha".into())),
        ("id", Json::U64(id)),
        ("t", Json::U64(tensor)),
        (
            "idx",
            Json::Arr(idx.iter().map(|x| Json::U64(*x)).collect()),
        ),
        (
            "val",
            Json::Arr(val.iter().map(|x| Json::F64(*x)).collect()),
        ),
        ("start", Json::Str("*".into())),
    ]))
}

/// The verdict line.
pub fn verdict(required: &[u64], nodes: u64) -> String {
    json::to_string(&obj(vec![
        ("kind", Json::Str("verdict".into())),
        ("source", Json::Str("authorized".into())),
        (
            "required",
            Json::Arr(required.iter().map(|x| Json::U64(*x)).collect()),
        ),
        ("min_margin", Json::F64(0.0)),
        ("nodes", Json::U64(nodes)),
    ]))
}

/// The published line.
pub fn published(result: &str) -> String {
    json::to_string(&obj(vec![
        ("kind", Json::Str("published".into())),
        ("result", Json::Str(result.into())),
    ]))
}

/// Join `lines` and append the `end` record with the correct digest.
pub fn finish(lines: &[String]) -> Vec<u8> {
    let mut body = String::new();
    for l in lines {
        body.push_str(l);
        body.push('\n');
    }
    let digest = sha256_hex(body.as_bytes());
    let end = json::to_string(&obj(vec![
        ("kind", Json::Str("end".into())),
        ("lines", Json::U64(lines.len() as u64 + 1)),
        ("sha256", Json::Str(digest)),
    ]));
    body.push_str(&end);
    body.push('\n');
    body.into_bytes()
}

// ---------------------------------------------------------------------------
// Scratch files and runs
// ---------------------------------------------------------------------------

/// A fresh scratch directory for one test.
pub fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("ny-proofcheck-tests")
        .join(name);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// A written fixture.
#[derive(Debug, Clone)]
pub struct Fixture {
    /// Scratch directory.
    pub dir: PathBuf,
    /// ONNX path.
    pub onnx: PathBuf,
    /// VNN-LIB path.
    pub vnnlib: PathBuf,
    /// ONNX bytes.
    pub onnx_bytes: Vec<u8>,
    /// VNN-LIB bytes.
    pub vnn_bytes: Vec<u8>,
}

impl Fixture {
    /// Write `onnx_bytes` and `vnn` under a fresh scratch dir.
    pub fn new(name: &str, onnx_bytes: Vec<u8>, vnn: &str) -> Self {
        let dir = scratch(name);
        let onnx = dir.join("net.onnx");
        let vnnlib = dir.join("prop.vnnlib");
        std::fs::write(&onnx, &onnx_bytes).expect("write onnx");
        std::fs::write(&vnnlib, vnn.as_bytes()).expect("write vnnlib");
        Self {
            dir,
            onnx,
            vnnlib,
            onnx_bytes,
            vnn_bytes: vnn.as_bytes().to_vec(),
        }
    }

    /// The header line for this fixture.
    pub fn header(&self) -> String {
        header(&self.onnx, &self.onnx_bytes, &self.vnnlib, &self.vnn_bytes)
    }

    /// Write `bytes` as `<name>.proof.jsonl` and check it.
    pub fn check(&self, name: &str, bytes: &[u8]) -> CheckOutcome {
        self.check_with(name, bytes, |_| {})
    }

    /// As [`Self::check`], with argument tweaks.
    pub fn check_with(
        &self,
        name: &str,
        bytes: &[u8],
        tweak: impl FnOnce(&mut CheckArgs),
    ) -> CheckOutcome {
        let path = self.dir.join(format!("{name}.proof.jsonl"));
        std::fs::write(&path, bytes).expect("write bundle");
        let mut args = CheckArgs::new(path);
        args.checker_build = "test".into();
        args.threads = Some(2);
        tweak(&mut args);
        check_bundle(&args).expect("operational success")
    }
}

/// Parse a report.
pub fn report(o: &CheckOutcome) -> Json {
    json::from_str(&o.report_json).expect("report is JSON")
}

/// The reason codes of a report.
pub fn reason_codes(o: &CheckOutcome) -> Vec<String> {
    let r = report(o);
    match &r["reasons"] {
        Json::Arr(v) => v
            .iter()
            .filter_map(|x| x["code"].as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// The base fixture: a 2-layer MLP with dyadic weights
// ---------------------------------------------------------------------------

/// `z = W1 x + b1` (3 ReLU inputs, tensor `z`), `h = relu(z)`,
/// `y = W2 h + b2`, with
///
/// * `z0 = x0 + x1`, `z1 = x0 - x1`, `z2 = x0 + x1 + 5` (always active);
/// * `y0 = h0 + h1 + b0`, `y1 = b1`.
///
/// On `x ∈ [-1, 1]^2` the margin `y0 - y1 = relu(z0) + relu(z1) + b0 - b1`
/// has exact minimum `b0 - b1` (at any `x` with `x0 <= -|x1|`): the property
/// "class 1 never reaches class 0" holds iff `b0 > b1`. The root CROWN bound
/// (adaptive α = 1 on both unstable neurons) is `2·x0 + b0 - b1 >= -2 + b0 -
/// b1 < 0`, so the root check FAILS and the split on `z0`, `z1` is needed;
/// every leaf passes with β = 1 on each active literal.
pub fn mlp_onnx(b0: f64, b1: f64) -> Vec<u8> {
    let w1 = [1.0, 1.0, 1.0, -1.0, 1.0, 1.0];
    let bias1 = [0.0, 0.0, 5.0];
    let w2 = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let bias2 = [b0, b1];
    model(
        "x",
        &[1, 2],
        &[
            node("Gemm", &["x", "W1", "B1"], &["z"], vec![A::I("transB", 1)]),
            node("Relu", &["z"], &["h"], vec![]),
            node("Gemm", &["h", "W2", "B2"], &["y"], vec![A::I("transB", 1)]),
        ],
        &[
            tf("W1", &[3, 2], &w1),
            tf("B1", &[3], &bias1),
            tf("W2", &[2, 3], &w2),
            tf("B2", &[2], &bias2),
        ],
        "y",
        &[1, 2],
    )
}

/// The base MLP of [`mlp_onnx`] rebuilt through the lowering's routing forms
/// (b2-checker-lowering), with the SAME function: a rank-1 input `[2]`,
/// `z = (W1 · x_col)ᵀ + b1` (constant on the left), `h = relu(z)`, then a
/// Gather permutation, two Slices and a Concat that restore `h`'s order, and
/// the base output Gemm. Default-domain opset 13.
pub fn routed_mlp_onnx(b0: f64, b1: f64) -> Vec<u8> {
    routed_mlp_onnx_opset(b0, b1, 13)
}

/// [`routed_mlp_onnx`] declaring default-domain `opset` (below 7 its constant
/// Add operand broadcasts by the legacy attributes, which the lowering
/// refuses).
pub fn routed_mlp_onnx_opset(b0: f64, b1: f64, opset: i64) -> Vec<u8> {
    model_ext(
        "x",
        &[Dm::V(2)],
        &[
            node("Reshape", &["x", "col"], &["xc"], vec![]),
            node("MatMul", &["W1", "xc"], &["m"], vec![]),
            node("Reshape", &["m", "row"], &["mr"], vec![]),
            node("Add", &["mr", "B1"], &["z"], vec![]),
            node("Relu", &["z"], &["h"], vec![]),
            node("Gather", &["h", "perm"], &["p"], vec![A::I("axis", 1)]),
            node("Slice", &["p", "s0", "e0", "a1"], &["p01"], vec![]),
            node("Slice", &["p", "s1", "e1", "a1"], &["p2"], vec![]),
            node("Concat", &["p01", "p2"], &["hh"], vec![A::I("axis", 1)]),
            node("Gemm", &["hh", "W2", "B2"], &["y"], vec![A::I("transB", 1)]),
        ],
        &[
            ti("col", &[2], &[2, 1]),
            ti("row", &[2], &[1, 3]),
            ti("perm", &[3], &[2, 0, 1]),
            ti("s0", &[1], &[1]),
            ti("e0", &[1], &[3]),
            ti("s1", &[1], &[0]),
            ti("e1", &[1], &[1]),
            ti("a1", &[1], &[1]),
            tf("W1", &[3, 2], &[1.0, 1.0, 1.0, -1.0, 1.0, 1.0]),
            tf("B1", &[3], &[0.0, 0.0, 5.0]),
            tf("W2", &[2, 3], &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            tf("B2", &[2], &[b0, b1]),
        ],
        "y",
        &[1, 2],
        opset,
    )
}

/// The base property: `x ∈ [-1, 1]^2`, class 0 true, class 1 adversarial.
pub fn mlp_vnnlib() -> String {
    robustness_vnnlib(&["-1", "-1"], &["1", "1"], 2, 0, &[1])
}

/// The canonical clause for class 1 vs true class 0.
pub fn clause01() -> String {
    "Y0:1,Y1:-1<=0".to_string()
}

/// The base bundle's four leaves with their β (site id 0 = tensor `z`).
pub fn mlp_leaves() -> Vec<(Vec<L>, Vec<f64>)> {
    vec![
        (vec![L::R(0, 0, true), L::R(0, 1, true)], vec![1.0, 1.0]),
        (vec![L::R(0, 0, true), L::R(0, 1, false)], vec![1.0, 0.0]),
        (vec![L::R(0, 0, false), L::R(0, 1, true)], vec![0.0, 1.0]),
        (vec![L::R(0, 0, false), L::R(0, 1, false)], vec![0.0, 0.0]),
    ]
}

/// The base bundle's lines (without `end`), leaves as given.
pub fn mlp_bundle_lines(fx: &Fixture, leaves: &[(Vec<L>, Vec<f64>)]) -> Vec<String> {
    let mut lines = vec![
        fx.header(),
        sites(&[(0, "z", 3, 1)]),
        clauses(0, &[(1, vec![clause01()])]),
    ];
    for (k, (lits, beta)) in leaves.iter().enumerate() {
        lines.push(node_line(
            k as u64,
            lits,
            "bound",
            Some(&[1]),
            Some(beta),
            None,
            false,
            1e-6,
        ));
    }
    lines.push(verdict(&[1], leaves.len() as u64));
    lines.push(published("unsat"));
    lines
}

// ---------------------------------------------------------------------------
// B2.3: frozen-dual prune records
// ---------------------------------------------------------------------------

/// A frozen-dual prune node: `close: bound`, `src: fd_prune`, one clause,
/// `hint.fd = {state, l0, l1}` (non-finite multipliers written as `null`).
#[allow(clippy::too_many_arguments)]
pub fn fd_node_line(
    id: u64,
    lits: &[L],
    dis: &[u64],
    state: u64,
    l0: &[f64],
    l1: &[f64],
    margin: f64,
) -> String {
    let nums = |v: &[f64]| {
        Json::Arr(
            v.iter()
                .map(|x| {
                    if x.is_finite() {
                        Json::F64(*x)
                    } else {
                        Json::Null
                    }
                })
                .collect(),
        )
    };
    let fd = obj(vec![
        ("state", Json::U64(state)),
        ("l0", nums(l0)),
        ("l1", nums(l1)),
    ]);
    json::to_string(&obj(vec![
        ("kind", Json::Str("node".into())),
        ("id", Json::U64(id)),
        ("region", Json::U64(0)),
        (
            "lits",
            Json::Arr(lits.iter().map(|l| lit_json(*l)).collect()),
        ),
        ("close", Json::Str("bound".into())),
        ("src", Json::Str("fd_prune".into())),
        (
            "dis",
            Json::Arr(dis.iter().map(|x| Json::U64(*x)).collect()),
        ),
        (
            "hint",
            obj(vec![
                ("beta", Json::Null),
                ("alpha", Json::Null),
                ("lift", Json::Bool(false)),
                ("margin", Json::F64(margin)),
                ("fd", fd),
            ]),
        ),
    ]))
}

/// A `hint.fd_state` line.
pub fn fd_state_line(id: u64, class: u64, order: &[(u64, u64)], prunes: u64) -> String {
    json::to_string(&obj(vec![
        ("kind", Json::Str("hint.fd_state".into())),
        ("id", Json::U64(id)),
        ("class", Json::U64(class)),
        ("t", Json::U64(0)),
        ("state_sha256", Json::Str("0".repeat(64))),
        ("n_gens", Json::U64(4)),
        ("in_gens", Json::U64(2)),
        ("unstable", Json::U64(2)),
        ("root", Json::F64(-1.0)),
        (
            "order",
            Json::Arr(
                order
                    .iter()
                    .map(|&(t, i)| Json::Arr(vec![Json::U64(t), Json::U64(i)]))
                    .collect(),
            ),
        ),
        ("prunes", Json::U64(prunes)),
    ]))
}

/// A `hint.fd_lambda` line.
pub fn fd_lambda_line(id: u64, tensor: u64, idx: &[u64], val: &[f64]) -> String {
    json::to_string(&obj(vec![
        ("kind", Json::Str("hint.fd_lambda".into())),
        ("id", Json::U64(id)),
        ("t", Json::U64(tensor)),
        (
            "idx",
            Json::Arr(idx.iter().map(|x| Json::U64(*x)).collect()),
        ),
        (
            "val",
            Json::Arr(val.iter().map(|x| Json::F64(*x)).collect()),
        ),
    ]))
}

// ---------------------------------------------------------------------------
// The ResNet-shaped fixture
// ---------------------------------------------------------------------------

/// A ResNet-shaped synthetic network on a `1x1x4x4` input: Conv 3x3 (2
/// channels, pad 1) -> BatchNormalization -> Relu (site `b1`, 32) -> Conv 3x3
/// (1 channel) -> residual Add with the input -> Relu (site `a1`, 16) ->
/// Flatten -> Gemm (4) -> Relu (site `g1`, 4) -> Gemm (2 logits, bias
/// `[5, 0]`: class 0 wins by ~5 everywhere on `[0, 1]^16`). Small seeded
/// dyadic weights.
pub fn conv_onnx() -> Vec<u8> {
    let mut s: u64 = 0x1234_5678;
    let mut w = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s % 33) as f64 - 16.0) / 256.0
            })
            .collect()
    };
    let c1w = w(2 * 9);
    let c1b = w(2);
    let c2w = w(2 * 9);
    let c2b = w(1);
    let g1w = w(4 * 16);
    let g1b = w(4);
    let g2w = w(2 * 4);
    model(
        "x",
        &[1, 1, 4, 4],
        &[
            node(
                "Conv",
                &["x", "C1W", "C1B"],
                &["c1"],
                vec![
                    A::Is("pads", vec![1, 1, 1, 1]),
                    A::Is("kernel_shape", vec![3, 3]),
                ],
            ),
            node(
                "BatchNormalization",
                &["c1", "S", "Bb", "M", "V"],
                &["b1"],
                vec![A::F("epsilon", 1e-5)],
            ),
            node("Relu", &["b1"], &["r1"], vec![]),
            node(
                "Conv",
                &["r1", "C2W", "C2B"],
                &["c2"],
                vec![A::Is("pads", vec![1, 1, 1, 1])],
            ),
            node("Add", &["c2", "x"], &["a1"], vec![]),
            node("Relu", &["a1"], &["r2"], vec![]),
            node("Flatten", &["r2"], &["f"], vec![A::I("axis", 1)]),
            node(
                "Gemm",
                &["f", "G1W", "G1B"],
                &["g1"],
                vec![A::I("transB", 1)],
            ),
            node("Relu", &["g1"], &["r3"], vec![]),
            node(
                "Gemm",
                &["r3", "G2W", "G2B"],
                &["y"],
                vec![A::I("transB", 1)],
            ),
        ],
        &[
            tf("C1W", &[2, 1, 3, 3], &c1w),
            tf("C1B", &[2], &c1b),
            tf("S", &[2], &[2.0, 0.5]),
            tf("Bb", &[2], &[0.0, 0.125]),
            tf("M", &[2], &[0.1, -0.1]),
            tf("V", &[2], &[0.25, 1.5]),
            tf("C2W", &[1, 2, 3, 3], &c2w),
            tf("C2B", &[1], &c2b),
            tf("G1W", &[4, 16], &g1w),
            tf("G1B", &[4], &g1b),
            tf("G2W", &[2, 4], &g2w),
            tf("G2B", &[2], &[5.0, 0.0]),
        ],
        "y",
        &[1, 2],
    )
}
