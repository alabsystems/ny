// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end proof-carrying certification of a REAL VNN-COMP ONNX network:
//   ONNX -> exact-rational DeepReluProblem (f32 -> n/2^k, lossless)
//   VNNLIB -> input box + output property
//   ny-cert exact CROWN (crown_deep) -> entailment + farkas certificates
//   (emitted as Clean's external-certificate JSON; checked by clean-extcert-verify)
//
// Usage: certify_onnx <model.onnx> <prop.vnnlib> <out_dir>

#![forbid(unsafe_code)]

use ny_cert::crown_deep::DeepReluProblem;
use ny_cert::rational::{Rat, RatError};
use ny_cert::schema::{
    entailment_to_json, farkas_to_json, ConstraintKind, EntailmentCertificate, FarkasCertificate,
    LinearConstraint,
};
use ny_cert::selfcheck::{check_entailment, check_farkas};
use std::collections::BTreeMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Protobuf wire reader (ONNX is protobuf-encoded): the shared, total
// `ny_cert::protobuf::Reader` — malformed input is a `WireError`, not a panic.
// ---------------------------------------------------------------------------
use ny_cert::protobuf::{Reader as Pb, WireError};

// ---------------------------------------------------------------------------
// ONNX structures we care about.
// ---------------------------------------------------------------------------
/// One parsed ONNX node attribute. Only the scalar `i`/`f` payloads are decoded
/// (all the Gemm attributes we must inspect — `transA`/`transB`/`alpha`/`beta` —
/// are scalars). An attribute whose payload is some other kind decodes with both
/// fields `None`, which every consumer treats as unsupported (fail closed).
#[derive(Debug, Clone)]
struct OnnxAttr {
    name: String,
    i: Option<i64>,
    f: Option<f32>,
}

#[derive(Debug, Clone)]
struct OnnxNode {
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attrs: Vec<OnnxAttr>,
}

#[derive(Debug, Clone)]
struct OnnxTensor {
    name: String,
    dims: Vec<i64>,
    floats: Vec<f32>, // already decoded (from raw_data or float_data)
}

#[derive(Debug)]
struct OnnxGraph {
    nodes: Vec<OnnxNode>,
    inits: BTreeMap<String, OnnxTensor>,
    /// Graph-level input tensor names (GraphProto.input; may include weights on
    /// old exporters — consumers filter against `inits`).
    graph_inputs: Vec<String>,
    /// Graph-level output tensor names (GraphProto.output).
    graph_outputs: Vec<String>,
}

/// Parse an AttributeProto: name=1(str), f=2(fixed32 float), i=3(varint int64).
/// Other payload kinds are skipped (attr decodes with `i`/`f` both None, which
/// consumers reject — fail closed).
fn parse_attr(buf: &[u8]) -> Result<OnnxAttr, WireError> {
    let mut pb = Pb::new(buf);
    let mut a = OnnxAttr {
        name: String::new(),
        i: None,
        f: None,
    };
    while !pb.eof() {
        let (fn_, wt) = pb.tag()?;
        match (fn_, wt) {
            (1, 2) => a.name = String::from_utf8_lossy(pb.bytes()?).into_owned(),
            (2, 5) => a.f = Some(f32::from_le_bytes(pb.fixed32()?)),
            (3, 0) => a.i = Some(pb.varint()? as i64),
            _ => pb.skip(wt)?,
        }
    }
    Ok(a)
}

fn parse_node(buf: &[u8]) -> Result<OnnxNode, WireError> {
    // NodeProto: input=1(str), output=2(str), op_type=4(str), attribute=5(msg)
    let mut pb = Pb::new(buf);
    let mut n = OnnxNode {
        op_type: String::new(),
        inputs: vec![],
        outputs: vec![],
        attrs: vec![],
    };
    while !pb.eof() {
        let (fn_, wt) = pb.tag()?;
        match (fn_, wt) {
            (1, 2) => n
                .inputs
                .push(String::from_utf8_lossy(pb.bytes()?).into_owned()),
            (2, 2) => n
                .outputs
                .push(String::from_utf8_lossy(pb.bytes()?).into_owned()),
            (4, 2) => n.op_type = String::from_utf8_lossy(pb.bytes()?).into_owned(),
            (5, 2) => n.attrs.push(parse_attr(pb.bytes()?)?),
            _ => pb.skip(wt)?,
        }
    }
    Ok(n)
}

/// Parse a ValueInfoProto far enough to extract its name (field 1).
fn parse_value_info_name(buf: &[u8]) -> Result<String, WireError> {
    let mut pb = Pb::new(buf);
    let mut name = String::new();
    while !pb.eof() {
        let (fn_, wt) = pb.tag()?;
        if fn_ == 1 && wt == 2 {
            name = String::from_utf8_lossy(pb.bytes()?).into_owned();
        } else {
            pb.skip(wt)?;
        }
    }
    Ok(name)
}

fn parse_tensor(buf: &[u8]) -> Result<OnnxTensor, WireError> {
    // TensorProto: dims=1(int64 repeated), data_type=2, float_data=4(repeated float),
    //              name=8(str), raw_data=9(bytes)
    let mut pb = Pb::new(buf);
    let mut t = OnnxTensor {
        name: String::new(),
        dims: vec![],
        floats: vec![],
    };
    let mut raw: Option<Vec<u8>> = None;
    let mut float_data: Vec<f32> = vec![];
    while !pb.eof() {
        let (fn_, wt) = pb.tag()?;
        match (fn_, wt) {
            (1, 0) => t.dims.push(pb.varint()? as i64),
            (1, 2) => {
                // packed repeated int64 dims
                let b = pb.bytes()?;
                let mut p2 = Pb::new(b);
                while !p2.eof() {
                    t.dims.push(p2.varint()? as i64);
                }
            }
            (4, 5) => float_data.push(f32::from_le_bytes(pb.fixed32()?)),
            (4, 2) => {
                // packed repeated float
                let b = pb.bytes()?;
                let mut p2 = Pb::new(b);
                while !p2.eof() {
                    float_data.push(f32::from_le_bytes(p2.fixed32()?));
                }
            }
            (8, 2) => t.name = String::from_utf8_lossy(pb.bytes()?).into_owned(),
            (9, 2) => raw = Some(pb.bytes()?.to_vec()),
            _ => pb.skip(wt)?,
        }
    }
    if let Some(r) = raw {
        // raw_data for FLOAT (dtype=1) is little-endian f32.
        for chunk in r.as_chunks::<4>().0 {
            t.floats.push(f32::from_le_bytes(*chunk));
        }
    } else {
        t.floats = float_data;
    }
    Ok(t)
}

fn parse_onnx(data: &[u8]) -> Result<OnnxGraph, OnnxError> {
    // ModelProto: graph=7 (GraphProto)
    let mut pb = Pb::new(data);
    let mut graph_bytes: Option<&[u8]> = None;
    while !pb.eof() {
        let (fn_, wt) = pb.tag()?;
        if fn_ == 7 && wt == 2 {
            graph_bytes = Some(pb.bytes()?);
        } else {
            pb.skip(wt)?;
        }
    }
    let gb = graph_bytes.ok_or(OnnxError::NoGraph)?;
    // GraphProto: node=1(NodeProto), initializer=5(TensorProto),
    //             input=11(ValueInfoProto), output=12(ValueInfoProto)
    let mut pb = Pb::new(gb);
    let mut nodes = vec![];
    let mut inits = BTreeMap::new();
    let mut graph_inputs = vec![];
    let mut graph_outputs = vec![];
    while !pb.eof() {
        let (fn_, wt) = pb.tag()?;
        match (fn_, wt) {
            (1, 2) => nodes.push(parse_node(pb.bytes()?)?),
            (5, 2) => {
                let t = parse_tensor(pb.bytes()?)?;
                inits.insert(t.name.clone(), t);
            }
            (11, 2) => graph_inputs.push(parse_value_info_name(pb.bytes()?)?),
            (12, 2) => graph_outputs.push(parse_value_info_name(pb.bytes()?)?),
            _ => pb.skip(wt)?,
        }
    }
    Ok(OnnxGraph {
        nodes,
        inits,
        graph_inputs,
        graph_outputs,
    })
}

/// Why an `.onnx` model could not be decoded.
#[derive(Debug)]
enum OnnxError {
    /// The protobuf encoding itself is malformed (truncated, overlong, ...).
    Wire(WireError),
    /// The ModelProto carries no `graph` field.
    NoGraph,
}

impl From<WireError> for OnnxError {
    fn from(e: WireError) -> Self {
        Self::Wire(e)
    }
}

impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Wire(e) => write!(f, "malformed ONNX protobuf: {e}"),
            Self::NoGraph => f.write_str("no graph in ONNX"),
        }
    }
}

/// Read and decode an `.onnx` file, or report why not and exit with status 2
/// (the usage/input-error status). A malformed model used to panic inside the
/// protobuf reader; it is now a diagnostic.
fn load_onnx_or_exit(path: &str) -> OnnxGraph {
    let data = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("certify_onnx: cannot read {path}: {e}");
        std::process::exit(2);
    });
    parse_onnx(&data).unwrap_or_else(|e| {
        eprintln!("certify_onnx: {path}: {e}");
        std::process::exit(2);
    })
}

// ---------------------------------------------------------------------------
// f32 -> exact Rat (n / 2^k), lossless.  f32 == m * 2^e with m an integer.
// ---------------------------------------------------------------------------
fn f32_to_rat(v: f32) -> Result<Rat, RatError> {
    let bits = v.to_bits();
    assert!(bits & 0x7f80_0000 != 0x7f80_0000, "non-finite weight {v}");
    Rat::from_f32_exact(v).ok_or(RatError::Poisoned)
}

// ---------------------------------------------------------------------------
// Build an affine chain (W,b per layer) + relu structure from the ONNX graph.
// Supports: Sub (const), Flatten (no-op for vectors), MatMul (x @ W), Add (const), Relu.
// ---------------------------------------------------------------------------
struct AffineLayer {
    w: Vec<Vec<Rat>>, // [out][in]
    b: Vec<Rat>,      // [out]
}

/// A loaded network: hidden layers (each followed by ReLU) and a linear
/// read-out. The read-out is its own field rather than the last element of one
/// `Vec`, so "every loaded net has a read-out" is a fact of the type and no
/// caller computes `layers.len() - 1`.
struct LoadedNet {
    hidden: Vec<AffineLayer>,
    readout: AffineLayer,
    input_dim: usize,
}

impl LoadedNet {
    /// Every affine layer in order: the hidden layers, then the read-out.
    fn layers(&self) -> impl Iterator<Item = &AffineLayer> {
        self.hidden.iter().chain(std::iter::once(&self.readout))
    }
}

/// Longest vector this binary allocates for one layer, input box or output:
/// 2^23 entries (32 MiB of `Rat` ids, 64 MiB of `f64`, 192 MiB for the row
/// table of a 2^23-row matrix). Model dims are untrusted input; past this
/// bound the loaders and the bound passes fail closed (`width_unsupported`)
/// instead of asking the allocator for an input-chosen size. It sits as high
/// as the verified lane's per-allocation bound (under 2^28 bytes) allows, far
/// above the supported networks (ACAS Xu: 50 wide, cersyve: < 100) and above
/// a 3x1024x1024 image input (3.1M), which an earlier 2^20 cap refused.
const MAX_LAYER_WIDTH: usize = 1 << 23;

/// Most entries in one matrix whose SHAPE the model chooses without backing
/// it with data: identity read-outs, DAG basis identities and pass-through
/// blocks, Gemm results over a basis. 2^26 entries (256 MiB of `Rat` ids,
/// e.g. an 8192 x 8192 identity). A per-dimension bound alone is not enough:
/// a 62-byte model declaring a `[2^20, 0]` Gemm asked for a 2^20 x 2^20
/// identity (2^40 entries). Weight matrices (`rat_mat`) share the bound: a
/// single layer of more than 2^26 weights (a 256 MiB f32 tensor) is refused.
const MAX_MATRIX_ENTRIES: usize = 1 << 26;

/// Fail closed on a width past [`MAX_LAYER_WIDTH`]: a diagnostic and exit
/// status 2 (the same status as a malformed model), not an allocation of an
/// input-chosen size.
fn width_unsupported(n: usize) -> ! {
    eprintln!("MODEL_UNSUPPORTED: vector width {n} exceeds MAX_LAYER_WIDTH ({MAX_LAYER_WIDTH})");
    std::process::exit(2);
}

/// `n` zeros; fails closed past `MAX_LAYER_WIDTH`.
fn zeros_f64(n: usize) -> Vec<f64> {
    if n > MAX_LAYER_WIDTH {
        width_unsupported(n);
    }
    vec![0.0; n]
}

/// `n` exact zeros; fails closed past `MAX_LAYER_WIDTH`.
fn zeros_rat(n: usize) -> Vec<Rat> {
    if n > MAX_LAYER_WIDTH {
        width_unsupported(n);
    }
    vec![Rat::ZERO; n]
}

/// Fail closed on a matrix shape past [`MAX_MATRIX_ENTRIES`]: a diagnostic
/// and exit status 2, not an allocation of an input-chosen size.
fn matrix_unsupported(rows: usize, cols: usize) -> ! {
    eprintln!(
        "MODEL_UNSUPPORTED: matrix {rows}x{cols} exceeds MAX_MATRIX_ENTRIES ({MAX_MATRIX_ENTRIES})"
    );
    std::process::exit(2);
}

/// `n` `false` flags; fails closed past `MAX_LAYER_WIDTH`.
fn falses(n: usize) -> Vec<bool> {
    if n > MAX_LAYER_WIDTH {
        width_unsupported(n);
    }
    vec![false; n]
}

/// A `rows x cols` exact zero matrix; fails closed when `rows * cols` is past
/// `MAX_MATRIX_ENTRIES` (or `rows` past `MAX_LAYER_WIDTH`, for `cols == 0`).
fn zero_rat_matrix(rows: usize, cols: usize) -> Vec<Vec<Rat>> {
    let within = rows
        .checked_mul(cols)
        .is_some_and(|entries| entries <= MAX_MATRIX_ENTRIES);
    if !within || rows > MAX_LAYER_WIDTH {
        matrix_unsupported(rows, cols);
    }
    if rows == 0 {
        return Vec::new();
    }
    vec![zeros_rat(cols); rows]
}

fn rat_mat(t: &OnnxTensor, rows: usize, cols: usize) -> Result<Vec<Vec<Rat>>, RatError> {
    // ONNX MatMul stores W as [in, out] row-major (in*out). We want [out][in].
    // `rows`/`cols` come from the model's (untrusted) dims: an overflowing
    // product is a size mismatch, not an arithmetic-overflow panic.
    assert_eq!(
        Some(t.floats.len()),
        rows.checked_mul(cols),
        "matrix size mismatch for {}",
        t.name
    );
    // Convert in the stored (row-major) order, so the first bad weight is the
    // one reported, then transpose: column `c` of the stored matrix is every
    // `cols`-th entry starting at `c`. No index arithmetic.
    let flat: Vec<Rat> = t
        .floats
        .iter()
        .map(|&f| f32_to_rat(f))
        .collect::<Result<_, _>>()?;
    if cols == 0 {
        return Ok(Vec::new()); // [out=0][in=rows]: no columns
    }
    let mut out = zero_rat_matrix(cols, rows); // [out=cols][in=rows]
    for (c, col) in out.iter_mut().enumerate() {
        for (slot, v) in col.iter_mut().zip(flat.iter().skip(c).step_by(cols)) {
            *slot = *v;
        }
    }
    Ok(out)
}

fn rat_vec(t: &OnnxTensor) -> Result<Vec<Rat>, RatError> {
    t.floats.iter().map(|&f| f32_to_rat(f)).collect()
}

fn load_net(g: &OnnxGraph) -> LoadedNet {
    // Walk nodes, tracking the symbolic affine transform applied to the input.
    // We fold Sub (subtract const vector) into a pending bias, MatMul sets pending W,
    // Add sets pending b, Relu closes a layer.
    let mut layers: Vec<AffineLayer> = vec![];
    let mut input_dim = 0usize;

    // pending affine accumulators between ReLUs (or until the read-out).
    // We represent the running pre-activation as W_run @ x_prev + b_run, where
    // x_prev is the previous ReLU output (or the input for the first layer).
    // A Sub before the first MatMul subtracts a const c: x' = x - c, so
    // MatMul(x') = W @ (x-c) = W@x - W@c -> folded into bias.
    let mut sub_const: Option<Vec<Rat>> = None; // pending input shift (x - sub_const)
    let mut cur_w: Option<Vec<Vec<Rat>>> = None;
    let mut cur_b: Option<Vec<Rat>> = None;
    // Activation dimension currently flowing into the next affine layer.
    // 0 means "unknown / input not yet sized".
    let mut cur_dim: usize = 0;

    for node in &g.nodes {
        match node.op_type.as_str() {
            "Sub" => {
                // input - const ; const is whichever input is an initializer.
                let cname = if g.inits.contains_key(&node.inputs[1]) {
                    &node.inputs[1]
                } else {
                    &node.inputs[0]
                };
                let t = &g.inits[cname];
                let c = rat_vec(t).expect("sub const");
                if input_dim == 0 {
                    input_dim = c.len();
                    cur_dim = c.len();
                }
                sub_const = Some(c);
            }
            "Flatten" => { /* no-op for 1-D vectors */ }
            "MatMul" => {
                // The weight is whichever MatMul input is an initializer.
                let wname = if g.inits.contains_key(&node.inputs[1]) {
                    &node.inputs[1]
                } else {
                    &node.inputs[0]
                };
                let t = &g.inits[wname];
                let (d0, d1) = (t.dims[0] as usize, t.dims[1] as usize);
                // Determine orientation so that W stored as [out][in], where `in`
                // matches the running activation dimension.
                //   act @ W : W is [in, out]  (in == d0)
                //   W @ act : W is [out, in]  (in == d1)
                let (rin, cout, transpose) = if cur_dim == 0 {
                    // First layer, input dim unknown. ONNX MatMul(act, W) => W=[in,out];
                    // MatMul(W, act) => W=[out,in]. Disambiguate by input order.
                    if g.inits.contains_key(&node.inputs[1]) {
                        (d0, d1, false) // act @ W : [in,out]
                    } else {
                        (d1, d0, true) // W @ act : [out,in]
                    }
                } else if d0 == cur_dim {
                    (d0, d1, false) // [in,out]
                } else if d1 == cur_dim {
                    (d1, d0, true) // [out,in]
                } else {
                    panic!("MatMul W dims {d0}x{d1} match neither side of cur_dim {cur_dim}");
                };
                if input_dim == 0 {
                    input_dim = rin;
                }
                // rat_mat reads flat [d0*d1] row-major and returns [d1][d0] (i.e.
                // out=d1, in=d0) for the canonical [in,out] layout. For the
                // transposed [out,in] layout we read it as [in=d1][out=d0] then
                // we already set rin=d1,cout=d0; rat_mat(t,d0,d1) returns [d1][d0]
                // = [cout][rin] only when NOT transposed. Handle both explicitly.
                let w: Vec<Vec<Rat>> = if !transpose {
                    rat_mat(t, d0, d1).expect("matmul W [in,out]") // [out=d1][in=d0]
                } else {
                    // W stored [out=d0, in=d1] row-major: w[o][i] = floats[o*d1 + i].
                    let mut w = vec![vec![Rat::ZERO; d1]; d0];
                    for o in 0..d0 {
                        for i in 0..d1 {
                            w[o][i] = f32_to_rat(t.floats[o * d1 + i]).unwrap();
                        }
                    }
                    w // [out=d0][in=d1]
                };
                let out_w = w.len();
                let in_w = w[0].len();
                cur_dim = out_w;
                let _ = (rin, cout);
                // If there's a pending input shift, fold it into a bias.
                // W is [out][in]. W @ (x - c) = W@x - W@c. Subtracted term per out o:
                //   sum_i W[o][i] * c[i].
                if let Some(c) = sub_const.take() {
                    assert_eq!(c.len(), in_w, "sub const dim != layer in dim");
                    let mut fold = vec![Rat::ZERO; out_w];
                    for o in 0..out_w {
                        let mut acc = Rat::ZERO;
                        for i in 0..in_w {
                            acc = acc.add(w[o][i].mul(c[i]).unwrap()).unwrap();
                        }
                        fold[o] = acc.neg();
                    }
                    cur_b = Some(fold);
                }
                cur_w = Some(w);
            }
            "Add" => {
                let bname = if g.inits.contains_key(&node.inputs[1]) {
                    &node.inputs[1]
                } else {
                    &node.inputs[0]
                };
                let t = &g.inits[bname];
                let bvec = rat_vec(t).expect("add b");
                // Add to any folded sub bias.
                cur_b = Some(match cur_b.take() {
                    Some(prev) => prev
                        .iter()
                        .zip(&bvec)
                        .map(|(a, b)| a.add(*b).unwrap())
                        .collect(),
                    None => bvec,
                });
            }
            "Relu" => {
                let w = cur_w.take().expect("relu without matmul");
                // A zero bias with one entry per output row of `w`.
                let b = cur_b.take().unwrap_or_else(|| zeros_rat(w.len()));
                layers.push(AffineLayer { w, b });
            }
            other => panic!("unsupported op {other}"),
        }
    }
    // Final linear read-out (the last MatMul+Add not followed by Relu).
    let readout = if let Some(w) = cur_w.take() {
        let b = cur_b.take().unwrap_or_else(|| zeros_rat(w.len()));
        AffineLayer { w, b }
    } else {
        // The network ENDS in a ReLU (the last hidden layer is the output, e.g.
        // y = ReLU(W x)).  DeepReluProblem requires a linear read-out after the
        // last ReLU, so synthesize an identity read-out  y = a_last.
        let last_width = layers.last().expect("no layers at all").w.len();
        let mut w = zero_rat_matrix(last_width, last_width);
        for (d, row) in w.iter_mut().enumerate() {
            if let Some(diag) = row.get_mut(d) {
                *diag = Rat::ONE; // square: `d < last_width` always
            }
        }
        AffineLayer {
            w,
            b: zeros_rat(last_width),
        }
    };
    LoadedNet {
        hidden: layers,
        readout,
        input_dim,
    }
}

// ---------------------------------------------------------------------------
// DAG-AWARE EXACT LOADER (cersyve shapes).
//
// Supports exactly: Gemm (default attrs only: transA=0, transB=0, alpha=1,
// beta=1), Relu, and Add of two computed tensors (residual merges), over an
// arbitrary DAG in topological node order. Everything is tracked SYMBOLICALLY:
// each tensor value is an exact-rational affine expression over "basis"
// variables — basis 0 is the network input x, basis r >= 1 is the output of
// the r-th ReLU node. The DAG is then flattened to an EQUIVALENT sequential
// `LoadedNet` (the algebra is exact over rationals):
//   * each ReLU is scheduled at level = 1 + max level of the basis vars its
//     pre-activation reads (input has level 0);
//   * values needed across levels are carried by PASSTHROUGH units that are
//     provably stable-active (preact >= 0 over the box), so ReLU is the
//     identity on them and the flattening introduces NO extra relaxation:
//       - the input x is carried as x - lo_root (>= 0 on the vnnlib box),
//       - a ReLU activation is carried as itself (>= 0 by definition).
// FAIL CLOSED (panic) on any op, attribute, or shape outside this fragment.
// ---------------------------------------------------------------------------

/// Exact affine expression over basis variables: `Σ_b M_b · v_b + cst`, where
/// `v_0 = x` (the input) and `v_r` (r >= 1) is the r-th ReLU node's output.
#[derive(Debug, Clone)]
struct Aff {
    /// basis id -> coefficient matrix `[rows][width(basis)]`.
    terms: BTreeMap<usize, Vec<Vec<Rat>>>,
    /// constant vector `[rows]`.
    cst: Vec<Rat>,
}

impl Aff {
    fn rows(&self) -> usize {
        self.cst.len()
    }

    /// Identity expression `v_basis` (width `dim`).
    /// `dim` is the model's (untrusted) Gemm input width or an existing ReLU
    /// width; past `MAX_MATRIX_ENTRIES` (`dim * dim`) this fails closed before
    /// allocating.
    fn identity(basis: usize, dim: usize) -> Aff {
        let mut m = zero_rat_matrix(dim, dim);
        for (d, row) in m.iter_mut().enumerate() {
            if let Some(diag) = row.get_mut(d) {
                *diag = Rat::ONE; // square: `d < dim` always
            }
        }
        let mut terms = BTreeMap::new();
        terms.insert(basis, m);
        Aff {
            terms,
            cst: zeros_rat(dim),
        }
    }

    /// Drop all-zero coefficient matrices (keeps ReLU levels minimal).
    fn prune(mut self) -> Aff {
        self.terms
            .retain(|_, m| m.iter().any(|row| row.iter().any(|c| !c.is_zero())));
        self
    }
}

/// The parsed DAG: per-ReLU pre-activation expressions plus the graph output,
/// all exact affine forms over the basis variables.
struct DagNet {
    input_dim: usize,
    /// `relu_pre[r-1]` = pre-activation of basis var `r` (over basis vars < r).
    relu_pre: Vec<Aff>,
    /// Graph output as an affine form over basis vars.
    output: Aff,
}

impl DagNet {
    fn basis_width(&self, b: usize) -> usize {
        if b == 0 {
            self.input_dim
        } else {
            self.relu_pre[b - 1].rows()
        }
    }
}

/// `out = xᵀW + c` applied to an affine expression (`W` given flat `[kin*kout]`
/// row-major, i.e. ONNX Gemm with transB=0): exact rational algebra.
fn gemm_apply(a: &Aff, wfloats: &[f32], kin: usize, kout: usize, bias: &[Rat]) -> Aff {
    assert_eq!(a.rows(), kin, "Gemm inner dimension mismatch");
    assert_eq!(
        Some(wfloats.len()),
        kin.checked_mul(kout),
        "Gemm weight size mismatch"
    );
    assert_eq!(bias.len(), kout, "Gemm bias size mismatch");
    if kout == 0 {
        // No output rows: every term matrix is empty and `prune` drops it.
        return Aff {
            terms: BTreeMap::new(),
            cst: Vec::new(),
        };
    }
    // Exact weights in stored row-major order (row `i` = input `i`, `kout`
    // entries); conversion order, and so the first bad weight, is unchanged.
    let w: Vec<Rat> = wfloats
        .iter()
        .map(|&f| f32_to_rat(f).expect("gemm weight"))
        .collect();
    let mut terms = BTreeMap::new();
    for (b, m) in &a.terms {
        let bw = m.first().map_or(0, Vec::len);
        // Every term matrix of `a` is `[a.rows()][bw]` by construction.
        assert!(
            m.len() == kin && m.iter().all(|r| r.len() == bw),
            "Gemm operand term for basis {b} is not [{kin}][{bw}]"
        );
        let mut out = zero_rat_matrix(kout, bw);
        // `w_i` is weight row `i`, `m_i` the operand's row `i`; `chunks_exact`
        // (kout >= 1 here) yields exactly `kin` rows.
        for (w_i, m_i) in w.chunks_exact(kout).zip(m) {
            for (orow, &wio) in out.iter_mut().zip(w_i) {
                if wio.is_zero() {
                    continue;
                }
                for (slot, &mc) in orow.iter_mut().zip(m_i) {
                    *slot = slot.add(wio.mul(mc).unwrap()).unwrap();
                }
            }
        }
        terms.insert(*b, out);
    }
    // cst[o] = bias[o] + sum_i w[i][o] * a.cst[i], summed in increasing `i`.
    let mut cst = bias.to_vec();
    for (w_i, &ci) in w.chunks_exact(kout).zip(&a.cst) {
        for (slot, &wio) in cst.iter_mut().zip(w_i) {
            *slot = slot.add(wio.mul(ci).unwrap()).unwrap();
        }
    }
    Aff { terms, cst }.prune()
}

/// Elementwise sum of two affine expressions.
fn aff_add(a: &Aff, b: &Aff) -> Aff {
    assert_eq!(a.rows(), b.rows(), "Add operand widths differ");
    let mut terms = a.terms.clone();
    for (bb, m) in &b.terms {
        match terms.get_mut(bb) {
            Some(acc) => {
                for (u, row) in m.iter().enumerate() {
                    for (c, v) in row.iter().enumerate() {
                        acc[u][c] = acc[u][c].add(*v).unwrap();
                    }
                }
            }
            None => {
                terms.insert(*bb, m.clone());
            }
        }
    }
    let cst = a
        .cst
        .iter()
        .zip(&b.cst)
        .map(|(x, y)| x.add(*y).unwrap())
        .collect();
    Aff { terms, cst }.prune()
}

/// Fail closed unless every Gemm attribute present is an explicit default
/// (transA=0, transB=0, alpha=1.0, beta=1.0). Cersyve nets carry no attrs.
fn check_gemm_attrs(node: &OnnxNode) {
    for a in &node.attrs {
        let ok = match a.name.as_str() {
            "transA" | "transB" => a.i == Some(0),
            "alpha" | "beta" => a.f == Some(1.0),
            _ => false,
        };
        assert!(
            ok,
            "unsupported Gemm attribute {:?} (i={:?}, f={:?}) — fail closed",
            a.name, a.i, a.f
        );
    }
}

/// Parse the supported DAG fragment into symbolic affine form. Panics (fail
/// closed) on anything outside {Gemm(default attrs), Relu, Add(value,value)}.
fn load_dag(g: &OnnxGraph) -> DagNet {
    let input_name = g
        .graph_inputs
        .iter()
        .find(|n| !g.inits.contains_key(*n))
        .expect("no non-initializer graph input")
        .clone();
    let mut vals: BTreeMap<String, Aff> = BTreeMap::new();
    let mut relu_pre: Vec<Aff> = vec![];
    let mut input_dim = 0usize;
    // Resolve a node input tensor to its symbolic value. The raw graph input is
    // materialized lazily once its width is known (from the first Gemm using it).
    let get_val =
        |vals: &BTreeMap<String, Aff>, name: &str| -> Option<Aff> { vals.get(name).cloned() };
    for node in &g.nodes {
        assert_eq!(
            node.outputs.len(),
            1,
            "multi-output node {:?} unsupported",
            node.op_type
        );
        let out_name = node.outputs[0].clone();
        let value = match node.op_type.as_str() {
            "Gemm" => {
                check_gemm_attrs(node);
                assert_eq!(node.inputs.len(), 3, "Gemm without bias unsupported");
                assert!(
                    !g.inits.contains_key(&node.inputs[0]),
                    "Gemm data input is an initializer — unsupported orientation"
                );
                let w = g
                    .inits
                    .get(&node.inputs[1])
                    .expect("Gemm B must be an initializer");
                let c = g
                    .inits
                    .get(&node.inputs[2])
                    .expect("Gemm C must be an initializer");
                assert_eq!(w.dims.len(), 2, "Gemm B must be 2-D");
                let (kin, kout) = (w.dims[0] as usize, w.dims[1] as usize);
                let bias = rat_vec(c).expect("gemm bias");
                assert_eq!(bias.len(), kout, "Gemm C length != output width");
                let a = match get_val(&vals, &node.inputs[0]) {
                    Some(a) => a,
                    None => {
                        assert_eq!(
                            node.inputs[0], input_name,
                            "Gemm input {} not yet computed (non-topological graph?)",
                            node.inputs[0]
                        );
                        if input_dim == 0 {
                            input_dim = kin;
                        }
                        assert_eq!(input_dim, kin, "inconsistent input width");
                        Aff::identity(0, kin)
                    }
                };
                gemm_apply(&a, &w.floats, kin, kout, &bias)
            }
            "Relu" => {
                let a = get_val(&vals, &node.inputs[0])
                    .expect("Relu on the raw input (or unknown tensor) unsupported");
                let width = a.rows();
                relu_pre.push(a);
                Aff::identity(relu_pre.len(), width)
            }
            "Add" => {
                assert!(
                    !g.inits.contains_key(&node.inputs[0])
                        && !g.inits.contains_key(&node.inputs[1]),
                    "Add with an initializer operand unsupported in the DAG loader"
                );
                let a = get_val(&vals, &node.inputs[0]).expect("Add lhs not computed");
                let b = get_val(&vals, &node.inputs[1]).expect("Add rhs not computed");
                aff_add(&a, &b)
            }
            other => panic!("unsupported op {other} in DAG loader — fail closed"),
        };
        vals.insert(out_name, value);
    }
    assert!(input_dim > 0, "input never consumed by a Gemm");
    let out_name = g.graph_outputs.first().expect("no graph output");
    let output = vals
        .get(out_name)
        .unwrap_or_else(|| panic!("graph output {out_name} not computed"))
        .clone();
    DagNet {
        input_dim,
        relu_pre,
        output,
    }
}

/// Flatten the DAG to an equivalent sequential `LoadedNet`. `root_lo` is the
/// vnnlib (root) input lower bound: the input passthrough is `x - root_lo`,
/// which stays >= 0 on every sub-box of the root box, keeping the passthrough
/// units stable-active (ReLU = identity, no extra relaxation). Exact algebra.
fn flatten_dag(dag: &DagNet, root_lo: &[Rat]) -> LoadedNet {
    let nr = dag.relu_pre.len();
    assert!(nr > 0, "graph has no ReLU — nothing to flatten");
    assert_eq!(root_lo.len(), dag.input_dim, "root_lo arity");
    // Levels: level[0] = 0 (input); level[r] = 1 + max level of read basis vars.
    let mut level = vec![0usize; nr + 1];
    for r in 1..=nr {
        let mut lv = 0usize;
        for b in dag.relu_pre[r - 1].terms.keys() {
            assert!(
                *b < r,
                "ReLU {r} reads a later basis var {b} — not topological"
            );
            lv = lv.max(level[*b]);
        }
        level[r] = lv + 1;
    }
    let depth = (1..=nr).map(|r| level[r]).max().unwrap();
    // need_until[b] = last layer whose ACTIVATION vector must expose b's value:
    // a ReLU at level L reads its inputs from layer L-1; the read-out reads
    // from layer `depth`.
    let mut need_until = vec![0usize; nr + 1];
    for r in 1..=nr {
        for b in dag.relu_pre[r - 1].terms.keys() {
            need_until[*b] = need_until[*b].max(level[r] - 1);
        }
    }
    for b in dag.output.terms.keys() {
        need_until[*b] = need_until[*b].max(depth);
    }
    // Unit layout per layer: ReLUs of this level first, then passthroughs.
    #[derive(Clone, Copy)]
    enum Unit {
        Relu(usize),
        Pass(usize), // basis id (0 = shifted input x - root_lo)
    }
    let mut layer_units: Vec<Vec<Unit>> = vec![vec![]]; // index 0 unused
    let mut slot: Vec<BTreeMap<usize, usize>> = vec![BTreeMap::new()]; // slot[L][basis] = col
    for lvl in 1..=depth {
        let mut units = vec![];
        let mut cols: BTreeMap<usize, usize> = BTreeMap::new();
        let mut col = 0usize;
        for r in 1..=nr {
            if level[r] == lvl {
                cols.insert(r, col);
                col += dag.basis_width(r);
                units.push(Unit::Relu(r));
            }
        }
        for b in 0..=nr {
            if level[b] < lvl && need_until[b] >= lvl {
                cols.insert(b, col);
                col += dag.basis_width(b);
                units.push(Unit::Pass(b));
            }
        }
        layer_units.push(units);
        slot.push(cols);
    }
    // Translate an affine expression over basis vars into a weight-row block
    // over layer `src` activations (src = 0 means the raw input x).
    let width_of_layer = |lvl: usize| -> usize {
        if lvl == 0 {
            dag.input_dim
        } else {
            layer_units[lvl]
                .iter()
                .map(|u| match u {
                    Unit::Relu(r) => dag.basis_width(*r),
                    Unit::Pass(b) => dag.basis_width(*b),
                })
                .sum()
        }
    };
    // Rows `row0 .. row0 + aff.rows()` of `w`/`bv` receive `aff`; a term over
    // basis `b` lands in the column block where layer `src` carries `b`.
    // Iterators over those blocks, not `row0 + u` / `col0 + c` indexing; the
    // block bounds the indexing used to enforce are asserted up front, so a
    // layout bug still fails closed instead of being truncated.
    let emit_aff = |aff: &Aff, src: usize, w: &mut [Vec<Rat>], bv: &mut [Rat], row0: usize| {
        let nrows = aff.rows();
        assert!(
            w.len() == bv.len() && row0.checked_add(nrows).is_some_and(|end| end <= w.len()),
            "affine block rows {row0}+{nrows} exceed the layer's {} rows",
            w.len()
        );
        for (b, m) in &aff.terms {
            let (col0, shift_const) = if src == 0 {
                assert_eq!(*b, 0, "level-1 preact reads a ReLU var — level bug");
                (0usize, false)
            } else if *b == 0 {
                let c = *slot[src]
                    .get(&0)
                    .unwrap_or_else(|| panic!("x not carried at layer {src}"));
                (c, true)
            } else {
                let c = *slot[src]
                    .get(b)
                    .unwrap_or_else(|| panic!("basis {b} not carried at layer {src}"));
                (c, false)
            };
            assert!(
                m.len() <= nrows,
                "term for basis {b} has more rows than the expression"
            );
            for ((wrow, bslot), mrow) in w.iter_mut().zip(bv.iter_mut()).skip(row0).zip(m) {
                assert!(
                    col0.checked_add(mrow.len())
                        .is_some_and(|end| end <= wrow.len()),
                    "basis {b} columns {col0}+{} exceed the layer width {}",
                    mrow.len(),
                    wrow.len()
                );
                for (slot, v) in wrow.iter_mut().skip(col0).zip(mrow) {
                    if !v.is_zero() {
                        *slot = slot.add(*v).unwrap();
                    }
                }
                if shift_const {
                    // carried value is x_c - root_lo[c]; add back v * root_lo[c]
                    // (same terms, same increasing-`c` order).
                    assert!(mrow.len() <= root_lo.len(), "input term wider than root_lo");
                    for (v, lo_c) in mrow.iter().zip(root_lo) {
                        if !v.is_zero() {
                            *bslot = bslot.add(v.mul(*lo_c).unwrap()).unwrap();
                        }
                    }
                }
            }
        }
        for (bslot, v) in bv.iter_mut().skip(row0).zip(&aff.cst) {
            *bslot = bslot.add(*v).unwrap();
        }
    };
    let mut layers: Vec<AffineLayer> = vec![];
    for lvl in 1..=depth {
        let rows = width_of_layer(lvl);
        let prev = width_of_layer(lvl - 1);
        let mut w = zero_rat_matrix(rows, prev);
        let mut bv = zeros_rat(rows);
        let mut row = 0usize;
        for unit in &layer_units[lvl] {
            match unit {
                Unit::Relu(r) => {
                    emit_aff(&dag.relu_pre[*r - 1], lvl - 1, &mut w, &mut bv, row);
                    row += dag.basis_width(*r);
                }
                Unit::Pass(0) => {
                    if lvl == 1 {
                        for i in 0..dag.input_dim {
                            w[row + i][i] = Rat::ONE;
                            bv[row + i] = root_lo[i].neg();
                        }
                    } else {
                        let c = *slot[lvl - 1].get(&0).expect("x chain broken");
                        for i in 0..dag.input_dim {
                            w[row + i][c + i] = Rat::ONE;
                        }
                    }
                    row += dag.input_dim;
                }
                Unit::Pass(b) => {
                    let c = *slot[lvl - 1]
                        .get(b)
                        .unwrap_or_else(|| panic!("pass chain for basis {b} broken"));
                    for i in 0..dag.basis_width(*b) {
                        w[row + i][c + i] = Rat::ONE;
                    }
                    row += dag.basis_width(*b);
                }
            }
        }
        assert_eq!(row, rows);
        layers.push(AffineLayer { w, b: bv });
    }
    // Linear read-out from layer `depth`.
    let out_dim = dag.output.rows();
    let prev = width_of_layer(depth);
    let mut w = zero_rat_matrix(out_dim, prev);
    let mut bv = zeros_rat(out_dim);
    emit_aff(&dag.output, depth, &mut w, &mut bv, 0);
    LoadedNet {
        hidden: layers,
        readout: AffineLayer { w, b: bv },
        input_dim: dag.input_dim,
    }
}

/// Exact forward evaluation of the parsed DAG at a rational point (all outputs).
fn eval_aff_at(aff: &Aff, basis: &[Vec<Rat>]) -> Vec<Rat> {
    let mut out = aff.cst.clone();
    for (b, m) in &aff.terms {
        let Some(v) = basis.get(*b) else {
            panic!("basis var {b} read before it is evaluated");
        };
        // `[rows][width(b)]` against `v` of `width(b)`: the shape the old
        // `out[u]`/`v[c]` indexing enforced, checked once.
        assert!(
            m.len() <= out.len() && m.iter().all(|row| row.len() <= v.len()),
            "term for basis {b} does not match the expression/basis widths"
        );
        for (slot, row) in out.iter_mut().zip(m) {
            for (coef, vc) in row.iter().zip(v) {
                if !coef.is_zero() {
                    *slot = slot.add(coef.mul(*vc).unwrap()).unwrap();
                }
            }
        }
    }
    out
}

fn dag_eval_exact(dag: &DagNet, x: &[Rat]) -> Vec<Rat> {
    assert_eq!(x.len(), dag.input_dim);
    let mut basis: Vec<Vec<Rat>> = vec![x.to_vec()];
    for pre in &dag.relu_pre {
        let z = eval_aff_at(pre, &basis);
        basis.push(
            z.iter()
                .map(|v| if v.is_positive() { *v } else { Rat::ZERO })
                .collect(),
        );
    }
    eval_aff_at(&dag.output, &basis)
}

/// Exact forward evaluation of a flattened `LoadedNet` (vector output).
fn loaded_eval_vec(net: &LoadedNet, x: &[Rat]) -> Vec<Rat> {
    assert_eq!(x.len(), net.input_dim);
    let mut act: Vec<Rat> = x.to_vec();
    for layer in &net.hidden {
        let mut next = Vec::new();
        for (row, b) in layer.w.iter().zip(&layer.b) {
            let mut z = *b;
            for (wv, a) in row.iter().zip(&act) {
                z = z.add(wv.mul(*a).unwrap()).unwrap();
            }
            next.push(if z.is_positive() { z } else { Rat::ZERO });
        }
        act = next;
    }
    let last = &net.readout;
    last.w
        .iter()
        .zip(&last.b)
        .map(|(row, b)| {
            let mut y = *b;
            for (wv, a) in row.iter().zip(&act) {
                y = y.add(wv.mul(*a).unwrap()).unwrap();
            }
            y
        })
        .collect()
}

/// Independent f32 interpreter over the ORIGINAL node list (natural summation
/// order) — the parity reference for the exact loader. Same fail-closed fragment.
fn f32_forward(g: &OnnxGraph, x: &[f32]) -> Vec<f32> {
    let input_name = g
        .graph_inputs
        .iter()
        .find(|n| !g.inits.contains_key(*n))
        .expect("no graph input");
    let mut vals: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    vals.insert(input_name.clone(), x.to_vec());
    for node in &g.nodes {
        let out = match node.op_type.as_str() {
            "Gemm" => {
                check_gemm_attrs(node);
                let a = vals.get(&node.inputs[0]).expect("gemm input").clone();
                let w = &g.inits[&node.inputs[1]];
                let c = &g.inits[&node.inputs[2]];
                let (kin, kout) = (w.dims[0] as usize, w.dims[1] as usize);
                assert_eq!(a.len(), kin);
                // The shape `load_dag` already enforces on this graph; checked
                // here too so the column walk below reads exactly `kin` weights.
                assert_eq!(
                    Some(w.floats.len()),
                    kin.checked_mul(kout),
                    "Gemm weight size mismatch"
                );
                assert_eq!(c.floats.len(), kout, "Gemm C length != output width");
                let mut y = c.floats.clone();
                // `kout == 0` leaves `y` empty: nothing to accumulate.
                if kout > 0 {
                    for (o, slot) in y.iter_mut().enumerate() {
                        // Column `o` of the row-major `[kin][kout]` weights is
                        // every `kout`-th entry from `o`; summed in increasing
                        // `i`, the natural order.
                        let mut acc = 0.0f32;
                        for (ai, wio) in a.iter().zip(w.floats.iter().skip(o).step_by(kout)) {
                            acc += ai * wio;
                        }
                        *slot += acc;
                    }
                }
                y
            }
            "Relu" => vals[&node.inputs[0]].iter().map(|v| v.max(0.0)).collect(),
            "Add" => {
                let a = &vals[&node.inputs[0]];
                let b = &vals[&node.inputs[1]];
                a.iter().zip(b).map(|(x, y)| x + y).collect()
            }
            other => panic!("unsupported op {other} in f32 forward"),
        };
        vals.insert(node.outputs[0].clone(), out);
    }
    vals[g.graph_outputs.first().expect("no output")].clone()
}

// ---------------------------------------------------------------------------
// VNNLIB parser (subset): input box on X_i, output property on Y_j.
// We return (lower[X], upper[X], properties) where each property is
// (var_index, kind, rhs) over Y. We handle the common single-output ACAS props
// and the test nets.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct VnnAtom {
    var: String,     // e.g. "Y_0" or "X_2"
    op: String,      // "<=" or ">="
    rhs_raw: String, // raw decimal token, parsed losslessly to Rat
}

fn parse_vnnlib(text: &str) -> Vec<VnnAtom> {
    // Extract all (assert (>= VAR NUM)) and (<= VAR NUM) atoms, flattening
    // and/or structure. For these benchmarks the box constraints are flat and
    // the property is a single clause; we collect every atom.
    let mut atoms = vec![];
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'(' {
            // try to read an op token
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j] == b' ') {
                j += 1;
            }
            // read token
            let start = j;
            while j < bytes.len()
                && !bytes[j].is_ascii_whitespace()
                && bytes[j] != b'('
                && bytes[j] != b')'
            {
                j += 1;
            }
            let tok = &text[start..j];
            if tok == "<=" || tok == ">=" {
                // next: a var token, then a number token
                let mut k = j;
                while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                    k += 1;
                }
                let vs = k;
                while k < bytes.len() && !bytes[k].is_ascii_whitespace() && bytes[k] != b')' {
                    k += 1;
                }
                let var = &text[vs..k];
                while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                    k += 1;
                }
                let ns = k;
                while k < bytes.len() && !bytes[k].is_ascii_whitespace() && bytes[k] != b')' {
                    k += 1;
                }
                let numtok = &text[ns..k];
                if (var.starts_with("X_") || var.starts_with("Y_"))
                    && var_index(var).is_some()
                    && numtok.parse::<f64>().is_ok()
                {
                    atoms.push(VnnAtom {
                        var: var.to_string(),
                        op: tok.to_string(),
                        rhs_raw: numtok.to_string(),
                    });
                }
            }
        }
        i += 1;
    }
    atoms
}

/// Parse the numeric index out of an `X_<n>` / `Y_<n>` VNNLIB variable.
///
/// Returns `None` for a malformed suffix — non-numeric (`Y_abc`), empty (`Y_`),
/// or overflowing (`X_99999999999999999999`) — so an untrusted/mistyped property
/// file fails closed instead of panicking on `.unwrap()`.
fn var_index(v: &str) -> Option<usize> {
    v.get(2..).and_then(|s| s.parse::<usize>().ok())
}

/// Input arity of a VNNLIB property: one more than the largest `X_i` index.
///
/// The index is untrusted text, so this fails closed instead of computing
/// `i + 1` (which overflowed for `X_18446744073709551615`) and then sizing the
/// box vectors by it (`X_100000000000` asked for a 10^11-element allocation).
/// A complete box needs a lower and an upper bound atom for every coordinate,
/// so a property whose arity exceeds its number of `X` atoms is incomplete
/// whatever else it says; rejecting it here keeps every complete property
/// accepted and bounds the box allocation by the size of the input file.
fn vnnlib_input_dim(atoms: &[VnnAtom]) -> Result<usize, String> {
    let mut n_x = 0usize;
    let mut max_idx: Option<usize> = None;
    for a in atoms.iter().filter(|a| a.var.starts_with("X_")) {
        n_x = n_x.saturating_add(1);
        if let Some(i) = var_index(&a.var) {
            max_idx = Some(max_idx.map_or(i, |m| m.max(i)));
        }
    }
    let max_idx = max_idx.ok_or_else(|| "vnnlib has no X_i atoms".to_string())?;
    // `checked_add` is the real overflow case (`X_<usize::MAX>`); the filter is
    // the completeness bound. Either failure is the same out-of-range error.
    let dim = max_idx
        .checked_add(1)
        .filter(|&dim| dim <= n_x)
        .ok_or_else(|| {
            format!(
                "input index X_{max_idx} is out of range: only {n_x} X bound atoms, so the box cannot be complete"
            )
        })?;
    // The box is a layer input: the same width bound as every loaded layer.
    if dim > MAX_LAYER_WIDTH {
        return Err(format!(
            "input dimension {dim} exceeds MAX_LAYER_WIDTH ({MAX_LAYER_WIDTH})"
        ));
    }
    Ok(dim)
}

/// Parse a decimal literal into an exact Rat (lossless: decimals are n/10^k).
fn decimal_to_rat(s: &str) -> Result<Rat, RatError> {
    let neg = s.starts_with('-');
    let s2 = s.trim_start_matches('-').trim_start_matches('+');
    let (int_part, frac_part) = match s2.split_once('.') {
        Some((a, b)) => (a, b),
        None => (s2, ""),
    };
    use num_bigint::BigInt;
    let digits = format!("{int_part}{frac_part}");
    let num: BigInt = if digits.is_empty() {
        BigInt::from(0)
    } else {
        digits.parse().map_err(|_| RatError::Overflow)?
    };
    let den: BigInt = BigInt::from(10).pow(frac_part.len() as u32);
    let n = if neg { -num } else { num };
    Rat::from_bigints(n, den)
}

/// ReLU envelope parameters `(p, q, r, t)` on a pre-activation box `[l, u]`:
/// lower envelope `a >= p*z + q`, upper envelope `a <= r*z + t`. Stable active
/// `a = z`; stable inactive `a = 0`; unstable: lower `a >= alpha*z` (alpha
/// adaptive), upper `a <= s*(z - l)` with `s = u/(u - l)`. Matches crown_deep.
fn relu_env(l: f64, u: f64) -> (f64, f64, f64, f64) {
    if l >= 0.0 {
        (1.0, 0.0, 1.0, 0.0)
    } else if u <= 0.0 {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        let s = u / (u - l);
        let alpha = if u >= -l { 1.0 } else { 0.0 };
        (alpha, 0.0, s, s * (-l))
    }
}

/// Width of the vector feeding layer `li` of an f64 net: the input width `n`
/// for layer 0, else the width of layer `li - 1`.
fn in_width_f64(weights: &[Vec<Vec<f64>>], li: usize, n: usize) -> usize {
    li.checked_sub(1)
        .and_then(|p| weights.get(p))
        .map_or(n, Vec::len)
}

/// Fail closed unless `(weights, biases, lo, hi)` is a well-formed f64 net
/// over an `n = lo.len()`-dimensional box: one bias per unit, every weight row
/// exactly as wide as the layer below, `hi` as wide as `lo`. The f64 passes
/// below walk these shapes with zipped iterators; this check keeps a malformed
/// net a panic (as the index arithmetic it replaces made it) instead of a
/// silently truncated sum. The loaders only ever build well-formed nets.
fn assert_f64_net_shapes(weights: &[Vec<Vec<f64>>], biases: &[Vec<f64>], lo: &[f64], hi: &[f64]) {
    assert_eq!(lo.len(), hi.len(), "f64 net: box lo/hi widths differ");
    assert_eq!(
        weights.len(),
        biases.len(),
        "f64 net: one bias vector per layer"
    );
    let mut prev = lo.len();
    for (li, (w, b)) in weights.iter().zip(biases).enumerate() {
        assert_eq!(b.len(), w.len(), "f64 net layer {li}: bias length != width");
        assert!(
            w.iter().all(|row| row.len() == prev),
            "f64 net layer {li}: a weight row is not {prev} wide"
        );
        prev = w.len();
    }
}

/// Contribution of the input box to a bound with coefficients `a_coeff` on the
/// inputs: each coefficient takes the box end that minimises (`want_lower`) or
/// maximises its term.
fn input_box_term(a_coeff: &[f64], lo: &[f64], hi: &[f64], want_lower: bool, mut acc: f64) -> f64 {
    for ((&d, &l), &h) in a_coeff.iter().zip(lo).zip(hi) {
        if d > 0.0 {
            acc += d * if want_lower { l } else { h };
        } else if d < 0.0 {
            acc += d * if want_lower { h } else { l };
        }
    }
    acc
}

/// CROWN backward bound (f64) of a linear functional over hidden layer `up_to`'s
/// pre-activations: given a coefficient vector `coeff` on z^(up_to) plus a const,
/// substitute back through the ReLU envelopes (using already-computed preact
/// bounds `pre_lo/pre_hi` for layers < up_to) and the affine layers down to the
/// input box, returning (lower, upper) of the functional.  `want_lower` selects
/// direction (each direction uses the matching envelope sign choice).
/// This is the standard "CROWN for intermediate bounds": far tighter than IBP on
/// deep nets, and it stays SOUND because every envelope it uses is a valid
/// over/under-approximation of ReLU on the (valid) preact box.
///
/// Callers validate the net with [`assert_f64_net_shapes`] first.
#[allow(clippy::too_many_arguments)]
fn crown_bound_z(
    weights: &[Vec<Vec<f64>>],
    biases: &[Vec<f64>],
    lo: &[f64],
    hi: &[f64],
    pre_lo: &[Vec<f64>],
    pre_hi: &[Vec<f64>],
    up_to: usize,     // bound z^(up_to), using preact bounds for layers < up_to
    z_coeff0: &[f64], // coefficient on z^(up_to)
    const0: f64,
    want_lower: bool,
) -> f64 {
    let n = lo.len();
    // We want min/max of  const0 + Σ z_coeff0[j] * z^(up_to)_j  over the box.
    // z^(up_to) = W^(up_to) a^(up_to-1) + b^(up_to). Replace z by affine in a^(L-1):
    // accumulate into a_coeff on a^(up_to-1), then walk down through envelopes.
    let li0 = up_to; // 0-indexed layer producing z^(up_to)
    let (Some(w_top), Some(b_top)) = (weights.get(li0), biases.get(li0)) else {
        panic!("crown_bound_z: layer {li0} out of range");
    };
    assert_eq!(
        z_coeff0.len(),
        w_top.len(),
        "crown_bound_z: coefficient width"
    );
    assert!(
        pre_lo.len() >= li0 && pre_hi.len() >= li0,
        "crown_bound_z: preact bounds missing below layer {li0}"
    );
    let mut const_acc = const0;
    // First eliminate z^(li0) through its affine layer to get coeff on a^(li0-1)
    // (or inputs if li0==0).
    let mut a_coeff = zeros_f64(in_width_f64(weights, li0, n));
    for ((&c, row), &bj) in z_coeff0.iter().zip(w_top).zip(b_top) {
        if c == 0.0 {
            continue;
        }
        for (acc, &wji) in a_coeff.iter_mut().zip(row) {
            *acc += c * wji;
        }
        const_acc += c * bj;
    }
    // Now walk layers li from li0-1 down to 0: a^(li) -> envelope -> z^(li) -> affine.
    let below = weights
        .iter()
        .zip(biases)
        .zip(pre_lo.iter().zip(pre_hi))
        .enumerate()
        .take(li0);
    for (li, ((w_li, b_li), (pl, ph))) in below.rev() {
        // a^(li) -> z^(li) via envelope. For lower bound (want_lower) of the whole
        // functional: a term d*a with d>0 uses the LOWER envelope a>=alpha*z; d<0
        // uses the UPPER envelope a<=s(z-l). For upper bound, swap.
        let mut z_coeff = zeros_f64(w_li.len());
        for (((zc, &d), &l), &u) in z_coeff.iter_mut().zip(&a_coeff).zip(pl).zip(ph) {
            let (p, q, r, t) = relu_env(l, u);
            // Choose envelope by direction and sign of d.
            // Lower bound of d*a: if d>0 -> a>=p*z+q -> contributes d*(p z + q);
            //                     if d<0 -> a<=r*z+t -> contributes d*(r z + t).
            // Upper bound of d*a: mirror.
            let use_lower_env = (d > 0.0) == want_lower;
            if d != 0.0 {
                if use_lower_env {
                    *zc += d * p;
                    const_acc += d * q;
                } else {
                    *zc += d * r;
                    const_acc += d * t;
                }
            }
        }
        // z^(li) -> affine over a^(li-1) (or inputs).
        let mut prev_coeff = zeros_f64(in_width_f64(weights, li, n));
        for ((&c, row), &bj) in z_coeff.iter().zip(w_li).zip(b_li) {
            if c == 0.0 {
                continue;
            }
            for (acc, &wji) in prev_coeff.iter_mut().zip(row) {
                *acc += c * wji;
            }
            const_acc += c * bj;
        }
        a_coeff = prev_coeff;
    }
    // a_coeff now on inputs.
    input_box_term(&a_coeff, lo, hi, want_lower, const_acc)
}

/// Compute tight per-layer pre-activation bounds via CROWN (not IBP). For each
/// hidden layer L and unit j, bound z^(L)_j = e_j·z^(L) over the box using
/// `crown_bound_z` with the preact bounds of layers < L already computed. Returns
/// (pre_lo, pre_hi). These bounds are valid (contain the true preact range), so
/// every ReLU envelope built from them is sound.
fn crown_intermediate_bounds_f64(
    weights: &[Vec<Vec<f64>>],
    biases: &[Vec<f64>],
    lo: &[f64],
    hi: &[f64],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    assert_f64_net_shapes(weights, biases, lo, hi);
    let mut pre_lo: Vec<Vec<f64>> = Vec::new();
    let mut pre_hi: Vec<Vec<f64>> = Vec::new();
    // Running IBP activation bounds — each unit's CROWN bound is INTERSECTED
    // with its IBP bound (both valid; IBP wins exactly where the lower envelope
    // loses the `a >= 0` fact, e.g. passthrough units of a flattened DAG).
    // Mirrors the exact intersection in crown_deep::preact_bounds_crown.
    let mut act_lo: Vec<f64> = lo.to_vec();
    let mut act_hi: Vec<f64> = hi.to_vec();
    for (li, (w_li, b_li)) in weights.iter().zip(biases).enumerate() {
        let width = w_li.len();
        let mut zl = zeros_f64(width);
        let mut zu = zeros_f64(width);
        let units = w_li.iter().zip(b_li).zip(zl.iter_mut().zip(zu.iter_mut()));
        for (j, ((row, &bj), (zlj, zuj))) in units.enumerate() {
            let mut e = zeros_f64(width);
            if let Some(ej) = e.get_mut(j) {
                *ej = 1.0; // `j < width`: `j` enumerates this layer's units
            }
            let cl = crown_bound_z(weights, biases, lo, hi, &pre_lo, &pre_hi, li, &e, 0.0, true);
            let cu = crown_bound_z(
                weights, biases, lo, hi, &pre_lo, &pre_hi, li, &e, 0.0, false,
            );
            let mut il = bj;
            let mut iu = bj;
            for ((&wji, &al), &ah) in row.iter().zip(&act_lo).zip(&act_hi) {
                if wji >= 0.0 {
                    il += wji * al;
                    iu += wji * ah;
                } else {
                    il += wji * ah;
                    iu += wji * al;
                }
            }
            *zlj = cl.max(il);
            *zuj = cu.min(iu);
        }
        act_lo = zl.iter().map(|&v| v.max(0.0)).collect();
        act_hi = zu.iter().map(|&v| v.max(0.0)).collect();
        pre_lo.push(zl);
        pre_hi.push(zu);
    }
    (pre_lo, pre_hi)
}

/// TIGHT f64 CROWN lower bound on the output `y'` using CROWN intermediate
/// bounds (matches the exact tight pass in crown_deep when NY_CERT_TIGHT_INTERM
/// is set). The output read-out is the last "layer": y' = out_weight·a^(k)+out_bias.
/// We treat the read-out as a functional on a^(k), substitute through the last
/// ReLU envelope into z^(k), and bound via `crown_bound_z` at up_to=k-1... but
/// since out is on a^(k) (the last hidden activation), we fold it: build coeff on
/// a^(k) then run one envelope step into z^(k)=z^(layer k) and CROWN down.
fn crown_lower_f64_tight(
    weights: &[Vec<Vec<f64>>],
    biases: &[Vec<f64>],
    out_weight: &[f64],
    out_bias: f64,
    lo: &[f64],
    hi: &[f64],
) -> f64 {
    let (pre_lo, pre_hi) = crown_intermediate_bounds_f64(weights, biases, lo, hi);
    // y' = out_bias + Σ out_weight[j] a^(k)_j. a^(k) is ReLU(z^(k)), z^(k)=layer k-1.
    // Fold the read-out (a functional on a^(k)=a^(last hidden)) through the last
    // envelope into z^(k) coeff, then CROWN down from layer (k-1).
    let Some(last) = weights.len().checked_sub(1) else {
        panic!("crown_lower_f64_tight: the network has no hidden layer");
    };
    let (Some(pl), Some(ph)) = (pre_lo.get(last), pre_hi.get(last)) else {
        panic!("crown_lower_f64_tight: no preact bounds for layer {last}");
    };
    assert_eq!(
        out_weight.len(),
        pl.len(),
        "read-out width != last hidden width"
    );
    let mut z_coeff = zeros_f64(pl.len());
    let mut const_acc = out_bias;
    for (((zc, &d), &l), &u) in z_coeff.iter_mut().zip(out_weight).zip(pl).zip(ph) {
        let (p, q, r, t) = relu_env(l, u);
        // Lower bound: d>0 uses lower env a>=p z+q; d<0 uses upper env a<=r z+t.
        if d > 0.0 {
            *zc += d * p;
            const_acc += d * q;
        } else if d < 0.0 {
            *zc += d * r;
            const_acc += d * t;
        }
    }
    crown_bound_z(
        weights, biases, lo, hi, &pre_lo, &pre_hi, last, &z_coeff, const_acc, true,
    )
}

/// A first-layer 2-neuron joint cut: relu(z1_i)+relu(z1_j) <= b, with i<j the
/// first-layer unit indices and b the box-corner max of (relu(z1_i)+relu(z1_j))
/// over the input box. Always sound (it is a valid joint upper bound). Used as an
/// extra >=0-multiplier Farkas premise that can TIGHTEN the leaf bound.
#[derive(Clone, Debug)]
struct ReluCut2 {
    i: usize,
    j: usize,
    b: f64, // f64 view; the exact pass recomputes b in rationals (outward).
}

/// Largest input dimension for which `cut2_box_b_f64` enumerates box corners
/// (2^20 corners per neuron pair).
const MAX_CUT2_CORNER_DIM: usize = 20;

/// Box-corner max of relu(z1_i(x)) + relu(z1_j(x)) over the input box (f64).
/// z1 is affine in the input, so relu(z1) is convex => the sum's max over the
/// box is attained at a corner. We enumerate corners exactly (input_dim small).
///
/// Corner enumeration is `2^n`; past [`MAX_CUT2_CORNER_DIM`] inputs it is not
/// attempted and the bound is `+inf`, which `discover_cuts2` never emits as a
/// cut (a cut is an optional tightening, so omitting it keeps the bound
/// sound). The old unguarded `1u32 << n` panicked for an input dimension of 32
/// or more and ran for 2^n iterations below that.
fn cut2_box_b_f64(w1: &[Vec<f64>], b1: &[f64], lo: &[f64], hi: &[f64], i: usize, j: usize) -> f64 {
    let n = lo.len();
    if n > MAX_CUT2_CORNER_DIM {
        return f64::INFINITY;
    }
    let (Some(&bi), Some(&bj), Some(wi), Some(wj)) = (b1.get(i), b1.get(j), w1.get(i), w1.get(j))
    else {
        panic!(
            "cut2: neuron pair ({i}, {j}) out of range for a first layer of width {}",
            w1.len()
        );
    };
    assert!(
        hi.len() >= n && wi.len() >= n && wj.len() >= n,
        "cut2: box/weight rows narrower than the {n}-dimensional input"
    );
    let mut best = f64::NEG_INFINITY;
    for mask in 0u32..(1u32 << n) {
        let mut zi = bi;
        let mut zj = bj;
        // Bit `d` of `mask` picks hi[d] (set) or lo[d] (clear); `bits` walks
        // the bits from the lowest, one input coordinate per step.
        let mut bits = mask;
        for (((&l, &h), &wid), &wjd) in lo.iter().zip(hi).zip(wi).zip(wj) {
            let x = if bits & 1 != 0 { h } else { l };
            bits >>= 1;
            zi += wid * x;
            zj += wjd * x;
        }
        let v = zi.max(0.0) + zj.max(0.0);
        if v > best {
            best = v;
        }
    }
    best
}

/// Discover candidate first-layer 2-neuron cuts: pairs of UNSTABLE first-layer
/// neurons (l<0<u) whose joint box-corner bound b is strictly tighter than the
/// sum of their per-neuron triangle upper-envelope maxima (= s_i*(u_i-l_i)... at
/// z=u_i gives relu's box max u_i). A cut is only worth emitting if b < u_i + u_j
/// (the trivial per-neuron sum), i.e. coupling actually shrinks the joint range.
fn discover_cuts2(
    w1: &[Vec<f64>],
    b1: &[f64],
    lo: &[f64],
    hi: &[f64],
    pre_lo1: &[f64],
    pre_hi1: &[f64],
    max_cuts: usize,
) -> Vec<ReluCut2> {
    let width = w1.len();
    assert!(
        pre_lo1.len() >= width && pre_hi1.len() >= width,
        "discover_cuts2: preact bounds narrower than the first layer"
    );
    // Unstable first-layer units, with their preact upper bound u.
    let mut unstable: Vec<(usize, f64)> = Vec::new();
    for (j, ((_, &l), &u)) in w1.iter().zip(pre_lo1).zip(pre_hi1).enumerate() {
        if l < 0.0 && u > 0.0 {
            unstable.push((j, u));
        }
    }
    // Score every unstable pair by the slack = (u_i + u_j) - b  (>0 means coupling
    // genuinely tightens the joint relu sum below the independent max). Pairs
    // are visited as (first, each later one), the first walking forward.
    let mut scored: Vec<(f64, usize, usize, f64)> = Vec::new();
    let mut rest = unstable.as_slice();
    while let Some((&(i, ui), later)) = rest.split_first() {
        for &(j, uj) in later {
            let b = cut2_box_b_f64(w1, b1, lo, hi, i, j);
            let indep = ui.max(0.0) + uj.max(0.0);
            let slack = indep - b;
            if slack > 1e-12 {
                scored.push((slack, i, j, b));
            }
        }
        rest = later;
    }
    scored.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    scored.truncate(max_cuts);
    scored
        .into_iter()
        .map(|(_, i, j, b)| ReluCut2 { i, j, b })
        .collect()
}

/// CROWN lower bound on y' using tight CROWN intermediate bounds AND first-layer
/// 2-neuron joint cuts. Backward pass identical to `crown_lower_f64_tight` down
/// to the FIRST hidden layer, where—after the running coefficient on a1 is
/// known—each cut (a_i + a_j <= b) is applied with multiplier μ = min(|c_i|,|c_j|)
/// over the SAME-sign (both contributing via the UPPER envelope) pair, replacing
/// μ units of each per-neuron envelope by the joint μ·b. Returns the (>=) bound m.
#[allow(clippy::too_many_arguments)]
fn crown_lower_f64_tight_cut(
    weights: &[Vec<Vec<f64>>],
    biases: &[Vec<f64>],
    out_weight: &[f64],
    out_bias: f64,
    lo: &[f64],
    hi: &[f64],
    cuts: &[ReluCut2],
) -> f64 {
    let n = lo.len();
    let (pre_lo, pre_hi) = crown_intermediate_bounds_f64(weights, biases, lo, hi);
    assert_eq!(
        out_weight.len(),
        in_width_f64(weights, weights.len(), n),
        "read-out width != last hidden width"
    );
    // Backward pass (same accumulation convention as crown_lower_f64): maintain
    // `-y <= const_acc` form; a_coeff is the running coeff on the current layer's
    // ACTIVATIONS a^(li). Start from the read-out on a^(k).
    let mut const_acc = -out_bias;
    let mut a_coeff = out_weight.to_vec();
    let layers = weights
        .iter()
        .zip(biases)
        .zip(pre_lo.iter().zip(&pre_hi))
        .enumerate();
    for (li, ((w_li, b_li), (pl, ph))) in layers.rev() {
        let width = w_li.len();
        let mut z_coeff = zeros_f64(width);
        // Per-neuron envelope params for this layer, (p,q,r,t) per unit.
        let env: Vec<(f64, f64, f64, f64)> =
            pl.iter().zip(ph).map(|(&l, &u)| relu_env(l, u)).collect();
        // First eliminate via per-neuron envelopes (as in the no-cut pass).
        // We additionally, ON THE FIRST LAYER, divert a μ-share of the negative-
        // coefficient pairs into joint cuts.  In the `-y <= const_acc` convention,
        // a term c*a with c<0 uses the UPPER envelope a <= r z + t and contributes
        // const_acc += |c|*t.  The cut a_i+a_j <= b lets us substitute, for a pair
        // with c_i<0, c_j<0 and μ = min(|c_i|,|c_j|):
        //   |c_i|*t_i + |c_j|*t_j  (envelope consts)  is replaced, for the μ-share,
        //   by μ*b  (joint const) with NO z_coeff contribution from that share.
        // This lowers const_acc (raising the bound -const_acc) exactly when
        //   μ*b  <  μ*(t_i + t_j)  i.e. b < t_i + t_j  — the coupling gain.
        let mut cut_share = zeros_f64(width); // |c_j|-share already taken by cuts
        if li == 0 {
            for cut in cuts {
                let (ci, cj) = (a_coeff[cut.i], a_coeff[cut.j]);
                if ci < 0.0 && cj < 0.0 {
                    let avail_i = (-ci) - cut_share[cut.i];
                    let avail_j = (-cj) - cut_share[cut.j];
                    let mu = avail_i.min(avail_j);
                    if mu <= 0.0 {
                        continue;
                    }
                    let (_, _, _, ti) = env[cut.i];
                    let (_, _, _, tj) = env[cut.j];
                    // Only apply if the joint const beats the per-neuron consts.
                    if cut.b < ti + tj - 1e-15 {
                        // Joint share contributes μ*b to const_acc (and NO z term).
                        const_acc += mu * cut.b;
                        cut_share[cut.i] += mu;
                        cut_share[cut.j] += mu;
                    }
                }
            }
        }
        let units = z_coeff.iter_mut().zip(&a_coeff).zip(&env).zip(&cut_share);
        for (((zc, &c), &(p, q, r, t)), &share) in units {
            if c > 0.0 {
                *zc += c * p;
                const_acc += c * (-q);
            } else if c < 0.0 {
                // The cut already absorbed `share` of |c| (no envelope, no z).
                let mag = (-c) - share;
                if mag > 0.0 {
                    // remaining magnitude uses the per-neuron upper envelope.
                    *zc += (-mag) * r; // c-share is negative => -mag
                    const_acc += mag * t;
                }
            }
        }
        let mut prev_coeff = zeros_f64(in_width_f64(weights, li, n));
        for ((&c, row), &bj) in z_coeff.iter().zip(w_li).zip(b_li) {
            if c == 0.0 {
                continue;
            }
            for (acc, &wji) in prev_coeff.iter_mut().zip(row) {
                *acc += c * wji;
            }
            const_acc += -(c * bj);
        }
        a_coeff = prev_coeff;
    }
    for ((&d, &l), &h) in a_coeff.iter().zip(lo).zip(hi) {
        if d > 0.0 {
            const_acc += -(d * l);
        } else if d < 0.0 {
            const_acc += (-d) * h;
        }
    }
    -const_acc
}

/// Dispatch: tight CROWN intermediate bounds when NYCERT_TIGHT=1, else IBP.
fn crown_lower_dispatch(
    tight: bool,
    weights: &[Vec<Vec<f64>>],
    biases: &[Vec<f64>],
    out_weight: &[f64],
    out_bias: f64,
    lo: &[f64],
    hi: &[f64],
) -> f64 {
    if tight {
        crown_lower_f64_tight(weights, biases, out_weight, out_bias, lo, hi)
    } else {
        crown_lower_f64(weights, biases, out_weight, out_bias, lo, hi)
    }
}

/// Fast f64 mirror of crown_deep::certify's backward pass: returns the CROWN
/// lower bound `m` on the transformed scalar output `y'` using IBP intermediate
/// bounds.  Same
/// algorithm as the exact pass (IBP preact + adaptive-alpha envelopes + backward
/// substitution) but in f64 — microseconds instead of seconds.  Used ONLY to
/// navigate the bisection (decide where/whether to split); the LEAF decision is
/// always re-confirmed by the exact bignum pass.  `safe iff m > u_bound`.
fn crown_lower_f64(
    weights: &[Vec<Vec<f64>>],
    biases: &[Vec<f64>],
    out_weight: &[f64],
    out_bias: f64,
    lo: &[f64],
    hi: &[f64],
) -> f64 {
    let n = lo.len();
    assert_f64_net_shapes(weights, biases, lo, hi);
    assert_eq!(
        out_weight.len(),
        in_width_f64(weights, weights.len(), n),
        "read-out width != last hidden width"
    );
    // IBP pre-activation bounds per hidden layer.
    let mut pre_lo: Vec<Vec<f64>> = Vec::new();
    let mut pre_hi: Vec<Vec<f64>> = Vec::new();
    let mut act_lo = lo.to_vec();
    let mut act_hi = hi.to_vec();
    for (w, b) in weights.iter().zip(biases) {
        let mut zl = Vec::new();
        let mut zu = Vec::new();
        for (row, &bias) in w.iter().zip(b) {
            let mut lmin = bias;
            let mut umax = bias;
            for ((&wji, &al), &ah) in row.iter().zip(&act_lo).zip(&act_hi) {
                if wji >= 0.0 {
                    lmin += wji * al;
                    umax += wji * ah;
                } else {
                    lmin += wji * ah;
                    umax += wji * al;
                }
            }
            zl.push(lmin);
            zu.push(umax);
        }
        act_lo = zl.iter().map(|&v| v.max(0.0)).collect();
        act_hi = zu.iter().map(|&v| v.max(0.0)).collect();
        pre_lo.push(zl);
        pre_hi.push(zu);
    }
    // Backward pass: const_acc with -y <= const_acc  =>  y >= -const_acc.
    let mut const_acc = -out_bias;
    let mut a_coeff = out_weight.to_vec();
    let layers = weights
        .iter()
        .zip(biases)
        .zip(pre_lo.iter().zip(&pre_hi))
        .enumerate();
    for (li, ((w_li, b_li), (pl, ph))) in layers.rev() {
        let mut z_coeff = zeros_f64(w_li.len());
        for (((zc, &c), &l), &u) in z_coeff.iter_mut().zip(&a_coeff).zip(pl).zip(ph) {
            // Envelope params matching crown_deep (see `relu_env`).
            let (p, q, r, t) = relu_env(l, u);
            if c > 0.0 {
                *zc += c * p;
                const_acc += c * (-q);
            } else if c < 0.0 {
                let mag = -c;
                *zc += c * r;
                const_acc += mag * t;
            }
        }
        let mut prev_coeff = zeros_f64(in_width_f64(weights, li, n));
        for ((&c, row), &bj) in z_coeff.iter().zip(w_li).zip(b_li) {
            if c == 0.0 {
                continue;
            }
            for (acc, &wji) in prev_coeff.iter_mut().zip(row) {
                *acc += c * wji;
            }
            const_acc += -(c * bj);
        }
        a_coeff = prev_coeff;
    }
    for ((&d, &l), &h) in a_coeff.iter().zip(lo).zip(hi) {
        if d > 0.0 {
            const_acc += -(d * l);
        } else if d < 0.0 {
            const_acc += (-d) * h;
        }
    }
    -const_acc
}

/// Parse an exact rational token "n/d" or "n" (decimal int) into a Rat.
fn parse_rat_token(s: &str) -> Rat {
    use num_bigint::BigInt;
    if let Some((n, d)) = s.split_once('/') {
        let num: BigInt = n.trim().parse().expect("rat num");
        let den: BigInt = d.trim().parse().expect("rat den");
        Rat::from_bigints(num, den).expect("rat")
    } else {
        // allow a plain decimal too (lossless n/10^k)
        decimal_to_rat(s).expect("decimal rat")
    }
}

/// Whether to outward-round intermediate CROWN bounds (`DeepReluProblem::interm_round`).
/// An EXPLICIT per-invocation `--interm-round` argv flag, not an ambient env read:
/// the certificate content this harness emits is thus a pure function of its
/// arguments, matching the production path (which sets the field directly).
fn interm_round_flag() -> bool {
    std::env::args().any(|a| a == "--interm-round")
}

/// Optional per-leaf alpha override (lower-envelope slopes per hidden layer).
/// NYCERT_ALPHA_FILE points to a JSON file: an array of layers, each an array of
/// rationals "n/d" (one per unit in that hidden layer). Any alpha in [0,1] is
/// SOUND (validated by DeepReluProblem.alpha_for). Returns None if unset.
fn load_alpha_override() -> Option<Vec<Vec<Rat>>> {
    let path = std::env::var("NYCERT_ALPHA_FILE").ok()?;
    let txt = std::fs::read_to_string(&path).expect("read alpha file");
    let v: ny_cert::json::Json = ny_cert::json::from_str(&txt).expect("alpha json");
    let layers = v.as_array().expect("alpha: top array");
    let out: Vec<Vec<Rat>> = layers
        .iter()
        .map(|layer| {
            layer
                .as_array()
                .expect("alpha: layer array")
                .iter()
                .map(|x| parse_rat_token(x.as_str().expect("alpha token str")))
                .collect()
        })
        .collect();
    Some(out)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // ---- Lean-instance emitter mode (self-contained corpus file, mirrors the
    // `lrat_to_lean` precedent): flatten the cersyve con+inv ONNX pair to exact
    // ℚ matrices and emit a `Crownproof/<Module>.lean` `ClampedSystem` instance
    // that applies `safe_forever`. See `emit_lean_instance`.
    if args.get(1).map(String::as_str) == Some("--emit-lean") {
        if args.len() != 7 {
            eprintln!(
                "usage: certify_onnx --emit-lean <con.onnx> <inv.onnx> <prop.vnnlib> <out.lean> <Module>"
            );
            std::process::exit(2);
        }
        emit_lean_instance(&args[2], &args[3], &args[4], &args[5], &args[6]);
        return;
    }
    if args.len() < 4 {
        eprintln!("usage: certify_onnx <model.onnx> <prop.vnnlib> <out_dir> [out_idx]");
        std::process::exit(2);
    }
    let onnx_path = &args[1];
    let vnnlib_path = &args[2];
    let out_dir = &args[3];
    // optional: which output index is the property's subject (default: infer from vnnlib)
    let forced_out: Option<usize> = args.get(4).and_then(|s| s.parse().ok());

    let t0 = Instant::now();
    let g = load_onnx_or_exit(onnx_path);

    // --- VNNLIB (parsed FIRST: the DAG flattener shifts its input passthrough
    // by the ROOT box lower bound, so the box must be known before loading) ---
    let vtext = std::fs::read_to_string(vnnlib_path).expect("read vnnlib");
    let atoms = parse_vnnlib(&vtext);
    // Input dimension = 1 + max X index present (the completeness assert below
    // rejects any gap, so this equals the true arity for every supported prop).
    let vnn_input_dim = vnnlib_input_dim(&atoms).unwrap_or_else(|e| {
        eprintln!("PARSE_FAILED: {e}");
        std::process::exit(2);
    });

    // Input box from X_i atoms.
    let mut lo = vec![Rat::ZERO; vnn_input_dim];
    let mut hi = vec![Rat::ZERO; vnn_input_dim];
    let mut lo_set = vec![false; vnn_input_dim];
    let mut hi_set = vec![false; vnn_input_dim];
    // Output property atoms (over Y_j).
    let mut y_atoms: Vec<VnnAtom> = vec![];
    for a in &atoms {
        if a.var.starts_with("X_") {
            // `parse_vnnlib` only emits atoms with a parseable index, so `None` is
            // unreachable here; an out-of-range index is dropped (the box-incomplete
            // assert below then reports it clearly). Both paths are fail-closed.
            let Some(idx) = var_index(&a.var) else {
                continue;
            };
            if idx >= vnn_input_dim {
                continue;
            }
            // parse the raw decimal token losslessly (decimals are n/10^k).
            // The vnnlib gate admits any f64-parseable token, but decimal_to_rat
            // only handles plain decimals — scientific notation (e.g. `1e10`) must
            // fail closed (exit), never panic on the verdict path.
            let r = match decimal_to_rat(&a.rhs_raw) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!(
                        "PARSE_FAILED: unsupported X threshold {:?}: {:?}",
                        a.rhs_raw, e
                    );
                    std::process::exit(2);
                }
            };
            if a.op == ">=" {
                lo[idx] = r;
                lo_set[idx] = true;
            } else {
                hi[idx] = r;
                hi_set[idx] = true;
            }
        } else {
            y_atoms.push(a.clone());
        }
    }
    for i in 0..vnn_input_dim {
        assert!(lo_set[i] && hi_set[i], "input X_{i} box incomplete");
    }

    // --- Load the network. Gemm graphs (the cersyve DAG shapes) go through the
    // DAG-aware exact loader + flattener; everything else through the original
    // sequential loader. Both fail closed outside their verified fragments. ---
    let is_dag = g.nodes.iter().any(|n| n.op_type == "Gemm");
    let dag = if is_dag { Some(load_dag(&g)) } else { None };
    let net = match &dag {
        Some(d) => {
            assert_eq!(
                d.input_dim, vnn_input_dim,
                "ONNX input dim != vnnlib box dim"
            );
            flatten_dag(d, &lo)
        }
        None => load_net(&g),
    };
    assert_eq!(
        net.input_dim, vnn_input_dim,
        "loaded input dim != vnnlib box dim"
    );
    let load_us = t0.elapsed().as_micros();

    let n_hidden = net.hidden.len();
    let widths: Vec<usize> = net.layers().map(|l| l.w.len()).collect();
    eprintln!(
        "[load] input_dim={} hidden_layers={} layer_widths={:?} loader={} ({} us)",
        net.input_dim,
        n_hidden,
        widths,
        if is_dag { "dag-flatten" } else { "sequential" },
        load_us
    );

    // --- EXACT FORWARD PARITY GATE (NYCERT_PARITY=1) -----------------------
    // For DAG-loaded nets: at pseudo-random f32-exact points inside the box,
    // (a) the flattened sequential net must equal the symbolic DAG forward
    //     EXACTLY (rational equality — the flattening algebra is exact), and
    // (b) the exact forward must match an independent f32 interpretation of
    //     the original ONNX node list within f32 rounding.
    if std::env::var("NYCERT_PARITY").ok().as_deref() == Some("1") {
        let Some(d) = &dag else {
            eprintln!("PARITY_UNSUPPORTED: sequential loader path");
            std::process::exit(2);
        };
        run_parity(&g, d, &net, &lo, &hi);
        return;
    }

    // --- Optional per-LEAF box override (float-fast tree / exact-slow leaf) ---
    // NYCERT_BOX_LO / NYCERT_BOX_HI : comma-separated EXACT rationals "n/d" or "n"
    // (one per input dim). When provided they REPLACE the vnnlib box. The driver
    // MUST guarantee the override lies inside the vnnlib box and that the union of
    // all leaf overrides covers it (covering is checked separately/in Lean).
    if let Ok(s) = std::env::var("NYCERT_BOX_LO") {
        let parts: Vec<&str> = s.split(',').collect();
        assert_eq!(parts.len(), net.input_dim, "NYCERT_BOX_LO arity");
        for (i, p) in parts.iter().enumerate() {
            lo[i] = parse_rat_token(p.trim());
        }
    }
    if let Ok(s) = std::env::var("NYCERT_BOX_HI") {
        let parts: Vec<&str> = s.split(',').collect();
        assert_eq!(parts.len(), net.input_dim, "NYCERT_BOX_HI arity");
        for (i, p) in parts.iter().enumerate() {
            hi[i] = parse_rat_token(p.trim());
        }
    }

    // --- CONJUNCTIVE UNSAFE REGION (NYCERT_CONJ=1) --------------------------
    // The cersyve properties assert the unsafe region  Y_0 <= 0  AND  Y_1 >= 0.
    // Refuting the conjunction on a box needs only ONE conjunct refuted there,
    // so the driver runs a complete exact branch-and-bound: f64 CROWN screens
    // navigate, every leaf verdict is an exact-rational certificate refuting
    // {box ∧ network relaxation ∧ one unsafe atom}, self-checked by the in-tree
    // mirror of Clean's verifier and emitted as Clean external-cert JSON.
    if std::env::var("NYCERT_CONJ").ok().as_deref() == Some("1") {
        conj_pipeline(&net, &lo, &hi, &y_atoms, out_dir, load_us);
    }

    // --- Optional TRUNCATION to the first K hidden layers (real ACAS weights) ---
    // NYCERT_TRUNC=K : keep only hidden layers 0..K (real W,b), and synthesize the
    // linear read-out  y' = Σ_j ow_j · a^(K)_j  over the K-th ReLU activations,
    // where ow is read from NYCERT_TRUNC_OW (comma rationals, len = width(K)) or
    // defaults to a fixed sign pattern. This yields a genuine real-ACAS SUB-network
    // (5 -> width^K -> 1) on which the first-layer relaxation is the dominant one,
    // making the first-layer multi-neuron cut a measurable decision lever. The
    // VNNLIB Y atom's threshold is reinterpreted as the bound on this y'.
    let trunc_readout: AffineLayer;
    let (hidden_layers, readout): (&[AffineLayer], &AffineLayer) =
        if let Ok(ks) = std::env::var("NYCERT_TRUNC") {
            let kk: usize = ks.parse().expect("NYCERT_TRUNC int");
            assert!(kk >= 1 && kk <= n_hidden, "NYCERT_TRUNC out of range");
            let width_k = net.hidden[kk - 1].w.len();
            // read-out weights on a^(K)
            let ow: Vec<Rat> = if let Ok(s) = std::env::var("NYCERT_TRUNC_OW") {
                let v: Vec<Rat> = s.split(',').map(|t| parse_rat_token(t.trim())).collect();
                assert_eq!(v.len(), width_k, "NYCERT_TRUNC_OW arity");
                v
            } else {
                // default: alternating +1/-1 read-out (a real, nontrivial functional).
                (0..width_k)
                    .map(|j| if j % 2 == 0 { Rat::ONE } else { Rat::ONE.neg() })
                    .collect()
            };
            trunc_readout = AffineLayer {
                w: vec![ow],
                b: vec![Rat::ZERO],
            };
            (&net.hidden[..kk], &trunc_readout)
        } else {
            (&net.hidden, &net.readout)
        };

    // Build DeepReluProblem with hidden layers and read-out.
    let hidden_w: Vec<Vec<Vec<Rat>>> = hidden_layers.iter().map(|l| l.w.clone()).collect();
    let hidden_b: Vec<Vec<Rat>> = hidden_layers.iter().map(|l| l.b.clone()).collect();
    let out_dim = readout.w.len();

    // Decide the output property. Each VNNLIB Y atom is part of the UNSAFE region.
    //   (>= Y_j c)  unsafe  -> safe iff Y_j < c       -> certify Y_j <= c (upper bound)
    //   (<= Y_j c)  unsafe  -> safe iff Y_j > c       -> certify Y_j >= c (lower bound)
    // We process the first Y atom (these single-output props have exactly one),
    // or the one matching forced_out.
    let prop = if let Some(oi) = forced_out {
        y_atoms
            .iter()
            .find(|a| var_index(&a.var) == Some(oi))
            .expect("forced out not in props")
            .clone()
    } else {
        y_atoms.first().expect("no Y property atom").clone()
    };
    let j = match var_index(&prop.var) {
        Some(j) if j < out_dim => j,
        other => {
            eprintln!(
                "malformed or out-of-range Y variable {:?} (index {:?}, out_dim {})",
                prop.var, other, out_dim
            );
            std::process::exit(1);
        }
    };
    let c = match decimal_to_rat(&prop.rhs_raw) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "PARSE_FAILED: unsupported Y threshold {:?}: {:?}",
                prop.rhs_raw, e
            );
            std::process::exit(2);
        }
    };

    // Read-out for output j: y = readout.w[j] . a_last + readout.b[j].
    let (out_weight, out_bias, want_upper) = if prop.op == ">=" {
        // certify Y_j <= c  <=>  certify (-Y_j) >= -c.  Negate read-out.
        let w: Vec<Rat> = readout.w[j].iter().map(|r| r.neg()).collect();
        (w, readout.b[j].neg(), true)
    } else {
        // certify Y_j >= c.
        (readout.w[j].clone(), readout.b[j], false)
    };
    // The transformed scalar output is  y' = Y_j  (lower prop)  or  y' = -Y_j (upper).
    // The unsafe region, expressed in y', is always  y' <= u_bound :
    //   lower prop  (<= Y_j c) :  unsafe Y_j <= c   => y' <= c
    //   upper prop  (>= Y_j c) :  unsafe Y_j >= c   => -y' >= c  => y' <= -c
    let mut u_bound = if want_upper { c.neg() } else { c };
    // NYCERT_UBOUND override: set the decision threshold directly (certify y' >=
    // -u_bound i.e. refute unsafe y' <= u_bound). Lets the experiment place the
    // threshold inside the cut-sensitive band (m_base, m_cut].
    if let Ok(s) = std::env::var("NYCERT_UBOUND") {
        u_bound = parse_rat_token(s.trim());
    }

    // --- FAST f64 SCREEN (navigation only) ---------------------------------
    // When NYCERT_F64_SCREEN=1, compute the CROWN bound in f64 (microseconds)
    // and print a SCREEN line, then EXIT without the slow exact pass. The driver
    // uses this to navigate the bisection. The eventual leaf is ALWAYS re-run
    // through the exact bignum pass + Clean kernel, so soundness never depends
    // on the f64 screen.
    if std::env::var("NYCERT_F64_SCREEN").ok().as_deref() == Some("1") {
        let f = |r: &Rat| rat_to_f64(r);
        let w64: Vec<Vec<Vec<f64>>> = hidden_w
            .iter()
            .map(|l| l.iter().map(|row| row.iter().map(&f).collect()).collect())
            .collect();
        let b64: Vec<Vec<f64>> = hidden_b
            .iter()
            .map(|l| l.iter().map(&f).collect())
            .collect();
        let ow64: Vec<f64> = out_weight.iter().map(&f).collect();
        let ob64 = f(&out_bias);
        let lo64: Vec<f64> = lo.iter().map(&f).collect();
        let hi64: Vec<f64> = hi.iter().map(&f).collect();
        let tight = std::env::var("NYCERT_TIGHT").ok().as_deref() == Some("1");
        let m64 = crown_lower_dispatch(tight, &w64, &b64, &ow64, ob64, &lo64, &hi64);
        let ub64 = f(&u_bound);
        let margin = m64 - ub64;
        println!(
            "SCREEN m={m64:.9} u_bound={ub64:.9} margin={margin:.9} decided={} tight={tight}",
            margin > 0.0
        );
        return;
    }

    // --- CUT MEASUREMENT (NYCERT_CUTMEASURE=1) -----------------------------
    // Compute, for the current box: the BASELINE tight CROWN bound (no cut), and
    // the cut-augmented tight CROWN bound (first-layer 2-neuron joint cuts), then
    // report both margins + the discovered cuts. Pure measurement: no cert emitted.
    if std::env::var("NYCERT_CUTMEASURE").ok().as_deref() == Some("1") {
        let f = |r: &Rat| rat_to_f64(r);
        let w64: Vec<Vec<Vec<f64>>> = hidden_w
            .iter()
            .map(|l| l.iter().map(|row| row.iter().map(&f).collect()).collect())
            .collect();
        let b64: Vec<Vec<f64>> = hidden_b
            .iter()
            .map(|l| l.iter().map(&f).collect())
            .collect();
        let ow64: Vec<f64> = out_weight.iter().map(&f).collect();
        let ob64 = f(&out_bias);
        let lo64: Vec<f64> = lo.iter().map(&f).collect();
        let hi64: Vec<f64> = hi.iter().map(&f).collect();
        let ub64 = f(&u_bound);
        let max_cuts: usize = std::env::var("NYCERT_MAXCUTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
        let (pre_lo, pre_hi) = crown_intermediate_bounds_f64(&w64, &b64, &lo64, &hi64);
        let cuts = discover_cuts2(
            &w64[0], &b64[0], &lo64, &hi64, &pre_lo[0], &pre_hi[0], max_cuts,
        );
        let m_base = crown_lower_f64_tight(&w64, &b64, &ow64, ob64, &lo64, &hi64);
        // A cut is an OPTIONAL valid premise: the with-cut bound is sound, but the
        // GREEDY diversion can drop a backward z-term, so we take the MAX (both are
        // valid lower bounds; max stays sound and is monotone in the cut's help).
        let m_cut_raw = crown_lower_f64_tight_cut(&w64, &b64, &ow64, ob64, &lo64, &hi64, &cuts);
        let m_cut = m_cut_raw.max(m_base);
        println!(
            "CUTMEASURE m_base={m_base:.9} m_cut={m_cut:.9} u_bound={ub64:.9} \
             margin_base={:.9} margin_cut={:.9} decided_base={} decided_cut={} \
             n_cuts={} delta_m={:.9}",
            m_base - ub64,
            m_cut - ub64,
            m_base > ub64,
            m_cut > ub64,
            cuts.len(),
            m_cut - m_base
        );
        return;
    }

    // --- BATCH f64 SCREEN: load ONNX once, screen many boxes from stdin -------
    // When NYCERT_BATCH_SCREEN=1, read lines "id lo0,lo1,...|hi0,hi1,..." (exact
    // rationals) from stdin and print "RES id margin decided" per line. The ONNX
    // is parsed once, so a deep bisection costs no per-box reload/spawn overhead.
    if std::env::var("NYCERT_BATCH_SCREEN").ok().as_deref() == Some("1") {
        use std::io::{BufRead, Write};
        let f = |r: &Rat| rat_to_f64(r);
        let w64: Vec<Vec<Vec<f64>>> = hidden_w
            .iter()
            .map(|l| l.iter().map(|row| row.iter().map(&f).collect()).collect())
            .collect();
        let b64: Vec<Vec<f64>> = hidden_b
            .iter()
            .map(|l| l.iter().map(&f).collect())
            .collect();
        let ow64: Vec<f64> = out_weight.iter().map(&f).collect();
        let ob64 = f(&out_bias);
        let ub64 = f(&u_bound);
        let n = net.input_dim;
        let tight = std::env::var("NYCERT_TIGHT").ok().as_deref() == Some("1");
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        let mut out = std::io::BufWriter::new(stdout.lock());
        for line in stdin.lock().lines() {
            let line = line.unwrap();
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if line == "QUIT" {
                break;
            }
            // "id box" where box = "lo0,..,loN|hi0,..,hiN"
            let (id, boxstr) = line.split_once(' ').expect("batch line: id box");
            let (los, his) = boxstr.split_once('|').expect("batch box: lo|hi");
            let lo64: Vec<f64> = los
                .split(',')
                .map(|t| f(&parse_rat_token(t.trim())))
                .collect();
            let hi64: Vec<f64> = his
                .split(',')
                .map(|t| f(&parse_rat_token(t.trim())))
                .collect();
            assert_eq!(lo64.len(), n);
            assert_eq!(hi64.len(), n);
            let m64 = crown_lower_dispatch(tight, &w64, &b64, &ow64, ob64, &lo64, &hi64);
            let margin = m64 - ub64;
            writeln!(out, "RES {id} {margin:.9} {}", margin > 0.0).unwrap();
            out.flush().unwrap();
        }
        return;
    }

    // --- IN-RUST f64 BISECTION (tree discovery) ------------------------------
    // When NYCERT_BISECT=<out.json>, run the WHOLE widest-coordinate midpoint
    // bisection internally (no per-node IPC): a box whose f64 CROWN bound refutes
    // the unsafe atom is a LEAF; otherwise bisect its widest coord at the exact
    // rational midpoint into closed half-boxes {x<=m},{x>=m} and recurse. Writes
    // the tree (leaf boxes + split structure, exact rationals) as JSON in the same
    // schema the Python driver used. Leaves are re-certified exactly in phase 2.
    // Env: NYCERT_BISECT_MAXDEPTH (default 40).
    if let Ok(treepath) = std::env::var("NYCERT_BISECT") {
        let f = |r: &Rat| rat_to_f64(r);
        let w64: Vec<Vec<Vec<f64>>> = hidden_w
            .iter()
            .map(|l| l.iter().map(|row| row.iter().map(&f).collect()).collect())
            .collect();
        let b64: Vec<Vec<f64>> = hidden_b
            .iter()
            .map(|l| l.iter().map(&f).collect())
            .collect();
        let ow64: Vec<f64> = out_weight.iter().map(&f).collect();
        let ob64 = f(&out_bias);
        let ub64 = f(&u_bound);
        let n = net.input_dim;
        let tight = std::env::var("NYCERT_TIGHT").ok().as_deref() == Some("1");
        // NYCERT_CUT=1: use first-layer 2-neuron joint cuts in the leaf-bound
        // decision (tighter bound => closes leaves a triangle bound cannot).
        let use_cut = std::env::var("NYCERT_CUT").ok().as_deref() == Some("1");
        let max_cuts: usize = std::env::var("NYCERT_MAXCUTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
        let max_depth: usize = std::env::var("NYCERT_BISECT_MAXDEPTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(40);
        // Node record kinds, serialized lazily as JSON strings.
        // We build the tree with an explicit stack. Each stack item carries the
        // exact Rat box and the path (list of (coord, mid_rat, side)).
        struct Item {
            lo: Vec<Rat>,
            hi: Vec<Rat>,
            path: Vec<(usize, Rat, bool)>,
            nid: usize,
        }
        let half = Rat::new(1, 2).unwrap();
        let mut stack: Vec<Item> = vec![Item {
            lo: lo.clone(),
            hi: hi.clone(),
            path: vec![],
            nid: 0,
        }];
        let mut next_node = 1usize;
        let mut node_json: Vec<String> = Vec::new(); // each "\"<id>\": {...}"
        let mut n_leaves = 0usize;
        let mut n_failed = 0usize;
        let mut explored = 0usize;
        let mut max_depth_used = 0usize;
        let rat_str = |r: &Rat| -> String {
            r.to_clean_string()
                .expect("refusing to serialize a poisoned rational arena")
        };
        let path_json = |path: &[(usize, Rat, bool)]| -> String {
            let parts: Vec<String> = path
                .iter()
                .map(|(c, m, le)| {
                    format!(
                        "[{},\"{}\",\"{}\"]",
                        c,
                        rat_str(m),
                        if *le { "le" } else { "ge" }
                    )
                })
                .collect();
            format!("[{}]", parts.join(","))
        };
        let box_json = |v: &[Rat]| -> String {
            let parts: Vec<String> = v.iter().map(|r| format!("\"{}\"", rat_str(r))).collect();
            format!("[{}]", parts.join(","))
        };
        while let Some(it) = stack.pop() {
            explored += 1;
            let depth = it.path.len();
            let lo64: Vec<f64> = it.lo.iter().map(&f).collect();
            let hi64: Vec<f64> = it.hi.iter().map(&f).collect();
            let m64 = if use_cut && tight {
                let m_base = crown_lower_f64_tight(&w64, &b64, &ow64, ob64, &lo64, &hi64);
                let (pl, ph) = crown_intermediate_bounds_f64(&w64, &b64, &lo64, &hi64);
                let cuts = discover_cuts2(&w64[0], &b64[0], &lo64, &hi64, &pl[0], &ph[0], max_cuts);
                let m_cut = crown_lower_f64_tight_cut(&w64, &b64, &ow64, ob64, &lo64, &hi64, &cuts);
                m_base.max(m_cut) // cut is an OPTIONAL valid premise -> max stays sound
            } else {
                crown_lower_dispatch(tight, &w64, &b64, &ow64, ob64, &lo64, &hi64)
            };
            let margin = m64 - ub64;
            if margin > 0.0 {
                n_leaves += 1;
                max_depth_used = max_depth_used.max(depth);
                node_json.push(format!(
                    "\"{}\":{{\"kind\":\"leaf\",\"id\":{},\"lo\":{},\"hi\":{},\"path\":{},\"screen_margin\":{:.9}}}",
                    it.nid, it.nid, box_json(&it.lo), box_json(&it.hi), path_json(&it.path), margin));
            } else if depth >= max_depth {
                n_failed += 1;
                node_json.push(format!(
                    "\"{}\":{{\"kind\":\"FAILED_DEPTH\",\"id\":{},\"lo\":{},\"hi\":{},\"path\":{},\"screen_margin\":{:.9}}}",
                    it.nid, it.nid, box_json(&it.lo), box_json(&it.hi), path_json(&it.path), margin));
            } else {
                // widest coordinate
                let mut c = 0usize;
                let mut best = Rat::ZERO;
                for i in 0..n {
                    let w = it.hi[i].sub(it.lo[i]).unwrap();
                    if i == 0 || w > best {
                        best = w;
                        c = i;
                    }
                }
                let mid = it.lo[c].add(it.hi[c]).unwrap().mul(half).unwrap();
                let lid = next_node;
                let rid = next_node + 1;
                next_node += 2;
                node_json.push(format!(
                    "\"{}\":{{\"kind\":\"split\",\"id\":{},\"coord\":{},\"mid\":\"{}\",\"lo_child\":{},\"hi_child\":{}}}",
                    it.nid, it.nid, c, rat_str(&mid), lid, rid));
                // lo child: x_c <= mid  -> hi[c]=mid
                let mut lo_hi = it.hi.clone();
                lo_hi[c] = mid;
                let mut lpath = it.path.clone();
                lpath.push((c, mid, true));
                stack.push(Item {
                    lo: it.lo.clone(),
                    hi: lo_hi,
                    path: lpath,
                    nid: lid,
                });
                // hi child: x_c >= mid  -> lo[c]=mid
                let mut hi_lo = it.lo.clone();
                hi_lo[c] = mid;
                let mut rpath = it.path.clone();
                rpath.push((c, mid, false));
                stack.push(Item {
                    lo: hi_lo,
                    hi: it.hi,
                    path: rpath,
                    nid: rid,
                });
            }
        }
        let root_lo_j = box_json(&lo);
        let root_hi_j = box_json(&hi);
        let json = format!(
            "{{\"root_lo\":{},\"root_hi\":{},\"leaves\":{},\"failed_leaves\":{},\"explored_nodes\":{},\"max_depth_used\":{},\"nodes\":{{{}}}}}",
            root_lo_j, root_hi_j, n_leaves, n_failed, explored, max_depth_used, node_json.join(","));
        assert!(
            !ny_cert::rational::poisoned(),
            "refusing to write a bisection tree after a rational arena fallback"
        );
        std::fs::write(&treepath, json).expect("write tree json");
        println!("BISECT leaves={n_leaves} failed={n_failed} explored={explored} max_depth={max_depth_used} out={treepath}");
        return;
    }

    let problem = DeepReluProblem {
        weights: hidden_w,
        biases: hidden_b,
        out_weight,
        out_bias,
        input_lower: lo.clone(),
        input_upper: hi.clone(),
        alpha: load_alpha_override(),
        interm_round: interm_round_flag(),
    };

    // certify() with any threshold <= m derives the same multipliers proving
    // y' >= m (m = CROWN lower bound). Use a very low threshold to recover m.
    let sentinel = Rat::from_int(-1_000_000_000);
    let t1 = Instant::now();
    // NYCERT_TIGHT=1 selects exact CROWN intermediate bounds (tight, sound) — the
    // same tightening the f64 tree-discovery screen uses, so the discovered leaves
    // close under the exact pass too.
    let tight_interm = std::env::var("NYCERT_TIGHT").ok().as_deref() == Some("1");
    // NYCERT_CUT=1: discover first-layer 2-neuron cut pairs (in f64) and emit them
    // as exact joint-cut Farkas premises (the verified multiReluCut lever). The
    // EXACT pass recomputes each B_ij from the leaf box corners in rationals, so
    // Clean checks the cut premise alongside the others.
    let use_cut = std::env::var("NYCERT_CUT").ok().as_deref() == Some("1");
    let certified = if use_cut && tight_interm {
        let f = |r: &Rat| rat_to_f64(r);
        let w64: Vec<Vec<Vec<f64>>> = problem
            .weights
            .iter()
            .map(|l| l.iter().map(|row| row.iter().map(&f).collect()).collect())
            .collect();
        let b64: Vec<Vec<f64>> = problem
            .biases
            .iter()
            .map(|l| l.iter().map(&f).collect())
            .collect();
        let lo64: Vec<f64> = lo.iter().map(&f).collect();
        let hi64: Vec<f64> = hi.iter().map(&f).collect();
        let max_cuts: usize = std::env::var("NYCERT_MAXCUTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
        let (pl, ph) = crown_intermediate_bounds_f64(&w64, &b64, &lo64, &hi64);
        let cuts2 = discover_cuts2(&w64[0], &b64[0], &lo64, &hi64, &pl[0], &ph[0], max_cuts);
        let pairs: Vec<(usize, usize)> = cuts2.iter().map(|c| (c.i, c.j)).collect();
        eprintln!(
            "[cut] discovered {} first-layer 2-neuron cut pairs",
            pairs.len()
        );
        match problem.certify_with_interm_cuts(sentinel, tight_interm, &pairs) {
            Ok(c) => c,
            Err(e) => {
                println!("CERTIFY_FAILED out={} dim={} : {e}", j, out_dim);
                std::process::exit(3);
            }
        }
    } else {
        match problem.certify_with_interm(sentinel, tight_interm) {
            Ok(c) => c,
            Err(e) => {
                println!("CERTIFY_FAILED out={} dim={} : {e}", j, out_dim);
                std::process::exit(3);
            }
        }
    };
    let certify_us = t1.elapsed().as_micros();

    let m = certified.lower_bound; // y' >= m, exact
    let m_f = rat_to_f64(&m);

    // SAFETY decision: the unsafe region y' <= u_bound is empty iff m > u_bound.
    let safe = m > u_bound;
    if want_upper {
        let ub = m.neg(); // Y_j <= -m
        eprintln!(
            "[crown] exact CROWN: Y_{j} <= {} (~{:.6}) ; unsafe region was Y_{j} >= {} (~{:.6}) -> {}",
            fmt_rat(&ub), -m_f, fmt_rat(&c), rat_to_f64(&c),
            if safe { "SAFE (refuted)" } else { "NOT proven" }
        );
    } else {
        eprintln!(
            "[crown] exact CROWN: Y_{j} >= {} (~{:.6}) ; unsafe region was Y_{j} <= {} (~{:.6}) -> {}",
            fmt_rat(&m), m_f, fmt_rat(&c), rat_to_f64(&c),
            if safe { "SAFE (refuted)" } else { "NOT proven" }
        );
    }
    eprintln!(
        "[crown] CROWN exact pass: {certify_us} us; premises={}",
        certified.entailment.premises.len()
    );

    // ---- Exact soundness cross-check: the certified bound m must not exceed
    // the TRUE transformed output y'(x) at any box corner (2^dim) or grid point.
    {
        let dim = lo.len();
        let mut worst: Option<Rat> = None;
        // All 2^dim corners (exact).
        if dim <= 16 {
            for mask in 0u32..(1u32 << dim) {
                let mut x = Vec::with_capacity(dim);
                for i in 0..dim {
                    x.push(if mask & (1 << i) != 0 { hi[i] } else { lo[i] });
                }
                let y = problem.eval(&x).expect("eval corner");
                assert!(m <= y, "UNSOUND: corner y'={y:?} < certified bound {m:?}");
                if worst.map_or(true, |w| y < w) {
                    worst = Some(y);
                }
            }
        }
        // A small interior grid (exact midpoints) as extra evidence.
        let mut x = Vec::with_capacity(dim);
        for i in 0..dim {
            x.push(
                lo[i]
                    .add(hi[i])
                    .unwrap()
                    .mul(Rat::new(1, 2).unwrap())
                    .unwrap(),
            );
        }
        let ymid = problem.eval(&x).expect("eval mid");
        assert!(m <= ymid, "UNSOUND at midpoint");
        if let Some(w) = worst {
            eprintln!("[check] exact eval over {} box corners + midpoint: min true y'={} >= bound {}  (SOUND)",
                1u64 << dim, fmt_rat(&w), fmt_rat(&m));
        }
    }

    // ---- Build the proof-carrying ENTAILMENT certificate (always emittable) ----
    // The CROWN backward pass yields non-negative multipliers under which the
    // network-relaxation premises + input box entail the exact scalar bound
    //   y' >= m   (m = certified.lower_bound, an exact bignum rational).
    // This is an UNCONDITIONAL, kernel-checkable theorem about the REAL network,
    // independent of whether m happens to be tight enough to decide the property.
    // (the cert's scalar variable is named "y" and equals our y'.)
    let entailment = EntailmentCertificate {
        premises: certified.entailment.premises.clone(),
        multipliers: certified.entailment.multipliers.clone(),
        conclusion: LinearConstraint::with_kind(ConstraintKind::Ge, &[("y", Rat::ONE)], m),
    };

    // Report the certified output bound in the ORIGINAL Y_j coordinates.
    //   lower prop : y' = Y_j        => certified  Y_j >= m
    //   upper prop : y' = -Y_j       => certified  Y_j <= -m
    let (bound_dir, bound_val) = if want_upper {
        ("<=", m.neg())
    } else {
        (">=", m)
    };
    eprintln!(
        "[bound] EXACT certified output bound: Y_{j} {bound_dir} {} (~{:.6})",
        fmt_rat(&bound_val),
        rat_to_f64(&bound_val)
    );

    // Emit JSON. (No i64 guard anymore: full bignum n/d strings are written.)
    std::fs::create_dir_all(out_dir).unwrap();
    let emit_ent = || -> Result<(), RatError> {
        let ent = entailment_to_json(&entailment)?;
        std::fs::write(
            format!("{out_dir}/entailment.json"),
            ny_cert::json::to_string(&ent),
        )
        .unwrap();
        Ok(())
    };
    match emit_ent() {
        Ok(()) => {
            eprintln!(
                "[emit] wrote {out_dir}/entailment.json (full bignum rationals; no i64 guard)"
            );
        }
        Err(e) => {
            println!("EMIT_FAILED: {e}");
            std::process::exit(4);
        }
    }

    // ---- Build the FARKAS refutation only when the property is actually decided.
    //   premises (deriving -y <= -m) PLUS the real unsafe atom y <= u_bound
    //   (Le, multiplier 1). Sum: 0 <= u_bound - m < 0 (since m > u_bound) =>
    //   non-strict contradiction, refuting  box ∧ network ∧ unsafe.
    if safe {
        let mut f_constraints = certified.entailment.premises.clone();
        let mut f_mult = certified.entailment.multipliers;
        f_constraints.push(LinearConstraint::with_kind(
            ConstraintKind::Le,
            &[("y", Rat::ONE)],
            u_bound,
        ));
        f_mult.push(Rat::ONE);
        let farkas = FarkasCertificate {
            constraints: f_constraints,
            multipliers: f_mult,
        };
        let emit_far = || -> Result<(), RatError> {
            let far = farkas_to_json(&farkas)?;
            std::fs::write(
                format!("{out_dir}/farkas.json"),
                ny_cert::json::to_string(&far),
            )
            .unwrap();
            Ok(())
        };
        match emit_far() {
            Ok(()) => {
                eprintln!("[emit] wrote {out_dir}/farkas.json (property DECIDED by exact CROWN)")
            }
            Err(e) => {
                println!("EMIT_FAILED (farkas): {e}");
                std::process::exit(4);
            }
        }
    } else {
        eprintln!(
            "[emit] property NOT decided by single-pass CROWN (bound loose); no farkas emitted"
        );
    }

    // Max numerator / denominator bit-width actually present in the emitted
    // entailment certificate — the quantitative measure of how far past the
    // i128 (127-bit) wall this real network's exact certificate reaches.
    let max_bits = cert_max_bits(&entailment);
    // Machine-readable margin = m - u_bound (the slack by which the unsafe atom is
    // refuted on this box). margin > 0 iff DECIDED. Printed as exact rational and
    // f64 so the BaB driver can decide whether to accept the leaf or split it.
    let margin = m.sub(u_bound).unwrap();
    println!(
        "MARGIN exact={} approx={:.9} decided={}",
        fmt_rat(&margin),
        rat_to_f64(&margin),
        safe
    );
    println!(
        "OK property={} certify={certify_us}us load={load_us}us premises={} max_cert_bits={}",
        if safe { "DECIDED" } else { "bound-only" },
        entailment.premises.len(),
        max_bits,
    );
}

/// EXACT FORWARD PARITY GATE. Points are f32-exact (sampled in f64, rounded to
/// f32, then losslessly lifted to rationals), so the exact and f32 paths
/// evaluate the SAME mathematical input.
fn run_parity(g: &OnnxGraph, dag: &DagNet, flat: &LoadedNet, lo: &[Rat], hi: &[Rat]) {
    let n = dag.input_dim;
    let lo64: Vec<f64> = lo.iter().map(rat_to_f64).collect();
    let hi64: Vec<f64> = hi.iter().map(rat_to_f64).collect();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let npoints: usize = std::env::var("NYCERT_PARITY_POINTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    let mut max_abs = 0f64;
    let mut max_rel = 0f64;
    let mut checked = 0usize;
    for p in 0..=npoints {
        let mut xf: Vec<f32> = Vec::with_capacity(n);
        for i in 0..n {
            // p == 0: box midpoint; else pseudo-random interior grid point.
            let k = if p == 0 { 2048 } else { next() % 4095 + 1 };
            let t = k as f64 / 4096.0;
            xf.push((lo64[i] + (hi64[i] - lo64[i]) * t) as f32);
        }
        let xr: Vec<Rat> = xf
            .iter()
            .map(|&v| f32_to_rat(v).expect("parity point"))
            .collect();
        // Skip points whose f32 rounding escaped the box (possible near bounds).
        if xr
            .iter()
            .zip(lo.iter().zip(hi))
            .any(|(x, (l, h))| x < l || x > h)
        {
            continue;
        }
        // (a) flattened sequential net == symbolic DAG forward, EXACTLY.
        let exact = dag_eval_exact(dag, &xr);
        let flatv = loaded_eval_vec(flat, &xr);
        assert_eq!(exact.len(), flatv.len(), "output arity mismatch");
        for (o, (a, b)) in exact.iter().zip(&flatv).enumerate() {
            assert!(
                a == b,
                "flatten/DAG exact forward mismatch at output {o}: {} vs {}",
                fmt_rat(a),
                fmt_rat(b)
            );
        }
        // (b) exact forward vs independent f32 interpreter, within f32 rounding.
        let f32v = f32_forward(g, &xf);
        assert_eq!(f32v.len(), exact.len());
        for (o, y) in f32v.iter().enumerate() {
            let e = rat_to_f64(&exact[o]);
            let d = (e - f64::from(*y)).abs();
            let rel = d / e.abs().max(1.0);
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(rel);
            assert!(
                rel <= 1e-3,
                "exact vs f32 forward diverges at output {o}: exact {e} f32 {y}"
            );
        }
        checked += 1;
    }
    assert!(checked >= 5, "too few in-box parity points ({checked})");
    println!(
        "PARITY_OK points={checked} outputs={} max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e}",
        dag.output.rows()
    );
}

/// Typed identity of one ORIGINAL VNNLIB unsafe row (index among output rows,
/// before any synthetic conic targets are added).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UnsafeSourceRowId(usize);

/// One exact original unsafe row in both output-variable and final-readout
/// coordinates. The latter is what the scalar exact CROWN worker certifies.
#[derive(Debug, Clone)]
struct UnsafeSourceRow {
    id: UnsafeSourceRowId,
    label: String,
    original_constraint: LinearConstraint,
    normalized_constraint: LinearConstraint,
    out_w: Vec<Rat>,
    out_b: Rat,
    u_bound: Rat,
}

/// One exact non-negative multiplier naming its original VNNLIB source row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConicSourceTerm {
    source_row: UnsafeSourceRowId,
    multiplier: Rat,
}

/// Exact provenance for a pure or synthetic target.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ConicProvenance {
    terms: Vec<ConicSourceTerm>,
}

impl ConicProvenance {
    fn pure(source_row: UnsafeSourceRowId) -> Self {
        Self {
            terms: vec![ConicSourceTerm {
                source_row,
                multiplier: Rat::ONE,
            }],
        }
    }
}

/// One validated refutation target derived from original unsafe-region rows:
/// the atom holds nowhere on a box iff the exact lower bound of
/// `y' = out_w·a + out_b` exceeds `u_bound`. Construction is private to
/// [`trusted_conj_target`], which exactly reconstructs all three values from
/// typed source rows/multipliers and checks the corresponding entailment.
struct ConjTarget {
    label: String,
    out_w: Vec<Rat>,
    out_b: Rat,
    u_bound: Rat,
    provenance: ConicProvenance,
    source_entailment: EntailmentCertificate,
}

/// Fail-closed parser for `NYCERT_MIX_SET="a:b,c:d,..."`.
///
/// Zero on one side is useful (it repeats a pure atom) and remains accepted;
/// negative weights are not conic, while `0:0` proves nothing, so both are
/// rejected before target construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MixPair {
    left: i64,
    right: i64,
}

fn parse_mix_set(raw: &str) -> Result<Vec<MixPair>, String> {
    if raw.trim().is_empty() {
        return Err("mix set is empty".to_string());
    }
    let mut mixes = Vec::new();
    for (index, pair) in raw.split(',').enumerate() {
        let pair = pair.trim();
        let Some((left, right)) = pair.split_once(':') else {
            return Err(format!("mix {index} must have form a:b (got {pair:?})"));
        };
        let left: i64 = left
            .trim()
            .parse()
            .map_err(|_| format!("mix {index} has invalid left multiplier"))?;
        let right: i64 = right
            .trim()
            .parse()
            .map_err(|_| format!("mix {index} has invalid right multiplier"))?;
        if left < 0 || right < 0 {
            return Err(format!(
                "mix {index} has negative multiplier ({left}:{right}); conic weights must be >= 0"
            ));
        }
        if left == 0 && right == 0 {
            return Err(format!(
                "mix {index} is 0:0; at least one conic multiplier must be positive"
            ));
        }
        mixes.push(MixPair { left, right });
    }
    Ok(mixes)
}

/// Reconstruct a target exactly from ORIGINAL VNNLIB unsafe rows and validate
/// the output-level conic entailment before any f64 screen or exact leaf can
/// consume it. This is the authority gate for synthetic targets.
fn trusted_conj_target(
    label: String,
    out_w: Vec<Rat>,
    out_b: Rat,
    u_bound: Rat,
    provenance: ConicProvenance,
    source_rows: &[UnsafeSourceRow],
) -> Result<ConjTarget, String> {
    if provenance.terms.is_empty() {
        return Err("target has no source rows".to_string());
    }
    let mut rebuilt_w = vec![Rat::ZERO; out_w.len()];
    let mut rebuilt_b = Rat::ZERO;
    let mut rebuilt_u = Rat::ZERO;
    let mut conclusion_coefficients: BTreeMap<String, Rat> = BTreeMap::new();
    let mut premises = Vec::new();
    let mut multipliers = Vec::new();
    let mut has_positive = false;

    for term in &provenance.terms {
        if term.multiplier.is_negative() {
            return Err(format!(
                "source row {} has negative multiplier",
                term.source_row.0
            ));
        }
        has_positive |= term.multiplier.is_positive();
        let row = source_rows
            .get(term.source_row.0)
            .ok_or_else(|| format!("missing source row {}", term.source_row.0))?;
        if row.id != term.source_row || row.out_w.len() != rebuilt_w.len() {
            return Err(format!(
                "source row {} shape/id mismatch",
                term.source_row.0
            ));
        }
        if row.normalized_constraint.kind != ConstraintKind::Le
            || row.normalized_constraint.constant != row.u_bound
        {
            return Err(format!(
                "source row {} normalization mismatch",
                term.source_row.0
            ));
        }
        for (slot, coefficient) in rebuilt_w.iter_mut().zip(&row.out_w) {
            *slot = slot
                .add(
                    coefficient
                        .mul(term.multiplier)
                        .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
        }
        rebuilt_b = rebuilt_b
            .add(row.out_b.mul(term.multiplier).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        rebuilt_u = rebuilt_u
            .add(
                row.u_bound
                    .mul(term.multiplier)
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
        for (name, coefficient) in &row.normalized_constraint.coefficients {
            let scaled = coefficient
                .mul(term.multiplier)
                .map_err(|e| e.to_string())?;
            let prior = conclusion_coefficients.remove(name).unwrap_or(Rat::ZERO);
            let combined = prior.add(scaled).map_err(|e| e.to_string())?;
            if !combined.is_zero() {
                conclusion_coefficients.insert(name.clone(), combined);
            }
        }
        premises.push(row.original_constraint.clone());
        multipliers.push(term.multiplier);
    }
    if !has_positive {
        return Err("all source multipliers are zero".to_string());
    }
    if rebuilt_w != out_w || rebuilt_b != out_b || rebuilt_u != u_bound {
        return Err("target does not exactly reconstruct from its source rows".to_string());
    }

    let source_entailment = EntailmentCertificate {
        premises,
        multipliers,
        conclusion: LinearConstraint {
            kind: ConstraintKind::Le,
            coefficients: conclusion_coefficients,
            constant: rebuilt_u,
        },
    };
    let (derived, claimed) =
        check_entailment(&source_entailment).map_err(|e| format!("source entailment: {e}"))?;
    if derived != claimed {
        return Err("source entailment was not the exact stated combination".to_string());
    }
    Ok(ConjTarget {
        label,
        out_w,
        out_b,
        u_bound,
        provenance,
        source_entailment,
    })
}

fn target_authority_json(index: usize, target: &ConjTarget) -> ny_cert::json::Json {
    let terms: Vec<ny_cert::json::Json> = target
        .provenance
        .terms
        .iter()
        .map(|term| {
            ny_cert::json!({
                "source_row": term.source_row.0,
                "multiplier": fmt_rat(&term.multiplier),
            })
        })
        .collect();
    ny_cert::json!({
        "target": index,
        "label": target.label.clone(),
        "source_terms": terms,
        "reconstructed_readout": {
            "weights": target.out_w.iter().map(fmt_rat).collect::<Vec<_>>(),
            "bias": fmt_rat(&target.out_b),
            "upper_bound": fmt_rat(&target.u_bound),
        },
        "source_entailment": entailment_to_json(&target.source_entailment)
            .expect("checked source entailment must serialize"),
        "leaf_certificate_composition": "the leaf Farkas refutes the reconstructed scalar atom; source_entailment proves the original VNNLIB unsafe rows entail the corresponding output-space row. NY checks the exact readout reconstruction, but Clean does not compose these artifacts",
    })
}

/// Metadata for a certified leaf of the conjunction branch-and-bound (the
/// Farkas certificate itself is checked, serialized, and written to disk by
/// the worker/driver as soon as the leaf closes — never accumulated in RAM).
struct ConjLeaf {
    id: usize,
    lo: Vec<Rat>,
    hi: Vec<Rat>,
    target: usize,
    margin_s: String,
    margin_f64: f64,
    max_bits: u64,
}

/// Outcome of one exact leaf attempt (all heavy rationals already serialized).
struct LeafOutcome {
    margin_s: String,
    margin_f64: f64,
    farkas_json: String,
    max_bits: u64,
    selfcheck_us: u128,
}

/// One live box of the conjunction branch-and-bound.
struct ConjNode {
    lo: Vec<Rat>,
    hi: Vec<Rat>,
    depth: usize,
    id: usize,
}

/// Immutable context needed to pick a branch coordinate for a box.
struct SplitCtx<'a> {
    smart_branch: bool,
    tight: bool,
    w64: &'a [Vec<Vec<f64>>],
    b64: &'a [Vec<f64>],
    t64: &'a [(Vec<f64>, f64, f64)],
    n: usize,
    half: Rat,
    max_depth: usize,
    label: &'a str,
}

/// Bisect `nd` (exact midpoint of the chosen coordinate) into two child boxes,
/// recording the split. Exits fail-closed at the depth cap.
///
/// Branch selection. Default: widest coordinate. `smart_branch`: pick the
/// coordinate whose WORSE child has the best screened margin (2·n·|targets|
/// extra f64 screens per split — a pure heuristic; the exact pass still
/// decides every leaf, so soundness is unaffected). Near-ties prefer the WIDER
/// coordinate so a plateau cannot pin the split on one coordinate forever.
fn split_node(
    nd: &ConjNode,
    bm: f64,
    work: &mut Vec<ConjNode>,
    splits: &mut BTreeMap<usize, (usize, Rat, usize, usize)>,
    next_id: &mut usize,
    ctx: &SplitCtx,
) {
    if nd.depth >= ctx.max_depth {
        println!(
            "CONJ_FAILED depth cap {} at node {} (screen margin {bm:.3e} on {})",
            ctx.max_depth, nd.id, ctx.label
        );
        std::process::exit(3);
    }
    let n = ctx.n;
    let mut c = 0usize;
    if ctx.smart_branch {
        let lo64: Vec<f64> = nd.lo.iter().map(rat_to_f64).collect();
        let hi64: Vec<f64> = nd.hi.iter().map(rat_to_f64).collect();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_width = f64::NEG_INFINITY;
        for i in 0..n {
            let width = hi64[i] - lo64[i];
            if width <= 0.0 {
                continue;
            }
            let m64 = f64::midpoint(lo64[i], hi64[i]);
            let mut worse = f64::INFINITY;
            for (clo, chi) in [(lo64[i], m64), (m64, hi64[i])] {
                let mut l2 = lo64.clone();
                let mut h2 = hi64.clone();
                l2[i] = clo;
                h2[i] = chi;
                let mut child_best = f64::NEG_INFINITY;
                for (ow, ob, ub) in ctx.t64 {
                    let m = crown_lower_dispatch(ctx.tight, ctx.w64, ctx.b64, ow, *ob, &l2, &h2);
                    child_best = child_best.max(m - ub);
                }
                worse = worse.min(child_best);
            }
            let eps = 1e-9 * (1.0 + best_score.abs().min(1e9));
            if worse > best_score + eps || (worse > best_score - eps && width > best_width) {
                best_score = worse.max(best_score);
                best_width = width;
                c = i;
            }
        }
    } else {
        let mut bestw = Rat::ZERO;
        for i in 0..n {
            let w = nd.hi[i].sub(nd.lo[i]).unwrap();
            if i == 0 || w > bestw {
                bestw = w;
                c = i;
            }
        }
    }
    let mid = nd.lo[c].add(nd.hi[c]).unwrap().mul(ctx.half).unwrap();
    // Node ids and depths are counters; checked, so an (unreachable) wrap is a
    // reported failure, not a reused id (release builds would wrap silently).
    let lid = *next_id;
    let (Some(rid), Some(after), Some(child_depth)) = (
        lid.checked_add(1),
        lid.checked_add(2),
        nd.depth.checked_add(1),
    ) else {
        println!(
            "CONJ_FAILED node id/depth counter exhausted at node {} on {}",
            nd.id, ctx.label
        );
        std::process::exit(3);
    };
    *next_id = after;
    splits.insert(nd.id, (c, mid, lid, rid));
    let mut lo_hi = nd.hi.clone();
    lo_hi[c] = mid;
    work.push(ConjNode {
        lo: nd.lo.clone(),
        hi: lo_hi,
        depth: child_depth,
        id: lid,
    });
    let mut hi_lo = nd.lo.clone();
    hi_lo[c] = mid;
    work.push(ConjNode {
        lo: hi_lo,
        hi: nd.hi.clone(),
        depth: child_depth,
        id: rid,
    });
}

/// ONE exact leaf certification. The conj driver runs this in a SHORT-LIVED
/// WORKER THREAD (never on the main thread):
///
/// `Rat` is a handle into a THREAD-LOCAL interning arena that only ever grows;
/// a long branch-and-bound in a single thread therefore accumulates every
/// intermediate bignum of every exact pass (multi-GB on the larger cersyve
/// trees). Isolating each exact pass in its own thread frees the whole arena
/// when the thread exits, keeping the driver's memory flat — and lets
/// `NYCERT_JOBS` boxes certify concurrently. All exact values cross the
/// boundary as canonical `n/d` strings (lossless), and the returned
/// certificate is already SELF-CHECKED (in-tree mirror of Clean's verifier)
/// and serialized to Clean's external-cert JSON. That JSON proves only the
/// leaf-local scalar contradiction; Clean does not consume `tree.json` and
/// therefore does not check the target's source-row composition or the tree
/// cover. Those separate obligations remain explicit in the manifest.
///
/// Returns `None` when the exact bound does not refute the atom on this box
/// (the caller splits further). Worker panics (soundness asserts) propagate.
// The Arc<Vec<..>> fields are shared read-only across worker threads as-is;
// converting to Arc<[..]> buys nothing here and churns the call sites.
#[allow(clippy::too_many_arguments, clippy::rc_buffer)]
fn exact_leaf_compute(
    hidden_w_s: std::sync::Arc<Vec<Vec<Vec<String>>>>,
    hidden_b_s: std::sync::Arc<Vec<Vec<String>>>,
    out_w_s: std::sync::Arc<Vec<String>>,
    out_b_s: String,
    u_bound_s: String,
    lo_s: Vec<String>,
    hi_s: Vec<String>,
    tight: bool,
    cuts: Vec<(usize, usize)>,
) -> Option<LeafOutcome> {
    {
        let p = |s: &String| parse_rat_token(s);
        let weights: Vec<Vec<Vec<Rat>>> = hidden_w_s
            .iter()
            .map(|l| l.iter().map(|r| r.iter().map(p).collect()).collect())
            .collect();
        let biases: Vec<Vec<Rat>> = hidden_b_s
            .iter()
            .map(|l| l.iter().map(p).collect())
            .collect();
        let out_weight: Vec<Rat> = out_w_s.iter().map(p).collect();
        let out_bias = parse_rat_token(&out_b_s);
        let u_bound = parse_rat_token(&u_bound_s);
        let lo: Vec<Rat> = lo_s.iter().map(p).collect();
        let hi: Vec<Rat> = hi_s.iter().map(p).collect();
        let n = lo.len();
        let problem = DeepReluProblem {
            weights,
            biases,
            out_weight,
            out_bias,
            input_lower: lo.clone(),
            input_upper: hi.clone(),
            alpha: None,
            interm_round: false,
        };
        // Joint first-layer 2-neuron cuts (kernel-backed `multiReluCut` premise):
        // the exact pass recomputes each cut bound B_ij from the box corners in
        // rationals, so the emitted premise is exactly checkable like the rest.
        let cert =
            match problem.certify_with_interm_cuts(Rat::from_int(-1_000_000_000), tight, &cuts) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("[conj] exact certify error: {e}");
                    return None;
                }
            };
        // The greedy cut diversion is not guaranteed to dominate the cut-free
        // pass; if the cut-augmented bound misses, fall back to the base pass
        // and keep the better certificate (both are valid).
        let cert = if !cuts.is_empty() && cert.lower_bound <= parse_rat_token(&u_bound_s) {
            match problem.certify_with_interm(Rat::from_int(-1_000_000_000), tight) {
                Ok(c) if c.lower_bound > cert.lower_bound => c,
                _ => cert,
            }
        } else {
            cert
        };
        let m = cert.lower_bound;
        if m <= u_bound {
            return None;
        }
        // Exact soundness spot-check: the certified bound must not exceed the
        // true output at any box corner.
        if n <= 16 {
            for mask in 0u32..(1u32 << n) {
                // Bit `i` of `mask` picks hi[i] (set) or lo[i] (clear); `bits`
                // walks the bits from the lowest, one coordinate per step.
                let x: Vec<Rat> = lo
                    .iter()
                    .zip(&hi)
                    .scan(mask, |bits, (l, h)| {
                        let pick_hi = *bits & 1 != 0;
                        *bits >>= 1;
                        Some(if pick_hi { *h } else { *l })
                    })
                    .collect();
                let y = problem.eval(&x).expect("corner eval");
                assert!(m <= y, "UNSOUND leaf: corner below certified bound");
            }
        }
        let margin = m.sub(u_bound).unwrap();
        let mut fc = cert.entailment.premises.clone();
        let mut fm = cert.entailment.multipliers.clone();
        // This scalar row is the exact target passed by `conj_pipeline`. Every
        // such target was admitted only by `trusted_conj_target`, which
        // reconstructs it from original VNNLIB rows and checks their conic
        // entailment. The leaf Farkas remains a certificate relative to this
        // scalar premise; `tree.json.target_authority` carries the checked
        // source entailment required to compose it back to the VNNLIB unsafe
        // region (the standalone leaf JSON is intentionally not overclaimed).
        fc.push(LinearConstraint::with_kind(
            ConstraintKind::Le,
            &[("y", Rat::ONE)],
            u_bound,
        ));
        fm.push(Rat::ONE);
        let farkas = FarkasCertificate {
            constraints: fc,
            multipliers: fm,
        };
        let entailment = EntailmentCertificate {
            premises: cert.entailment.premises.clone(),
            multipliers: cert.entailment.multipliers,
            conclusion: LinearConstraint::with_kind(ConstraintKind::Ge, &[("y", Rat::ONE)], m),
        };
        // Self-check (in-tree mirror of Clean's verifier) before serializing.
        let t = Instant::now();
        let c = check_farkas(&farkas).expect("leaf farkas self-check");
        assert!(!c.is_positive(), "farkas residual not contradictory");
        let (derived, claimed) = check_entailment(&entailment).expect("leaf entailment");
        assert!(derived <= claimed);
        let selfcheck_us = t.elapsed().as_micros();
        let far = farkas_to_json(&farkas).expect("farkas json");
        Some(LeafOutcome {
            margin_s: fmt_rat(&margin),
            margin_f64: rat_to_f64(&margin),
            farkas_json: ny_cert::json::to_string(&far),
            max_bits: cert_max_bits(&entailment),
            selfcheck_us,
        })
    }
}

/// Complete exact branch-and-bound refuting a CONJUNCTIVE unsafe region
/// (all `y_atoms` simultaneously). A box is closed when ONE atom is refuted on
/// it by an exact-rational Farkas certificate; otherwise it is bisected at the
/// exact midpoint of its widest coordinate. The f64 CROWN screen only navigates
/// — every leaf is decided by the exact bignum pass, self-checked with the
/// in-tree mirror of Clean's verifier, and emitted as Clean external-cert JSON.
/// Clean checks each leaf's local arithmetic only; it does not establish the
/// external source-row composition recorded in `target_authority` or the tree
/// cover recorded in `tree.json`.
fn conj_pipeline(
    net: &LoadedNet,
    root_lo: &[Rat],
    root_hi: &[Rat],
    y_atoms: &[VnnAtom],
    out_dir: &str,
    load_us: u128,
) -> ! {
    let n_hidden = net.hidden.len();
    assert!(n_hidden >= 1, "need at least one hidden layer");
    let hidden_w: Vec<Vec<Vec<Rat>>> = net.hidden.iter().map(|l| l.w.clone()).collect();
    let hidden_b: Vec<Vec<Rat>> = net.hidden.iter().map(|l| l.b.clone()).collect();
    let readout = &net.readout;
    let out_dim = readout.w.len();

    // Build the ORIGINAL exact source rows first. Synthetic targets may only be
    // constructed from these rows through `trusted_conj_target` below.
    let mut source_rows: Vec<UnsafeSourceRow> = vec![];
    for a in y_atoms {
        let j = match var_index(&a.var) {
            Some(j) if j < out_dim => j,
            other => {
                eprintln!("CONJ: bad Y variable {:?} (index {other:?})", a.var);
                std::process::exit(2);
            }
        };
        let c = match decimal_to_rat(&a.rhs_raw) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("CONJ: unparseable Y threshold {:?}: {e:?}", a.rhs_raw);
                std::process::exit(2);
            }
        };
        let source_row = UnsafeSourceRowId(source_rows.len());
        let (label, original_constraint, normalized_constraint, out_w, out_b, u_bound) = if a.op
            == "<="
        {
            // unsafe atom Y_j <= c ; refuted on a box iff  min Y_j > c.
            (
                format!("Y_{j}<={}", a.rhs_raw),
                LinearConstraint::with_kind(ConstraintKind::Le, &[(a.var.as_str(), Rat::ONE)], c),
                LinearConstraint::with_kind(ConstraintKind::Le, &[(a.var.as_str(), Rat::ONE)], c),
                readout.w[j].clone(),
                readout.b[j],
                c,
            )
        } else {
            // unsafe atom Y_j >= c ; refuted iff  min (-Y_j) > -c.
            (
                format!("Y_{j}>={}", a.rhs_raw),
                LinearConstraint::with_kind(ConstraintKind::Ge, &[(a.var.as_str(), Rat::ONE)], c),
                LinearConstraint::with_kind(
                    ConstraintKind::Le,
                    &[(a.var.as_str(), Rat::ONE.neg())],
                    c.neg(),
                ),
                readout.w[j].iter().map(|r| r.neg()).collect(),
                readout.b[j].neg(),
                c.neg(),
            )
        };
        source_rows.push(UnsafeSourceRow {
            id: source_row,
            label,
            original_constraint,
            normalized_constraint,
            out_w,
            out_b,
            u_bound,
        });
    }
    assert!(!source_rows.is_empty(), "CONJ: no Y atoms");

    let reject_authority = |reason: String| -> ! {
        eprintln!("CONJ_AUTHORITY_REJECTED: {reason}");
        std::process::exit(2);
    };
    let mut targets: Vec<ConjTarget> = source_rows
        .iter()
        .map(|row| {
            trusted_conj_target(
                row.label.clone(),
                row.out_w.clone(),
                row.out_b,
                row.u_bound,
                ConicProvenance::pure(row.id),
                &source_rows,
            )
            .unwrap_or_else(|reason| reject_authority(reason))
        })
        .collect();
    // CONIC MIXES: every target is normalized to "unsafe ⊨ y'_t ≤ u_t", so any
    // non-negative combination Σ λ_t·y'_t ≤ Σ λ_t·u_t is ALSO entailed by the
    // unsafe region — refuting it on a box refutes the conjunction there. A
    // mixed functional can close junction boxes (both atom boundaries nearby)
    // that neither pure atom can, collapsing the deepest parts of the tree.
    // The leaf Farkas pushes the combined atom; tree.json records the mix.
    if targets.len() == 2 && std::env::var("NYCERT_MIX").ok().as_deref() != Some("0") {
        // NYCERT_MIX_SET overrides the default ratio ladder: "1:1,2:1,1:2,...".
        let mixes = match std::env::var("NYCERT_MIX_SET") {
            Ok(raw) => parse_mix_set(&raw).unwrap_or_else(|reason| reject_authority(reason)),
            Err(_) => vec![
                MixPair { left: 1, right: 1 },
                MixPair { left: 2, right: 1 },
                MixPair { left: 1, right: 2 },
            ],
        };
        for mix in mixes {
            let (l0, l1) = (mix.left, mix.right);
            let (a, b) = (&targets[0], &targets[1]);
            let s0 = Rat::from_int(i128::from(l0));
            let s1 = Rat::from_int(i128::from(l1));
            let out_w: Vec<Rat> = a
                .out_w
                .iter()
                .zip(&b.out_w)
                .map(|(x, y)| x.mul(s0).unwrap().add(y.mul(s1).unwrap()).unwrap())
                .collect();
            let label = format!("{l0}*[{}] + {l1}*[{}]", a.label, b.label);
            let out_b = a
                .out_b
                .mul(s0)
                .unwrap()
                .add(b.out_b.mul(s1).unwrap())
                .unwrap();
            let u_bound = a
                .u_bound
                .mul(s0)
                .unwrap()
                .add(b.u_bound.mul(s1).unwrap())
                .unwrap();
            let provenance = ConicProvenance {
                terms: vec![
                    ConicSourceTerm {
                        source_row: source_rows[0].id,
                        multiplier: s0,
                    },
                    ConicSourceTerm {
                        source_row: source_rows[1].id,
                        multiplier: s1,
                    },
                ],
            };
            targets.push(
                trusted_conj_target(label, out_w, out_b, u_bound, provenance, &source_rows)
                    .unwrap_or_else(|reason| reject_authority(reason)),
            );
        }
    }

    // f64 mirrors for the navigation screen.
    let f = |r: &Rat| rat_to_f64(r);
    let w64: Vec<Vec<Vec<f64>>> = hidden_w
        .iter()
        .map(|l| l.iter().map(|row| row.iter().map(&f).collect()).collect())
        .collect();
    let b64: Vec<Vec<f64>> = hidden_b
        .iter()
        .map(|l| l.iter().map(&f).collect())
        .collect();
    let t64: Vec<(Vec<f64>, f64, f64)> = targets
        .iter()
        .map(|t| (t.out_w.iter().map(&f).collect(), f(&t.out_b), f(&t.u_bound)))
        .collect();

    let tight = std::env::var("NYCERT_TIGHT").ok().as_deref() == Some("1");
    let smart_branch = std::env::var("NYCERT_SMART_BRANCH").ok().as_deref() == Some("1");
    // NYCERT_CUT=1 (with tight): first-layer 2-neuron joint cuts as extra
    // Farkas premises (the kernel-backed multiReluCut lever) — tighter leaves.
    let use_cut = std::env::var("NYCERT_CUT").ok().as_deref() == Some("1") && tight;
    let max_cuts: usize = std::env::var("NYCERT_MAXCUTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let max_depth: usize = std::env::var("NYCERT_BISECT_MAXDEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let max_leaves: usize = std::env::var("NYCERT_MAX_LEAVES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let half = Rat::new(1, 2).unwrap();

    // Lossless string mirrors of the exact net, shared with the per-leaf worker
    // threads (each worker re-interns into its own thread-local arena, which is
    // FREED when the worker exits — see `exact_leaf_worker`).
    use std::sync::Arc;
    let hidden_w_s: Arc<Vec<Vec<Vec<String>>>> = Arc::new(
        hidden_w
            .iter()
            .map(|l| {
                l.iter()
                    .map(|row| row.iter().map(fmt_rat).collect())
                    .collect()
            })
            .collect(),
    );
    let hidden_b_s: Arc<Vec<Vec<String>>> = Arc::new(
        hidden_b
            .iter()
            .map(|l| l.iter().map(fmt_rat).collect())
            .collect(),
    );
    let targets_s: Vec<(Arc<Vec<String>>, String, String)> = targets
        .iter()
        .map(|t| {
            (
                Arc::new(t.out_w.iter().map(fmt_rat).collect::<Vec<String>>()),
                fmt_rat(&t.out_b),
                fmt_rat(&t.u_bound),
            )
        })
        .collect();
    std::fs::create_dir_all(out_dir).unwrap();

    let mut work = vec![ConjNode {
        lo: root_lo.to_vec(),
        hi: root_hi.to_vec(),
        depth: 0,
        id: 0,
    }];
    let mut next_id = 1usize;
    let mut splits: BTreeMap<usize, (usize, Rat, usize, usize)> = BTreeMap::new();
    let mut leaves: Vec<ConjLeaf> = vec![];
    let mut explored = 0usize;
    let mut exact_calls = 0usize;
    let mut exact_us = 0u128;
    let mut check_us = 0u128;
    let n = net.input_dim;
    // NYCERT_JOBS>1: run up to that many exact leaf certifications CONCURRENTLY
    // (independent boxes; each in its own arena-isolated worker thread). Purely
    // a throughput knob — every leaf is still decided by its own exact pass.
    let jobs: usize = std::env::var("NYCERT_JOBS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    // A worker panic is a soundness assert firing; never hang on it — abort.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default_hook(info);
        eprintln!("CONJ: worker panic — aborting");
        std::process::exit(4);
    }));
    let screen_only = std::env::var("NYCERT_CONJ_SCREEN").ok().as_deref() == Some("1");
    let t_solve = Instant::now();
    // (nd, target, cut pairs) batch awaiting exact certification.
    let mut batch: Vec<(ConjNode, usize, Vec<(usize, usize)>)> = vec![];
    while !work.is_empty() || !batch.is_empty() {
        // --- Phase 1: screen/split until `jobs` exact candidates are queued ---
        while batch.len() < jobs {
            let Some(nd) = work.pop() else { break };
            explored += 1;
            if leaves.len() >= max_leaves {
                println!("CONJ_FAILED leaf budget exceeded ({max_leaves})");
                std::process::exit(3);
            }
            let lo64: Vec<f64> = nd.lo.iter().map(&f).collect();
            let hi64: Vec<f64> = nd.hi.iter().map(&f).collect();
            // Optional joint first-layer 2-neuron cuts for this box (NYCERT_CUT=1).
            let box_cuts: Vec<ReluCut2> = if use_cut {
                let (pl, ph) = crown_intermediate_bounds_f64(&w64, &b64, &lo64, &hi64);
                discover_cuts2(&w64[0], &b64[0], &lo64, &hi64, &pl[0], &ph[0], max_cuts)
            } else {
                vec![]
            };
            // Screen every target; try the exact pass on the best-margin one.
            let mut best: Option<(usize, f64)> = None;
            for (ti, (ow, ob, ub)) in t64.iter().enumerate() {
                let mut m = crown_lower_dispatch(tight, &w64, &b64, ow, *ob, &lo64, &hi64);
                if !box_cuts.is_empty() {
                    // The cut screen is a second VALID lower bound; max stays sound.
                    m = m.max(crown_lower_f64_tight_cut(
                        &w64, &b64, ow, *ob, &lo64, &hi64, &box_cuts,
                    ));
                }
                let mg = m - ub;
                if best.map_or(true, |(_, b)| mg > b) {
                    best = Some((ti, mg));
                }
            }
            let (bt, bm) = best.expect("targets nonempty");
            // NYCERT_CONJ_SCREEN=1: f64-screen-only dry run (NO exact pass, NO
            // certificates) — measures the tree size/shape a full run will need.
            if bm > 0.0 && screen_only {
                leaves.push(ConjLeaf {
                    id: nd.id,
                    lo: nd.lo.clone(),
                    hi: nd.hi.clone(),
                    target: bt,
                    margin_s: String::new(),
                    margin_f64: bm,
                    max_bits: 0,
                });
                continue;
            }
            if bm > 0.0 {
                // Defer the exact pass to the parallel phase below.
                batch.push((nd, bt, box_cuts.iter().map(|c| (c.i, c.j)).collect()));
                continue;
            }
            split_node(
                &nd,
                bm,
                &mut work,
                &mut splits,
                &mut next_id,
                &SplitCtx {
                    smart_branch,
                    tight,
                    w64: &w64,
                    b64: &b64,
                    t64: &t64,
                    n,
                    half,
                    max_depth,
                    label: &targets[bt].label,
                },
            );
        } // end phase 1

        // --- Phase 2: certify the queued boxes in parallel worker threads ---
        if batch.is_empty() {
            continue;
        }
        let t1 = Instant::now();
        // `batch` is refilled each outer-loop iteration; drain empties it in
        // place to reuse the allocation (into_iter would move it out).
        #[allow(clippy::iter_with_drain)]
        let handles: Vec<(
            ConjNode,
            usize,
            std::thread::JoinHandle<Option<LeafOutcome>>,
        )> = batch
            .drain(..)
            .map(|(nd, bt, cuts)| {
                // A statistics counter: saturating, never a wrap or a panic.
                exact_calls = exact_calls.saturating_add(1);
                let (ow_s, ob_s, ub_s) = &targets_s[bt];
                let hw = Arc::clone(&hidden_w_s);
                let hb = Arc::clone(&hidden_b_s);
                let ow = Arc::clone(ow_s);
                let ob = ob_s.clone();
                let ub = ub_s.clone();
                let lo_s: Vec<String> = nd.lo.iter().map(fmt_rat).collect();
                let hi_s: Vec<String> = nd.hi.iter().map(fmt_rat).collect();
                let h = std::thread::spawn(move || {
                    exact_leaf_compute(hw, hb, ow, ob, ub, lo_s, hi_s, tight, cuts)
                });
                (nd, bt, h)
            })
            .collect();
        for (nd, bt, h) in handles {
            match h.join().expect("exact leaf worker panicked") {
                Some(o) => {
                    // Stream the checked certificate straight to disk.
                    std::fs::write(
                        format!("{out_dir}/leaf_{}.farkas.json", nd.id),
                        &o.farkas_json,
                    )
                    .unwrap();
                    check_us += o.selfcheck_us;
                    leaves.push(ConjLeaf {
                        id: nd.id,
                        lo: nd.lo.clone(),
                        hi: nd.hi.clone(),
                        target: bt,
                        margin_s: o.margin_s,
                        margin_f64: o.margin_f64,
                        max_bits: o.max_bits,
                    });
                }
                None => split_node(
                    &nd,
                    0.0,
                    &mut work,
                    &mut splits,
                    &mut next_id,
                    &SplitCtx {
                        smart_branch,
                        tight,
                        w64: &w64,
                        b64: &b64,
                        t64: &t64,
                        n,
                        half,
                        max_depth,
                        label: &targets[bt].label,
                    },
                ),
            }
        }
        exact_us += t1.elapsed().as_micros();
    }
    let solve_us = t_solve.elapsed().as_micros();

    // --- Structural covering re-check: the split tree partitions the root box
    // and every leaf carries a certificate. Walked independently of the solve
    // loop's bookkeeping (from the recorded splits + leaf boxes only). ---
    let leaf_by_id: BTreeMap<usize, &ConjLeaf> = leaves.iter().map(|l| (l.id, l)).collect();
    let mut cover_ok = true;
    let mut stack: Vec<(usize, Vec<Rat>, Vec<Rat>)> = vec![(0, root_lo.to_vec(), root_hi.to_vec())];
    while let Some((id, blo, bhi)) = stack.pop() {
        if let Some((c, mid, lid, rid)) = splits.get(&id) {
            if !(blo[*c] < *mid && *mid < bhi[*c]) {
                cover_ok = false;
                break;
            }
            let mut lh = bhi.clone();
            lh[*c] = *mid;
            stack.push((*lid, blo.clone(), lh));
            let mut hl = blo;
            hl[*c] = *mid;
            stack.push((*rid, hl, bhi));
        } else if let Some(leaf) = leaf_by_id.get(&id) {
            if leaf.lo != blo || leaf.hi != bhi {
                cover_ok = false;
                break;
            }
        } else {
            cover_ok = false;
            break;
        }
    }
    assert!(cover_ok, "CONJ: covering re-check failed");

    // --- Emit the tree manifest ---------------------------------------------
    let rat_s = |r: &Rat| fmt_rat(r);
    let box_json = |v: &[Rat]| -> ny_cert::json::Json {
        ny_cert::json::Json::Arr(
            v.iter()
                .map(|r| ny_cert::json::Json::Str(rat_s(r)))
                .collect(),
        )
    };
    let mut max_bits = 0u64;
    let mut leaf_manifest = vec![];
    for leaf in &leaves {
        max_bits = max_bits.max(leaf.max_bits);
        if screen_only {
            // f64-screen leaf: no exact margin exists and no certificate file
            // was written, so the manifest must not name one.
            leaf_manifest.push(ny_cert::json!({
                "id": leaf.id,
                "lo": box_json(&leaf.lo),
                "hi": box_json(&leaf.hi),
                "screened_target": leaf.target,
                "screened_atom": targets[leaf.target].label.clone(),
                "margin_approx": leaf.margin_f64,
            }));
        } else {
            // Every certified leaf was decided by the exact pass (its worker
            // wrote the self-checked Farkas cert before pushing the leaf); a
            // leaf without an exact margin must never reach `decided=true`.
            assert!(
                !leaf.margin_s.is_empty(),
                "CONJ: leaf {} carries no exact certificate",
                leaf.id
            );
            leaf_manifest.push(ny_cert::json!({
                "id": leaf.id,
                "lo": box_json(&leaf.lo),
                "hi": box_json(&leaf.hi),
                "refuted_target": leaf.target,
                "refuted_atom": targets[leaf.target].label.clone(),
                "margin": leaf.margin_s.clone(),
                "margin_approx": leaf.margin_f64,
                "farkas": format!("leaf_{}.farkas.json", leaf.id),
            }));
        }
    }
    let split_manifest: Vec<ny_cert::json::Json> = splits
        .iter()
        .map(|(id, (c, mid, lid, rid))| {
            ny_cert::json!({
                "id": *id, "coord": *c, "mid": rat_s(mid),
                "lo_child": *lid, "hi_child": *rid,
            })
        })
        .collect();
    let original_unsafe_rows: Vec<ny_cert::json::Json> = source_rows
        .iter()
        .map(|row| {
            ny_cert::json!({
                "source_row": row.id.0,
                "label": row.label.clone(),
            })
        })
        .collect();
    let target_authority: Vec<ny_cert::json::Json> = targets
        .iter()
        .enumerate()
        .map(|(index, target)| target_authority_json(index, target))
        .collect();

    // Screen-only dry run: nothing was proved — no exact pass ran and no
    // certificates exist. Emit a shape-only manifest under a distinct name
    // and a non-verdict line; never the certified tree type or
    // `CONJ_RESULT decided=true`.
    if screen_only {
        let tree = ny_cert::json!({
            "type": "conjunctive_screen_only_tree",
            "version": "1.0",
            "unsafe_atoms": targets.iter().map(|t| t.label.clone()).collect::<Vec<_>>(),
            "original_unsafe_rows": original_unsafe_rows,
            "target_authority": target_authority,
            "semantics": "f64 CROWN screen dry run (no directed rounding, no exact pass, no certificates): each leaf's screen margin was positive on its box; the split tree partitions the root box (re-checked structurally). A tree size/shape measurement, NOT a verdict.",
            "root_lo": box_json(root_lo),
            "root_hi": box_json(root_hi),
            "splits": split_manifest,
            "leaves": leaf_manifest,
        });
        assert!(
            !ny_cert::rational::poisoned(),
            "refusing to write a screen tree after a rational arena fallback"
        );
        std::fs::write(
            format!("{out_dir}/screen_tree.json"),
            ny_cert::json::to_string_pretty(&tree),
        )
        .unwrap();
        println!(
            "CONJ_SCREEN_ONLY leaves={} splits={} explored={} \
             load_us={load_us} solve_us={solve_us} cover_ok={cover_ok} tight={tight} \
             (f64 screen only: no certificates, not a verdict)",
            leaves.len(),
            splits.len(),
            explored,
        );
        std::process::exit(0);
    }

    // Certified emit: the per-leaf Farkas certs were self-checked and written
    // by the workers as each leaf closed.
    let tree = ny_cert::json!({
        "type": "conjunctive_unsat_tree",
        "version": "1.0",
        "unsafe_atoms": targets.iter().map(|t| t.label.clone()).collect::<Vec<_>>(),
        "original_unsafe_rows": original_unsafe_rows,
        "target_authority": target_authority,
        "semantics": "unsafe region = AND of original_unsafe_rows; each leaf refutes ONE exactly reconstructed target on its box. target_authority carries a checked non-negative source-row entailment for every pure or conic target; the leaf Farkas is composition-dependent on that linkage. The split tree partitions the root box (re-checked structurally)",
        "root_lo": box_json(root_lo),
        "root_hi": box_json(root_hi),
        "splits": split_manifest,
        "leaves": leaf_manifest,
    });
    assert!(
        !ny_cert::rational::poisoned(),
        "refusing to write a certificate tree after a rational arena fallback"
    );
    std::fs::write(
        format!("{out_dir}/tree.json"),
        ny_cert::json::to_string_pretty(&tree),
    )
    .unwrap();

    println!(
        "CONJ_RESULT decided=true leaves={} splits={} explored={} exact_calls={} \
         load_us={load_us} solve_us={solve_us} exact_us={exact_us} selfcheck_us={check_us} \
         max_cert_bits={max_bits} cover_ok={cover_ok} tight={tight}",
        leaves.len(),
        splits.len(),
        explored,
        exact_calls,
    );
    std::process::exit(0);
}

/// Largest bit-length of any numerator or denominator appearing in the emitted
/// entailment certificate (multipliers, premise coefficients/constants, and the
/// conclusion). Demonstrates the certificate genuinely exceeds the i128 wall.
fn cert_max_bits(cert: &EntailmentCertificate) -> u64 {
    let bits = |r: &Rat| {
        let (num, den) = r
            .checked_parts()
            .expect("refusing to inspect a poisoned rational arena");
        num.bits().max(den.bits())
    };
    let mut m = 0u64;
    for mu in &cert.multipliers {
        m = m.max(bits(mu));
    }
    for p in &cert.premises {
        m = m.max(bits(&p.constant));
        for c in p.coefficients.values() {
            m = m.max(bits(c));
        }
    }
    m = m.max(bits(&cert.conclusion.constant));
    for c in cert.conclusion.coefficients.values() {
        m = m.max(bits(c));
    }
    m
}

fn fmt_rat(r: &Rat) -> String {
    r.to_clean_string()
        .expect("refusing to format a poisoned rational arena")
}

/// Derive the snake-case system stem used for the emitted theorem names from the
/// Lean module: strips the `CersyveInstance_` prefix (if present) and converts
/// the remaining UpperCamel stem to snake_case
/// (`CersyveInstance_DoubleIntegrator` → `double_integrator`,
/// `CersyveInstance_Pendulum` → `pendulum`).
fn module_to_sys(module: &str) -> String {
    let stem = module.strip_prefix("CersyveInstance_").unwrap_or(module);
    let mut out = String::new();
    for (i, c) in stem.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// Approximate a (possibly huge) exact rational as `f64` for human-readable
/// diagnostics only. Never used in a certificate or a soundness decision.
fn rat_to_f64(r: &Rat) -> f64 {
    r.to_f64_approx()
}

// ---------------------------------------------------------------------------
// LEAN-INSTANCE EMITTER (`--emit-lean`).
//
// Transcribes the exact-rational flattened cersyve con+inv nets into a
// self-contained `Crownproof/<Module>.lean` `ClampedSystem` instance that
// applies the kernel-checked `safe_forever` induction. Same trust class as
// `lrat_to_lean`: this does NO reasoning, only exact syntactic transcription of
// the ℚ matrices the DAG loader already produced (parity-gated via
// `NYCERT_PARITY=1`); everything semantic is re-checked by the Lean kernel.
// ---------------------------------------------------------------------------

/// One exact rational as a Lean `ℚ` literal `(mkRat n d)` (denominator kept
/// positive by `Rat`'s `BigRational` normalization). Lossless.
fn rat_lean(r: &Rat) -> String {
    let (num, den) = r
        .checked_parts()
        .expect("refusing to emit a poisoned rational arena");
    format!("(mkRat ({num}) ({den}))")
}

/// Emit a flattened `LoadedNet` as a Lean `Net` literal (hidden ReLU layers +
/// linear read-out). `w` is `[out][in]`, so unit `o` is `(row_o, b_o)` with
/// `row_o = W[o][·]` the input weights and `b_o` the bias; the Lean `Net.eval`
/// computes `bias + dot row x` per unit — the same forward pass.
fn emit_net_lean(name: &str, net: &LoadedNet, out: &mut String) {
    use std::fmt::Write as _;
    let emit_layer = |layer: &AffineLayer, out: &mut String| {
        out.push_str("    [ ");
        for (o, (row, b)) in layer.w.iter().zip(&layer.b).enumerate() {
            if o > 0 {
                out.push_str("\n    , ");
            }
            let cells: Vec<String> = row.iter().map(rat_lean).collect();
            let _ = write!(out, "([{}], {})", cells.join(", "), rat_lean(b));
        }
        out.push_str(" ]");
    };
    let _ = writeln!(out, "noncomputable def {name} : Net :=");
    out.push_str("  { hidden :=\n");
    out.push_str("  [\n");
    for (li, layer) in net.hidden.iter().enumerate() {
        if li > 0 {
            out.push_str("  ,\n");
        }
        emit_layer(layer, out);
        out.push('\n');
    }
    out.push_str("  ]\n");
    out.push_str("  , readout :=\n");
    emit_layer(&net.readout, out);
    out.push_str(" }\n\n");
}

/// Parse the vnnlib input box + output atoms (mirrors `main`'s inline logic).
fn emit_parse_box(vtext: &str) -> (Vec<Rat>, Vec<Rat>, Vec<VnnAtom>) {
    let atoms = parse_vnnlib(vtext);
    let dim = vnnlib_input_dim(&atoms).unwrap_or_else(|e| {
        eprintln!("PARSE_FAILED: {e}");
        std::process::exit(2);
    });
    // `vnnlib_input_dim` already refuses a dimension past `MAX_LAYER_WIDTH`;
    // the allocation helpers check it again (failing closed, never
    // truncating) where the verifier can see the bound.
    let mut lo = zeros_rat(dim);
    let mut hi = zeros_rat(dim);
    let mut lo_set = falses(dim);
    let mut hi_set = falses(dim);
    let mut y = vec![];
    for a in &atoms {
        if a.var.starts_with("X_") {
            let Some(idx) = var_index(&a.var) else {
                continue;
            };
            if idx >= dim {
                continue;
            }
            let r = decimal_to_rat(&a.rhs_raw).expect("x threshold");
            if a.op == ">=" {
                lo[idx] = r;
                lo_set[idx] = true;
            } else {
                hi[idx] = r;
                hi_set[idx] = true;
            }
        } else {
            y.push(a.clone());
        }
    }
    for i in 0..dim {
        assert!(lo_set[i] && hi_set[i], "input X_{i} box incomplete");
    }
    (lo, hi, y)
}

/// Load + flatten one cersyve ONNX net against the shared root box lower bound.
fn emit_load_flat(onnx_path: &str, lo: &[Rat]) -> LoadedNet {
    let g = load_onnx_or_exit(onnx_path);
    let dag = load_dag(&g);
    assert_eq!(dag.input_dim, lo.len(), "ONNX input dim != vnnlib box dim");
    flatten_dag(&dag, lo)
}

#[allow(clippy::too_many_lines)]
fn emit_lean_instance(
    con_path: &str,
    inv_path: &str,
    vnnlib_path: &str,
    out_path: &str,
    module: &str,
) {
    use std::fmt::Write as _;
    // Validate module name (uppercase-initial Lean identifier).
    assert!(
        !module.is_empty()
            && module
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
            && module
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_uppercase()),
        "Module {module:?} must be [A-Z][A-Za-z0-9_]*"
    );
    let vtext = std::fs::read_to_string(vnnlib_path).expect("read vnnlib");
    let (lo, hi, y_atoms) = emit_parse_box(&vtext);
    let dim = lo.len();
    assert!(
        dim >= 1,
        "vnnlib input box must have at least one dimension"
    );

    // The cersyve unsafe region is  Y_0 <= t0  AND  Y_1 >= t1.  The clean
    // `V x ≤ 0` framing needs both thresholds = 0 (they are, for cersyve); fail
    // closed otherwise so a mistyped property never yields a misframed theorem.
    let t0 = y_atoms
        .iter()
        .find(|a| a.var == "Y_0" && a.op == "<=")
        .map(|a| decimal_to_rat(&a.rhs_raw).expect("Y_0 threshold"))
        .expect("no (<= Y_0 _) atom");
    let t1 = y_atoms
        .iter()
        .find(|a| a.var == "Y_1" && a.op == ">=")
        .map(|a| decimal_to_rat(&a.rhs_raw).expect("Y_1 threshold"))
        .expect("no (>= Y_1 _) atom");
    assert!(
        t0.is_zero() && t1.is_zero(),
        "emitter assumes zero output thresholds (Y_0<=0, Y_1>=0); got {t0:?},{t1:?}"
    );

    let con = emit_load_flat(con_path, &lo);
    let inv = emit_load_flat(inv_path, &lo);
    eprintln!(
        "[emit-lean] con widths={:?} inv widths={:?}",
        con.layers().map(|l| l.w.len()).collect::<Vec<_>>(),
        inv.layers().map(|l| l.w.len()).collect::<Vec<_>>()
    );

    // Snake-case system stem for theorem names / docstrings, derived from the
    // module (`CersyveInstance_Pendulum` → `pendulum`, `..._DoubleIntegrator` →
    // `double_integrator`), so the emitted theorem is `{sys}_safe_forever`.
    let sys = module_to_sys(module);
    // Human-readable box product for the docstring, e.g. `[-1,1]×[-1,1]×[-1,1]`.
    // (`lo`/`hi` are both `dim` long: `emit_parse_box` builds them pairwise.)
    let box_product: String = lo
        .iter()
        .zip(&hi)
        .map(|(l, h)| format!("[{},{}]", fmt_rat(l), fmt_rat(h)))
        .collect::<Vec<_>>()
        .join("×");
    // The Lean domain conjunction over every input coordinate (exact ℚ bounds).
    let domain_body: String = lo
        .iter()
        .zip(&hi)
        .enumerate()
        .map(|(i, (l, h))| {
            format!(
                "{} ≤ x.getD {} 0 ∧ x.getD {} 0 ≤ {}",
                rat_lean(l),
                i,
                i,
                rat_lean(h)
            )
        })
        .collect::<Vec<_>>()
        .join(" ∧\n  ");

    let mut out = String::new();
    let _ = write!(
        out,
        r#"/-
  {module}.lean — MACHINE-GENERATED by `ny-cert`'s `certify_onnx --emit-lean`.
  Do not edit by hand.

  A concrete `Crownproof.Cersyve.ClampedSystem` instance for the VNN-COMP 2025
  cersyve `{sys}` finetune pair, composed with the kernel-checked
  `safe_forever` induction (`NyProof/CersyveInduction.lean`).

  # What is CONCRETE here (Lean literals, kernel-checked)
  * `Ncon`, `Ninv` : the two ONNX graphs flattened to EXACT-RATIONAL affine+ReLU
    `Net`s by the `certify_onnx` DAG loader (Gemm/Relu/Add DAG → sequential
    ReLU net; stable-active passthroughs, no relaxation).  Emitted here as ℚ
    weight/bias literals `(mkRat n d)`.
  * `D`   := the vnnlib input box  {box_product} (state dimension {dim}).
  * `V x` := `(Ninv.eval x)[0]` — the shared value/certificate network output.
  * `Safe x` := `(Ncon.eval x)[1] < 0` — the `con` constraint margin negative.
  * `{sys}_safe_forever` — a SINGLE application of `safe_forever`,
    fully kernel-checked, yielding unbounded (∀ k) safety.

  # RESIDUAL TRUST (precisely, and ONLY, this — everything else is the kernel)
  1. TRANSCRIPTION.  The ℚ matrices `Ncon`/`Ninv` equal the ONNX initializers
     under `certify_onnx`'s lossless f32→ℚ (n/2^k) read + exact DAG flattening.
     Same trust class as `lrat_to_lean`: syntax only, PARITY-TESTABLE via
     `NYCERT_PARITY=1` (flattened == symbolic DAG exactly; both == an
     independent f32 ONNX interpreter within f32 rounding, measured max diff
     ~1.5e-7 across 26 points × 2 outputs, per net).
  2. `bind_step`.  The inv net's SECOND output equals `V (step x)` — i.e. the
     inv ONNX graph is `x ↦ (V x, V (clamp (f̂ x)))`, the value composed with the
     clamped learned dynamics.  The dynamics map `step : ℝ^{dim}→ℝ^{dim}` itself
     is NOT present in the con/inv files (it lives in a separate Cersyve.jl
     dynamics net), so it is carried ABSTRACTLY as a hypothesis-parameter here;
     `bind_step` is exactly the graph→transition binding this instance makes
     explicit.
  3. `bind_val`.  The con and inv nets share the value net V bit-for-bit
     (Cersyve.jl), so `(Ncon.eval x)[0] = V x`.
  4. `inv_cert` / `con_cert`.  The two BOX-UNIVERSAL one-step facts.  These are
     exactly what ny-cert's exact-rational Farkas branch-and-bound certifies: a
     COMPLETE BaB whose every leaf emits a Farkas certificate (premises = input
     box + exact affine layer eqs + ReLU envelopes + one refuted unsafe atom),
     each self-checked by ny-cert's in-tree mirror of Clean's verifier AND
     re-checked by Clean's external certificate verifier, with the leaf boxes
     re-walked to confirm they partition the vnnlib box (`cover_ok`).  Measured
     for `{sys}` con AND inv (exact-leaf counts settings-dependent) in
     `docs/MEASURED_CERSYVE_SAFE_FOREVER.md` v2.  They are passed here as named
     hypotheses; KERNEL-PORTING those certificates (decide over the exact BaB
     leaves) is the remaining step (B).
-/
import NyProof.CersyveInduction
import Mathlib.Tactic.Linarith

namespace Crownproof

namespace {module}

open Crownproof.Cersyve

-- The two flattened nets are large exact-ℚ list literals; raise the elaboration
-- recursion limit so the kernel can typecheck them.
set_option maxRecDepth 100000

/-! ### Exact-rational affine+ReLU network evaluation (self-contained). -/

/-- Exact rational dot product `Σ rowᵢ · xᵢ`. -/
def dot (row x : List ℚ) : ℚ :=
  (List.zipWith (fun a b => a * b) row x).foldl (fun s t => s + t) 0

/-- One affine layer: a list of `(weight-row, bias)` units. -/
def affine (layer : List (List ℚ × ℚ)) (x : List ℚ) : List ℚ :=
  layer.map (fun ub => ub.2 + dot ub.1 x)

/-- Scalar ReLU. -/
def reluScalar (q : ℚ) : ℚ := if 0 ≤ q then q else 0

/-- Affine layer followed by elementwise ReLU. -/
def reluLayer (layer : List (List ℚ × ℚ)) (x : List ℚ) : List ℚ :=
  (affine layer x).map reluScalar

/-- Apply a stack of hidden ReLU layers. -/
def evalHidden : List (List (List ℚ × ℚ)) → List ℚ → List ℚ
  | [], x => x
  | layer :: rest, x => evalHidden rest (reluLayer layer x)

/-- A sequential ReLU network: hidden ReLU layers then a linear read-out. -/
structure Net where
  hidden : List (List (List ℚ × ℚ))
  readout : List (List ℚ × ℚ)

/-- Forward pass. -/
def Net.eval (N : Net) (x : List ℚ) : List ℚ :=
  affine N.readout (evalHidden N.hidden x)

/-! ### The two flattened cersyve networks (exact ℚ literals). -/

"#
    );

    emit_net_lean("Ncon", &con, &mut out);
    emit_net_lean("Ninv", &inv, &mut out);

    let _ = write!(
        out,
        r#"/-! ### `ClampedSystem` data: domain, value, safety. -/

/-- Operating domain: the vnnlib input box `{box_product}`. -/
def D (x : List ℚ) : Prop :=
  {domain_body}

/-- The value/certificate network: the shared first output. -/
noncomputable def V (x : List ℚ) : ℚ := (Ninv.eval x).getD 0 0

/-- The safety predicate: the `con` constraint margin (second output) negative. -/
def Safe (x : List ℚ) : Prop := (Ncon.eval x).getD 1 0 < 0

/-! ### SAFE FOREVER for the Lean-modeled {sys} system.

Parameters (residual trust — see the file header):
* `step`, `step_mem`  — the abstract clamped dynamics ℝ^{dim}→ℝ^{dim} (not in the
  con/inv files) and the box-clamp membership fact.
* `bind_step`         — inv net's 2nd output = `V (step x)` (graph composition).
* `bind_val`          — con/inv share the value net.
* `inv_cert`,`con_cert` — the Clean-verified exact-rational Farkas BaB one-step
  certs (box-universal); kernel-porting them is the remaining step (B).

Conclusion: started in the certified region `S = {{x ∈ D : V x ≤ 0}}`, the
closed loop is `Safe` at EVERY time step — the unbounded claim, one application
of the kernel-checked `safe_forever`. -/
theorem {sys}_safe_forever
    (step : List ℚ → List ℚ)
    (step_mem : ∀ x, D x → D (step x))
    (bind_step : ∀ x, (Ninv.eval x).getD 1 0 = V (step x))
    (bind_val : ∀ x, (Ncon.eval x).getD 0 0 = V x)
    (inv_cert : ∀ x, D x → (Ninv.eval x).getD 0 0 ≤ 0 → (Ninv.eval x).getD 1 0 < 0)
    (con_cert : ∀ x, D x → (Ncon.eval x).getD 0 0 ≤ 0 → (Ncon.eval x).getD 1 0 < 0)
    (x₀ : List ℚ) (hx₀D : D x₀) (hx₀V : V x₀ ≤ 0) :
    ∀ k : ℕ, Safe (trajectory ⟨D, step, step_mem, V, Safe⟩ x₀ k) := by
  -- `V x` is `(Ninv.eval x).getD 0 0` and `Safe x` is `(Ncon.eval x).getD 1 0 < 0`
  -- definitionally, so the two certs discharge `hinv`/`hcon` directly.
  have hinv : ∀ x, D x → V x ≤ 0 → V (step x) ≤ 0 := by
    intro x hx hV
    have h : (Ninv.eval x).getD 1 0 < 0 := inv_cert x hx hV
    rw [← bind_step x]
    exact h.le
  have hcon : ∀ x, D x → V x ≤ 0 → Safe x := by
    intro x hx hV
    have hpre : (Ncon.eval x).getD 0 0 ≤ 0 := by rw [bind_val x]; exact hV
    exact con_cert x hx hpre
  exact safe_at_every_step ⟨D, step, step_mem, V, Safe⟩ hinv hcon x₀ hx₀D hx₀V

/-- The full invariant form: in `D`, in the sublevel set, and `Safe`, forever. -/
theorem {sys}_safe_forever_full
    (step : List ℚ → List ℚ)
    (step_mem : ∀ x, D x → D (step x))
    (bind_step : ∀ x, (Ninv.eval x).getD 1 0 = V (step x))
    (bind_val : ∀ x, (Ncon.eval x).getD 0 0 = V x)
    (inv_cert : ∀ x, D x → (Ninv.eval x).getD 0 0 ≤ 0 → (Ninv.eval x).getD 1 0 < 0)
    (con_cert : ∀ x, D x → (Ncon.eval x).getD 0 0 ≤ 0 → (Ncon.eval x).getD 1 0 < 0)
    (x₀ : List ℚ) (hx₀D : D x₀) (hx₀V : V x₀ ≤ 0) :
    ∀ k : ℕ,
      D (trajectory ⟨D, step, step_mem, V, Safe⟩ x₀ k) ∧
      V (trajectory ⟨D, step, step_mem, V, Safe⟩ x₀ k) ≤ 0 ∧
      Safe (trajectory ⟨D, step, step_mem, V, Safe⟩ x₀ k) := by
  have hinv : ∀ x, D x → V x ≤ 0 → V (step x) ≤ 0 := by
    intro x hx hV
    have h : (Ninv.eval x).getD 1 0 < 0 := inv_cert x hx hV
    rw [← bind_step x]
    exact h.le
  have hcon : ∀ x, D x → V x ≤ 0 → Safe x := by
    intro x hx hV
    have hpre : (Ncon.eval x).getD 0 0 ≤ 0 := by rw [bind_val x]; exact hV
    exact con_cert x hx hpre
  exact safe_forever ⟨D, step, step_mem, V, Safe⟩ hinv hcon x₀ hx₀D hx₀V

end {module}

end Crownproof

/-! ## Trust-base check — must reduce to the standard axioms only. -/

#print axioms Crownproof.{module}.{sys}_safe_forever
#print axioms Crownproof.{module}.{sys}_safe_forever_full
"#
    );

    assert!(
        !ny_cert::rational::poisoned(),
        "refusing to write Lean after a rational arena fallback"
    );
    std::fs::write(out_path, &out).expect("write lean file");
    eprintln!(
        "[emit-lean] wrote {out_path} ({} bytes, module {module})",
        out.len()
    );
}

// ---------------------------------------------------------------------------
// Unit tests for the DAG loader + exact flattening algebra (always run; no
// benchmark files needed — graphs are built in memory).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod dag_tests {
    use super::*;

    fn tensor(name: &str, dims: &[i64], floats: &[f32]) -> OnnxTensor {
        OnnxTensor {
            name: name.to_string(),
            dims: dims.to_vec(),
            floats: floats.to_vec(),
        }
    }

    fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> OnnxNode {
        OnnxNode {
            op_type: op.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attrs: vec![],
        }
    }

    /// Miniature cersyve-shaped DAG (residual Add merges, two branches of
    /// different ReLU depth):
    ///   h1 = ReLU(x W0 + b0)            (2 -> 3)
    ///   h2 = ReLU(h1 W1 + b1)           (3 -> 2)
    ///   headA = h2 W2 + b2              (2 -> 2)
    ///   g  = ReLU(x W4 + b4)            (2 -> 1)
    ///   s  = x W5 + b5                  (2 -> 1)
    ///   a0 = g + s
    ///   headB = a0 W6 + b6              (1 -> 2)
    ///   out = headA + headB
    fn mini_dag() -> OnnxGraph {
        let mut inits = BTreeMap::new();
        inits.insert(
            "W0".into(),
            tensor("W0", &[2, 3], &[0.5, -1.25, 2.0, 1.0, 0.75, -0.5]),
        );
        inits.insert("b0".into(), tensor("b0", &[3], &[0.125, -0.25, 0.5]));
        inits.insert(
            "W1".into(),
            tensor("W1", &[3, 2], &[1.0, -0.5, 0.25, 2.0, -1.0, 0.5]),
        );
        inits.insert("b1".into(), tensor("b1", &[2], &[-0.125, 0.25]));
        inits.insert("W2".into(), tensor("W2", &[2, 2], &[1.5, -0.75, 0.5, 1.0]));
        inits.insert("b2".into(), tensor("b2", &[2], &[0.0, -1.0]));
        inits.insert("W4".into(), tensor("W4", &[2, 1], &[-0.5, 1.25]));
        inits.insert("b4".into(), tensor("b4", &[1], &[0.25]));
        inits.insert("W5".into(), tensor("W5", &[2, 1], &[0.75, -0.25]));
        inits.insert("b5".into(), tensor("b5", &[1], &[-0.5]));
        inits.insert("W6".into(), tensor("W6", &[1, 2], &[2.0, -1.5]));
        inits.insert("b6".into(), tensor("b6", &[2], &[0.5, 0.25]));
        OnnxGraph {
            nodes: vec![
                node("Gemm", &["x", "W0", "b0"], &["z0"]),
                node("Relu", &["z0"], &["h1"]),
                node("Gemm", &["h1", "W1", "b1"], &["z1"]),
                node("Relu", &["z1"], &["h2"]),
                node("Gemm", &["h2", "W2", "b2"], &["headA"]),
                node("Gemm", &["x", "W4", "b4"], &["z4"]),
                node("Relu", &["z4"], &["g"]),
                node("Gemm", &["x", "W5", "b5"], &["s"]),
                node("Add", &["g", "s"], &["a0"]),
                node("Gemm", &["a0", "W6", "b6"], &["headB"]),
                node("Add", &["headA", "headB"], &["out"]),
            ],
            inits,
            graph_inputs: vec!["x".into()],
            graph_outputs: vec!["out".into()],
        }
    }

    /// Reference forward pass computed directly in exact rationals from the
    /// f32 weights (independent of the Aff algebra).
    fn reference_forward(x0: Rat, x1: Rat) -> Vec<Rat> {
        let r = |v: f32| f32_to_rat(v).unwrap();
        let relu = |v: Rat| if v.is_positive() { v } else { Rat::ZERO };
        let dot2 = |a: Rat, b: Rat, wa: f32, wb: f32, bias: f32| {
            a.mul(r(wa))
                .unwrap()
                .add(b.mul(r(wb)).unwrap())
                .unwrap()
                .add(r(bias))
                .unwrap()
        };
        // h1 (W0 is [2,3] row-major [in][out])
        let h1: Vec<Rat> = (0..3)
            .map(|o| {
                relu(dot2(
                    x0,
                    x1,
                    [0.5, -1.25, 2.0][o],
                    [1.0, 0.75, -0.5][o],
                    [0.125, -0.25, 0.5][o],
                ))
            })
            .collect();
        // h2 (W1 [3,2])
        let w1 = [[1.0, -0.5], [0.25, 2.0], [-1.0, 0.5]];
        let b1 = [-0.125, 0.25];
        let h2: Vec<Rat> = (0..2)
            .map(|o| {
                let mut acc = r(b1[o]);
                for (i, h) in h1.iter().enumerate() {
                    acc = acc.add(h.mul(r(w1[i][o])).unwrap()).unwrap();
                }
                relu(acc)
            })
            .collect();
        // headA (W2 [2,2])
        let w2 = [[1.5, -0.75], [0.5, 1.0]];
        let b2 = [0.0, -1.0];
        let head_a: Vec<Rat> = (0..2)
            .map(|o| {
                let mut acc = r(b2[o]);
                for (i, h) in h2.iter().enumerate() {
                    acc = acc.add(h.mul(r(w2[i][o])).unwrap()).unwrap();
                }
                acc
            })
            .collect();
        let g = relu(dot2(x0, x1, -0.5, 1.25, 0.25));
        let s = dot2(x0, x1, 0.75, -0.25, -0.5);
        let a0 = g.add(s).unwrap();
        let head_b: Vec<Rat> = (0..2)
            .map(|o| {
                a0.mul(r([2.0, -1.5][o]))
                    .unwrap()
                    .add(r([0.5, 0.25][o]))
                    .unwrap()
            })
            .collect();
        (0..2).map(|o| head_a[o].add(head_b[o]).unwrap()).collect()
    }

    #[test]
    fn mix_set_parser_rejects_non_conic_and_zero_authority() {
        for raw in ["-1:1", "1:-1", "0:0", "", "1", "x:1", "1:x"] {
            assert!(parse_mix_set(raw).is_err(), "raw={raw:?}");
        }
        assert_eq!(
            parse_mix_set("1:1, 2:0, 0:3").unwrap(),
            vec![
                MixPair { left: 1, right: 1 },
                MixPair { left: 2, right: 0 },
                MixPair { left: 0, right: 3 },
            ]
        );
    }

    fn authority_source_rows() -> Vec<UnsafeSourceRow> {
        vec![
            UnsafeSourceRow {
                id: UnsafeSourceRowId(0),
                label: "Y_0<=5".to_string(),
                original_constraint: LinearConstraint::with_kind(
                    ConstraintKind::Le,
                    &[("Y_0", Rat::ONE)],
                    Rat::from_int(5),
                ),
                normalized_constraint: LinearConstraint::with_kind(
                    ConstraintKind::Le,
                    &[("Y_0", Rat::ONE)],
                    Rat::from_int(5),
                ),
                out_w: vec![Rat::ONE, Rat::from_int(2)],
                out_b: Rat::from_int(3),
                u_bound: Rat::from_int(5),
            },
            UnsafeSourceRow {
                id: UnsafeSourceRowId(1),
                label: "Y_1>=7".to_string(),
                original_constraint: LinearConstraint::with_kind(
                    ConstraintKind::Ge,
                    &[("Y_1", Rat::ONE)],
                    Rat::from_int(7),
                ),
                normalized_constraint: LinearConstraint::with_kind(
                    ConstraintKind::Le,
                    &[("Y_1", Rat::from_int(-1))],
                    Rat::from_int(-7),
                ),
                out_w: vec![Rat::from_int(-3), Rat::from_int(-4)],
                out_b: Rat::from_int(-6),
                u_bound: Rat::from_int(-7),
            },
        ]
    }

    #[test]
    fn trusted_conic_target_reconstructs_exact_rows_and_rejects_forgery() {
        let rows = authority_source_rows();
        let provenance = ConicProvenance {
            terms: vec![
                ConicSourceTerm {
                    source_row: UnsafeSourceRowId(0),
                    multiplier: Rat::from_int(2),
                },
                ConicSourceTerm {
                    source_row: UnsafeSourceRowId(1),
                    multiplier: Rat::ONE,
                },
            ],
        };
        let target = trusted_conj_target(
            "2*row0 + row1".to_string(),
            vec![Rat::from_int(-1), Rat::ZERO],
            Rat::ZERO,
            Rat::from_int(3),
            provenance.clone(),
            &rows,
        )
        .expect("exact conic reconstruction");
        assert_eq!(target.provenance, provenance);
        assert!(check_entailment(&target.source_entailment).is_ok());

        assert!(trusted_conj_target(
            "forged bound".to_string(),
            vec![Rat::from_int(-1), Rat::ZERO],
            Rat::ZERO,
            Rat::from_int(4),
            provenance,
            &rows,
        )
        .is_err());
    }

    #[test]
    fn trusted_conic_target_rejects_negative_and_all_zero_provenance() {
        let rows = authority_source_rows();
        for multipliers in [[-1, 1], [0, 0]] {
            let provenance = ConicProvenance {
                terms: multipliers
                    .into_iter()
                    .enumerate()
                    .map(|(source_row, multiplier)| ConicSourceTerm {
                        source_row: UnsafeSourceRowId(source_row),
                        multiplier: Rat::from_int(multiplier),
                    })
                    .collect(),
            };
            assert!(trusted_conj_target(
                "invalid".to_string(),
                vec![Rat::ZERO, Rat::ZERO],
                Rat::ZERO,
                Rat::ZERO,
                provenance,
                &rows,
            )
            .is_err());
        }
    }

    #[test]
    fn dag_loader_and_flatten_match_reference_exactly() {
        let g = mini_dag();
        let dag = load_dag(&g);
        assert_eq!(dag.input_dim, 2);
        assert_eq!(dag.relu_pre.len(), 3);
        let lo = vec![Rat::new(-1, 1).unwrap(); 2];
        let flat = flatten_dag(&dag, &lo);
        // depth 2: level(h1)=1, level(g)=1, level(h2)=2; readout reads h2, g, x.
        assert_eq!(flat.hidden.len(), 2);
        // Points inside the box [-1,1]^2 (rational, incl. non-dyadic).
        let pts = [
            (Rat::new(0, 1).unwrap(), Rat::new(0, 1).unwrap()),
            (Rat::new(1, 3).unwrap(), Rat::new(-2, 3).unwrap()),
            (Rat::new(-1, 1).unwrap(), Rat::new(1, 1).unwrap()),
            (Rat::new(7, 8).unwrap(), Rat::new(-5, 16).unwrap()),
            (Rat::new(-99, 100).unwrap(), Rat::new(99, 100).unwrap()),
        ];
        for (x0, x1) in pts {
            let x = vec![x0, x1];
            let want = reference_forward(x0, x1);
            assert_eq!(dag_eval_exact(&dag, &x), want, "DAG eval != reference");
            assert_eq!(
                loaded_eval_vec(&flat, &x),
                want,
                "flatten eval != reference"
            );
        }
    }

    #[test]
    fn flatten_matches_dag_on_sub_box_of_root() {
        // The passthrough shift uses the ROOT box lower bound; the flattened net
        // must stay exact on any point of any SUB-box (x >= root lo).
        let g = mini_dag();
        let dag = load_dag(&g);
        let lo = vec![Rat::new(-2, 1).unwrap(); 2];
        let flat = flatten_dag(&dag, &lo);
        let x = vec![Rat::new(-2, 1).unwrap(), Rat::new(2, 1).unwrap()];
        assert_eq!(dag_eval_exact(&dag, &x), loaded_eval_vec(&flat, &x));
    }

    #[test]
    fn f32_forward_tracks_exact_forward() {
        let g = mini_dag();
        let dag = load_dag(&g);
        let xf = [0.375f32, -0.625f32]; // f32-exact dyadics
        let x: Vec<Rat> = xf.iter().map(|&v| f32_to_rat(v).unwrap()).collect();
        let exact = dag_eval_exact(&dag, &x);
        let approx = f32_forward(&g, &xf);
        for (e, a) in exact.iter().zip(&approx) {
            assert!((rat_to_f64(e) - f64::from(*a)).abs() < 1e-4);
        }
    }

    #[test]
    #[should_panic(expected = "unsupported Gemm attribute")]
    fn fails_closed_on_transb() {
        let mut g = mini_dag();
        g.nodes[0].attrs.push(OnnxAttr {
            name: "transB".into(),
            i: Some(1),
            f: None,
        });
        let _ = load_dag(&g);
    }

    #[test]
    #[should_panic(expected = "unsupported op")]
    fn fails_closed_on_unknown_op() {
        let mut g = mini_dag();
        g.nodes.push(node("Mul", &["out", "out"], &["out2"]));
        g.graph_outputs = vec!["out2".into()];
        let _ = load_dag(&g);
    }

    #[test]
    #[should_panic(expected = "initializer operand")]
    fn fails_closed_on_const_add() {
        let mut g = mini_dag();
        g.nodes.push(node("Add", &["out", "b6"], &["out2"]));
        g.graph_outputs = vec!["out2".into()];
        let _ = load_dag(&g);
    }
}

#[cfg(test)]
mod onnx_wire_tests {
    use super::*;

    fn varint(mut v: u64, out: &mut Vec<u8>) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }

    /// One length-delimited field (wire type 2).
    fn ld(field: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        varint((field << 3) | 2, &mut out);
        varint(payload.len() as u64, &mut out);
        out.extend_from_slice(payload);
        out
    }

    /// ModelProto{ graph: GraphProto{ node Relu(x)->y, init w=[1.5, -2],
    /// input x, output y } }.
    fn model() -> Vec<u8> {
        let mut node = ld(1, b"x");
        node.extend(ld(2, b"y"));
        node.extend(ld(4, b"Relu"));
        let mut tensor = ld(1, &[2]); // packed dims [2]
        tensor.extend(ld(8, b"w"));
        let mut raw = 1.5f32.to_le_bytes().to_vec();
        raw.extend((-2.0f32).to_le_bytes());
        tensor.extend(ld(9, &raw));
        let mut graph = ld(1, &node);
        graph.extend(ld(5, &tensor));
        graph.extend(ld(11, &ld(1, b"x")));
        graph.extend(ld(12, &ld(1, b"y")));
        let mut m = vec![0x08, 0x07]; // ir_version = 7 (skipped varint field)
        m.extend(ld(7, &graph));
        m
    }

    #[test]
    fn well_formed_model_decodes() {
        let g = parse_onnx(&model()).expect("well-formed model");
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.nodes[0].op_type, "Relu");
        assert_eq!(g.nodes[0].inputs, vec!["x".to_string()]);
        let w = g.inits.get("w").expect("initializer w");
        assert_eq!(w.dims, vec![2]);
        assert_eq!(w.floats, vec![1.5, -2.0]);
        assert_eq!(g.graph_inputs, vec!["x".to_string()]);
        assert_eq!(g.graph_outputs, vec!["y".to_string()]);
    }

    #[test]
    fn every_truncation_is_an_error_not_a_panic() {
        // The old indexing reader panicked (index/slice out of range) on these.
        let m = model();
        for cut in 1..m.len() {
            assert!(parse_onnx(&m[..cut]).is_err(), "prefix of {cut} bytes");
        }
    }

    #[test]
    fn missing_graph_and_bad_wire_type_are_errors() {
        assert!(matches!(parse_onnx(&[0x08, 0x07]), Err(OnnxError::NoGraph)));
        // field 1, wire type 3 (start-group): the old reader `panic!`ed.
        assert!(matches!(
            parse_onnx(&[0x0b]),
            Err(OnnxError::Wire(WireError::UnsupportedWireType {
                wire_type: 3
            }))
        ));
    }
}

#[cfg(test)]
mod input_bound_tests {
    use super::*;

    fn atoms(text: &str) -> Vec<VnnAtom> {
        parse_vnnlib(text)
    }

    #[test]
    fn vnnlib_input_dim_of_a_complete_box() {
        let a = atoms(
            "(assert (>= X_0 -1.0))(assert (<= X_0 1.0))\
             (assert (>= X_1 0.0))(assert (<= X_1 0.5))(assert (<= Y_0 0.0))",
        );
        assert_eq!(vnnlib_input_dim(&a), Ok(2));
    }

    #[test]
    fn vnnlib_max_index_is_an_error_not_an_overflow_panic() {
        // The old `.map(|i| i + 1)` overflowed on usize::MAX.
        let a = atoms("(assert (>= X_18446744073709551615 0.0))");
        assert!(vnnlib_input_dim(&a).unwrap_err().contains("out of range"));
    }

    #[test]
    fn vnnlib_huge_index_is_rejected_before_allocating_the_box() {
        // The old code sized `vec![Rat::ZERO; dim]` by this index (10^11).
        let a = atoms("(assert (>= X_100000000000 0.0))(assert (<= X_100000000000 1.0))");
        assert!(vnnlib_input_dim(&a).is_err());
    }

    #[test]
    fn vnnlib_without_x_atoms_is_an_error() {
        let a = atoms("(assert (<= Y_0 0.0))");
        assert_eq!(
            vnnlib_input_dim(&a),
            Err("vnnlib has no X_i atoms".to_string())
        );
    }

    #[test]
    fn cut2_corner_bound_matches_enumeration_in_range() {
        // z_i = x0 + x1, z_j = x0 - x1 on [-1,1]^2: max relu(z_i)+relu(z_j) = 2.
        let w1 = vec![vec![1.0, 1.0], vec![1.0, -1.0]];
        let b1 = vec![0.0, 0.0];
        let (lo, hi) = (vec![-1.0, -1.0], vec![1.0, 1.0]);
        assert_eq!(cut2_box_b_f64(&w1, &b1, &lo, &hi, 0, 1), 2.0);
    }

    #[test]
    fn cut2_past_corner_cap_yields_no_cut_instead_of_a_shift_panic() {
        // 40 inputs: the old `1u32 << n` panicked (and 2^n corners is infeasible).
        let n = 40;
        let w1 = vec![vec![1.0; n], vec![-1.0; n]];
        let b1 = vec![0.0, 0.0];
        let (lo, hi) = (vec![-1.0; n], vec![1.0; n]);
        assert_eq!(cut2_box_b_f64(&w1, &b1, &lo, &hi, 0, 1), f64::INFINITY);
        // ... so discover_cuts2 emits no cut for the (unstable) pair.
        let (pl, ph) = (vec![-1.0, -1.0], vec![1.0, 1.0]);
        assert!(discover_cuts2(&w1, &b1, &lo, &hi, &pl, &ph, 8).is_empty());
    }

    #[test]
    fn matrix_size_with_overflowing_dims_is_a_size_mismatch() {
        let t = OnnxTensor {
            name: "w".into(),
            dims: vec![],
            floats: vec![1.0],
        };
        let r = std::panic::catch_unwind(|| rat_mat(&t, usize::MAX, 2));
        let msg = r.expect_err("must reject");
        let msg = msg.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(msg.contains("matrix size mismatch"), "{msg}");
    }
}

/// The index-free rewrites of the loaders and the f64 bound passes compute
/// what the index arithmetic computed, and keep its fail-closed shape checks.
#[cfg(test)]
mod index_free_tests {
    use super::*;

    fn t(dims: &[i64], floats: &[f32]) -> OnnxTensor {
        OnnxTensor {
            name: "w".into(),
            dims: dims.to_vec(),
            floats: floats.to_vec(),
        }
    }

    fn r(n: i128, d: i128) -> Rat {
        Rat::new(n, d).unwrap()
    }

    #[test]
    fn rat_mat_transposes_like_the_index_formula() {
        // stored [rows=2][cols=3] row-major; out[c][r] = floats[r*3 + c].
        let floats = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = rat_mat(&t(&[2, 3], &floats), 2, 3).unwrap();
        for (c, col) in out.iter().enumerate() {
            for (rr, v) in col.iter().enumerate() {
                assert_eq!(*v, f32_to_rat(floats[rr * 3 + c]).unwrap());
            }
        }
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|col| col.len() == 2));
    }

    #[test]
    fn rat_mat_with_no_columns_is_empty() {
        assert_eq!(
            rat_mat(&t(&[4, 0], &[]), 4, 0).unwrap(),
            [] as [Vec<Rat>; 0]
        );
    }

    #[test]
    fn gemm_apply_matches_the_dense_formula() {
        // a = identity over a 2-wide input; W = [[1,2,3],[4,5,6]] ([kin=2][kout=3]).
        let a = Aff::identity(0, 2);
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias = vec![r(1, 2), Rat::ZERO, r(-1, 4)];
        let out = gemm_apply(&a, &w, 2, 3, &bias);
        assert_eq!(out.cst, bias);
        let m = &out.terms[&0];
        for (o, row) in m.iter().enumerate() {
            for (i, v) in row.iter().enumerate() {
                assert_eq!(*v, f32_to_rat(w[i * 3 + o]).unwrap(), "W^T[{o}][{i}]");
            }
        }
    }

    #[test]
    fn gemm_apply_with_no_outputs_is_empty() {
        let a = Aff::identity(0, 2);
        let out = gemm_apply(&a, &[], 2, 0, &[]);
        assert!(out.terms.is_empty() && out.cst.is_empty());
    }

    #[test]
    fn relu_terminated_net_gets_an_identity_readout() {
        // x (2) -> MatMul W [2,3] -> Relu: no trailing affine layer.
        let mut inits = BTreeMap::new();
        inits.insert(
            "W".to_string(),
            t(&[2, 3], &[1.0, 0.0, -1.0, 0.5, 2.0, 0.0]),
        );
        let g = OnnxGraph {
            nodes: vec![
                OnnxNode {
                    op_type: "MatMul".into(),
                    inputs: vec!["x".into(), "W".into()],
                    outputs: vec!["m".into()],
                    attrs: vec![],
                },
                OnnxNode {
                    op_type: "Relu".into(),
                    inputs: vec!["m".into()],
                    outputs: vec!["h".into()],
                    attrs: vec![],
                },
            ],
            inits,
            graph_inputs: vec!["x".into()],
            graph_outputs: vec!["h".into()],
        };
        let net = load_net(&g);
        assert_eq!(net.hidden.len(), 1);
        assert_eq!(net.layers().count(), 2);
        assert_eq!(net.readout.b, vec![Rat::ZERO; 3]);
        for (d, row) in net.readout.w.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                assert_eq!(*v, if c == d { Rat::ONE } else { Rat::ZERO });
            }
        }
        // y = relu(W^T x): x = (1, 2) -> (1 + 1, 0 + 4, -1 + 0) -> (2, 4, 0).
        let y = loaded_eval_vec(&net, &[Rat::ONE, Rat::from_int(2)]);
        assert_eq!(y, vec![Rat::from_int(2), Rat::from_int(4), Rat::ZERO]);
    }

    /// The pre-rewrite `crown_lower_f64` (index arithmetic), kept verbatim as
    /// the reference for the zipped version.
    #[allow(clippy::needless_range_loop)]
    fn crown_lower_f64_reference(
        weights: &[Vec<Vec<f64>>],
        biases: &[Vec<f64>],
        out_weight: &[f64],
        out_bias: f64,
        lo: &[f64],
        hi: &[f64],
    ) -> f64 {
        let k = weights.len();
        let n = lo.len();
        let mut pre_lo: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut pre_hi: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut act_lo = lo.to_vec();
        let mut act_hi = hi.to_vec();
        for li in 0..k {
            let mut zl = Vec::new();
            let mut zu = Vec::new();
            for (row, &bias) in weights[li].iter().zip(&biases[li]) {
                let (mut lmin, mut umax) = (bias, bias);
                for (i, &wji) in row.iter().enumerate() {
                    if wji >= 0.0 {
                        lmin += wji * act_lo[i];
                        umax += wji * act_hi[i];
                    } else {
                        lmin += wji * act_hi[i];
                        umax += wji * act_lo[i];
                    }
                }
                zl.push(lmin);
                zu.push(umax);
            }
            act_lo = zl.iter().map(|&v| v.max(0.0)).collect();
            act_hi = zu.iter().map(|&v| v.max(0.0)).collect();
            pre_lo.push(zl);
            pre_hi.push(zu);
        }
        let mut const_acc = -out_bias;
        let mut a_coeff = out_weight.to_vec();
        for li in (0..k).rev() {
            let width = weights[li].len();
            let mut z_coeff = vec![0.0f64; width];
            for j in 0..width {
                let c = a_coeff[j];
                let (p, q, r, t) = relu_env(pre_lo[li][j], pre_hi[li][j]);
                if c > 0.0 {
                    z_coeff[j] += c * p;
                    const_acc += c * (-q);
                } else if c < 0.0 {
                    z_coeff[j] += c * r;
                    const_acc += (-c) * t;
                }
            }
            let prev_dim = if li == 0 { n } else { weights[li - 1].len() };
            let mut prev_coeff = vec![0.0f64; prev_dim];
            for j in 0..width {
                let c = z_coeff[j];
                if c == 0.0 {
                    continue;
                }
                for (i, &wji) in weights[li][j].iter().enumerate() {
                    prev_coeff[i] += c * wji;
                }
                const_acc += -(c * biases[li][j]);
            }
            a_coeff = prev_coeff;
        }
        for i in 0..n {
            let d = a_coeff[i];
            if d > 0.0 {
                const_acc += -(d * lo[i]);
            } else if d < 0.0 {
                const_acc += (-d) * hi[i];
            }
        }
        -const_acc
    }

    fn net_2_3_2() -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
        (
            vec![
                vec![vec![1.0, -0.5], vec![0.25, 2.0], vec![-1.0, 0.75]],
                vec![vec![0.5, -1.0, 1.5], vec![-0.25, 0.5, 1.0]],
            ],
            vec![vec![0.125, -0.25, 0.5], vec![-0.5, 0.25]],
        )
    }

    #[test]
    fn crown_lower_f64_matches_the_index_reference_bit_for_bit() {
        let (w, b) = net_2_3_2();
        let ow = [1.0, -2.0];
        for (lo, hi) in [
            ([-1.0, -1.0], [1.0, 1.0]),
            ([0.0, -0.5], [0.5, 0.25]),
            ([-2.0, 1.0], [-1.0, 3.0]),
        ] {
            let got = crown_lower_f64(&w, &b, &ow, 0.375, &lo, &hi);
            let want = crown_lower_f64_reference(&w, &b, &ow, 0.375, &lo, &hi);
            assert_eq!(got.to_bits(), want.to_bits(), "box {lo:?}..{hi:?}");
        }
    }

    #[test]
    fn tight_and_cut_bounds_are_no_looser_than_ibp_crown() {
        // Tight CROWN intermediate bounds intersect IBP, and the cut pass with
        // no cuts is the tight pass's backward pass: same or better bounds.
        let (w, b) = net_2_3_2();
        let ow = [1.0, -2.0];
        let (lo, hi) = ([-1.0, -1.0], [1.0, 1.0]);
        let ibp = crown_lower_f64(&w, &b, &ow, 0.0, &lo, &hi);
        let tight = crown_lower_f64_tight(&w, &b, &ow, 0.0, &lo, &hi);
        let cut0 = crown_lower_f64_tight_cut(&w, &b, &ow, 0.0, &lo, &hi, &[]);
        assert!(tight >= ibp - 1e-12, "tight {tight} < ibp {ibp}");
        assert!(
            (cut0 - tight).abs() <= 1e-12,
            "no-cut pass {cut0} != tight {tight}"
        );
    }

    #[test]
    #[should_panic(expected = "weight row is not 2 wide")]
    fn ragged_f64_net_fails_closed() {
        let (mut w, b) = net_2_3_2();
        w[0][1].pop();
        let _ = crown_intermediate_bounds_f64(&w, &b, &[0.0, 0.0], &[1.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "no hidden layer")]
    fn tight_bound_without_hidden_layers_fails_closed() {
        let _ = crown_lower_f64_tight(&[], &[], &[1.0], 0.0, &[0.0], &[1.0]);
    }
}

#[cfg(test)]
mod input_width_tests {
    use super::*;

    #[test]
    fn vnnlib_box_wider_than_2_pow_20_is_still_accepted() {
        // A complete box over 2^20 + 1 inputs. A first cut of the width cap
        // sat at 2^20 and refused this valid (if large) property, which the
        // unbounded loader before it accepted; the cap is now far above it.
        let n = (1usize << 20) + 1;
        let mut text = String::new();
        for i in 0..n {
            text.push_str(&format!("(assert (>= X_{i} 0))(assert (<= X_{i} 1))"));
        }
        assert_eq!(vnnlib_input_dim(&parse_vnnlib(&text)), Ok(n));
    }
}
