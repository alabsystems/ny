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
#[derive(Debug, Clone)]
struct OnnxNode {
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    /// Integer-list attributes we care about for Conv: `kernel_shape`, `strides`,
    /// `pads`, `dilations`, `group`. A single-int attr (`i`) is stored as a
    /// length-1 vec. Non-integer attrs are ignored (never needed for the affine
    /// lowering — a Conv is an exact linear map determined entirely by these).
    attr_ints: BTreeMap<String, Vec<i64>>,
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
    /// Declared shapes of graph inputs (ValueInfoProto), keyed by name. Used to
    /// size the first Conv (needs Cin×Hin×Win) and to seed shape tracking. A
    /// leading batch dim of 1 (or an unknown/`0` dim) is kept as-is here and
    /// stripped by the loader.
    input_shapes: BTreeMap<String, Vec<i64>>,
}

/// Parse a ValueInfoProto -> (name, shape). ValueInfoProto: name=1(str),
/// type=2(TypeProto). TypeProto: tensor_type=1. Tensor: elem_type=1, shape=2.
/// TensorShapeProto: dim=1(Dimension). Dimension: dim_value=1(int64). Unknown
/// (symbolic) dims parse as 0.
fn parse_value_info(buf: &[u8]) -> Result<Option<(String, Vec<i64>)>, WireError> {
    let mut pb = Pb::new(buf);
    let mut name = String::new();
    let mut shape: Option<Vec<i64>> = None;
    while !pb.eof() {
        let (fn_, wt) = pb.tag()?;
        match (fn_, wt) {
            (1, 2) => name = String::from_utf8_lossy(pb.bytes()?).into_owned(),
            (2, 2) => shape = parse_type_proto(pb.bytes()?)?,
            _ => pb.skip(wt)?,
        }
    }
    if name.is_empty() {
        return Ok(None);
    }
    Ok(Some((name, shape.unwrap_or_default())))
}

fn parse_type_proto(buf: &[u8]) -> Result<Option<Vec<i64>>, WireError> {
    let mut pb = Pb::new(buf);
    let mut shape = None;
    while !pb.eof() {
        let (fn_, wt) = pb.tag()?;
        match (fn_, wt) {
            (1, 2) => shape = parse_tensor_type(pb.bytes()?)?, // tensor_type
            _ => pb.skip(wt)?,
        }
    }
    Ok(shape)
}

fn parse_tensor_type(buf: &[u8]) -> Result<Option<Vec<i64>>, WireError> {
    let mut pb = Pb::new(buf);
    let mut shape = None;
    while !pb.eof() {
        let (fn_, wt) = pb.tag()?;
        match (fn_, wt) {
            (2, 2) => shape = Some(parse_tensor_shape(pb.bytes()?)?), // shape
            _ => pb.skip(wt)?,
        }
    }
    Ok(shape)
}

fn parse_tensor_shape(buf: &[u8]) -> Result<Vec<i64>, WireError> {
    let mut pb = Pb::new(buf);
    let mut dims = vec![];
    while !pb.eof() {
        let (fn_, wt) = pb.tag()?;
        match (fn_, wt) {
            (1, 2) => dims.push(parse_dimension(pb.bytes()?)?), // dim
            _ => pb.skip(wt)?,
        }
    }
    Ok(dims)
}

fn parse_dimension(buf: &[u8]) -> Result<i64, WireError> {
    let mut pb = Pb::new(buf);
    let mut v = 0i64; // 0 == unknown/symbolic
    while !pb.eof() {
        let (fn_, wt) = pb.tag()?;
        match (fn_, wt) {
            (1, 0) => v = pb.varint()? as i64, // dim_value
            _ => pb.skip(wt)?,
        }
    }
    Ok(v)
}

fn parse_node(buf: &[u8]) -> Result<OnnxNode, WireError> {
    // NodeProto: input=1(str), output=2(str), op_type=4(str), attribute=5(AttributeProto)
    let mut pb = Pb::new(buf);
    let mut n = OnnxNode {
        op_type: String::new(),
        inputs: vec![],
        outputs: vec![],
        attr_ints: BTreeMap::new(),
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
            (5, 2) => {
                if let Some((name, ints)) = parse_attr_ints(pb.bytes()?)? {
                    n.attr_ints.insert(name, ints);
                }
            }
            _ => pb.skip(wt)?,
        }
    }
    Ok(n)
}

/// Parse an AttributeProto, extracting the name and any integer payload.
/// AttributeProto: name=1(str), i=3(int64 single), ints=8(int64 repeated,
/// packed or unpacked). Returns `None` when the attribute carries no integer
/// list we can use (e.g. a string/tensor attr) — those are irrelevant to the
/// exact Conv->affine lowering.
fn parse_attr_ints(buf: &[u8]) -> Result<Option<(String, Vec<i64>)>, WireError> {
    let mut pb = Pb::new(buf);
    let mut name = String::new();
    let mut single: Option<i64> = None;
    let mut list: Vec<i64> = vec![];
    while !pb.eof() {
        let (fn_, wt) = pb.tag()?;
        match (fn_, wt) {
            (1, 2) => name = String::from_utf8_lossy(pb.bytes()?).into_owned(),
            (3, 0) => single = Some(pb.varint()? as i64),
            (8, 0) => list.push(pb.varint()? as i64), // unpacked repeated int64
            (8, 2) => {
                // packed repeated int64
                let b = pb.bytes()?;
                let mut p2 = Pb::new(b);
                while !p2.eof() {
                    list.push(p2.varint()? as i64);
                }
            }
            _ => pb.skip(wt)?,
        }
    }
    if name.is_empty() {
        return Ok(None);
    }
    if !list.is_empty() {
        Ok(Some((name, list)))
    } else {
        Ok(single.map(|s| (name, vec![s])))
    }
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
    // GraphProto: node=1(NodeProto), initializer=5(TensorProto)
    let mut pb = Pb::new(gb);
    let mut nodes = vec![];
    let mut inits = BTreeMap::new();
    let mut input_shapes = BTreeMap::new();
    while !pb.eof() {
        let (fn_, wt) = pb.tag()?;
        match (fn_, wt) {
            (1, 2) => nodes.push(parse_node(pb.bytes()?)?),
            (5, 2) => {
                let t = parse_tensor(pb.bytes()?)?;
                inits.insert(t.name.clone(), t);
            }
            (11, 2) => {
                // GraphProto.input (ValueInfoProto)
                if let Some((name, shape)) = parse_value_info(pb.bytes()?)? {
                    input_shapes.insert(name, shape);
                }
            }
            _ => pb.skip(wt)?,
        }
    }
    Ok(OnnxGraph {
        nodes,
        inits,
        input_shapes,
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
        eprintln!("certify_onnx_diff: cannot read {path}: {e}");
        std::process::exit(2);
    });
    parse_onnx(&data).unwrap_or_else(|e| {
        eprintln!("certify_onnx_diff: {path}: {e}");
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

/// One affine layer `W·a + b` applied exactly (no activation). Rows of `W` and
/// entries of `b` pair up; each dot product runs over `a` in increasing index.
fn affine_exact(layer: &AffineLayer, a: &[Rat]) -> Vec<Rat> {
    assert!(
        layer.b.len() >= layer.w.len(),
        "layer bias shorter than its width"
    );
    let mut z = Vec::new();
    for (row, b) in layer.w.iter().zip(&layer.b) {
        assert!(
            row.len() <= a.len(),
            "weight row wider than the layer input"
        );
        let mut acc = *b;
        for (wi, ai) in row.iter().zip(a) {
            acc = acc.add(wi.mul(*ai).unwrap()).unwrap();
        }
        z.push(acc);
    }
    z
}

impl LoadedNet {
    /// Every affine layer in order: the hidden layers, then the read-out.
    fn layers(&self) -> impl Iterator<Item = &AffineLayer> {
        self.hidden.iter().chain(std::iter::once(&self.readout))
    }

    /// Exact forward evaluation of the FULL network (all output logits) at input
    /// `x`. Every hidden layer is `relu(W·x + b)`; the last layer is the linear
    /// read-out. Used by the ONNX-equivalence self-check (compared against
    /// onnxruntime) to prove the Conv->affine lowering is exact.
    fn eval_full(&self, x: &[Rat]) -> Vec<Rat> {
        let mut a: Vec<Rat> = x.to_vec();
        for layer in &self.hidden {
            let mut z = affine_exact(layer, &a);
            for zi in z.iter_mut() {
                if zi.is_negative() {
                    *zi = Rat::ZERO;
                }
            }
            a = z;
        }
        affine_exact(&self.readout, &a)
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

/// Lower an ONNX `Conv` layer to an EXACT sparse-affine map `y = W·x + b` over
/// the flattened (row-major NCHW, batch dropped) input, and return the output
/// spatial shape `(Cout, Hout, Wout)`.
///
/// # Exactness (SOUNDNESS-CRITICAL)
/// A convolution is, by definition, an affine map: each output element is a
/// fixed linear combination of a patch of input elements plus the filter bias.
/// This routine writes exactly the ONNX Conv weights into the `[out][in]`
/// matrix positions dictated by the sliding-window (im2col) index arithmetic —
/// every coefficient is `f32_to_rat` of the *same* filter weight ONNX uses, and
/// the constant is `f32_to_rat` of the *same* bias. There is NO rounding: the
/// lowered map equals the ONNX Conv semantics bit-for-bit over the rationals.
/// A forward self-check in `main` re-confirms this against onnxruntime.
///
/// Supports strides, symmetric/asymmetric `pads` (zero-padding), `dilations`,
/// and grouped convolution (`group`). Only the default `auto_pad=NOTSET` /
/// explicit-`pads` form is handled; `SAME_*` autopadding is rejected upstream by
/// requiring an explicit `pads` attribute (or defaulting to zero).
#[allow(clippy::too_many_arguments)]
fn conv_to_affine(
    w: &OnnxTensor,            // filter [Cout, Cin/group, kh, kw]
    bias: Option<&OnnxTensor>, // [Cout] or None
    in_c: usize,
    in_h: usize,
    in_w: usize,
    strides: (usize, usize),
    pads: (usize, usize, usize, usize), // (top, left, bottom, right)
    dils: (usize, usize),
    group: usize,
) -> (Vec<Vec<Rat>>, Vec<Rat>, (usize, usize, usize)) {
    assert!(
        w.dims.len() == 4,
        "Conv weight must be 4-D [Cout,Cin/g,kh,kw], got {:?}",
        w.dims
    );
    let cout = w.dims[0] as usize;
    let cin_g = w.dims[1] as usize; // Cin per group
    let kh = w.dims[2] as usize;
    let kw = w.dims[3] as usize;
    let (sh, sw) = strides;
    let (pt, pl, pb_, pr) = pads;
    let (dh, dw) = dils;
    assert_eq!(
        cin_g * group,
        in_c,
        "Conv group/channel mismatch: (Cin/g)*group = {}*{} != Cin {}",
        cin_g,
        group,
        in_c
    );
    assert_eq!(
        cout % group,
        0,
        "Conv Cout {cout} not divisible by group {group}"
    );
    let cout_g = cout / group; // output channels per group
                               // Output spatial size (ONNX Conv formula, floor division).
    let out_h = (in_h + pt + pb_).saturating_sub(dh * (kh - 1) + 1) / sh + 1;
    let out_w = (in_w + pl + pr).saturating_sub(dw * (kw - 1) + 1) / sw + 1;
    let n_out = cout * out_h * out_w;
    let n_in = in_c * in_h * in_w;
    // Precompute exact rational filter weights once (avoid re-parsing per output
    // pixel — the same filter is reused across all H×W positions).
    let mut wrat = vec![Rat::ZERO; w.floats.len()];
    for (i, &f) in w.floats.iter().enumerate() {
        wrat[i] = f32_to_rat(f).expect("conv weight -> rat");
    }
    let brat: Vec<Rat> = match bias {
        Some(t) => rat_vec(t).expect("conv bias -> rat"),
        None => vec![Rat::ZERO; cout],
    };
    assert_eq!(brat.len(), cout, "conv bias length != Cout");

    let mut mat = vec![vec![Rat::ZERO; n_in]; n_out];
    let mut b = vec![Rat::ZERO; n_out];
    for co in 0..cout {
        let g = co / cout_g; // which group this output channel belongs to
        let ci_base = g * cin_g; // first input channel of that group
        for oh in 0..out_h {
            for ow in 0..out_w {
                let row = (co * out_h + oh) * out_w + ow;
                b[row] = brat[co];
                for ci in 0..cin_g {
                    for ki in 0..kh {
                        // signed input row/col (padding may push negative)
                        let ih = (oh * sh + ki * dh) as i64 - pt as i64;
                        if ih < 0 || ih as usize >= in_h {
                            continue;
                        }
                        let ih = ih as usize;
                        for kj in 0..kw {
                            let iw = (ow * sw + kj * dw) as i64 - pl as i64;
                            if iw < 0 || iw as usize >= in_w {
                                continue;
                            }
                            let iw = iw as usize;
                            let col = ((ci_base + ci) * in_h + ih) * in_w + iw;
                            let wi = ((co * cin_g + ci) * kh + ki) * kw + kj;
                            // A padded/dilated grid never revisits the same
                            // (col) for one output row, so an assign (not add)
                            // is exact.
                            mat[row][col] = wrat[wi];
                        }
                    }
                }
            }
        }
    }
    (mat, b, (cout, out_h, out_w))
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

    // Current activation SPATIAL shape (batch dropped), row-major NCHW. Empty
    // until we know it. Needed by Conv (which requires Cin×Hin×Win); Flatten and
    // affine ops collapse it to a single flat dim. Seeded from the graph input
    // whose name is not an initializer (the true model input).
    let mut cur_shape: Vec<usize> = vec![];
    if let Some((_, shp)) = g
        .input_shapes
        .iter()
        .find(|(name, _)| !g.inits.contains_key(*name))
    {
        // Drop a leading batch dim (value 1, or unknown 0 which we treat as 1).
        let dims: Vec<usize> = shp
            .iter()
            .map(|&d| if d <= 0 { 1 } else { d as usize })
            .collect();
        let spatial: Vec<usize> = if dims.len() >= 2 && dims[0] == 1 {
            dims[1..].to_vec()
        } else {
            dims
        };
        let prod: usize = spatial.iter().product();
        if prod > 0 {
            input_dim = prod;
            cur_dim = prod;
            cur_shape = spatial;
        }
    }

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
            "Flatten" | "Reshape" => {
                // Collapse the spatial shape to a single flat dim. The ONNX
                // memory order for NCHW is exactly the row-major order our
                // sparse-affine rows already use, so this is a pure no-op on the
                // value vector — only the tracked shape changes. (Reshape's
                // target-shape initializer is ignored: for a vector-carrying
                // graph the only valid reshape flattens to [1, N].)
                if !cur_shape.is_empty() {
                    let flat: usize = cur_shape.iter().product();
                    cur_shape = vec![flat];
                }
            }
            "Conv" => {
                // Conv is an exact sparse-affine map (im2col). Requires a known
                // input spatial shape Cin×Hin×Win.
                assert!(
                    cur_shape.len() == 3,
                    "Conv needs a 3-D CHW input shape, have {:?} (input shape not \
                     tracked — is the graph input's declared shape present?)",
                    cur_shape
                );
                let (in_c, in_h, in_w) = (cur_shape[0], cur_shape[1], cur_shape[2]);
                let wt = &g.inits[&node.inputs[1]];
                let bias = node.inputs.get(2).and_then(|n| g.inits.get(n));
                // Attributes (ONNX defaults when absent: stride 1, pad 0,
                // dilation 1, group 1).
                let kh = wt.dims[2] as usize;
                let kw = wt.dims[3] as usize;
                let get2 = |name: &str, def: usize| -> (usize, usize) {
                    match node.attr_ints.get(name) {
                        Some(v) if v.len() == 2 => (v[0] as usize, v[1] as usize),
                        Some(v) if v.len() == 1 => (v[0] as usize, v[0] as usize),
                        _ => (def, def),
                    }
                };
                let strides = get2("strides", 1);
                let dils = get2("dilations", 1);
                let pads = match node.attr_ints.get("pads") {
                    // ONNX pads = [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
                    // For 2-D: [top, left, bottom, right].
                    Some(p) if p.len() == 4 => {
                        (p[0] as usize, p[1] as usize, p[2] as usize, p[3] as usize)
                    }
                    Some(p) if p.len() == 2 => {
                        (p[0] as usize, p[1] as usize, p[0] as usize, p[1] as usize)
                    }
                    _ => (0, 0, 0, 0),
                };
                assert!(
                    !node.attr_ints.contains_key("auto_pad"),
                    "Conv auto_pad unsupported; provide explicit pads"
                );
                let group = node
                    .attr_ints
                    .get("group")
                    .and_then(|v| v.first())
                    .map(|&x| x as usize)
                    .unwrap_or(1);
                let _ = (kh, kw);
                let (mut w, mut b, out_shape) =
                    conv_to_affine(wt, bias, in_c, in_h, in_w, strides, pads, dils, group);
                // Fold a pending input shift  x' = x - c  into the bias:
                // W·(x-c) = W·x - W·c.
                if let Some(c) = sub_const.take() {
                    let in_dim = w[0].len();
                    assert_eq!(c.len(), in_dim, "sub const dim != conv in dim");
                    for (row, brow) in b.iter_mut().enumerate() {
                        let mut acc = Rat::ZERO;
                        for i in 0..in_dim {
                            acc = acc.add(w[row][i].mul(c[i]).unwrap()).unwrap();
                        }
                        *brow = brow.sub(acc).unwrap();
                    }
                }
                cur_dim = w.len();
                cur_shape = vec![out_shape.0, out_shape.1, out_shape.2];
                // A trailing Add (rare — bias is usually the conv's 3rd input)
                // still folds via the existing Add arm into cur_b.
                let _ = &mut w;
                cur_w = Some(w);
                cur_b = Some(b);
            }
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
                cur_shape = vec![out_w]; // MatMul output is a flat vector
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
/// Returns `None` for a malformed suffix so an untrusted/mistyped property file
/// fails closed instead of panicking on `.unwrap()`.
fn var_index(v: &str) -> Option<usize> {
    v.get(2..).and_then(|s| s.parse::<usize>().ok())
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: certify_onnx <model.onnx> <prop.vnnlib> <out_dir> [out_idx]");
        std::process::exit(2);
    }
    let onnx_path = &args[1];
    let vnnlib_path = &args[2];
    let out_dir = &args[3];
    // DIFFERENCE read-out: certify  g = Y_a - Y_b >= m  on the box.
    //   idx_a, idx_b are REQUIRED for this binary.
    // The unsafe atom we refute is  g <= 0  (i.e. Y_a <= Y_b), refuted iff m > 0.
    // For ACAS prop_3/prop_4 ("COC = Y_0 is minimal" is unsafe), the unsafe
    // conjunction is  Y_0 <= Y_j  for ALL j in {1,2,3,4}; refuting ANY ONE
    // conjunct (proving Y_0 - Y_j > 0 for some j on this leaf) refutes the whole
    // conjunction on the leaf. So we set a=0, b=j and look for m>0.
    let idx_a: usize = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .expect("certify_onnx_diff needs idx_a");
    let idx_b: usize = args
        .get(5)
        .and_then(|s| s.parse().ok())
        .expect("certify_onnx_diff needs idx_b");

    let t0 = Instant::now();
    let g = load_onnx_or_exit(onnx_path);
    let net = load_net(&g);
    let load_us = t0.elapsed().as_micros();

    let n_hidden = net.hidden.len();
    let widths: Vec<usize> = net.layers().map(|l| l.w.len()).collect();
    eprintln!(
        "[load] input_dim={} hidden_layers={} layer_widths={:?} ({} us)",
        net.input_dim, n_hidden, widths, load_us
    );

    // ---- ONNX-equivalence self-check hook (SOUNDNESS) --------------------------
    // When `NYCERT_FWD_IN` names a file of whitespace-separated decimals (one
    // network input), evaluate the FULL lowered net exactly and write the output
    // logits (exact `n/d` rationals, one per line) to `NYCERT_FWD_OUT`, then exit.
    // The Python harness feeds the SAME inputs to onnxruntime and asserts the
    // lowered Conv->affine map reproduces ONNX Conv bit-for-bit. This proves the
    // exactness precondition the certificate's soundness rests on.
    if let (Ok(inp), Ok(outp)) = (
        std::env::var("NYCERT_FWD_IN"),
        std::env::var("NYCERT_FWD_OUT"),
    ) {
        let txt = std::fs::read_to_string(&inp).expect("read fwd input");
        let x: Vec<Rat> = txt
            .split_whitespace()
            .map(|tok| decimal_to_rat(tok).expect("fwd input decimal -> rat"))
            .collect();
        assert_eq!(
            x.len(),
            net.input_dim,
            "fwd input len {} != net input_dim {}",
            x.len(),
            net.input_dim
        );
        let y = net.eval_full(&x);
        let mut s = String::new();
        for yi in &y {
            s.push_str(&fmt_rat(yi));
            s.push('\n');
        }
        std::fs::write(&outp, s).expect("write fwd output");
        eprintln!("[fwd] wrote {} logits to {outp}", y.len());
        return;
    }

    // --- VNNLIB ---
    let vtext = std::fs::read_to_string(vnnlib_path).expect("read vnnlib");
    let atoms = parse_vnnlib(&vtext);

    // Input box from X_i atoms.
    let mut lo = vec![Rat::ZERO; net.input_dim];
    let mut hi = vec![Rat::ZERO; net.input_dim];
    let mut lo_set = vec![false; net.input_dim];
    let mut hi_set = vec![false; net.input_dim];
    for a in &atoms {
        if a.var.starts_with("X_") {
            // `parse_vnnlib` only emits parseable-index atoms; drop any out-of-range
            // index (leaves the box incomplete → reported clearly downstream).
            let Some(idx) = var_index(&a.var) else {
                continue;
            };
            if idx >= net.input_dim {
                continue;
            }
            // parse the raw decimal token losslessly (decimals are n/10^k).
            // Scientific notation passes the f64 gate but decimal_to_rat rejects it;
            // fail closed (exit) rather than panic on the verdict path.
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
        }
        // Y atoms are NOT read from the file in the diff binary: the difference
        // property is fully specified by (idx_a, idx_b) on the CLI.
    }
    for i in 0..net.input_dim {
        assert!(lo_set[i] && hi_set[i], "input X_{i} box incomplete");
    }

    // Build DeepReluProblem with hidden layers and read-out.
    let hidden_w: Vec<Vec<Vec<Rat>>> = net.hidden.iter().map(|l| l.w.clone()).collect();
    let hidden_b: Vec<Vec<Rat>> = net.hidden.iter().map(|l| l.b.clone()).collect();
    let readout = &net.readout;
    let out_dim = readout.w.len();

    // DIFFERENCE read-out: g = Y_a - Y_b = (w_a - w_b).a_last + (b_a - b_b).
    //   Y_a = readout.w[idx_a] . a_last + readout.b[idx_a]
    //   Y_b = readout.w[idx_b] . a_last + readout.b[idx_b]
    // We certify a lower bound  g >= m.  The unsafe atom (one conjunct of the
    // ACAS "Y_0 is minimal" region, with a=0,b=j) is  Y_a <= Y_b  i.e.  g <= 0.
    // It is REFUTED (empty on this leaf) iff  m > 0.
    assert!(
        idx_a < out_dim && idx_b < out_dim,
        "out idx out of range (out_dim={out_dim})"
    );
    let out_weight: Vec<Rat> = readout.w[idx_a]
        .iter()
        .zip(readout.w[idx_b].iter())
        .map(|(wa, wb)| wa.sub(*wb).unwrap())
        .collect();
    let out_bias: Rat = readout.b[idx_a].sub(readout.b[idx_b]).unwrap();
    // The transformed scalar output is  y' = g = Y_a - Y_b.
    // Unsafe region in y':  y' <= u_bound  with  u_bound = 0.
    let u_bound = Rat::ZERO;

    let problem = DeepReluProblem {
        weights: hidden_w,
        biases: hidden_b,
        out_weight,
        out_bias,
        input_lower: lo.clone(),
        input_upper: hi.clone(),
        alpha: None,
        interm_round: false,
    };

    // certify() with any threshold <= m derives the same multipliers proving
    // y' >= m (m = CROWN lower bound). Use a very low threshold to recover m.
    let sentinel = Rat::from_int(-1_000_000_000);
    let t1 = Instant::now();
    let certified = match problem.certify(sentinel) {
        Ok(c) => c,
        Err(e) => {
            println!(
                "CERTIFY_FAILED a={} b={} dim={} : {e}",
                idx_a, idx_b, out_dim
            );
            std::process::exit(3);
        }
    };
    let certify_us = t1.elapsed().as_micros();

    let m = certified.lower_bound; // y' = (Y_a - Y_b) >= m, exact
    let m_f = rat_to_f64(&m);

    // SAFETY decision: the unsafe region y' <= 0 is empty iff m > 0.
    let safe = m > u_bound;
    eprintln!(
        "[crown] exact CROWN: (Y_{idx_a} - Y_{idx_b}) >= {} (~{:.6}) ; unsafe atom was Y_{idx_a} <= Y_{idx_b} (i.e. diff <= 0) -> {}",
        fmt_rat(&m), m_f,
        if safe { "SAFE (refuted: Y_a > Y_b on this leaf)" } else { "NOT proven (diff bound not yet > 0)" }
    );
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

    // Report the certified output bound in the difference coordinates.
    //   y' = (Y_a - Y_b)  => certified  (Y_a - Y_b) >= m
    eprintln!(
        "[bound] EXACT certified output bound: (Y_{idx_a} - Y_{idx_b}) >= {} (~{:.6})",
        fmt_rat(&m),
        rat_to_f64(&m)
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
    // ---- Exact-rational checker acceptance (STEP 3) ----------------------------
    // Re-check the emitted ENTAILMENT with the same exact-rational verifier Clean's
    // kernel mirrors (`farkas_premise_combination`-backed): the non-negative
    // multiplier combination of the premises must reproduce the conclusion
    // `y >= m`. This is the acceptance the certificate must pass.
    match check_entailment(&entailment) {
        Ok((derived, claimed)) => eprintln!(
            "[check] exact-rational CHECKER ACCEPTED entailment: derived {} <= claimed {} (y >= m verified)",
            fmt_rat(&derived),
            fmt_rat(&claimed)
        ),
        Err(e) => {
            println!("CHECKER_REJECTED entailment: {e:?}");
            std::process::exit(5);
        }
    }

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
        // The Farkas refutation must be ACCEPTED by the exact-rational checker:
        // its non-negative combination cancels all variables and yields a
        // negative constant (a contradiction), refuting box ∧ network ∧ unsafe.
        match check_farkas(&farkas) {
            Ok(c) => eprintln!(
                "[check] exact-rational CHECKER ACCEPTED farkas refutation: contradiction constant {} < 0 (UNSAT certified)",
                fmt_rat(&c)
            ),
            Err(e) => {
                println!("CHECKER_REJECTED farkas: {e:?}");
                std::process::exit(5);
            }
        }
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
    println!(
        "OK property={} certify={certify_us}us load={load_us}us premises={} max_cert_bits={}",
        if safe { "DECIDED" } else { "bound-only" },
        entailment.premises.len(),
        max_bits,
    );
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

/// Approximate a (possibly huge) exact rational as `f64` for human-readable
/// diagnostics only. Never used in a certificate or a soundness decision.
fn rat_to_f64(r: &Rat) -> f64 {
    r.to_f64_approx()
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

    /// ModelProto{ graph: GraphProto{ node Conv(x)->y with kernel_shape=[3,3],
    /// init w=[0.5], input x: [1, 4] } }.
    fn model() -> Vec<u8> {
        let mut attr = ld(1, b"kernel_shape");
        attr.extend(ld(8, &[3, 3])); // packed ints
        let mut node = ld(1, b"x");
        node.extend(ld(2, b"y"));
        node.extend(ld(4, b"Conv"));
        node.extend(ld(5, &attr));
        let mut tensor = vec![0x08, 0x01]; // unpacked dims [1]
        tensor.extend(ld(8, b"w"));
        tensor.extend(ld(9, &0.5f32.to_le_bytes()));
        let mut shape = ld(1, &[0x08, 0x01]); // dim_value 1
        shape.extend(ld(1, &[0x08, 0x04])); // dim_value 4
        let tensor_type = ld(2, &shape);
        let type_proto = ld(1, &tensor_type);
        let mut vi = ld(1, b"x");
        vi.extend(ld(2, &type_proto));
        let mut graph = ld(1, &node);
        graph.extend(ld(5, &tensor));
        graph.extend(ld(11, &vi));
        ld(7, &graph)
    }

    #[test]
    fn well_formed_model_decodes() {
        let g = parse_onnx(&model()).expect("well-formed model");
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.nodes[0].op_type, "Conv");
        assert_eq!(g.nodes[0].attr_ints.get("kernel_shape"), Some(&vec![3, 3]));
        let w = g.inits.get("w").expect("initializer w");
        assert_eq!(w.dims, vec![1]);
        assert_eq!(w.floats, vec![0.5]);
        assert_eq!(g.input_shapes.get("x"), Some(&vec![1, 4]));
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
        assert!(matches!(parse_onnx(&[]), Err(OnnxError::NoGraph)));
        assert!(matches!(
            parse_onnx(&[0x0c]),
            Err(OnnxError::Wire(WireError::UnsupportedWireType {
                wire_type: 4
            }))
        ));
    }
}
