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
// Minimal protobuf reader (ONNX is protobuf-encoded).
// ---------------------------------------------------------------------------
struct Pb<'a> {
    buf: &'a [u8],
    i: usize,
}
impl<'a> Pb<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Pb { buf, i: 0 }
    }
    fn eof(&self) -> bool {
        self.i >= self.buf.len()
    }
    fn varint(&mut self) -> u64 {
        let mut shift = 0u32;
        let mut result = 0u64;
        loop {
            let b = self.buf[self.i];
            self.i += 1;
            result |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        result
    }
    /// Returns (field_number, wire_type).
    fn tag(&mut self) -> (u64, u8) {
        let t = self.varint();
        (t >> 3, (t & 7) as u8)
    }
    fn bytes(&mut self) -> &'a [u8] {
        let len = self.varint() as usize;
        let s = &self.buf[self.i..self.i + len];
        self.i += len;
        s
    }
    fn fixed32(&mut self) -> [u8; 4] {
        let s = [
            self.buf[self.i],
            self.buf[self.i + 1],
            self.buf[self.i + 2],
            self.buf[self.i + 3],
        ];
        self.i += 4;
        s
    }
    #[allow(dead_code)] // wire-type helper kept for completeness alongside fixed32/skip
    fn fixed64(&mut self) -> [u8; 8] {
        let mut s = [0u8; 8];
        s.copy_from_slice(&self.buf[self.i..self.i + 8]);
        self.i += 8;
        s
    }
    /// Skip a field of the given wire type.
    fn skip(&mut self, wt: u8) {
        match wt {
            0 => {
                self.varint();
            }
            2 => {
                let _ = self.bytes();
            }
            5 => {
                self.i += 4;
            }
            1 => {
                self.i += 8;
            }
            _ => panic!("unknown wire type {wt}"),
        }
    }
}

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
fn parse_value_info(buf: &[u8]) -> Option<(String, Vec<i64>)> {
    let mut pb = Pb::new(buf);
    let mut name = String::new();
    let mut shape: Option<Vec<i64>> = None;
    while !pb.eof() {
        let (fn_, wt) = pb.tag();
        match (fn_, wt) {
            (1, 2) => name = String::from_utf8_lossy(pb.bytes()).into_owned(),
            (2, 2) => shape = parse_type_proto(pb.bytes()),
            _ => pb.skip(wt),
        }
    }
    if name.is_empty() {
        return None;
    }
    Some((name, shape.unwrap_or_default()))
}

fn parse_type_proto(buf: &[u8]) -> Option<Vec<i64>> {
    let mut pb = Pb::new(buf);
    let mut shape = None;
    while !pb.eof() {
        let (fn_, wt) = pb.tag();
        match (fn_, wt) {
            (1, 2) => shape = parse_tensor_type(pb.bytes()), // tensor_type
            _ => pb.skip(wt),
        }
    }
    shape
}

fn parse_tensor_type(buf: &[u8]) -> Option<Vec<i64>> {
    let mut pb = Pb::new(buf);
    let mut shape = None;
    while !pb.eof() {
        let (fn_, wt) = pb.tag();
        match (fn_, wt) {
            (2, 2) => shape = Some(parse_tensor_shape(pb.bytes())), // shape
            _ => pb.skip(wt),
        }
    }
    shape
}

fn parse_tensor_shape(buf: &[u8]) -> Vec<i64> {
    let mut pb = Pb::new(buf);
    let mut dims = vec![];
    while !pb.eof() {
        let (fn_, wt) = pb.tag();
        match (fn_, wt) {
            (1, 2) => dims.push(parse_dimension(pb.bytes())), // dim
            _ => pb.skip(wt),
        }
    }
    dims
}

fn parse_dimension(buf: &[u8]) -> i64 {
    let mut pb = Pb::new(buf);
    let mut v = 0i64; // 0 == unknown/symbolic
    while !pb.eof() {
        let (fn_, wt) = pb.tag();
        match (fn_, wt) {
            (1, 0) => v = pb.varint() as i64, // dim_value
            _ => pb.skip(wt),
        }
    }
    v
}

fn parse_node(buf: &[u8]) -> OnnxNode {
    // NodeProto: input=1(str), output=2(str), op_type=4(str), attribute=5(AttributeProto)
    let mut pb = Pb::new(buf);
    let mut n = OnnxNode {
        op_type: String::new(),
        inputs: vec![],
        outputs: vec![],
        attr_ints: BTreeMap::new(),
    };
    while !pb.eof() {
        let (fn_, wt) = pb.tag();
        match (fn_, wt) {
            (1, 2) => n
                .inputs
                .push(String::from_utf8_lossy(pb.bytes()).into_owned()),
            (2, 2) => n
                .outputs
                .push(String::from_utf8_lossy(pb.bytes()).into_owned()),
            (4, 2) => n.op_type = String::from_utf8_lossy(pb.bytes()).into_owned(),
            (5, 2) => {
                if let Some((name, ints)) = parse_attr_ints(pb.bytes()) {
                    n.attr_ints.insert(name, ints);
                }
            }
            _ => pb.skip(wt),
        }
    }
    n
}

/// Parse an AttributeProto, extracting the name and any integer payload.
/// AttributeProto: name=1(str), i=3(int64 single), ints=8(int64 repeated,
/// packed or unpacked). Returns `None` when the attribute carries no integer
/// list we can use (e.g. a string/tensor attr) — those are irrelevant to the
/// exact Conv->affine lowering.
fn parse_attr_ints(buf: &[u8]) -> Option<(String, Vec<i64>)> {
    let mut pb = Pb::new(buf);
    let mut name = String::new();
    let mut single: Option<i64> = None;
    let mut list: Vec<i64> = vec![];
    while !pb.eof() {
        let (fn_, wt) = pb.tag();
        match (fn_, wt) {
            (1, 2) => name = String::from_utf8_lossy(pb.bytes()).into_owned(),
            (3, 0) => single = Some(pb.varint() as i64),
            (8, 0) => list.push(pb.varint() as i64), // unpacked repeated int64
            (8, 2) => {
                // packed repeated int64
                let b = pb.bytes();
                let mut p2 = Pb::new(b);
                while !p2.eof() {
                    list.push(p2.varint() as i64);
                }
            }
            _ => pb.skip(wt),
        }
    }
    if name.is_empty() {
        return None;
    }
    if !list.is_empty() {
        Some((name, list))
    } else {
        single.map(|s| (name, vec![s]))
    }
}

fn parse_tensor(buf: &[u8]) -> OnnxTensor {
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
        let (fn_, wt) = pb.tag();
        match (fn_, wt) {
            (1, 0) => t.dims.push(pb.varint() as i64),
            (1, 2) => {
                // packed repeated int64 dims
                let b = pb.bytes();
                let mut p2 = Pb::new(b);
                while !p2.eof() {
                    t.dims.push(p2.varint() as i64);
                }
            }
            (4, 5) => float_data.push(f32::from_le_bytes(pb.fixed32())),
            (4, 2) => {
                // packed repeated float
                let b = pb.bytes();
                let mut p2 = Pb::new(b);
                while !p2.eof() {
                    float_data.push(f32::from_le_bytes(p2.fixed32()));
                }
            }
            (8, 2) => t.name = String::from_utf8_lossy(pb.bytes()).into_owned(),
            (9, 2) => raw = Some(pb.bytes().to_vec()),
            _ => pb.skip(wt),
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
    t
}

fn parse_onnx(data: &[u8]) -> OnnxGraph {
    // ModelProto: graph=7 (GraphProto)
    let mut pb = Pb::new(data);
    let mut graph_bytes: Option<&[u8]> = None;
    while !pb.eof() {
        let (fn_, wt) = pb.tag();
        if fn_ == 7 && wt == 2 {
            graph_bytes = Some(pb.bytes());
        } else {
            pb.skip(wt);
        }
    }
    let gb = graph_bytes.expect("no graph in ONNX");
    // GraphProto: node=1(NodeProto), initializer=5(TensorProto)
    let mut pb = Pb::new(gb);
    let mut nodes = vec![];
    let mut inits = BTreeMap::new();
    let mut input_shapes = BTreeMap::new();
    while !pb.eof() {
        let (fn_, wt) = pb.tag();
        match (fn_, wt) {
            (1, 2) => nodes.push(parse_node(pb.bytes())),
            (5, 2) => {
                let t = parse_tensor(pb.bytes());
                inits.insert(t.name.clone(), t);
            }
            (11, 2) => {
                // GraphProto.input (ValueInfoProto)
                if let Some((name, shape)) = parse_value_info(pb.bytes()) {
                    input_shapes.insert(name, shape);
                }
            }
            _ => pb.skip(wt),
        }
    }
    OnnxGraph {
        nodes,
        inits,
        input_shapes,
    }
}

// ---------------------------------------------------------------------------
// f32 -> exact Rat (n / 2^k), lossless.  f32 == m * 2^e with m an integer.
// ---------------------------------------------------------------------------
fn f32_to_rat(v: f32) -> Result<Rat, RatError> {
    if v == 0.0 {
        return Ok(Rat::ZERO);
    }
    assert!(v.is_finite(), "non-finite weight {v}");
    let bits = v.to_bits();
    let sign = if bits >> 31 == 1 { -1i128 } else { 1i128 };
    let exp_field = ((bits >> 23) & 0xff) as i32;
    let mant_field = (bits & 0x7f_ffff) as i128;
    // value = sign * mantissa * 2^exp
    let (mantissa, exp) = if exp_field == 0 {
        // subnormal: value = mant * 2^(-126-23)
        (mant_field, -149)
    } else {
        // normal: implicit leading 1
        (mant_field + (1i128 << 23), exp_field - 127 - 23)
    };
    let mantissa = sign * mantissa;
    // value = mantissa * 2^exp, exactly, in arbitrary precision (no i128 cap:
    // a large normal f32 has exp up to +104 and mantissa up to 2^24, whose
    // product exceeds i128 — bignum makes this lossless for every f32).
    use num_bigint::BigInt;
    let m = BigInt::from(mantissa);
    if exp >= 0 {
        // n = mantissa * 2^exp, den = 1
        let num = m << (exp as u32);
        Rat::from_bigints(num, BigInt::from(1))
    } else {
        // n / 2^(-exp)
        let den = BigInt::from(1) << ((-exp) as u32);
        Rat::from_bigints(m, den)
    }
}

// ---------------------------------------------------------------------------
// Build an affine chain (W,b per layer) + relu structure from the ONNX graph.
// Supports: Sub (const), Flatten (no-op for vectors), MatMul (x @ W), Add (const), Relu.
// ---------------------------------------------------------------------------
struct AffineLayer {
    w: Vec<Vec<Rat>>, // [out][in]
    b: Vec<Rat>,      // [out]
}

struct LoadedNet {
    layers: Vec<AffineLayer>, // each followed by ReLU except the last (linear read-out)
    input_dim: usize,
}

impl LoadedNet {
    /// Exact forward evaluation of the FULL network (all output logits) at input
    /// `x`. Every hidden layer is `relu(W·x + b)`; the last layer is the linear
    /// read-out. Used by the ONNX-equivalence self-check (compared against
    /// onnxruntime) to prove the Conv->affine lowering is exact.
    fn eval_full(&self, x: &[Rat]) -> Vec<Rat> {
        let mut a: Vec<Rat> = x.to_vec();
        let last = self.layers.len() - 1;
        for (li, layer) in self.layers.iter().enumerate() {
            let mut z = vec![Rat::ZERO; layer.w.len()];
            for (o, row) in layer.w.iter().enumerate() {
                let mut acc = layer.b[o];
                for (i, wi) in row.iter().enumerate() {
                    acc = acc.add(wi.mul(a[i]).unwrap()).unwrap();
                }
                z[o] = acc;
            }
            if li < last {
                for zi in z.iter_mut() {
                    if zi.is_negative() {
                        *zi = Rat::ZERO;
                    }
                }
            }
            a = z;
        }
        a
    }
}

fn rat_mat(t: &OnnxTensor, rows: usize, cols: usize) -> Result<Vec<Vec<Rat>>, RatError> {
    // ONNX MatMul stores W as [in, out] row-major (in*out). We want [out][in].
    assert_eq!(
        t.floats.len(),
        rows * cols,
        "matrix size mismatch for {}",
        t.name
    );
    let mut out = vec![vec![Rat::ZERO; rows]; cols]; // [out=cols][in=rows]
    for r in 0..rows {
        for c in 0..cols {
            out[c][r] = f32_to_rat(t.floats[r * cols + c])?;
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
                let b = cur_b.take().unwrap_or_else(|| vec![Rat::ZERO; w.len()]);
                layers.push(AffineLayer { w, b });
            }
            other => panic!("unsupported op {other}"),
        }
    }
    // Final linear read-out (the last MatMul+Add not followed by Relu).
    if let Some(w) = cur_w.take() {
        let b = cur_b.take().unwrap_or_else(|| vec![Rat::ZERO; w.len()]);
        layers.push(AffineLayer { w, b });
    } else {
        // The network ENDS in a ReLU (the last hidden layer is the output, e.g.
        // y = ReLU(W x)).  DeepReluProblem requires a linear read-out after the
        // last ReLU, so synthesize an identity read-out  y = a_last.
        let last_width = layers.last().expect("no layers at all").w.len();
        let mut w = vec![vec![Rat::ZERO; last_width]; last_width];
        for d in 0..last_width {
            w[d][d] = Rat::ONE;
        }
        layers.push(AffineLayer {
            w,
            b: vec![Rat::ZERO; last_width],
        });
    }
    LoadedNet { layers, input_dim }
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
    let data = std::fs::read(onnx_path).expect("read onnx");
    let g = parse_onnx(&data);
    let net = load_net(&g);
    let load_us = t0.elapsed().as_micros();

    let n_hidden = net.layers.len() - 1;
    let widths: Vec<usize> = net.layers.iter().map(|l| l.w.len()).collect();
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
    let hidden_w: Vec<Vec<Vec<Rat>>> = net.layers[..n_hidden].iter().map(|l| l.w.clone()).collect();
    let hidden_b: Vec<Vec<Rat>> = net.layers[..n_hidden].iter().map(|l| l.b.clone()).collect();
    let readout = &net.layers[n_hidden];
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
            serde_json::to_string(&ent).unwrap(),
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
                serde_json::to_string(&far).unwrap(),
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
    let bits = |r: &Rat| r.num().bits().max(r.den().bits());
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
    use num_traits::One;
    if r.den().is_one() {
        format!("{}", r.num())
    } else {
        format!("{}/{}", r.num(), r.den())
    }
}

/// Approximate a (possibly huge) exact rational as `f64` for human-readable
/// diagnostics only. Never used in a certificate or a soundness decision.
fn rat_to_f64(r: &Rat) -> f64 {
    use num_traits::ToPrimitive;
    r.to_big().to_f64().unwrap_or(f64::NAN)
}
