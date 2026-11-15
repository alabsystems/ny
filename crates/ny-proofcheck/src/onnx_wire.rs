// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! A TOTAL, bounds-checked protobuf reader for the subset of the ONNX schema
//! the checker lowers. Written for this crate (no prost, no generated code,
//! no shared parser with any producer): every read returns a `Result`, so a
//! truncated or hostile file is an error, never a panic or an out-of-bounds
//! read. Unknown fields are skipped by wire type; groups (wire types 3/4)
//! are refused.
//!
//! Field numbers follow `onnx.proto` (IR version 3+). Only what the lowering
//! needs is decoded: the model's graph and opset imports; the graph's nodes,
//! initializers, inputs and outputs; a node's inputs, outputs, op type,
//! domain and attributes; an attribute's scalar/tensor/int-list/float-list
//! payloads (a sub-graph attribute is only FLAGGED, the lowering refuses it);
//! a tensor's dims, type, name, typed data and raw data (external data is
//! flagged and refused).

use std::fmt;

/// Why the wire reader refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireError(pub String);

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

type Res<T> = Result<T, WireError>;

fn err<T>(what: &str) -> Res<T> {
    Err(WireError(what.to_string()))
}

/// A cursor over one message's bytes.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    const fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    const fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn byte(&mut self) -> Res<u8> {
        let b = *self
            .buf
            .get(self.pos)
            .ok_or_else(|| WireError("truncated varint".into()))?;
        self.pos += 1;
        Ok(b)
    }

    fn varint(&mut self) -> Res<u64> {
        let mut out: u64 = 0;
        for shift in 0..10u32 {
            let b = self.byte()?;
            let low = u64::from(b & 0x7f);
            if shift == 9 && low > 1 {
                return err("varint overflows 64 bits");
            }
            out |= low << (7 * shift);
            if b & 0x80 == 0 {
                return Ok(out);
            }
        }
        err("varint longer than 10 bytes")
    }

    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| WireError("length overflow".into()))?;
        let s = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| WireError("truncated field".into()))?;
        self.pos = end;
        Ok(s)
    }

    fn len_delimited(&mut self) -> Res<&'a [u8]> {
        let n = self.varint()?;
        let n = usize::try_from(n).map_err(|_| WireError("length exceeds usize".into()))?;
        self.take(n)
    }

    fn fixed32(&mut self) -> Res<u32> {
        let s = self.take(4)?;
        let mut a = [0u8; 4];
        a.copy_from_slice(s);
        Ok(u32::from_le_bytes(a))
    }

    fn fixed64(&mut self) -> Res<u64> {
        let s = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(u64::from_le_bytes(a))
    }

    /// `(field number, wire type)`, or `None` at the end of the message.
    fn key(&mut self) -> Res<Option<(u64, u8)>> {
        if self.done() {
            return Ok(None);
        }
        let k = self.varint()?;
        let wire = (k & 7) as u8;
        let field = k >> 3;
        if field == 0 {
            return err("field number 0");
        }
        Ok(Some((field, wire)))
    }

    fn skip(&mut self, wire: u8) -> Res<()> {
        match wire {
            0 => self.varint().map(|_| ()),
            1 => self.fixed64().map(|_| ()),
            2 => self.len_delimited().map(|_| ()),
            5 => self.fixed32().map(|_| ()),
            _ => err("unsupported wire type (group or reserved)"),
        }
    }
}

fn string(bytes: &[u8]) -> Res<String> {
    String::from_utf8(bytes.to_vec()).map_err(|_| WireError("string is not UTF-8".into()))
}

/// Append a repeated int64/varint field, packed (wire 2) or not (wire 0).
fn read_i64s(r: &mut Reader<'_>, wire: u8, out: &mut Vec<i64>) -> Res<()> {
    match wire {
        0 => out.push(r.varint()? as i64),
        2 => {
            let mut sub = Reader::new(r.len_delimited()?);
            while !sub.done() {
                out.push(sub.varint()? as i64);
            }
        }
        _ => return err("bad wire type for an int64 field"),
    }
    Ok(())
}

/// Append a repeated float field, packed (wire 2) or not (wire 5).
fn read_f32s(r: &mut Reader<'_>, wire: u8, out: &mut Vec<f32>) -> Res<()> {
    match wire {
        5 => out.push(f32::from_bits(r.fixed32()?)),
        2 => {
            let bytes = r.len_delimited()?;
            if bytes.len() % 4 != 0 {
                return err("packed float field length not a multiple of 4");
            }
            for c in bytes.as_chunks::<4>().0 {
                let mut a = [0u8; 4];
                a.copy_from_slice(c);
                out.push(f32::from_bits(u32::from_le_bytes(a)));
            }
        }
        _ => return err("bad wire type for a float field"),
    }
    Ok(())
}

/// `TensorProto`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tensor {
    /// Name.
    pub name: String,
    /// Dimensions.
    pub dims: Vec<i64>,
    /// `TensorProto.DataType`.
    pub data_type: i64,
    /// `float_data`.
    pub float_data: Vec<f32>,
    /// `int64_data`.
    pub int64_data: Vec<i64>,
    /// `raw_data`, when present.
    pub raw: Option<Vec<u8>>,
    /// Whether any non-float/int64 typed payload (int32, string, double,
    /// uint64) was present: the lowering refuses such tensors.
    pub other_payload: bool,
    /// Whether `data_location` is EXTERNAL or `external_data` is present.
    pub external: bool,
}

/// ONNX `TensorProto.DataType` FLOAT.
pub const DT_FLOAT: i64 = 1;
/// ONNX `TensorProto.DataType` INT64.
pub const DT_INT64: i64 = 7;

impl Tensor {
    fn parse(bytes: &[u8]) -> Res<Self> {
        let mut t = Self::default();
        let mut r = Reader::new(bytes);
        while let Some((field, wire)) = r.key()? {
            match (field, wire) {
                (1, 0 | 2) => read_i64s(&mut r, wire, &mut t.dims)?,
                (2, 0) => t.data_type = r.varint()? as i64,
                (4, 2 | 5) => read_f32s(&mut r, wire, &mut t.float_data)?,
                (7, 0 | 2) => read_i64s(&mut r, wire, &mut t.int64_data)?,
                (8, 2) => t.name = string(r.len_delimited()?)?,
                (9, 2) => t.raw = Some(r.len_delimited()?.to_vec()),
                (5 | 6 | 10 | 11, _) => {
                    t.other_payload = true;
                    r.skip(wire)?;
                }
                (13, 2) => {
                    t.external = true;
                    r.skip(wire)?;
                }
                (14, 0) => {
                    if r.varint()? != 0 {
                        t.external = true;
                    }
                }
                (3, _) => return err("segmented tensors are not supported"),
                _ => r.skip(wire)?,
            }
        }
        Ok(t)
    }

    /// Element count from `dims` (empty dims = scalar = 1). `None` on a
    /// negative dim or overflow.
    #[must_use]
    pub fn numel(&self) -> Option<usize> {
        let mut n: usize = 1;
        for &d in &self.dims {
            let d = usize::try_from(d).ok()?;
            n = n.checked_mul(d)?;
        }
        Some(n)
    }

    /// The FLOAT payload as exact f64 values (from `raw_data` little-endian
    /// or `float_data`), checked against `dims`.
    pub fn floats(&self) -> Res<Vec<f64>> {
        if self.external {
            return err("external tensor data is not supported");
        }
        if self.data_type != DT_FLOAT || self.other_payload {
            return err("tensor is not FLOAT");
        }
        let n = self
            .numel()
            .ok_or_else(|| WireError("bad tensor dims".into()))?;
        let out: Vec<f64> = match &self.raw {
            Some(raw) => {
                if !self.float_data.is_empty() {
                    return err("tensor carries both raw_data and float_data");
                }
                if raw.len() != n.saturating_mul(4) {
                    return err("raw_data length does not match dims");
                }
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| {
                        let mut a = [0u8; 4];
                        a.copy_from_slice(c);
                        f64::from(f32::from_bits(u32::from_le_bytes(a)))
                    })
                    .collect()
            }
            None => {
                if self.float_data.len() != n {
                    return err("float_data length does not match dims");
                }
                self.float_data.iter().map(|&v| f64::from(v)).collect()
            }
        };
        Ok(out)
    }

    /// The INT64 payload, checked against `dims`.
    pub fn int64s(&self) -> Res<Vec<i64>> {
        if self.external {
            return err("external tensor data is not supported");
        }
        if self.data_type != DT_INT64 || self.other_payload {
            return err("tensor is not INT64");
        }
        let n = self
            .numel()
            .ok_or_else(|| WireError("bad tensor dims".into()))?;
        match &self.raw {
            Some(raw) => {
                if raw.len() != n.saturating_mul(8) {
                    return err("raw_data length does not match dims");
                }
                Ok(raw
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| {
                        let mut a = [0u8; 8];
                        a.copy_from_slice(c);
                        i64::from_le_bytes(a)
                    })
                    .collect())
            }
            None => {
                if self.int64_data.len() != n {
                    return err("int64_data length does not match dims");
                }
                Ok(self.int64_data.clone())
            }
        }
    }
}

/// `AttributeProto` (the decoded subset).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Attr {
    /// Name.
    pub name: String,
    /// `f`.
    pub f: Option<f32>,
    /// `i`.
    pub i: Option<i64>,
    /// `s`.
    pub s: Option<Vec<u8>>,
    /// `t`.
    pub t: Option<Tensor>,
    /// `floats`.
    pub floats: Vec<f32>,
    /// `ints`.
    pub ints: Vec<i64>,
    /// A graph / sparse-tensor / type / strings / tensors payload was present.
    pub unsupported_payload: bool,
}

impl Attr {
    fn parse(bytes: &[u8]) -> Res<Self> {
        let mut a = Self::default();
        let mut r = Reader::new(bytes);
        while let Some((field, wire)) = r.key()? {
            match (field, wire) {
                (1, 2) => a.name = string(r.len_delimited()?)?,
                (2, 5) => a.f = Some(f32::from_bits(r.fixed32()?)),
                (3, 0) => a.i = Some(r.varint()? as i64),
                (4, 2) => a.s = Some(r.len_delimited()?.to_vec()),
                (5, 2) => a.t = Some(Tensor::parse(r.len_delimited()?)?),
                (7, 2 | 5) => read_f32s(&mut r, wire, &mut a.floats)?,
                (8, 0 | 2) => read_i64s(&mut r, wire, &mut a.ints)?,
                (6 | 9 | 10 | 11 | 14 | 15 | 22 | 23 | 21, _) => {
                    a.unsupported_payload = true;
                    r.skip(wire)?;
                }
                _ => r.skip(wire)?,
            }
        }
        Ok(a)
    }
}

/// `NodeProto`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Node {
    /// Inputs (empty string = omitted optional input).
    pub inputs: Vec<String>,
    /// Outputs.
    pub outputs: Vec<String>,
    /// Op type.
    pub op_type: String,
    /// Domain (`""` = default).
    pub domain: String,
    /// Attributes.
    pub attrs: Vec<Attr>,
}

impl Node {
    fn parse(bytes: &[u8]) -> Res<Self> {
        let mut n = Self::default();
        let mut r = Reader::new(bytes);
        while let Some((field, wire)) = r.key()? {
            match (field, wire) {
                (1, 2) => n.inputs.push(string(r.len_delimited()?)?),
                (2, 2) => n.outputs.push(string(r.len_delimited()?)?),
                (4, 2) => n.op_type = string(r.len_delimited()?)?,
                (7, 2) => n.domain = string(r.len_delimited()?)?,
                (5, 2) => n.attrs.push(Attr::parse(r.len_delimited()?)?),
                _ => r.skip(wire)?,
            }
        }
        Ok(n)
    }

    /// The attribute named `name`, if any.
    #[must_use]
    pub fn attr(&self, name: &str) -> Option<&Attr> {
        self.attrs.iter().find(|a| a.name == name)
    }
}

/// One dimension of a declared shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    /// A concrete size.
    Value(i64),
    /// A symbolic size.
    Param(String),
    /// Neither was given.
    Unknown,
}

/// `ValueInfoProto`, decoded to its name and (for tensor types) its shape.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ValueInfo {
    /// Name.
    pub name: String,
    /// Tensor element type, when given.
    pub elem_type: Option<i64>,
    /// Shape, when given.
    pub dims: Option<Vec<Dim>>,
}

fn parse_dim(bytes: &[u8]) -> Res<Dim> {
    let mut d = Dim::Unknown;
    let mut r = Reader::new(bytes);
    while let Some((field, wire)) = r.key()? {
        match (field, wire) {
            (1, 0) => d = Dim::Value(r.varint()? as i64),
            (2, 2) => d = Dim::Param(string(r.len_delimited()?)?),
            _ => r.skip(wire)?,
        }
    }
    Ok(d)
}

fn parse_shape(bytes: &[u8]) -> Res<Vec<Dim>> {
    let mut dims = Vec::new();
    let mut r = Reader::new(bytes);
    while let Some((field, wire)) = r.key()? {
        match (field, wire) {
            (1, 2) => dims.push(parse_dim(r.len_delimited()?)?),
            _ => r.skip(wire)?,
        }
    }
    Ok(dims)
}

impl ValueInfo {
    fn parse(bytes: &[u8]) -> Res<Self> {
        let mut v = Self::default();
        let mut r = Reader::new(bytes);
        while let Some((field, wire)) = r.key()? {
            match (field, wire) {
                (1, 2) => v.name = string(r.len_delimited()?)?,
                (2, 2) => {
                    // TypeProto: tensor_type = 1.
                    let mut tr = Reader::new(r.len_delimited()?);
                    while let Some((tf, tw)) = tr.key()? {
                        match (tf, tw) {
                            (1, 2) => {
                                let mut ttr = Reader::new(tr.len_delimited()?);
                                while let Some((f2, w2)) = ttr.key()? {
                                    match (f2, w2) {
                                        (1, 0) => v.elem_type = Some(ttr.varint()? as i64),
                                        (2, 2) => v.dims = Some(parse_shape(ttr.len_delimited()?)?),
                                        _ => ttr.skip(w2)?,
                                    }
                                }
                            }
                            _ => tr.skip(tw)?,
                        }
                    }
                }
                _ => r.skip(wire)?,
            }
        }
        Ok(v)
    }
}

/// `GraphProto` (the decoded subset).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    /// Nodes, in file order (ONNX requires topological order).
    pub nodes: Vec<Node>,
    /// Initializers.
    pub initializers: Vec<Tensor>,
    /// Declared inputs.
    pub inputs: Vec<ValueInfo>,
    /// Declared outputs.
    pub outputs: Vec<ValueInfo>,
    /// A sparse initializer was present (refused by the lowering).
    pub sparse_initializers: bool,
}

impl Graph {
    fn parse(bytes: &[u8]) -> Res<Self> {
        let mut g = Self::default();
        let mut r = Reader::new(bytes);
        while let Some((field, wire)) = r.key()? {
            match (field, wire) {
                (1, 2) => g.nodes.push(Node::parse(r.len_delimited()?)?),
                (5, 2) => g.initializers.push(Tensor::parse(r.len_delimited()?)?),
                (11, 2) => g.inputs.push(ValueInfo::parse(r.len_delimited()?)?),
                (12, 2) => g.outputs.push(ValueInfo::parse(r.len_delimited()?)?),
                (15, _) => {
                    g.sparse_initializers = true;
                    r.skip(wire)?;
                }
                _ => r.skip(wire)?,
            }
        }
        Ok(g)
    }
}

/// `ModelProto` (the decoded subset).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Model {
    /// The main graph.
    pub graph: Graph,
    /// `(domain, version)` opset imports.
    pub opsets: Vec<(String, i64)>,
    /// Model-local functions were present (refused by the lowering when a
    /// node names a non-default domain).
    pub functions: bool,
}

/// Decode a serialized `ModelProto`.
pub fn parse_model(bytes: &[u8]) -> Result<Model, WireError> {
    let mut m = Model::default();
    let mut graph = None;
    let mut r = Reader::new(bytes);
    while let Some((field, wire)) = r.key()? {
        match (field, wire) {
            (7, 2) => {
                if graph.is_some() {
                    return err("model carries two graphs");
                }
                graph = Some(Graph::parse(r.len_delimited()?)?);
            }
            (8, 2) => {
                let mut or = Reader::new(r.len_delimited()?);
                let mut domain = String::new();
                let mut version = 0i64;
                while let Some((f, w)) = or.key()? {
                    match (f, w) {
                        (1, 2) => domain = string(or.len_delimited()?)?,
                        (2, 0) => version = or.varint()? as i64,
                        _ => or.skip(w)?,
                    }
                }
                m.opsets.push((domain, version));
            }
            (25, _) => {
                m.functions = true;
                r.skip(wire)?;
            }
            _ => r.skip(wire)?,
        }
    }
    m.graph = graph.ok_or_else(|| WireError("model has no graph".into()))?;
    Ok(m)
}
