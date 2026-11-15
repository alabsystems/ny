// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The checker's own IR, lowered from the decoded ONNX graph.
//!
//! Admission is EXACTLY the margin-row family's op set (the lane's extractor
//! arms), and nothing else:
//!
//! | ONNX op | IR | conditions |
//! |---|---|---|
//! | Conv | [`Op::Conv`] | constant 4-D weight, optional bias; `group = 1`, `dilations = 1`, explicit pads (`auto_pad` absent or `NOTSET`) |
//! | BatchNormalization | [`Op::Bn`] | inference form; **kept unfolded**: `s = γ/√(var+ε)` is an outward interval `[s_lo, s_hi]`, the shift `β − s·mean` likewise |
//! | Gemm | [`Op::Gemm`] | `alpha = beta = 1`, `transA = 0`, constant `B`, optional constant `C` |
//! | Add | [`Op::Add`] / [`Op::AddConst`] | two same-shape activations, or an activation plus a constant that broadcasts to it without growing it (see Shapes) |
//! | Relu | [`Op::Relu`] | |
//! | Flatten | [`Op::Identity`] | any `axis` in `[-r, r]` (a flat-order-preserving reshape to 2-D) |
//! | Reshape | [`Op::Identity`] | constant 1-D INT64 shape, `allowzero = 0`: `0` copies the input dim, one `-1` is inferred, the element count is kept (any static target) |
//! | Identity / Dropout | [`Op::Identity`] | one input (inference form) |
//! | Constant | (initializer) | `value` attribute only |
//! | MatMul | [`Op::Gemm`] | one activation and one constant FLOAT operand, either side, with numpy's rank-1 promotion and batch broadcasting: lowered to the EXACT dense matrix of the linear map (values placed, nothing computed; at most `2^24` entries or the constant's own size) |
//! | Slice | [`Op::Select`] | activation data; constant 1-D INT64 `starts`/`ends` and optional `axes`/`steps` (opset >= 10 input form) |
//! | Gather | [`Op::Select`] | activation data, constant INT64 indices (in range; negative only at opset >= 11) |
//! | Concat | [`Op::Select`] | activations only, equal rank, dims equal off `axis` |
//! | Sub | [`Op::AddConst`] / [`Op::Scale`] | activation minus a broadcastable constant (exact negation), or a constant minus an activation (`s = -1`) (B2.2) |
//! | Mul | [`Op::Scale`] | activation times a broadcastable constant (exact `s`) (B2.2) |
//! | Div | [`Op::Scale`] | activation divided by a broadcastable nonzero constant: `s = 1/c` as an OUTWARD interval (B2.2) |
//! | Neg | [`Op::Scale`] | `s = -1` (B2.2) |
//!
//! [`Op::Scale`] is the elementwise interval-affine map `y = s·x + sh` with
//! `s ∈ [s_lo, s_hi]`, `sh ∈ [sh_lo, sh_hi]` — BatchNormalization's own
//! arithmetic with one "channel" per element, so it inherits BN's certified
//! forward (`ibp`) and back-substitution (`backsub`) charges unchanged.
//! [`Op::Select`] is exact index routing: every output element is a COPY of
//! one input element (no arithmetic, so nothing to round); its backward
//! scatters coefficients back, charged like any accumulation with the
//! largest number of outputs that read one input element as its fan-in.
//!
//! # Shapes
//!
//! Every tensor carries its EXACT static ONNX shape ([`TensorInfo::shape`]).
//! The graph input may have any rank >= 1; a symbolic leading dim of a
//! rank >= 2 input is the batch and resolves to 1, every other dim must be a
//! positive constant. Element order is ONNX's flat C order throughout, so
//! the input's flat index is the bundle's `onnx-input0-flat/v1` index (a
//! size-1 batch dimension does not move it) and a Relu site's flat index is
//! the producer's `onnx-relu-input/v1` index. The admitted forms that need a
//! batch require it to be 1: Conv `[1, C, H, W]`, Gemm `[1, K]`,
//! BatchNormalization `[1, C, ...]`, and the graph output `[1, N]`.
//! Broadcasting is ONNX's right-aligned rule, admitted only when the
//! activation operand does not grow (each constant dim is 1 or equal to the
//! aligned activation dim; extra leading constant dims are 1), and only from
//! opset 7: below it Add / Sub / Mul / Div broadcast by the legacy
//! `broadcast` / `axis` attributes, so a constant operand there (or such an
//! attribute anywhere) is refused.
//!
//! Anything else — another op, a non-default domain, a sub-graph attribute,
//! external data, a NaN or Inf parameter, a constant-only op (constant
//! folding), an activation-dependent index, a reshape that changes the
//! element count, an implementation-defined Slice edge — is
//! [`LowerError::Unsupported`] (report verdict `unsupported`, exit 31) with a
//! reason naming the op and the condition.
//!
//! Tensor identity is the ONNX tensor NAME, so a proof literal naming a Relu
//! node's `input[0]` resolves to the same mathematical quantity the producer
//! split, whatever the producer folded (the margin-row lane folds BN into the
//! preceding Conv; the Relu's input tensor is the BN output either way).

use std::collections::HashMap;

use crate::arith::{down, up};
use crate::onnx_wire::{Dim, Model, Node, Tensor as WireTensor, DT_FLOAT, DT_INT64};

/// Why the lowering refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    /// Outside the admitted op set or shape discipline.
    Unsupported(String),
}

impl LowerError {
    /// Log-ready text.
    #[must_use]
    pub fn text(&self) -> &str {
        match self {
            Self::Unsupported(s) => s,
        }
    }
}

fn unsup<T>(what: impl Into<String>) -> Result<T, LowerError> {
    Err(LowerError::Unsupported(what.into()))
}

/// A tensor of the IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// ONNX name.
    pub name: String,
    /// The exact static ONNX shape (a symbolic batch dim of the graph input
    /// resolved to 1). Elements are in flat C order.
    pub shape: Vec<usize>,
    /// Element count.
    pub size: usize,
}

/// A convolution (group 1, dilation 1).
#[derive(Debug, Clone, PartialEq)]
pub struct Conv {
    /// Input tensor.
    pub x: usize,
    /// Weights `[co][ci][kh][kw]`.
    pub w: Vec<f64>,
    /// Bias `[co]`.
    pub b: Vec<f64>,
    /// Output channels.
    pub co: usize,
    /// Input channels.
    pub ci: usize,
    /// Kernel height.
    pub kh: usize,
    /// Kernel width.
    pub kw: usize,
    /// Stride (h, w).
    pub stride: (usize, usize),
    /// Pads (top, left).
    pub pad: (usize, usize),
    /// Input spatial (h, w).
    pub ihw: (usize, usize),
    /// Output spatial (h, w).
    pub ohw: (usize, usize),
}

/// A dense layer `y = W x + b`, `W` row-major `(n_out, n_in)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Gemm {
    /// Input tensor.
    pub x: usize,
    /// Row-major weights.
    pub w: Vec<f64>,
    /// Bias.
    pub b: Vec<f64>,
    /// Rows.
    pub n_out: usize,
    /// Columns.
    pub n_in: usize,
}

/// Unfolded inference BatchNormalization, `y = s·x + sh` per channel, with
/// `s ∈ [s_lo, s_hi]` and `sh ∈ [sh_lo, sh_hi]` certified enclosures of the
/// real-arithmetic parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Bn {
    /// Input tensor.
    pub x: usize,
    /// Channels.
    pub c: usize,
    /// Elements per channel.
    pub hw: usize,
    /// Scale lower ends.
    pub s_lo: Vec<f64>,
    /// Scale upper ends.
    pub s_hi: Vec<f64>,
    /// Scale midpoints (the value the backward pass multiplies by).
    pub s_mid: Vec<f64>,
    /// Upper bound on `|s - s_mid|` over the enclosure.
    pub s_rad: Vec<f64>,
    /// Shift lower ends.
    pub sh_lo: Vec<f64>,
    /// Shift upper ends.
    pub sh_hi: Vec<f64>,
}

/// Exact index routing (Slice, Gather with constant indices, Concat): output
/// element `k` is a copy of element `src[k].1` of tensor `xs[src[k].0]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Select {
    /// Source tensors in operand order (a tensor may repeat: `Concat(a, a)`).
    pub xs: Vec<usize>,
    /// Per output element: `(operand slot in xs, flat element of that operand)`.
    pub src: Vec<(usize, usize)>,
}

impl Select {
    /// Each DISTINCT source tensor (first-occurrence order) with its
    /// backward fan-in: the largest number of output elements that read any
    /// one of its elements (>= 1).
    #[must_use]
    pub fn sources(&self, tensors: &[TensorInfo]) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = Vec::new();
        for &t in &self.xs {
            if out.iter().any(|&(u, _)| u == t) {
                continue;
            }
            let mut count = vec![0usize; tensors.get(t).map_or(0, |ti| ti.size)];
            for &(s, i) in &self.src {
                if self.xs.get(s) == Some(&t) {
                    if let Some(c) = count.get_mut(i) {
                        *c += 1;
                    }
                }
            }
            out.push((t, count.iter().copied().max().unwrap_or(0).max(1)));
        }
        out
    }
}

/// One IR operation.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// Convolution.
    Conv(Conv),
    /// Dense layer.
    Gemm(Gemm),
    /// BatchNormalization (unfolded).
    Bn(Bn),
    /// Exact index routing (Slice / Gather / Concat).
    Select(Select),
    /// Elementwise interval-affine `y = s·x + sh` (Sub / Mul / Div / Neg by
    /// constants): BN's arithmetic with `c` = the element count, `hw = 1`.
    Scale(Bn),
    /// Sum of two same-shape tensors.
    Add {
        /// Left.
        a: usize,
        /// Right.
        b: usize,
    },
    /// Tensor plus a constant (broadcast to full size).
    AddConst {
        /// Input.
        x: usize,
        /// Constant, one value per element.
        c: Vec<f64>,
    },
    /// Elementwise ReLU.
    Relu {
        /// Input (a "site" tensor).
        x: usize,
    },
    /// Flatten / flatten-equivalent Reshape.
    Identity {
        /// Input.
        x: usize,
    },
}

impl Op {
    /// Input tensors.
    #[must_use]
    pub fn inputs(&self) -> Vec<usize> {
        match self {
            Self::Conv(c) => vec![c.x],
            Self::Gemm(g) => vec![g.x],
            Self::Bn(b) | Self::Scale(b) => vec![b.x],
            Self::Select(s) => s.xs.clone(),
            Self::Add { a, b } => vec![*a, *b],
            Self::AddConst { x, .. } | Self::Relu { x } | Self::Identity { x } => vec![*x],
        }
    }

    /// A stable name for reports.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Conv(_) => "Conv",
            Self::Gemm(_) => "Gemm",
            Self::Bn(_) => "BatchNormalization",
            Self::Select(_) => "Select",
            Self::Scale(_) => "Scale",
            Self::Add { .. } => "Add",
            Self::AddConst { .. } => "AddConst",
            Self::Relu { .. } => "Relu",
            Self::Identity { .. } => "Identity",
        }
    }
}

/// One IR node: an op and the tensor it produces.
#[derive(Debug, Clone, PartialEq)]
pub struct IrNode {
    /// The op.
    pub op: Op,
    /// Output tensor.
    pub out: usize,
}

/// The lowered network. `nodes` are in topological order; `producer[t]` is
/// the node producing tensor `t` (`None` for the input).
#[derive(Debug, Clone, PartialEq)]
pub struct Net {
    /// Tensors.
    pub tensors: Vec<TensorInfo>,
    /// Nodes, topologically ordered.
    pub nodes: Vec<IrNode>,
    /// Input tensor id.
    pub input: usize,
    /// Output tensor id.
    pub output: usize,
    /// Producer node per tensor.
    pub producer: Vec<Option<usize>>,
    /// For each tensor, the Relu node consuming it (a proof "site"), if any.
    pub relu_consumer: Vec<Option<usize>>,
    /// Name -> tensor id.
    pub by_name: HashMap<String, usize>,
}

impl Net {
    /// Input size.
    #[must_use]
    pub fn n_in(&self) -> usize {
        self.tensors[self.input].size
    }

    /// Output size.
    #[must_use]
    pub fn n_out(&self) -> usize {
        self.tensors[self.output].size
    }

    /// Whether tensor `t` is the input of some Relu (a legal literal site).
    #[must_use]
    pub fn is_site(&self, t: usize) -> bool {
        self.relu_consumer.get(t).is_some_and(Option::is_some)
    }

    /// Topological index of the node producing `t` (`None` for the input).
    #[must_use]
    pub fn topo(&self, t: usize) -> Option<usize> {
        self.producer.get(t).copied().flatten()
    }

    /// ReLU neurons (total size of site tensors).
    #[must_use]
    pub fn relu_neurons(&self) -> usize {
        (0..self.tensors.len())
            .filter(|&t| self.is_site(t))
            .map(|t| self.tensors[t].size)
            .sum()
    }
}

fn attr_ints<'a>(n: &'a Node, name: &str) -> Option<&'a [i64]> {
    n.attr(name).map(|a| a.ints.as_slice())
}

fn attr_i(n: &Node, name: &str) -> Option<i64> {
    n.attr(name).and_then(|a| a.i)
}

fn attr_f(n: &Node, name: &str) -> Option<f32> {
    n.attr(name).and_then(|a| a.f)
}

fn to_usize(v: i64, what: &str) -> Result<usize, LowerError> {
    usize::try_from(v).map_err(|_| LowerError::Unsupported(format!("{what}: negative value {v}")))
}

fn finite(v: &[f64], what: &str) -> Result<(), LowerError> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        unsup(format!("{what}: NaN or Inf parameter"))
    }
}

/// Element count of a shape (`None` on overflow; the empty shape is 1).
fn numel(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d))
}

/// An ONNX axis normalized against rank `r`: `[-r, r)` (or `[-r, r]` when
/// `inclusive`, Flatten's range) maps to `[0, r)` (`[0, r]`).
fn norm_axis(axis: i64, r: usize, inclusive: bool, what: &str) -> Result<usize, LowerError> {
    let r = r as i128;
    let a = i128::from(axis);
    let a = if a < 0 { a + r } else { a };
    let hi = if inclusive { r } else { r - 1 };
    if a < 0 || a > hi {
        return unsup(format!("{what}: axis {axis} out of range for rank {r}"));
    }
    Ok(a as usize)
}

/// C-order strides of `shape`.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        s[k] = s[k + 1] * shape[k + 1];
    }
    s
}

/// The largest dense matrix [`Builder::lower_matmul`] materializes (unless
/// the constant operand is itself larger).
const MATMUL_DENSE_CAP: usize = 1 << 24;

struct Builder<'m> {
    inits: HashMap<String, &'m WireTensor>,
    consts: HashMap<String, WireTensor>,
    tensors: Vec<TensorInfo>,
    by_name: HashMap<String, usize>,
    nodes: Vec<IrNode>,
    /// The default-domain opset version, when the model declares one.
    opset: Option<i64>,
}

impl Builder<'_> {
    fn constant(&self, name: &str) -> Option<&WireTensor> {
        self.consts
            .get(name)
            .or_else(|| self.inits.get(name).copied())
    }

    fn act(&self, name: &str) -> Result<usize, LowerError> {
        self.by_name.get(name).copied().ok_or_else(|| {
            LowerError::Unsupported(format!(
                "tensor `{name}` is not a computed activation (graph order or constant operand)"
            ))
        })
    }

    fn new_tensor(&mut self, name: &str, shape: Vec<usize>) -> Result<usize, LowerError> {
        if self.by_name.contains_key(name)
            || self.inits.contains_key(name)
            || self.consts.contains_key(name)
        {
            return unsup(format!("tensor `{name}` defined twice"));
        }
        let size = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
        let Some(size) = size else {
            return unsup(format!("tensor `{name}`: size overflow"));
        };
        if size == 0 {
            return unsup(format!("tensor `{name}`: empty"));
        }
        let id = self.tensors.len();
        self.tensors.push(TensorInfo {
            name: name.to_string(),
            shape,
            size,
        });
        self.by_name.insert(name.to_string(), id);
        Ok(id)
    }

    fn floats(&self, name: &str, what: &str) -> Result<(Vec<i64>, Vec<f64>), LowerError> {
        let t = self.constant(name).ok_or_else(|| {
            LowerError::Unsupported(format!("{what}: `{name}` is not a constant"))
        })?;
        let v = t
            .floats()
            .map_err(|e| LowerError::Unsupported(format!("{what}: {e}")))?;
        finite(&v, what)?;
        Ok((t.dims.clone(), v))
    }

    /// A constant INT64 tensor `(dims, values)`; an activation is refused.
    fn ints(&self, name: &str, what: &str) -> Result<(Vec<i64>, Vec<i64>), LowerError> {
        if self.by_name.contains_key(name) {
            return unsup(format!(
                "{what}: `{name}` is computed from the input (only constant indices are admitted)"
            ));
        }
        let t = self.constant(name).ok_or_else(|| {
            LowerError::Unsupported(format!("{what}: `{name}` is not a constant"))
        })?;
        if t.data_type != DT_INT64 {
            return unsup(format!(
                "{what}: `{name}` is not INT64 (data type {})",
                t.data_type
            ));
        }
        let v = t
            .int64s()
            .map_err(|e| LowerError::Unsupported(format!("{what}: {e}")))?;
        Ok((t.dims.clone(), v))
    }

    /// A constant 1-D INT64 vector.
    fn int_vec(&self, name: &str, what: &str) -> Result<Vec<i64>, LowerError> {
        let (dims, v) = self.ints(name, what)?;
        if dims.len() != 1 {
            return unsup(format!("{what}: must be 1-D, has rank {}", dims.len()));
        }
        Ok(v)
    }

    /// An optional input slot: present and non-empty.
    fn opt_input(n: &Node, k: usize) -> Option<&str> {
        n.inputs
            .get(k)
            .map(String::as_str)
            .filter(|s| !s.is_empty())
    }

    /// Push an [`Op::Select`] producing `n`'s single output of `shape`.
    fn push_select(
        &mut self,
        n: &Node,
        xs: Vec<usize>,
        src: Vec<(usize, usize)>,
        shape: Vec<usize>,
    ) -> Result<(), LowerError> {
        if numel(&shape) != Some(src.len()) {
            return unsup(format!("{}: internal shape bookkeeping", n.op_type));
        }
        for &(s, i) in &src {
            let ok = xs.get(s).is_some_and(|&t| i < self.tensors[t].size);
            if !ok {
                return unsup(format!("{}: internal index bookkeeping", n.op_type));
            }
        }
        let out = self.new_tensor(Self::one_output(n)?, shape)?;
        self.nodes.push(IrNode {
            op: Op::Select(Select { xs, src }),
            out,
        });
        Ok(())
    }

    fn one_output(n: &Node) -> Result<&str, LowerError> {
        if n.outputs.len() != 1 || n.outputs[0].is_empty() {
            return unsup(format!("{}: exactly one output required", n.op_type));
        }
        Ok(&n.outputs[0])
    }

    fn lower_node(&mut self, n: &Node) -> Result<(), LowerError> {
        if !(n.domain.is_empty() || n.domain == "ai.onnx") {
            return unsup(format!("op {} in domain `{}`", n.op_type, n.domain));
        }
        if n.attrs.iter().any(|a| a.unsupported_payload) {
            return unsup(format!(
                "{}: sub-graph or unsupported attribute payload",
                n.op_type
            ));
        }
        match n.op_type.as_str() {
            "Constant" => {
                let out = Self::one_output(n)?;
                if n.attrs.len() != 1 || n.attrs[0].name != "value" {
                    return unsup("Constant: only the `value` attribute is supported");
                }
                let t = n.attrs[0]
                    .t
                    .clone()
                    .ok_or_else(|| LowerError::Unsupported("Constant: missing tensor".into()))?;
                if self.by_name.contains_key(out) || self.consts.contains_key(out) {
                    return unsup(format!("tensor `{out}` defined twice"));
                }
                self.consts.insert(out.to_string(), t);
                Ok(())
            }
            "Relu" => {
                let x = self.act(n.inputs.first().map_or("", String::as_str))?;
                if n.inputs.len() != 1 {
                    return unsup("Relu: one input");
                }
                let shape = self.tensors[x].shape.clone();
                let out = self.new_tensor(Self::one_output(n)?, shape)?;
                self.nodes.push(IrNode {
                    op: Op::Relu { x },
                    out,
                });
                Ok(())
            }
            "Flatten" => {
                if n.inputs.len() != 1 {
                    return unsup("Flatten: one input");
                }
                let x = self.act(&n.inputs[0])?;
                let shape = self.tensors[x].shape.clone();
                let a = norm_axis(attr_i(n, "axis").unwrap_or(1), shape.len(), true, "Flatten")?;
                let (Some(outer), Some(inner)) = (numel(&shape[..a]), numel(&shape[a..])) else {
                    return unsup("Flatten: size overflow");
                };
                let out = self.new_tensor(Self::one_output(n)?, vec![outer, inner])?;
                self.nodes.push(IrNode {
                    op: Op::Identity { x },
                    out,
                });
                Ok(())
            }
            "Reshape" => self.lower_reshape(n),
            "Slice" => self.lower_slice(n),
            "Gather" => self.lower_gather(n),
            "Concat" => self.lower_concat(n),
            "Identity" | "Dropout" => {
                if n.inputs.is_empty() || n.outputs.is_empty() || n.outputs[0].is_empty() {
                    return unsup(format!("{}: one input and one output", n.op_type));
                }
                if n.op_type == "Dropout" && n.inputs.get(2).is_some_and(|t| !t.is_empty()) {
                    return unsup("Dropout: training_mode input");
                }
                if n.op_type == "Identity" && n.inputs.len() != 1 {
                    return unsup("Identity: one input");
                }
                let x = self.act(&n.inputs[0])?;
                let shape = self.tensors[x].shape.clone();
                let out = self.new_tensor(&n.outputs[0], shape)?;
                self.nodes.push(IrNode {
                    op: Op::Identity { x },
                    out,
                });
                Ok(())
            }
            "MatMul" => self.lower_matmul(n),
            "Sub" => self.lower_sub(n),
            "Mul" | "Div" => self.lower_mul_div(n),
            "Neg" => {
                if n.inputs.len() != 1 {
                    return unsup("Neg: one input");
                }
                let x = self.act(&n.inputs[0])?;
                let size = self.tensors[x].size;
                let shape = self.tensors[x].shape.clone();
                let s = vec![(-1.0, -1.0); size];
                let sh = vec![(0.0, 0.0); size];
                self.push_scale(n, x, &s, &sh, shape)
            }
            "Add" => self.lower_add(n),
            "Conv" => self.lower_conv(n),
            "Gemm" => self.lower_gemm(n),
            "BatchNormalization" => self.lower_bn(n),
            other => unsup(format!(
                "op `{other}` is outside the checker's admission set"
            )),
        }
    }

    fn lower_reshape(&mut self, n: &Node) -> Result<(), LowerError> {
        if n.inputs.len() != 2 {
            return unsup("Reshape: two inputs");
        }
        let x = self.act(&n.inputs[0])?;
        if attr_i(n, "allowzero").unwrap_or(0) != 0 {
            return unsup("Reshape: allowzero");
        }
        if self.by_name.contains_key(&n.inputs[1]) {
            return unsup("Reshape: shape is computed from the input (not constant)");
        }
        if self.constant(&n.inputs[1]).is_none() {
            return unsup("Reshape: shape is not constant");
        }
        let dims = self.int_vec(&n.inputs[1], "Reshape shape")?;
        let size = self.tensors[x].size;
        // Resolve against the input's exact ONNX shape: `0` copies the dim at
        // the same axis (allowzero = 0), one `-1` is inferred from the count.
        let in_shape = self.tensors[x].shape.clone();
        let mut resolved: Vec<i128> = Vec::with_capacity(dims.len());
        let mut infer = None;
        for (ax, &d) in dims.iter().enumerate() {
            match d {
                -1 => {
                    if infer.is_some() {
                        return unsup("Reshape: two -1 dims");
                    }
                    infer = Some(ax);
                    resolved.push(1);
                }
                0 => {
                    let Some(&v) = in_shape.get(ax) else {
                        return unsup(format!(
                            "Reshape: dim {ax} is 0 (copy) but the input has rank {}",
                            in_shape.len()
                        ));
                    };
                    resolved.push(v as i128);
                }
                d if d > 0 => resolved.push(i128::from(d)),
                _ => return unsup(format!("Reshape: negative dim {d}")),
            }
        }
        let known: i128 = resolved.iter().product();
        let total = size as i128;
        if let Some(ax) = infer {
            if known == 0 || total % known != 0 {
                return unsup(format!(
                    "Reshape: cannot infer the -1 dim ({total} elements over {known})"
                ));
            }
            resolved[ax] = total / known;
        }
        if resolved.iter().product::<i128>() != total {
            return unsup(format!(
                "Reshape: element count changes ({total} -> {})",
                resolved.iter().product::<i128>()
            ));
        }
        let mut shape = Vec::with_capacity(resolved.len());
        for &d in &resolved {
            let Ok(d) = usize::try_from(d) else {
                return unsup("Reshape: dim out of range");
            };
            shape.push(d);
        }
        let out = self.new_tensor(Self::one_output(n)?, shape)?;
        self.nodes.push(IrNode {
            op: Op::Identity { x },
            out,
        });
        Ok(())
    }

    /// Slice (opset >= 10 input form): `data` an activation, `starts`/`ends`
    /// and the optional `axes`/`steps` constant 1-D INT64 vectors. Per axis
    /// (ONNX semantics): a negative start/end is counted from the end
    /// (`+ dim`); then for `step > 0` both clamp into `[0, dim]`, for
    /// `step < 0` the start into `[0, dim - 1]` and the end into
    /// `[-1, dim - 1]`; the axis keeps `ceil((end - start) / step)` elements
    /// `start + j·step`. A negative-step start that still lies before the
    /// first element (`start + dim < 0`) is refused: numpy (the ONNX
    /// reference) and the specification's clamp disagree there.
    fn lower_slice(&mut self, n: &Node) -> Result<(), LowerError> {
        if n.inputs.len() == 1 {
            return unsup(
                "Slice: the opset-1 attribute form (starts/ends attributes) is outside the admission set",
            );
        }
        if !(3..=5).contains(&n.inputs.len()) {
            return unsup("Slice: 3 to 5 inputs");
        }
        let x = self.act(&n.inputs[0]).map_err(|_| {
            LowerError::Unsupported(
                "Slice: data must be a computed activation (slicing a constant is constant folding)"
                    .into(),
            )
        })?;
        let shape = self.tensors[x].shape.clone();
        let r = shape.len();
        let starts = self.int_vec(&n.inputs[1], "Slice starts")?;
        let ends = self.int_vec(&n.inputs[2], "Slice ends")?;
        let m = starts.len();
        if m == 0 {
            return unsup("Slice: empty starts");
        }
        let axes = match Self::opt_input(n, 3) {
            Some(name) => self.int_vec(name, "Slice axes")?,
            None => (0..m as i64).collect(),
        };
        let steps = match Self::opt_input(n, 4) {
            Some(name) => self.int_vec(name, "Slice steps")?,
            None => vec![1; m],
        };
        if ends.len() != m || axes.len() != m || steps.len() != m {
            return unsup("Slice: starts/ends/axes/steps lengths differ");
        }
        // Per data axis: (first index, step, count); untouched axes keep all.
        let mut plan: Vec<(i128, i128, usize)> = shape.iter().map(|&d| (0, 1, d)).collect();
        let mut seen = vec![false; r];
        for j in 0..m {
            let a = norm_axis(axes[j], r, false, "Slice")?;
            if seen[a] {
                return unsup(format!("Slice: axis {a} repeated"));
            }
            seen[a] = true;
            let dim = shape[a] as i128;
            let st = i128::from(steps[j]);
            if st == 0 {
                return unsup("Slice: zero step");
            }
            let mut s = i128::from(starts[j]);
            let mut e = i128::from(ends[j]);
            if s < 0 {
                s += dim;
            }
            if e < 0 {
                e += dim;
            }
            let count = if st > 0 {
                s = s.clamp(0, dim);
                e = e.clamp(0, dim);
                if e > s {
                    (e - s + st - 1) / st
                } else {
                    0
                }
            } else {
                if s < 0 {
                    return unsup(format!(
                        "Slice: axis {a}: a negative-step start before the first element is \
                         implementation-defined"
                    ));
                }
                s = s.clamp(0, dim - 1);
                e = e.clamp(-1, dim - 1);
                if s > e {
                    (s - e - st - 1) / (-st)
                } else {
                    0
                }
            };
            if count <= 0 {
                return unsup(format!("Slice: axis {a} selects no element"));
            }
            plan[a] = (s, st, count as usize);
        }
        let out_shape: Vec<usize> = plan.iter().map(|p| p.2).collect();
        let Some(total) = numel(&out_shape) else {
            return unsup("Slice: size overflow");
        };
        let in_strides = strides(&shape);
        let mut src = Vec::with_capacity(total);
        let mut idx = vec![0usize; r];
        for _ in 0..total {
            let mut flat = 0usize;
            for a in 0..r {
                let (s0, st, _) = plan[a];
                let p = s0 + st * idx[a] as i128;
                // In range by construction: s0 in [0, dim) when the axis keeps
                // an element, and the count stops before the clamped end.
                if p < 0 || p >= shape[a] as i128 {
                    return unsup("Slice: internal index out of range");
                }
                flat += p as usize * in_strides[a];
            }
            src.push((0usize, flat));
            for a in (0..r).rev() {
                idx[a] += 1;
                if idx[a] < out_shape[a] {
                    break;
                }
                idx[a] = 0;
            }
        }
        self.push_select(n, vec![x], src, out_shape)
    }

    /// Gather with constant INT64 indices along `axis`: output shape
    /// `data[..axis] ++ indices ++ data[axis+1..]`.
    fn lower_gather(&mut self, n: &Node) -> Result<(), LowerError> {
        if n.inputs.len() != 2 {
            return unsup("Gather: two inputs");
        }
        if self.by_name.contains_key(&n.inputs[1]) {
            return unsup("Gather: indices computed from the input are outside the admission set");
        }
        let x = self.act(&n.inputs[0]).map_err(|_| {
            LowerError::Unsupported(
                "Gather: data must be a computed activation (gathering from a constant is constant folding)"
                    .into(),
            )
        })?;
        let (idims, idx) = self.ints(&n.inputs[1], "Gather indices")?;
        let shape = self.tensors[x].shape.clone();
        if shape.is_empty() {
            return unsup("Gather: rank-0 data");
        }
        let a = norm_axis(attr_i(n, "axis").unwrap_or(0), shape.len(), false, "Gather")?;
        let dim = shape[a];
        let mut ii = Vec::with_capacity(idx.len());
        for &v in &idx {
            let w = if v < 0 {
                if self.opset.is_none_or(|o| o < 11) {
                    return unsup(format!(
                        "Gather: negative index {v} needs opset >= 11 (declared {:?})",
                        self.opset
                    ));
                }
                i128::from(v) + dim as i128
            } else {
                i128::from(v)
            };
            if w < 0 || w >= dim as i128 {
                return unsup(format!("Gather: index {v} out of range for dim {dim}"));
            }
            ii.push(w as usize);
        }
        let mut idims_u = Vec::with_capacity(idims.len());
        for &d in &idims {
            idims_u.push(to_usize(d, "Gather indices dim")?);
        }
        if numel(&idims_u) != Some(ii.len()) {
            return unsup("Gather: indices dims do not match their data");
        }
        let mut out_shape = shape[..a].to_vec();
        out_shape.extend_from_slice(&idims_u);
        out_shape.extend_from_slice(&shape[a + 1..]);
        let outer: usize = shape[..a].iter().product();
        let inner: usize = shape[a + 1..].iter().product();
        let mut src = Vec::with_capacity(outer * ii.len() * inner);
        for o in 0..outer {
            for &q in &ii {
                for i in 0..inner {
                    src.push((0usize, (o * dim + q) * inner + i));
                }
            }
        }
        self.push_select(n, vec![x], src, out_shape)
    }

    /// Concat of activations along `axis` (equal rank, dims equal off the
    /// axis).
    fn lower_concat(&mut self, n: &Node) -> Result<(), LowerError> {
        if n.inputs.is_empty() {
            return unsup("Concat: no input");
        }
        let Some(axis) = attr_i(n, "axis") else {
            return unsup("Concat: the axis attribute is required");
        };
        let mut xs = Vec::with_capacity(n.inputs.len());
        for name in &n.inputs {
            let Some(&t) = self.by_name.get(name) else {
                return unsup(format!(
                    "Concat: operand `{name}` is not a computed activation (a constant operand is outside the admission set)"
                ));
            };
            xs.push(t);
        }
        let first = self.tensors[xs[0]].shape.clone();
        let r = first.len();
        if r == 0 {
            return unsup("Concat: rank-0 operand");
        }
        let a = norm_axis(axis, r, false, "Concat")?;
        let mut along = 0usize;
        for &t in &xs {
            let s = &self.tensors[t].shape;
            if s.len() != r || (0..r).any(|k| k != a && s[k] != first[k]) {
                return unsup(format!(
                    "Concat: operand shape {s:?} does not match {first:?} off axis {a}"
                ));
            }
            along = along
                .checked_add(s[a])
                .ok_or_else(|| LowerError::Unsupported("Concat: size overflow".into()))?;
        }
        let mut out_shape = first.clone();
        out_shape[a] = along;
        let outer: usize = first[..a].iter().product();
        let inner: usize = first[a + 1..].iter().product();
        let mut src = Vec::new();
        for o in 0..outer {
            for (slot, &t) in xs.iter().enumerate() {
                let d = self.tensors[t].shape[a];
                for j in 0..d {
                    for i in 0..inner {
                        src.push((slot, (o * d + j) * inner + i));
                    }
                }
            }
        }
        self.push_select(n, xs, src, out_shape)
    }

    /// Push an elementwise interval-affine node `y = s·x + sh` (one enclosure
    /// pair per element) as [`Op::Scale`].
    fn push_scale(
        &mut self,
        n: &Node,
        x: usize,
        s: &[(f64, f64)],
        sh: &[(f64, f64)],
        shape: Vec<usize>,
    ) -> Result<(), LowerError> {
        let size = self.tensors[x].size;
        if numel(&shape) != Some(size) {
            return unsup(format!("{}: internal shape bookkeeping", n.op_type));
        }
        if s.len() != size || sh.len() != size {
            return unsup(format!("{}: parameter length", n.op_type));
        }
        let mut bn = Bn {
            x,
            c: size,
            hw: 1,
            s_lo: Vec::with_capacity(size),
            s_hi: Vec::with_capacity(size),
            s_mid: Vec::with_capacity(size),
            s_rad: Vec::with_capacity(size),
            sh_lo: Vec::with_capacity(size),
            sh_hi: Vec::with_capacity(size),
        };
        for (&(s_lo, s_hi), &(sh_lo, sh_hi)) in s.iter().zip(sh) {
            if !matches!(
                s_lo.partial_cmp(&s_hi),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            ) || !matches!(
                sh_lo.partial_cmp(&sh_hi),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            ) {
                return unsup(format!("{}: non-ordered parameter enclosure", n.op_type));
            }
            let mid = if s_lo == s_hi {
                s_lo
            } else {
                0.5 * s_lo + 0.5 * s_hi
            };
            let rad = if s_lo == s_hi {
                0.0
            } else {
                up((s_hi - mid).max(mid - s_lo))
            };
            if ![s_lo, s_hi, mid, rad, sh_lo, sh_hi]
                .iter()
                .all(|v| v.is_finite())
            {
                return unsup(format!("{}: non-finite parameter enclosure", n.op_type));
            }
            bn.s_lo.push(s_lo);
            bn.s_hi.push(s_hi);
            bn.s_mid.push(mid);
            bn.s_rad.push(rad);
            bn.sh_lo.push(sh_lo);
            bn.sh_hi.push(sh_hi);
        }
        let out = self.new_tensor(Self::one_output(n)?, shape)?;
        self.nodes.push(IrNode {
            op: Op::Scale(bn),
            out,
        });
        Ok(())
    }

    /// MatMul with one activation `x` and one constant FLOAT operand `C`, in
    /// either order, under numpy semantics: a rank-1 left operand `[K]` is
    /// promoted to `[1, K]` (the 1 dropped from the output), a rank-1 right
    /// operand `[K]` to `[K, 1]` (likewise), and the batch dims (all but the
    /// last two) broadcast right-aligned (equal, or 1). The map is linear in
    /// `x`, so it lowers to [`Op::Gemm`] with the EXACT dense matrix
    /// (`n_out × n_in`, zero bias): every entry is either one of `C`'s values
    /// placed or `0`, nothing is computed (`x · W` row vectors give `Wᵀ`, as
    /// before). The dense matrix is refused above [`MATMUL_DENSE_CAP`] entries
    /// (unless `C` is itself larger).
    #[allow(clippy::too_many_lines)]
    fn lower_matmul(&mut self, n: &Node) -> Result<(), LowerError> {
        if n.inputs.len() != 2 {
            return unsup("MatMul: two inputs");
        }
        let a = self.by_name.get(&n.inputs[0]).copied();
        let b = self.by_name.get(&n.inputs[1]).copied();
        let (x, cname, x_left) = match (a, b) {
            (Some(x), None) => (x, &n.inputs[1], true),
            (None, Some(x)) => (x, &n.inputs[0], false),
            (Some(_), Some(_)) => {
                return unsup("MatMul: activation · activation is outside the admission set")
            }
            (None, None) => {
                return unsup(
                    "MatMul: both operands are constants (constant folding is outside the admission set)",
                )
            }
        };
        let (cd, cv) = self.floats(cname, "MatMul constant")?;
        let mut cdims = Vec::with_capacity(cd.len());
        for &d in &cd {
            cdims.push(to_usize(d, "MatMul constant dim")?);
        }
        let xdims = self.tensors[x].shape.clone();
        let (adims, bdims) = if x_left {
            (xdims, cdims)
        } else {
            (cdims, xdims)
        };
        if adims.is_empty() || bdims.is_empty() {
            return unsup("MatMul: a rank-0 operand");
        }
        let a_vec = adims.len() == 1;
        let b_vec = bdims.len() == 1;
        let a2 = if a_vec { vec![1, adims[0]] } else { adims };
        let b2 = if b_vec { vec![bdims[0], 1] } else { bdims };
        let (m, ka) = (a2[a2.len() - 2], a2[a2.len() - 1]);
        let (kb, nn) = (b2[b2.len() - 2], b2[b2.len() - 1]);
        if ka != kb {
            return unsup(format!("MatMul: inner dims differ ({ka} vs {kb})"));
        }
        let k = ka;
        let ab = &a2[..a2.len() - 2];
        let bb = &b2[..b2.len() - 2];
        // Broadcast the batch dims (right-aligned).
        let rank = ab.len().max(bb.len());
        let pad = |s: &[usize]| -> Vec<usize> {
            let mut v = vec![1usize; rank - s.len()];
            v.extend_from_slice(s);
            v
        };
        let (pa, pb) = (pad(ab), pad(bb));
        let mut ob = Vec::with_capacity(rank);
        for (&da, &db) in pa.iter().zip(&pb) {
            ob.push(if da == db || db == 1 {
                da
            } else if da == 1 {
                db
            } else {
                return unsup(format!(
                    "MatMul: batch dims {pa:?} and {pb:?} do not broadcast"
                ));
            });
        }
        let mut oshape = ob.clone();
        if !a_vec {
            oshape.push(m);
        }
        if !b_vec {
            oshape.push(nn);
        }
        let n_in = self.tensors[x].size;
        let Some(nb) = numel(&ob) else {
            return unsup("MatMul: size overflow");
        };
        let Some(n_out) = nb.checked_mul(m).and_then(|v| v.checked_mul(nn)) else {
            return unsup("MatMul: size overflow");
        };
        let Some(dense) = n_out.checked_mul(n_in) else {
            return unsup("MatMul: size overflow");
        };
        if dense > MATMUL_DENSE_CAP.max(cv.len()) {
            return unsup(format!(
                "MatMul: the dense lowering needs {dense} entries (cap {MATMUL_DENSE_CAP})"
            ));
        }
        let (sa, sb) = (strides(&pa), strides(&pb));
        let so = strides(&ob);
        let mut w = vec![0.0f64; dense];
        for o in 0..nb {
            // The operands' batch offsets of output batch element `o`.
            let (mut ia, mut ib) = (0usize, 0usize);
            for d in 0..rank {
                let q = (o / so[d]) % ob[d];
                if pa[d] != 1 {
                    ia += q * sa[d];
                }
                if pb[d] != 1 {
                    ib += q * sb[d];
                }
            }
            for mi in 0..m {
                for ni in 0..nn {
                    let row = (o * m + mi) * nn + ni;
                    for kk in 0..k {
                        let a_idx = (ia * m + mi) * k + kk;
                        let b_idx = (ib * k + kk) * nn + ni;
                        let (col, v) = if x_left {
                            (a_idx, cv[b_idx])
                        } else {
                            (b_idx, cv[a_idx])
                        };
                        w[row * n_in + col] = v;
                    }
                }
            }
        }
        let out = self.new_tensor(Self::one_output(n)?, oshape)?;
        if self.tensors[out].size != n_out {
            return unsup("MatMul: internal shape bookkeeping");
        }
        self.nodes.push(IrNode {
            op: Op::Gemm(Gemm {
                x,
                w,
                b: vec![0.0; n_out],
                n_out,
                n_in,
            }),
            out,
        });
        Ok(())
    }

    /// review(checker-lowering): Add / Sub / Mul / Div with a constant
    /// operand go through [`broadcast`]'s RIGHT-ALIGNED rule, which is the
    /// rule only from opset 7. Below it these ops broadcast by the legacy
    /// `broadcast` / `axis` attributes (`B` aligned at `axis`), so a `[3]`
    /// constant against `[1, 3, 2, 3]` with `axis = 1` is per-channel there
    /// and last-dim here: the same bytes, two networks. Refused rather than
    /// guessed (a model without a default-domain opset counts as legacy, as
    /// for Gather's negative indices); from opset 7 the attributes do not
    /// exist, so a node carrying one is refused too.
    fn legacy_broadcast_guard(&self, n: &Node) -> Result<(), LowerError> {
        if n.attr("broadcast").is_some() || n.attr("axis").is_some() {
            return unsup(format!(
                "{}: the legacy broadcast / axis attributes (opset < 7) are outside the admission set",
                n.op_type
            ));
        }
        if self.opset.is_none_or(|o| o < 7) {
            return unsup(format!(
                "{}: a constant operand below opset 7 (declared {:?}) broadcasts by the legacy \
                 attribute rule, outside the admission set",
                n.op_type, self.opset
            ));
        }
        Ok(())
    }

    /// `x − c` (an exact [`Op::AddConst`] of `−c`: binary negation is exact)
    /// or `c − x` ([`Op::Scale`] with `s = −1`, `sh = c`, both exact).
    fn lower_sub(&mut self, n: &Node) -> Result<(), LowerError> {
        if n.inputs.len() != 2 {
            return unsup("Sub: two inputs");
        }
        let a = self.by_name.get(&n.inputs[0]).copied();
        let b = self.by_name.get(&n.inputs[1]).copied();
        match (a, b) {
            (Some(x), None) => {
                self.legacy_broadcast_guard(n)?;
                let (dims, v) = self.floats(&n.inputs[1], "Sub constant")?;
                let (c, shape) = broadcast(&dims, &v, &self.tensors[x].shape)?;
                let c: Vec<f64> = c.iter().map(|v| -v).collect();
                let out = self.new_tensor(Self::one_output(n)?, shape)?;
                self.nodes.push(IrNode {
                    op: Op::AddConst { x, c },
                    out,
                });
                Ok(())
            }
            (None, Some(x)) => {
                self.legacy_broadcast_guard(n)?;
                let (dims, v) = self.floats(&n.inputs[0], "Sub constant")?;
                let (c, shape) = broadcast(&dims, &v, &self.tensors[x].shape)?;
                let s = vec![(-1.0, -1.0); c.len()];
                let sh: Vec<(f64, f64)> = c.iter().map(|&v| (v, v)).collect();
                self.push_scale(n, x, &s, &sh, shape)
            }
            (Some(_), Some(_)) => {
                unsup("Sub: activation − activation is outside the admission set")
            }
            (None, None) => unsup("Sub: both operands are constants"),
        }
    }

    /// `x · c` / `c · x` (exact `s = c`) and `x / c` (`s = 1/c` enclosed one
    /// ulp outward on each side; `c = 0` is refused).
    fn lower_mul_div(&mut self, n: &Node) -> Result<(), LowerError> {
        if n.inputs.len() != 2 {
            return unsup(format!("{}: two inputs", n.op_type));
        }
        let div = n.op_type == "Div";
        let a = self.by_name.get(&n.inputs[0]).copied();
        let b = self.by_name.get(&n.inputs[1]).copied();
        let (x, cname) = match (a, b) {
            (Some(x), None) => (x, &n.inputs[1]),
            (None, Some(x)) if !div => (x, &n.inputs[0]),
            (None, Some(_)) => return unsup("Div: constant / activation is not affine"),
            (Some(_), Some(_)) => {
                return unsup(format!(
                    "{}: activation × activation is outside the admission set",
                    n.op_type
                ))
            }
            (None, None) => return unsup(format!("{}: both operands are constants", n.op_type)),
        };
        self.legacy_broadcast_guard(n)?;
        let (dims, v) = self.floats(cname, "Mul/Div constant")?;
        let (c, shape) = broadcast(&dims, &v, &self.tensors[x].shape)?;
        let mut s = Vec::with_capacity(c.len());
        for &cv in &c {
            if div {
                if cv == 0.0 {
                    return unsup("Div: division by zero");
                }
                let q = 1.0 / cv;
                s.push((down(q), up(q)));
            } else {
                s.push((cv, cv));
            }
        }
        let sh = vec![(0.0, 0.0); c.len()];
        self.push_scale(n, x, &s, &sh, shape)
    }

    fn lower_add(&mut self, n: &Node) -> Result<(), LowerError> {
        if n.inputs.len() != 2 {
            return unsup("Add: two inputs");
        }
        let a = self.by_name.get(&n.inputs[0]).copied();
        let b = self.by_name.get(&n.inputs[1]).copied();
        match (a, b) {
            (Some(a), Some(b)) => {
                if self.tensors[a].shape != self.tensors[b].shape {
                    return unsup("Add: operand shapes differ");
                }
                let shape = self.tensors[a].shape.clone();
                let out = self.new_tensor(Self::one_output(n)?, shape)?;
                self.nodes.push(IrNode {
                    op: Op::Add { a, b },
                    out,
                });
                Ok(())
            }
            (Some(x), None) | (None, Some(x)) => {
                let cname = if a.is_some() {
                    &n.inputs[1]
                } else {
                    &n.inputs[0]
                };
                self.legacy_broadcast_guard(n)?;
                let (dims, v) = self.floats(cname, "Add constant")?;
                let (c, shape) = broadcast(&dims, &v, &self.tensors[x].shape)?;
                let out = self.new_tensor(Self::one_output(n)?, shape)?;
                self.nodes.push(IrNode {
                    op: Op::AddConst { x, c },
                    out,
                });
                Ok(())
            }
            (None, None) => unsup("Add: both operands are constants"),
        }
    }

    fn lower_conv(&mut self, n: &Node) -> Result<(), LowerError> {
        if n.inputs.len() < 2 || n.inputs.len() > 3 {
            return unsup("Conv: 2 or 3 inputs");
        }
        let x = self.act(&n.inputs[0])?;
        let full = self.tensors[x].shape.clone();
        if full.len() != 4 || full[0] != 1 {
            return unsup(format!("Conv: input must be [1, C, H, W], is {full:?}"));
        }
        let ishape = &full[1..];
        let (wd, w) = self.floats(&n.inputs[1], "Conv weight")?;
        if wd.len() != 4 {
            return unsup("Conv: weight must be 4-D");
        }
        let co = to_usize(wd[0], "Conv")?;
        let ci = to_usize(wd[1], "Conv")?;
        let kh = to_usize(wd[2], "Conv")?;
        let kw = to_usize(wd[3], "Conv")?;
        if ci != ishape[0] || co == 0 || kh == 0 || kw == 0 {
            return unsup("Conv: weight/input channel mismatch");
        }
        let b = match n.inputs.get(2).filter(|s| !s.is_empty()) {
            Some(name) => {
                let (_, b) = self.floats(name, "Conv bias")?;
                if b.len() != co {
                    return unsup("Conv: bias length");
                }
                b
            }
            None => vec![0.0; co],
        };
        if let Some(a) = n.attr("auto_pad") {
            if a.s.as_deref() != Some(b"NOTSET") {
                return unsup("Conv: auto_pad");
            }
        }
        if attr_i(n, "group").unwrap_or(1) != 1 {
            return unsup("Conv: group != 1");
        }
        if attr_ints(n, "dilations").is_some_and(|d| d.iter().any(|&v| v != 1)) {
            return unsup("Conv: dilation != 1");
        }
        if let Some(ks) = attr_ints(n, "kernel_shape") {
            if ks != [kh as i64, kw as i64] {
                return unsup("Conv: kernel_shape mismatch");
            }
        }
        let strides = attr_ints(n, "strides").map_or(vec![1, 1], <[i64]>::to_vec);
        let pads = attr_ints(n, "pads").map_or(vec![0, 0, 0, 0], <[i64]>::to_vec);
        if strides.len() != 2 || pads.len() != 4 {
            return unsup("Conv: 2-D strides and 4 pads required");
        }
        let sh = to_usize(strides[0], "Conv stride")?;
        let sw = to_usize(strides[1], "Conv stride")?;
        if sh == 0 || sw == 0 {
            return unsup("Conv: zero stride");
        }
        let (pt, pl, pb, pr) = (
            to_usize(pads[0], "Conv pad")?,
            to_usize(pads[1], "Conv pad")?,
            to_usize(pads[2], "Conv pad")?,
            to_usize(pads[3], "Conv pad")?,
        );
        let (ih, iw) = (ishape[1], ishape[2]);
        let eh = (ih + pt + pb).checked_sub(kh);
        let ew = (iw + pl + pr).checked_sub(kw);
        let (Some(eh), Some(ew)) = (eh, ew) else {
            return unsup("Conv: kernel larger than padded input");
        };
        let oh = eh / sh + 1;
        let ow = ew / sw + 1;
        let out = self.new_tensor(Self::one_output(n)?, vec![1, co, oh, ow])?;
        self.nodes.push(IrNode {
            op: Op::Conv(Conv {
                x,
                w,
                b,
                co,
                ci,
                kh,
                kw,
                stride: (sh, sw),
                pad: (pt, pl),
                ihw: (ih, iw),
                ohw: (oh, ow),
            }),
            out,
        });
        Ok(())
    }

    fn lower_gemm(&mut self, n: &Node) -> Result<(), LowerError> {
        if n.inputs.len() < 2 || n.inputs.len() > 3 {
            return unsup("Gemm: 2 or 3 inputs");
        }
        let x = self.act(&n.inputs[0])?;
        let xs = &self.tensors[x].shape;
        if xs.len() != 2 || xs[0] != 1 {
            return unsup(format!("Gemm: input must be [1, K], is {xs:?}"));
        }
        if attr_f(n, "alpha").is_some_and(|a| a != 1.0)
            || attr_f(n, "beta").is_some_and(|b| b != 1.0)
        {
            return unsup("Gemm: alpha/beta must be 1");
        }
        if attr_i(n, "transA").unwrap_or(0) != 0 {
            return unsup("Gemm: transA");
        }
        let trans_b = attr_i(n, "transB").unwrap_or(0) != 0;
        let (bd, raw) = self.floats(&n.inputs[1], "Gemm weight")?;
        if bd.len() != 2 {
            return unsup("Gemm: weight must be 2-D");
        }
        let d0 = to_usize(bd[0], "Gemm")?;
        let d1 = to_usize(bd[1], "Gemm")?;
        let (w, n_out, n_in) = if trans_b {
            (raw, d0, d1)
        } else {
            let mut w = vec![0.0; d0 * d1];
            for i in 0..d0 {
                for j in 0..d1 {
                    w[j * d0 + i] = raw[i * d1 + j];
                }
            }
            (w, d1, d0)
        };
        if n_in != self.tensors[x].size {
            return unsup("Gemm: input width mismatch");
        }
        let b = match n.inputs.get(2).filter(|s| !s.is_empty()) {
            Some(name) => {
                let (cd, c) = self.floats(name, "Gemm bias")?;
                let ok_shape = cd.iter().filter(|&&d| d != 1).count() <= 1;
                if !ok_shape {
                    return unsup("Gemm: bias must be a vector");
                }
                if c.len() == n_out {
                    c
                } else if c.len() == 1 {
                    vec![c[0]; n_out]
                } else {
                    return unsup("Gemm: bias length");
                }
            }
            None => vec![0.0; n_out],
        };
        let out = self.new_tensor(Self::one_output(n)?, vec![1, n_out])?;
        self.nodes.push(IrNode {
            op: Op::Gemm(Gemm {
                x,
                w,
                b,
                n_out,
                n_in,
            }),
            out,
        });
        Ok(())
    }

    fn lower_bn(&mut self, n: &Node) -> Result<(), LowerError> {
        if n.inputs.len() != 5 {
            return unsup("BatchNormalization: five inputs");
        }
        if n.outputs.len() != 1 {
            return unsup("BatchNormalization: training-mode outputs");
        }
        if attr_i(n, "training_mode").unwrap_or(0) != 0 {
            return unsup("BatchNormalization: training_mode");
        }
        let x = self.act(&n.inputs[0])?;
        let shape = self.tensors[x].shape.clone();
        if shape.len() < 2 || shape[0] != 1 {
            return unsup(format!(
                "BatchNormalization: input must be [1, C, ...], is {shape:?}"
            ));
        }
        let c = shape[1];
        let size = self.tensors[x].size;
        let hw = size / c;
        let gamma = self.floats(&n.inputs[1], "BN scale")?.1;
        let beta = self.floats(&n.inputs[2], "BN bias")?.1;
        let mean = self.floats(&n.inputs[3], "BN mean")?.1;
        let var = self.floats(&n.inputs[4], "BN var")?.1;
        if [&gamma, &beta, &mean, &var].iter().any(|v| v.len() != c) {
            return unsup("BatchNormalization: parameter length");
        }
        let eps = f64::from(attr_f(n, "epsilon").unwrap_or(1e-5));
        if !eps.is_finite() {
            return unsup("BatchNormalization: epsilon");
        }
        let mut bn = Bn {
            x,
            c,
            hw,
            s_lo: vec![0.0; c],
            s_hi: vec![0.0; c],
            s_mid: vec![0.0; c],
            s_rad: vec![0.0; c],
            sh_lo: vec![0.0; c],
            sh_hi: vec![0.0; c],
        };
        for ch in 0..c {
            let Some((s_lo, s_hi)) = bn_scale(gamma[ch], var[ch], eps) else {
                return unsup("BatchNormalization: var + eps not positive or non-finite scale");
            };
            let mid = f64::midpoint(s_lo, s_hi);
            let rad = up((s_hi - mid).max(mid - s_lo));
            // shift = beta - s * mean, s in [s_lo, s_hi].
            let p = [s_lo * mean[ch], s_hi * mean[ch]];
            let (p_lo, p_hi) = (down(p[0].min(p[1])), up(p[0].max(p[1])));
            let sh_lo = down(beta[ch] - p_hi);
            let sh_hi = up(beta[ch] - p_lo);
            if ![mid, rad, sh_lo, sh_hi].iter().all(|v| v.is_finite()) {
                return unsup("BatchNormalization: non-finite parameter enclosure");
            }
            bn.s_lo[ch] = s_lo;
            bn.s_hi[ch] = s_hi;
            bn.s_mid[ch] = mid;
            bn.s_rad[ch] = rad;
            bn.sh_lo[ch] = sh_lo;
            bn.sh_hi[ch] = sh_hi;
        }
        let out = self.new_tensor(Self::one_output(n)?, shape)?;
        self.nodes.push(IrNode {
            op: Op::Bn(bn),
            out,
        });
        Ok(())
    }
}

/// `γ / √(var + ε)` enclosed: one outward step after every correctly rounded
/// operation. `None` when `var + ε` is not certifiably positive.
#[must_use]
pub fn bn_scale(gamma: f64, var: f64, eps: f64) -> Option<(f64, f64)> {
    let t = var + eps;
    let (t_lo, t_hi) = (down(t), up(t));
    if t_lo.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || !t_hi.is_finite() {
        return None;
    }
    let r_lo = down(t_lo.sqrt());
    let r_hi = up(t_hi.sqrt());
    if r_lo.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    let (lo, hi) = if gamma >= 0.0 {
        (down(gamma / r_hi), up(gamma / r_lo))
    } else {
        (down(gamma / r_lo), up(gamma / r_hi))
    };
    (lo.is_finite() && hi.is_finite()).then_some((lo, hi))
}

/// Broadcast a constant of ONNX dims `dims` against an activation of exact
/// `shape` by ONNX's right-aligned rule, admitted only when the activation
/// does not grow: every aligned constant dim is 1 or equal to the
/// activation's, and extra leading constant dims are 1 (they prepend size-1
/// dims to the result, which moves no element). Returns one constant value
/// per result element (flat C order) and the result's shape.
fn broadcast(
    dims: &[i64],
    v: &[f64],
    shape: &[usize],
) -> Result<(Vec<f64>, Vec<usize>), LowerError> {
    let mut cd = Vec::with_capacity(dims.len());
    for &d in dims {
        cd.push(to_usize(d, "constant operand dim")?);
    }
    if numel(&cd) != Some(v.len()) {
        return unsup("constant operand: dims do not match its data");
    }
    let extra = cd.len().saturating_sub(shape.len());
    if cd[..extra].iter().any(|&d| d != 1) {
        return unsup(format!(
            "constant operand {cd:?} would grow the activation {shape:?} (outside the admission set)"
        ));
    }
    let tail = &cd[extra..];
    let rank = shape.len();
    let off = rank - tail.len();
    let cpad: Vec<usize> = (0..rank)
        .map(|k| if k < off { 1 } else { tail[k - off] })
        .collect();
    if cpad.iter().zip(shape).any(|(&c, &s)| c != 1 && c != s) {
        return unsup(format!(
            "constant operand {cd:?} does not broadcast to the activation {shape:?} without growing it"
        ));
    }
    let mut out_shape = vec![1usize; extra];
    out_shape.extend_from_slice(shape);
    let size: usize = shape.iter().product();
    let (cs, ss) = (strides(&cpad), strides(shape));
    let mut out = Vec::with_capacity(size);
    for f in 0..size {
        let mut ci = 0usize;
        for k in 0..rank {
            if cpad[k] != 1 {
                ci += ((f / ss[k]) % shape[k]) * cs[k];
            }
        }
        out.push(v[ci]);
    }
    Ok((out, out_shape))
}

/// Lower a decoded model to the checker's IR.
pub fn lower(model: &Model) -> Result<Net, LowerError> {
    let g = &model.graph;
    if g.sparse_initializers {
        return unsup("sparse initializers");
    }
    let mut b = Builder {
        inits: g.initializers.iter().map(|t| (t.name.clone(), t)).collect(),
        consts: HashMap::new(),
        tensors: Vec::new(),
        by_name: HashMap::new(),
        nodes: Vec::new(),
        opset: model
            .opsets
            .iter()
            .find(|(d, _)| d.is_empty() || d == "ai.onnx")
            .map(|&(_, v)| v),
    };
    // Initializers of other element types are refused lazily: only an op that
    // reads one (through `floats` / `int64s`) can fail on it.
    let real_inputs: Vec<_> = g
        .inputs
        .iter()
        .filter(|i| !b.inits.contains_key(&i.name))
        .collect();
    if real_inputs.len() != 1 {
        return unsup(format!(
            "exactly one non-initializer graph input required, found {}",
            real_inputs.len()
        ));
    }
    let inp = real_inputs[0];
    if inp.elem_type.is_some_and(|e| e != DT_FLOAT) {
        return unsup("graph input is not FLOAT");
    }
    let dims = inp
        .dims
        .as_ref()
        .ok_or_else(|| LowerError::Unsupported("graph input has no shape".into()))?;
    if dims.is_empty() {
        return unsup("graph input is a scalar (rank 0)");
    }
    // Any rank >= 1, kept exactly (the bundle's onnx-input0-flat/v1 index is
    // the flat C-order index over it). A symbolic LEADING dim of a rank >= 2
    // input is the batch and resolves to 1; every other dim is a positive
    // constant.
    let mut shape = Vec::with_capacity(dims.len());
    for (k, d) in dims.iter().enumerate() {
        match d {
            Dim::Value(v) if *v > 0 => shape.push(*v as usize),
            Dim::Param(_) | Dim::Unknown if k == 0 && dims.len() >= 2 => shape.push(1),
            _ => {
                return unsup(format!(
                    "graph input dim {k} is symbolic or non-positive ({d:?}); only a leading batch \
                     dim of a rank >= 2 input may be symbolic"
                ))
            }
        }
    }
    let input = b.new_tensor(&inp.name, shape)?;
    for n in &g.nodes {
        b.lower_node(n)?;
    }
    if g.outputs.len() != 1 {
        return unsup("exactly one graph output required");
    }
    let output = b.act(&g.outputs[0].name)?;
    let oshape = &b.tensors[output].shape;
    if oshape.len() != 2 || oshape[0] != 1 {
        return unsup(format!("graph output must be [1, N], is {oshape:?}"));
    }
    let mut producer = vec![None; b.tensors.len()];
    let mut relu_consumer = vec![None; b.tensors.len()];
    for (k, node) in b.nodes.iter().enumerate() {
        producer[node.out] = Some(k);
        if let Op::Relu { x } = node.op {
            if relu_consumer[x].is_some() {
                return unsup("a tensor feeds two Relu nodes (ambiguous site)");
            }
            relu_consumer[x] = Some(k);
        }
    }
    if producer[output].is_none() {
        return unsup("graph output is the input");
    }
    Ok(Net {
        tensors: b.tensors,
        nodes: b.nodes,
        input,
        output,
        producer,
        relu_consumer,
        by_name: b.by_name,
    })
}
