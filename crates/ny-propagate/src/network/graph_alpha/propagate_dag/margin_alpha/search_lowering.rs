// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-root-alpha (ENGINE_PROGRAM §2 V2): lower the graph lane's IR into the
//! plain-data search copy the untrusted backend reads
//! ([`ny_core::search::GraphSearchNet`]), and SELF-CHECK the copy (G1).
//!
//! # What is lowered, and how
//!
//! Only the ancestors of the output node, in the lane's execution order.
//!
//! * Structural: `Linear` (a last-axis dense layer, the lane's own f32 weights
//!   widened exactly), `Conv2d` (gather form; `groups = 1`), `ReLU`,
//!   `BilinearCrown` / var×var `MatMul` (same-batch `[b][m][k]·[b][k][n]`, the
//!   lane's `alpha_shape`), `MulBinary` (operands broadcast by the lane's own
//!   `broadcast_flat_index_map`), `Exp` / `Reciprocal` / `Clip`, and the
//!   monolithic last-axis `Softmax` (with the lane's OWN frozen LSE affine
//!   bounds per group, `SoftmaxLayer::softmax_lse_affine_bounds`).
//! * Probed: every other admitted exact-affine node (reshapes, transposes,
//!   concats with embedded constants, slices, batch norms, constant arithmetic,
//!   residual adds, reductions) is recovered by evaluating the NODE'S OWN IBP
//!   at concrete inputs — an index-coded probe for pure data movement, an
//!   elementwise probe, then a bounded basis probe — and every recovered map is
//!   verified at a random point before it is accepted. So the axis, batch and
//!   broadcast conventions of those layers are the lane's own, not a copy.
//!
//! Anything else fails closed with a named refusal.
//!
//! # G1: the self-check
//!
//! [`self_check`] evaluates the lowered copy with its TRUE functions in f64 at
//! the box centre and at an interior point, and compares EVERY lowered node
//! with the lane's own point forward (`collect_node_activations_pointwise`).
//! A mismatch refuses the whole phase.
//!
//! # Trust
//!
//! Nothing here is certified and nothing certified reads it (R3): a lowering
//! bug costs tightness or a refusal. The certified boxes are copied as
//! REFERENCE boxes (f64 widening of the lane's f32 endpoints, exact).

use std::collections::{HashMap, HashSet};

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::search::{
    GraphSearchNet, GraphSearchOp, SearchConv, SparseTerm, TensorBox, UnaryKind, GATHER_PAD,
};
use ny_core::{GemmEngine, NyError};
use ny_tensor::BoundedTensor;

use crate::layers::{BilinearCrownLayer, Conv2dLayer, Layer, LinearLayer, SoftmaxLayer};
use crate::network::core::graph::ibp::dispatch::dispatch_ibp_for_node;
use crate::network::core::{GraphNetwork, GraphNode, NETWORK_INPUT};

/// The α family of a lowered site, with the lane's parameter shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SiteKind {
    /// One lower slope per pre-activation neuron.
    Relu,
    /// `[4, m, n, k]` interpolation parameters.
    Bilinear { m: usize, n: usize, k: usize },
    /// `[2, n]` interpolation parameters.
    MulBinary { n: usize },
}

/// One α site of the lowered copy.
#[derive(Debug, Clone)]
pub(crate) struct AlphaSite {
    /// Op index in the search copy.
    pub(crate) op: usize,
    /// The lane node the site's α is keyed by.
    pub(crate) node: String,
    /// A ReLU's pre-activation node (empty for products).
    pub(crate) input_node: String,
    pub(crate) kind: SiteKind,
}

/// The lowered copy plus the bookkeeping the ingest gate needs.
pub(crate) struct GraphLowering {
    pub(crate) net: GraphSearchNet,
    /// Reference boxes of every relaxed op's operands, by tensor id.
    pub(crate) boxes: Vec<TensorBox>,
    pub(crate) sites: Vec<AlphaSite>,
    /// Lane node → tensor id of the copy.
    pub(crate) tensor_of: HashMap<String, usize>,
    /// Nodes lowered structurally / recovered by probing.
    pub(crate) structural: usize,
    pub(crate) probed: usize,
}

/// What the self-check compared.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SelfCheck {
    /// (point, node) pairs compared.
    pub(crate) compared: usize,
    /// Worst scaled discrepancy seen (`|copy - lane| / (1 + max|lane|)`).
    pub(crate) worst: f64,
}

/// Probes of a basis probe (`Σ input sizes`) and of its output table.
const BASIS_MAX_INPUTS: usize = 8192;
const BASIS_MAX_CELLS: usize = 1 << 26;

/// Index codes stay exact in f32 below this (2^22) and the second code set is
/// shifted by it.
const CODE_OFFSET: f32 = 4_194_304.0;

/// Self-check tolerance, relative to `1 + max|lane value|` of the node.
pub(crate) const SELF_CHECK_TOL: f64 = 1e-3;

fn flat_f64(values: &ArrayD<f32>) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

/// Exact-affine layers the probe lowering may recover (anything else that is
/// not lowered structurally refuses).
fn probe_admissible(layer: &Layer) -> bool {
    matches!(
        layer,
        Layer::Reshape(_)
            | Layer::Flatten(_)
            | Layer::Transpose(_)
            | Layer::Concat(_)
            | Layer::Squeeze(_)
            | Layer::Unsqueeze(_)
            | Layer::Slice(_)
            | Layer::Gather(_)
            | Layer::Tile(_)
            | Layer::Pad(_)
            | Layer::Add(_)
            | Layer::Sub(_)
            | Layer::AddConstant(_)
            | Layer::SubConstant(_)
            | Layer::MulConstant(_)
            | Layer::DivConstant(_)
            | Layer::BatchNorm(_)
            | Layer::ReduceSum(_)
            | Layer::ReduceMean(_)
            | Layer::ExpandLikeLastAxis(_)
            | Layer::Linear(_)
            | Layer::Conv2d(_)
    )
}

/// Deterministic probe values in `[-1, 1]`.
fn random_inputs(sizes: &[usize], seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    sizes
        .iter()
        .map(|&s| {
            (0..s)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let unit = ((state >> 40) as f32) / ((1u64 << 24) as f32);
                    2.0 * unit - 1.0
                })
                .collect()
        })
        .collect()
}

/// One node evaluated by its own IBP at concrete inputs.
struct Probe<'a> {
    node: &'a GraphNode,
    name: &'a str,
    /// Distinct dynamic input names, first-appearance order.
    names: &'a [&'a str],
    shapes: &'a [Vec<usize>],
    out_size: usize,
}

impl Probe<'_> {
    /// The node's output at `values` (one flat vector per distinct input), as
    /// f64 interval midpoints; `None` when the node refuses or goes non-finite.
    fn eval(&self, values: &[Vec<f32>]) -> Option<Vec<f64>> {
        let mut tensors = Vec::with_capacity(values.len());
        for (v, shape) in values.iter().zip(self.shapes) {
            let array = ArrayD::from_shape_vec(IxDyn(shape), v.clone()).ok()?;
            tensors.push(BoundedTensor::concrete(array).ok()?);
        }
        let mut resolve = |input: &str| -> ny_core::Result<BoundedTensor> {
            self.names
                .iter()
                .position(|n| *n == input)
                .map(|i| tensors[i].clone())
                .ok_or_else(|| NyError::InternalError(format!("probe: unknown input '{input}'")))
        };
        let out = dispatch_ibp_for_node(self.node, self.name, &mut resolve).ok()?;
        if out.len() != self.out_size {
            return None;
        }
        let mid: Vec<f64> = out
            .lower()
            .iter()
            .zip(out.upper().iter())
            .map(|(&l, &u)| f64::midpoint(f64::from(l), f64::from(u)))
            .collect();
        mid.iter().all(|v| v.is_finite()).then_some(mid)
    }
}

type AffineMap = (Vec<SparseTerm>, Vec<f64>);

/// Pure data movement: every output is one input element or a constant.
fn probe_gather(p: &Probe<'_>, sizes: &[usize]) -> Option<AffineMap> {
    let total: usize = sizes.iter().sum();
    if total == 0 || total + 1 >= CODE_OFFSET as usize {
        return None;
    }
    let codes = |offset: f32| -> Vec<Vec<f32>> {
        let mut base = 0usize;
        sizes
            .iter()
            .map(|&s| {
                let v: Vec<f32> = (0..s).map(|i| (base + i + 1) as f32 + offset).collect();
                base += s;
                v
            })
            .collect()
    };
    let fa = p.eval(&codes(0.0))?;
    let fb = p.eval(&codes(CODE_OFFSET))?;
    let mut src: Vec<Option<(usize, usize)>> = Vec::with_capacity(p.out_size);
    let mut bias = vec![0.0f64; p.out_size];
    for o in 0..p.out_size {
        let (a, b) = (fa[o], fb[o]);
        if b - a == f64::from(CODE_OFFSET) && a.fract() == 0.0 && a >= 1.0 && a <= total as f64 {
            let mut code = a as usize - 1;
            let mut found = None;
            for (slot, &s) in sizes.iter().enumerate() {
                if code < s {
                    found = Some((slot, code));
                    break;
                }
                code -= s;
            }
            src.push(Some(found?));
        } else if a == b {
            src.push(None);
            bias[o] = a;
        } else {
            return None;
        }
    }
    let r = random_inputs(sizes, 0x6a7e);
    let fr = p.eval(&r)?;
    for o in 0..p.out_size {
        let expect = match src[o] {
            Some((slot, i)) => f64::from(r[slot][i]),
            None => bias[o],
        };
        if (fr[o] - expect).abs() > 1e-6 * (1.0 + expect.abs()) {
            return None;
        }
    }
    let terms = src
        .iter()
        .enumerate()
        .filter_map(|(o, s)| {
            s.map(|(slot, i)| SparseTerm {
                out: o as u32,
                slot: slot as u32,
                index: i as u32,
                weight: 1.0,
            })
        })
        .collect();
    Some((terms, bias))
}

/// Elementwise affine over equally sized inputs: `out[o] = bias[o] + Σ_k
/// c_k[o] · in_k[o]`.
fn probe_elementwise(p: &Probe<'_>, sizes: &[usize]) -> Option<AffineMap> {
    if sizes.iter().any(|&s| s != p.out_size) {
        return None;
    }
    let zeros: Vec<Vec<f32>> = sizes.iter().map(|&s| vec![0.0; s]).collect();
    let bias = p.eval(&zeros)?;
    let mut coef = vec![vec![0.0f64; p.out_size]; sizes.len()];
    for slot in 0..sizes.len() {
        let mut inputs = zeros.clone();
        inputs[slot] = vec![1.0; sizes[slot]];
        let f = p.eval(&inputs)?;
        for o in 0..p.out_size {
            coef[slot][o] = f[o] - bias[o];
        }
    }
    let r = random_inputs(sizes, 0x51ed);
    let fr = p.eval(&r)?;
    for o in 0..p.out_size {
        let mut expect = bias[o];
        let mut magnitude = bias[o].abs();
        for slot in 0..sizes.len() {
            let t = coef[slot][o] * f64::from(r[slot][o]);
            expect += t;
            magnitude += t.abs();
        }
        if (fr[o] - expect).abs() > 1e-4 * (1.0 + magnitude) {
            return None;
        }
    }
    let mut terms = Vec::new();
    for (slot, c) in coef.iter().enumerate() {
        for (o, &w) in c.iter().enumerate() {
            if w != 0.0 {
                terms.push(SparseTerm {
                    out: o as u32,
                    slot: slot as u32,
                    index: o as u32,
                    weight: w,
                });
            }
        }
    }
    Some((terms, bias))
}

/// Any affine map, one basis vector at a time (bounded).
fn probe_basis(p: &Probe<'_>, sizes: &[usize]) -> Option<AffineMap> {
    let total: usize = sizes.iter().sum();
    if total == 0 || total > BASIS_MAX_INPUTS || total.saturating_mul(p.out_size) > BASIS_MAX_CELLS
    {
        return None;
    }
    let mut inputs: Vec<Vec<f32>> = sizes.iter().map(|&s| vec![0.0; s]).collect();
    let bias = p.eval(&inputs)?;
    let floor = 4.0 * f64::from(f32::EPSILON);
    let mut terms = Vec::new();
    for (slot, &s) in sizes.iter().enumerate() {
        for i in 0..s {
            inputs[slot][i] = 1.0;
            let f = p.eval(&inputs);
            inputs[slot][i] = 0.0;
            let f = f?;
            for o in 0..p.out_size {
                let w = f[o] - bias[o];
                if w.abs() > floor * (1.0 + bias[o].abs()) {
                    terms.push(SparseTerm {
                        out: o as u32,
                        slot: slot as u32,
                        index: i as u32,
                        weight: w,
                    });
                }
            }
        }
    }
    let r = random_inputs(sizes, 0xba5e);
    let fr = p.eval(&r)?;
    let mut expect = bias.clone();
    let mut magnitude: Vec<f64> = bias.iter().map(|b| b.abs()).collect();
    for t in &terms {
        let v = t.weight * f64::from(r[t.slot as usize][t.index as usize]);
        expect[t.out as usize] += v;
        magnitude[t.out as usize] += v.abs();
    }
    for o in 0..p.out_size {
        if (fr[o] - expect[o]).abs() > 1e-4 * (1.0 + magnitude[o]) {
            return None;
        }
    }
    Some((terms, bias))
}

struct Builder<'a> {
    input: &'a BoundedTensor,
    node_bounds: &'a HashMap<String, BoundedTensor>,
    ops: Vec<GraphSearchOp>,
    /// Flat size of every tensor (0 = the input).
    sizes: Vec<usize>,
    boxes: HashMap<usize, (Vec<f64>, Vec<f64>)>,
    tensor_of: HashMap<String, usize>,
    sites: Vec<AlphaSite>,
    structural: usize,
    probed: usize,
}

impl<'a> Builder<'a> {
    fn tensor(&self, name: &str) -> Result<usize, String> {
        if name == NETWORK_INPUT {
            return Ok(0);
        }
        self.tensor_of
            .get(name)
            .copied()
            .ok_or_else(|| format!("'{name}' is read before it is lowered"))
    }

    fn bounds_of(&self, name: &str) -> Result<&'a BoundedTensor, String> {
        if name == NETWORK_INPUT {
            return Ok(self.input);
        }
        self.node_bounds
            .get(name)
            .ok_or_else(|| format!("no certified box for '{name}'"))
    }

    fn shape_of(&self, name: &str) -> Result<Vec<usize>, String> {
        Ok(self.bounds_of(name)?.shape().to_vec())
    }

    /// Record the certified box of `name` as the reference box of `tensor`.
    fn note_box(&mut self, tensor: usize, name: &str) -> Result<(), String> {
        if !self.boxes.contains_key(&tensor) {
            let b = self.bounds_of(name)?;
            self.boxes
                .insert(tensor, (flat_f64(b.lower()), flat_f64(b.upper())));
        }
        Ok(())
    }

    fn push(&mut self, op: GraphSearchOp, size: usize) -> usize {
        self.ops.push(op);
        self.sizes.push(size);
        self.ops.len()
    }

    fn unary_input(node: &GraphNode) -> Result<&str, String> {
        node.require_unary_input().map_err(|e| e.to_string())
    }

    fn lower_node(&mut self, name: &str, node: &GraphNode) -> Result<(), String> {
        let out_shape = self.shape_of(name)?;
        let out_size: usize = out_shape.iter().product();
        let tensor = match &node.layer {
            Layer::Linear(linear) => match self.lower_linear(node, linear, out_size)? {
                Some(t) => t,
                None => self.lower_probe(name, node, out_size)?,
            },
            Layer::Conv2d(conv) => match self.lower_conv(node, conv, &out_shape)? {
                Some(t) => t,
                None => self.lower_probe(name, node, out_size)?,
            },
            Layer::ReLU(_) => {
                let input = Self::unary_input(node)?;
                let t_in = self.tensor(input)?;
                self.note_box(t_in, input)?;
                let t = self.push(GraphSearchOp::Relu { input: t_in }, out_size);
                self.sites.push(AlphaSite {
                    op: t - 1,
                    node: name.to_string(),
                    input_node: input.to_string(),
                    kind: SiteKind::Relu,
                });
                self.structural += 1;
                t
            }
            Layer::BilinearCrown(layer) => {
                self.lower_bilinear(name, node, layer.transpose_b, layer.scale, out_size)?
            }
            Layer::MatMul(layer) => {
                self.lower_bilinear(name, node, layer.transpose_b, layer.scale, out_size)?
            }
            Layer::MulBinary(_) => self.lower_mul(name, node, &out_shape)?,
            Layer::Exp(_) => self.lower_unary(node, UnaryKind::Exp, out_size)?,
            Layer::Reciprocal(_) => self.lower_unary(node, UnaryKind::Reciprocal, out_size)?,
            Layer::Clip(clip) => self.lower_unary(
                node,
                UnaryKind::Clip {
                    lo: f64::from(clip.min),
                    hi: f64::from(clip.max),
                },
                out_size,
            )?,
            Layer::Softmax(softmax) => self.lower_softmax(node, softmax, &out_shape)?,
            layer if probe_admissible(layer) => self.lower_probe(name, node, out_size)?,
            layer => {
                return Err(format!(
                    "unsupported layer {} at '{name}'",
                    layer.layer_type()
                ))
            }
        };
        if self.sizes[tensor] != out_size {
            return Err(format!(
                "'{name}': lowered width {} disagrees with the node's {out_size}",
                self.sizes[tensor]
            ));
        }
        self.tensor_of.insert(name.to_string(), tensor);
        Ok(())
    }

    fn lower_linear(
        &mut self,
        node: &GraphNode,
        linear: &LinearLayer,
        out_size: usize,
    ) -> Result<Option<usize>, String> {
        let input = Self::unary_input(node)?;
        let shape = self.shape_of(input)?;
        let (n_in, n_out) = (linear.in_features(), linear.out_features());
        let size: usize = shape.iter().product();
        if n_in == 0 || shape.last() != Some(&n_in) || !size.is_multiple_of(n_in) {
            return Ok(None);
        }
        let rows = size / n_in;
        if rows * n_out != out_size {
            return Ok(None);
        }
        let weight: Vec<f64> = linear.weight().iter().map(|&v| f64::from(v)).collect();
        let bias = linear.bias().map_or_else(
            || vec![0.0; n_out],
            |b| b.iter().map(|&v| f64::from(v)).collect(),
        );
        if weight.len() != n_in * n_out || bias.len() != n_out {
            return Ok(None);
        }
        let t_in = self.tensor(input)?;
        self.structural += 1;
        Ok(Some(self.push(
            GraphSearchOp::Dense {
                input: t_in,
                rows,
                n_in,
                n_out,
                weight,
                bias,
            },
            out_size,
        )))
    }

    fn lower_conv(
        &mut self,
        node: &GraphNode,
        conv: &Conv2dLayer,
        out_shape: &[usize],
    ) -> Result<Option<usize>, String> {
        let input = Self::unary_input(node)?;
        let in_shape = self.shape_of(input)?;
        let kshape = conv.kernel.shape();
        if kshape.len() != 4 || conv.groups != 1 || in_shape.len() < 3 || out_shape.len() < 3 {
            return Ok(None);
        }
        let lead_ones = |s: &[usize]| s[..s.len() - 3].iter().all(|&d| d == 1);
        if !lead_ones(in_shape.as_slice()) || !lead_ones(out_shape) {
            return Ok(None);
        }
        let (cout, cin, kh, kw) = (kshape[0], kshape[1], kshape[2], kshape[3]);
        let r = in_shape.len();
        let (c, h, w) = (in_shape[r - 3], in_shape[r - 2], in_shape[r - 1]);
        let q = out_shape.len();
        let (oc, oh, ow) = (out_shape[q - 3], out_shape[q - 2], out_shape[q - 1]);
        if c != cin || oc != cout {
            return Ok(None);
        }
        let (sh, sw) = conv.stride;
        let (ph, pw) = conv.padding;
        let (dh, dw) = conv.dilation;
        let expected = |i: usize, k: usize, s: usize, p: usize, d: usize| -> Option<usize> {
            if s == 0 || k == 0 || d == 0 {
                return None;
            }
            let span = d.checked_mul(k - 1)?.checked_add(1)?;
            (i + 2 * p).checked_sub(span).map(|v| v / s + 1)
        };
        if expected(h, kh, sh, ph, dh) != Some(oh) || expected(w, kw, sw, pw, dw) != Some(ow) {
            return Ok(None);
        }
        let (k, p) = (cin * kh * kw, oh * ow);
        let mut gather = vec![GATHER_PAD; k * p];
        for oy in 0..oh {
            for ox in 0..ow {
                let sp = oy * ow + ox;
                for ci in 0..cin {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = (oy * sh + ky * dh) as isize - ph as isize;
                            let ix = (ox * sw + kx * dw) as isize - pw as isize;
                            if iy >= 0 && (iy as usize) < h && ix >= 0 && (ix as usize) < w {
                                let src = ci * h * w + (iy as usize) * w + ix as usize;
                                gather[(ci * kh * kw + ky * kw + kx) * p + sp] =
                                    u32::try_from(src).map_err(|e| e.to_string())?;
                            }
                        }
                    }
                }
            }
        }
        let weight = flat_f64(&conv.kernel);
        let bias = conv.bias.as_ref().map_or_else(
            || vec![0.0; cout],
            |b| b.iter().map(|&v| f64::from(v)).collect(),
        );
        let t_in = self.tensor(input)?;
        self.structural += 1;
        Ok(Some(self.push(
            GraphSearchOp::Conv(Box::new(SearchConv {
                input: t_in,
                cout,
                k,
                p,
                gather,
                weight,
                bias,
            })),
            cout * p,
        )))
    }

    fn lower_bilinear(
        &mut self,
        name: &str,
        node: &GraphNode,
        transpose_b: bool,
        scale: Option<f32>,
        out_size: usize,
    ) -> Result<usize, String> {
        let (a_name, b_name) = node.require_binary_inputs().map_err(|e| e.to_string())?;
        let (a_shape, b_shape) = (self.shape_of(a_name)?, self.shape_of(b_name)?);
        let layer = BilinearCrownLayer::try_new(transpose_b, scale).map_err(|e| e.to_string())?;
        let (m, n, k) = layer
            .alpha_shape(&a_shape, &b_shape)
            .map_err(|e| e.to_string())?;
        let size_a: usize = a_shape.iter().product();
        let size_b: usize = b_shape.iter().product();
        let per_a = m * k;
        if per_a == 0 || !size_a.is_multiple_of(per_a) {
            return Err(format!(
                "'{name}': operand {a_shape:?} is not [.., {m}, {k}]"
            ));
        }
        let batch = size_a / per_a;
        if batch * k * n != size_b || batch * m * n != out_size {
            return Err(format!(
                "'{name}': {a_shape:?} x {b_shape:?} is not a same-batch product"
            ));
        }
        let (ta, tb) = (self.tensor(a_name)?, self.tensor(b_name)?);
        self.note_box(ta, a_name)?;
        self.note_box(tb, b_name)?;
        let t = self.push(
            GraphSearchOp::Bilinear {
                a: ta,
                b: tb,
                batch,
                m,
                n,
                k,
                transpose_b,
                scale: f64::from(scale.unwrap_or(1.0)),
            },
            out_size,
        );
        self.sites.push(AlphaSite {
            op: t - 1,
            node: name.to_string(),
            input_node: String::new(),
            kind: SiteKind::Bilinear { m, n, k },
        });
        self.structural += 1;
        Ok(t)
    }

    /// `name`'s tensor at `out_shape`: itself when the shapes agree, else a
    /// broadcast gather (the lane's own index map) with the gathered box.
    fn broadcast_operand(&mut self, name: &str, out_shape: &[usize]) -> Result<usize, String> {
        let shape = self.shape_of(name)?;
        let t = self.tensor(name)?;
        if shape == out_shape {
            self.note_box(t, name)?;
            return Ok(t);
        }
        if crate::shape::broadcast_shapes(&shape, out_shape).as_deref() != Some(out_shape) {
            return Err(format!(
                "operand '{name}' {shape:?} does not broadcast to {out_shape:?}"
            ));
        }
        let map = crate::shape::broadcast_flat_index_map(out_shape, &shape);
        let b = self.bounds_of(name)?;
        let (l, u) = (flat_f64(b.lower()), flat_f64(b.upper()));
        if map.iter().any(|&i| i >= l.len()) {
            return Err(format!("operand '{name}': broadcast index out of range"));
        }
        let terms = map
            .iter()
            .enumerate()
            .map(|(o, &i)| SparseTerm {
                out: o as u32,
                slot: 0,
                index: i as u32,
                weight: 1.0,
            })
            .collect();
        let bl: Vec<f64> = map.iter().map(|&i| l[i]).collect();
        let bu: Vec<f64> = map.iter().map(|&i| u[i]).collect();
        let tt = self.push(
            GraphSearchOp::Sparse {
                inputs: vec![t],
                n_out: map.len(),
                terms,
                bias: vec![0.0; map.len()],
            },
            map.len(),
        );
        self.boxes.insert(tt, (bl, bu));
        Ok(tt)
    }

    fn lower_mul(
        &mut self,
        name: &str,
        node: &GraphNode,
        out_shape: &[usize],
    ) -> Result<usize, String> {
        let (a_name, b_name) = node.require_binary_inputs().map_err(|e| e.to_string())?;
        let out_size: usize = out_shape.iter().product();
        let ta = self.broadcast_operand(a_name, out_shape)?;
        let tb = self.broadcast_operand(b_name, out_shape)?;
        let t = self.push(GraphSearchOp::MulBinary { a: ta, b: tb }, out_size);
        self.sites.push(AlphaSite {
            op: t - 1,
            node: name.to_string(),
            input_node: String::new(),
            kind: SiteKind::MulBinary { n: out_size },
        });
        self.structural += 1;
        Ok(t)
    }

    fn lower_unary(
        &mut self,
        node: &GraphNode,
        kind: UnaryKind,
        out_size: usize,
    ) -> Result<usize, String> {
        let input = Self::unary_input(node)?;
        let t_in = self.tensor(input)?;
        if self.sizes[t_in] != out_size {
            return Err(format!("{} changes the width of '{input}'", kind.token()));
        }
        self.note_box(t_in, input)?;
        self.structural += 1;
        Ok(self.push(GraphSearchOp::Unary { input: t_in, kind }, out_size))
    }

    fn lower_softmax(
        &mut self,
        node: &GraphNode,
        softmax: &SoftmaxLayer,
        out_shape: &[usize],
    ) -> Result<usize, String> {
        let input = Self::unary_input(node)?;
        let in_shape = self.shape_of(input)?;
        if in_shape != out_shape || in_shape.is_empty() {
            return Err(format!("softmax '{input}': shape {in_shape:?} changes"));
        }
        let rank = i32::try_from(in_shape.len()).map_err(|e| e.to_string())?;
        if !(softmax.axis == -1 || softmax.axis == rank - 1 || softmax.axis == rank) {
            return Err(format!(
                "softmax axis {} is not the last axis of {in_shape:?}",
                softmax.axis
            ));
        }
        let len = in_shape[in_shape.len() - 1];
        let n: usize = in_shape.iter().product();
        if len == 0 || !n.is_multiple_of(len) {
            return Err("softmax: empty last axis".into());
        }
        let groups = n / len;
        let b = self.bounds_of(input)?;
        let lo: Vec<f32> = b.lower().iter().copied().collect();
        let hi: Vec<f32> = b.upper().iter().copied().collect();
        // Constant [0, 1] per output unless the lane's LSE relaxation exists.
        let mut lower_a = vec![0.0; n * len];
        let mut lower_b = vec![0.0; n];
        let mut upper_a = vec![0.0; n * len];
        let mut upper_b = vec![1.0; n];
        for g in 0..groups {
            let pl = Array1::from(lo[g * len..(g + 1) * len].to_vec());
            let pu = Array1::from(hi[g * len..(g + 1) * len].to_vec());
            let Some((la, lb, ua, ub)) = softmax.softmax_lse_affine_bounds(&pl, &pu) else {
                continue;
            };
            let well_formed = la.dim() == (len, len)
                && ua.dim() == (len, len)
                && lb.len() == len
                && ub.len() == len
                && la
                    .iter()
                    .chain(ua.iter())
                    .chain(lb.iter())
                    .chain(ub.iter())
                    .all(|v| v.is_finite());
            if !well_formed {
                continue;
            }
            for i in 0..len {
                let o = g * len + i;
                for q in 0..len {
                    lower_a[o * len + q] = f64::from(la[[i, q]]);
                    upper_a[o * len + q] = f64::from(ua[[i, q]]);
                }
                lower_b[o] = f64::from(lb[i]);
                upper_b[o] = f64::from(ub[i]);
            }
        }
        let t_in = self.tensor(input)?;
        self.note_box(t_in, input)?;
        self.structural += 1;
        Ok(self.push(
            GraphSearchOp::Softmax {
                input: t_in,
                groups,
                len,
                lower_a,
                lower_b,
                upper_a,
                upper_b,
            },
            n,
        ))
    }

    fn lower_probe(
        &mut self,
        name: &str,
        node: &GraphNode,
        out_size: usize,
    ) -> Result<usize, String> {
        let mut names: Vec<&str> = Vec::new();
        for input in &node.inputs {
            if !names.contains(&input.as_str()) {
                names.push(input.as_str());
            }
        }
        if names.is_empty() {
            return Err(format!("'{name}' has no dynamic input"));
        }
        let shapes = names
            .iter()
            .map(|n| self.shape_of(n))
            .collect::<Result<Vec<_>, String>>()?;
        let tensors = names
            .iter()
            .map(|n| self.tensor(n))
            .collect::<Result<Vec<_>, String>>()?;
        let sizes: Vec<usize> = shapes.iter().map(|s| s.iter().product()).collect();
        for (&t, &s) in tensors.iter().zip(&sizes) {
            if self.sizes[t] != s {
                return Err(format!("'{name}': input width disagrees with its box"));
            }
        }
        let probe = Probe {
            node,
            name,
            names: &names,
            shapes: &shapes,
            out_size,
        };
        let (terms, bias) = probe_gather(&probe, &sizes)
            .or_else(|| probe_elementwise(&probe, &sizes))
            .or_else(|| probe_basis(&probe, &sizes))
            .ok_or_else(|| {
                format!(
                    "'{name}' ({}) is not an affine map the probes recover",
                    node.layer.layer_type()
                )
            })?;
        self.probed += 1;
        Ok(self.push(
            GraphSearchOp::Sparse {
                inputs: tensors,
                n_out: out_size,
                terms,
                bias,
            },
            out_size,
        ))
    }
}

/// Lower the ancestors of the graph's output, in `exec_order`.
pub(crate) fn lower_graph(
    graph: &GraphNetwork,
    exec_order: &[String],
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
) -> Result<GraphLowering, String> {
    let output = graph.output_node.as_str();
    let mut needed: HashSet<&str> = HashSet::new();
    let mut stack = vec![output];
    while let Some(name) = stack.pop() {
        if name == NETWORK_INPUT || !needed.insert(name) {
            continue;
        }
        let node = graph
            .nodes
            .get(name)
            .ok_or_else(|| format!("the output's ancestry names a missing node '{name}'"))?;
        stack.extend(node.inputs.iter().map(String::as_str));
    }
    let mut builder = Builder {
        input,
        node_bounds,
        ops: Vec::new(),
        sizes: vec![input.len()],
        boxes: HashMap::new(),
        tensor_of: HashMap::new(),
        sites: Vec::new(),
        structural: 0,
        probed: 0,
    };
    for name in exec_order {
        if !needed.contains(name.as_str()) {
            continue;
        }
        let node = graph
            .nodes
            .get(name.as_str())
            .ok_or_else(|| format!("exec order names a missing node '{name}'"))?;
        builder.lower_node(name, node)?;
    }
    let out_tensor = builder.tensor(output)?;
    if out_tensor != builder.ops.len() || out_tensor == 0 {
        return Err("the output is not the last lowered tensor".into());
    }
    if builder.boxes.contains_key(&0) {
        // The input box is the lane's own box, exactly.
        builder
            .boxes
            .insert(0, (flat_f64(input.lower()), flat_f64(input.upper())));
    }
    let net = GraphSearchNet {
        n_in: input.len(),
        ops: builder.ops,
    };
    let sizes = net.tensor_sizes()?;
    if sizes != builder.sizes {
        return Err("the lowered sizes disagree with the builder's".into());
    }
    let mut boxes: Vec<TensorBox> = builder
        .boxes
        .into_iter()
        .map(|(tensor, (l, u))| TensorBox { tensor, l, u })
        .collect();
    boxes.sort_by_key(|b| b.tensor);
    Ok(GraphLowering {
        net,
        boxes,
        sites: builder.sites,
        tensor_of: builder.tensor_of,
        structural: builder.structural,
        probed: builder.probed,
    })
}

/// G1: the copy's TRUE-function f64 forward must agree with the lane's own
/// point forward on EVERY lowered node, at the box centre and at an interior
/// point. Returns what was compared; any disagreement refuses.
pub(crate) fn self_check(
    lowering: &GraphLowering,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
) -> Result<SelfCheck, String> {
    let lo: Vec<f32> = input.lower().iter().copied().collect();
    let hi: Vec<f32> = input.upper().iter().copied().collect();
    let mut compared = 0usize;
    let mut worst = 0.0f64;
    let mut names: Vec<(&String, &usize)> = lowering.tensor_of.iter().collect();
    names.sort();
    for point in 0..2usize {
        let x: Vec<f32> = lo
            .iter()
            .zip(&hi)
            .enumerate()
            .map(|(i, (&l, &u))| {
                let t = if point == 0 {
                    0.5
                } else {
                    0.25 + 0.5 * ((i as f64) * 0.618_033_988_75).fract()
                };
                let v = (f64::from(l) + (f64::from(u) - f64::from(l)) * t) as f32;
                v.clamp(l, u)
            })
            .collect();
        let array = ArrayD::from_shape_vec(input.lower().raw_dim(), x.clone())
            .map_err(|e| e.to_string())?;
        let degenerate = BoundedTensor::new(array.clone(), array).map_err(|e| e.to_string())?;
        let lane = graph
            .collect_node_activations_pointwise(&degenerate, engine)
            .map_err(|e| format!("lane point forward: {e}"))?;
        let x64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let values = lowering.net.forward_f64(&x64)?;
        for &(name, &t) in &names {
            let lane_values = lane
                .get(name)
                .ok_or_else(|| format!("the lane's point forward has no '{name}'"))?;
            let lane_values = flat_f64(lane_values.lower());
            if lane_values.len() != values[t].len() {
                return Err(format!("'{name}': width disagrees with the lane"));
            }
            let scale = lane_values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            if !scale.is_finite() {
                return Err(format!("'{name}': the lane's point value is not finite"));
            }
            let tol = SELF_CHECK_TOL * (1.0 + scale);
            for (a, b) in values[t].iter().zip(&lane_values) {
                let e = (a - b).abs();
                if !matches!(
                    e.partial_cmp(&tol),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                ) {
                    return Err(format!(
                        "'{name}' differs from the lane by {e:.3e} (tolerance {tol:.1e})"
                    ));
                }
                worst = worst.max(e / (1.0 + scale));
            }
            compared += 1;
        }
    }
    Ok(SelfCheck { compared, worst })
}
