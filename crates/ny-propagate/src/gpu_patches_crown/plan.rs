// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#vgg-gpu-crown`: extraction of the admitted sequential chain.
//!
//! The pass admits ONLY a single-input chain of `Conv2d` (dilation 1,
//! groups 1), `Linear`, `ReLU`, non-overlapping unpadded `MaxPool2d`, and the
//! data-order-preserving reshapes (`Flatten` / `Reshape` / `Squeeze` /
//! `Unsqueeze`). That is VGG16's whole op surface. Anything else — a DAG, a
//! residual `Add`, a sigmoid, an overlapping or padded pool, a grouped or
//! dilated conv — refuses the pass before any allocation, and a refusal is
//! byte-identical to the dark route.
//!
//! Every parameter is taken EXACTLY: f32 weights and biases are widened to
//! binary64 bit-exactly (`f32_to_f64_exact`), and the f32 GEMM operands are the
//! original f32 values. There is no folding, fusion or re-quantisation here.

use std::sync::OnceLock;

use ny_core::{f32_to_f64_exact, f64_to_f32_up};
use ny_tensor::BoundedTensor;

use crate::layers::Layer;
use crate::{GraphNetwork, GraphNode, NETWORK_INPUT};

/// Element shape of one chain value. A flat value is `(n, 1, 1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shape3 {
    pub(crate) c: usize,
    pub(crate) h: usize,
    pub(crate) w: usize,
}

impl Shape3 {
    /// A flat value of `n` elements.
    pub(crate) const fn flat(n: usize) -> Self {
        Self { c: n, h: 1, w: 1 }
    }

    /// Number of elements (`c * h * w`; checked at plan time).
    pub(crate) const fn numel(self) -> usize {
        self.c * self.h * self.w
    }

    /// One channel plane (`h * w`).
    pub(crate) const fn plane(self) -> usize {
        self.h * self.w
    }
}

/// `true` for a nonzero binary32 subnormal (bit test: independent of DAZ).
#[inline]
pub(crate) fn is_f32_subnormal(v: f32) -> bool {
    let bits = v.to_bits() & 0x7fff_ffff;
    bits != 0 && bits < 0x0080_0000
}

/// `|w|` as an f32 UPPER bound that is never subnormal: a subnormal magnitude
/// widens to `f32::MIN_POSITIVE`, so a flush-to-zero device cannot silently
/// drop it from the `|A||B|` certificate product.
#[inline]
pub(crate) fn abs_up_weight_f32(v: f32) -> f32 {
    let bits = v.to_bits() & 0x7fff_ffff;
    if bits != 0 && bits < 0x0080_0000 {
        f32::MIN_POSITIVE
    } else {
        f32::from_bits(bits)
    }
}

/// `|x|` rounded UP to f32, never subnormal (see `f64_to_f32_up`). An
/// out-of-range magnitude becomes `+inf`, which every caller refuses.
#[inline]
pub(crate) fn abs_up_f32(x: f64) -> f32 {
    f64_to_f32_up(x.abs())
}

/// One admitted `Conv2d` (dilation 1, groups 1).
pub(crate) struct ConvSpec {
    pub(crate) cout: usize,
    pub(crate) cin: usize,
    pub(crate) kh: usize,
    pub(crate) kw: usize,
    pub(crate) sh: usize,
    pub(crate) sw: usize,
    pub(crate) ph: usize,
    pub(crate) pw: usize,
    /// `(cout, cin, kh, kw)` row-major, the exact binary64 widening of the
    /// stored f32 kernel.
    pub(crate) w: Vec<f64>,
    /// Per-output-channel bias (zeros when the layer has none).
    pub(crate) bias: Vec<f64>,
    /// Some kernel entry is a nonzero f32 subnormal (the signed f32 GEMM route
    /// refuses such a layer; see `product`).
    pub(crate) subnormal_f32: bool,
    /// `max |w|` (for the f32 route's flush-to-zero addend).
    pub(crate) wmax: f64,
    gemm_a64: OnceLock<Vec<f64>>,
    gemm_a32: OnceLock<Vec<f32>>,
    gemm_a32_abs: OnceLock<Vec<f32>>,
}

impl ConvSpec {
    /// Build from explicit parts (validated).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        cout: usize,
        cin: usize,
        kh: usize,
        kw: usize,
        stride: (usize, usize),
        padding: (usize, usize),
        w: Vec<f64>,
        bias: Vec<f64>,
        subnormal_f32: bool,
    ) -> Result<Self, String> {
        if cout == 0 || cin == 0 || kh == 0 || kw == 0 || stride.0 == 0 || stride.1 == 0 {
            return Err("degenerate conv geometry".into());
        }
        let expected = cout
            .checked_mul(cin)
            .and_then(|v| v.checked_mul(kh))
            .and_then(|v| v.checked_mul(kw))
            .ok_or("conv kernel size overflow")?;
        if w.len() != expected || bias.len() != cout {
            return Err(format!(
                "conv parameter arity: kernel {} (expected {expected}), bias {} (expected {cout})",
                w.len(),
                bias.len()
            ));
        }
        if w.iter().chain(bias.iter()).any(|v| !v.is_finite()) {
            return Err("non-finite conv parameter".into());
        }
        let wmax = w.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        Ok(Self {
            cout,
            cin,
            kh,
            kw,
            sh: stride.0,
            sw: stride.1,
            ph: padding.0,
            pw: padding.1,
            w,
            bias,
            subnormal_f32,
            wmax,
            gemm_a64: OnceLock::new(),
            gemm_a32: OnceLock::new(),
            gemm_a32_abs: OnceLock::new(),
        })
    }

    /// Full contraction length of one backward output entry:
    /// `kh * kw * cout` (absent taps are EXACT zeros, see `product`).
    pub(crate) const fn contraction(&self) -> usize {
        self.kh * self.kw * self.cout
    }

    /// Kernel entry `w[c][ci][ky][kx]`.
    #[inline]
    pub(crate) fn weight(&self, c: usize, ci: usize, ky: usize, kx: usize) -> f64 {
        self.w[((c * self.cin + ci) * self.kh + ky) * self.kw + kx]
    }

    /// The GEMM left operand `A (cin x kh*kw*cout)` with
    /// `A[ci][(ky*kw + kx)*cout + c] = w[c][ci][ky][kx]`.
    fn arranged<T>(&self, map: impl Fn(f64) -> T) -> Vec<T> {
        let mut a = Vec::with_capacity(self.cin * self.contraction());
        for ci in 0..self.cin {
            for ky in 0..self.kh {
                for kx in 0..self.kw {
                    for c in 0..self.cout {
                        a.push(map(self.weight(c, ci, ky, kx)));
                    }
                }
            }
        }
        a
    }

    /// Signed binary64 GEMM operand (exact).
    pub(crate) fn gemm_a64(&self) -> &[f64] {
        self.gemm_a64.get_or_init(|| self.arranged(|v| v))
    }

    /// Signed f32 GEMM operand: every entry is an exact widening of an f32,
    /// so the narrowing back is exact.
    pub(crate) fn gemm_a32(&self) -> &[f32] {
        self.gemm_a32.get_or_init(|| self.arranged(|v| v as f32))
    }

    /// `|W|` f32 GEMM operand, never subnormal (upper bound of `|w|`).
    pub(crate) fn gemm_a32_abs(&self) -> &[f32] {
        self.gemm_a32_abs
            .get_or_init(|| self.arranged(|v| abs_up_weight_f32(v as f32)))
    }
}

/// One admitted `Linear` (borrowing the layer's own f32 weight storage: the
/// fc6 matrix of VGG16 alone is 102.8 M entries, and an f64 copy would be
/// 822 MB).
pub(crate) struct LinearSpec<'g> {
    pub(crate) nout: usize,
    pub(crate) nin: usize,
    /// `(nout, nin)` row-major f32.
    pub(crate) w: &'g [f32],
    pub(crate) bias: Vec<f64>,
    pub(crate) subnormal_f32: bool,
    pub(crate) wmax: f64,
}

impl<'g> LinearSpec<'g> {
    /// Build from explicit parts (validated).
    pub(crate) fn new(
        nout: usize,
        nin: usize,
        w: &'g [f32],
        bias: Vec<f64>,
    ) -> Result<Self, String> {
        if nout == 0 || nin == 0 {
            return Err("degenerate linear geometry".into());
        }
        if nout.checked_mul(nin) != Some(w.len()) || bias.len() != nout {
            return Err("linear parameter arity".into());
        }
        if w.iter().any(|v| !v.is_finite()) || bias.iter().any(|v| !v.is_finite()) {
            return Err("non-finite linear parameter".into());
        }
        let subnormal_f32 = w.iter().any(|&v| is_f32_subnormal(v));
        let wmax = w
            .iter()
            .fold(0.0_f64, |m, &v| m.max(f32_to_f64_exact(v).abs()));
        Ok(Self {
            nout,
            nin,
            w,
            bias,
            subnormal_f32,
            wmax,
        })
    }

    /// `W[o][j]` widened exactly.
    #[inline]
    pub(crate) fn weight(&self, o: usize, j: usize) -> f64 {
        f32_to_f64_exact(self.w[o * self.nin + j])
    }
}

/// One admitted `MaxPool2d`: unpadded, windows non-overlapping.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PoolSpec {
    pub(crate) kh: usize,
    pub(crate) kw: usize,
    pub(crate) sh: usize,
    pub(crate) sw: usize,
}

/// The operator of one chain link.
pub(crate) enum OpKind<'g> {
    /// Boxed: a `ConvSpec` is 224 bytes against `LinearSpec`'s 72
    /// (clippy::large_enum_variant, workspace threshold 64).
    Conv(Box<ConvSpec>),
    Linear(LinearSpec<'g>),
    Relu,
    MaxPool(PoolSpec),
    /// Flatten / Reshape / Squeeze / Unsqueeze: identity on the row-major data.
    Reshape,
}

/// One chain link: value `i` (its `input`) -> value `i + 1` (its `output`).
pub(crate) struct ChainOp<'g> {
    pub(crate) name: String,
    pub(crate) kind: OpKind<'g>,
    pub(crate) input: Shape3,
    pub(crate) output: Shape3,
}

impl ChainOp<'_> {
    /// Conv or Linear.
    pub(crate) fn is_affine(&self) -> bool {
        matches!(self.kind, OpKind::Conv(_) | OpKind::Linear(_))
    }
}

/// The admitted chain. Value `0` is the network input; value `i + 1` is the
/// output of `ops[i]`; the last value is the graph output.
pub(crate) struct ChainPlan<'g> {
    pub(crate) input: Shape3,
    pub(crate) ops: Vec<ChainOp<'g>>,
}

impl<'g> ChainPlan<'g> {
    /// Shape of value `v`.
    pub(crate) fn value_shape(&self, v: usize) -> Shape3 {
        if v == 0 {
            self.input
        } else {
            self.ops[v - 1].output
        }
    }

    /// Index of the output value.
    pub(crate) fn output_value(&self) -> usize {
        self.ops.len()
    }

    /// Does value `v` feed a ReLU (so its bounds decide relaxations)?
    pub(crate) fn feeds_relu(&self, v: usize) -> bool {
        self.ops
            .get(v)
            .is_some_and(|op| matches!(op.kind, OpKind::Relu))
    }

    /// Extract the chain, or say why it is refused.
    pub(crate) fn build(graph: &'g GraphNetwork, input: Shape3) -> Result<Self, String> {
        if input.c == 0 || input.h == 0 || input.w == 0 {
            return Err("degenerate input shape".into());
        }
        let order = graph
            .exec_order()
            .map_err(|e| format!("no execution order: {e}"))?;
        if order.is_empty() {
            return Err("empty graph".into());
        }
        let mut ops = Vec::with_capacity(order.len());
        let mut prev: &str = NETWORK_INPUT;
        let mut shape = input;
        for name in order {
            let node: &'g GraphNode = graph
                .node(name)
                .ok_or_else(|| format!("missing node '{name}'"))?;
            let inputs = node.inputs();
            if inputs.len() != 1 || inputs[0] != prev {
                return Err(format!(
                    "node '{name}' is not a single-input link of a sequential chain"
                ));
            }
            let (kind, output) = lower_node(node.layer(), shape)
                .map_err(|why| format!("node '{name}' ({}): {why}", node.layer().layer_type()))?;
            ops.push(ChainOp {
                name: name.clone(),
                kind,
                input: shape,
                output,
            });
            shape = output;
            prev = name.as_str();
        }
        let out = graph.output_name();
        if !out.is_empty() && out != prev {
            return Err(format!(
                "graph output '{out}' is not the last chain link '{prev}'"
            ));
        }
        Ok(Self { input, ops })
    }
}

fn conv_out_dim(n: usize, k: usize, s: usize, p: usize) -> Result<usize, String> {
    let padded = p
        .checked_mul(2)
        .and_then(|two_p| n.checked_add(two_p))
        .ok_or("conv size overflow")?;
    if padded < k {
        return Err("conv kernel larger than the padded input".into());
    }
    Ok((padded - k) / s + 1)
}

fn checked_numel(s: Shape3) -> Result<Shape3, String> {
    s.c.checked_mul(s.h)
        .and_then(|v| v.checked_mul(s.w))
        .map(|_| s)
        .ok_or_else(|| "shape size overflow".to_string())
}

fn lower_node<'g>(layer: &'g Layer, shape: Shape3) -> Result<(OpKind<'g>, Shape3), String> {
    match layer {
        Layer::Conv2d(conv) => {
            let ks = conv.kernel.shape();
            if ks.len() != 4 {
                return Err("kernel is not 4-D".into());
            }
            let (cout, cin, kh, kw) = (ks[0], ks[1], ks[2], ks[3]);
            if conv.groups != 1 {
                return Err("grouped conv".into());
            }
            if conv.dilation != (1, 1) {
                return Err("dilated conv".into());
            }
            if cin != shape.c {
                return Err(format!(
                    "input has {} channels, kernel expects {cin}",
                    shape.c
                ));
            }
            let (sh, sw) = conv.stride;
            let (ph, pw) = conv.padding;
            if sh == 0 || sw == 0 {
                return Err("zero stride".into());
            }
            let oh = conv_out_dim(shape.h, kh, sh, ph)?;
            let ow = conv_out_dim(shape.w, kw, sw, pw)?;
            let w: Vec<f64> = conv.kernel.iter().map(|&v| f32_to_f64_exact(v)).collect();
            let subnormal = conv.kernel.iter().any(|&v| is_f32_subnormal(v));
            let bias: Vec<f64> = match conv.bias.as_ref() {
                Some(b) => b.iter().map(|&v| f32_to_f64_exact(v)).collect(),
                None => vec![0.0; cout],
            };
            let spec = ConvSpec::new(cout, cin, kh, kw, (sh, sw), (ph, pw), w, bias, subnormal)?;
            let out = checked_numel(Shape3 {
                c: cout,
                h: oh,
                w: ow,
            })?;
            Ok((OpKind::Conv(Box::new(spec)), out))
        }
        Layer::Linear(lin) => {
            let (nout, nin) = (lin.out_features(), lin.in_features());
            if nin != shape.numel() {
                return Err(format!(
                    "input has {} elements, weight expects {nin}",
                    shape.numel()
                ));
            }
            let w: &'g [f32] = lin
                .weight()
                .as_slice()
                .ok_or("weight is not in standard row-major layout")?;
            let bias: Vec<f64> = match lin.bias() {
                Some(b) => b.iter().map(|&v| f32_to_f64_exact(v)).collect(),
                None => vec![0.0; nout],
            };
            let spec = LinearSpec::new(nout, nin, w, bias)?;
            Ok((OpKind::Linear(spec), Shape3::flat(nout)))
        }
        Layer::ReLU(_) => Ok((OpKind::Relu, shape)),
        Layer::MaxPool2d(pool) => {
            let (kh, kw) = pool.kernel_size;
            let (sh, sw) = pool.stride;
            if pool.padding != (0, 0) {
                return Err("padded max-pool".into());
            }
            if kh == 0 || kw == 0 || sh < kh || sw < kw {
                return Err("overlapping or degenerate max-pool windows".into());
            }
            if shape.h < kh || shape.w < kw {
                return Err("max-pool window larger than the input".into());
            }
            let out = Shape3 {
                c: shape.c,
                h: (shape.h - kh) / sh + 1,
                w: (shape.w - kw) / sw + 1,
            };
            Ok((OpKind::MaxPool(PoolSpec { kh, kw, sh, sw }), out))
        }
        Layer::Flatten(_) | Layer::Reshape(_) | Layer::Squeeze(_) | Layer::Unsqueeze(_) => {
            Ok((OpKind::Reshape, Shape3::flat(shape.numel())))
        }
        _ => Err("unsupported operator".into()),
    }
}

/// Recover the `(C, H, W)` element shape of the network input (the
/// `dd_zonotope::input_chw` recipe): leading unit axes are stripped; a flat
/// box is reconstructed from the FIRST `Conv2d`'s declared `input_shape` and
/// kernel input channels; a flat box feeding a `Linear` stays flat. The
/// product is always re-checked against `input.len()`.
pub(crate) fn input_shape(graph: &GraphNetwork, input: &BoundedTensor) -> Option<Shape3> {
    let mut dims: Vec<usize> = input.shape().to_vec();
    while dims.len() > 3 && dims[0] == 1 {
        dims.remove(0);
    }
    let n = input.len();
    if dims.len() == 3 {
        let s = Shape3 {
            c: dims[0],
            h: dims[1],
            w: dims[2],
        };
        if s.c.checked_mul(s.h).and_then(|v| v.checked_mul(s.w)) == Some(n) {
            return Some(s);
        }
    }
    let order = graph.exec_order().ok()?;
    let first = graph.node(order.first()?)?;
    match first.layer() {
        Layer::Conv2d(conv) => {
            let (h, w) = conv.input_shape?;
            let c = *conv.kernel.shape().get(1)?;
            if c.checked_mul(h)?.checked_mul(w)? == n {
                Some(Shape3 { c, h, w })
            } else {
                None
            }
        }
        Layer::Linear(lin) if lin.in_features() == n => Some(Shape3::flat(n)),
        _ => None,
    }
}
