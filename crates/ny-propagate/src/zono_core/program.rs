// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! A compiled, single-input sequential zonotope program over a
//! [`GraphNetwork`], and its certified forward pass.
//!
//! OP SET (the Z1 replica's, `z1lib.py:67-132`): `Linear` (ONNX Gemm),
//! `Reshape`/`Flatten`, `ConvTranspose2d`, `Conv2d`, `BatchNorm` and `ReLU`.
//! Every other layer, any node with more than one input, and any graph that
//! is not one straight chain from the network input to the output node is
//! REFUSED at compile time — the cGAN nonlinear-activations (Tanh/Sigmoid)
//! and upsample models are the real-model refusals; the small-transformer
//! model (row 19) never gets here, because ny refuses to build its graph at
//! all (`child_zono_refuses_unsupported_ops`). Grouped/dilated convolutions
//! and `output_padding` are refused too: they are outside what Z1 measured.
//!
//! Shapes come from ny's own IBP forward on a sample box
//! ([`ZonoProgram::compile_with_ibp_shapes`]), so every published entry has
//! exactly the shape the input-split maps use and a shape mismatch can only
//! make the downstream merge skip an entry, never misapply one.
//!
//! #m8-cgan-leaf-batch: [`ZonoProgram::forward_batch`] runs a tranche of
//! children concurrently with [`RowPar::Rayon`], i.e. every `K`-row pass of
//! every child dispatched as the reference's own row blocks (kernels) or as
//! independent rows / column blocks (element-wise passes). Each child's
//! outcome is bitwise its [`ZonoProgram::forward`] outcome with
//! [`RowPar::Seq`] (`zono_forward_batch_is_bit_identical_to_per_child_forward`):
//! no arithmetic changes, so the zono_core contract above carries over
//! verbatim and there is no key.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;

use crate::layers::{
    BatchNormChannelAxisHint, BatchNormLayer, Conv2dLayer, ConvTranspose2dLayer, Layer, LinearLayer,
};
use crate::network::NETWORK_INPUT;
use crate::GraphNetwork;

use super::affine::{
    apply_affine_par, apply_channel_affine_par, AffineOp, BatchNormCoefficients, ChannelAffine,
};
use super::concretize::{concretize, count_unstable, intersect_keep_looser, to_f32_outward};
use super::ids::{SymbolAllocator, SymbolId};
use super::kernels::{check_deadline, ConvGeom, LinearKernel, RowPar, CHUNK_ELEMS};
use super::relu::apply_relu_par;
use super::rounding::widen_f32_interval;
use super::state::{InputSymbol, ZonoState};
use super::ZonoRefusal;

/// What one program step does.
#[derive(Debug, Clone)]
pub enum StepOp {
    /// `y = A x + b`. Boxed, like `ChannelAffine`: inline they made the enum
    /// 176 bytes with unit variants beside them (clippy::large_enum_variant,
    /// workspace threshold 64).
    Affine(Box<AffineOp>),
    /// Per-element certified affine (BatchNorm).
    ChannelAffine(Box<ChannelAffine>),
    /// Pure relabelling of the flat values (Reshape / Flatten).
    Reshape,
    /// DeepZ ReLU.
    Relu,
}

/// One step: the node it computes, the node it reads, and both shapes.
#[derive(Debug, Clone)]
pub struct ZonoStep {
    /// Graph node computed by this step.
    pub node: String,
    /// Node (or [`NETWORK_INPUT`]) this step reads.
    pub input: String,
    /// Shape of the input as ny propagates it.
    pub input_shape: Vec<usize>,
    /// Shape of the output as ny propagates it.
    pub out_shape: Vec<usize>,
    /// The transformer.
    pub op: StepOp,
}

/// A compiled sequential program.
#[derive(Debug, Clone)]
pub struct ZonoProgram {
    input_len: usize,
    input_shape: Vec<usize>,
    steps: Vec<ZonoStep>,
    /// Coefficient representation of every compiled BatchNorm step, in
    /// order (#zono-exact-bn): the lane's own observable of which charge it
    /// applies.
    batch_norm: Vec<(String, BatchNormCoefficients)>,
}

/// Options for one forward pass.
#[derive(Debug, Clone, Copy)]
pub struct ZonoForwardOptions {
    /// Generator budget; rows beyond it are folded into the remainder after
    /// each ReLU (input symbols are never folded). The cGAN default is 16384,
    /// which no measured leaf reaches (Z1: at most 4,153).
    pub max_generators: usize,
    /// Keep the per-ReLU boxes and per-node magnitude bounds needed by
    /// [`super::spec_crown`] (tests and diagnostics; costs memory).
    pub record_trace: bool,
    /// Row schedule of every `K`-row pass (#m8-cgan-leaf-batch).
    /// [`RowPar::Seq`] is the reference; [`RowPar::Rayon`] is bitwise
    /// identical and is what [`ZonoProgram::forward_batch`] uses.
    pub row_par: RowPar,
}

/// Per-ReLU accounting of one forward pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReluSummary {
    /// Pre-activation node (the ReLU's input).
    pub node: String,
    /// Unstable neurons under the zonotope's own concretisation.
    pub zono_unstable: usize,
    /// Unstable neurons under the inherited maps alone (`None`: no inherited
    /// entry for this node).
    pub inherited_unstable: Option<usize>,
    /// Unstable neurons after the intersection (what DeepZ relaxed).
    pub unstable: usize,
}

/// Data [`super::spec_crown`] needs, recorded only on request.
#[derive(Debug, Clone)]
pub struct ZonoTrace {
    /// Per-ReLU certified pre-activation boxes (after intersection), in order.
    pub pre_boxes: Vec<(Vec<f64>, Vec<f64>)>,
    /// Per-node magnitude bounds: index 0 is the input, `i + 1` is step `i`.
    pub magnitudes: Vec<Vec<f64>>,
    /// The one-ulp-widened binary64 input box the forward pass enclosed.
    pub input_lo: Vec<f64>,
    /// Upper end of `input_lo`'s box.
    pub input_hi: Vec<f64>,
}

/// Cost and charge statistics of one forward pass.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ZonoForwardStats {
    /// Generator rows at the output.
    pub gens_final: usize,
    /// Most generator rows held at once.
    pub gens_peak: usize,
    /// Rows folded into the remainder by the budget.
    pub folded: usize,
    /// Largest remainder radius over every node (the certified charge).
    pub max_remainder: f64,
    /// Multiply-accumulates spent in linear kernels.
    pub macs: f64,
    /// Wall time of the pass.
    pub elapsed: Duration,
    /// Generator-kernel row blocks run as rayon tasks under
    /// [`RowPar::Rayon`] (0 under [`RowPar::Seq`]). Telemetry only.
    pub row_tasks: usize,
}

/// One child of [`ZonoProgram::forward_batch`]: its box, the inherited maps
/// its ReLUs intersect with, and its own deadline.
#[derive(Debug, Clone)]
pub struct ZonoBatchChild<'a> {
    /// The child's `f32` input box (compiled input shape).
    pub input: &'a BoundedTensor,
    /// Certified maps intersected into every ReLU's pre-activation box.
    pub inherited: Vec<&'a HashMap<String, BoundedTensor>>,
    /// This child's deadline (expiry refuses THIS child only).
    pub deadline: Option<Instant>,
}

/// Result of one forward pass.
#[derive(Debug, Clone)]
pub struct ZonoForwardOutcome {
    /// Certified `f32` pre-activation boxes (zonotope intersected with the
    /// inherited maps), keyed by the ReLU INPUT node, with ny's shapes.
    pub published: HashMap<String, BoundedTensor>,
    /// Per-ReLU accounting, in order.
    pub relus: Vec<ReluSummary>,
    /// The input symbols (for pointwise form checks).
    pub input_symbols: Vec<InputSymbol>,
    /// Certified output box.
    pub output_lo: Vec<f64>,
    /// Upper end of the output box.
    pub output_hi: Vec<f64>,
    /// Trace for [`super::spec_crown`], when requested.
    pub trace: Option<ZonoTrace>,
    /// Statistics.
    pub stats: ZonoForwardStats,
}

/// A read-only view of the state after one step, for observers (tests).
pub struct ZonoNodeView<'a> {
    /// Step index (`None` for the input).
    pub step: Option<usize>,
    /// Node name ([`NETWORK_INPUT`] for the input).
    pub node: &'a str,
    /// Node shape as ny propagates it.
    pub shape: &'a [usize],
    /// The state.
    pub state: &'a ZonoState,
    /// `state.abs_generator_sum()`.
    pub abs_g: &'a [f64],
    /// The input symbols of this pass.
    pub input_symbols: &'a [InputSymbol],
}

fn numel(shape: &[usize]) -> Result<usize, ZonoRefusal> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| ZonoRefusal::Shape(format!("shape {shape:?} overflows usize")))
}

/// `(C, H, W)` of an image shape: the last three dims, every leading dim 1.
fn image_dims(shape: &[usize]) -> Option<(usize, usize, usize)> {
    if shape.len() < 3 {
        return None;
    }
    let (lead, tail) = shape.split_at(shape.len() - 3);
    if lead.iter().any(|&d| d != 1) {
        return None;
    }
    Some((tail[0], tail[1], tail[2]))
}

fn to_f64(values: impl Iterator<Item = f32>) -> Vec<f64> {
    values.map(f64::from).collect()
}

fn abs_of(values: &[f64]) -> Vec<f64> {
    values.iter().map(|v| v.abs()).collect()
}

fn per_channel_full(
    bias: Option<&ndarray::Array1<f32>>,
    channels: usize,
    per_channel: usize,
    node: &str,
) -> Result<Vec<f64>, ZonoRefusal> {
    match bias {
        None => Ok(vec![0.0; channels * per_channel]),
        Some(b) if b.len() == channels => Ok(b
            .iter()
            .flat_map(|&v| std::iter::repeat_n(f64::from(v), per_channel))
            .collect()),
        Some(b) => Err(ZonoRefusal::Shape(format!(
            "node `{node}`: bias of {} values for {channels} channels",
            b.len()
        ))),
    }
}

fn dense_op(
    node: &str,
    layer: &LinearLayer,
    in_shape: &[usize],
    n_in: usize,
    n_out: usize,
) -> Result<AffineOp, ZonoRefusal> {
    if in_shape.len() > 1 && in_shape[..in_shape.len() - 1].iter().any(|&d| d != 1) {
        return Err(ZonoRefusal::Shape(format!(
            "node `{node}`: Linear on a non-vector input {in_shape:?}"
        )));
    }
    let weight = layer.weight();
    let (rows, cols) = weight.dim();
    if cols != n_in || rows != n_out {
        return Err(ZonoRefusal::Shape(format!(
            "node `{node}`: Linear weight {rows}x{cols} for {n_in} -> {n_out}"
        )));
    }
    let w = to_f64(weight.as_standard_layout().iter().copied());
    let w_abs = abs_of(&w);
    let bias = match layer.bias() {
        None => vec![0.0; rows],
        Some(b) if b.len() == rows => to_f64(b.iter().copied()),
        Some(b) => {
            return Err(ZonoRefusal::Shape(format!(
                "node `{node}`: Linear bias of {} values for {rows} outputs",
                b.len()
            )))
        }
    };
    AffineOp::new(
        LinearKernel::Dense {
            n_in: cols,
            n_out: rows,
            w,
            w_abs,
        },
        bias,
    )
}

pub(super) fn conv_op(
    node: &str,
    layer: &Conv2dLayer,
    in_shape: &[usize],
    out_shape: &[usize],
) -> Result<AffineOp, ZonoRefusal> {
    let kernel = layer.kernel();
    if kernel.ndim() != 4 {
        return Err(ZonoRefusal::Shape(format!(
            "node `{node}`: Conv2d kernel rank {}",
            kernel.ndim()
        )));
    }
    if layer.groups != 1 || layer.dilation != (1, 1) {
        return Err(ZonoRefusal::UnsupportedGraph(format!(
            "node `{node}`: grouped or dilated Conv2d (groups={}, dilation={:?})",
            layer.groups, layer.dilation
        )));
    }
    let ks = kernel.shape();
    let (co, ci, kh, kw) = (ks[0], ks[1], ks[2], ks[3]);
    let (c, h, w) = image_dims(in_shape).ok_or_else(|| {
        ZonoRefusal::Shape(format!("node `{node}`: Conv2d input shape {in_shape:?}"))
    })?;
    if c != ci {
        return Err(ZonoRefusal::Shape(format!(
            "node `{node}`: Conv2d kernel expects {ci} channels, input has {c}"
        )));
    }
    let geom = ConvGeom::new(
        c,
        h,
        w,
        kh,
        kw,
        layer.stride.0,
        layer.stride.1,
        layer.padding.0,
        layer.padding.1,
    )?;
    if image_dims(out_shape) != Some((co, geom.out_h, geom.out_w)) {
        return Err(ZonoRefusal::Shape(format!(
            "node `{node}`: Conv2d output {out_shape:?} vs computed {co}x{}x{}",
            geom.out_h, geom.out_w
        )));
    }
    let k = to_f64(kernel.as_standard_layout().iter().copied());
    let k_abs = abs_of(&k);
    let bias = per_channel_full(layer.bias(), co, geom.grid_len(), node)?;
    AffineOp::new(
        LinearKernel::Conv2d {
            geom,
            out_channels: co,
            k,
            k_abs,
        },
        bias,
    )
}

fn conv_transpose_op(
    node: &str,
    layer: &ConvTranspose2dLayer,
    in_shape: &[usize],
    out_shape: &[usize],
) -> Result<AffineOp, ZonoRefusal> {
    let kernel = layer.kernel();
    if kernel.ndim() != 4 {
        return Err(ZonoRefusal::Shape(format!(
            "node `{node}`: ConvTranspose2d kernel rank {}",
            kernel.ndim()
        )));
    }
    if layer.dilation != (1, 1) || layer.output_padding != (0, 0) {
        return Err(ZonoRefusal::UnsupportedGraph(format!(
            "node `{node}`: ConvTranspose2d with dilation {:?} / output_padding {:?}",
            layer.dilation, layer.output_padding
        )));
    }
    let ks = kernel.shape();
    let (ci, co, kh, kw) = (ks[0], ks[1], ks[2], ks[3]);
    let (c, h, w) = image_dims(in_shape).ok_or_else(|| {
        ZonoRefusal::Shape(format!(
            "node `{node}`: ConvTranspose2d input shape {in_shape:?}"
        ))
    })?;
    if c != ci || h == 0 || w == 0 {
        return Err(ZonoRefusal::Shape(format!(
            "node `{node}`: ConvTranspose2d kernel expects {ci} channels, input {in_shape:?}"
        )));
    }
    let (sh, sw) = layer.stride;
    let (ph, pw) = layer.padding;
    let full_h = (h - 1)
        .checked_mul(sh)
        .and_then(|v| v.checked_add(kh))
        .ok_or_else(|| ZonoRefusal::Shape(format!("node `{node}`: output height overflow")))?;
    let full_w = (w - 1)
        .checked_mul(sw)
        .and_then(|v| v.checked_add(kw))
        .ok_or_else(|| ZonoRefusal::Shape(format!("node `{node}`: output width overflow")))?;
    if full_h <= 2 * ph || full_w <= 2 * pw {
        return Err(ZonoRefusal::Shape(format!(
            "node `{node}`: ConvTranspose2d padding {ph}x{pw} crops the whole output"
        )));
    }
    let (oh, ow) = (full_h - 2 * ph, full_w - 2 * pw);
    if image_dims(out_shape) != Some((co, oh, ow)) {
        return Err(ZonoRefusal::Shape(format!(
            "node `{node}`: ConvTranspose2d output {out_shape:?} vs computed {co}x{oh}x{ow}"
        )));
    }
    let geom = ConvGeom::new(co, oh, ow, kh, kw, sh, sw, ph, pw)?;
    if geom.out_h != h || geom.out_w != w {
        return Err(ZonoRefusal::Internal(format!(
            "node `{node}`: ConvTranspose2d grid {}x{} does not reproduce the input {h}x{w}",
            geom.out_h, geom.out_w
        )));
    }
    let k = to_f64(kernel.as_standard_layout().iter().copied());
    let k_abs = abs_of(&k);
    let bias = per_channel_full(layer.bias(), co, oh * ow, node)?;
    AffineOp::new(
        LinearKernel::ConvTranspose2d {
            geom,
            in_channels: ci,
            k,
            k_abs,
        },
        bias,
    )
}

/// The channel axis BatchNorm propagation uses for `shape` — the same
/// resolution as `batch_norm::math::detect_input_layout` (hint first, then the
/// rank rules), restated here because that helper is private to the layer.
fn bn_channel_axis(
    shape: &[usize],
    channels: usize,
    hint: Option<BatchNormChannelAxisHint>,
) -> Option<usize> {
    if shape.is_empty() {
        return None;
    }
    let axis = match hint {
        Some(BatchNormChannelAxisHint::Fixed(axis)) => axis,
        Some(BatchNormChannelAxisHint::OnnxNchw { authored_rank }) => {
            if shape.len() == authored_rank {
                1
            } else if shape.len() + 1 == authored_rank {
                0
            } else {
                return None;
            }
        }
        None => match shape.len() {
            0 => return None,
            1 => 0,
            2 => {
                if shape[1] == channels {
                    1
                } else if shape[0] == channels {
                    0
                } else {
                    1
                }
            }
            3 => usize::from(shape[0] != channels),
            _ => 1,
        },
    };
    (axis < shape.len()).then_some(axis)
}

fn batch_norm_op(
    node: &str,
    layer: &BatchNormLayer,
    in_shape: &[usize],
    out_shape: &[usize],
) -> Result<(ChannelAffine, BatchNormCoefficients), ZonoRefusal> {
    if in_shape != out_shape {
        return Err(ZonoRefusal::Shape(format!(
            "node `{node}`: BatchNorm {in_shape:?} -> {out_shape:?}"
        )));
    }
    layer
        .validate_affine_parameters()
        .map_err(|e| ZonoRefusal::Shape(format!("node `{node}`: BatchNorm parameters: {e}")))?;
    let channels = layer.num_channels;
    let axis = bn_channel_axis(in_shape, channels, layer.channel_axis_hint).ok_or_else(|| {
        ZonoRefusal::Shape(format!(
            "node `{node}`: BatchNorm channel axis unresolved for {in_shape:?}"
        ))
    })?;
    if in_shape[axis] != channels || numel(&in_shape[..axis])? != 1 {
        return Err(ZonoRefusal::Shape(format!(
            "node `{node}`: BatchNorm over {in_shape:?} with {channels} channels at axis {axis}"
        )));
    }
    let per_channel = numel(&in_shape[axis + 1..])?.max(1);
    let n = numel(in_shape)?;
    let (per_channel_op, source) = batch_norm_channel_coefficients(layer);
    if [
        &per_channel_op.scale,
        &per_channel_op.shift,
        &per_channel_op.scale_err,
        &per_channel_op.shift_err,
    ]
    .iter()
    .any(|v| v.len() != channels)
    {
        return Err(ZonoRefusal::Shape(format!(
            "node `{node}`: BatchNorm vectors do not have {channels} entries"
        )));
    }
    let expand =
        |v: &[f64]| -> Vec<f64> { (0..n).map(|j| v[(j / per_channel) % channels]).collect() };
    Ok((
        ChannelAffine {
            scale: expand(&per_channel_op.scale),
            shift: expand(&per_channel_op.shift),
            scale_err: expand(&per_channel_op.scale_err),
            shift_err: expand(&per_channel_op.shift_err),
        },
        source,
    ))
}

/// The PER-CHANNEL coefficients a zonotope BatchNorm step charges, and which
/// representation they came from (#zono-exact-bn).
///
/// [`BatchNormCoefficients::ExactF64`] whenever the layer carries an intact
/// binary64 enclosure of its exact-real coefficients
/// (`BatchNormLayer::exact_affine_channels`: every layer built by
/// `BatchNormLayer::new`, i.e. every ONNX-converted BatchNorm): binary64
/// centres with radii of a few binary64 ulps, so the model charge
/// `scale_err * m + shift_err` is ~`1e-15` relative instead of the binary32
/// affine's ~`1e-7`. Otherwise [`BatchNormCoefficients::Binary32`]: the stored
/// binary32 affine and its certified `scale_err` / `bias_err`, exactly as
/// before this path existed. Either way `|s_real - scale| <= scale_err` and
/// `|t_real - shift| <= shift_err` hold per channel, which is all
/// [`apply_channel_affine`] and `spec_crown` assume.
fn batch_norm_channel_coefficients(
    layer: &BatchNormLayer,
) -> (ChannelAffine, BatchNormCoefficients) {
    match layer.exact_affine_channels() {
        Some(exact) => (
            ChannelAffine {
                scale: exact.scale,
                shift: exact.bias,
                scale_err: exact.scale_err,
                shift_err: exact.bias_err,
            },
            BatchNormCoefficients::ExactF64,
        ),
        None => (
            ChannelAffine {
                scale: to_f64(layer.scale.iter().copied()),
                shift: to_f64(layer.bias.iter().copied()),
                scale_err: to_f64(layer.scale_err.iter().copied()),
                shift_err: to_f64(layer.bias_err.iter().copied()),
            },
            BatchNormCoefficients::Binary32,
        ),
    }
}

/// Structural pre-check: refuse any node the program cannot represent BEFORE
/// spending a shape pass on it.
fn precheck_graph(graph: &GraphNetwork) -> Result<(), ZonoRefusal> {
    let order = graph
        .exec_order()
        .map_err(|e| ZonoRefusal::UnsupportedGraph(format!("execution order: {e}")))?;
    if order.is_empty() {
        return Err(ZonoRefusal::UnsupportedGraph("empty graph".to_string()));
    }
    let mut prev = NETWORK_INPUT;
    for name in order {
        let node = graph
            .node(name)
            .ok_or_else(|| ZonoRefusal::UnsupportedGraph(format!("node `{name}` missing")))?;
        match node.layer() {
            Layer::Linear(_)
            | Layer::Reshape(_)
            | Layer::Flatten(_)
            | Layer::Conv2d(_)
            | Layer::ConvTranspose2d(_)
            | Layer::BatchNorm(_)
            | Layer::ReLU(_) => {}
            other => {
                return Err(ZonoRefusal::UnsupportedLayer {
                    node: name.clone(),
                    layer: other.layer_type(),
                })
            }
        }
        if node.inputs().len() != 1 || node.inputs()[0] != prev {
            return Err(ZonoRefusal::UnsupportedGraph(format!(
                "node `{name}` reads {:?}, not the single previous node `{prev}`",
                node.inputs()
            )));
        }
        prev = name.as_str();
    }
    let output = graph.output_name();
    if !output.is_empty() && output != prev {
        return Err(ZonoRefusal::UnsupportedGraph(format!(
            "output node `{output}` is not the end of the chain `{prev}`"
        )));
    }
    Ok(())
}

impl ZonoProgram {
    /// Compile with shapes from ny's own IBP forward on `sample` (any box of
    /// the right shape; bounds are discarded, only shapes are kept). The
    /// structural pre-check runs first, so an unsupported graph refuses
    /// without a shape pass.
    pub fn compile_with_ibp_shapes(
        graph: &GraphNetwork,
        sample: &BoundedTensor,
    ) -> Result<Self, ZonoRefusal> {
        Self::compile_with_ibp_shapes_and_deadline(graph, sample, None)
    }

    /// [`Self::compile_with_ibp_shapes`] with the shape pass bounded by
    /// `deadline` (expiry refuses with [`ZonoRefusal::Deadline`]).
    pub fn compile_with_ibp_shapes_and_deadline(
        graph: &GraphNetwork,
        sample: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<Self, ZonoRefusal> {
        precheck_graph(graph)?;
        let shapes: HashMap<String, Vec<usize>> = graph
            .collect_node_bounds_with_engine_and_deadline(sample, None, deadline)
            .map_err(|e| {
                if e.is_deadline_exceeded() {
                    ZonoRefusal::Deadline
                } else {
                    ZonoRefusal::UnsupportedGraph(format!("shape pass: {e}"))
                }
            })?
            .into_iter()
            .map(|(name, bounds)| (name, bounds.shape().to_vec()))
            .collect();
        Self::compile(graph, &shapes, sample.shape())
    }

    /// Compile from explicit per-node shapes and the input shape.
    pub fn compile(
        graph: &GraphNetwork,
        node_shapes: &HashMap<String, Vec<usize>>,
        input_shape: &[usize],
    ) -> Result<Self, ZonoRefusal> {
        precheck_graph(graph)?;
        let order = graph
            .exec_order()
            .map_err(|e| ZonoRefusal::UnsupportedGraph(format!("execution order: {e}")))?;
        let input_len = numel(input_shape)?;
        let mut prev = NETWORK_INPUT.to_string();
        let mut prev_shape = input_shape.to_vec();
        let mut steps = Vec::with_capacity(order.len());
        let mut batch_norm = Vec::new();
        for name in order {
            let node = graph
                .node(name)
                .ok_or_else(|| ZonoRefusal::UnsupportedGraph(format!("node `{name}` missing")))?;
            let out_shape = node_shapes
                .get(name)
                .cloned()
                .ok_or_else(|| ZonoRefusal::Shape(format!("no shape for node `{name}`")))?;
            let n_in = numel(&prev_shape)?;
            let n_out = numel(&out_shape)?;
            let op = match node.layer() {
                Layer::Linear(layer) => {
                    StepOp::Affine(Box::new(dense_op(name, layer, &prev_shape, n_in, n_out)?))
                }
                Layer::Reshape(_) | Layer::Flatten(_) => {
                    if n_in != n_out {
                        return Err(ZonoRefusal::Shape(format!(
                            "node `{name}`: reshape {prev_shape:?} -> {out_shape:?}"
                        )));
                    }
                    StepOp::Reshape
                }
                Layer::Conv2d(layer) => {
                    StepOp::Affine(Box::new(conv_op(name, layer, &prev_shape, &out_shape)?))
                }
                Layer::ConvTranspose2d(layer) => StepOp::Affine(Box::new(conv_transpose_op(
                    name,
                    layer,
                    &prev_shape,
                    &out_shape,
                )?)),
                Layer::BatchNorm(layer) => {
                    let (op, source) = batch_norm_op(name, layer, &prev_shape, &out_shape)?;
                    batch_norm.push((name.clone(), source));
                    StepOp::ChannelAffine(Box::new(op))
                }
                Layer::ReLU(_) => {
                    if prev_shape != out_shape {
                        return Err(ZonoRefusal::Shape(format!(
                            "node `{name}`: ReLU {prev_shape:?} -> {out_shape:?}"
                        )));
                    }
                    StepOp::Relu
                }
                other => {
                    return Err(ZonoRefusal::UnsupportedLayer {
                        node: name.clone(),
                        layer: other.layer_type(),
                    })
                }
            };
            steps.push(ZonoStep {
                node: name.clone(),
                input: prev.clone(),
                input_shape: prev_shape.clone(),
                out_shape: out_shape.clone(),
                op,
            });
            prev.clone_from(name);
            prev_shape = out_shape;
        }
        Ok(Self {
            input_len,
            input_shape: input_shape.to_vec(),
            steps,
            batch_norm,
        })
    }

    /// The coefficient representation each compiled BatchNorm step charges,
    /// in program order (#zono-exact-bn). Every layer built by
    /// `BatchNormLayer::new` (every ONNX-converted BatchNorm) is
    /// [`BatchNormCoefficients::ExactF64`]; a test or lane that depends on the
    /// tight charge must assert it here rather than infer it from a margin.
    #[must_use]
    pub fn batch_norm_coefficients(&self) -> &[(String, BatchNormCoefficients)] {
        &self.batch_norm
    }

    /// Flat input length.
    #[must_use]
    pub fn input_len(&self) -> usize {
        self.input_len
    }

    /// Input shape the program was compiled for.
    #[must_use]
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    /// Flat output length.
    #[must_use]
    pub fn output_len(&self) -> usize {
        self.steps
            .last()
            .map_or(self.input_len, |step| step.out_shape.iter().product())
    }

    /// The compiled steps.
    #[must_use]
    pub fn steps(&self) -> &[ZonoStep] {
        &self.steps
    }

    /// Plain binary64 concrete forward of `rows` flat input points (row-major
    /// `rows x input_len`) through the compiled program. `visit` sees the
    /// input (`None`) and every step's output (`Some(step)`) as a row-major
    /// block. For containment SAMPLING in tests and diagnostics — it uses the
    /// compiled coefficients (the model's `f32` weights; for BatchNorm the
    /// binary64 centres of [`batch_norm_coefficients`](Self::batch_norm_coefficients)'s
    /// representation) and is not itself a certificate.
    pub fn evaluate_points(
        &self,
        points: &[f64],
        rows: usize,
        mut visit: impl FnMut(Option<usize>, &[f64]),
    ) -> Result<(), ZonoRefusal> {
        if points.len() != rows * self.input_len {
            return Err(ZonoRefusal::Shape(format!(
                "{} point values for {rows} rows of {}",
                points.len(),
                self.input_len
            )));
        }
        let mut current = points.to_vec();
        visit(None, &current);
        for (index, step) in self.steps.iter().enumerate() {
            match &step.op {
                StepOp::Reshape => {}
                StepOp::Affine(op) => {
                    let n_out = op.kernel.n_out();
                    let mut next = vec![0.0f64; rows * n_out];
                    op.kernel
                        .forward_rows(&current, rows, false, &mut next, None)?;
                    for row in next.chunks_exact_mut(n_out.max(1)) {
                        for (v, b) in row.iter_mut().zip(&op.bias) {
                            *v += *b;
                        }
                    }
                    current = next;
                }
                StepOp::ChannelAffine(op) => {
                    let n = op.scale.len();
                    for row in current.chunks_exact_mut(n.max(1)) {
                        for ((v, s), t) in row.iter_mut().zip(&op.scale).zip(&op.shift) {
                            *v = *v * s + t;
                        }
                    }
                }
                StepOp::Relu => {
                    for v in &mut current {
                        *v = v.max(0.0);
                    }
                }
            }
            visit(Some(index), &current);
        }
        Ok(())
    }

    /// The certified intersection of every inherited map's entry for the
    /// ReLU input of `step` (entries with a different shape are skipped; NaN
    /// endpoints are ignored), or `None` when no map has a usable entry.
    fn inherited_box(
        step: &ZonoStep,
        inherited: &[&HashMap<String, BoundedTensor>],
    ) -> Option<(Vec<f64>, Vec<f64>)> {
        let mut acc: Option<(Vec<f64>, Vec<f64>)> = None;
        for map in inherited {
            let Some(entry) = map.get(step.input.as_str()) else {
                continue;
            };
            if entry.shape() != step.input_shape.as_slice() {
                continue;
            }
            let lo: Vec<f64> = entry
                .lower()
                .iter()
                .map(|&v| {
                    if v.is_nan() {
                        f64::NEG_INFINITY
                    } else {
                        f64::from(v)
                    }
                })
                .collect();
            let hi: Vec<f64> = entry
                .upper()
                .iter()
                .map(|&v| {
                    if v.is_nan() {
                        f64::INFINITY
                    } else {
                        f64::from(v)
                    }
                })
                .collect();
            match acc.as_mut() {
                None => acc = Some((lo, hi)),
                Some((acc_lo, acc_hi)) => {
                    intersect_keep_looser(acc_lo, acc_hi, &lo, &hi);
                }
            }
        }
        acc
    }

    /// Per-ReLU upper bound on the unstable count: neurons the inherited maps
    /// leave unstable (all of them when a ReLU has no inherited entry). Since
    /// the forward pass intersects with those same maps, it can never relax
    /// more neurons than this, so it bounds the fresh generators per ReLU.
    #[must_use]
    pub fn unstable_upper_bounds(
        &self,
        inherited: &[&HashMap<String, BoundedTensor>],
    ) -> Vec<usize> {
        self.steps
            .iter()
            .filter(|step| matches!(step.op, StepOp::Relu))
            .map(|step| match Self::inherited_box(step, inherited) {
                Some((lo, hi)) => count_unstable(&lo, &hi),
                None => step.input_shape.iter().product(),
            })
            .collect()
    }

    /// Conservative peak resident bytes of one forward pass: the widest
    /// simultaneous pair of dense generator blocks (input and output of a
    /// step), bounded with the per-ReLU unstable counts and the generator
    /// budget, plus kernel scratch and the published map. A POLICY bound for
    /// admission, never a soundness input.
    #[must_use]
    pub fn estimate_peak_bytes(
        &self,
        input_symbols: usize,
        max_generators: usize,
        unstable_bounds: &[usize],
    ) -> u64 {
        let budget = max_generators.max(input_symbols).max(1);
        let mut k = input_symbols;
        let mut relu = 0usize;
        let mut peak_elems: u128 = 0;
        let mut published_elems: u128 = 0;
        let mut widest: u128 = 0;
        for step in &self.steps {
            let n_in = step.input_shape.iter().product::<usize>() as u128;
            let n_out = step.out_shape.iter().product::<usize>() as u128;
            widest = widest.max(n_in).max(n_out);
            let live = match step.op {
                StepOp::Affine(_) => k as u128 * (n_in + n_out),
                StepOp::ChannelAffine(_) => k as u128 * 2 * n_in,
                StepOp::Reshape => k as u128 * n_in,
                StepOp::Relu => {
                    let unstable = unstable_bounds.get(relu).copied().unwrap_or(usize::MAX / 4);
                    relu += 1;
                    let during = k.saturating_add(unstable);
                    // `forward` keeps the pre-ReLU state (k rows) alive while
                    // `apply_relu` builds the relaxed one (`during` rows), and a
                    // fold then builds the kept copy (`budget` rows) while both
                    // are still live — all three coexist when the budget binds.
                    let fold_copy = if during > budget { budget } else { 0 };
                    let live = (k as u128 + during as u128 + fold_copy as u128) * n_in;
                    k = during.min(budget);
                    published_elems += n_in;
                    live
                }
            };
            peak_elems = peak_elems.max(live);
        }
        // Two kernel scratch buffers, ~12 per-element vectors, the published
        // f32 map (lower + upper), and a fixed floor for allocator slack.
        let scratch = 2 * CHUNK_ELEMS as u128;
        let vectors = 12 * widest;
        let bytes = (peak_elems + scratch + vectors) * 8 + published_elems * 8 + (64u128 << 20);
        u64::try_from(bytes).unwrap_or(u64::MAX)
    }

    /// Conservative bytes of row-task scratch that [`RowPar::Rayon`] kernels
    /// hold AT ONCE across a whole rayon pool of `threads` workers, whatever
    /// the number of concurrent children (#m8-cgan-leaf-batch): at most one
    /// live task per worker, each holding at most the largest per-task
    /// scratch any step allocates ([`LinearKernel::task_scratch_elems`], the
    /// formula the kernels allocate with). Added ONCE per batched tranche on
    /// top of [`Self::estimate_peak_bytes`] of its children. A POLICY bound
    /// for admission, never a soundness input.
    #[must_use]
    pub fn row_task_scratch_bytes(&self, threads: usize) -> u64 {
        let per_task = self
            .steps
            .iter()
            .filter_map(|step| match &step.op {
                StepOp::Affine(op) => Some(op.kernel.task_scratch_elems(usize::MAX)),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        u64::try_from(threads.max(1))
            .unwrap_or(u64::MAX)
            .saturating_mul(u64::try_from(per_task).unwrap_or(u64::MAX))
            .saturating_mul(8)
    }

    /// Run [`Self::forward`] for every child of a tranche CONCURRENTLY, each
    /// with [`RowPar::Rayon`] (#m8-cgan-leaf-batch), so all workers stay busy
    /// even when memory admission holds the tranche narrow.
    ///
    /// Returns, in child order, each child's outcome and its wall time in
    /// nanoseconds. Each outcome is bitwise the [`RowPar::Seq`] forward of
    /// that child alone (the row passes run the reference's own blocks); a
    /// refusal (NaN, deadline, shape) is that child's alone and publishes
    /// nothing for it. `opts.row_par` is ignored (always
    /// [`RowPar::Rayon`]); no observer is attached.
    #[must_use]
    pub fn forward_batch(
        &self,
        children: &[ZonoBatchChild<'_>],
        opts: &ZonoForwardOptions,
    ) -> Vec<(Result<ZonoForwardOutcome, ZonoRefusal>, u64)> {
        let row_opts = ZonoForwardOptions {
            row_par: RowPar::Rayon,
            ..*opts
        };
        let run = |child: &ZonoBatchChild<'_>| {
            let started = Instant::now();
            let result = self.forward(
                child.input,
                &row_opts,
                &child.inherited,
                child.deadline,
                None,
            );
            (
                result,
                u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            )
        };
        if children.len() > 1 {
            children
                .par_iter()
                .map(|child| {
                    let _rayon_task_guard = crate::faer_parallelism::RayonTaskGuard::new();
                    run(child)
                })
                .collect()
        } else {
            children.iter().map(run).collect()
        }
    }

    /// The certified forward pass on `input` (an `f32` box of the compiled
    /// input shape), intersecting every ReLU's pre-activation box with the
    /// `inherited` maps first.
    ///
    /// Fails closed ([`ZonoRefusal`]) on a shape mismatch, a NaN/Inf anywhere
    /// in the state, or the deadline; a refused pass publishes nothing.
    pub fn forward(
        &self,
        input: &BoundedTensor,
        opts: &ZonoForwardOptions,
        inherited: &[&HashMap<String, BoundedTensor>],
        deadline: Option<Instant>,
        mut observer: Option<&mut dyn FnMut(&ZonoNodeView<'_>)>,
    ) -> Result<ZonoForwardOutcome, ZonoRefusal> {
        let started = Instant::now();
        if input.len() != self.input_len {
            return Err(ZonoRefusal::Shape(format!(
                "input box of {} values for a program of {}",
                input.len(),
                self.input_len
            )));
        }
        let raw_lo: Vec<f64> = input.lower().iter().map(|&v| f64::from(v)).collect();
        let raw_hi: Vec<f64> = input.upper().iter().map(|&v| f64::from(v)).collect();
        let mut alloc = SymbolAllocator::new();
        let (mut state, input_symbols) = ZonoState::from_box(&raw_lo, &raw_hi, &mut alloc)?;
        let protected_below = u64::try_from(input_symbols.len()).unwrap_or(u64::MAX);
        let protect = move |id: SymbolId| id.0 < protected_below;
        let budget = opts.max_generators.max(input_symbols.len()).max(1);

        let mut trace = opts.record_trace.then(|| {
            let (lo, hi): (Vec<f64>, Vec<f64>) = input
                .lower()
                .iter()
                .zip(input.upper().iter())
                .map(|(&l, &u)| widen_f32_interval(l, u))
                .unzip();
            ZonoTrace {
                pre_boxes: Vec::new(),
                magnitudes: Vec::with_capacity(self.steps.len() + 1),
                input_lo: lo,
                input_hi: hi,
            }
        });
        let par = opts.row_par;
        let mut row_tasks = 0usize;
        let mut abs_g = state.abs_generator_sum_par(par);
        state.ensure_finite(&abs_g)?;
        if let Some(t) = trace.as_mut() {
            t.magnitudes.push(state.magnitude(&abs_g));
        }
        if let Some(obs) = observer.as_deref_mut() {
            obs(&ZonoNodeView {
                step: None,
                node: NETWORK_INPUT,
                shape: &self.input_shape,
                state: &state,
                abs_g: &abs_g,
                input_symbols: &input_symbols,
            });
        }

        let mut published = HashMap::new();
        let mut relus = Vec::new();
        let mut stats = ZonoForwardStats {
            gens_peak: state.num_generators(),
            max_remainder: state.max_remainder(),
            ..ZonoForwardStats::default()
        };
        for (index, step) in self.steps.iter().enumerate() {
            check_deadline(deadline)?;
            let next = match &step.op {
                StepOp::Reshape => None,
                StepOp::Affine(op) => {
                    #[allow(clippy::cast_precision_loss)]
                    let macs = ((state.num_generators() + 3) * op.kernel.macs_per_row()) as f64;
                    stats.macs += macs;
                    let (next, tasks) = apply_affine_par(&state, &abs_g, op, deadline, par)?;
                    if par == RowPar::Rayon {
                        row_tasks += tasks;
                    }
                    Some(next)
                }
                StepOp::ChannelAffine(op) => {
                    Some(apply_channel_affine_par(&state, &abs_g, op, par)?)
                }
                StepOp::Relu => {
                    let (mut lo, mut hi) = concretize(&state, &abs_g);
                    let zono_unstable = count_unstable(&lo, &hi);
                    let inherited_box = Self::inherited_box(step, inherited);
                    let inherited_unstable =
                        inherited_box.as_ref().map(|(l, h)| count_unstable(l, h));
                    if let Some((il, ih)) = inherited_box.as_ref() {
                        intersect_keep_looser(&mut lo, &mut hi, il, ih);
                    }
                    let unstable = count_unstable(&lo, &hi);
                    if step.input != NETWORK_INPUT {
                        let (lo32, hi32) = to_f32_outward(&lo, &hi);
                        let lower = ArrayD::from_shape_vec(IxDyn(&step.input_shape), lo32)
                            .map_err(|e| ZonoRefusal::Internal(format!("publish lower: {e}")))?;
                        let upper = ArrayD::from_shape_vec(IxDyn(&step.input_shape), hi32)
                            .map_err(|e| ZonoRefusal::Internal(format!("publish upper: {e}")))?;
                        let tensor = BoundedTensor::new_allow_infinite(lower, upper)
                            .map_err(|e| ZonoRefusal::Internal(format!("publish tensor: {e}")))?;
                        published.insert(step.input.clone(), tensor);
                    }
                    let (mut relu_state, _report) =
                        apply_relu_par(&state, &abs_g, &lo, &hi, &mut alloc, par)?;
                    stats.gens_peak = stats.gens_peak.max(relu_state.num_generators());
                    stats.folded += relu_state.fold_to_budget(budget, protect);
                    if let Some(t) = trace.as_mut() {
                        t.pre_boxes.push((lo, hi));
                    }
                    relus.push(ReluSummary {
                        node: step.input.clone(),
                        zono_unstable,
                        inherited_unstable,
                        unstable,
                    });
                    Some(relu_state)
                }
            };
            if let Some(next_state) = next {
                state = next_state;
                abs_g = state.abs_generator_sum_par(par);
                state.ensure_finite(&abs_g)?;
            }
            stats.gens_peak = stats.gens_peak.max(state.num_generators());
            stats.max_remainder = stats.max_remainder.max(state.max_remainder());
            if let Some(t) = trace.as_mut() {
                t.magnitudes.push(state.magnitude(&abs_g));
            }
            if let Some(obs) = observer.as_deref_mut() {
                obs(&ZonoNodeView {
                    step: Some(index),
                    node: &step.node,
                    shape: &step.out_shape,
                    state: &state,
                    abs_g: &abs_g,
                    input_symbols: &input_symbols,
                });
            }
        }
        let (output_lo, output_hi) = concretize(&state, &abs_g);
        stats.gens_final = state.num_generators();
        stats.row_tasks = row_tasks;
        stats.elapsed = started.elapsed();
        Ok(ZonoForwardOutcome {
            published,
            relus,
            input_symbols,
            output_lo,
            output_hi,
            trace,
            stats,
        })
    }
}
