// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! A compiled DAG zonotope program over a [`GraphNetwork`] and its certified
//! forward pass (V4 `#fwd-root-screen`, the dense MulBinary/trig ml4acopf
//! power-flow graphs).
//!
//! It extends the shared core (`zono_core`) instead of forking it: states,
//! rounding, ID-aligned addition, the DeepZ ReLU and the dense kernels are the
//! V2 core's; this module adds a multi-input DAG walk and the element-wise
//! transformers in [`super::elementwise`].
//!
//! # Op set (everything else REFUSES at compile time)
//!
//! | ny layer | transformer |
//! |---|---|
//! | `Linear` (last axis, any leading batch) | exact affine, Higham-charged |
//! | `AddConstant`/`SubConstant`/`MulConstant`/`DivConstant` | per-element affine |
//! | `Add`/`Sub` (live, live) | ID-aligned sum ([`add_aligned`]) |
//! | `MulBinary` (live, live) | ID-aligned product, fresh second-order symbol |
//! | `MulBinary(x, x)`, `PowConstant(2)` | square |
//! | `ReLU` | DeepZ, one fresh symbol per unstable neuron |
//! | `Sigmoid`/`Sin`/`Cos` | chord band ([`smooth_band`]) |
//! | `Floor` | `[x - 1, x]` band or exact constant |
//! | ReLU lookup table (C1's motif) | exact piecewise-linear band |
//! | `Slice`/`Gather`/`Transpose`/`Unsqueeze`/`Squeeze`/`Reshape`/`Flatten`/`Concat` | exact selection |
//!
//! # The vit op set ([`DagOpSet::Vit`], `#vit-zono-root`)
//!
//! Compiled ONLY when the caller asks for [`DagOpSet::Vit`]; the default
//! [`DagOpSet::Base`] (every `#fwd-root-screen` caller) admits exactly the
//! table above, byte for byte. The vit set adds (transformers in
//! [`super::vit_ops`]):
//!
//! | ny layer | transformer |
//! |---|---|
//! | `Conv2d` (patch embedding) | exact affine, Higham-charged ([`apply_affine`]) |
//! | `BatchNorm` | certified per-element affine with ny's own layout expansion and the layer's certified coefficient radii ([`apply_channel_affine`]) |
//! | `BilinearCrown` / live-live `MatMul` | ID-aligned bilinear product, fresh certified remainder symbol per element |
//! | `Exp`, `Reciprocal`, `Clip` | chord/tangent bands (certified `exp`, `1/x` on `x > 0` with wave-6's `[l, +inf)` case, exact clip) |
//! | `ReduceSum`, `ReduceMean` (any axes) | sparse Higham-charged sum (`/ count` for the mean), map verified against ny's layer on integer probes |
//! | `Softmax` (direct) | certified pairwise-difference box, one fresh symbol per element |
//!
//! Data movement is never re-implemented: each selection node's element map
//! is PROBED from ny's own layer on two integer-coded concrete inputs (codes
//! below `2^24`, exact in `f32`), so the zonotope moves exactly the elements
//! ny's IBP moves. A probe whose output is not an exact code-shifted copy (or
//! an unchanged constant) refuses.
//!
//! # Relaxation intervals
//!
//! Every node's box is the zonotope's outward concretisation INTERSECTED with
//! ny's IBP box for that node (and, at a lookup table, with C1's exact range).
//! All are enclosures of the node's true values over the same input box, so
//! the intersection is one too and is never empty; a crossed element proves a
//! defect in one of the two enclosures, so it REFUSES the whole pass (never a
//! collapse, and never a silent pick of one side). The ReLU, the smooth bands,
//! Floor and the lookup band are sized on that box — sound because each band
//! needs only an enclosure of the true input (see [`apply_band`]).
//!
//! # Fail closed
//!
//! Unsupported layer or arity, a shape the program cannot account for, a probe
//! that is not a selection, NaN/Inf anywhere, a crossed box intersection, an
//! IBP map computed over a different input box, the byte budget, or the
//! deadline each REFUSE ([`ZonoRefusal`]); a refused pass publishes nothing.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::time::{Duration, Instant};

use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::layers::{BoundPropagation, Layer};
use crate::network::{check_lookup_shapes, NETWORK_INPUT};
use crate::shape::{broadcast_flat_index_map, broadcast_shapes};
use crate::{GraphNetwork, GraphNode};

use super::affine::{
    apply_affine, apply_channel_affine, AffineOp, BatchNormCoefficients, ChannelAffine,
};
use super::concretize::{concretize, count_unstable, intersect_keep_looser};
use super::elementwise::{
    apply_band, element_affine, floor_band, gather, product, pwl_band, pwl_exact_range, select,
    smooth_band, square, ElementScale, SelectSource, SmoothKind,
};
use super::ids::{add_aligned, SymbolAllocator};
use super::kernels::{check_deadline, LinearKernel, CHUNK_ELEMS};
use super::relu::apply_relu;
use super::rounding::{gamma, next_down, next_up, TINY};
use super::state::{InputSymbol, ZonoState};
use super::vit_ops::{
    bilinear_matmul, box_state, clip_band, clip_range, clip_value, compile_drops_scale, exp_band,
    exp_range, group_sum, matmul_eval, reciprocal_band, reciprocal_range, reduce_group_map,
    softmax_box, softmax_eval, softmax_lanes, MatMulDims,
};
use super::ZonoRefusal;

/// Which op set a [`DagProgram`] may compile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DagOpSet {
    /// The `#fwd-root-screen` op set (the module table), unchanged.
    #[default]
    Base,
    /// Base plus the vit_2023 op set (`#vit-zono-root`; module docs).
    Vit,
}

/// What one DAG step computes.
#[derive(Debug, Clone)]
pub enum DagOp {
    /// Exact selection from the step's inputs (or constants).
    Select {
        /// Source of every output element.
        sources: Vec<SelectSource>,
        /// The map is the identity on a single input (pure relabel).
        identity: bool,
    },
    /// `y[p, o] = sum_i W[o, i] x[p, i] + b[o]` for every leading index `p`.
    Linear {
        /// Dense `out x in` kernel.
        kernel: LinearKernel,
        /// Bias per output feature.
        bias: Vec<f64>,
        /// Number of leading positions `p`.
        batch: usize,
    },
    /// `y_j = x[map_j] * a_j + t_j` or `x[map_j] / d_j + t_j`.
    ElementAffine {
        /// Input element read by each output element (broadcast map).
        map: Vec<usize>,
        /// Scale per output element.
        scale: ElementScale,
        /// Shift per output element.
        shift: Vec<f64>,
    },
    /// `y_j = a[map_a_j] +/- b[map_b_j]`.
    AddSub {
        /// Broadcast map of the first input.
        map_a: Vec<usize>,
        /// Broadcast map of the second input.
        map_b: Vec<usize>,
        /// `a - b` instead of `a + b`.
        subtract: bool,
    },
    /// `y_j = a[map_a_j] * b[map_b_j]` (two distinct live inputs).
    Product {
        /// Broadcast map of the first input.
        map_a: Vec<usize>,
        /// Broadcast map of the second input.
        map_b: Vec<usize>,
    },
    /// `y = x * x` element-wise.
    Square,
    /// DeepZ ReLU.
    Relu,
    /// Chord band of a smooth function.
    Smooth(SmoothKind),
    /// Floor band.
    Floor,
    /// Exact band of a ReLU lookup table over its scalar input.
    Pwl {
        /// Breakpoints.
        thresholds: Vec<f32>,
        /// Hinge weights.
        weights: Vec<f32>,
        /// Constant after the weighted sum.
        bias: f32,
    },
    /// `y = A x + b` over the whole flat value (vit: the Conv2d patch
    /// embedding).
    Affine(AffineOp),
    /// Certified per-element affine (vit: BatchNormalization).
    ChannelAffine(ChannelAffine),
    /// `y_j = sum_{i : out_of[i] = j} x_i` (ReduceSum), divided by the exact
    /// group `count` when `mean` (ReduceMean).
    GroupSum {
        /// Output element of every input element.
        out_of: Vec<usize>,
        /// Output length.
        n_out: usize,
        /// Inputs summed into every output.
        count: usize,
        /// Divide by `count` (ReduceMean).
        mean: bool,
    },
    /// Live x live batched MatMul / BilinearCrown.
    MatMul(MatMulDims),
    /// Exp band.
    Exp,
    /// Reciprocal band (input strictly positive).
    Reciprocal,
    /// Clip band.
    Clip {
        /// Lower clip value (exact binary64 image of the layer's `f32`).
        lo: f64,
        /// Upper clip value.
        hi: f64,
    },
    /// Direct Softmax: certified pairwise-difference box per lane.
    Softmax {
        /// Flat input indices of every normalised lane.
        lanes: Vec<Vec<usize>>,
    },
}

impl DagOp {
    /// Stable lowercase kind token (observables, census).
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Select { .. } => "select",
            Self::Linear { .. } => "linear",
            Self::ElementAffine { .. } => "affine",
            Self::AddSub { .. } => "addsub",
            Self::Product { .. } => "mul",
            Self::Square => "square",
            Self::Relu => "relu",
            Self::Smooth(kind) => kind.token(),
            Self::Floor => "floor",
            Self::Pwl { .. } => "pwl",
            Self::Affine(_) => "conv",
            Self::ChannelAffine(_) => "bn",
            Self::GroupSum { .. } => "reduce",
            Self::MatMul(_) => "matmul",
            Self::Exp => "exp",
            Self::Reciprocal => "recip",
            Self::Clip { .. } => "clip",
            Self::Softmax { .. } => "softmax",
        }
    }
}

/// The kind tokens in census order (a zero count is omitted from the census,
/// so a Base program's census is unchanged by the vit kinds).
pub const DAG_OP_KINDS: [&str; 22] = [
    "linear", "select", "affine", "addsub", "mul", "square", "relu", "sigmoid", "sin", "cos",
    "floor", "pwl", "conv", "bn", "reduce", "matmul", "exp", "recip", "clip", "softmax", "skipped",
    "nodes",
];

/// One compiled step.
#[derive(Debug, Clone)]
pub struct DagStep {
    /// Graph node computed by this step.
    pub node: String,
    /// Live inputs the transformer reads (node names or [`NETWORK_INPUT`]).
    pub inputs: Vec<String>,
    /// Output shape as ny propagates it.
    pub out_shape: Vec<usize>,
    /// The transformer.
    pub op: DagOp,
}

/// A compiled DAG program.
#[derive(Debug, Clone)]
pub struct DagProgram {
    input_len: usize,
    input_shape: Vec<usize>,
    output: String,
    steps: Vec<DagStep>,
    skipped: Vec<String>,
    last_use: HashMap<String, usize>,
    /// Bit patterns of the `f32` input box the IBP map returned by
    /// [`DagProgram::compile_with_ibp`] was computed over (`None` for
    /// [`DagProgram::compile`], which computes no map). An intersecting
    /// forward pass refuses any other input box: a map from one box is not an
    /// enclosure over another (the stale-map-across-domains defect class).
    ibp_input: Option<(Vec<u32>, Vec<u32>)>,
    /// Coefficient representation of every compiled BatchNorm step, in
    /// order (#zono-exact-bn).
    batch_norm: Vec<(String, BatchNormCoefficients)>,
}

/// Options for one forward pass.
#[derive(Debug, Clone, Copy)]
pub struct DagForwardOptions {
    /// Intersect every node box with ny's IBP box (and lookup tables with C1's
    /// exact range). Production is always `true`; `false` reproduces the F1
    /// replica's pure ZONO pass for the parity pin.
    pub intersect_ibp: bool,
    /// Live-state byte budget; a step that would exceed it REFUSES.
    pub byte_budget: u64,
}

/// Cost and charge statistics of one forward pass.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DagForwardStats {
    /// Symbols allocated (input symbols included).
    pub symbols: u64,
    /// Input symbols.
    pub input_symbols: usize,
    /// Fresh symbols from unstable ReLUs.
    pub fresh_relu: usize,
    /// Fresh symbols from products (second-order terms).
    pub fresh_mul: usize,
    /// Fresh symbols from squares.
    pub fresh_square: usize,
    /// Fresh symbols from smooth, Floor and lookup-table bands (and, in the
    /// vit set, the Exp / Reciprocal / Clip bands).
    pub fresh_band: usize,
    /// Fresh symbols from live-live MatMul second-order remainders (vit set).
    pub fresh_bilinear: usize,
    /// Fresh symbols from direct-Softmax boxes (vit set).
    pub fresh_softmax: usize,
    /// Unstable ReLU neurons after intersection.
    pub relu_unstable: usize,
    /// Most generator rows any node held.
    pub gens_peak: usize,
    /// Largest live-state byte count observed.
    pub live_bytes_peak: u64,
    /// Largest remainder radius over every node (the certified charge).
    pub max_remainder: f64,
    /// Elements whose box strictly shrank through the IBP / C1 intersection.
    pub ibp_tightened: usize,
    /// Wall time of the pass.
    pub elapsed: Duration,
}

/// Result of one forward pass.
#[derive(Debug, Clone)]
pub struct DagForwardOutcome {
    /// Certified output box (concretisation intersected with IBP when armed).
    pub output_lo: Vec<f64>,
    /// Upper end of the output box.
    pub output_hi: Vec<f64>,
    /// Output shape.
    pub output_shape: Vec<usize>,
    /// The input symbols (for pointwise form checks).
    pub input_symbols: Vec<InputSymbol>,
    /// Statistics.
    pub stats: DagForwardStats,
}

/// A read-only view of one node after its step, for observers (tests).
pub struct DagNodeView<'a> {
    /// Node name ([`NETWORK_INPUT`] for the input).
    pub node: &'a str,
    /// Node shape as ny propagates it.
    pub shape: &'a [usize],
    /// The state.
    pub state: &'a ZonoState,
    /// `state.abs_generator_sum()`.
    pub abs_g: &'a [f64],
    /// The node's box (concretisation, intersected when armed).
    pub lo: &'a [f64],
    /// Upper end of the node box.
    pub hi: &'a [f64],
    /// The input symbols of this pass.
    pub input_symbols: &'a [InputSymbol],
}

/// A computed node: state, its certified absolute sums and its box.
struct NodeVal {
    state: ZonoState,
    abs_g: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
}

impl NodeVal {
    fn bytes(&self) -> u64 {
        let n = self.state.len() as u64;
        self.state
            .generator_bytes()
            .saturating_add(n.saturating_mul(8 * 5))
    }
}

/// Bit patterns of an `f32` box's lower and upper ends (bitwise identity,
/// so `-0.0`/`0.0` and every NaN payload are told apart).
fn input_box_bits(input: &BoundedTensor) -> (Vec<u32>, Vec<u32>) {
    (
        input.lower().iter().map(|v| v.to_bits()).collect(),
        input.upper().iter().map(|v| v.to_bits()).collect(),
    )
}

fn numel(shape: &[usize]) -> Result<usize, ZonoRefusal> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| ZonoRefusal::Shape(format!("shape {shape:?} overflows usize")))
}

/// Integer code offset between the two selection probes (`2^23`); every code
/// and code + offset stays below `2^24`, so both are exact in `f32`.
const PROBE_SHIFT: f32 = 8_388_608.0;
/// Largest total probe code (exclusive).
const PROBE_MAX_CODES: usize = 1 << 23;

/// Layers the program represents (a structural pre-check before any shape
/// pass). The lookup-table `ReduceSum` head is admitted only as C1's motif in
/// the Base set; the vit set also admits the layers of [`DagOpSet::Vit`].
fn layer_supported(graph: &GraphNetwork, node: &GraphNode, opset: DagOpSet) -> bool {
    let vit = opset == DagOpSet::Vit
        && matches!(
            node.layer(),
            Layer::Conv2d(_)
                | Layer::BatchNorm(_)
                | Layer::BilinearCrown(_)
                | Layer::MatMul(_)
                | Layer::Exp(_)
                | Layer::Reciprocal(_)
                | Layer::Clip(_)
                | Layer::Softmax(_)
                | Layer::ReduceSum(_)
                | Layer::ReduceMean(_)
        );
    vit || match node.layer() {
        Layer::Linear(_)
        | Layer::ReLU(_)
        | Layer::Sigmoid(_)
        | Layer::Sin(_)
        | Layer::Cos(_)
        | Layer::Floor(_)
        | Layer::PowConstant(_)
        | Layer::MulBinary(_)
        | Layer::Add(_)
        | Layer::Sub(_)
        | Layer::AddConstant(_)
        | Layer::SubConstant(_)
        | Layer::MulConstant(_)
        | Layer::DivConstant(_)
        | Layer::Slice(_)
        | Layer::Transpose(_)
        | Layer::Unsqueeze(_)
        | Layer::Squeeze(_)
        | Layer::Reshape(_)
        | Layer::Flatten(_)
        | Layer::Concat(_) => true,
        Layer::Gather(gather) => {
            gather.constant_indices().is_some() && !gather.is_runtime_last_axis_len()
        }
        Layer::ReduceSum(_) => graph.match_pwl_lookup(node).is_some(),
        _ => false,
    }
}

/// Structural pre-check: refuse any node the program cannot represent BEFORE
/// spending an IBP pass on it (the Base op set).
pub fn precheck_dag(graph: &GraphNetwork) -> Result<(), ZonoRefusal> {
    precheck_dag_opset(graph, DagOpSet::Base)
}

/// [`precheck_dag`] for an explicit op set.
pub fn precheck_dag_opset(graph: &GraphNetwork, opset: DagOpSet) -> Result<(), ZonoRefusal> {
    graph
        .ensure_proof_eligible()
        .map_err(|error| ZonoRefusal::UnsupportedGraph(error.to_string()))?;
    let order = graph
        .exec_order()
        .map_err(|e| ZonoRefusal::UnsupportedGraph(format!("execution order: {e}")))?;
    if order.is_empty() {
        return Err(ZonoRefusal::UnsupportedGraph("empty graph".to_string()));
    }
    for name in order {
        let node = graph
            .node(name)
            .ok_or_else(|| ZonoRefusal::UnsupportedGraph(format!("node `{name}` missing")))?;
        if !layer_supported(graph, node, opset) {
            return Err(ZonoRefusal::UnsupportedLayer {
                node: name.clone(),
                layer: node.layer().layer_type(),
            });
        }
    }
    Ok(())
}

/// A matched lookup-table motif ending at a head node.
struct PwlMotif {
    scalar: String,
    thresholds: Vec<f32>,
    weights: Vec<f32>,
    bias: f32,
}

/// Probe the element map of a pure data-movement node from ny's own layer.
fn probe_selection(
    node: &GraphNode,
    name: &str,
    input_shapes: &[Vec<usize>],
    out_shape: &[usize],
) -> Result<Vec<SelectSource>, ZonoRefusal> {
    let sizes: Vec<usize> = input_shapes
        .iter()
        .map(|shape| numel(shape))
        .collect::<Result<_, _>>()?;
    let total: usize = sizes.iter().sum();
    if total == 0 || total >= PROBE_MAX_CODES {
        return Err(ZonoRefusal::UnsupportedGraph(format!(
            "node `{name}`: {total} probe codes (limit {PROBE_MAX_CODES})"
        )));
    }
    let run = |offset: f32| -> Result<BoundedTensor, ZonoRefusal> {
        let mut base = 0usize;
        let mut probes = Vec::with_capacity(input_shapes.len());
        for (shape, &count) in input_shapes.iter().zip(&sizes) {
            #[allow(clippy::cast_precision_loss)]
            let values: Vec<f32> = (0..count).map(|j| (base + j + 1) as f32 + offset).collect();
            base += count;
            let array = ArrayD::from_shape_vec(IxDyn(shape), values)
                .map_err(|e| ZonoRefusal::Internal(format!("probe `{name}`: {e}")))?;
            probes.push(
                BoundedTensor::concrete(array)
                    .map_err(|e| ZonoRefusal::Internal(format!("probe `{name}`: {e}")))?,
            );
        }
        let refused = |e: ny_core::NyError| {
            ZonoRefusal::UnsupportedGraph(format!("probe of `{name}` failed: {e}"))
        };
        match node.layer() {
            Layer::Concat(concat) => {
                let mut ordered: Vec<BoundedTensor> = Vec::new();
                match concat.constant_inputs.as_ref() {
                    Some(slots) => {
                        let mut live = probes.iter();
                        for slot in slots {
                            match slot {
                                Some(constant) => ordered.push(constant.clone()),
                                None => ordered.push(
                                    live.next()
                                        .ok_or_else(|| {
                                            ZonoRefusal::UnsupportedGraph(format!(
                                                "concat `{name}`: fewer graph inputs than slots"
                                            ))
                                        })?
                                        .clone(),
                                ),
                            }
                        }
                        if live.next().is_some() {
                            return Err(ZonoRefusal::UnsupportedGraph(format!(
                                "concat `{name}`: more graph inputs than slots"
                            )));
                        }
                    }
                    None => ordered.extend(probes.iter().cloned()),
                }
                let refs: Vec<&BoundedTensor> = ordered.iter().collect();
                concat.propagate_ibp_nary(&refs).map_err(refused)
            }
            layer => {
                if probes.len() != 1 {
                    return Err(ZonoRefusal::UnsupportedGraph(format!(
                        "node `{name}`: {} inputs for a unary data-movement layer",
                        probes.len()
                    )));
                }
                layer.propagate_ibp(&probes[0]).map_err(refused)
            }
        }
    };
    let first = run(0.0)?;
    let second = run(PROBE_SHIFT)?;
    if first.shape() != out_shape || second.shape() != out_shape {
        return Err(ZonoRefusal::Shape(format!(
            "node `{name}`: probe shape {:?} vs IBP shape {out_shape:?}",
            first.shape()
        )));
    }
    let mut bases = Vec::with_capacity(sizes.len());
    let mut acc = 0usize;
    for &count in &sizes {
        bases.push(acc);
        acc += count;
    }
    let mut sources = Vec::with_capacity(first.len());
    for (((&l0, &u0), &l1), &u1) in first
        .lower()
        .iter()
        .zip(first.upper().iter())
        .zip(second.lower().iter())
        .zip(second.upper().iter())
    {
        let is_code = l0 == u0
            && l1 == u1
            && l1 - l0 == PROBE_SHIFT
            && l0 >= 1.0
            && l0.fract() == 0.0
            && f64::from(l0) <= total as f64;
        if is_code {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let code = l0 as usize - 1;
            let input = bases
                .iter()
                .rposition(|&b| b <= code)
                .ok_or_else(|| ZonoRefusal::Internal(format!("probe `{name}` code {code}")))?;
            let elem = code - bases[input];
            if elem >= sizes[input] {
                return Err(ZonoRefusal::Internal(format!(
                    "probe `{name}` code {code} past input {input}"
                )));
            }
            sources.push(SelectSource::Live { input, elem });
        } else if l0 == l1 && u0 == u1 && l0.is_finite() && u0.is_finite() && l0 <= u0 {
            sources.push(SelectSource::Const {
                lo: f64::from(l0),
                hi: f64::from(u0),
            });
        } else {
            return Err(ZonoRefusal::UnsupportedGraph(format!(
                "node `{name}` ({}) is not an exact selection under probing",
                node.layer().layer_type()
            )));
        }
    }
    Ok(sources)
}

/// Standard ONNX broadcast of two shapes; the result must be `out_shape`.
fn broadcast_maps(
    name: &str,
    a: &[usize],
    b: &[usize],
    out_shape: &[usize],
) -> Result<(Vec<usize>, Vec<usize>), ZonoRefusal> {
    let target = broadcast_shapes(a, b).ok_or_else(|| {
        ZonoRefusal::Shape(format!("node `{name}`: {a:?} and {b:?} do not broadcast"))
    })?;
    if target != out_shape {
        return Err(ZonoRefusal::Shape(format!(
            "node `{name}`: broadcast {target:?} vs IBP shape {out_shape:?}"
        )));
    }
    Ok((
        broadcast_flat_index_map(out_shape, a),
        broadcast_flat_index_map(out_shape, b),
    ))
}

/// `(input map, constant values per output element)` for a live input of
/// `in_shape` combined with `constant` broadcast as `effective` (the layer's
/// own broadcast shape for the constant).
fn constant_maps(
    name: &str,
    in_shape: &[usize],
    constant: &ArrayD<f32>,
    effective: &[usize],
    out_shape: &[usize],
) -> Result<(Vec<usize>, Vec<f64>), ZonoRefusal> {
    let values: Vec<f64> = constant.iter().map(|&v| f64::from(v)).collect();
    if numel(effective)? != values.len() {
        return Err(ZonoRefusal::Shape(format!(
            "node `{name}`: constant of {} values viewed as {effective:?}",
            values.len()
        )));
    }
    let (map, const_map) = broadcast_maps(name, in_shape, effective, out_shape)?;
    let per_output = const_map.iter().map(|&k| values[k]).collect();
    Ok((map, per_output))
}

fn require_inputs(node: &GraphNode, name: &str, count: usize) -> Result<(), ZonoRefusal> {
    if node.inputs().len() == count {
        Ok(())
    } else {
        Err(ZonoRefusal::UnsupportedGraph(format!(
            "node `{name}` ({}) has {} inputs, expected {count}",
            node.layer().layer_type(),
            node.inputs().len()
        )))
    }
}

/// Dimensions of a live-live `MatMul`/`BilinearCrown` node from ny's own IBP
/// shapes: `A [.., m, k]`, `B [.., k, n]` (`[.., n, k]` when `transpose_b`),
/// IDENTICAL leading (batch) axes — the only form ny's MatMul IBP accepts —
/// and output `[.., m, n]`. Anything else refuses.
fn matmul_dims(
    node: &GraphNode,
    name: &str,
    input_shapes: &[Vec<usize>],
    out_shape: &[usize],
    transpose_b: bool,
    scale: Option<f32>,
) -> Result<MatMulDims, ZonoRefusal> {
    require_inputs(node, name, 2)?;
    let (a, b) = (&input_shapes[0], &input_shapes[1]);
    if a.len() < 2 || b.len() < 2 || a.len() != b.len() {
        return Err(ZonoRefusal::Shape(format!(
            "node `{name}`: MatMul operands {a:?} x {b:?}"
        )));
    }
    let r = a.len();
    if a[..r - 2] != b[..r - 2] {
        return Err(ZonoRefusal::Shape(format!(
            "node `{name}`: MatMul batch axes {a:?} x {b:?}"
        )));
    }
    let (m, k) = (a[r - 2], a[r - 1]);
    let (kb, n) = if transpose_b {
        (b[r - 1], b[r - 2])
    } else {
        (b[r - 2], b[r - 1])
    };
    let mut expected = a[..r - 2].to_vec();
    expected.extend([m, n]);
    if kb != k || expected != out_shape {
        return Err(ZonoRefusal::Shape(format!(
            "node `{name}`: MatMul {a:?} x {b:?} (transpose_b={transpose_b}) -> {out_shape:?}"
        )));
    }
    let batch = numel(&a[..r - 2])?;
    let scale = if compile_drops_scale() {
        None
    } else {
        scale.map(f64::from)
    };
    MatMulDims::new(batch, m, k, n, transpose_b, scale)
}

/// Integer probe codes for [`verify_reduce_map`]: small positive integers, so
/// every group sum is an exact `f32` integer (`count * 97 < 2^24`).
const REDUCE_PROBE_MAX_COUNT: usize = 1 << 16;

/// Cross-check a derived reduction map against ny's OWN layer: two
/// integer-coded probes (exact in `f32`) go through the layer's IBP, and every
/// output must enclose the map's exact sum (or mean). A map that disagrees
/// with the layer REFUSES — the zonotope must move exactly the elements ny's
/// IBP moves.
fn verify_reduce_map(
    node: &GraphNode,
    name: &str,
    in_shape: &[usize],
    out_of: &[usize],
    n_out: usize,
    count: usize,
    mean: bool,
) -> Result<(), ZonoRefusal> {
    if count > REDUCE_PROBE_MAX_COUNT {
        return Err(ZonoRefusal::UnsupportedGraph(format!(
            "node `{name}`: reduction of {count} terms is past the probe limit"
        )));
    }
    let codings: [fn(usize) -> usize; 2] = [|i| i % 97 + 1, |i| (i * 31 + 7) % 89 + 1];
    for code in codings {
        #[allow(clippy::cast_precision_loss)]
        let values: Vec<f32> = (0..out_of.len()).map(|i| code(i) as f32).collect();
        let array = ArrayD::from_shape_vec(IxDyn(in_shape), values)
            .map_err(|e| ZonoRefusal::Internal(format!("reduce probe `{name}`: {e}")))?;
        let probe = BoundedTensor::concrete(array)
            .map_err(|e| ZonoRefusal::Internal(format!("reduce probe `{name}`: {e}")))?;
        let out = node.layer().propagate_ibp(&probe).map_err(|e| {
            ZonoRefusal::UnsupportedGraph(format!("reduce probe of `{name}` failed: {e}"))
        })?;
        if out.len() != n_out {
            return Err(ZonoRefusal::Shape(format!(
                "node `{name}`: reduce probe gave {} values, map {n_out}",
                out.len()
            )));
        }
        let mut sums = vec![0.0f64; n_out];
        for (i, &j) in out_of.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let value = code(i) as f64;
            sums[j] += value;
        }
        #[allow(clippy::cast_precision_loss)]
        let divisor = if mean { count as f64 } else { 1.0 };
        for (j, (&l, &u)) in out.lower().iter().zip(out.upper().iter()).enumerate() {
            let expected = sums[j] / divisor;
            if !(f64::from(l) <= expected && expected <= f64::from(u)) {
                return Err(ZonoRefusal::UnsupportedGraph(format!(
                    "node `{name}`: reduction map disagrees with ny's layer at output {j}: \
                     {expected} vs [{l}, {u}]"
                )));
            }
        }
    }
    Ok(())
}

impl DagProgram {
    /// Compile with shapes (and the per-node boxes the forward intersects
    /// with) from ny's own IBP collection on `input`, bounded by `deadline`.
    /// The structural pre-check runs first, so an unsupported graph refuses
    /// without an IBP pass.
    pub fn compile_with_ibp(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<(Self, HashMap<String, BoundedTensor>), ZonoRefusal> {
        Self::compile_with_ibp_opset(graph, input, deadline, DagOpSet::Base)
    }

    /// [`Self::compile_with_ibp`] for an explicit op set.
    pub fn compile_with_ibp_opset(
        graph: &GraphNetwork,
        input: &BoundedTensor,
        deadline: Option<Instant>,
        opset: DagOpSet,
    ) -> Result<(Self, HashMap<String, BoundedTensor>), ZonoRefusal> {
        precheck_dag_opset(graph, opset)?;
        let ibp = graph
            .collect_node_bounds_with_engine_and_deadline(input, None, deadline)
            .map_err(|e| {
                if e.is_deadline_exceeded() {
                    ZonoRefusal::Deadline
                } else {
                    ZonoRefusal::UnsupportedGraph(format!("IBP shape pass: {e}"))
                }
            })?;
        let shapes: HashMap<String, Vec<usize>> = ibp
            .iter()
            .map(|(name, bounds)| (name.clone(), bounds.shape().to_vec()))
            .collect();
        let mut program = Self::compile_opset(graph, &shapes, input.shape(), opset)?;
        program.ibp_input = Some(input_box_bits(input));
        Ok((program, ibp))
    }

    /// Compile from explicit per-node shapes and the input shape.
    pub fn compile(
        graph: &GraphNetwork,
        node_shapes: &HashMap<String, Vec<usize>>,
        input_shape: &[usize],
    ) -> Result<Self, ZonoRefusal> {
        Self::compile_opset(graph, node_shapes, input_shape, DagOpSet::Base)
    }

    /// [`Self::compile`] for an explicit op set.
    pub fn compile_opset(
        graph: &GraphNetwork,
        node_shapes: &HashMap<String, Vec<usize>>,
        input_shape: &[usize],
        opset: DagOpSet,
    ) -> Result<Self, ZonoRefusal> {
        precheck_dag_opset(graph, opset)?;
        let order = graph
            .exec_order()
            .map_err(|e| ZonoRefusal::UnsupportedGraph(format!("execution order: {e}")))?;
        let output = if graph.output_name().is_empty() {
            order
                .last()
                .cloned()
                .ok_or_else(|| ZonoRefusal::UnsupportedGraph("empty graph".to_string()))?
        } else {
            graph.output_name().to_string()
        };
        let input_len = numel(input_shape)?;
        let input_shape_vec = input_shape.to_vec();
        let shape_of = |name: &str| -> Result<Vec<usize>, ZonoRefusal> {
            if name == NETWORK_INPUT {
                Ok(input_shape_vec.clone())
            } else {
                node_shapes
                    .get(name)
                    .cloned()
                    .ok_or_else(|| ZonoRefusal::Shape(format!("no shape for node `{name}`")))
            }
        };

        // Consumer counts (the graph output counts as one consumer).
        let mut consumers: HashMap<String, usize> = HashMap::new();
        for name in order {
            let node = graph
                .node(name)
                .ok_or_else(|| ZonoRefusal::UnsupportedGraph(format!("node `{name}` missing")))?;
            for input in node.inputs() {
                *consumers.entry(input.clone()).or_insert(0) += 1;
            }
        }
        *consumers.entry(output.clone()).or_insert(0) += 1;

        // Lookup-table motifs (C1's structural match and runtime shape check).
        let mut motifs: HashMap<String, PwlMotif> = HashMap::new();
        let mut skip: HashSet<String> = HashSet::new();
        for name in order {
            let Some(node) = graph.node(name) else {
                continue;
            };
            if !matches!(node.layer(), Layer::Linear(_) | Layer::ReduceSum(_)) {
                continue;
            }
            let Some(table) = graph.match_pwl_lookup(node) else {
                continue;
            };
            let (Ok(scalar_shape), Ok(sub_shape), Ok(out_shape)) = (
                shape_of(table.scalar_input.as_str()),
                shape_of(table.sub_node.as_str()),
                shape_of(name.as_str()),
            ) else {
                continue;
            };
            let mul_shape = match table.mul_node.as_deref() {
                Some(mul) => match shape_of(mul) {
                    Ok(shape) => Some(shape),
                    Err(_) => continue,
                },
                None => None,
            };
            if check_lookup_shapes(
                &table,
                &scalar_shape,
                &sub_shape,
                mul_shape.as_deref(),
                &out_shape,
            )
            .is_err()
            {
                continue;
            }
            let relu = match table.mul_node.as_deref() {
                Some(mul) => graph.node(mul).and_then(|m| m.inputs().first().cloned()),
                None => node.inputs().first().cloned(),
            };
            let Some(relu) = relu else {
                continue;
            };
            let mut chain = vec![table.sub_node.clone(), relu];
            if let Some(mul) = table.mul_node.clone() {
                chain.push(mul);
            }
            // Skip the hinge nodes only when nothing but the motif reads them;
            // otherwise they are computed normally for their other readers.
            if chain
                .iter()
                .all(|member| consumers.get(member).copied() == Some(1))
            {
                skip.extend(chain);
            }
            motifs.insert(
                name.clone(),
                PwlMotif {
                    scalar: table.scalar_input.clone(),
                    thresholds: table.thresholds.clone(),
                    weights: table.weights.clone(),
                    bias: table.bias,
                },
            );
        }
        if skip.contains(&output) {
            return Err(ZonoRefusal::UnsupportedGraph(
                "the graph output is inside a lookup-table motif".to_string(),
            ));
        }

        let mut steps = Vec::with_capacity(order.len());
        let mut skipped = Vec::new();
        let mut batch_norm = Vec::new();
        for name in order {
            if skip.contains(name) {
                skipped.push(name.clone());
                continue;
            }
            let node = graph
                .node(name)
                .ok_or_else(|| ZonoRefusal::UnsupportedGraph(format!("node `{name}` missing")))?;
            let out_shape = shape_of(name.as_str())?;
            let (inputs, op) = match motifs.remove(name) {
                Some(motif) => {
                    let scalar_len = numel(&shape_of(motif.scalar.as_str())?)?;
                    if scalar_len != numel(&out_shape)? {
                        return Err(ZonoRefusal::Shape(format!(
                            "lookup table `{name}`: {scalar_len} scalars for {out_shape:?}"
                        )));
                    }
                    (
                        vec![motif.scalar],
                        DagOp::Pwl {
                            thresholds: motif.thresholds,
                            weights: motif.weights,
                            bias: motif.bias,
                        },
                    )
                }
                None => Self::compile_node(node, name, &shape_of, &out_shape, opset)?,
            };
            if let (Layer::BatchNorm(layer), DagOp::ChannelAffine(_)) = (node.layer(), &op) {
                // The same predicate `expanded_exact_affine_parameters` uses to
                // choose the exact enclosure over the binary32 affine.
                let source = if layer.exact_affine_channels().is_some() {
                    BatchNormCoefficients::ExactF64
                } else {
                    BatchNormCoefficients::Binary32
                };
                batch_norm.push((name.clone(), source));
            }
            steps.push(DagStep {
                node: name.clone(),
                inputs,
                out_shape,
                op,
            });
        }
        if !steps.iter().any(|step| step.node == output) {
            return Err(ZonoRefusal::UnsupportedGraph(format!(
                "output node `{output}` is not computed"
            )));
        }
        let mut last_use = HashMap::new();
        for (index, step) in steps.iter().enumerate() {
            for input in &step.inputs {
                last_use.insert(input.clone(), index);
            }
        }
        Ok(Self {
            input_len,
            input_shape: input_shape.to_vec(),
            output,
            steps,
            skipped,
            last_use,
            ibp_input: None,
            batch_norm,
        })
    }

    /// The coefficient representation each compiled BatchNorm step charges,
    /// in program order (#zono-exact-bn; see
    /// [`super::program::ZonoProgram::batch_norm_coefficients`]).
    #[must_use]
    pub fn batch_norm_coefficients(&self) -> &[(String, BatchNormCoefficients)] {
        &self.batch_norm
    }

    fn compile_node<F>(
        node: &GraphNode,
        name: &str,
        shape_of: &F,
        out_shape: &[usize],
        opset: DagOpSet,
    ) -> Result<(Vec<String>, DagOp), ZonoRefusal>
    where
        F: Fn(&str) -> Result<Vec<usize>, ZonoRefusal>,
    {
        let inputs: Vec<String> = node.inputs().to_vec();
        let input_shapes: Vec<Vec<usize>> = inputs
            .iter()
            .map(|input| shape_of(input.as_str()))
            .collect::<Result<_, _>>()?;
        let n_out = numel(out_shape)?;
        let elementwise = |count: usize| -> Result<(), ZonoRefusal> {
            require_inputs(node, name, count)?;
            if input_shapes.iter().any(|s| s.as_slice() != out_shape) {
                return Err(ZonoRefusal::Shape(format!(
                    "node `{name}`: element-wise op {input_shapes:?} -> {out_shape:?}"
                )));
            }
            Ok(())
        };
        let op = match node.layer() {
            Layer::Linear(layer) => {
                require_inputs(node, name, 1)?;
                let in_shape = &input_shapes[0];
                let weight = layer.weight();
                let (rows, cols) = weight.dim();
                let n_in = numel(in_shape)?;
                if in_shape.last() != Some(&cols) || cols == 0 || n_in % cols != 0 {
                    return Err(ZonoRefusal::Shape(format!(
                        "node `{name}`: Linear {rows}x{cols} on input {in_shape:?}"
                    )));
                }
                let batch = n_in / cols;
                if batch.checked_mul(rows) != Some(n_out) {
                    return Err(ZonoRefusal::Shape(format!(
                        "node `{name}`: Linear output {out_shape:?} for batch {batch} x {rows}"
                    )));
                }
                let w: Vec<f64> = weight.iter().map(|&v| f64::from(v)).collect();
                let w_abs: Vec<f64> = w.iter().map(|v| v.abs()).collect();
                let bias = match layer.bias() {
                    None => vec![0.0; rows],
                    Some(b) if b.len() == rows => b.iter().map(|&v| f64::from(v)).collect(),
                    Some(b) => {
                        return Err(ZonoRefusal::Shape(format!(
                            "node `{name}`: Linear bias of {} values for {rows} outputs",
                            b.len()
                        )))
                    }
                };
                DagOp::Linear {
                    kernel: LinearKernel::Dense {
                        n_in: cols,
                        n_out: rows,
                        w,
                        w_abs,
                    },
                    bias,
                    batch,
                }
            }
            Layer::ReLU(_) => {
                elementwise(1)?;
                DagOp::Relu
            }
            Layer::Sigmoid(_) => {
                elementwise(1)?;
                DagOp::Smooth(SmoothKind::Sigmoid)
            }
            Layer::Sin(_) => {
                elementwise(1)?;
                DagOp::Smooth(SmoothKind::Sin)
            }
            Layer::Cos(_) => {
                elementwise(1)?;
                DagOp::Smooth(SmoothKind::Cos)
            }
            Layer::Floor(_) => {
                elementwise(1)?;
                DagOp::Floor
            }
            Layer::PowConstant(pow) => {
                elementwise(1)?;
                if pow.exponent() != 2.0 {
                    return Err(ZonoRefusal::UnsupportedGraph(format!(
                        "node `{name}`: PowConstant exponent {} (only 2 is supported)",
                        pow.exponent()
                    )));
                }
                DagOp::Square
            }
            Layer::MulBinary(_) => {
                require_inputs(node, name, 2)?;
                if inputs[0] == inputs[1] {
                    elementwise(2)?;
                    return Ok((vec![inputs[0].clone()], DagOp::Square));
                }
                let (map_a, map_b) =
                    broadcast_maps(name, &input_shapes[0], &input_shapes[1], out_shape)?;
                DagOp::Product { map_a, map_b }
            }
            Layer::Add(_) | Layer::Sub(_) => {
                require_inputs(node, name, 2)?;
                let (map_a, map_b) =
                    broadcast_maps(name, &input_shapes[0], &input_shapes[1], out_shape)?;
                DagOp::AddSub {
                    map_a,
                    map_b,
                    subtract: matches!(node.layer(), Layer::Sub(_)),
                }
            }
            Layer::AddConstant(layer) => {
                require_inputs(node, name, 1)?;
                let effective = layer
                    .effective_broadcast_shape(&input_shapes[0])
                    .map_err(|e| ZonoRefusal::UnsupportedGraph(format!("node `{name}`: {e}")))?;
                let (map, shift) = constant_maps(
                    name,
                    &input_shapes[0],
                    layer.constant(),
                    &effective,
                    out_shape,
                )?;
                DagOp::ElementAffine {
                    map,
                    scale: ElementScale::Mul(vec![1.0; n_out]),
                    shift,
                }
            }
            Layer::SubConstant(layer) => {
                require_inputs(node, name, 1)?;
                let constant = layer.constant();
                let (map, values) = constant_maps(
                    name,
                    &input_shapes[0],
                    constant,
                    constant.shape(),
                    out_shape,
                )?;
                let (factor, shift) = if layer.reverse {
                    (-1.0, values)
                } else {
                    (1.0, values.into_iter().map(|c| -c).collect())
                };
                DagOp::ElementAffine {
                    map,
                    scale: ElementScale::Mul(vec![factor; n_out]),
                    shift,
                }
            }
            Layer::MulConstant(layer) => {
                require_inputs(node, name, 1)?;
                let constant = layer.constant();
                let (map, factors) = constant_maps(
                    name,
                    &input_shapes[0],
                    constant,
                    constant.shape(),
                    out_shape,
                )?;
                DagOp::ElementAffine {
                    map,
                    scale: ElementScale::Mul(factors),
                    shift: vec![0.0; n_out],
                }
            }
            Layer::DivConstant(layer) => {
                require_inputs(node, name, 1)?;
                let constant = layer.constant();
                let (map, divisors) = constant_maps(
                    name,
                    &input_shapes[0],
                    constant,
                    constant.shape(),
                    out_shape,
                )?;
                if divisors.iter().any(|d| *d == 0.0 || !d.is_finite()) {
                    return Err(ZonoRefusal::NonFinite("DivConstant divisor"));
                }
                DagOp::ElementAffine {
                    map,
                    scale: ElementScale::Div(divisors),
                    shift: vec![0.0; n_out],
                }
            }
            Layer::Slice(_)
            | Layer::Gather(_)
            | Layer::Transpose(_)
            | Layer::Unsqueeze(_)
            | Layer::Squeeze(_)
            | Layer::Reshape(_)
            | Layer::Flatten(_) => {
                require_inputs(node, name, 1)?;
                let sources = probe_selection(node, name, &input_shapes, out_shape)?;
                let identity = input_shapes[0].iter().product::<usize>() == n_out
                    && sources
                        .iter()
                        .enumerate()
                        .all(|(j, source)| *source == SelectSource::Live { input: 0, elem: j });
                DagOp::Select { sources, identity }
            }
            Layer::Concat(_) => {
                if inputs.is_empty() {
                    return Err(ZonoRefusal::UnsupportedGraph(format!(
                        "concat `{name}` has no graph inputs"
                    )));
                }
                let sources = probe_selection(node, name, &input_shapes, out_shape)?;
                let identity = inputs.len() == 1
                    && input_shapes[0].iter().product::<usize>() == n_out
                    && sources
                        .iter()
                        .enumerate()
                        .all(|(j, source)| *source == SelectSource::Live { input: 0, elem: j });
                DagOp::Select { sources, identity }
            }
            // ---- the vit op set (`#vit-zono-root`); unreachable for Base,
            // whose pre-check refuses these layers before compilation ----
            Layer::Conv2d(layer) if opset == DagOpSet::Vit => {
                require_inputs(node, name, 1)?;
                DagOp::Affine(super::program::conv_op(
                    name,
                    layer,
                    &input_shapes[0],
                    out_shape,
                )?)
            }
            Layer::BatchNorm(layer) if opset == DagOpSet::Vit => {
                elementwise(1)?;
                // ny's OWN shape-aware expansion (the source of truth its
                // BatchNorm IBP/CROWN share), with the certified coefficient
                // radii: never an uncharged fold (plan U10). The binary64
                // enclosure of the exact-real coefficients when the layer
                // carries an intact one (#zono-exact-bn), else the binary32
                // affine and its radii, exactly as before.
                let exact = layer
                    .expanded_exact_affine_parameters(&input_shapes[0], n_out)
                    .map_err(|e| {
                        ZonoRefusal::UnsupportedGraph(format!("node `{name}`: BatchNorm: {e}"))
                    })?;
                let op = if let Some(exact) = exact {
                    ChannelAffine {
                        scale: exact.scale,
                        shift: exact.bias,
                        scale_err: exact.scale_err,
                        shift_err: exact.bias_err,
                    }
                } else {
                    let (scale, shift, scale_err, shift_err) = layer
                        .expanded_affine_parameters(&input_shapes[0], n_out)
                        .map_err(|e| {
                            ZonoRefusal::UnsupportedGraph(format!("node `{name}`: BatchNorm: {e}"))
                        })?;
                    let to_f64 = |v: &ndarray::Array1<f32>| -> Vec<f64> {
                        v.iter().map(|&x| f64::from(x)).collect()
                    };
                    ChannelAffine {
                        scale: to_f64(&scale),
                        shift: to_f64(&shift),
                        scale_err: to_f64(&scale_err),
                        shift_err: to_f64(&shift_err),
                    }
                };
                if [&op.scale, &op.shift, &op.scale_err, &op.shift_err]
                    .iter()
                    .any(|v| v.len() != n_out)
                    || op
                        .scale_err
                        .iter()
                        .chain(&op.shift_err)
                        .any(|e| !e.is_finite() || *e < 0.0)
                    || op.scale.iter().chain(&op.shift).any(|v| !v.is_finite())
                {
                    return Err(ZonoRefusal::Shape(format!(
                        "node `{name}`: BatchNorm expansion is not {n_out} finite entries"
                    )));
                }
                DagOp::ChannelAffine(op)
            }
            Layer::BilinearCrown(layer) if opset == DagOpSet::Vit => DagOp::MatMul(matmul_dims(
                node,
                name,
                &input_shapes,
                out_shape,
                layer.transpose_b(),
                layer.scale(),
            )?),
            Layer::MatMul(layer) if opset == DagOpSet::Vit => DagOp::MatMul(matmul_dims(
                node,
                name,
                &input_shapes,
                out_shape,
                layer.transpose_b(),
                layer.scale(),
            )?),
            Layer::Exp(_) if opset == DagOpSet::Vit => {
                elementwise(1)?;
                DagOp::Exp
            }
            Layer::Reciprocal(_) if opset == DagOpSet::Vit => {
                elementwise(1)?;
                DagOp::Reciprocal
            }
            Layer::Clip(layer) if opset == DagOpSet::Vit => {
                elementwise(1)?;
                let (lo, hi) = (f64::from(layer.min), f64::from(layer.max));
                if lo.is_nan() || hi.is_nan() || lo > hi {
                    return Err(ZonoRefusal::UnsupportedGraph(format!(
                        "node `{name}`: Clip [{lo}, {hi}]"
                    )));
                }
                DagOp::Clip { lo, hi }
            }
            Layer::Softmax(layer) if opset == DagOpSet::Vit => {
                elementwise(1)?;
                let axis = ny_core::resolve_axis_i32(layer.axis, out_shape.len(), "Softmax")
                    .map_err(|e| {
                        ZonoRefusal::UnsupportedGraph(format!("node `{name}`: Softmax axis: {e}"))
                    })?;
                DagOp::Softmax {
                    lanes: softmax_lanes(out_shape, axis)?,
                }
            }
            Layer::ReduceSum(layer) if opset == DagOpSet::Vit => {
                require_inputs(node, name, 1)?;
                let (out_of, n_red, count) =
                    reduce_group_map(&input_shapes[0], &layer.axes, layer.keepdims, out_shape)?;
                verify_reduce_map(node, name, &input_shapes[0], &out_of, n_red, count, false)?;
                DagOp::GroupSum {
                    out_of,
                    n_out: n_red,
                    count,
                    mean: false,
                }
            }
            Layer::ReduceMean(layer) if opset == DagOpSet::Vit => {
                require_inputs(node, name, 1)?;
                let (out_of, n_red, count) =
                    reduce_group_map(&input_shapes[0], &layer.axes, layer.keepdims, out_shape)?;
                verify_reduce_map(node, name, &input_shapes[0], &out_of, n_red, count, true)?;
                DagOp::GroupSum {
                    out_of,
                    n_out: n_red,
                    count,
                    mean: true,
                }
            }
            other => {
                return Err(ZonoRefusal::UnsupportedLayer {
                    node: name.to_string(),
                    layer: other.layer_type(),
                })
            }
        };
        Ok((inputs, op))
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

    /// The output node.
    #[must_use]
    pub fn output(&self) -> &str {
        &self.output
    }

    /// The compiled steps.
    #[must_use]
    pub fn steps(&self) -> &[DagStep] {
        &self.steps
    }

    /// Lookup-table hinge nodes the program does not compute (their only
    /// reader is the table's exact band).
    #[must_use]
    pub fn skipped(&self) -> &[String] {
        &self.skipped
    }

    /// Op census `kind:count,...` in [`DAG_OP_KINDS`] order (zero kinds
    /// omitted), for the observable.
    #[must_use]
    pub fn census(&self) -> String {
        let mut counts: HashMap<&'static str, usize> = HashMap::new();
        for step in &self.steps {
            *counts.entry(step.op.kind()).or_insert(0) += 1;
        }
        counts.insert("skipped", self.skipped.len());
        counts.insert("nodes", self.steps.len());
        DAG_OP_KINDS
            .iter()
            .filter_map(|kind| {
                counts
                    .get(kind)
                    .copied()
                    .filter(|&count| count > 0)
                    .map(|count| format!("{kind}:{count}"))
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Plain binary64 concrete forward of `rows` flat input points (row-major
    /// `rows x input_len`) through the compiled program; `visit` sees the input
    /// (as [`NETWORK_INPUT`]) and every step's output, per point. For
    /// containment SAMPLING in tests — it is not itself a certificate. A
    /// constant interval `[lo, hi]` evaluates at its midpoint.
    pub fn evaluate_points(
        &self,
        points: &[f64],
        rows: usize,
        mut visit: impl FnMut(usize, &str, &[f64]),
    ) -> Result<(), ZonoRefusal> {
        if points.len() != rows * self.input_len {
            return Err(ZonoRefusal::Shape(format!(
                "{} point values for {rows} rows of {}",
                points.len(),
                self.input_len
            )));
        }
        for row in 0..rows {
            let x = &points[row * self.input_len..(row + 1) * self.input_len];
            let mut env: HashMap<&str, Rc<[f64]>> = HashMap::new();
            env.insert(NETWORK_INPUT, Rc::from(x));
            visit(row, NETWORK_INPUT, x);
            for step in &self.steps {
                let args: Vec<Rc<[f64]>> = step
                    .inputs
                    .iter()
                    .map(|name| {
                        env.get(name.as_str()).cloned().ok_or_else(|| {
                            ZonoRefusal::Internal(format!("evaluate: `{name}` not computed"))
                        })
                    })
                    .collect::<Result<_, _>>()?;
                let values = eval_step(step, &args)?;
                visit(row, &step.node, &values);
                env.insert(step.node.as_str(), Rc::from(values));
            }
        }
        Ok(())
    }

    /// The certified forward pass on `input` (an `f32` box of the compiled
    /// input shape). `ibp` holds ny's per-node boxes over the same input
    /// (`compile_with_ibp`'s second result); when `opts.intersect_ibp`, every
    /// node box is intersected with its entry.
    ///
    /// Fails closed ([`ZonoRefusal`]) on a shape mismatch, NaN/Inf, the byte
    /// budget or the deadline; a refused pass publishes nothing.
    pub fn forward(
        &self,
        input: &BoundedTensor,
        opts: &DagForwardOptions,
        ibp: &HashMap<String, BoundedTensor>,
        deadline: Option<Instant>,
        mut observer: Option<&mut dyn FnMut(&DagNodeView<'_>)>,
    ) -> Result<DagForwardOutcome, ZonoRefusal> {
        let started = Instant::now();
        if input.len() != self.input_len {
            return Err(ZonoRefusal::Shape(format!(
                "input box of {} values for a program of {}",
                input.len(),
                self.input_len
            )));
        }
        // The per-node IBP boxes are enclosures over the box they were
        // computed on and over nothing else. An intersecting pass therefore
        // runs only on EXACTLY that box (bitwise), and only with a map whose
        // provenance the program recorded (`compile_with_ibp`).
        if opts.intersect_ibp && self.ibp_input.as_ref() != Some(&input_box_bits(input)) {
            return Err(ZonoRefusal::Internal(
                "intersect_ibp requires the IBP map compile_with_ibp computed over this exact \
                 input box"
                    .to_string(),
            ));
        }
        let raw_lo: Vec<f64> = input.lower().iter().map(|&v| f64::from(v)).collect();
        let raw_hi: Vec<f64> = input.upper().iter().map(|&v| f64::from(v)).collect();
        let mut alloc = SymbolAllocator::new();
        let (state, input_symbols) = ZonoState::from_box(&raw_lo, &raw_hi, &mut alloc)?;
        let mut stats = DagForwardStats {
            input_symbols: input_symbols.len(),
            ..DagForwardStats::default()
        };
        let input_val = {
            let abs_g = state.abs_generator_sum();
            state.ensure_finite(&abs_g)?;
            let (mut lo, mut hi) = concretize(&state, &abs_g);
            // The binary64 image of the f32 input box encloses the input.
            intersect_or_refuse(&mut lo, &mut hi, &raw_lo, &raw_hi, "input box")?;
            NodeVal {
                state,
                abs_g,
                lo,
                hi,
            }
        };
        if let Some(obs) = observer.as_deref_mut() {
            obs(&DagNodeView {
                node: NETWORK_INPUT,
                shape: &self.input_shape,
                state: &input_val.state,
                abs_g: &input_val.abs_g,
                lo: &input_val.lo,
                hi: &input_val.hi,
                input_symbols: &input_symbols,
            });
        }
        stats.gens_peak = input_val.state.num_generators();
        stats.max_remainder = input_val.state.max_remainder();
        let mut env: HashMap<String, Rc<NodeVal>> = HashMap::new();
        env.insert(NETWORK_INPUT.to_string(), Rc::new(input_val));

        for (index, step) in self.steps.iter().enumerate() {
            check_deadline(deadline)?;
            let args: Vec<Rc<NodeVal>> = step
                .inputs
                .iter()
                .map(|name| {
                    env.get(name).cloned().ok_or_else(|| {
                        ZonoRefusal::Internal(format!("forward: `{name}` not computed"))
                    })
                })
                .collect::<Result<_, _>>()?;
            let live = live_bytes(&env);
            let estimate = estimate_step_bytes(step, &args)?;
            if live.saturating_add(estimate) > opts.byte_budget {
                return Err(ZonoRefusal::Memory {
                    estimate_bytes: live.saturating_add(estimate),
                    budget_bytes: opts.byte_budget,
                });
            }
            let ibp_entry = if opts.intersect_ibp {
                ibp.get(&step.node)
            } else {
                None
            };
            let value: Rc<NodeVal> = match &step.op {
                DagOp::Select { identity: true, .. } => Rc::clone(&args[0]),
                DagOp::Select { sources, .. } => {
                    let states: Vec<&ZonoState> = args.iter().map(|a| &a.state).collect();
                    let state = select(&states, sources)?;
                    // Selected boxes are the inputs' boxes, element for element.
                    let (lo, hi): (Vec<f64>, Vec<f64>) = sources
                        .iter()
                        .map(|source| match *source {
                            SelectSource::Live { input, elem } => {
                                (args[input].lo[elem], args[input].hi[elem])
                            }
                            SelectSource::Const { lo, hi } => (lo, hi),
                        })
                        .unzip();
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        Some((lo.as_slice(), hi.as_slice())),
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                DagOp::Linear {
                    kernel,
                    bias,
                    batch,
                } => {
                    let state = apply_linear_batched(
                        &args[0].state,
                        &args[0].abs_g,
                        kernel,
                        bias,
                        *batch,
                        deadline,
                    )?;
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        None,
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                DagOp::ElementAffine { map, scale, shift } => {
                    let x = gather(&args[0].state, map)?;
                    let state = element_affine(&x, scale, shift)?;
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        None,
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                DagOp::AddSub {
                    map_a,
                    map_b,
                    subtract,
                } => {
                    let a = gather(&args[0].state, map_a)?;
                    let b = gather(&args[1].state, map_b)?;
                    let state = add_aligned(&a, &b, *subtract)?;
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        None,
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                DagOp::Product { map_a, map_b } => {
                    let a = gather(&args[0].state, map_a)?;
                    let b = gather(&args[1].state, map_b)?;
                    let (state, fresh) = product(&a, &b, &mut alloc)?;
                    stats.fresh_mul += fresh;
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        None,
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                DagOp::Square => {
                    let (state, fresh) = square(&args[0].state, &mut alloc)?;
                    stats.fresh_square += fresh;
                    // x^2 over the input box: [0 or min^2, max^2], outward.
                    // The lower end is `next_down(fl(m * m))`: round-to-
                    // nearest is within half an ulp, so one ulp down is below
                    // the exact square in EVERY range. (A relative `(1 - 4u)`
                    // factor is not: in the subnormal range the rounding
                    // error is absolute, e.g. `l*l = 3e-324` rounds UP to
                    // `4.9e-324` and no relative factor brings it back.) An
                    // overflowed square keeps the trivially valid `0`.
                    let square_floor = |m: f64| {
                        let sq = m * m;
                        if sq.is_finite() {
                            next_down(sq).max(0.0)
                        } else {
                            0.0
                        }
                    };
                    let (lo, hi): (Vec<f64>, Vec<f64>) = args[0]
                        .lo
                        .iter()
                        .zip(&args[0].hi)
                        .map(|(&l, &u)| {
                            let big = l.abs().max(u.abs());
                            let top = next_up(big * big);
                            let bottom = if l > 0.0 {
                                square_floor(l)
                            } else if u < 0.0 {
                                square_floor(u)
                            } else {
                                0.0
                            };
                            (bottom, top)
                        })
                        .unzip();
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        Some((lo.as_slice(), hi.as_slice())),
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                DagOp::Relu => {
                    let arg = &args[0];
                    let (state, report) =
                        apply_relu(&arg.state, &arg.abs_g, &arg.lo, &arg.hi, &mut alloc)?;
                    stats.fresh_relu += report.unstable;
                    stats.relu_unstable += report.unstable;
                    let lo: Vec<f64> = arg.lo.iter().map(|&l| l.max(0.0)).collect();
                    let hi: Vec<f64> = arg.hi.iter().map(|&u| u.max(0.0)).collect();
                    Rc::new(finish(
                        super::elementwise::drop_zero_rows(state),
                        ibp_entry,
                        Some((lo.as_slice(), hi.as_slice())),
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                DagOp::Smooth(kind) => {
                    let arg = &args[0];
                    let band = smooth_band(*kind, &arg.lo, &arg.hi)?;
                    let (state, fresh) = apply_band(&arg.state, &band, &mut alloc)?;
                    stats.fresh_band += fresh;
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        None,
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                DagOp::Floor => {
                    let arg = &args[0];
                    let band = floor_band(&arg.lo, &arg.hi)?;
                    let (state, fresh) = apply_band(&arg.state, &band, &mut alloc)?;
                    stats.fresh_band += fresh;
                    let lo: Vec<f64> = arg.lo.iter().map(|l| l.floor()).collect();
                    let hi: Vec<f64> = arg.hi.iter().map(|u| u.floor()).collect();
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        Some((lo.as_slice(), hi.as_slice())),
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                DagOp::Pwl {
                    thresholds,
                    weights,
                    bias,
                } => {
                    let arg = &args[0];
                    let band = pwl_band(&arg.lo, &arg.hi, thresholds, weights, *bias)?;
                    let (state, fresh) = apply_band(&arg.state, &band, &mut alloc)?;
                    stats.fresh_band += fresh;
                    let extra = if opts.intersect_ibp {
                        Some(pwl_exact_range(
                            &arg.lo, &arg.hi, thresholds, weights, *bias,
                        ))
                    } else {
                        None
                    };
                    let extra_ref = extra.as_ref().map(|(l, h)| (l.as_slice(), h.as_slice()));
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        extra_ref,
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                // ---- vit op set (`#vit-zono-root`) ----
                DagOp::Affine(op) => {
                    let state = apply_affine(&args[0].state, &args[0].abs_g, op, deadline)?;
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        None,
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                DagOp::ChannelAffine(op) => {
                    let state = apply_channel_affine(&args[0].state, &args[0].abs_g, op)?;
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        None,
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                DagOp::GroupSum {
                    out_of,
                    n_out,
                    count,
                    mean,
                } => {
                    let summed = group_sum(&args[0].state, &args[0].abs_g, out_of, *n_out, *count)?;
                    let state = if *mean {
                        #[allow(clippy::cast_precision_loss)]
                        let divisor = *count as f64;
                        element_affine(
                            &summed,
                            &ElementScale::Div(vec![divisor; *n_out]),
                            &vec![0.0; *n_out],
                        )?
                    } else {
                        summed
                    };
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        None,
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                DagOp::MatMul(dims) => {
                    let (state, fresh) = bilinear_matmul(
                        &args[0].state,
                        &args[1].state,
                        dims,
                        &mut alloc,
                        deadline,
                    )?;
                    stats.fresh_bilinear += fresh;
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        None,
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                DagOp::Exp => {
                    let arg = &args[0];
                    let band = exp_band(&arg.lo, &arg.hi)?;
                    let (state, fresh) = apply_band(&arg.state, &band, &mut alloc)?;
                    stats.fresh_band += fresh;
                    let (lo, hi) = exp_range(&arg.lo, &arg.hi);
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        Some((lo.as_slice(), hi.as_slice())),
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                DagOp::Reciprocal => {
                    let arg = &args[0];
                    let band = reciprocal_band(&arg.lo, &arg.hi)?;
                    let (state, fresh) = apply_band(&arg.state, &band, &mut alloc)?;
                    stats.fresh_band += fresh;
                    let (lo, hi) = reciprocal_range(&arg.lo, &arg.hi);
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        Some((lo.as_slice(), hi.as_slice())),
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                DagOp::Clip { lo, hi } => {
                    let arg = &args[0];
                    let band = clip_band(&arg.lo, &arg.hi, *lo, *hi)?;
                    let (state, fresh) = apply_band(&arg.state, &band, &mut alloc)?;
                    stats.fresh_band += fresh;
                    let (range_lo, range_hi) = clip_range(&arg.lo, &arg.hi, *lo, *hi);
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        Some((range_lo.as_slice(), range_hi.as_slice())),
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
                DagOp::Softmax { lanes } => {
                    let arg = &args[0];
                    let (lo, hi) =
                        softmax_box(&arg.state, &arg.abs_g, &arg.lo, &arg.hi, lanes, deadline)?;
                    let (state, fresh) = box_state(&lo, &hi, &mut alloc)?;
                    stats.fresh_softmax += fresh;
                    Rc::new(finish(
                        state,
                        ibp_entry,
                        Some((lo.as_slice(), hi.as_slice())),
                        opts.intersect_ibp,
                        &mut stats,
                    )?)
                }
            };
            if value.state.len() != numel(&step.out_shape)? {
                return Err(ZonoRefusal::Shape(format!(
                    "node `{}`: {} values for shape {:?}",
                    step.node,
                    value.state.len(),
                    step.out_shape
                )));
            }
            drop(args);
            for input in &step.inputs {
                if self.last_use.get(input) == Some(&index) && *input != self.output {
                    env.remove(input);
                }
            }
            stats.gens_peak = stats.gens_peak.max(value.state.num_generators());
            stats.max_remainder = stats.max_remainder.max(value.state.max_remainder());
            if let Some(obs) = observer.as_deref_mut() {
                obs(&DagNodeView {
                    node: &step.node,
                    shape: &step.out_shape,
                    state: &value.state,
                    abs_g: &value.abs_g,
                    lo: &value.lo,
                    hi: &value.hi,
                    input_symbols: &input_symbols,
                });
            }
            // A node nobody reads later (and not the output) is dropped now.
            if self.last_use.contains_key(&step.node) || step.node == self.output {
                env.insert(step.node.clone(), value);
            }
            stats.live_bytes_peak = stats.live_bytes_peak.max(live_bytes(&env));
        }
        let out = env
            .get(&self.output)
            .ok_or_else(|| ZonoRefusal::Internal("output not computed".to_string()))?;
        stats.symbols = alloc.allocated();
        stats.elapsed = started.elapsed();
        let output_shape = self
            .steps
            .iter()
            .find(|step| step.node == self.output)
            .map(|step| step.out_shape.clone())
            .unwrap_or_default();
        Ok(DagForwardOutcome {
            output_lo: out.lo.clone(),
            output_hi: out.hi.clone(),
            output_shape,
            input_symbols,
            stats,
        })
    }
}

/// Finish a node: certified absolute sums, finiteness, and the box
/// `concretise ∩ extra ∩ IBP` (each an enclosure of the node's true values).
/// With `intersect == false` (the replica-parity pass) the box is the pure
/// concretisation: no IBP entry and no op-derived extra enclosure.
fn finish(
    state: ZonoState,
    ibp: Option<&BoundedTensor>,
    extra: Option<(&[f64], &[f64])>,
    intersect: bool,
    stats: &mut DagForwardStats,
) -> Result<NodeVal, ZonoRefusal> {
    let abs_g = state.abs_generator_sum();
    state.ensure_finite(&abs_g)?;
    let (mut lo, mut hi) = concretize(&state, &abs_g);
    if let Some((el, eh)) = extra.filter(|_| intersect) {
        if el.len() == lo.len() && eh.len() == hi.len() {
            stats.ibp_tightened += intersect_or_refuse(&mut lo, &mut hi, el, eh, "op range")?;
        }
    }
    if let Some(entry) = ibp.filter(|_| intersect) {
        if entry.len() == lo.len() {
            let il: Vec<f64> = entry
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
            let ih: Vec<f64> = entry
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
            stats.ibp_tightened += intersect_or_refuse(&mut lo, &mut hi, &il, &ih, "IBP box")?;
        }
    }
    Ok(NodeVal {
        state,
        abs_g,
        lo,
        hi,
    })
}

/// [`intersect_keep_looser`], but a CROSSED element REFUSES the pass.
///
/// Both boxes enclose the node's true values over the same non-empty input
/// box, so in a correct program their intersection is never empty. A crossing
/// therefore proves that one of the two enclosures is wrong, and nothing says
/// it is the other one: keeping the zonotope's side (what
/// `intersect_keep_looser` does, correctly, for the shrink-only map merges it
/// was written for) would let a defective transformer go on to refute
/// clauses. The screen publishes refutations, so it fails closed instead —
/// the same stance as C1's `contradiction` (a stop-and-report event). Costs
/// nothing when both enclosures are sound. NaN endpoints are ignored, exactly
/// as in `intersect_keep_looser`.
fn intersect_or_refuse(
    lo: &mut [f64],
    hi: &mut [f64],
    other_lo: &[f64],
    other_hi: &[f64],
    what: &'static str,
) -> Result<usize, ZonoRefusal> {
    let count = lo.len().min(other_lo.len()).min(other_hi.len());
    for j in 0..count {
        let (ol, oh) = (other_lo[j], other_hi[j]);
        if ol.is_nan() || oh.is_nan() {
            continue;
        }
        if lo[j].max(ol) > hi[j].min(oh) {
            return Err(ZonoRefusal::Internal(format!(
                "{what} contradicts the zonotope box at element {j}: [{:e}, {:e}] vs [{ol:e}, {oh:e}]",
                lo[j], hi[j]
            )));
        }
    }
    Ok(intersect_keep_looser(lo, hi, other_lo, other_hi))
}

/// Bytes held by every distinct live node value.
fn live_bytes(env: &HashMap<String, Rc<NodeVal>>) -> u64 {
    let mut seen: HashSet<*const NodeVal> = HashSet::new();
    let mut total = 0u64;
    for value in env.values() {
        if seen.insert(Rc::as_ptr(value)) {
            total = total.saturating_add(value.bytes());
        }
    }
    total
}

/// Conservative transient bytes of one step (policy only, never soundness):
/// the output's dense rows plus the gathered operand copies.
fn estimate_step_bytes(step: &DagStep, args: &[Rc<NodeVal>]) -> Result<u64, ZonoRefusal> {
    let n_out = numel(&step.out_shape)? as u64;
    let k_in: u64 = args.iter().map(|a| a.state.num_generators() as u64).sum();
    let (k_out, copies) = match step.op {
        DagOp::Select { identity: true, .. } => (0, 0),
        // `select` builds the dense union matrix and `drop_zero_rows` may
        // then copy the kept rows while it is still alive: two copies.
        DagOp::Select { .. } => (k_in, 2),
        DagOp::Linear { .. } | DagOp::Affine(_) => (k_in, 1),
        DagOp::ElementAffine { .. } | DagOp::ChannelAffine(_) => (k_in, 2),
        // The sum, then (ReduceMean) the element-wise division.
        DagOp::GroupSum { .. } => (k_in, 3),
        DagOp::AddSub { .. } => (k_in, 3),
        DagOp::Product { .. } | DagOp::MatMul(_) => (k_in + n_out, 3),
        DagOp::Square
        | DagOp::Relu
        | DagOp::Smooth(_)
        | DagOp::Floor
        | DagOp::Pwl { .. }
        | DagOp::Exp
        | DagOp::Reciprocal
        | DagOp::Clip { .. } => (k_in + n_out, 2),
        // The output is a box with one fresh symbol per element.
        DagOp::Softmax { .. } => (n_out, 2),
    };
    let scratch = match &step.op {
        DagOp::Linear { .. } | DagOp::Affine(_) => 2 * CHUNK_ELEMS as u64,
        // Two 1 Mi-element GEMM buffers per side.
        DagOp::MatMul(_) => 4 << 20,
        // One gathered lane (K x lane) plus two lane x lane tables per worker
        // thread.
        DagOp::Softmax { lanes } => {
            let widest = lanes.iter().map(Vec::len).max().unwrap_or(0) as u64;
            (rayon::current_num_threads() as u64).saturating_mul(
                k_in.saturating_mul(widest)
                    .saturating_add(2u64.saturating_mul(widest).saturating_mul(widest)),
            )
        }
        _ => 0,
    };
    Ok(8u64
        .saturating_mul(copies)
        .saturating_mul(k_out)
        .saturating_mul(n_out)
        .saturating_add(8 * 8 * n_out)
        .saturating_add(scratch.saturating_mul(8)))
}

/// `y[p, o] = sum_i W[o, i] x[p, i] + b[o]` for `batch` leading positions.
///
/// SOUND (the V2 affine charge per position). For `x = c + G e + rho`,
/// `W x + b = (W c + b) + (W G) e + W rho`. The computed centre and rows are
/// within `gamma_{f+1}(|W||c| + |b|)` and `gamma_f |W||G_k|` (`f` = fan-in,
/// any summation order), and `|W rho| <= |W| r`; with `m >= |c| + sum|G| + r`
/// the remainder `(|W| r + gamma_{f+2}(|W| m + |b|))(1 + gamma_{f+5})`,
/// rounded up, dominates all of it. Every generator row is `batch` rows of
/// `in` values, so the dense kernel maps all of them in one call.
fn apply_linear_batched(
    state: &ZonoState,
    abs_g: &[f64],
    kernel: &LinearKernel,
    bias: &[f64],
    batch: usize,
    deadline: Option<Instant>,
) -> Result<ZonoState, ZonoRefusal> {
    let (n_in, n_out) = (kernel.n_in(), kernel.n_out());
    if state.len() != batch * n_in || abs_g.len() != batch * n_in || bias.len() != n_out {
        return Err(ZonoRefusal::Shape(format!(
            "linear of {n_in} -> {n_out} x {batch} on {} values",
            state.len()
        )));
    }
    let total_out = batch * n_out;
    let k = state.num_generators();
    let m = state.magnitude(abs_g);
    let mut center = vec![0.0f64; total_out];
    kernel.forward_rows(state.center(), batch, false, &mut center, deadline)?;
    for (q, c) in center.iter_mut().enumerate() {
        *c += bias[q % n_out];
    }
    let mut am = vec![0.0f64; total_out];
    kernel.forward_rows(&m, batch, true, &mut am, deadline)?;
    let mut ar = vec![0.0f64; total_out];
    kernel.forward_rows(state.remainder(), batch, true, &mut ar, deadline)?;
    let mut gens = vec![0.0f64; k * total_out];
    kernel.forward_rows(state.generators(), k * batch, false, &mut gens, deadline)?;
    let fan = kernel.fan_in();
    let g_round = gamma(fan + 2);
    let g_eval = 1.0 + gamma(fan + 5);
    let rem: Vec<f64> = (0..total_out)
        .map(|q| next_up((ar[q] + g_round * (am[q] + bias[q % n_out].abs())) * g_eval + TINY))
        .collect();
    Ok(ZonoState::from_parts_trusted(
        total_out,
        center,
        gens,
        state.ids().to_vec(),
        rem,
    ))
}

/// Plain binary64 evaluation of one step (the sampling oracle).
fn eval_step(step: &DagStep, args: &[Rc<[f64]>]) -> Result<Vec<f64>, ZonoRefusal> {
    let unary = |f: &dyn Fn(f64) -> f64| -> Vec<f64> { args[0].iter().map(|&v| f(v)).collect() };
    Ok(match &step.op {
        DagOp::Select { sources, .. } => sources
            .iter()
            .map(|source| match *source {
                SelectSource::Live { input, elem } => args[input][elem],
                SelectSource::Const { lo, hi } => {
                    if lo == hi {
                        lo
                    } else {
                        0.5 * lo + 0.5 * hi
                    }
                }
            })
            .collect(),
        DagOp::Linear {
            kernel,
            bias,
            batch,
        } => {
            let LinearKernel::Dense { n_in, n_out, w, .. } = kernel else {
                return Err(ZonoRefusal::Internal("non-dense linear step".to_string()));
            };
            let x = &args[0];
            let mut out = vec![0.0f64; batch * n_out];
            for p in 0..*batch {
                for o in 0..*n_out {
                    let mut acc = bias[o];
                    for i in 0..*n_in {
                        acc += w[o * n_in + i] * x[p * n_in + i];
                    }
                    out[p * n_out + o] = acc;
                }
            }
            out
        }
        DagOp::ElementAffine { map, scale, shift } => match scale {
            ElementScale::Mul(a) => map
                .iter()
                .enumerate()
                .map(|(j, &i)| args[0][i] * a[j] + shift[j])
                .collect(),
            ElementScale::Div(d) => map
                .iter()
                .enumerate()
                .map(|(j, &i)| args[0][i] / d[j] + shift[j])
                .collect(),
        },
        DagOp::AddSub {
            map_a,
            map_b,
            subtract,
        } => map_a
            .iter()
            .zip(map_b)
            .map(|(&i, &k)| {
                if *subtract {
                    args[0][i] - args[1][k]
                } else {
                    args[0][i] + args[1][k]
                }
            })
            .collect(),
        DagOp::Product { map_a, map_b } => map_a
            .iter()
            .zip(map_b)
            .map(|(&i, &k)| args[0][i] * args[1][k])
            .collect(),
        DagOp::Square => unary(&|v| v * v),
        DagOp::Relu => unary(&|v| v.max(0.0)),
        DagOp::Smooth(kind) => unary(&|v| kind.eval(v)),
        DagOp::Floor => unary(&f64::floor),
        DagOp::Pwl {
            thresholds,
            weights,
            bias,
        } => unary(&|t| {
            f64::from(*bias)
                + thresholds
                    .iter()
                    .zip(weights)
                    .map(|(&b, &w)| f64::from(w) * (t - f64::from(b)).max(0.0))
                    .sum::<f64>()
        }),
        DagOp::Affine(op) => {
            let mut out = vec![0.0f64; op.kernel.n_out()];
            op.kernel.forward_rows(&args[0], 1, false, &mut out, None)?;
            for (value, bias) in out.iter_mut().zip(&op.bias) {
                *value += *bias;
            }
            out
        }
        DagOp::ChannelAffine(op) => args[0]
            .iter()
            .enumerate()
            .map(|(j, &v)| v * op.scale[j] + op.shift[j])
            .collect(),
        DagOp::GroupSum {
            out_of,
            n_out,
            count,
            mean,
        } => {
            let mut out = vec![0.0f64; *n_out];
            for (i, &j) in out_of.iter().enumerate() {
                out[j] += args[0][i];
            }
            if *mean {
                #[allow(clippy::cast_precision_loss)]
                let divisor = *count as f64;
                out.iter_mut().for_each(|v| *v /= divisor);
            }
            out
        }
        DagOp::MatMul(dims) => matmul_eval(dims, &args[0], &args[1]),
        DagOp::Exp => unary(&f64::exp),
        DagOp::Reciprocal => unary(&|v| 1.0 / v),
        DagOp::Clip { lo, hi } => unary(&|v| clip_value(v, *lo, *hi)),
        DagOp::Softmax { lanes } => softmax_eval(lanes, &args[0]),
    })
}

/// Unstable count of a node box (diagnostics).
#[must_use]
pub fn unstable_in_box(lo: &[f64], hi: &[f64]) -> usize {
    count_unstable(lo, hi)
}

#[cfg(test)]
mod intersect_tests {
    use super::intersect_or_refuse;
    use crate::zono_core::ZonoRefusal;

    /// A crossed box intersection proves one enclosure wrong, so the screen
    /// REFUSES instead of keeping either side; an overlapping one intersects
    /// shrink-only, and a NaN endpoint is ignored.
    #[test]
    fn fwd_root_screen_crossed_intersection_refuses() {
        let (mut lo, mut hi) = (vec![0.0, 1.0], vec![1.0, 2.0]);
        let tightened = intersect_or_refuse(&mut lo, &mut hi, &[0.5, f64::NAN], &[2.0, 0.0], "t")
            .expect("overlapping boxes intersect");
        assert_eq!(tightened, 1);
        assert_eq!((lo[0], hi[0]), (0.5, 1.0));
        assert_eq!((lo[1], hi[1]), (1.0, 2.0), "a NaN endpoint is ignored");

        let (mut lo, mut hi) = (vec![0.0], vec![1.0]);
        match intersect_or_refuse(&mut lo, &mut hi, &[1.5], &[2.0], "t") {
            Err(ZonoRefusal::Internal(why)) => assert!(why.contains("contradicts"), "{why}"),
            other => panic!("a crossed intersection must refuse, got {other:?}"),
        }
        assert_eq!(
            (lo[0], hi[0]),
            (0.0, 1.0),
            "a refusal leaves the box untouched"
        );
    }
}
