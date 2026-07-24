// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Network preprocessing utilities for HiGHS MIP verification.
// Extracts parameters, folds constants, strips shape layers.
//
// Part of #1763.

use anyhow::Result;
use ndarray::{Array1, Array2, ArrayD};
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
use ny_core::Bound;
use ny_propagate::layers::LinearLayer;
use ny_propagate::{Layer, Network};
use ny_tensor::BoundedTensor;
use std::ops::Deref;

/// Extracted network parameters: (weights, biases, layer_dims).
pub(super) type NetworkParams = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>);

/// Validate the exact topology encoded by [`ny_mip::encode_feedforward`].
///
/// The encoder receives only affine parameters, not the original activation
/// sequence, and therefore inserts one ReLU after every non-final Linear.  A
/// membership-only check ("every layer is Linear or ReLU") is insufficient:
/// adjacent Linear layers, a leading/trailing ReLU, or consecutive ReLUs would
/// be encoded as a different network.  Refuse every such sequence before an
/// exact solver can contribute a certified verdict.
pub(super) fn validate_mip_feedforward_topology(network: &Network) -> Result<()> {
    let layers = network.layers();
    if layers.is_empty() {
        anyhow::bail!("MIP feedforward topology is empty; expected [Linear, ReLU]*, Linear");
    }

    for (index, layer) in layers.iter().enumerate() {
        let expected_linear = index % 2 == 0;
        let matches_encoder = if expected_linear {
            matches!(layer, Layer::Linear(_))
        } else {
            matches!(layer, Layer::ReLU(_))
        };
        if !matches_encoder {
            let expected = if expected_linear { "Linear" } else { "ReLU" };
            anyhow::bail!(
                "MIP feedforward topology mismatch at layer {index}: encoder expects {expected}, found {} (required topology: [Linear, ReLU]*, Linear)",
                layer.layer_type(),
            );
        }
    }

    if layers.len().is_multiple_of(2) {
        let index = layers.len() - 1;
        anyhow::bail!(
            "MIP feedforward topology mismatch at layer {index}: trailing ReLU has no following Linear (required topology: [Linear, ReLU]*, Linear)"
        );
    }

    Ok(())
}

/// Extract weights, biases, and layer dimensions from the exact
/// `[Linear, ReLU]*, Linear` topology consumed by the MIP encoder.
pub(super) fn extract_linear_relu_params(network: &Network) -> Result<NetworkParams> {
    // Defense in depth: callers cannot silently erase an unsupported activation
    // sequence merely by invoking the parameter extractor directly.
    validate_mip_feedforward_topology(network)?;

    let mut weights: Vec<Vec<f64>> = Vec::new();
    let mut biases: Vec<Vec<f64>> = Vec::new();
    let mut layer_dims: Vec<usize> = Vec::new();
    let mut first_layer = true;

    for layer in network.layers() {
        if let Layer::Linear(linear) = layer {
            let (out_dim, in_dim) = linear.weight.dim();
            if first_layer {
                layer_dims.push(in_dim);
                first_layer = false;
            }
            layer_dims.push(out_dim);
            let weight_vec: Vec<f64> = linear.weight.iter().map(|&x| x as f64).collect();
            weights.push(weight_vec);
            let bias_vec: Vec<f64> = match &linear.bias {
                Some(b) => b.iter().map(|&x| x as f64).collect(),
                None => vec![0.0; out_dim],
            };
            biases.push(bias_vec);
        }
    }

    Ok((weights, biases, layer_dims))
}

/// Convert BoundedTensor to Vec<Bound>.
pub(super) fn bounded_tensor_to_bounds(tensor: &BoundedTensor) -> Result<Vec<Bound>> {
    tensor
        .lower()
        .iter()
        .zip(tensor.upper().iter())
        .map(|(&l, &u)| Bound::try_new(l, u).map_err(|e| anyhow::anyhow!("Invalid bound: {}", e)))
        .collect()
}

/// Convert IBP intermediate bounds for MIP encoding.
///
/// Returns pre-activation bounds for each hidden layer's ReLU
/// (used as tight Big-M values in the encoding).
///
/// Looks past constant-op layers (AddConstant, SubConstant) between
/// Linear and ReLU, since `fold_constant_layers` merges these into
/// the Linear's bias. Uses the output of the layer just before the
/// ReLU as the pre-activation bounds (which includes any bias additions).
pub(super) fn convert_intermediate_bounds(
    layer_outputs: &[BoundedTensor],
    network: &Network,
) -> Result<Vec<Vec<Bound>>> {
    let mut intermediate_bounds = Vec::new();
    let layers = network.layers();

    for (i, layer) in layers.iter().enumerate() {
        // Match both Linear and Conv2d as affine layers preceding ReLU.
        // Conv2d→ReLU pairs produce pre-activation bounds that the MIP encoder
        // needs for Big-M encoding, just like Linear→ReLU pairs. (#3218)
        if matches!(layer, Layer::Linear(_) | Layer::Conv2d(_)) {
            // Find the next ReLU, skipping constant-op layers that get folded
            // into the Linear's bias by fold_constant_layers.
            let mut pre_relu_idx = None;
            for (j, lj) in layers.iter().enumerate().skip(i + 1) {
                match lj {
                    Layer::ReLU(_) => {
                        // The layer just before ReLU has the pre-activation bounds
                        pre_relu_idx = Some(j - 1);
                        break;
                    }
                    Layer::AddConstant(_) | Layer::SubConstant(_) => continue,
                    _ => break,
                }
            }

            let is_last_affine = layers[i + 1..]
                .iter()
                .all(|l| !matches!(l, Layer::Linear(_) | Layer::Conv2d(_)));

            if let Some(bounds_idx) = pre_relu_idx {
                if !is_last_affine && bounds_idx < layer_outputs.len() {
                    let bounds = bounded_tensor_to_bounds(&layer_outputs[bounds_idx])?;
                    intermediate_bounds.push(bounds);
                }
            }
        }
    }
    Ok(intermediate_bounds)
}

/// Strip shape-only layers (Flatten, Reshape) that don't affect computation.
///
/// ONNX models commonly start with a Flatten layer to convert 2D image tensors
/// into 1D vectors. For MIP encoding, input is already flat (Vec<Bound>), so
/// these layers are no-ops that we skip.
pub(super) fn strip_shape_layers(network: &Network) -> Network {
    let mut stripped = Network::new();
    for layer in network.layers() {
        match layer {
            Layer::Flatten(_) | Layer::Reshape(_) => {}
            _ => stripped.add_layer(layer.clone()),
        }
    }
    stripped
}

/// Fold constant-operation layers into adjacent Linear layers.
///
/// Many ONNX models use separate MatMul + BiasAdd nodes instead of a fused
/// Gemm node. This produces Linear → AddConstant → ReLU instead of
/// Linear(with_bias) → ReLU. This function folds AddConstant and SubConstant
/// layers into the preceding or following Linear layer's bias.
///
/// The reduction is deliberately chain-aware.  A single left-to-right pair
/// pass is insufficient for the ACAS-Xu normalization prelude
///
/// ```text
/// SubConstant → Linear → AddConstant
/// ```
///
/// because folding the first pair creates a new `Linear → AddConstant`
/// adjacency behind the scan cursor.  The stack reduction below retries the
/// new top pair after every fold, reaching the same fixed point without an
/// unbounded whole-network retry loop.
///
/// # Certified-UNSAT soundness
///
/// A naïve `f64` dot followed by an `f32` folded bias is only approximately
/// equivalent to `W (x + c) + b`; certifying the rounded network UNSAT would
/// therefore not certify the ONNX network.  Every bias expression here is
/// accumulated as an exact dyadic [`BigRational`].  The exact value is carried
/// separately to the MIP parameter extractor and the lane fails closed unless
/// that value is exactly representable by the encoder's `f64` coefficient
/// type.  The `Network` stored below contains a rounded `f32` structural bias
/// solely because [`LinearLayer`] requires one; MIP solving must use
/// [`FoldedMipNetwork::exact_biases`], never that structural copy.
pub(super) fn fold_constant_layers(network: &Network) -> Result<FoldedMipNetwork> {
    let mut stack: Vec<FoldEntry> = Vec::with_capacity(network.layers().len());

    for layer in network.layers() {
        stack.push(FoldEntry::from_layer(layer)?);
        loop {
            let Some(right) = stack.pop() else {
                break;
            };
            let Some(left) = stack.pop() else {
                stack.push(right);
                break;
            };

            if let Some(replacement) = fold_constant_linear_pair(&left, &right)? {
                stack.push(replacement);
                // The replacement may itself be adjacent to another foldable
                // constant op, so reduce the new stack top before consuming the
                // next input layer.
                continue;
            }

            stack.push(left);
            stack.push(right);
            break;
        }
    }

    let mut folded = Network::new();
    let mut exact_biases = Vec::new();
    for entry in stack {
        match entry {
            FoldEntry::Linear(linear) => {
                let exact_bias = linear
                    .bias
                    .iter()
                    .enumerate()
                    .map(|(row, value)| exact_rational_to_f64(value, row))
                    .collect::<Result<Vec<_>>>()?;
                // Structural only. The exact f64 sidecar above is what the MIP
                // encoder consumes; see the soundness contract on this function.
                let structural_bias = exact_bias
                    .iter()
                    .map(|&value| checked_f32(value, "structural folded Linear"))
                    .collect::<Result<Vec<_>>>()?;
                folded.add_layer(Layer::Linear(LinearLayer::new(
                    linear.layer.weight.clone(),
                    Some(Array1::from_vec(structural_bias)),
                )?));
                exact_biases.push(exact_bias);
            }
            FoldEntry::Other(layer) => folded.add_layer(layer),
        }
    }
    Ok(FoldedMipNetwork {
        network: folded,
        exact_biases,
    })
}

/// A structurally folded Linear+ReLU candidate plus the exact f64 biases which
/// preserve the original constant-op algebra for certified MIP solving.
#[derive(Debug)]
pub(super) struct FoldedMipNetwork {
    network: Network,
    exact_biases: Vec<Vec<f64>>,
}

impl FoldedMipNetwork {
    pub(super) fn exact_biases(&self) -> &[Vec<f64>] {
        &self.exact_biases
    }
}

impl Deref for FoldedMipNetwork {
    type Target = Network;

    fn deref(&self) -> &Self::Target {
        &self.network
    }
}

#[derive(Clone)]
struct ExactLinear {
    layer: LinearLayer,
    bias: Vec<BigRational>,
}

enum FoldEntry {
    Linear(ExactLinear),
    Other(Layer),
}

impl FoldEntry {
    fn from_layer(layer: &Layer) -> Result<Self> {
        let Layer::Linear(linear) = layer else {
            return Ok(Self::Other(layer.clone()));
        };
        if linear.weight.iter().any(|value| !value.is_finite()) {
            anyhow::bail!("constant folding: Linear has a non-finite weight");
        }
        let out_dim = linear.weight.nrows();
        let bias = match &linear.bias {
            Some(bias) => bias
                .iter()
                .enumerate()
                .map(|(row, &value)| rat_f32(value, &format!("Linear bias row {row}")))
                .collect::<Result<Vec<_>>>()?,
            None => vec![BigRational::zero(); out_dim],
        };
        Ok(Self::Linear(ExactLinear {
            layer: linear.clone(),
            bias,
        }))
    }
}

/// Fold exactly one adjacent constant/linear pair.
///
/// For a feature-wise constant `c`:
///
/// * `(W x + b) + c = W x + (b + c)`
/// * `(W x + b) - c = W x + (b - c)`
/// * `W (x + c) + b = W x + (b + W c)`
/// * `W (x - c) + b = W x + (b - W c)`
///
/// Reverse subtraction (`c - x`) is intentionally not folded.  Leaving it in
/// the network makes the downstream Linear+ReLU validator reject the MIP lane,
/// which is the sound fail-closed outcome until weight negation is implemented.
fn fold_constant_linear_pair(left: &FoldEntry, right: &FoldEntry) -> Result<Option<FoldEntry>> {
    let folded = match (left, right) {
        (FoldEntry::Linear(linear), FoldEntry::Other(Layer::AddConstant(add))) => {
            Some(fold_constant_after_linear(linear, add.constant(), false)?)
        }
        (FoldEntry::Linear(linear), FoldEntry::Other(Layer::SubConstant(sub))) if !sub.reverse => {
            Some(fold_constant_after_linear(linear, sub.constant(), true)?)
        }
        (FoldEntry::Other(Layer::AddConstant(add)), FoldEntry::Linear(linear)) => {
            Some(fold_constant_before_linear(linear, add.constant(), false)?)
        }
        (FoldEntry::Other(Layer::SubConstant(sub)), FoldEntry::Linear(linear)) if !sub.reverse => {
            Some(fold_constant_before_linear(linear, sub.constant(), true)?)
        }
        _ => None,
    };
    Ok(folded)
}

/// Resolve a constant tensor to the per-feature vector represented by the flat
/// MIP network.
///
/// Scalar constants broadcast to every feature.  Non-scalars are accepted only
/// when their shape is `[1, ..., 1, feature_dim]`; that is exactly the ONNX
/// broadcast form which adds the same feature vector to every leading batch
/// position produced/consumed by a Linear layer.  Merely having the same number
/// of elements is not sufficient: e.g. `[feature_dim, 1]` broadcasts along a
/// different axis and must fail closed because this preprocessing stage no
/// longer has the full intermediate shape.
fn constant_feature_vector(
    constant: &ArrayD<f32>,
    feature_dim: usize,
    context: &str,
) -> Result<Vec<f32>> {
    if feature_dim == 0 {
        anyhow::bail!("{context}: zero-sized Linear feature dimension");
    }
    if constant.is_empty() {
        anyhow::bail!("{context}: empty constant tensor");
    }
    if constant.len() == 1 {
        let value = constant
            .iter()
            .next()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("{context}: empty scalar constant"))?;
        return Ok(vec![value; feature_dim]);
    }

    let shape = constant.shape();
    let is_feature_vector = constant.len() == feature_dim
        && shape.last().copied() == Some(feature_dim)
        && shape[..shape.len().saturating_sub(1)]
            .iter()
            .all(|&dim| dim == 1);
    if !is_feature_vector {
        anyhow::bail!(
            "{context}: constant shape {:?} is not a scalar or a leading-singleton feature vector of dimension {feature_dim}",
            shape
        );
    }

    Ok(constant.iter().copied().collect())
}

fn checked_f32(value: f64, context: &str) -> Result<f32> {
    if !value.is_finite() || value < f32::MIN as f64 || value > f32::MAX as f64 {
        anyhow::bail!("{context}: folded bias is not representable as finite f32 ({value})");
    }
    let value = value as f32;
    if !value.is_finite() {
        anyhow::bail!("{context}: folded bias rounded to a non-finite f32");
    }
    Ok(value)
}

fn rat_f32(value: f32, context: &str) -> Result<BigRational> {
    BigRational::from_float(value as f64)
        .ok_or_else(|| anyhow::anyhow!("{context}: non-finite f32 coefficient {value}"))
}

/// Convert an exact folded dyadic to the MIP coefficient type without changing
/// its value.  Refusing an inexact conversion is what keeps certified UNSAT
/// admission tied to the original, pre-fold network.
fn exact_rational_to_f64(value: &BigRational, row: usize) -> Result<f64> {
    let encoded = value.to_f64().ok_or_else(|| {
        anyhow::anyhow!("folded Linear bias row {row} is outside finite f64 range")
    })?;
    if !encoded.is_finite()
        || BigRational::from_float(encoded).is_none_or(|round_trip| round_trip != *value)
    {
        anyhow::bail!(
            "folded Linear bias row {row} is not exactly representable as f64; \
             refusing an approximate certified-MIP encoding"
        );
    }
    Ok(encoded)
}

fn fold_constant_after_linear(
    linear: &ExactLinear,
    constant: &ArrayD<f32>,
    subtract: bool,
) -> Result<FoldEntry> {
    let (out_dim, _) = linear.layer.weight.dim();
    let constant = constant_feature_vector(constant, out_dim, "constant after Linear")?;
    let mut folded = linear.clone();
    for row in 0..out_dim {
        let constant = rat_f32(constant[row], "constant after Linear")?;
        if subtract {
            folded.bias[row] -= constant;
        } else {
            folded.bias[row] += constant;
        }
    }
    Ok(FoldEntry::Linear(folded))
}

fn fold_constant_before_linear(
    linear: &ExactLinear,
    constant: &ArrayD<f32>,
    subtract: bool,
) -> Result<FoldEntry> {
    let (out_dim, in_dim) = linear.layer.weight.dim();
    let constant = constant_feature_vector(constant, in_dim, "constant before Linear")?;
    let constant = constant
        .iter()
        .enumerate()
        .map(|(col, &value)| rat_f32(value, &format!("constant before Linear column {col}")))
        .collect::<Result<Vec<_>>>()?;
    let mut folded = linear.clone();
    for row in 0..out_dim {
        let mut contribution = BigRational::zero();
        for (col, c) in constant.iter().enumerate() {
            let weight = rat_f32(
                linear.layer.weight[[row, col]],
                &format!("constant-before-Linear weight [{row},{col}]"),
            )?;
            contribution += weight * c;
        }
        if subtract {
            folded.bias[row] -= contribution;
        } else {
            folded.bias[row] += contribution;
        }
    }
    Ok(FoldEntry::Linear(folded))
}

/// Unfold Conv2d layers into equivalent Linear layers for MIP encoding.
///
/// A Conv2d with kernel W[OC,IC,KH,KW], stride (SH,SW), padding (PH,PW)
/// on input (IC,IH,IW) is mathematically equivalent to a Linear layer with
/// weight matrix of shape (OC*OH*OW, IC*IH*IW) — the Toeplitz/im2col matrix.
///
/// This enables the MIP encoder (which only supports Linear+ReLU) to verify
/// CNN models like malbeware (Conv→ReLU→Flatten→Gemm).
///
/// Reference: alpha-beta-CROWN encodes Conv2d directly as MIP constraints
/// (BoundConv.build_solver, convolution.py:212). We use the equivalent
/// approach of unfolding to a dense Linear layer, which reuses the existing
/// encode_feedforward MIP path. Tractable for small models (malbeware ~4096 inputs).
///
/// `model_input_shape` is the ONNX model input shape (e.g., `[1, 64, 64]` for
/// 1-channel 64x64 images). Used to infer Conv2d spatial dimensions when the
/// Conv2d layer's `input_shape` field is not set (sequential network path).
///
/// Part of #3218.
pub(super) fn unfold_conv2d_to_linear(
    network: &Network,
    model_input_shape: &[usize],
) -> Result<Network> {
    let mut unfolded = Network::new();
    // Track current spatial shape as [C, H, W] for inferring Conv2d input_shape.
    // Initialized from model_input_shape — last 3 dims if 3D+, or inferred from flat.
    let mut current_spatial: Option<(usize, usize, usize)> = if model_input_shape.len() >= 3 {
        let n = model_input_shape.len();
        Some((
            model_input_shape[n - 3],
            model_input_shape[n - 2],
            model_input_shape[n - 1],
        ))
    } else {
        None
    };

    for layer in network.layers() {
        match layer {
            Layer::Conv2d(conv) => {
                // Use the Conv2d's own input_shape if available, otherwise infer
                // from the tracked spatial shape.
                let (ih, iw) = conv
                    .input_shape
                    .or_else(|| current_spatial.map(|(_c, h, w)| (h, w)))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Conv2d layer requires input_shape for MIP unfolding. \
                         Neither Conv2d.input_shape nor model_input_shape provides spatial dims."
                        )
                    })?;
                let ic = conv.in_channels();
                let oc = conv.out_channels();
                let (kh, kw) = conv.kernel_size();
                let (sh, sw) = conv.stride;
                let (ph, pw) = conv.padding;
                let (oh, ow) = conv
                    .output_size(ih, iw)
                    .map_err(|e| anyhow::anyhow!("Conv2d output_size: {}", e))?;

                // The dense im2col loop below assumes groups=1 and dilation=(1,1).
                // A grouped conv's kernel has dim1 = in_channels/groups, so
                // `conv.kernel[[oc_i, ic_i, ..]]` with `ic_i in 0..in_channels`
                // would index OUT OF BOUNDS (panic = lost instance under
                // panic=abort); a dilated conv would use the wrong source-pixel
                // stride and build the WRONG matrix -> an UNSOUND MIP verdict
                // (-150). Neither occurs on the current VNN-COMP benchmark set,
                // but bail SOUNDLY here so a future grouped/dilated conv degrades
                // the MIP escalation to the inconclusive BaB verdict instead of
                // crashing or emitting a wrong result.
                if conv.groups != 1 || conv.dilation != (1, 1) {
                    anyhow::bail!(
                        "Conv2d MIP unfolding unsupported for groups={} dilation={:?}; \
                         the complete verifier falls back to the inconclusive \
                         bound-propagation result.",
                        conv.groups,
                        conv.dilation,
                    );
                }

                let in_flat = ic * ih * iw;
                let out_flat = oc * oh * ow;

                // Guard against pathologically large dense unfoldings before we
                // attempt the allocation. The im2col matrix is `out_flat *
                // in_flat` f32 (4 bytes each); for large-spatial conv stacks
                // (e.g. 1.74M-param soundnessbench with 64x64 inputs) this is
                // multiple GB and would either OOM-abort the process or stall
                // for the whole wall-clock budget. Bailing with an error here
                // is SOUND: the caller degrades the MIP escalation to the
                // already-sound inconclusive BaB verdict (`unknown`/`timeout`)
                // rather than crashing or emitting a wrong verdict. This never
                // affects tractable nets — oval's largest unfold is ~6M
                // elements (24 MB), far below the cap.
                const MAX_UNFOLD_ELEMS: usize = 256_000_000; // ~1 GB f32 weight matrix
                let elems = out_flat.saturating_mul(in_flat);
                if elems > MAX_UNFOLD_ELEMS {
                    anyhow::bail!(
                        "Conv2d MIP unfolding too large: {oc}x{oh}x{ow} (out) x {ic}x{ih}x{iw} (in) \
                         = {elems} dense weight elements (~{:.1} GB) exceeds the {MAX_UNFOLD_ELEMS}-element cap. \
                         The MILP encoding is intractable for this layer; the complete verifier falls back \
                         to the inconclusive bound-propagation result.",
                        (elems as f64) * 4.0 / 1e9,
                    );
                }

                // Build dense weight matrix W_unfolded[out_idx, in_idx]
                // where out_idx = oc_i * OH*OW + oh_i * OW + ow_i
                // and   in_idx  = ic_i * IH*IW + ih_i * IW + iw_i
                let mut weight = Array2::<f32>::zeros((out_flat, in_flat));

                for oc_i in 0..oc {
                    for oh_i in 0..oh {
                        for ow_i in 0..ow {
                            let out_idx = oc_i * oh * ow + oh_i * ow + ow_i;
                            for ic_i in 0..ic {
                                for kh_i in 0..kh {
                                    for kw_i in 0..kw {
                                        let src_h = oh_i * sh + kh_i;
                                        let src_w = ow_i * sw + kw_i;
                                        if src_h >= ph
                                            && src_h < ih + ph
                                            && src_w >= pw
                                            && src_w < iw + pw
                                        {
                                            let actual_ih = src_h - ph;
                                            let actual_iw = src_w - pw;
                                            let in_idx =
                                                ic_i * ih * iw + actual_ih * iw + actual_iw;
                                            weight[[out_idx, in_idx]] =
                                                conv.kernel[[oc_i, ic_i, kh_i, kw_i]];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Broadcast bias: each output channel c gets bias[c] for all spatial positions
                let bias = conv.bias.as_ref().map(|b| {
                    let mut flat_bias = Array1::<f32>::zeros(out_flat);
                    for c in 0..oc {
                        for h in 0..oh {
                            for w in 0..ow {
                                flat_bias[c * oh * ow + h * ow + w] = b[c];
                            }
                        }
                    }
                    flat_bias
                });

                let linear = LinearLayer::new(weight, bias)
                    .map_err(|e| anyhow::anyhow!("Failed to create unfolded Linear: {}", e))?;
                unfolded.add_layer(Layer::Linear(linear));
                // Update tracked spatial shape to Conv2d output
                current_spatial = Some((oc, oh, ow));
            }
            _ => {
                // Track shape changes through non-Conv layers
                match layer {
                    Layer::Flatten(_) | Layer::Reshape(_) => {
                        current_spatial = None; // spatial dims consumed
                    }
                    _ => {} // ReLU, Linear, etc. don't change spatial tracking
                }
                unfolded.add_layer(layer.clone());
            }
        }
    }
    Ok(unfolded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, ArrayD, IxDyn};
    use ny_propagate::layers::{AddConstantLayer, Conv2dLayer, FlattenLayer, SubConstantLayer};

    fn linear(weight: &[f32], out_dim: usize, in_dim: usize, bias: &[f32]) -> Layer {
        let weight = Array2::from_shape_vec((out_dim, in_dim), weight.to_vec()).unwrap();
        let bias = Array1::from_vec(bias.to_vec());
        Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap())
    }

    fn vector_constant(values: &[f32]) -> ArrayD<f32> {
        Array1::from_vec(values.to_vec()).into_dyn()
    }

    fn assert_close(actual: f32, expected: f32, context: &str) {
        let tolerance = 2.0e-6_f32.max(expected.abs() * 2.0e-6);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}: expected {expected}, got {actual} (tolerance {tolerance})"
        );
    }

    /// Evaluate the affine/ReLU subset in f64, independently of the fold.
    /// This checks the algebra on samples without conflating it with IBP's
    /// intentional directed-rounding width.
    fn eval_foldable_network(network: &Network, input: &[f64]) -> Vec<f64> {
        let mut value = input.to_vec();
        for layer in network.layers() {
            match layer {
                Layer::Linear(linear) => {
                    let (out_dim, in_dim) = linear.weight.dim();
                    assert_eq!(value.len(), in_dim);
                    value = (0..out_dim)
                        .map(|row| {
                            let bias = linear.bias.as_ref().map_or(0.0, |b| b[row] as f64);
                            bias + (0..in_dim)
                                .map(|col| linear.weight[[row, col]] as f64 * value[col])
                                .sum::<f64>()
                        })
                        .collect();
                }
                Layer::AddConstant(add) => {
                    let c =
                        constant_feature_vector(add.constant(), value.len(), "test AddConstant")
                            .unwrap();
                    for (x, c) in value.iter_mut().zip(c) {
                        *x += c as f64;
                    }
                }
                Layer::SubConstant(sub) if !sub.reverse => {
                    let c =
                        constant_feature_vector(sub.constant(), value.len(), "test SubConstant")
                            .unwrap();
                    for (x, c) in value.iter_mut().zip(c) {
                        *x -= c as f64;
                    }
                }
                Layer::ReLU(_) => {
                    for x in &mut value {
                        *x = x.max(0.0);
                    }
                }
                other => panic!("unsupported test layer: {other:?}"),
            }
        }
        value
    }

    fn eval_foldable_network_exact(network: &Network, input: &[f32]) -> Vec<BigRational> {
        let mut value = input
            .iter()
            .map(|&x| rat_f32(x, "exact test input").unwrap())
            .collect::<Vec<_>>();
        for layer in network.layers() {
            match layer {
                Layer::Linear(linear) => {
                    let (out_dim, in_dim) = linear.weight.dim();
                    assert_eq!(value.len(), in_dim);
                    value = (0..out_dim)
                        .map(|row| {
                            let mut output =
                                linear.bias.as_ref().map_or_else(BigRational::zero, |bias| {
                                    rat_f32(bias[row], "exact test bias").unwrap()
                                });
                            for col in 0..in_dim {
                                output += rat_f32(linear.weight[[row, col]], "exact test weight")
                                    .unwrap()
                                    * &value[col];
                            }
                            output
                        })
                        .collect();
                }
                Layer::AddConstant(add) => {
                    let c = constant_feature_vector(
                        add.constant(),
                        value.len(),
                        "exact test AddConstant",
                    )
                    .unwrap();
                    for (x, c) in value.iter_mut().zip(c) {
                        *x += rat_f32(c, "exact test AddConstant value").unwrap();
                    }
                }
                Layer::SubConstant(sub) if !sub.reverse => {
                    let c = constant_feature_vector(
                        sub.constant(),
                        value.len(),
                        "exact test SubConstant",
                    )
                    .unwrap();
                    for (x, c) in value.iter_mut().zip(c) {
                        *x -= rat_f32(c, "exact test SubConstant value").unwrap();
                    }
                }
                Layer::ReLU(_) => {
                    for x in &mut value {
                        if *x < BigRational::zero() {
                            *x = BigRational::zero();
                        }
                    }
                }
                other => panic!("unsupported exact test layer: {other:?}"),
            }
        }
        value
    }

    fn eval_folded_mip_exact(folded: &FoldedMipNetwork, input: &[f32]) -> Vec<BigRational> {
        let mut value = input
            .iter()
            .map(|&x| rat_f32(x, "folded exact test input").unwrap())
            .collect::<Vec<_>>();
        let mut bias_index = 0usize;
        for layer in folded.layers() {
            match layer {
                Layer::Linear(linear) => {
                    let bias = &folded.exact_biases()[bias_index];
                    bias_index += 1;
                    let (out_dim, in_dim) = linear.weight.dim();
                    assert_eq!(value.len(), in_dim);
                    value = (0..out_dim)
                        .map(|row| {
                            let mut output = BigRational::from_float(bias[row]).unwrap();
                            for col in 0..in_dim {
                                output +=
                                    rat_f32(linear.weight[[row, col]], "folded exact test weight")
                                        .unwrap()
                                        * &value[col];
                            }
                            output
                        })
                        .collect();
                }
                Layer::ReLU(_) => {
                    for x in &mut value {
                        if *x < BigRational::zero() {
                            *x = BigRational::zero();
                        }
                    }
                }
                other => panic!("folded MIP test network retained unsupported layer: {other:?}"),
            }
        }
        assert_eq!(bias_index, folded.exact_biases().len());
        value
    }

    fn eval_folded_mip_f64(folded: &FoldedMipNetwork, input: &[f64]) -> Vec<f64> {
        let mut value = input.to_vec();
        let mut bias_index = 0usize;
        for layer in folded.layers() {
            match layer {
                Layer::Linear(linear) => {
                    let bias = &folded.exact_biases()[bias_index];
                    bias_index += 1;
                    let (out_dim, in_dim) = linear.weight.dim();
                    value = (0..out_dim)
                        .map(|row| {
                            bias[row]
                                + (0..in_dim)
                                    .map(|col| linear.weight[[row, col]] as f64 * value[col])
                                    .sum::<f64>()
                        })
                        .collect();
                }
                Layer::ReLU(_) => {
                    for x in &mut value {
                        *x = x.max(0.0);
                    }
                }
                other => panic!("folded f64 test retained unsupported layer: {other:?}"),
            }
        }
        value
    }

    fn unit_linear() -> Layer {
        linear(&[1.0], 1, 1, &[0.0])
    }

    #[test]
    fn mip_feedforward_topology_accepts_acas_alternation() {
        // ACAS-Xu rows 66/74 fold to seven affine layers separated by six
        // ReLUs.  Keep that exact encoder contract accepted.
        let mut network = Network::new();
        for affine_index in 0..7 {
            network.add_layer(unit_linear());
            if affine_index < 6 {
                network.add_layer(Layer::ReLU(Default::default()));
            }
        }

        validate_mip_feedforward_topology(&network).unwrap();
        let (weights, biases, dims) = extract_linear_relu_params(&network).unwrap();
        assert_eq!(weights.len(), 7);
        assert_eq!(biases.len(), 7);
        assert_eq!(dims, vec![1; 8]);
        let intermediate_bounds = vec![vec![Bound::new(-1.0, 1.0)]; 6];
        ny_mip::encode_feedforward(
            &weights,
            &biases,
            &dims,
            &[Bound::new(-1.0, 1.0)],
            &intermediate_bounds,
        )
        .expect("accepted topology must have exactly the encoder's implicit ReLU count");
    }

    #[test]
    fn mip_feedforward_topology_rejects_adjacent_linears() {
        let mut network = Network::new();
        network.add_layer(unit_linear());
        network.add_layer(unit_linear());

        let error = validate_mip_feedforward_topology(&network)
            .unwrap_err()
            .to_string();
        assert!(error.contains("layer 1: encoder expects ReLU, found Linear"));
        // The extractor is a second fail-closed seam: no caller can erase the
        // malformed activation sequence before handing it to the encoder.
        assert!(extract_linear_relu_params(&network).is_err());
    }

    #[test]
    fn mip_feedforward_topology_rejects_other_malformed_sequences() {
        let mut leading_relu = Network::new();
        leading_relu.add_layer(Layer::ReLU(Default::default()));
        leading_relu.add_layer(unit_linear());
        let error = validate_mip_feedforward_topology(&leading_relu)
            .unwrap_err()
            .to_string();
        assert!(error.contains("layer 0: encoder expects Linear, found ReLU"));

        let mut trailing_relu = Network::new();
        trailing_relu.add_layer(unit_linear());
        trailing_relu.add_layer(Layer::ReLU(Default::default()));
        let error = validate_mip_feedforward_topology(&trailing_relu)
            .unwrap_err()
            .to_string();
        assert!(error.contains("trailing ReLU has no following Linear"));

        let mut double_relu = Network::new();
        double_relu.add_layer(unit_linear());
        double_relu.add_layer(Layer::ReLU(Default::default()));
        double_relu.add_layer(Layer::ReLU(Default::default()));
        double_relu.add_layer(unit_linear());
        let error = validate_mip_feedforward_topology(&double_relu)
            .unwrap_err()
            .to_string();
        assert!(error.contains("layer 2: encoder expects Linear, found ReLU"));

        let mut unsupported = Network::new();
        unsupported.add_layer(unit_linear());
        unsupported.add_layer(Layer::AddConstant(AddConstantLayer::new(vector_constant(
            &[1.0],
        ))));
        unsupported.add_layer(unit_linear());
        let error = validate_mip_feedforward_topology(&unsupported)
            .unwrap_err()
            .to_string();
        assert!(error.contains("layer 1: encoder expects ReLU, found Add"));
    }

    /// Regression for the measured ACAS-Xu row-66/row-74 topology.  The old
    /// cursor pass folded `SubConstant -> Linear`, advanced past the new Linear,
    /// and left the following 50-element AddConstant at MIP layer 1.
    #[test]
    fn fold_constant_layers_reaches_fixed_point_across_acas_bias_chain() {
        let mut network = Network::new();
        network.add_layer(Layer::SubConstant(SubConstantLayer::new(vector_constant(
            &[1.0, -2.0, 0.5],
        ))));
        network.add_layer(linear(
            &[2.0, -1.0, 0.5, -3.0, 4.0, 2.0],
            2,
            3,
            &[0.1, -0.2],
        ));
        network.add_layer(Layer::AddConstant(AddConstantLayer::new(vector_constant(
            &[0.25, -0.75],
        ))));
        network.add_layer(Layer::ReLU(Default::default()));

        let folded = fold_constant_layers(&network).unwrap();
        assert_eq!(folded.layers().len(), 2);
        let Layer::Linear(linear) = &folded.layers()[0] else {
            panic!("ACAS constant prelude must reduce to one Linear");
        };
        let bias = linear.bias.as_ref().unwrap();
        // b' = b - W*[1,-2,.5] + [.25,-.75]
        assert_close(bias[0], -3.9, "ACAS folded bias[0]");
        assert_close(bias[1], 9.05, "ACAS folded bias[1]");
        assert!(matches!(folded.layers()[1], Layer::ReLU(_)));

        for input in [[-4.0, 0.25, 3.0], [0.0, 0.0, 0.0], [1.25, -3.5, 0.75]] {
            let original = eval_foldable_network(&network, &input);
            let reduced = eval_foldable_network(&folded, &input);
            for (idx, (&actual, &expected)) in reduced.iter().zip(&original).enumerate() {
                assert!(
                    (actual - expected).abs() <= 2.0e-6,
                    "sample {input:?}, output {idx}: folded={actual}, original={expected}"
                );
            }
        }
    }

    #[test]
    fn fold_add_before_linear_uses_b_plus_wc_through_both_sided_chains() {
        let mut network = Network::new();
        network.add_layer(Layer::AddConstant(AddConstantLayer::new(vector_constant(
            &[0.5, -1.0, 2.0],
        ))));
        network.add_layer(Layer::SubConstant(SubConstantLayer::new(vector_constant(
            &[0.25, 0.5, -1.0],
        ))));
        network.add_layer(linear(
            &[1.5, -2.0, 0.25, -4.0, 0.5, 3.0],
            2,
            3,
            &[0.75, -0.25],
        ));
        network.add_layer(Layer::AddConstant(AddConstantLayer::new(vector_constant(
            &[2.0, -3.0],
        ))));
        network.add_layer(Layer::SubConstant(SubConstantLayer::new(vector_constant(
            &[0.5, 1.0],
        ))));

        let folded = fold_constant_layers(&network).unwrap();
        assert_eq!(folded.layers().len(), 1);
        let Layer::Linear(linear) = &folded.layers()[0] else {
            panic!("both-sided constant chain must reduce to Linear");
        };
        let bias = linear.bias.as_ref().unwrap();
        // The pre-linear shift is c=[.25,-1.5,3], then post shift=[1.5,-4].
        // b' = b + Wc + post.
        assert_close(bias[0], 6.375, "both-sided folded bias[0]");
        assert_close(bias[1], 3.0, "both-sided folded bias[1]");

        for input in [[-2.0, 1.0, 0.0], [0.125, -0.25, 0.5], [10.0, -8.0, 3.0]] {
            let original = eval_foldable_network(&network, &input);
            let reduced = eval_foldable_network(&folded, &input);
            for (actual, expected) in reduced.iter().zip(original) {
                assert!((actual - expected).abs() <= 2.0e-6);
            }
        }
    }

    #[test]
    fn fold_constant_layers_accepts_scalar_and_leading_singleton_feature_broadcasts() {
        let mut network = Network::new();
        let leading_singletons =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
        network.add_layer(Layer::AddConstant(AddConstantLayer::new(
            leading_singletons,
        )));
        network.add_layer(linear(&[1.0, 1.0, 1.0], 1, 3, &[4.0]));
        network.add_layer(Layer::SubConstant(SubConstantLayer::scalar(2.5)));

        let folded = fold_constant_layers(&network).unwrap();
        let Layer::Linear(linear) = &folded.layers()[0] else {
            panic!("broadcast-safe constants must fold");
        };
        assert_eq!(folded.layers().len(), 1);
        // 4 + [1,1,1] dot [1,2,3] - 2.5 = 7.5
        assert_close(linear.bias.as_ref().unwrap()[0], 7.5, "broadcast bias");
    }

    #[test]
    fn fold_constant_layers_rejects_ambiguous_or_mismatched_broadcast_axes() {
        let mut wrong_axis = Network::new();
        wrong_axis.add_layer(Layer::AddConstant(AddConstantLayer::new(
            ArrayD::from_shape_vec(IxDyn(&[3, 1]), vec![1.0, 2.0, 3.0]).unwrap(),
        )));
        wrong_axis.add_layer(linear(&[1.0, 1.0, 1.0], 1, 3, &[0.0]));
        let err = fold_constant_layers(&wrong_axis).unwrap_err().to_string();
        assert!(err.contains("not a scalar or a leading-singleton feature vector"));

        let mut wrong_len = Network::new();
        wrong_len.add_layer(linear(&[1.0, 1.0, 1.0], 1, 3, &[0.0]));
        wrong_len.add_layer(Layer::AddConstant(AddConstantLayer::new(vector_constant(
            &[1.0, 2.0],
        ))));
        let err = fold_constant_layers(&wrong_len).unwrap_err().to_string();
        assert!(err.contains("feature vector of dimension 1"));
    }

    #[test]
    fn fold_constant_layers_fails_closed_on_non_finite_folded_bias() {
        let mut network = Network::new();
        network.add_layer(Layer::AddConstant(AddConstantLayer::new(vector_constant(
            &[2.0],
        ))));
        network.add_layer(linear(&[f32::MAX], 1, 1, &[0.0]));

        let err = fold_constant_layers(&network).unwrap_err().to_string();
        assert!(err.contains("not representable as finite f32"));
    }

    #[test]
    fn fold_constant_layers_fails_closed_when_exact_bias_is_not_f64_representable() {
        let mut network = Network::new();
        // 1 + 2^-149 is an exact dyadic result of f32 parameters, but the tiny
        // addend is far below f64's 53-bit significand at magnitude 1.  A
        // nearest-f64 fold would silently drop it, so certified MIP must refuse.
        network.add_layer(Layer::AddConstant(AddConstantLayer::new(vector_constant(
            &[f32::from_bits(1)],
        ))));
        network.add_layer(linear(&[1.0], 1, 1, &[1.0]));

        let err = fold_constant_layers(&network).unwrap_err().to_string();
        assert!(err.contains("not exactly representable as f64"));
    }

    #[test]
    fn fold_constant_layers_preserves_reverse_subtraction_for_fail_closed_validation() {
        let mut network = Network::new();
        network.add_layer(Layer::SubConstant(SubConstantLayer::new_reverse(
            vector_constant(&[1.0, 2.0]),
        )));
        network.add_layer(linear(&[1.0, 0.0, 0.0, 1.0], 2, 2, &[0.0, 0.0]));

        let folded = fold_constant_layers(&network).unwrap();
        assert_eq!(folded.layers().len(), 2);
        assert!(matches!(
            &folded.layers()[0],
            Layer::SubConstant(sub) if sub.reverse
        ));
        assert!(matches!(&folded.layers()[1], Layer::Linear(_)));
    }

    fn audit_external_acas_fold(
        model_path: &std::path::Path,
        vnnlib_path: Option<&std::path::Path>,
    ) {
        let model = ny_onnx::load_onnx(model_path).expect("load external ACAS fold-audit model");
        let network = model
            .to_propagate_network()
            .expect("convert external ACAS model to sequential network");
        let network = strip_shape_layers(&network);
        let folded = fold_constant_layers(&network).expect("fold external ACAS constants exactly");
        validate_mip_feedforward_topology(&folded)
            .expect("external ACAS fold must exactly match the implicit-ReLU MIP encoder");
        assert_eq!(
            folded.layers().len(),
            13,
            "ACAS rows 66/74 must fold to seven Linear and six ReLU layers"
        );

        let input_dim = folded
            .layers()
            .iter()
            .find_map(|layer| match layer {
                Layer::Linear(linear) => Some(linear.weight.ncols()),
                _ => None,
            })
            .expect("external ACAS model has a Linear layer");
        let mut samples = vec![
            vec![0.0_f32; input_dim],
            (0..input_dim)
                .map(|i| if i % 2 == 0 { 0.125 } else { -0.25 })
                .collect(),
        ];
        if let Some(vnnlib_path) = vnnlib_path {
            let spec = ny_onnx::vnnlib::load_vnnlib(vnnlib_path)
                .expect("load external ACAS fold-audit VNNLIB");
            let (lower, upper) = spec.split_input_bounds_f32();
            assert_eq!(lower.len(), input_dim);
            assert_eq!(upper.len(), input_dim);
            let midpoint = lower
                .iter()
                .zip(&upper)
                .map(|(&lo, &hi)| f32::midpoint(lo, hi))
                .collect();
            samples.extend([lower, upper, midpoint]);
        }

        let mut max_f64_error = 0.0_f64;
        for (sample_index, sample) in samples.iter().enumerate() {
            let original_exact = eval_foldable_network_exact(&network, sample);
            let folded_exact = eval_folded_mip_exact(&folded, sample);
            assert_eq!(
                folded_exact, original_exact,
                "exact ACAS fold mismatch at deterministic sample {sample_index}"
            );

            let sample_f64 = sample.iter().map(|&x| x as f64).collect::<Vec<_>>();
            let original_f64 = eval_foldable_network(&network, &sample_f64);
            let folded_f64 = eval_folded_mip_f64(&folded, &sample_f64);
            for (&original, &reduced) in original_f64.iter().zip(&folded_f64) {
                max_f64_error = max_f64_error.max((original - reduced).abs());
            }
        }
        eprintln!(
            "ACAS_FOLD_EQUIVALENCE model={} samples={} exact_max_error=0 max_f64_eval_error={max_f64_error:.17e}",
            model_path.display(),
            samples.len(),
        );
    }

    /// Opt-in real-model audit used by the score campaign.  It is a no-op in
    /// ordinary CI; provide `NY_TEST_ACAS_FOLD_MODEL` and optionally
    /// `NY_TEST_ACAS_FOLD_VNNLIB` to prove exact algebraic parity and exact MIP
    /// topology on the sealed ACAS network and deterministic property samples.
    #[test]
    fn external_acas_constant_fold_is_exact_on_deterministic_samples() {
        let Some(model_path) = std::env::var_os("NY_TEST_ACAS_FOLD_MODEL") else {
            return;
        };
        let vnnlib_path =
            std::env::var_os("NY_TEST_ACAS_FOLD_VNNLIB").map(std::path::PathBuf::from);
        audit_external_acas_fold(
            &std::path::PathBuf::from(model_path),
            vnnlib_path.as_deref(),
        );
    }

    fn audit_external_acas_v2_row(row_index: usize, onnx: &str) {
        let Some(version_root) = std::env::var_os("NY_TEST_ACAS_V2_ROOT") else {
            return;
        };
        let version_root = std::path::PathBuf::from(version_root);
        let vnnlib = "vnnlib/prop_2.vnnlib";
        let expected_row = format!("onnx/{onnx},{vnnlib},116");
        let instances = std::fs::read_to_string(version_root.join("instances.csv"))
            .expect("read external ACAS v2 instances.csv");
        assert_eq!(
            instances.lines().nth(row_index - 1),
            Some(expected_row.as_str()),
            "ACAS physical row binding changed"
        );
        audit_external_acas_fold(
            &version_root.join("onnx").join(onnx),
            Some(&version_root.join(vnnlib)),
        );
    }

    /// Opt-in exact regression for VNN-COMP ACAS v2 physical row 66.
    #[test]
    fn external_acas_row66_fold_matches_mip_topology_and_algebra() {
        audit_external_acas_v2_row(66, "ACASXU_run2a_3_3_batch_2000.onnx");
    }

    /// Opt-in exact regression for VNN-COMP ACAS v2 physical row 74.
    #[test]
    fn external_acas_row74_fold_matches_mip_topology_and_algebra() {
        audit_external_acas_v2_row(74, "ACASXU_run2a_4_2_batch_2000.onnx");
    }

    /// Verify Conv2d→Linear unfolding produces the correct dense weight matrix.
    /// Uses a 1-channel 4x4 input with a 2x1 filter (2x2 kernel, stride 1, no padding).
    /// The unfolded Linear should produce identical output to the Conv2d.
    #[test]
    fn test_unfold_conv2d_identity_no_padding() {
        // Conv2d: IC=1, OC=2, kernel 2x2, stride 1, no padding
        // Input: 1x4x4 → Output: 2x3x3
        let kernel_data = vec![
            // OC=0, IC=0, 2x2 kernel
            1.0f32, 0.0, 0.0, 1.0, // identity-like
            // OC=1, IC=0, 2x2 kernel
            0.0, 1.0, 1.0, 0.0, // anti-diagonal
        ];
        let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 2, 2]), kernel_data).unwrap();
        let bias = Some(Array1::from_vec(vec![0.5, -0.5]));
        let conv = Conv2dLayer::with_input_shape(kernel, bias, (1, 1), (0, 0), 4, 4).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Conv2d(conv));

        let unfolded = unfold_conv2d_to_linear(&network, &[1, 4, 4]).unwrap();
        let layers = unfolded.layers();
        assert_eq!(layers.len(), 1);

        let linear = match &layers[0] {
            Layer::Linear(l) => l,
            _ => panic!("Expected Linear layer"),
        };

        // Output: 2 channels * 3 rows * 3 cols = 18 outputs
        // Input: 1 channel * 4 rows * 4 cols = 16 inputs
        assert_eq!(linear.weight.dim(), (18, 16));
        assert!(linear.bias.is_some());
        let bias = linear.bias.as_ref().unwrap();
        assert_eq!(bias.len(), 18);

        // Verify bias broadcast: first 9 entries should be 0.5, next 9 should be -0.5
        for i in 0..9 {
            assert_eq!(bias[i], 0.5, "bias[{}] for OC=0", i);
        }
        for i in 9..18 {
            assert_eq!(bias[i], -0.5, "bias[{}] for OC=1", i);
        }

        // Verify specific weight entries for output (0, 0, 0) = OC=0, OH=0, OW=0
        // This output pixel depends on input positions (0,0), (0,1), (1,0), (1,1)
        // with kernel weights [1, 0, 0, 1]
        let out_idx = 0; // oc=0, oh=0, ow=0
        assert_eq!(linear.weight[[out_idx, 0]], 1.0); // (0,0) * 1
        assert_eq!(linear.weight[[out_idx, 1]], 0.0); // (0,1) * 0
        assert_eq!(linear.weight[[out_idx, 4]], 0.0); // (1,0) * 0
        assert_eq!(linear.weight[[out_idx, 5]], 1.0); // (1,1) * 1

        // Verify concrete computation: apply the Linear to a known input
        // Input: 4x4 all ones → Conv2d output at (0,0,0) should be 1*1 + 0*1 + 0*1 + 1*1 = 2
        let input_flat: Vec<f32> = vec![1.0; 16];
        let output: f32 = (0..16)
            .map(|j| linear.weight[[out_idx, j]] * input_flat[j])
            .sum::<f32>()
            + bias[out_idx];
        assert_eq!(output, 2.5); // 2.0 + 0.5 bias
    }

    /// Verify Conv2d→Linear unfolding with padding.
    #[test]
    fn test_unfold_conv2d_with_padding() {
        // Conv2d: IC=1, OC=1, kernel 3x3, stride 1, padding 1
        // Input: 1x3x3 → Output: 1x3x3 (same spatial with pad=1)
        let kernel_data = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, // identity kernel (center = 1)
            0.0, 0.0, 0.0f32,
        ];
        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), kernel_data).unwrap();
        let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (1, 1), 3, 3).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Conv2d(conv));

        let unfolded = unfold_conv2d_to_linear(&network, &[1, 3, 3]).unwrap();
        let linear = match &unfolded.layers()[0] {
            Layer::Linear(l) => l,
            _ => panic!("Expected Linear"),
        };

        // Same spatial: 9 outputs, 9 inputs
        assert_eq!(linear.weight.dim(), (9, 9));

        // Identity kernel with padding=1 should produce identity matrix
        for i in 0..9 {
            for j in 0..9 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(linear.weight[[i, j]], expected, "weight[{},{}]", i, j);
            }
        }
    }

    /// Verify the full pipeline: Conv2d → ReLU → Flatten → Linear
    /// unfolds to Linear → ReLU → Linear (Flatten stripped).
    #[test]
    fn test_unfold_conv2d_full_pipeline() {
        // Conv2d: IC=1, OC=1, kernel 2x2, stride 2, no pad
        // Input: 1x4x4 → Output: 1x2x2 = 4 outputs
        let kernel =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0f32, 1.0, 1.0, 1.0]).unwrap();
        let conv = Conv2dLayer::with_input_shape(kernel, None, (2, 2), (0, 0), 4, 4).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Conv2d(conv));
        network.add_layer(Layer::ReLU(Default::default()));
        network.add_layer(Layer::Flatten(FlattenLayer { axis: 1 }));
        // Final linear: 4 inputs → 2 outputs
        let weight = Array2::from_shape_vec((2, 4), vec![1.0; 8]).unwrap();
        let linear = LinearLayer::new(weight, None).unwrap();
        network.add_layer(Layer::Linear(linear));

        // Unfold Conv2d → Linear
        let unfolded = unfold_conv2d_to_linear(&network, &[1, 4, 4]).unwrap();
        assert_eq!(unfolded.layers().len(), 4); // Linear, ReLU, Flatten, Linear

        // Strip Flatten → 3 layers
        let stripped = strip_shape_layers(&unfolded);
        assert_eq!(stripped.layers().len(), 3); // Linear, ReLU, Linear

        // All layers should be Linear or ReLU
        for layer in stripped.layers() {
            assert!(
                matches!(layer, Layer::Linear(_) | Layer::ReLU(_)),
                "Unexpected layer: {:?}",
                layer
            );
        }

        // Extract params should succeed
        let (weights, biases, dims) = extract_linear_relu_params(&stripped).unwrap();
        assert_eq!(weights.len(), 2); // two Linear layers
        assert_eq!(dims, vec![16, 4, 2]); // 16 inputs → 4 → 2
        assert_eq!(biases[0].len(), 4);
        assert_eq!(biases[1].len(), 2);
    }
}
