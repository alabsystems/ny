// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Private, bounded pure-chain extraction with explicit raw-BN error charges.
//!
//! This does not authenticate an ONNX load. The root owner must supply a graph
//! loaded with `OnnxLoadConfig::certified_graph_lowering()` (PreserveRaw), its
//! original input box, and bounds/alpha derived from THAT graph and domain.
//! A graph whose loader already discarded BN errors cannot be repaired here.
//! Only the bounded f64 backend that charges `CertifiedWeightError` may consume
//! these descriptors. Existing extraction and ordinary GPU dispatch are unchanged.

use super::{bake_relu_layer, has_active_non_relu_alpha, resolve_pre};
use crate::bounds::GraphAlphaState;
use crate::layers::{normalization::BatchNormLayer, Layer};
use crate::network::core::{try_extract_single_gpu_layer, GraphNetwork, NETWORK_INPUT};
use ny_core::{is_crown_coeff_safe, CertifiedWeightError, GpuCrownLayer, GpuResnetSegment};
use ny_tensor::BoundedTensor;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

const MAX_NODES: usize = 128;
const MAX_DIM: usize = 1 << 20;
const MAX_BN_QUOTE_CHANNELS: usize = 8192;
const MAX_CALLER_BYTES: usize = 256 << 20;
const POLL_WORDS: usize = 4096;

/// Caller-owned extraction peak only, NOT a backend/device/process reservation.
/// The root owner must add bounded f64 fold/device workspace before dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExtractionLimits {
    pub max_nodes: usize,
    pub max_caller_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExtractionQuote {
    pub nodes: usize,
    pub charged_bn_layers: usize,
    pub float_words: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub total_weight_words: usize,
    pub total_macs_per_row: usize,
    pub total_affine_output_dim: usize,
    pub total_activation_dim: usize,
    pub total_conv_col_words_per_row: usize,
    pub max_frontier_dim: usize,
    pub max_gemm_output_per_row: usize,
    pub caller_owned_peak_bytes: usize,
}

/// Conservative aggregate extraction + bounded-f64-fold workspace. Shared by
/// root-cut evaluation and selected identity rows, which consume the same
/// charged descriptors and backend. This is an admission bound, not a lease.
pub(crate) fn bounded_fold_bytes(q: ExtractionQuote, rows: usize) -> Option<usize> {
    let dims = q.nodes.checked_mul(q.max_frontier_dim)?;
    q.caller_owned_peak_bytes
        .checked_add(
            256usize
                .checked_mul(rows)?
                .checked_mul(q.max_frontier_dim.max(q.max_gemm_output_per_row))?,
        )?
        .checked_add(32usize.checked_mul(q.total_weight_words)?)?
        .checked_add(64usize.checked_mul(rows)?.checked_mul(dims)?)?
        .checked_add(
            64usize
                .checked_mul(rows)?
                .checked_mul(q.input_dim.checked_add(q.output_dim)?)?,
        )?
        .checked_add(1 << 20)
}

/// Static operation charge for materialization and complete checked folds.
/// Counts both coefficient sides, magnitudes and inherited errors. It does not
/// predict elapsed time or authorize an unbounded accelerator dispatch.
pub(crate) fn bounded_fold_work(q: ExtractionQuote, rows: u64, repetitions: u64) -> Option<u64> {
    let macs = u64::try_from(q.total_macs_per_row).ok()?;
    let dims = u64::try_from(
        q.total_affine_output_dim
            .checked_add(q.total_activation_dim)?
            .checked_add(q.input_dim)?
            .checked_add(q.output_dim)?,
    )
    .ok()?;
    let cols = u64::try_from(q.total_conv_col_words_per_row).ok()?;
    // Six contractions, two operations per MAC; col2im/copy/validation and
    // bias/error/activation/concretization loops are charged separately.
    let fold = 12u64
        .checked_mul(rows)?
        .checked_mul(macs)?
        .checked_add(128u64.checked_mul(rows)?.checked_mul(cols)?)?
        .checked_add(512u64.checked_mul(rows)?.checked_mul(dims)?)?;
    fold.checked_mul(repetitions)?
        .checked_add(64u64.checked_mul(u64::try_from(q.total_activation_dim).ok()?)?)?
        .checked_add(
            32u64
                .checked_mul(u64::try_from(q.float_words).ok()?)?
                .checked_mul(repetitions.checked_add(1)?)?,
        )?
        .checked_add(4096u64.checked_mul(u64::try_from(q.nodes).ok()?)?)
}

pub(crate) struct ChargedRawBnChain {
    pub segments: Vec<GpuResnetSegment>,
    pub relu_names: Vec<String>,
    pub frontier_abs: Vec<Vec<f32>>,
    pub node_abs: Vec<Vec<f32>>,
    #[cfg_attr(not(test), expect(dead_code, reason = "preserve source dimensions"))]
    pub input_dim: usize,
    pub output_dim: usize,
    pub quote: ExtractionQuote,
}

// Fixed labels only: no graph-name copying, extra source scan, or proof authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MaterializationFailure {
    stage: &'static str,
    reason: &'static str,
    node_ordinal: usize,
    layer_kind: &'static str,
}

#[cfg(test)]
thread_local! {
    static MATERIALIZATION_FAILURE: std::cell::Cell<Option<MaterializationFailure>> = const { std::cell::Cell::new(None) };
    static MATERIALIZATION_FAILURE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn layer_kind(layer: &Layer) -> &'static str {
    match layer {
        Layer::Linear(_) => "linear",
        Layer::Conv2d(_) => "conv2d",
        Layer::BatchNorm(_) => "batch-norm",
        Layer::ReLU(_) => "relu",
        Layer::Flatten(_) => "flatten",
        Layer::Reshape(_) => "reshape",
        _ => "unsupported",
    }
}

struct Guard<'a> {
    deadline: Instant,
    poll: &'a mut dyn FnMut() -> bool,
    failed: bool,
    report_materialization: bool,
    completed: bool,
    failure: MaterializationFailure,
}
impl<'a> Guard<'a> {
    fn new(deadline: Instant, poll: &'a mut dyn FnMut() -> bool, materialize: bool) -> Self {
        Self {
            deadline,
            poll,
            failed: false,
            report_materialization: materialize,
            completed: false,
            failure: MaterializationFailure {
                stage: "entry",
                reason: "guard",
                // MAX_NODES denotes no current node. Otherwise the ordinal is
                // backward from start_node, the order already used by steps.
                node_ordinal: MAX_NODES,
                layer_kind: "none",
            },
        }
    }
    fn at(&mut self, stage: &'static str, reason: &'static str) {
        self.failure.stage = stage;
        self.failure.reason = reason;
    }
    fn node(&mut self, ordinal: usize, kind: &'static str) {
        self.failure.node_ordinal = ordinal.min(MAX_NODES);
        self.failure.layer_kind = kind;
    }
    fn refuse<T>(&mut self, reason: &'static str) -> Option<T> {
        self.failure.reason = reason;
        None
    }
    fn check(&mut self) -> Option<()> {
        // Same short-circuit order and exactly the original clock/poll calls.
        // Preserve the first cancellation reason after the sticky failure.
        if self.failed {
            None
        } else if Instant::now() >= self.deadline {
            self.failed = true;
            self.refuse("deadline")
        } else if !(self.poll)() {
            self.failed = true;
            self.refuse("poll")
        } else {
            Some(())
        }
    }
    fn values(&mut self, values: &[f32]) -> Option<()> {
        for chunk in values.chunks(POLL_WORDS) {
            self.check()?;
            if chunk.iter().any(|&x| !is_crown_coeff_safe(x)) {
                return self.refuse("unsafe-value");
            }
        }
        Some(())
    }
    fn bounds(&mut self, b: &BoundedTensor) -> Option<()> {
        if b.is_empty() || b.len() > MAX_DIM {
            return self.refuse("bounds-shape");
        }
        self.failure.reason = "bounds-layout";
        let lo = b.lower().as_slice()?;
        let hi = b.upper().as_slice()?;
        if lo.len() != hi.len() {
            return self.refuse("bounds-shape");
        }
        for (l, u) in lo.chunks(POLL_WORDS).zip(hi.chunks(POLL_WORDS)) {
            self.check()?;
            if l.iter()
                .zip(u)
                .any(|(&a, &b)| !is_crown_coeff_safe(a) || !is_crown_coeff_safe(b) || a > b)
            {
                return self.refuse("bounds-value-or-order");
            }
        }
        Some(())
    }
}

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        if !self.report_materialization || self.completed {
            return;
        }
        let f = self.failure;
        tracing::info!(
            stage = f.stage,
            reason = f.reason,
            node_ordinal = f.node_ordinal,
            layer_kind = f.layer_kind,
            cancellation = self.failed,
            "Certified raw chain materialization refused"
        );
        #[cfg(test)]
        {
            MATERIALIZATION_FAILURE.with(|slot| slot.set(Some(f)));
            MATERIALIZATION_FAILURE_COUNT.with(|count| count.set(count.get() + 1));
        }
    }
}

// The quote already reserves 8192 bytes for fixed local scratch. Check actual
// ABI sizes rather than assuming reference/Option/Instant layouts. The 2048
// remainder covers scalar walk locals, temporary quotes/headers and the fixed
// diagnostic event fields. Payload allocations keep their original quotes.
fn fixed_stack_bytes() -> Option<usize> {
    size_of::<[Option<(&str, &Layer, &BoundedTensor)>; MAX_NODES]>()
        .checked_add(size_of::<Guard<'_>>())?
        .checked_add(size_of::<ExtractionQuote>())?
        .checked_add(2048)
}

/// Convert the authenticated per-channel ABSOLUTE errors to the existing
/// per-entry relative weight contract. The product of two finite binary32
/// values is exact in binary64 (48 significand bits, representable exponent),
/// so the post-division product check is authoritative even for subnormals.
fn charge(bn: &BatchNormLayer, guard: &mut Guard<'_>) -> Option<CertifiedWeightError> {
    guard.at("raw-bn-charge", "guard");
    guard.check()?;
    guard.failure.reason = "proof-environment";
    ny_core::require_f64_interval_proof_environment().ok()?;
    guard.failure.reason = "affine-parameters";
    bn.validate_affine_parameters().ok()?;
    guard.failure.reason = "parameter-layout";
    let s = bn.scale.as_slice()?;
    let e = bn.scale_err.as_slice()?;
    let b = bn.bias.as_slice()?;
    let be = bn.bias_err.as_slice()?;
    guard.values(s)?;
    guard.values(e)?;
    guard.values(b)?;
    guard.values(be)?;
    let mut rho = 0.0f32;
    let mut delta = 0.0f32;
    for (i, ((&s, &e), &be)) in s.iter().zip(e).zip(be).enumerate() {
        if i % POLL_WORDS == 0 {
            guard.check()?;
        }
        if e != 0.0 {
            if s == 0.0 {
                return guard.refuse("zero-scale-with-error");
            } // relative form cannot encode this error
            let magnitude = ny_core::f32_to_f64_exact(s).abs();
            let error = ny_core::f32_to_f64_exact(e);
            let mut r = (error / magnitude) as f32;
            if ny_core::f32_to_f64_exact(r) * magnitude < error {
                r = ny_tensor::next_up_f32(r);
            }
            if !is_crown_coeff_safe(r)
                || r <= 0.0
                || ny_core::f32_to_f64_exact(r) * magnitude < error
            {
                return guard.refuse("relative-error-unrepresentable");
            }
            rho = rho.max(r);
        }
        delta = delta.max(be);
    }
    guard.check()?;
    Some(CertifiedWeightError {
        weight_rel_err: rho,
        bias_abs_err: delta,
    })
}

fn bn_shape(bn: &BatchNormLayer, pre: &BoundedTensor) -> Option<(usize, usize, usize)> {
    // The existing layout helper validates four value vectors. Bound that
    // otherwise hidden scan before invoking it, including malformed arrays.
    let expected = [bn.num_channels];
    if bn.num_channels == 0
        || bn.num_channels > MAX_BN_QUOTE_CHANNELS
        || [
            bn.scale.shape(),
            bn.bias.shape(),
            bn.scale_err.shape(),
            bn.bias_err.shape(),
        ]
        .iter()
        .any(|shape| *shape != expected)
    {
        return None;
    }
    let (c, epc) = bn
        .gpu_extraction_layout(pre.shape(), Some(pre.len()))
        .ok()?;
    if c == 0 || epc == 0 || c.checked_mul(epc)? != pre.len() {
        return None;
    }
    Some((c, epc, c.checked_mul(c)?.checked_add(pre.len())?))
}

fn bn_layer(
    bn: &BatchNormLayer,
    pre: &BoundedTensor,
    guard: &mut Guard<'_>,
) -> Option<GpuCrownLayer> {
    guard.at("raw-bn-layout", "layout");
    let (c, epc, _) = bn_shape(bn, pre)?;
    let cert_err = charge(bn, guard)?;
    guard.at("raw-bn-materialize", "guard-or-layout");
    guard.check()?;
    let scale = bn.scale.as_slice()?;
    let bias = bn.bias.as_slice()?;
    let mut weight = Vec::with_capacity(c.checked_mul(c)?);
    for i in 0..c.checked_mul(c)? {
        if i % POLL_WORDS == 0 {
            guard.check()?;
        }
        weight.push(if i / c == i % c { scale[i / c] } else { 0.0 });
    }
    let mut expanded = Vec::with_capacity(pre.len());
    for i in 0..pre.len() {
        if i % POLL_WORDS == 0 {
            guard.check()?;
        }
        expanded.push(bias[i / epc]);
    }
    guard.check()?;
    let layer = GpuCrownLayer::Conv2d {
        weight_col: Arc::from(weight),
        bias_expanded: Some(Arc::from(expanded)),
        out_channels: c,
        in_channels: c,
        kernel_h: 1,
        kernel_w: 1,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
        out_h: 1,
        out_w: epc,
        in_h: 1,
        in_w: epc,
        cert_err,
    };
    guard.check()?;
    Some(layer)
}

// Shape/source validation and payload count BEFORE extraction allocates a layer.
// The copied immutable weights remain exact binary32 coefficients. This is NOT
// permission to substitute a default-folded BN approximation for a raw layer.
fn layer_words(
    layer: &Layer,
    pre: &BoundedTensor,
    max_words: usize,
    validate_values: bool,
    guard: &mut Guard<'_>,
) -> Option<(usize, usize)> {
    if pre.is_empty() || pre.len() > MAX_DIM {
        return None;
    }
    if validate_values {
        guard.bounds(pre)?;
    }
    guard.failure.reason = "layer-layout-or-payload";
    match layer {
        Layer::Linear(l) => {
            let (n, m) = (l.in_features(), l.out_features());
            if n != pre.len() || m == 0 || m > MAX_DIM {
                return None;
            }
            let w = l.weight.as_slice()?;
            if n.checked_mul(m)? != w.len() {
                return None;
            }
            if w.len()
                .checked_add(l.bias.as_ref().map_or(0, |b| b.len()))?
                > max_words
            {
                return None;
            }
            if validate_values {
                guard.values(w)?;
            }
            let b = if let Some(b) = l.bias.as_ref() {
                let b = b.as_slice()?;
                if b.len() != m {
                    return None;
                }
                if validate_values {
                    guard.values(b)?;
                }
                m
            } else {
                0
            };
            Some((w.len().checked_add(b)?, m))
        }
        Layer::Conv2d(c) => {
            if c.groups != 1
                || c.dilation != (1, 1)
                || c.kernel.ndim() != 4
                || c.stride.0 == 0
                || c.stride.1 == 0
            {
                return None;
            }
            let (ic, ih, iw) = match pre.shape() {
                [ic, h, w] | [1, ic, h, w] => (*ic, *h, *w),
                _ => return None,
            };
            let k = c.kernel.shape();
            if ic == 0
                || ih == 0
                || iw == 0
                || k.contains(&0)
                || k[1] != ic
                || ic.checked_mul(ih)?.checked_mul(iw)? != pre.len()
            {
                return None;
            }
            let (oh, ow) = c.output_size(ih, iw).ok()?;
            let dim = k[0].checked_mul(oh)?.checked_mul(ow)?;
            if dim == 0 || dim > MAX_DIM {
                return None;
            }
            let w = c.kernel.as_slice()?;
            if w.len()
                .checked_add(if c.bias.is_some() { dim } else { 0 })?
                > max_words
            {
                return None;
            }
            if validate_values {
                guard.values(w)?;
            }
            let b = if let Some(b) = c.bias.as_ref() {
                if b.len() != k[0] {
                    return None;
                }
                if validate_values {
                    guard.values(b.as_slice()?)?;
                }
                dim
            } else {
                0
            };
            Some((w.len().checked_add(b)?, dim))
        }
        Layer::BatchNorm(bn) => {
            guard.at("raw-bn-layout", "layout-or-payload");
            let (_, _, words) = bn_shape(bn, pre)?;
            if words > max_words {
                return None;
            }
            if validate_values {
                charge(bn, guard)?;
            }
            Some((words, pre.len()))
        }
        Layer::ReLU(_) => {
            let words = pre.len().checked_mul(4)?;
            if words > max_words {
                return None;
            }
            Some((words, pre.len()))
        }
        Layer::Flatten(_) | Layer::Reshape(_) => Some((0, pre.len())),
        _ => None,
    }
}

fn abs_bounds(b: &BoundedTensor, guard: &mut Guard<'_>) -> Option<Vec<f32>> {
    let mut out = Vec::with_capacity(b.len());
    for (i, (&l, &u)) in b.lower().iter().zip(b.upper().iter()).enumerate() {
        if i % POLL_WORDS == 0 {
            guard.check()?;
        }
        out.push(l.abs().max(u.abs()));
    }
    Some(out)
}

/// Preserve the exact four buffers baked from this source's alpha and box.
/// Dual alpha holds TWO through-origin minorants, not a lower intercept in
/// its third vector. The checked backend routes each minorant on its own bound
/// side; the source checker below independently authenticates both and the
/// shared chord. No owner alpha, descriptor payload or arithmetic is rewritten.
///
/// Retain the existing bounded shape/bounds/poll pass and its conservative
/// conversion work reservation. No allocation or extra network fold is added.
fn checked_relu_candidate(
    layer: GpuCrownLayer,
    pre: &BoundedTensor,
    guard: &mut Guard<'_>,
) -> Option<GpuCrownLayer> {
    guard.at("relu-dual-preservation", "dual-layout");
    if let GpuCrownLayer::ActivationReluDualAlpha {
        lower_pos_slope,
        cross_slope,
        upper_neg_slope,
        cross_intercept,
        num_neurons,
    } = &layer
    {
        if *num_neurons != pre.len()
            || [
                lower_pos_slope.len(),
                cross_slope.len(),
                upper_neg_slope.len(),
                cross_intercept.len(),
            ]
            .into_iter()
            .any(|n| n != *num_neurons)
        {
            return None;
        }
        guard.bounds(pre)?;
        for i in 0..*num_neurons {
            if i % POLL_WORDS == 0 {
                guard.check()?;
            }
        }
        guard.check()?;
    }
    Some(layer)
}

fn valid_relu(layer: &GpuCrownLayer, pre: &BoundedTensor, guard: &mut Guard<'_>) -> Option<()> {
    guard.at("relu-layout-values", "descriptor-layout");
    // The third slice is interpreted only after fixing the exact variant.
    // Ordinary: lower intercept. Dual: upper-path lower minorant slope.
    let (lower_slope, upper_slope, third, upper_intercept, num_neurons, dual) = match layer {
        GpuCrownLayer::Activation {
            lower_slope,
            upper_slope,
            lower_intercept,
            upper_intercept,
            num_neurons,
        } => (
            lower_slope,
            upper_slope,
            lower_intercept,
            upper_intercept,
            *num_neurons,
            false,
        ),
        GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope,
            cross_slope,
            upper_neg_slope,
            cross_intercept,
            num_neurons,
        } => (
            lower_pos_slope,
            cross_slope,
            upper_neg_slope,
            cross_intercept,
            *num_neurons,
            true,
        ),
        _ => return None,
    };
    if num_neurons != pre.len() {
        return None;
    }
    for values in [lower_slope, upper_slope, third, upper_intercept] {
        if values.len() != pre.len() {
            return None;
        }
        guard.values(values)?;
    }
    let lo = pre.lower().as_slice()?;
    let hi = pre.upper().as_slice()?;
    for i in 0..pre.len() {
        if i % POLL_WORDS == 0 {
            guard.check()?;
        }
        // Every through-origin alpha in [0,1] is a global ReLU minorant.
        // Ordinary descriptors may additionally use a nonpositive intercept.
        if !(0.0..=1.0).contains(&lower_slope[i])
            || (!dual && third[i] > 0.0)
            || (dual && !(0.0..=1.0).contains(&third[i]))
        {
            guard.at("relu-exact-lower", "invalid-global-minorant");
            return None;
        }
        // The ReLU is convex, so an affine upper bound at BOTH interval
        // endpoints is an upper bound throughout. f32*f32 is exact in f64;
        // TwoSum retains the only addition residual. Since target is exactly
        // representable, this lexicographic comparison is exact (no tolerance).
        for x in [lo[i], hi[i]] {
            let p = ny_core::f32_to_f64_exact(upper_slope[i]) * ny_core::f32_to_f64_exact(x);
            let (s, e) = ny_core::dd::two_sum(p, ny_core::f32_to_f64_exact(upper_intercept[i]));
            let target = ny_core::f32_to_f64_exact(x.max(0.0));
            if !(s > target || (s == target && e >= 0.0)) {
                guard.at("relu-exact-upper", "endpoint-not-enclosed");
                return None;
            }
        }
    }
    Some(())
}

/// No engine/global accessor, GPU dispatch, graph clone, or bound-map clone.
/// Every path must reach the ORIGINAL network input; no frozen-node stops.
/// `poll` cancellation is sticky and all partial buffers remain private.
#[allow(clippy::too_many_arguments)]
fn extract_impl<V1: Borrow<BoundedTensor>, V2: Borrow<BoundedTensor>>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    start_node: &str,
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
    alpha_state: Option<&GraphAlphaState>,
    limits: ExtractionLimits,
    deadline: Instant,
    poll: &mut dyn FnMut() -> bool,
    materialize: bool,
) -> Option<(ExtractionQuote, Option<ChargedRawBnChain>)> {
    let mut guard = Guard::new(deadline, poll, materialize);
    guard.check()?;
    guard.at("environment", "proof-environment");
    ny_core::require_f64_interval_proof_environment().ok()?;
    guard.failure.reason = "dd-selfcheck";
    if !ny_core::dd_selfcheck::dd_selfcheck_ok() {
        return None;
    }
    guard.at("admission", "limits-or-fixed-stack");
    if fixed_stack_bytes()? > 8192 {
        return None;
    }
    if limits.max_nodes == 0
        || limits.max_nodes > MAX_NODES
        || limits.max_caller_bytes == 0
        || limits.max_caller_bytes > MAX_CALLER_BYTES
    {
        return None;
    }
    let refusal_bytes = 8192usize.checked_add(limits.max_nodes.checked_mul(size_of::<(
        &str,
        &Layer,
        &BoundedTensor,
    )>())?)?;
    if refusal_bytes > limits.max_caller_bytes {
        return None;
    }
    if input.is_empty() || input.len() > MAX_DIM {
        return None;
    }
    if materialize {
        guard.at("input-bounds", "shape-or-layout");
        guard.bounds(input)?;
    }
    // Fixed stack REFERENCES only; quote-only mode performs no heap allocation.
    let mut steps: [Option<(&str, &Layer, &BoundedTensor)>; MAX_NODES] = [None; MAX_NODES];
    let mut steps_len = 0usize;
    let mut total_weight_words = 0usize;
    let mut total_macs_per_row = 0usize;
    let mut total_affine_output_dim = 0usize;
    let mut total_activation_dim = 0usize;
    let mut total_conv_col_words_per_row = 0usize;
    let mut max_frontier_dim = input.len();
    let mut max_gemm_output_per_row = input.len();
    let mut current = start_node;
    let mut words = input.len(); // final input abs vector
    let mut names_bytes = 0usize;
    let mut charged_bn_layers = 0usize;
    let mut quoted_bn_channels = 0usize;
    let mut needed_output = None;
    let mut output_dim = 0usize;
    while current != NETWORK_INPUT {
        guard.node(steps_len, "unknown");
        guard.at("layout", "node-or-parent-or-bounds");
        guard.check()?;
        if steps_len >= limits.max_nodes
            || current.len() > 1024
            || has_active_non_relu_alpha(current, alpha_state)
        {
            return None;
        }
        let node = graph.nodes.get(current)?;
        guard.node(steps_len, layer_kind(&node.layer));
        let parent = node.require_unary_input().ok()?;
        let pre = resolve_pre(input, parent, crown_bounds, ibp_bounds)?;
        if let Layer::BatchNorm(bn) = &node.layer {
            quoted_bn_channels = quoted_bn_channels.checked_add(bn.num_channels)?;
            if quoted_bn_channels > MAX_BN_QUOTE_CHANNELS {
                return None;
            }
        }
        let available = limits
            .max_caller_bytes
            .checked_sub(8192)?
            .checked_div(8)?
            .checked_sub(words)?;
        guard.at("source-layout-values", "shape-or-values");
        let (layer_words, dim) = layer_words(&node.layer, pre, available, materialize, &mut guard)?;
        guard.at("layout", "dimension-or-quote-arithmetic");
        if needed_output.is_some_and(|n| n != dim) {
            return None;
        }
        if steps_len == 0 {
            output_dim = dim;
        }
        needed_output = Some(pre.len());
        words = words.checked_add(layer_words)?;
        if matches!(node.layer, Layer::ReLU(_)) {
            words = words.checked_add(pre.len())?;
            names_bytes = names_bytes.checked_add(current.len())?;
        }
        if matches!(node.layer, Layer::BatchNorm(_)) {
            charged_bn_layers += 1;
        }
        let (weights, gemm_out, macs) = match &node.layer {
            Layer::Linear(l) => (l.weight.len(), l.in_features(), l.weight.len()),
            Layer::Conv2d(c) => {
                let shape = pre.shape();
                let (oh, ow) = c
                    .output_size(shape[shape.len() - 2], shape[shape.len() - 1])
                    .ok()?;
                let k = c.kernel.shape();
                (
                    c.kernel.len(),
                    oh.checked_mul(ow)?
                        .checked_mul(k[1])?
                        .checked_mul(k[2])?
                        .checked_mul(k[3])?,
                    oh.checked_mul(ow)?.checked_mul(c.kernel.len())?,
                )
            }
            Layer::BatchNorm(bn) => (
                bn.num_channels.checked_mul(bn.num_channels)?,
                pre.len(),
                bn.num_channels.checked_mul(pre.len())?,
            ),
            _ => (0, 0, 0),
        };
        if weights != 0 {
            total_affine_output_dim = total_affine_output_dim.checked_add(dim)?;
        }
        if matches!(node.layer, Layer::ReLU(_)) {
            total_activation_dim = total_activation_dim.checked_add(dim)?;
        }
        if matches!(node.layer, Layer::Conv2d(_) | Layer::BatchNorm(_)) {
            total_conv_col_words_per_row = total_conv_col_words_per_row.checked_add(gemm_out)?;
        }
        total_weight_words = total_weight_words.checked_add(weights)?;
        total_macs_per_row = total_macs_per_row.checked_add(macs)?;
        max_frontier_dim = max_frontier_dim.max(pre.len()).max(dim);
        max_gemm_output_per_row = max_gemm_output_per_row.max(gemm_out);
        steps[steps_len] = Some((current, &node.layer, pre));
        steps_len += 1;
        current = parent;
    }
    if steps_len == 0 || needed_output != Some(input.len()) {
        return None;
    }
    guard.node(MAX_NODES, "none");
    guard.at("payload-admission", "headers-or-payload");
    // Two copies of every payload conservatively include Vec→Arc overlap and
    // geometric Vec capacity. Headers/references/name bytes are charged too.
    let headers = size_of::<GpuCrownLayer>()
        .checked_add(4 * size_of::<Vec<f32>>())?
        .checked_add(4 * size_of::<String>())?
        .checked_add(4 * size_of::<usize>())?;
    if headers > 512 {
        return None;
    } // quoted metadata ceiling, checked on the actual ABI
    let peak = words
        .checked_mul(8)?
        .checked_add(names_bytes.checked_mul(2)?)?
        .checked_add(limits.max_nodes.checked_mul(headers)?.checked_mul(2)?)?
        .checked_add(8192)?;
    if peak > limits.max_caller_bytes {
        return None;
    }
    let quote = ExtractionQuote {
        nodes: steps_len,
        charged_bn_layers,
        float_words: words,
        input_dim: input.len(),
        output_dim,
        total_weight_words,
        total_macs_per_row,
        total_affine_output_dim,
        total_activation_dim,
        total_conv_col_words_per_row,
        max_frontier_dim,
        max_gemm_output_per_row,
        caller_owned_peak_bytes: peak,
    };
    guard.at("scan-final-guard", "guard");
    guard.check()?;
    if !materialize {
        return Some((quote, None));
    }
    let mut chain = Vec::with_capacity(steps_len);
    let mut names = Vec::with_capacity(steps_len);
    let mut node_abs = Vec::with_capacity(steps_len);
    for (ordinal, (name, layer, pre)) in steps[..steps_len].iter().copied().flatten().enumerate() {
        guard.node(ordinal, layer_kind(layer));
        guard.at("materialize-node", "guard");
        guard.check()?;
        match layer {
            Layer::BatchNorm(bn) => chain.push(bn_layer(bn, pre, &mut guard)?),
            Layer::ReLU(_) => {
                guard.at("relu-bake", "bake-refused");
                let relu = bake_relu_layer(name, pre, alpha_state)?;
                let relu = checked_relu_candidate(relu, pre, &mut guard)?;
                valid_relu(&relu, pre, &mut guard)?;
                chain.push(relu);
                names.push(name.to_owned());
                guard.at("relu-abs-bounds", "guard");
                node_abs.push(abs_bounds(pre, &mut guard)?);
            }
            _ => {
                guard.at("ordinary-layer", "extraction-refused");
                try_extract_single_gpu_layer(layer, pre, &mut chain)?;
            }
        }
        guard.at("node-final-guard", "guard");
        guard.check()?;
    }
    guard.node(MAX_NODES, "none");
    guard.at("chain-completion", "empty-chain");
    if chain.is_empty() {
        return None;
    }
    guard.at("input-abs-bounds", "guard");
    let frontier = abs_bounds(input, &mut guard)?;
    guard.at("final-guard", "guard");
    guard.check()?;
    guard.completed = true;
    Some((
        quote,
        Some(ChargedRawBnChain {
            segments: vec![GpuResnetSegment::Chain(chain)],
            relu_names: names,
            frontier_abs: vec![frontier],
            node_abs,
            input_dim: input.len(),
            output_dim,
            quote,
        }),
    ))
}

/// Shape quote: no descriptor/bound-map allocation or full weight scan.
/// The existing BN layout helper also validates at most8192 channel entries
/// per chain (four vectors); this bounded setup scan precedes full admission.
/// Source values and error authority are rechecked after admission. The fixed
/// stack/refusal scratch is
/// covered by the 8192-byte quote allowance; a caller can quote both sequential
/// stages before acquiring its aggregate extraction+backend memory lease.
#[allow(clippy::too_many_arguments)]
pub(crate) fn quote_charged_raw_bn_chain<V1: Borrow<BoundedTensor>, V2: Borrow<BoundedTensor>>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    start_node: &str,
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
    alpha_state: Option<&GraphAlphaState>,
    limits: ExtractionLimits,
    deadline: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<ExtractionQuote> {
    extract_impl(
        graph,
        input,
        start_node,
        crown_bounds,
        ibp_bounds,
        alpha_state,
        limits,
        deadline,
        poll,
        false,
    )
    .map(|x| x.0)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn extract_charged_raw_bn_chain<V1: Borrow<BoundedTensor>, V2: Borrow<BoundedTensor>>(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    start_node: &str,
    crown_bounds: &HashMap<String, V1>,
    ibp_bounds: &HashMap<String, V2>,
    alpha_state: Option<&GraphAlphaState>,
    limits: ExtractionLimits,
    deadline: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<ChargedRawBnChain> {
    extract_impl(
        graph,
        input,
        start_node,
        crown_bounds,
        ibp_bounds,
        alpha_state,
        limits,
        deadline,
        poll,
        true,
    )
    .and_then(|x| x.1)
}

#[cfg(test)]
#[path = "raw_bn_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "raw_relu_fallback_tests.rs"]
mod relu_fallback_tests;
