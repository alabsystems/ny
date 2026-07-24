// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Layer query methods: type names, classification predicates, batched CROWN support checks.

use super::{for_each_elementwise_activation, Layer};
use crate::layers::common::BoundPropagation;

impl Layer {
    /// Get a string describing the layer type.
    pub fn layer_type(&self) -> &'static str {
        match self {
            Layer::Linear(_) => "Linear",
            Layer::Conv1d(_) => "Conv1d",
            Layer::Conv2d(_) => "Conv2d",
            Layer::ConvTranspose1d(_) => "ConvTranspose1d",
            Layer::ConvTranspose2d(_) => "ConvTranspose2d",
            Layer::AveragePool(_) => "AveragePool",
            Layer::MaxPool2d(_) => "MaxPool2d",
            Layer::ReLU(_) => "ReLU",
            Layer::LeakyReLU(_) => "LeakyReLU",
            Layer::Clip(_) => "Clip",
            Layer::Elu(_) => "Elu",
            Layer::Selu(_) => "Selu",
            Layer::PRelu(_) => "PRelu",
            Layer::HardSigmoid(_) => "HardSigmoid",
            Layer::HardSwish(_) => "HardSwish",
            Layer::SiLU(_) => "SiLU",
            Layer::Exp(_) => "Exp",
            Layer::Log(_) => "Log",
            Layer::Celu(_) => "Celu",
            Layer::Mish(_) => "Mish",
            Layer::LogSoftmax(_) => "LogSoftmax",
            Layer::LogSumExp(_) => "LogSumExp",
            Layer::ThresholdedRelu(_) => "ThresholdedRelu",
            Layer::Shrink(_) => "Shrink",
            Layer::Softsign(_) => "Softsign",
            Layer::Snake(_) => "Snake",
            Layer::Softmax(_) => "Softmax",
            Layer::CausalSoftmax(_) => "CausalSoftmax",
            Layer::SelfAttention(_) => "SelfAttention",
            Layer::GELU(_) => "GELU",
            Layer::LayerNorm(_) => "LayerNorm",
            Layer::RmsNorm(_) => "RmsNorm",
            Layer::InstanceNorm1d(_) => "InstanceNorm1d",
            Layer::GroupNorm(_) => "GroupNorm",
            Layer::AdaIN1d(_) => "AdaIN1d",
            Layer::BatchNorm(_) => "BatchNorm",
            Layer::MatMul(_) => "MatMul",
            Layer::MulBinary(_) => "MulBinary",
            Layer::Add(_) => "Add",
            Layer::Concat(_) => "Concat",
            Layer::Sub(_) => "Sub",
            Layer::Div(_) => "Div",
            Layer::Atan2(_) => "Atan2",
            Layer::BilinearCrown(_) => "BilinearCrown",
            Layer::MinBinary(_) => "MinBinary",
            Layer::MaxBinary(_) => "MaxBinary",
            Layer::AddConstant(_) => "AddConstant",
            Layer::Transpose(_) => "Transpose",
            Layer::Reshape(_) => "Reshape",
            Layer::Flatten(_) => "Flatten",
            Layer::Pad(_) => "Pad",
            Layer::MulConstant(_) => "MulConstant",
            Layer::Abs(_) => "Abs",
            Layer::Sqrt(_) => "Sqrt",
            Layer::DivConstant(_) => "DivConstant",
            Layer::SubConstant(_) => "SubConstant",
            Layer::PowConstant(_) => "PowConstant",
            Layer::ReduceMean(_) => "ReduceMean",
            Layer::ReduceSum(_) => "ReduceSum",
            Layer::ReduceMax(_) => "ReduceMax",
            Layer::ReduceMin(_) => "ReduceMin",
            Layer::Topk(_) => "Topk",
            Layer::ArgMax(_) => "ArgMax",
            Layer::ArgMin(_) => "ArgMin",
            Layer::ArgSort(_) => "ArgSort",
            Layer::CumSum(_) => "CumSum",
            Layer::Tanh(_) => "Tanh",
            Layer::Sigmoid(_) => "Sigmoid",
            Layer::Softplus(_) => "Softplus",
            Layer::Sin(_) => "Sin",
            Layer::Cos(_) => "Cos",
            Layer::Tan(_) => "Tan",
            Layer::Arctan(_) => "Arctan",
            Layer::RoPE(_) => "RoPE",
            Layer::Tile(_) => "Tile",
            Layer::ExpandLikeLastAxis(_) => "ExpandLikeLastAxis",
            Layer::Gather(_) => "Gather",
            Layer::ScatterAdd(_) => "ScatterAdd",
            Layer::IndexAdd(_) => "IndexAdd",
            Layer::ScatterNd(_) => "ScatterND",
            Layer::Slice(_) => "Slice",
            Layer::Squeeze(_) => "Squeeze",
            Layer::Unsqueeze(_) => "Unsqueeze",
            Layer::Resize(_) => "Resize",
            Layer::Compare(_) => "Compare",
            Layer::CompareTensor(_) => "CompareTensor",
            Layer::Where(_) => "Where",
            Layer::NonZero(_) => "NonZero",
            Layer::Floor(_) => "Floor",
            Layer::Ceil(_) => "Ceil",
            Layer::Round(_) => "Round",
            Layer::Trunc(_) => "Trunc",
            Layer::Sign(_) => "Sign",
            Layer::Reciprocal(_) => "Reciprocal",
            Layer::SkipMerge(_) => "SkipMerge",
            Layer::OpaqueSkip(_) => "OpaqueSkip",
            Layer::QdqPerturbation(_) => "QdqPerturbation",
        }
    }

    /// Check if this layer is a binary operation (requires two inputs).
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            Layer::MatMul(_)
                | Layer::MulBinary(_)
                | Layer::Add(_)
                | Layer::Concat(_)
                | Layer::Sub(_)
                | Layer::Div(_)
                | Layer::Atan2(_)
                | Layer::BilinearCrown(_)
                | Layer::MinBinary(_)
                | Layer::MaxBinary(_)
                | Layer::ExpandLikeLastAxis(_)
                | Layer::CompareTensor(_)
        ) || matches!(self, Layer::ScatterNd(scatter) if scatter.activation_input_count() == 2)
            || matches!(self, Layer::ScatterAdd(scatter) if scatter.activation_input_count() == 2)
            || matches!(self, Layer::IndexAdd(index) if index.activation_input_count() == 2)
    }

    /// Check if this layer is a ternary operation (requires three inputs).
    pub fn is_ternary(&self) -> bool {
        matches!(self, Layer::Where(_) | Layer::SelfAttention(_))
            || matches!(self, Layer::ScatterNd(scatter) if scatter.activation_input_count() == 3)
            || matches!(self, Layer::ScatterAdd(scatter) if scatter.activation_input_count() == 3)
            || matches!(self, Layer::IndexAdd(index) if index.activation_input_count() == 3)
            || matches!(self, Layer::AdaIN1d(adain) if adain.requires_style_inputs())
    }

    /// Minimum number of graph-edge inputs this layer requires.
    ///
    /// Used by `GraphNode::new()` to catch arity mismatches at construction time
    /// instead of deferring to propagation-time `require_*_inputs()` checks (#2481).
    ///
    /// Special cases:
    /// - `Where`: returns 1 because the embedded-constants variant needs only the
    ///   condition input. The 3-input variant is validated at propagation time.
    /// - `Concat`: returns 1 when the layer has embedded constant inputs (e.g.,
    ///   ViT CLS token from ConstantOfShape). The N-ary variant validates at
    ///   propagation time. Returns 2 otherwise.
    /// - `ScatterND`: returns the number of non-embedded activation inputs.
    /// - `OpaqueSkip`: returns 1 (propagates through first input only).
    pub fn min_inputs(&self) -> usize {
        if let Layer::AdaIN1d(adain) = self {
            // Variable-style AdaIN always needs exactly 3 inputs (x, style_gamma, style_beta).
            // Fixed-style AdaIN is unary (style embedded in the layer).
            if adain.requires_style_inputs() {
                3
            } else {
                1
            }
        } else if let Layer::ScatterNd(scatter) = self {
            scatter.activation_input_count()
        } else if let Layer::ScatterAdd(scatter) = self {
            scatter.activation_input_count()
        } else if let Layer::IndexAdd(index) = self {
            index.activation_input_count()
        } else if self.is_ternary() {
            // Where with embedded constants needs only 1 input, but
            // SelfAttention always needs 3. Return 1 as the conservative
            // minimum — propagation-time checks enforce the exact count.
            1
        } else if let Layer::Concat(c) = self {
            // Concat with embedded constants can have fewer graph inputs
            // (e.g., ViT CLS token concatenated with patch embeddings).
            // Fully-constant Concat (all ONNX inputs embedded) has zero
            // graph-level inputs — valid after ONNX lowering (#4112).
            if c.constant_inputs
                .as_ref()
                .is_some_and(|ci| !ci.is_empty() && ci.iter().all(|x| x.is_some()))
            {
                0
            } else if c.constant_inputs.is_some() {
                1
            } else {
                2
            }
        } else if self.is_binary() {
            2
        } else {
            1
        }
    }

    /// Check if this layer is an elementwise activation with standard batched CROWN signature.
    ///
    /// These layers all share the same `propagate_linear_batched_with_bounds(&BatchedLinearBounds,
    /// &BoundedTensor) -> Result<BatchedLinearBounds>` calling convention, enabling unified
    /// dispatch in both `Layer::propagate_crown_backward_batched` and
    /// `GraphNetwork::propagate_crown_batched_with_relaxation`.
    ///
    /// Excludes Softmax (extra `soundness_mode` parameter) and LayerNorm (complex non-elementwise).
    /// Generated from `for_each_elementwise_activation!` — the single source of truth.
    pub fn is_elementwise_activation(&self) -> bool {
        macro_rules! check_elementwise {
            ($($Variant:ident),*) => {
                matches!(self, $(Layer::$Variant(_))|*)
            };
        }
        for_each_elementwise_activation!(check_elementwise)
    }

    /// Whether this layer's CROWN backward soundly preserves a certified
    /// coefficient-error interval carried on the incoming [`LinearBounds`]
    /// (#vnncomp-aw-soundness).
    ///
    /// Single source of truth for the err-soundness gate used by both the
    /// sequential and graph CROWN backward dispatchers. The set is:
    /// - **err-producing layers** `Linear`, `Conv2d` (their backward both
    ///   computes the error AND propagates any incoming error), and every
    ///   **element-wise activation** (all flow through
    ///   `crown_elementwise_backward`, taught to propagate the error);
    /// - **pure pass-through reshape layers** `Flatten`, `Reshape`, `Squeeze`,
    ///   `Unsqueeze` whose `propagate_linear` returns `Cow::Borrowed(bounds)`,
    ///   leaving the coefficient *and* error matrices untouched.
    ///
    /// Every other layer reconstructs its `LinearBounds` without the error
    /// matrices, so the dispatcher must degrade the network to IBP (always
    /// sound) rather than let it silently drop the soundness penalty.
    pub fn propagates_coeff_err(&self) -> bool {
        self.is_elementwise_activation()
            || matches!(
                self,
                Layer::Linear(_)
                    | Layer::Conv2d(_)
                    // f32-arithmetic ops whose backward emits BOTH a fresh γ_n·S
                    // (or per-coeff ULP) coefficient error AND a prop term that
                    // re-propagates incoming err (#vnncomp-aw-soundness). These
                    // MUST NOT stay in `is_exact_linear_coeff_err_carrier` (whose
                    // carrier path OVERWRITES the fresh err with carried-incoming).
                    | Layer::Conv1d(_)
                    | Layer::ConvTranspose1d(_)
                    | Layer::ConvTranspose2d(_)
                    // BatchNorm (#cgan-conv-err-compose): its CROWN backward is an
                    // EXACT per-column scaling that now propagates incoming err as
                    // `e·|scale|` PLUS its fresh per-coeff ULP term (crown_scalar.rs).
                    // It must NOT be a carrier (the carrier path would OVERWRITE the
                    // fresh err), and it must no longer take the dispatcher's
                    // output-box discharge — that fold converted a u-scale RELATIVE
                    // coefficient error into an ABSOLUTE width penalty at
                    // intermediate-box magnitude, the dominant cGAN CROWN looseness
                    // (BN_5 2.05× / Conv_19 404× vs the exact affine composition).
                    | Layer::BatchNorm(_)
                    | Layer::MulConstant(_)
                    | Layer::DivConstant(_)
                    | Layer::Tile(_)
                    // Gather scatter-add: on DUPLICATE indices its backward
                    // f32-accumulates k coeffs into one input column and emits a fresh
                    // gamma_k*S coefficient error PLUS a prop term that re-propagates
                    // incoming err (#vnncomp-aw-soundness). Like Tile it MUST NOT stay
                    // in `is_exact_linear_coeff_err_carrier` (whose carrier path would
                    // OVERWRITE the fresh gamma_k*S via `attach_err_from_carried`).
                    | Layer::Gather(_)
                    // Resize (nearest): its backward scatter-adds scale_h*scale_w DUPLICATE
                    // output coeffs into each input cell (gather-class), emitting a fresh
                    // gamma_k*S err + prop — same reason as Gather/Tile. (#vnncomp-aw-soundness)
                    | Layer::Resize(_)
                    // Pad: Reflect mode duplicate-fan-in scatter-add (gamma_k*S) + Constant
                    // mode directed bias fold both emit fresh err — must NOT stay a carrier.
                    | Layer::Pad(_)
                    // NOTE: AveragePool is DELIBERATELY NOT here. It is a SCATTER-type
                    // linear op (each input column sums several output-coeff errors), so
                    // propagating its incoming err incurs a triangle-inequality loss:
                    // `max|outbox_c| ≤ Σ_{j∈win_c} weight·max|inbox_j|`, hence discharging
                    // over AvgPool's OWN (CROWN-tightened) output box is provably ≤
                    // propagate-then-discharge-later. Measured: even the EXACT per-column
                    // composition (average.rs) is 1.10× (2×2/s2) to 1.26× (3×3/s1) LOOSER
                    // than discharge (test_avgpool_carried_coeff_err_encloses_and_ab_width).
                    // This is the SAME reason Conv2d/ConvTranspose eager-discharge in the
                    // dispatcher. The BatchNorm win (eb651c59) transfers ONLY to DIAGONAL /
                    // exact-1:1-map linear ops (BN, Mul/DivConstant) where err composition
                    // is exact — NOT to scatter/averaging ops. Keep AvgPool as a
                    // discharge (non-carrier) op.
                    | Layer::Flatten(_)
                    | Layer::Reshape(_)
                    | Layer::Squeeze(_)
                    | Layer::Unsqueeze(_)
            )
    }

    /// Whether this layer is an EXACT linear graph op whose CROWN backward applies
    /// a fixed linear column transform `T` to the coefficient matrix (permute /
    /// select / tile / scale-by-constant / split / `A→bias` fold) and introduces
    /// NO nonlinear relaxation (#vnncomp-aw-soundness).
    ///
    /// For these ops the certified coefficient error is propagated by re-running
    /// the SAME backward transform on the incoming error matrices (taken as
    /// non-negative coefficients via [`LinearBounds::coeff_err_carrier`]): the
    /// carried result's coefficient magnitudes bound `T_abs(err_in)` and its bias
    /// magnitudes bound the error the op moved into the bias. The dispatcher uses
    /// this to carry the error soundly (and tightly) instead of degrading the
    /// affected rows to `[-inf, +inf]` via `discharge_coeff_err_to_conservative`.
    ///
    /// EXCLUDES nonlinear / relaxation-producing layers (ReLU, activations handled
    /// by `propagates_coeff_err`, MatMul/Mul/Bilinear McCormick, Where, Div, etc.)
    /// whose backward is not a fixed linear transform of the coefficient matrix.
    pub fn is_exact_linear_coeff_err_carrier(&self) -> bool {
        matches!(
            self,
            Layer::Add(_)
                | Layer::Sub(_)
                | Layer::Concat(_)
                | Layer::Slice(_)
                | Layer::Transpose(_)
                // Gather AND Pad moved to `propagates_coeff_err`: their backends emit a fresh
                // coefficient error (Gather/Pad-Reflect duplicate scatter-add gamma_k*S;
                // Pad-Constant directed bias fold) that the carrier path would clobber.
                // (#vnncomp-aw-soundness)
                | Layer::AddConstant(_)
                | Layer::SubConstant(_)
        )
    }

    /// Check if this layer supports batched CROWN propagation (for Network gate checks).
    ///
    /// Used by `Network::propagate_crown_batched` and `StreamingVerifier` gate functions
    /// to decide whether to use batched CROWN or fall back to regular CROWN.
    /// This is the single source of truth for the batched CROWN allow-list, replacing
    /// duplicated `matches!` blocks across 4 gate functions (#1753).
    pub fn supports_batched_crown(&self) -> bool {
        self.is_elementwise_activation()
            || matches!(
                self,
                Layer::Linear(_)
                    | Layer::Softmax(_)
                    | Layer::CausalSoftmax(_)
                    | Layer::LogSoftmax(_)
                    | Layer::LogSumExp(_)
                    | Layer::LayerNorm(_)
                    | Layer::RmsNorm(_)
                    | Layer::InstanceNorm1d(_)
                    | Layer::GroupNorm(_)
                    | Layer::BatchNorm(_)
                    | Layer::Conv1d(_)
                    | Layer::ConvTranspose1d(_)
                    | Layer::Conv2d(_)
                    | Layer::ConvTranspose2d(_)
                    | Layer::Flatten(_)
                    | Layer::Pad(_)
                    | Layer::Reshape(_)
                    | Layer::Squeeze(_)
                    | Layer::Unsqueeze(_)
                    | Layer::Resize(_)
                    | Layer::Transpose(_)
                    | Layer::AddConstant(_)
                    | Layer::SubConstant(_)
                    | Layer::MulConstant(_)
                    | Layer::DivConstant(_)
                    | Layer::RoPE(_)
                    | Layer::Slice(_)
                    | Layer::Tile(_)
                    | Layer::ReduceSum(_)
                    | Layer::ReduceMean(_)
                    | Layer::ReduceMax(_)
                    | Layer::ReduceMin(_)
                    | Layer::QdqPerturbation(_)
            )
            // Fixed-style AdaIN supports batched CROWN; variable-style does not.
            || matches!(self, Layer::AdaIN1d(adain) if !adain.requires_style_inputs())
    }

    /// Which producer inputs may need tightened intermediate bounds.
    ///
    /// Returns the input indices that feed nonlinear relaxation surfaces
    /// requiring pre-activation bounds for CROWN backward. Used by the
    /// graph-alpha collector to decide which upstream nodes are worth
    /// tightening with CROWN-IBP, avoiding O(N²) backward passes on
    /// nodes that no downstream consumer needs.
    ///
    /// - Unary nonlinear layers: `[0]` (need bounds on their single input)
    /// - Binary relaxation layers: `[0, 1]` (both inputs may be perturbed)
    /// - Exact/linear/shape layers: `[]` (no relaxation, no bounds needed)
    ///
    /// Reuses `requires_pre_activation_bounds()` for the unary case and
    /// special-cases only the binary relaxation surfaces plus exact shape-aware
    /// layers that use that trait for shape plumbing rather than nonlinear
    /// relaxation. This keeps demand metadata aligned with "which producers
    /// need tightening" instead of "which backward path needs the input shape."
    ///
    /// Reference: alpha-beta-CROWN `update_requires_input_bounds` per-operator.
    /// Source: `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/base.py:143`
    pub fn required_input_bound_indices(&self) -> &'static [usize] {
        match self {
            // Binary relaxation layers: both inputs may need tightening
            // when both sides are perturbed.
            Layer::MatMul(_) | Layer::BilinearCrown(_) | Layer::MulBinary(_) => &[0, 1],
            // Exact linear operators that borrow pre-activation bounds only
            // to recover input shape/channel metadata for backward.
            Layer::AveragePool(_)
            | Layer::BatchNorm(_)
            | Layer::Pad(_)
            | Layer::Resize(_)
            | Layer::Slice(_)
            | Layer::ReduceMean(_)
            | Layer::ReduceSum(_)
            | Layer::CumSum(_) => &[],
            // Unary nonlinear layers derive demand from the existing
            // pre-activation bounds contract.
            _ if self.requires_pre_activation_bounds() => &[0],
            // Linear, shape, and constant layers: exact backward, no
            // relaxation, no intermediate bounds needed.
            _ => &[],
        }
    }

    /// Check if this layer supports batched CROWN when Conv2d is present.
    ///
    /// Conv2d networks flatten spatial dimensions, which is incompatible with
    /// Softmax and LayerNorm. All other batched-CROWN-supported layers work.
    pub fn supports_batched_crown_with_conv2d(&self) -> bool {
        self.supports_batched_crown()
            && !matches!(
                self,
                Layer::Softmax(_)
                    | Layer::CausalSoftmax(_)
                    | Layer::LogSoftmax(_)
                    | Layer::LogSumExp(_)
                    | Layer::LayerNorm(_)
                    | Layer::RmsNorm(_)
                    | Layer::InstanceNorm1d(_)
                    | Layer::GroupNorm(_)
                    | Layer::AdaIN1d(_)
            )
    }
}
