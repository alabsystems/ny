// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Layer query methods: type names, classification predicates, batched CROWN support checks.

use super::{for_each_elementwise_activation, Layer};
use crate::layers::common::BoundPropagation;

impl Layer {
    /// Diagnostic-only storage cannot enter any proof or precomputed-bound
    /// fallback, even when that fallback would bypass the layer dispatcher.
    pub fn is_diagnostic_point_only(&self) -> bool {
        matches!(self, Layer::PointLinear(_))
    }

    /// Get a string describing the layer type.
    pub fn layer_type(&self) -> &'static str {
        match self {
            Layer::Linear(_) => "Linear",
            Layer::PointLinear(_) => "PointLinear",
            Layer::IntervalConst(ic) => ic.kind_name(),
            Layer::Conv1d(_) => "Conv1d",
            Layer::Conv2d(_) => "Conv2d",
            Layer::Conv3d(_) => "Conv3d",
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
            Layer::Erf(_) => "Erf",
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
            Layer::EpsilonExpand(_) => "EpsilonExpand",
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
    /// instead of deferring to propagation-time `require_*_inputs()` checks.
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
            // graph-level inputs — valid after ONNX lowering.
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
                    // Interval-valued frozen constant: its backward RUNS the
                    // Linear backward (which propagates the incoming err and
                    // emits its own fresh gamma_n*S) and then ADDS the enclosure
                    // term `(|A| + err_in)*R` on top. It therefore belongs here,
                    // beside `Linear`, and it MUST NOT be an
                    // `is_exact_linear_coeff_err_carrier`: that path OVERWRITES
                    // the fresh err via `attach_err_from_carried`, which would
                    // silently delete the enclosure and leave the MIDPOINT
                    // product standing as if it were exact — the unsound point
                    // fold, one layer down.
                    | Layer::IntervalConst(_)
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
                    // AddConstant (#vnncomp-aw-soundness, acasxu prop_1 recovery): its
                    // CROWN backward passes `A` through UNTOUCHED (so an incoming
                    // certificate `|A_stored − A_true| ≤ e` holds verbatim on the
                    // result) and charges the `A→bias` fold's error by MAGNITUDE — the
                    // bias is widened OUTWARD by `Σ_j e_j·|c_j|`, Pad's certified
                    // constant-fold idiom (see add_constant/crown.rs). That directed
                    // charge DOMINATES the signed carrier fold `|Σ_j e_j·c_j|` on every
                    // constant (equal exactly when the constant is sign-uniform), so
                    // the mixed-sign carrier eviction below stays CLOSED while the
                    // sequential lane keeps a FINITE row instead of the
                    // `discharge_coeff_err_to_conservative` vacuous-row degrade —
                    // which cost acasxu's single-output prop_1 family its sub-second
                    // unsat, because every acasxu FC bias is mixed-sign. Scalar
                    // dispatchers check THIS predicate first, so the sign-uniform
                    // carrier admission below now serves only the BATCHED lane (whose
                    // carrier ADDs the carried err instead of overwriting — see
                    // layers/layer_enum/dispatch.rs).
                    //
                    // SubConstant deliberately does NOT join it here: its scalar
                    // backward's broadcast column-REDUCTION path
                    // (`propagate_linear_broadcast_backward`) DOES propagate the
                    // incoming err (the p-term, sub_constant.rs:146-166) — but that
                    // path never charges the bias fold with the magnitude term
                    // Σ_j e_j·|c_j|, so declaring SubConstant a direct propagator
                    // would UNDER-CHARGE the folded bias on that path. It keeps the
                    // carrier/discharge policy (sound, at worst looser).
                    | Layer::AddConstant(_)
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
    ///
    /// It ALSO excludes an `A→bias` fold whose constant can CANCEL under the
    /// carrier — see `constant_fold_cannot_cancel` below, and the note in the
    /// body. That admission is per-INSTANCE, not per-variant.
    pub fn is_exact_linear_coeff_err_carrier(&self) -> bool {
        // `A→bias` FOLD OPS ARE ADMITTED ONLY WHEN THE FOLD CANNOT CANCEL
        // (#vnncomp-aw-soundness).
        //
        // AddConstant/SubConstant are the only members of this set whose backward
        // moves coefficients INTO THE BIAS (`b' = b ± A @ c`; see
        // `AddConstantLayer::crown_backward_with_operand_shape` and
        // `SubConstantLayer::propagate_linear` /
        // `propagate_linear_broadcast_backward`, all of which reach the fold
        // through `arithmetic::common::dot_bias_f64` with the SIGNED `c`).
        //
        // Run on `LinearBounds::coeff_err_carrier` — whose coefficients are the
        // non-negative `e = |err|` and whose bias is zero — that fold produces
        // `Σ_j e_j·c_j`, and `LinearBounds::attach_err_from_carried` then takes
        // `max(|B_lo|, |B_up|)` as the bias widening. The charge an error term
        // requires is `Σ_j e_j·|c_j|`: an error accumulates MAGNITUDES, because
        // cancellation in the VALUE does not cancel the ERROR. The two differ
        // without limit on a mixed-sign `c` — the ordinary CNN channel bias. With
        // `e = [1, 1]` and `c = [+1, -1]` the required charge is 2 and the carrier
        // publishes `|1·(+1) + 1·(-1)| = 0`: a bound TIGHTER than the arithmetic
        // justifies, i.e. a false `unsat`, not a loose one.
        //
        // The sibling that gets this right is Pad's constant fold
        // (`layers/transform/pad.rs`):
        //
        //     const_lo_err[row] += (e[[row, out_flat]] as f64).abs() * v.abs();
        //
        // — MAGNITUDE on both factors, applied outward (`-` into the lower bias,
        // `+` into the upper) — and Pad was moved OUT of this set into
        // `propagates_coeff_err()` precisely so the carrier could not clobber that
        // charge.
        //
        // The VALUE fold is untouched: these layers still shift the bias by the
        // signed `A @ c`, so the certified function is unchanged. Only the ERROR
        // policy moves. Where the constant is SIGN-UNIFORM the carrier's charge is
        // already the magnitude charge (`e ≥ 0`, so every term of `Σ_j e_j·c_j`
        // shares one sign and `|Σ_j e_j·c_j| = Σ_j e_j·|c_j|` identically), and
        // the tight carrier path is kept. Every other case leaves the set and
        // takes the dispatchers' PRECISE discharge instead —
        // `fold_coeff_err_into_bias` over the node's own output box in
        // `network/backward_dispatch/dispatch.rs` and
        // `layers/layer_enum/dispatch.rs`, `discharge_coeff_err_to_conservative`
        // in `network/core/sequential/crown/patches_step.rs`. Looser, never wrong.
        if let Layer::AddConstant(l) = self {
            return constant_fold_cannot_cancel(l.constant());
        }
        if let Layer::SubConstant(l) = self {
            // `reverse` (`y = c - x`) negates the coefficient matrix wholesale and
            // leaves the fold's summands untouched, so the same test applies.
            return constant_fold_cannot_cancel(l.constant());
        }
        // Gather AND Pad moved to `propagates_coeff_err`: their backends emit a
        // fresh coefficient error (Gather/Pad-Reflect duplicate scatter-add
        // gamma_k*S; Pad-Constant directed bias fold) that the carrier path would
        // clobber. (#vnncomp-aw-soundness)
        matches!(
            self,
            Layer::Add(_)
                | Layer::Sub(_)
                | Layer::Concat(_)
                | Layer::Slice(_)
                | Layer::Transpose(_)
        )
    }

    /// Check if this layer supports batched CROWN propagation (for Network gate checks).
    ///
    /// Used by `Network::propagate_crown_batched` and `StreamingVerifier` gate functions
    /// to decide whether to use batched CROWN or fall back to regular CROWN.
    /// This is the single source of truth for the batched CROWN allow-list, replacing
    /// duplicated `matches!` blocks across 4 gate functions.
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
                    // CumSum HAS a real batched CROWN backward
                    // (`CumsumLayer::propagate_linear_batched`, dispatched from
                    // `layer_enum/dispatch.rs`): the grouped last-axis prefix/suffix
                    // scan, with its own certified `γ_axislen·S` coefficient error.
                    // It was missing from this allow-list, so a single CumSum node
                    // disqualified an ENTIRE network from the batched path even
                    // though every other layer was supported. Refusals inside the
                    // backward (non-last-axis layout, shape-changing bounds) surface
                    // as `UnsupportedOp`, and every caller of this gate treats that
                    // as "fall back to the scalar CROWN path" — so listing CumSum
                    // here fails OPEN to the previous behaviour, never to a verdict.
                    | Layer::CumSum(_)
                    | Layer::QdqPerturbation(_)
                    | Layer::EpsilonExpand(_)
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
            // Point-only has no backward adapter. Empty demand does not grant
            // exact-linear authority; every proof dispatcher refuses it.
            Layer::PointLinear(_) => &[],
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

/// Whether a constant-shift layer's `A→bias` fold is free of SIGN CANCELLATION
/// under the non-negative coefficient-error carrier (#vnncomp-aw-soundness).
///
/// `LinearBounds::coeff_err_carrier` hands the op a coefficient matrix of
/// `e = |err| ≥ 0` and a zero bias, so `AddConstant`/`SubConstant` fold
/// `Σ_j e_j·c_j` into the carried bias while the certified charge the caller
/// needs is `Σ_j e_j·|c_j|`. The two are EQUAL exactly when `c` never changes
/// sign — every summand then shares one sign, so the sum's magnitude is the sum
/// of the magnitudes. `0.0` and `-0.0` contribute nothing either way and count
/// as neutral, so a constant of mixed positives-and-zeros (or
/// negatives-and-zeros) still qualifies.
///
/// Testing the AUTHORED constant is sufficient for every layout the backwards
/// build: `AddConstantLayer::backward_constant_columns` and
/// `SubConstantLayer::propagate_linear` only ever RESHAPE, BROADCAST, or TILE
/// the constant's own entries into `c_flat` — replication cannot introduce a
/// sign that is not already present — and `SubConstant`'s `y = x - c` /
/// `y = c - x` sign is applied to the whole fold, outside the sum.
///
/// Answers `false` for a non-finite entry and for an empty constant, so those
/// leave the carrier set and take the dispatchers' discharge path (looser, never
/// wrong) rather than reaching a fold whose charge cannot be trusted.
fn constant_fold_cannot_cancel(constant: &ndarray::ArrayD<f32>) -> bool {
    if constant.is_empty() {
        return false;
    }
    let mut seen_positive = false;
    let mut seen_negative = false;
    for &v in constant.iter() {
        if !v.is_finite() {
            return false;
        }
        seen_positive |= v > 0.0;
        seen_negative |= v < 0.0;
        if seen_positive && seen_negative {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod query_batched_gate_tests {
    use ndarray::{ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    use super::Layer;
    use crate::bounds::BatchedLinearBounds;
    use crate::layers::reduction::CumsumLayer;

    /// The batched-CROWN allow-list must admit CumSum: its batched backward is
    /// implemented and dispatched. Before this, one CumSum node disqualified a
    /// whole network from the batched path.
    #[test]
    fn cumsum_is_in_the_batched_crown_allow_list() {
        let layer = Layer::CumSum(CumsumLayer::new(-1, false, false));
        assert!(
            layer.supports_batched_crown(),
            "CumSum has a real batched CROWN backward and must be gated in"
        );
        assert!(
            layer.supports_batched_crown_with_conv2d(),
            "CumSum is not one of the spatial-flattening exclusions"
        );
    }

    /// Proof that the gate is not a lie: the dispatched batched backward really
    /// runs for a last-axis CumSum and produces the exact suffix-scan
    /// coefficients (`y_i = sum_{j>=i} x_j` => backward coeff row is the suffix
    /// sum of the incoming row).
    #[test]
    fn cumsum_batched_backward_actually_dispatches() {
        let layer = Layer::CumSum(CumsumLayer::new(-1, false, false));
        let bounds = BatchedLinearBounds::identity(&[3]).expect("identity bounds");
        let pre = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[3]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[3]), 1.0_f32),
        )
        .expect("pre-activation bounds");

        let out = layer
            .propagate_crown_backward_batched(&bounds, Some(&pre), None)
            .expect("CumSum batched CROWN backward must be implemented");
        // Row 0 of the identity is e_0; the (non-reverse, inclusive) CumSum
        // backward turns it into the suffix indicator [1, 0, 0] -> unchanged,
        // while row 2 (e_2) becomes [1, 1, 1].
        assert_eq!(out.lower_a().shape(), &[3, 3]);
        assert_eq!(out.lower_a()[[2, 0]], 1.0);
        assert_eq!(out.lower_a()[[2, 1]], 1.0);
        assert_eq!(out.lower_a()[[2, 2]], 1.0);
        assert_eq!(out.lower_a()[[0, 1]], 0.0);
    }
}

#[cfg(test)]
mod constant_fold_carrier_tests {
    //! The `A→bias` fold half of the coefficient-error carrier contract
    //! (#vnncomp-aw-soundness).
    //!
    //! These tests drive the PRODUCTION composition, not a kernel in isolation:
    //! the gate `Layer::is_exact_linear_coeff_err_carrier`, then
    //! `LinearBounds::coeff_err_carrier` → the layer's own
    //! `propagate_crown_backward` → `LinearBounds::attach_err_from_carried`,
    //! exactly as `dispatch_exact_linear_with_err`
    //! (`network/backward_dispatch/dispatch.rs`) and the batched carrier arm
    //! (`layers/layer_enum/dispatch.rs`) compose them.

    use ndarray::{Array1, Array2, ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    use super::Layer;
    use crate::bounds::LinearBounds;
    use crate::layers::arithmetic::{AddConstantLayer, SubConstantLayer};
    use crate::layers::common::BoundPropagation;

    fn constant(values: [f32; 2]) -> ArrayD<f32> {
        ArrayD::from_shape_vec(IxDyn(&[2]), values.to_vec()).expect("constant tensor")
    }

    /// One row of incoming CROWN bounds over a 2-position layer OUTPUT, carrying
    /// a certified coefficient error of `e` on both relaxations.
    fn bounds_with_err(a: [f32; 2], e: [f32; 2]) -> LinearBounds {
        let mut bounds = LinearBounds::new(
            Array2::from_shape_vec((1, 2), a.to_vec()).expect("lower_a"),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), a.to_vec()).expect("upper_a"),
            Array1::zeros(1),
        )
        .expect("incoming bounds");
        bounds.set_coeff_err(
            Array2::from_shape_vec((1, 2), e.to_vec()).expect("lower err"),
            Array2::from_shape_vec((1, 2), e.to_vec()).expect("upper err"),
        );
        assert!(bounds.has_coeff_err(), "the fixture must carry an error");
        bounds
    }

    /// The carrier branch, mirroring `dispatch_exact_linear_with_err`: the real
    /// run on error-free coefficients, the carrier run on `|err|`, then attach.
    fn carrier_path(layer: &Layer, bounds: &LinearBounds, pre: &BoundedTensor) -> LinearBounds {
        let carrier = bounds
            .coeff_err_carrier()
            .expect("bounds carry an error, so a carrier exists");
        let mut plain = bounds.clone();
        plain.lower_a_err = None;
        plain.upper_a_err = None;
        let mut real = layer
            .propagate_crown_backward(&plain, Some(pre))
            .expect("real backward");
        let carried = layer
            .propagate_crown_backward(&carrier, Some(pre))
            .expect("carrier backward");
        real.attach_err_from_carried(&carried);
        real
    }

    /// THE GATE. For every constant-shift instance this predicate ADMITS to the
    /// carrier set, the carrier's own bias fold must publish a widening of at
    /// least the MAGNITUDE charge `Σ_j e_j·|c_j|` on each side.
    ///
    /// Both layers pass `A` through (`SubConstant` may negate it) and shift the
    /// bias by `±A @ c`, so the real run leaves `lower_b == upper_b` and the
    /// published width is exactly twice the charge `attach_err_from_carried`
    /// folded in — a layer-agnostic, sign-agnostic observable.
    ///
    /// FAILS on the pre-fix tree: `c = [+1, -1]` with `e = [1, 1]` needs a charge
    /// of 2 (width 4) and the signed carrier fold publishes `|1·(+1) + 1·(-1)| =
    /// 0` (width 0).
    #[test]
    fn admitted_constant_shift_carriers_cover_the_magnitude_charge() {
        let pre = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0_f32, 0.0]).expect("pre lower"),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, 1.0]).expect("pre upper"),
        )
        .expect("pre-activation box");
        let e = [1.0_f32, 1.0];
        let mut admitted = 0_usize;
        for c in [
            [1.0_f32, -1.0],  // mixed sign: the defect's own instance
            [-1.0_f32, 1.0],  // mixed sign, other order
            [3.0_f32, -0.25], // mixed sign, unequal magnitudes
            [1.0_f32, 2.0],   // sign-uniform positive
            [-1.0_f32, -2.0], // sign-uniform negative
            [0.0_f32, -3.0],  // zero is sign-neutral
        ] {
            let required =
                f64::from(e[0]) * f64::from(c[0]).abs() + f64::from(e[1]) * f64::from(c[1]).abs();
            for layer in [
                Layer::AddConstant(AddConstantLayer::new(constant(c))),
                Layer::SubConstant(SubConstantLayer::new(constant(c))),
                Layer::SubConstant(SubConstantLayer::new_reverse(constant(c))),
            ] {
                if !layer.is_exact_linear_coeff_err_carrier() {
                    // Refused: the dispatchers discharge the error over the node's
                    // own box instead, which never under-charges. Nothing to prove.
                    continue;
                }
                admitted += 1;
                let out = carrier_path(&layer, &bounds_with_err([1.0, 1.0], e), &pre);
                let width = f64::from(out.upper_b()[0]) - f64::from(out.lower_b()[0]);
                assert!(
                    width >= 2.0 * required - 1e-4,
                    "{} admitted to the coeff-err carrier set with c = {:?}, but its carried \
                     bias charge is {:.6} where the magnitude charge SUM_j e_j*|c_j| = {:.6} \
                     is required — a bound tighter than the arithmetic justifies",
                    layer.layer_type(),
                    c,
                    width / 2.0,
                    required
                );
            }
        }
        assert!(
            admitted >= 6,
            "the sign-uniform constants must STAY in the carrier set (admitted = {admitted}); \
             a blanket refusal would make this gate vacuous"
        );
    }

    /// The defect's own instance, named: a mixed-sign constant must NOT be in the
    /// carrier set, while its sign-uniform siblings must stay in it.
    #[test]
    fn mixed_sign_constant_shift_is_refused_from_the_carrier_set() {
        for layer in [
            Layer::AddConstant(AddConstantLayer::new(constant([1.0, -1.0]))),
            Layer::SubConstant(SubConstantLayer::new(constant([1.0, -1.0]))),
            Layer::SubConstant(SubConstantLayer::new_reverse(constant([1.0, -1.0]))),
        ] {
            assert!(
                !layer.is_exact_linear_coeff_err_carrier(),
                "{}: the carrier's bias fold SUM_j e_j*c_j CANCELS on a mixed-sign constant, \
                 so it must take the dispatchers' discharge path instead",
                layer.layer_type()
            );
        }
        for layer in [
            Layer::AddConstant(AddConstantLayer::new(constant([1.0, 2.0]))),
            Layer::AddConstant(AddConstantLayer::new(constant([-1.0, -2.0]))),
            Layer::AddConstant(AddConstantLayer::new(constant([0.0, 2.0]))),
            Layer::SubConstant(SubConstantLayer::new(constant([1.0, 2.0]))),
            Layer::SubConstant(SubConstantLayer::scalar(-0.5)),
        ] {
            assert!(
                layer.is_exact_linear_coeff_err_carrier(),
                "{}: a sign-uniform constant cannot cancel, so the tight carrier path must be \
                 kept — refusing it would be an unnecessary loss of precision",
                layer.layer_type()
            );
        }
    }

    /// `constant_fold_cannot_cancel` fails closed on the degenerate inputs.
    #[test]
    fn constant_fold_cancellation_test_fails_closed() {
        use super::constant_fold_cannot_cancel;
        assert!(constant_fold_cannot_cancel(&constant([2.0, 0.0])));
        assert!(constant_fold_cannot_cancel(&constant([-0.0, -2.0])));
        assert!(!constant_fold_cannot_cancel(&constant([f32::NAN, 1.0])));
        assert!(!constant_fold_cannot_cancel(&constant([
            f32::INFINITY,
            1.0
        ])));
        assert!(!constant_fold_cannot_cancel(&ArrayD::<f32>::zeros(IxDyn(
            &[0]
        ))));
    }
}

#[cfg(test)]
mod constant_fold_direct_propagation_tests {
    //! The DIRECT err-propagation half of the constant-shift contract
    //! (#vnncomp-aw-soundness, acasxu prop_1 recovery).
    //!
    //! `constant_fold_carrier_tests` above proves the carrier set refuses the
    //! mixed-sign `A→bias` fold, whose signed carrier charge `|Σ_j e_j·c_j|`
    //! CANCELS. These tests prove the replacement fast path: `AddConstant` is a
    //! `propagates_coeff_err()` DIRECT propagator whose own CROWN backward
    //! passes the err planes through untouched (`A` is untouched, so the
    //! certificate transfers verbatim) and charges the fold by MAGNITUDE
    //! `Σ_j e_j·|c_j|` — dominating the signed fold on every constant, equal to
    //! it exactly on the sign-uniform ones. The original eviction stays closed:
    //! mixed-sign instances remain OUT of the carrier set, and the row stays
    //! FINITE instead of taking the sequential lane's
    //! `discharge_coeff_err_to_conservative` vacuous-row degrade.

    use ndarray::{Array1, Array2, ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    use super::Layer;
    use crate::bounds::LinearBounds;
    use crate::layers::arithmetic::AddConstantLayer;
    use crate::layers::common::BoundPropagation;

    fn constant(values: [f32; 2]) -> ArrayD<f32> {
        ArrayD::from_shape_vec(IxDyn(&[2]), values.to_vec()).expect("constant tensor")
    }

    /// One row of incoming CROWN bounds over a 2-position layer OUTPUT, carrying
    /// a certified coefficient error of `e` on both relaxations.
    fn bounds_with_err(a: [f32; 2], e: [f32; 2]) -> LinearBounds {
        let mut bounds = LinearBounds::new(
            Array2::from_shape_vec((1, 2), a.to_vec()).expect("lower_a"),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), a.to_vec()).expect("upper_a"),
            Array1::zeros(1),
        )
        .expect("incoming bounds");
        bounds.set_coeff_err(
            Array2::from_shape_vec((1, 2), e.to_vec()).expect("lower err"),
            Array2::from_shape_vec((1, 2), e.to_vec()).expect("upper err"),
        );
        assert!(bounds.has_coeff_err(), "the fixture must carry an error");
        bounds
    }

    fn pre() -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0_f32, 0.0]).expect("pre lower"),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, 1.0]).expect("pre upper"),
        )
        .expect("pre-activation box")
    }

    /// THE RECOVERED FAST PATH. For EVERY constant sign pattern — including the
    /// mixed-sign instances the carrier set refuses (every acasxu FC bias) — the
    /// direct backward keeps the row FINITE, widens the bias OUTWARD by at least
    /// the magnitude charge `Σ_j e_j·|c_j|`, and passes the err planes through
    /// unchanged.
    #[test]
    fn add_constant_backward_propagates_incoming_err_with_the_magnitude_charge() {
        let e = [1.0_f32, 0.5];
        for c in [
            [1.0_f32, -1.0],  // mixed sign: the refused-carrier instance
            [-3.0_f32, 0.25], // mixed sign, unequal magnitudes
            [1.0_f32, 2.0],   // sign-uniform positive (parity with the carrier)
            [-1.0_f32, -2.0], // sign-uniform negative
        ] {
            let layer = Layer::AddConstant(AddConstantLayer::new(constant(c)));
            assert!(
                layer.propagates_coeff_err(),
                "AddConstant must be a DIRECT err propagator so the sequential \
                 lane never discharges its rows to [-inf, +inf]"
            );
            let with_err = bounds_with_err([1.0, 1.0], e);
            let mut plain = with_err.clone();
            plain.lower_a_err = None;
            plain.upper_a_err = None;
            let real = layer
                .propagate_crown_backward(&plain, Some(&pre()))
                .expect("plain backward");
            let out = layer
                .propagate_crown_backward(&with_err, Some(&pre()))
                .expect("err-carrying backward");
            let required =
                f64::from(e[0]) * f64::from(c[0]).abs() + f64::from(e[1]) * f64::from(c[1]).abs();
            assert!(
                out.lower_b()[0].is_finite() && out.upper_b()[0].is_finite(),
                "c = {c:?}: the direct path must publish a FINITE row — a \
                 vacuous [-inf, +inf] here is exactly the acasxu prop_1 loss"
            );
            assert!(
                f64::from(out.lower_b()[0]) <= f64::from(real.lower_b()[0]) - required + 1e-4,
                "c = {c:?}: lower bias must be widened DOWN by the magnitude \
                 charge {required} (got {} vs plain {})",
                out.lower_b()[0],
                real.lower_b()[0]
            );
            assert!(
                f64::from(out.upper_b()[0]) >= f64::from(real.upper_b()[0]) + required - 1e-4,
                "c = {c:?}: upper bias must be widened UP by the magnitude \
                 charge {required} (got {} vs plain {})",
                out.upper_b()[0],
                real.upper_b()[0]
            );
            let le = out.lower_a_err().expect("lower err plane must survive");
            let ue = out.upper_a_err().expect("upper err plane must survive");
            for j in 0..2 {
                assert_eq!(
                    le[[0, j]],
                    e[j],
                    "c = {c:?}: A passes through untouched, so the lower err \
                     plane must transfer verbatim"
                );
                assert_eq!(
                    ue[[0, j]],
                    e[j],
                    "c = {c:?}: A passes through untouched, so the upper err \
                     plane must transfer verbatim"
                );
            }
        }
    }

    /// NEGATIVE — the ORIGINAL asymmetry stays closed. On the defect's own
    /// instance `c = [+1, −1]`, `e = [1, 1]` the signed fold `Σ_j e_j·c_j`
    /// publishes a charge of ZERO; the direct path must publish the magnitude
    /// charge 2 on EACH side (width ≥ 4). This test fails on any reintroduction
    /// of the signed fold into the direct path — and the mixed-sign instance
    /// must remain evicted from the carrier set, so no dispatcher can reach the
    /// cancelling carrier fold either.
    #[test]
    fn mixed_sign_direct_charge_is_by_magnitude_and_the_carrier_eviction_stays() {
        let layer = Layer::AddConstant(AddConstantLayer::new(constant([1.0, -1.0])));
        assert!(
            !layer.is_exact_linear_coeff_err_carrier(),
            "the mixed-sign carrier eviction (a5ffc1bcc) must stay closed"
        );
        let out = layer
            .propagate_crown_backward(&bounds_with_err([1.0, 1.0], [1.0, 1.0]), Some(&pre()))
            .expect("backward");
        let width = f64::from(out.upper_b()[0]) - f64::from(out.lower_b()[0]);
        assert!(
            width >= 4.0 - 1e-4,
            "mixed-sign fold must charge magnitudes (SUM_j e_j*|c_j| = 2 per \
             side, width 4); the signed fold cancels to 0 — got width {width}"
        );
    }

    /// No incoming certificate → byte-identical previous behavior: no err
    /// planes materialize and the fold is the plain directed `A @ c`.
    #[test]
    fn add_constant_backward_without_incoming_err_is_unchanged() {
        let layer = Layer::AddConstant(AddConstantLayer::new(constant([1.0, -1.0])));
        let plain = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).expect("lower_a"),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).expect("upper_a"),
            Array1::zeros(1),
        )
        .expect("plain bounds");
        let out = layer
            .propagate_crown_backward(&plain, Some(&pre()))
            .expect("backward");
        assert!(
            !out.has_coeff_err(),
            "an error-free input must not grow error planes"
        );
        assert!(
            (f64::from(out.lower_b()[0])).abs() < 1e-4
                && (f64::from(out.upper_b()[0])).abs() < 1e-4,
            "plain fold is A @ c = 1*(+1) + 1*(-1) = 0 up to directed rounding"
        );
    }
}
