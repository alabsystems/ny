// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Opt-in, policy-aware precision widening for deployed-precision verdicts (P8).
//!
//! NY proves bounds under an f32 *idealization*. Real models execute in lower
//! precisions on the GPU — f16 (Metal), bf16 (CUDA), or quantized GGUF blocks —
//! so a bound proven in f32 is not automatically valid for the bits that actually
//! run. This module offers two paths that differ in soundness strength:
//!
//! 1. [`verify_with_precision_policy`] — **representation-only, HEURISTIC.** It
//!    widens the verifier's OUTPUT bounds onto the deployed-precision grid. This
//!    is a heuristic *sanity* pass, NOT a sound deployed-precision verdict.
//! 2. [`verify_with_sound_precision`] — **layer-aware, SOUND.** It accounts for
//!    accumulation rounding inside each reduction/Linear/Conv/MatMul layer using
//!    [`ny_core::summation_error_bound`] before re-checking. A `Verified` from it
//!    is sound for the deployed precision (see that function's docs).
//!
//! # What [`verify_with_precision_policy`] models — and what it does NOT
//!
//! It rounds each output bound's endpoints OUTWARD onto the deployed-precision
//! grid (see [`ny_core::round_to_precision_outward`]). That models **only the
//! rounding of STORING the already-computed output value at precision `p`**. It
//! does **NOT** model the rounding that happens *inside* the computation — the
//! term-by-term accumulation drift of a dot product / GEMM / reduction realized
//! in f16/bf16.
//!
//! That distinction is not academic. Summing 5000 ones in f16 saturates near
//! 2048 (once the running sum exceeds 2048 the f16 ULP is 2 and adding 1 is
//! lost), while the f32-idealized result is 5000. Representation-only widening of
//! the f32 output `[5000, 5000]` moves it by a single f16 ULP — nowhere near
//! enough to contain 2048. So a `Verified` produced by the representation-only
//! path **cannot be claimed sound for deployed f16/bf16 compute**.
//!
//! For that reason [`verify_with_precision_policy`] records a HEURISTIC soundness
//! provenance ([`ny_core::VerificationSoundnessMode::Heuristic`]); it never
//! returns a result labeled `Sound` "verified at deployed precision". Use
//! [`verify_with_sound_precision`] when a sound deployed-precision verdict is
//! required.
//!
//! (There is no dedicated [`ny_core::HeuristicUsed`] enum variant for precision
//! widening today — a documented gap, see the crate integration notes. The
//! marker is carried via [`ny_core::SoundnessProvenance::heuristic`], which sets
//! the `Heuristic` mode without listing an enumerated heuristic.)
//!
//! # ADDITIVE + OPT-IN
//!
//! All entry points are no-ops for the default [`MixedPrecisionPolicy`]
//! (`{ compute: F32, accumulate: F32 }`): bounds are returned byte-for-byte
//! unchanged and the verdict is identical to today's f32 path. Widening activates
//! only when a caller sets a non-F32 compute or accumulate precision.
//!
//! # Monotonicity of the verdict re-check (both paths)
//!
//! [`ny_core`]'s verdict rule is monotone: an output bound `[c_lo, c_hi]`
//! satisfies a required bound `[r_lo, r_hi]` iff `c_lo >= r_lo && c_hi <= r_hi`.
//! Widening can only *decrease* `c_lo` and *increase* `c_hi`, so a `Verified`
//! verdict can only ever flip to `Unknown` after widening — never the reverse.
//! This is the safe (fail-closed) direction. The difference between the two paths
//! is *how much* they widen: the representation-only path widens too little to be
//! sound for accumulation, which is exactly why its verdict is labeled Heuristic.

use ny_core::{
    widen_bounds_for_precision, widen_bounds_for_precision_owned, Bound, SoundnessProvenance,
    UnknownReason, VerificationResult, VerificationSpec,
};

/// Target precisions for a mixed-precision verification run (re-exported from
/// `ny-build`). Default is all-F32 (today's idealized behavior; no widening).
pub use ny_build::MixedPrecisionPolicy;
/// Floating-point precision tag (F32 default / F16 / Bf16) the widening keys off.
pub use ny_core::FloatPrecision;

/// The single precision an output-rounding widening must key off of for `policy`.
///
/// A mixed-precision policy declares a `compute` and an `accumulate` precision.
/// A deployed output passes through both kinds of rounding, so to over-approximate
/// it we widen to the **coarser** (more lossy) of the two — see
/// [`FloatPrecision::coarser`]. If both are F32 the result is F32 (no widening).
///
/// SOUNDNESS: widening to the coarser grid is a superset of widening to the finer
/// grid, and bounds the rounding of *either* precision step.
#[must_use]
pub fn effective_widening_precision(policy: &MixedPrecisionPolicy) -> FloatPrecision {
    policy.compute.coarser(policy.accumulate)
}

/// Widen every bound IN PLACE onto `policy`'s deployed-precision grid
/// (REPRESENTATION rounding only).
///
/// Selects the effective widening precision via [`effective_widening_precision`]
/// and rounds each bound outward onto that grid. For the default (all-F32) policy
/// this is a strict no-op — every bound is left byte-for-byte unchanged.
///
/// SCOPE: this brackets the rounding of *storing* each value at the deployed
/// precision. It does NOT account for accumulation rounding, so it is not by
/// itself enough to make a verdict sound for deployed compute — see the module
/// docs and [`verify_with_sound_precision`].
///
/// SOUNDNESS (of the representation step): each bound's lower can only
/// decrease-or-stay and upper can only increase-or-stay; the result is an
/// element-wise superset of the input. Never narrows a bound.
pub fn widen_bounds_for_policy(bounds: &mut [Bound], policy: &MixedPrecisionPolicy) {
    widen_bounds_for_precision(bounds, effective_widening_precision(policy));
}

/// Returning variant of [`widen_bounds_for_policy`].
///
/// Produces a fresh `Vec<Bound>` widened to `policy`'s deployed precision, leaving
/// the input untouched. For the default (all-F32) policy this is an exact copy.
#[must_use]
pub fn widen_bounds_for_policy_owned(
    bounds: &[Bound],
    policy: &MixedPrecisionPolicy,
) -> Vec<Bound> {
    widen_bounds_for_precision_owned(bounds, effective_widening_precision(policy))
}

/// Apply REPRESENTATION-ONLY precision widening to a finished
/// [`VerificationResult`] and re-decide the verdict against `spec`.
///
/// This is the post-processing core shared by [`verify_with_precision_policy`].
/// It is a HEURISTIC pass: it widens output bounds onto the deployed-precision
/// grid but does NOT model accumulation rounding, so its `Verified` is not a
/// sound deployed-precision verdict (see module docs). It takes a result produced
/// by NY's normal f32 propagation and:
///
/// 1. **Default policy (all-F32):** returns `result` completely unchanged — same
///    variant, same bounds, same provenance. Byte-for-byte today's behavior.
/// 2. **Non-F32 policy:**
///    - `Verified` / `Unknown`: widens the result's output bounds outward to the
///      deployed grid, then re-checks them against `spec.output_bounds()`. If the
///      widened bounds still satisfy every required bound the verdict stays
///      `Verified` (carrying the widened bounds); otherwise it is downgraded to
///      `Unknown { reason: BoundsTooLoose }`. Either way the recorded provenance is
///      marked `Heuristic` — a `Verified` here is a heuristic signal, NOT a sound
///      deployed-precision claim.
///    - `Violated`: returned unchanged. A concrete counterexample found in the
///      f32 idealization is reported as-is; widening output *intervals* cannot
///      make a found violation sound to drop, and re-validating the counterexample
///      at the deployed precision is out of scope for this representation-rounding
///      pass.
///    - `Timeout`: returned unchanged (no completed bounds to re-check).
///
/// MONOTONICITY: widening only ever loosens the computed bounds, and the verdict
/// rule is monotone (see module docs), so this pass cannot turn `Unknown` into
/// `Verified`. It is fail-closed relative to its own (representation-only) model,
/// but that model under-approximates the deployed error — hence Heuristic.
#[must_use]
pub fn widen_result_for_policy(
    result: VerificationResult,
    spec: &VerificationSpec,
    policy: &MixedPrecisionPolicy,
) -> VerificationResult {
    // OPT-IN no-op: the idealized all-f32 policy reproduces today's result exactly.
    if policy.is_idealized_f32() {
        return result;
    }

    let precision = effective_widening_precision(policy);
    let required = spec.output_bounds();

    match result {
        VerificationResult::Verified {
            provenance,
            output_bounds,
            proof: _,
            actual_method,
        } => {
            // Re-decide the verdict on the widened bounds. Dropping any prior proof
            // object is deliberate: a proof certified for the f32 bounds does not
            // certify the widened (deployed-precision) bounds.
            let widened = widen_bounds_for_precision_owned(&output_bounds, precision);
            let prov = mark_precision_heuristic(&provenance);
            decide_against_required(widened, required, prov, actual_method)
        }
        VerificationResult::Unknown {
            provenance,
            bounds,
            reason: _,
            actual_method,
        } => {
            // Already inconclusive in f32. Widening can only keep it inconclusive
            // (or, in the degenerate case where the loose f32 bounds happened to
            // straddle the requirement, it stays inconclusive). Re-decide so the
            // reported bounds are the deployed-precision ones, but the outcome can
            // only be Unknown here because the f32 bounds already failed the check
            // and widening cannot tighten them.
            let widened = widen_bounds_for_precision_owned(&bounds, precision);
            let prov = mark_precision_heuristic(&provenance);
            decide_against_required(widened, required, prov, actual_method)
        }
        // A concrete counterexample / timeout is passed through unchanged.
        other => other,
    }
}

/// Combine `provenance` with the no-detail precision-widening heuristic marker.
///
/// The run is labeled `Heuristic` to flag that the verdict was produced by the
/// REPRESENTATION-ONLY widening pass, which does not model accumulation rounding
/// and therefore cannot be claimed sound for deployed f16/bf16 compute. There is
/// no dedicated [`ny_core::HeuristicUsed`] variant for precision widening today
/// (a documented gap — see integration notes); [`SoundnessProvenance::heuristic`]
/// records the `Heuristic` mode without inventing a misleading enum entry.
fn mark_precision_heuristic(provenance: &SoundnessProvenance) -> SoundnessProvenance {
    provenance.combine(&SoundnessProvenance::heuristic())
}

/// Decide `Verified` vs `Unknown` for `computed` against `required`, mirroring
/// `ny_core`'s monotone verdict rule (`c_lo >= r_lo && c_hi <= r_hi` per index).
///
/// If `required` is empty (no output property to check) the bounds cannot be
/// validated, so the result is conservatively `Unknown` rather than a vacuous
/// `Verified`. If the network produced fewer bounds than `required`, the same
/// conservative `Unknown` is returned (never silently truncate a requirement).
fn decide_against_required(
    computed: Vec<Bound>,
    required: &[Bound],
    provenance: SoundnessProvenance,
    actual_method: Option<ny_core::MethodUsed>,
) -> VerificationResult {
    let unknown = |bounds: Vec<Bound>, gap: Option<f32>| VerificationResult::Unknown {
        provenance: provenance.clone(),
        bounds,
        reason: UnknownReason::BoundsTooLoose { gap },
        actual_method: actual_method.clone(),
    };

    if required.is_empty() || computed.len() < required.len() {
        return unknown(computed, None);
    }

    let mut worst_gap = 0.0_f32;
    let mut satisfied = true;
    for (c, r) in computed.iter().zip(required.iter()) {
        let lower_gap = if c.lower() < r.lower() {
            r.lower() - c.lower()
        } else {
            0.0
        };
        let upper_gap = if c.upper() > r.upper() {
            c.upper() - r.upper()
        } else {
            0.0
        };
        let gap = lower_gap.max(upper_gap);
        if gap > 0.0 {
            satisfied = false;
            if gap > worst_gap {
                worst_gap = gap;
            }
        }
    }

    if satisfied {
        VerificationResult::Verified {
            provenance,
            output_bounds: computed,
            proof: None,
            actual_method,
        }
    } else {
        unknown(
            computed,
            if worst_gap > 0.0 {
                Some(worst_gap)
            } else {
                None
            },
        )
    }
}

/// Verify `network` against `spec`, then widen the OUTPUT bounds onto `policy`'s
/// deployed-precision grid as a HEURISTIC (representation-only) sanity pass.
///
/// **This is NOT a sound deployed-precision verdict.** It models only the
/// rounding of *storing* the output at precision `p`; it ignores the
/// accumulation rounding that happens *inside* f16/bf16 reductions (see the
/// module docs and the 5000-ones example). For a sound deployed-precision
/// verdict use [`verify_with_sound_precision`].
///
/// It runs NY's normal f32 propagation via [`verifier`](crate::verify::Verifier),
/// then applies [`widen_result_for_policy`], which rounds the output bounds
/// outward and re-checks them.
///
/// - **Default policy (all-F32):** identical to `verifier.verify(network, spec)` —
///   no widening, no provenance change, today's exact behavior.
/// - **Non-F32 policy:** the output bounds are widened by the representation step
///   and the verdict is re-checked; the result is labeled
///   [`Heuristic`](ny_core::VerificationSoundnessMode::Heuristic). A `Verified`
///   here means "the representation-rounding sanity pass did not break the f32
///   verdict", NOT "verified at the deployed precision".
///
/// MONOTONICITY: widening only loosens bounds and the verdict rule is monotone,
/// so this pass cannot turn an f32 `Unknown` into `Verified`. But because it
/// under-models accumulation, a `Verified` from it is only a heuristic signal —
/// hence the provenance label.
///
/// # Errors
/// Propagates any error from the underlying [`Verifier::verify`].
pub fn verify_with_precision_policy(
    verifier: &crate::verify::Verifier,
    network: &crate::graph::SequentialNetwork,
    spec: &VerificationSpec,
    policy: &MixedPrecisionPolicy,
) -> ny_core::Result<VerificationResult> {
    let result = verifier.verify(network, spec)?;
    Ok(widen_result_for_policy(result, spec, policy))
}

// ===========================================================================
// PART D — Genuinely-sound, layer-aware deployed-precision verification.
//
// Unlike `verify_with_precision_policy` (representation-only / Heuristic), this
// path accounts for ACCUMULATION rounding inside reduction/Linear layers using
// `ny_core::summation_error_bound`. When it returns `Verified` with `Sound`
// provenance, the verdict is sound for the deployed precision.
// ===========================================================================

use ny_core::FloatPrecision as FP;
use ny_propagate::layers::{BoundPropagation, Layer};
use ny_propagate::{GraphNetwork, GraphNode, NETWORK_INPUT};
use ny_tensor::BoundedTensor;

/// Tracks whether every accumulating layer's error was exactly accounted for.
///
/// The layer-aware pass models the accumulation error of `Linear` layers exactly
/// (their weights are available). Convolution / `MatMul` accumulating layers are
/// widened conservatively to `[-inf, +inf]` (still sound) but cannot be reported
/// as exactly accounted, so the run is downgraded to `Heuristic` with a recorded
/// reason rather than claiming a `Sound` deployed-precision verdict for them.
struct AccountingState {
    /// `true` while every accumulating layer encountered so far was modeled
    /// exactly. Becomes `false` (with a reason) the first time an accumulating
    /// layer is only conservatively bounded.
    all_accounted: bool,
    /// Human-readable reason recorded when `all_accounted` flips to `false`.
    reason: Option<String>,
}

impl AccountingState {
    fn new() -> Self {
        Self {
            all_accounted: true,
            reason: None,
        }
    }

    fn record_unaccounted(&mut self, reason: impl Into<String>) {
        if self.all_accounted {
            self.all_accounted = false;
            self.reason = Some(reason.into());
        }
    }
}

/// Verify `graph` against `spec` at `policy`'s deployed precision, accounting for
/// per-layer ACCUMULATION rounding so a `Verified` verdict is SOUND for the
/// deployed f16/bf16 compute.
///
/// # What is modeled
///
/// The pass runs a self-contained forward IBP over the graph in topological
/// order, reusing NY's own per-layer IBP (which is inclusion-monotone and
/// directed-rounded). For `Linear` layers it then RECOMPUTES each output's
/// deployed interval by DIRECT INTERVAL ARITHMETIC AT DEPLOYED PRECISION,
/// rounding OUTWARD to the actual grid at every step (see [`sound_widen_linear`]).
/// This is sound BY CONSTRUCTION rather than by an error budget:
///
/// - Inputs and weights are rounded OUTWARD to the compute grid before use (via
///   [`ny_core::round_to_precision_outward`]).
/// - Each per-term product is the interval hull of the four corner products,
///   narrowed to f32 outward, then rounded OUTWARD to the compute grid — so a
///   per-product overflow into compute precision becomes `±inf` automatically.
/// - Products are accumulated left-to-right with each partial-sum store rounded
///   OUTWARD to the accumulate grid — so a running-sum overflow becomes `±inf`
///   automatically, and the single-store rounding of a one-term reduction is
///   modeled like any other.
/// - The bias is folded in as one extra accumulated term with its own
///   accumulate-grid store.
/// - The final accumulator is rounded OUTWARD to the output store grid (the
///   COARSER of compute/accumulate) for the output store rounding.
///
/// Because interval arithmetic with directed outward rounding contains every
/// real execution, the recomputed interval over-approximates every value the
/// deployed-precision Linear can produce — no `gamma_N` accumulation estimate and
/// no per-source error budget that could under-charge. Subnormals, bias, n=1
/// stores, compute-vs-accumulate coarseness, and overflow are all handled by the
/// grid-rounding operator itself.
///
/// # Exactly-accounted vs. conservatively-bounded layers
///
/// - `Linear` accumulating layers are modeled **soundly by construction** with
///   direct outward-rounded interval arithmetic (weights/bias and fan-in =
///   `in_features` are available, so the per-output reduction is reconstructible).
/// - Convolution / `MatMul` accumulating layers are widened **conservatively** to
///   `[-inf, +inf]` (sound but coarse) because their weight/fan-in product is not
///   reachable through this facade. The run is then labeled `Heuristic` with a
///   recorded reason rather than claiming a `Sound` deployed-precision verdict.
/// - `ReduceSum` / `ReduceMean` are likewise widened to `[-inf, +inf]` +
///   `Heuristic`: the per-output fan-in (reduced-axis -> output mapping) and the
///   mean's 1/N scaling are not reachable here, and a single full-set fold cannot
///   soundly stand in for arbitrary mixed-sign subset sums.
/// - All non-accumulating layers (activations, reshapes, element-wise ops) are
///   propagated by NY's existing sound IBP; their soundness comes for free from
///   inclusion-monotonicity applied to the already-widened inputs.
///
/// # Provenance
///
/// `Sound` iff the policy is non-trivial **and** every accumulating layer was
/// accounted for exactly; otherwise `Heuristic`. The default (all-F32) policy is
/// a strict no-op that delegates to [`Verifier::verify_graph`] and returns its
/// verdict unchanged (so `F32` is exactly the normal verdict, provenance and
/// all).
///
/// # Errors
/// Propagates layer/propagation errors and shape errors from the underlying IBP.
pub fn verify_with_sound_precision(
    net: &GraphNetwork,
    spec: &VerificationSpec,
    policy: &MixedPrecisionPolicy,
) -> ny_core::Result<VerificationResult> {
    // OPT-IN strict no-op: the idealized all-f32 policy is exactly the normal
    // verdict (same engine, same provenance) — no widening at all.
    if policy.is_idealized_f32() {
        let verifier = crate::verify::Verifier::new(crate::verify::PropagationConfig::default());
        return verifier.verify_graph(net, spec);
    }

    // Precisions: `compute` rounds per-element products/activations; `accumulate`
    // is the reduction running-sum precision. Use each where it applies.
    let compute_p = policy.compute;
    let accumulate_p = policy.accumulate;

    // Build the input bounded tensor from the spec (same contract the verifier
    // uses), then run our layer-aware forward IBP.
    let input = crate::verify::Verifier::bounds_to_tensor(spec.input_bounds(), spec.input_shape())?;

    let mut accounting = AccountingState::new();
    let output_tensor = sound_forward_ibp(net, &input, compute_p, accumulate_p, &mut accounting)?;

    // Flatten the deployed output bounds to a per-element Vec<Bound>.
    let computed = bounded_tensor_to_bounds(&output_tensor);

    // Provenance: Sound only if every accumulating layer was accounted for.
    let provenance = if accounting.all_accounted {
        SoundnessProvenance::sound()
    } else {
        // Record the gap explicitly; never silently claim Sound.
        SoundnessProvenance::heuristic()
    };

    Ok(decide_against_required(
        computed,
        spec.output_bounds(),
        provenance,
        Some(ny_core::MethodUsed::Ibp),
    ))
}

/// Run a self-contained, SOUND, layer-aware forward IBP over `net`.
///
/// Returns the deployed-precision output [`BoundedTensor`]. Per accumulating
/// layer, the output is additively widened by the accumulation + representation
/// error so the result over-approximates the deployed computation.
fn sound_forward_ibp(
    net: &GraphNetwork,
    input: &BoundedTensor,
    compute_p: FP,
    accumulate_p: FP,
    accounting: &mut AccountingState,
) -> ny_core::Result<BoundedTensor> {
    use std::collections::HashMap;

    let order = net.topological_sort()?;
    let mut cache: HashMap<String, BoundedTensor> = HashMap::with_capacity(order.len());

    // Helper: resolve a named input to its (deployed) bounded tensor.
    let resolve =
        |name: &str, cache: &HashMap<String, BoundedTensor>| -> ny_core::Result<BoundedTensor> {
            if name == NETWORK_INPUT {
                Ok(input.clone())
            } else {
                cache.get(name).cloned().ok_or_else(|| {
                    ny_core::NyError::InvalidSpec(format!(
                        "verify_with_sound_precision: missing bounds for input node '{name}'"
                    ))
                })
            }
        };

    for node_name in &order {
        let node: &GraphNode = net.node(node_name).ok_or_else(|| {
            ny_core::NyError::InvalidSpec(format!(
                "verify_with_sound_precision: node '{node_name}' not found"
            ))
        })?;
        let layer = node.layer();
        let inputs = node.inputs();

        // 1. Idealized layer output computed on the DEPLOYED (already-widened)
        //    inputs, via NY's existing sound per-layer IBP.
        let mut out = match inputs.len() {
            0 => {
                // No declared inputs: treat as taking the network input.
                let a = input.clone();
                layer.propagate_ibp(&a)?
            }
            1 => {
                let a = resolve(&inputs[0], &cache)?;
                layer.propagate_ibp(&a)?
            }
            2 => {
                let a = resolve(&inputs[0], &cache)?;
                let b = resolve(&inputs[1], &cache)?;
                layer.propagate_ibp_binary(&a, &b)?
            }
            n => {
                // Ternary+ ops (e.g. SelfAttention, Where): a uniform sound IBP
                // call is not available through this facade, and synthesizing a
                // safe stand-in tensor would risk an UNSOUND result. Soundness
                // first: fail closed rather than emit a possibly-wrong verdict.
                return Err(ny_core::NyError::UnsupportedOp(format!(
                    "verify_with_sound_precision: node '{node_name}' has {n} inputs \
                     (ternary+); the sound layer-aware path does not model it. Use the \
                     standard verifier or extend this path before claiming a \
                     deployed-precision verdict."
                )));
            }
        };

        // 2. If this is an ACCUMULATING layer, widen its output by the
        //    accumulation + representation error for the deployed precision.
        if let Some(in0) = inputs.first() {
            let in_tensor =
                resolve(in0, &cache).or_else(|_| Ok::<_, ny_core::NyError>(input.clone()))?;
            out = widen_accumulating_layer_output(
                layer,
                &in_tensor,
                out,
                compute_p,
                accumulate_p,
                accounting,
            )?;
        }

        cache.insert(node_name.clone(), out);
    }

    let output_name = net.output_name();
    cache.get(output_name).cloned().ok_or_else(|| {
        ny_core::NyError::InvalidSpec(format!(
            "verify_with_sound_precision: output node '{output_name}' produced no bounds"
        ))
    })
}

/// If `layer` is an accumulating layer, return `out` additively widened by the
/// SOUND accumulation + representation error; otherwise return `out` unchanged.
///
/// `in_tensor` is the (deployed) input interval to the layer; `out` is the
/// idealized output interval already computed by the layer's IBP.
fn widen_accumulating_layer_output(
    layer: &Layer,
    in_tensor: &BoundedTensor,
    out: BoundedTensor,
    compute_p: FP,
    accumulate_p: FP,
    accounting: &mut AccountingState,
) -> ny_core::Result<BoundedTensor> {
    match layer {
        Layer::Linear(lin) => sound_widen_linear(lin, in_tensor, &out, compute_p, accumulate_p),
        // Convolution / MatMul accumulate too, but their weight*fan-in product is
        // not reachable here. Conservatively widen to [-inf, +inf] (sound) and
        // record that the run is no longer exactly accounted.
        Layer::Conv1d(_)
        | Layer::Conv2d(_)
        | Layer::ConvTranspose1d(_)
        | Layer::ConvTranspose2d(_)
        | Layer::MatMul(_)
        | Layer::BilinearCrown(_) => {
            accounting.record_unaccounted(format!(
                "{} accumulation not modeled exactly (weights/fan-in not reachable); \
                 output widened conservatively",
                layer.layer_type()
            ));
            let lower = out.lower().mapv(|_| f32::NEG_INFINITY);
            let upper = out.upper().mapv(|_| f32::INFINITY);
            BoundedTensor::new_allow_infinite(lower, upper)
        }
        // Reductions that sum/mean also accumulate. Following the design change
        // (DIRECT interval arithmetic, no error budget), a per-output direct fold
        // would require the exact per-output fan-in (the reduced-axis -> output
        // mapping), which is NOT recoverable through this facade: a subset sum of
        // mixed-sign terms need not lie inside a full-set fold, so we cannot
        // soundly reuse a single fold across outputs, and ReduceMean's 1/N scaling
        // is likewise not modeled here. Rather than reintroduce an error-ESTIMATE
        // (the very thing this rewrite removes) we conservatively widen to
        // [-inf, +inf] (sound by construction — it contains every deployed value,
        // including overflow to ±inf) and DOWNGRADE the run to Heuristic with a
        // recorded reason. (See the Linear path for the direct-IA construction
        // applied where the per-output fan-in IS available.)
        Layer::ReduceSum(_) | Layer::ReduceMean(_) => {
            accounting.record_unaccounted(format!(
                "{} reduction not modeled by direct interval arithmetic (per-output \
                 fan-in / mean scaling not reachable); output widened conservatively \
                 to [-inf, +inf] and the run is labeled Heuristic",
                layer.layer_type()
            ));
            let lower = out.lower().mapv(|_| f32::NEG_INFINITY);
            let upper = out.upper().mapv(|_| f32::INFINITY);
            BoundedTensor::new_allow_infinite(lower, upper)
        }
        // Non-accumulating layers: no accumulation widening needed; the IBP output
        // is already sound on the deployed (widened) inputs.
        _ => Ok(out),
    }
}

// ===========================================================================
// SOUND Linear deployed-precision widening — DIRECT INTERVAL ARITHMETIC.
//
// DESIGN CHANGE (rev3): the prior Linear branch ESTIMATED the deployed error
// (gamma_N accumulation bound + per-source error terms) and three independent
// audits kept finding UNDER-widening corners in that approach:
//   (a) n_terms<=1 charged zero accumulation but a single-term reduction still
//       does one accumulate-precision store rounding;
//   (b) per-product overflow into the COMPUTE precision was not detected (the
//       overflow guard was keyed only on the accumulate max_normal) — e.g.
//       compute=F16, product 120064 -> +inf was missed;
//   (c) accumulator rounding was charged at compute_p when accumulate_p is the
//       coarser store grid;
//   (d) reductions lacked overflow guards.
//
// These are all symptoms of ESTIMATING error instead of COMPUTING it. We now
// replace the estimate with DIRECT INTERVAL ARITHMETIC AT DEPLOYED PRECISION,
// rounding OUTWARD to the actual grid at every step. This is sound BY
// CONSTRUCTION: interval arithmetic with directed outward rounding contains
// every real execution, and the grid-rounding operator
// `round_to_precision_outward` saturates to ±inf on overflow, so per-product
// overflow, accumulate overflow, n=1 stores, compute-vs-accumulate coarseness,
// subnormals, and bias are ALL handled with no error budget to under-charge.
//
// GUIDING PRINCIPLE: widening MORE is always sound; only being too tight is
// unsound. Every step here over-approximates; any ±inf endpoint is preserved.
// ===========================================================================

/// Convert an exact f64 value to f32 with OUTWARD directed rounding.
///
/// `up == true` rounds toward +inf (use for an upper endpoint), `up == false`
/// rounds toward -inf (use for a lower endpoint). The plain `as f32` cast is
/// round-to-nearest, which can move the value the WRONG way (a lower endpoint
/// could round up, excluding the true value); we therefore always nudge one
/// f32 ULP in the outward direction. Over-widening by at most one ULP is sound
/// and is in any case re-absorbed by the subsequent grid rounding.
///
/// Non-finite inputs pass through (`+inf`/`-inf`/`NaN` map to f32 `+inf`/`-inf`/
/// `NaN`; a NaN sum can only arise from `+inf + -inf` corners and is widened to
/// the appropriate infinity by the callers' overflow handling).
#[inline]
fn f64_to_f32_outward(x: f64, up: bool) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x == f64::INFINITY {
        return f32::INFINITY;
    }
    if x == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    let f = x as f32;
    if up {
        // Round-to-nearest may have landed below x; force >= x.
        ny_tensor::rounding::next_up_f32(f)
    } else {
        // Round-to-nearest may have landed above x; force <= x.
        ny_tensor::rounding::next_down_f32(f)
    }
}

/// Recompute each Linear output's deployed-precision interval, sound for EVERY
/// hardware reduction ORDER — not just left-to-right.
///
/// For each output row `j` with weights `W[j,:]`, rounded input bounds, bias
/// `b_j`, and policy `{compute, accumulate}`:
///
/// 1. INPUTS are rounded to the compute grid via
///    [`ny_core::round_to_precision_outward`] (deployed inputs are stored at
///    compute precision).
/// 2. WEIGHTS are rounded to the compute grid (a point weight becomes a 1- or
///    2-point grid bracket containing the deployed stored weight).
/// 3. Each PER-TERM PRODUCT is the interval hull of the four corner products
///    `{wl·xl, wl·xu, wu·xl, wu·xu}` computed in f64, narrowed to f32 outward,
///    then rounded OUTWARD to the compute grid (models the per-product rounding
///    to compute precision; overflow -> ±inf automatically). The bias is one
///    extra term, bracketed on the compute grid.
///
///    H2 (NaN / inf*0): `inf*0`, `0*inf`, and `inf-inf` produce NaN, and f64
///    `min`/`max` SILENTLY DROP a NaN operand — which would collapse a genuinely
///    unconstrained term to a finite interval. So before forming the hull we
///    check `is_nan` EXPLICITLY on all four corner products and on either input
///    endpoint; if ANY is NaN the deployed product is unconstrained and the term
///    interval is set to `[-inf, +inf]`.
///
/// 4. ACCUMULATE order-independently over all `N` terms (products + bias):
///    - `sum_lo` = exact f64 sum of the `tl_i`, narrowed OUTWARD to f32
///      (`next_down`); `sum_hi` = exact f64 sum of the `th_i`, narrowed OUTWARD
///      (`next_up`). The EXACT real sum of the deployed terms is order-
///      independent and lies in `[sum_lo, sum_hi]`.
///    - `pos` = f64 sum of `max(0, th_i)` = the largest partial sum reachable by
///      ANY order (add every positive term first); `neg` = f64 sum of
///      `min(0, tl_i)` = the most negative partial sum reachable by any order.
///      In ANY reduction order EVERY partial sum lies in `[neg, pos]`.
///
///    H1 (reduction-order overflow): if `pos >= M` or `neg <= -M`
///    (`M = accumulate.max_normal()`), or any of `sum_lo / sum_hi / pos / neg`
///    is non-finite, SOME legal order can drive a partial sum to round to ±inf,
///    so the output is set to `[-inf, +inf]`. This subsumes per-product overflow
///    (a `+inf` term makes `pos` non-finite) and the H2 `[-inf,+inf]` term.
///
/// 5. ACCUMULATION SLACK (sound for any order): the deployed sum in any order
///    differs from the exact real sum by the rounding of at most `N`
///    accumulate-precision stores (a left fold of N terms does N stores: the
///    first `acc = round(0 + t0)` rounds too, since t0 was computed at compute
///    precision and may not be representable in a coarser accumulate grid), each
///    with absolute error `<= u_acc * |partial sum| <= u_acc * max(pos, -neg)`
///    (`u_acc` = accumulate unit roundoff), FLOORED in the subnormal regime by
///    `half_subnormal_acc` per store. So
///    `E = max(K * u_acc * max(pos, -neg), K * half_subnormal_acc)` with `K = N`
///    (over-approximates any order's store count); `E = 0` for `N == 0`. `E` is
///    computed in f64 and rounded UP to f32.
/// 6. The pre-store interval is `[sum_lo - E, sum_hi + E]`, then rounded OUTWARD
///    to the OUTPUT store precision (the COARSER of {compute, accumulate}).
///    Overflow -> ±inf is preserved.
///
/// SOUNDNESS (order-independent): floating-point summation is NOT associative, so
/// a left-to-right interval fold is NOT sufficient — a different legal order can
/// overflow a partial sum that L-to-R never reaches. This construction is sound
/// for EVERY order because (a) the exact real sum of the deployed terms is
/// order-independent and contained in `[sum_lo, sum_hi]`; (b) in any order every
/// partial sum lies in `[neg, pos]`, so the overflow guard fires whenever any
/// order could overflow, and otherwise `E` dominates the total rounding error of
/// every order (each of the `<= N` stores is charged its worst-case error over
/// `[neg, pos]`); (c) the H2 NaN check makes an `inf*0` term unconstrained rather
/// than silently finite. The product step itself contains every real product of
/// operands in the rounded input/weight intervals, and the final store contains
/// the written output. `out` is used only as a shape/length reference.
fn sound_widen_linear(
    lin: &ny_propagate::layers::LinearLayer,
    in_tensor: &BoundedTensor,
    out: &BoundedTensor,
    compute_p: FP,
    accumulate_p: FP,
) -> ny_core::Result<BoundedTensor> {
    let weight = &lin.weight; // shape [out_features, in_features]
    let in_features = lin.in_features();
    let out_features = lin.out_features().max(1);

    // The OUTPUT store grid: the COARSER of the two precisions. Whichever grid
    // the layer writes its output at (compute or accumulate), the coarser grid
    // is a sound over-approximation of that store rounding (a superset of either
    // finer grid's rounding). Documented choice per the design.
    let output_store_p = compute_p.coarser(accumulate_p);

    // --- (1) INPUT rounding to the compute grid. ---
    // The deployed hardware stores/uses inputs at compute precision, so each
    // input interval is rounded OUTWARD to the compute grid before use. We build
    // a per-position rounded input interval when the input vector length matches
    // in_features (the clean single-vector case); otherwise we fall back to a
    // GLOBAL rounded max-abs applied to every term — a sound over-approximation
    // (replacing each input interval by [-M, M] only widens the result).
    let in_lower = in_tensor.lower();
    let in_upper = in_tensor.upper();
    let per_position = in_lower.len() == in_features && in_features > 0;

    let rounded_inputs: Vec<(f32, f32)> = if per_position {
        in_lower
            .iter()
            .zip(in_upper.iter())
            .map(|(&l, &u)| ny_core::round_to_precision_outward(l, u, compute_p))
            .collect()
    } else {
        // Global rounded max-abs M; every term sees [-M, M] on the compute grid.
        let m_raw = max_abs_over(in_lower, in_upper);
        let (lo, hi) = ny_core::round_to_precision_outward(-m_raw, m_raw, compute_p);
        vec![(lo, hi); in_features]
    };

    // Output element count and the row mapping (last axis is out_features).
    let out_len = out.lower().len();
    let layout_is_standard = out_features > 0 && out_len.is_multiple_of(out_features);

    // Per-row deployed interval [yl, yu], then map to the flat output (falling
    // back to the worst row when the layout is non-standard).
    struct RowResult {
        yl: f32,
        yu: f32,
    }
    let mut rows: Vec<RowResult> = Vec::with_capacity(out_features);

    // ORDER-INDEPENDENT accumulation constants (depend only on the accumulate
    // precision, not on the row): the overflow threshold, the unit roundoff, and
    // the per-store subnormal floor (half the smallest subnormal). All computed
    // exactly in f64 so the slack `E` never underflows in the subnormal regime.
    let m_acc = f64::from(accumulate_p.max_normal());
    let u_acc = f64::from(accumulate_p.unit_roundoff());
    let half_sub_acc = f64::from(accumulate_p.smallest_subnormal()) * 0.5;

    for j in 0..out_features {
        // (3) Build the per-term product intervals on the compute grid, plus the
        // order-independent reachability sums. We do NOT fold left-to-right (that
        // is unsound across reduction orders, H1); we collect the exact-sum
        // endpoints and the worst reachable partial sums [neg, pos] instead.
        let mut sum_lo64 = 0.0_f64; // exact f64 sum of tl_i
        let mut sum_hi64 = 0.0_f64; // exact f64 sum of th_i
        let mut pos64 = 0.0_f64; // sum of max(0, th_i): largest reachable partial sum
        let mut neg64 = 0.0_f64; // sum of min(0, tl_i): most negative reachable partial sum
        let mut n_terms: usize = 0;

        // Fold one already-formed term interval [tl, th] into the accumulators.
        let mut add_term = |tl: f32, th: f32| {
            n_terms += 1;
            let tl = f64::from(tl);
            let th = f64::from(th);
            sum_lo64 += tl;
            sum_hi64 += th;
            pos64 += th.max(0.0);
            neg64 += tl.min(0.0);
        };

        for i in 0..in_features {
            // (2) WEIGHT rounding to the compute grid: the deployed stored weight
            // round_nearest(w) lies inside [wl, wu].
            let w = weight[[j, i]];
            let (wl, wu) = ny_core::round_to_precision_outward(w, w, compute_p);
            let (xl, xu) = rounded_inputs[i];

            // (3) PER-TERM PRODUCT interval: hull of the 4 corner products in f64,
            // narrowed to f32 OUTWARD, then rounded OUTWARD to the compute grid
            // (per-product overflow -> ±inf here).
            let c1 = f64::from(wl) * f64::from(xl);
            let c2 = f64::from(wl) * f64::from(xu);
            let c3 = f64::from(wu) * f64::from(xl);
            let c4 = f64::from(wu) * f64::from(xu);

            // H2 (NaN / inf*0): a NaN corner (inf*0 / 0*inf / inf-inf) or a NaN
            // input/weight endpoint means the deployed product is UNCONSTRAINED.
            // f64 min/max would silently DROP the NaN and collapse the term to a
            // finite interval, so check is_nan explicitly on every corner and
            // endpoint FIRST. When unconstrained, the term is [-inf, +inf].
            let any_nan = c1.is_nan()
                || c2.is_nan()
                || c3.is_nan()
                || c4.is_nan()
                || wl.is_nan()
                || wu.is_nan()
                || xl.is_nan()
                || xu.is_nan();
            let (tl, th) = if any_nan {
                (f32::NEG_INFINITY, f32::INFINITY)
            } else {
                let term_lo64 = c1.min(c2).min(c3).min(c4);
                let term_hi64 = c1.max(c2).max(c3).max(c4);
                let term = (
                    f64_to_f32_outward(term_lo64, false),
                    f64_to_f32_outward(term_hi64, true),
                );
                ny_core::round_to_precision_outward(term.0, term.1, compute_p)
            };
            add_term(tl, th);
        }

        // Bias: one extra term, bracketed on the compute grid (no product). A NaN
        // bias bracket (from an overflow corner) is unconstrained too.
        if let Some(bias) = lin.bias.as_ref() {
            if let Some(&b) = bias.get(j) {
                let (bl, bu) = ny_core::round_to_precision_outward(b, b, compute_p);
                if bl.is_nan() || bu.is_nan() {
                    add_term(f32::NEG_INFINITY, f32::INFINITY);
                } else {
                    add_term(bl, bu);
                }
            }
        }

        // (4) ORDER-INDEPENDENT OVERFLOW GUARD (H1): in ANY reduction order every
        // partial sum lies in [neg, pos]. If that range can reach the accumulate
        // overflow threshold (or any sum is already non-finite, e.g. a per-product
        // +inf term), SOME legal order rounds a partial sum to ±inf -> widen.
        let overflow = !sum_lo64.is_finite()
            || !sum_hi64.is_finite()
            || !pos64.is_finite()
            || !neg64.is_finite()
            || pos64 >= m_acc
            || neg64 <= -m_acc;

        let (yl, yu) = if overflow {
            (f32::NEG_INFINITY, f32::INFINITY)
        } else {
            // (5) ACCUMULATION SLACK E, sound for any order. A left fold of N terms
            // performs N accumulate-grid stores: acc = round(0 + t0) is itself a
            // rounding step (t0 was computed at compute precision and may not be
            // representable in a coarser accumulate grid), then one store per
            // subsequent term. So K = N stores (not N-1); N == 0 charges nothing.
            // K = N over-approximates any reduction order's store count, so it is
            // sound for every order (5th-audit fix: was N-1, undercharged by one).
            let k = n_terms as f64;
            let max_mag = pos64.max(-neg64); // both >= 0 here
            let e_rel = k * u_acc * max_mag;
            let e_floor = k * half_sub_acc;
            let e64 = e_rel.max(e_floor);

            // (6) Pre-store interval [sum_lo - E, sum_hi + E], narrowed OUTWARD to
            // f32, then rounded OUTWARD to the output store grid.
            let lo64 = sum_lo64 - e64;
            let hi64 = sum_hi64 + e64;
            let lo = f64_to_f32_outward(lo64, false);
            let hi = f64_to_f32_outward(hi64, true);
            ny_core::round_to_precision_outward(lo, hi, output_store_p)
        };

        rows.push(RowResult { yl, yu });
    }

    // Worst-case row (for the non-standard-layout fallback): widest interval,
    // NaN-safe (a NaN endpoint can only arise from an +inf/-inf product corner;
    // treat it as the matching infinity so it never narrows the hull).
    let worst_yl = rows.iter().map(|r| r.yl).fold(f32::INFINITY, |m, v| {
        if v.is_nan() {
            f32::NEG_INFINITY
        } else {
            m.min(v)
        }
    });
    let worst_yu = rows.iter().map(|r| r.yu).fold(f32::NEG_INFINITY, |m, v| {
        if v.is_nan() {
            f32::INFINITY
        } else {
            m.max(v)
        }
    });

    // Place each output element. The recomputed [yl, yu] REPLACES the idealized
    // `out` interval (it already accounts for all deployed rounding); `out` is
    // only a shape/length reference for placement.
    let mut new_lower = out.lower().clone();
    let mut new_upper = out.upper().clone();
    for (flat_idx, (lo, up)) in new_lower.iter_mut().zip(new_upper.iter_mut()).enumerate() {
        let (yl, yu) = if layout_is_standard {
            let r = &rows[flat_idx % out_features];
            (r.yl, r.yu)
        } else {
            // Non-standard layout: cannot reliably map element -> row, so use the
            // worst row for EVERY element (sound over-approximation).
            (worst_yl, worst_yu)
        };
        // A NaN endpoint (only from +inf/-inf product corners) widens to the
        // matching infinity — never a finite value that could exclude a real one.
        *lo = if yl.is_nan() { f32::NEG_INFINITY } else { yl };
        *up = if yu.is_nan() { f32::INFINITY } else { yu };
    }

    BoundedTensor::new_allow_infinite(new_lower, new_upper)
}

/// Maximum |v| over the union of `lower` and `upper` arrays (sound, rounded up).
fn max_abs_over(lower: &ndarray::ArrayD<f32>, upper: &ndarray::ArrayD<f32>) -> f32 {
    let mut m = 0.0_f32;
    for &v in lower.iter() {
        if v.abs() > m {
            m = v.abs();
        }
    }
    for &v in upper.iter() {
        if v.abs() > m {
            m = v.abs();
        }
    }
    ny_tensor::rounding::next_up_f32(m)
}

/// Flatten a [`BoundedTensor`] to a row-major `Vec<Bound>` (infinity-admitting).
fn bounded_tensor_to_bounds(t: &BoundedTensor) -> Vec<Bound> {
    let lower = t.lower();
    let upper = t.upper();
    lower
        .iter()
        .zip(upper.iter())
        .map(|(&l, &u)| Bound::new_allow_infinite(l, u))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_core::{MethodUsed, VerificationSoundnessMode};

    /// A `Verified` result carrying the given output bounds and a sound provenance.
    fn verified(bounds: Vec<Bound>) -> VerificationResult {
        VerificationResult::Verified {
            provenance: SoundnessProvenance::sound(),
            output_bounds: bounds,
            proof: None,
            actual_method: Some(MethodUsed::Ibp),
        }
    }

    /// A spec whose required output property is `required` (input bounds are a
    /// throwaway since these tests exercise only the output re-check).
    fn spec_requiring(required: Vec<Bound>) -> VerificationSpec {
        VerificationSpec::new(vec![Bound::new(-1.0, 1.0)], required).unwrap()
    }

    #[test]
    fn f32_policy_is_strict_noop_on_bounds() {
        let policy = MixedPrecisionPolicy::default();
        assert!(policy.is_idealized_f32());
        let original = vec![Bound::new(0.1, 0.2), Bound::new(-1.0 / 3.0, 2.7182817)];
        let mut bounds = original.clone();
        widen_bounds_for_policy(&mut bounds, &policy);
        for (got, exp) in bounds.iter().zip(original.iter()) {
            assert_eq!(got.lower().to_bits(), exp.lower().to_bits());
            assert_eq!(got.upper().to_bits(), exp.upper().to_bits());
        }
        let owned = widen_bounds_for_policy_owned(&original, &policy);
        for (got, exp) in owned.iter().zip(original.iter()) {
            assert_eq!(got.lower().to_bits(), exp.lower().to_bits());
            assert_eq!(got.upper().to_bits(), exp.upper().to_bits());
        }
    }

    #[test]
    fn f32_policy_result_is_unchanged_identity() {
        // The whole result (variant, bounds, provenance) must be byte-identical.
        let policy = MixedPrecisionPolicy::default();
        let spec = spec_requiring(vec![Bound::new(-10.0, 10.0)]);
        let result = verified(vec![Bound::new(0.1, 0.2)]);
        let out = widen_result_for_policy(result, &spec, &policy);
        match out {
            VerificationResult::Verified {
                output_bounds,
                provenance,
                ..
            } => {
                assert_eq!(output_bounds.len(), 1);
                assert_eq!(output_bounds[0].lower().to_bits(), 0.1_f32.to_bits());
                assert_eq!(output_bounds[0].upper().to_bits(), 0.2_f32.to_bits());
                // Provenance stays Sound under the f32 no-op (no heuristic marker).
                assert_eq!(provenance.mode(), VerificationSoundnessMode::Sound);
            }
            other => panic!("expected Verified, got {other:?}"),
        }
    }

    #[test]
    fn f16_policy_widens_outward_never_narrows() {
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);
        assert_eq!(effective_widening_precision(&policy), FloatPrecision::F16);
        // Values not exactly representable in f16 so widening is observable.
        let original = vec![Bound::new(0.1, 0.2), Bound::new(-2.7182817, 3.1400003)];
        let widened = widen_bounds_for_policy_owned(&original, &policy);
        let mut any_strictly_wider = false;
        for (w, o) in widened.iter().zip(original.iter()) {
            assert!(w.lower() <= o.lower(), "lower must not increase");
            assert!(w.upper() >= o.upper(), "upper must not decrease");
            if w.lower() < o.lower() || w.upper() > o.upper() {
                any_strictly_wider = true;
            }
        }
        assert!(
            any_strictly_wider,
            "f16 widening should strictly widen at least one non-representable endpoint"
        );
    }

    #[test]
    fn mixed_policy_uses_coarser_precision() {
        // f16 compute + f32 accumulate => coarser is f16 (still widens).
        let policy = MixedPrecisionPolicy::new(FloatPrecision::F16, FloatPrecision::F32);
        assert_eq!(effective_widening_precision(&policy), FloatPrecision::F16);
        // bf16 compute + f16 accumulate => coarser is bf16.
        let policy2 = MixedPrecisionPolicy::new(FloatPrecision::Bf16, FloatPrecision::F16);
        assert_eq!(effective_widening_precision(&policy2), FloatPrecision::Bf16);
    }

    #[test]
    fn verdict_with_margin_survives_widening() {
        // Output bounds [0.1, 0.2] with a generous requirement [-10, 10]: widening
        // by a few f16 ULPs cannot escape the requirement, so it stays Verified.
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);
        let spec = spec_requiring(vec![Bound::new(-10.0, 10.0)]);
        let result = verified(vec![Bound::new(0.1, 0.2)]);
        let out = widen_result_for_policy(result, &spec, &policy);
        match out {
            VerificationResult::Verified {
                output_bounds,
                provenance,
                ..
            } => {
                // Bounds are the widened (deployed-precision) ones, still inside req.
                assert!(output_bounds[0].lower() >= -10.0);
                assert!(output_bounds[0].upper() <= 10.0);
                // The precision-widening heuristic label is recorded.
                assert_eq!(provenance.mode(), VerificationSoundnessMode::Heuristic);
            }
            other => panic!("expected Verified with margin, got {other:?}"),
        }
    }

    #[test]
    fn borderline_verdict_flips_to_unknown_not_falsely_verified() {
        // Construct a borderline case: the computed upper bound sits EXACTLY on the
        // requirement's upper edge for a value that is NOT f16-representable, so
        // outward widening pushes it strictly past the edge and the verdict must
        // downgrade to Unknown (never stay falsely Verified).
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);
        // 0.1 is not representable in f16; round-up gives a strictly larger value.
        let edge = 0.1_f32;
        let widened_edge = ny_core::widen_bound(&Bound::new(edge, edge), FloatPrecision::F16);
        assert!(
            widened_edge.upper() > edge,
            "test setup: f16 widening must push the edge strictly up"
        );
        // f32 verdict holds exactly: computed [0.0, edge] within required [0.0, edge].
        let spec = spec_requiring(vec![Bound::new(0.0, edge)]);
        let result = verified(vec![Bound::new(0.0, edge)]);
        // Sanity: it WOULD be Verified under the f32 (no-op) policy.
        let f32_out =
            widen_result_for_policy(result.clone(), &spec, &MixedPrecisionPolicy::default());
        assert!(
            f32_out.is_verified(),
            "f32 borderline case must be Verified"
        );
        // Under f16 it must flip to Unknown.
        let out = widen_result_for_policy(result, &spec, &policy);
        match out {
            VerificationResult::Unknown {
                reason, provenance, ..
            } => {
                assert!(
                    matches!(reason, UnknownReason::BoundsTooLoose { .. }),
                    "expected BoundsTooLoose, got {reason:?}"
                );
                assert_eq!(provenance.mode(), VerificationSoundnessMode::Heuristic);
            }
            other => panic!("borderline f16 verdict must flip to Unknown, got {other:?}"),
        }
    }

    #[test]
    fn lower_edge_borderline_also_flips() {
        // Symmetric check on the lower endpoint: computed lower exactly on the
        // requirement's lower edge, value not f16-representable, widening pushes it
        // below the edge => Unknown.
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);
        let edge = -0.1_f32;
        let spec = spec_requiring(vec![Bound::new(edge, 1.0)]);
        let result = verified(vec![Bound::new(edge, 0.5)]);
        assert!(
            widen_result_for_policy(result.clone(), &spec, &MixedPrecisionPolicy::default())
                .is_verified()
        );
        let out = widen_result_for_policy(result, &spec, &policy);
        assert!(
            matches!(out, VerificationResult::Unknown { .. }),
            "lower-edge borderline must flip to Unknown"
        );
    }

    #[test]
    fn violated_and_timeout_pass_through_unchanged() {
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::Bf16);
        let spec = spec_requiring(vec![Bound::new(0.0, 1.0)]);

        let violated = VerificationResult::Violated {
            provenance: SoundnessProvenance::sound(),
            counterexample: vec![0.5],
            output: vec![2.0],
            details: None,
            actual_method: Some(MethodUsed::Ibp),
        };
        let out = widen_result_for_policy(violated, &spec, &policy);
        assert!(matches!(out, VerificationResult::Violated { .. }));

        let timeout = VerificationResult::Timeout {
            provenance: SoundnessProvenance::sound(),
            partial_bounds: Some(vec![Bound::new(0.0, 1.0)]),
            actual_method: Some(MethodUsed::Ibp),
        };
        let out = widen_result_for_policy(timeout, &spec, &policy);
        assert!(matches!(out, VerificationResult::Timeout { .. }));
    }

    #[test]
    fn unknown_stays_unknown_with_widened_bounds() {
        // An f32 Unknown (bounds outside requirement) stays Unknown; widening only
        // loosens, never rescues.
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);
        let spec = spec_requiring(vec![Bound::new(0.0, 1.0)]);
        let unknown = VerificationResult::Unknown {
            provenance: SoundnessProvenance::sound(),
            bounds: vec![Bound::new(-5.0, 5.0)], // already outside [0,1]
            reason: UnknownReason::BoundsTooLoose { gap: Some(4.0) },
            actual_method: Some(MethodUsed::Ibp),
        };
        let out = widen_result_for_policy(unknown, &spec, &policy);
        assert!(matches!(out, VerificationResult::Unknown { .. }));
    }

    #[test]
    fn representation_only_path_never_labels_sound_for_non_f32() {
        // PART A honesty: for ANY non-F32 policy, the representation-only path must
        // label its verdict Heuristic — never a Sound "verified at deployed
        // precision". Check both surviving-Verified and flipped-Unknown outcomes.
        for p in [FloatPrecision::F16, FloatPrecision::Bf16] {
            let policy = MixedPrecisionPolicy::uniform(p);
            // (a) Survives as Verified with margin -> Heuristic.
            let spec = spec_requiring(vec![Bound::new(-10.0, 10.0)]);
            let out = widen_result_for_policy(verified(vec![Bound::new(0.1, 0.2)]), &spec, &policy);
            let prov = match &out {
                VerificationResult::Verified { provenance, .. } => provenance,
                VerificationResult::Unknown { provenance, .. } => provenance,
                other => panic!("unexpected variant {other:?}"),
            };
            assert_eq!(
                prov.mode(),
                VerificationSoundnessMode::Heuristic,
                "p={p:?}: representation-only verdict must be Heuristic, not Sound"
            );
        }
    }

    #[test]
    fn representation_only_widening_under_models_accumulation_5000_ones() {
        // PART A honesty (the bug, pinned): representation-only widening of the
        // f32-idealized point bound [5000, 5000] does NOT contain the deployed f16
        // value (~2048). This is exactly why the representation-only verdict cannot
        // be claimed Sound, and why the sound accumulation path is required.
        let widened = ny_core::widen_bound(&Bound::new(5000.0, 5000.0), FloatPrecision::F16);
        // A few f16 ULPs at 5000 is ~4, nowhere near reaching 2048.
        assert!(
            widened.lower() > 2048.0,
            "representation-only widening [{}, {}] wrongly excludes the deployed ~2048 — \
             demonstrating it under-models accumulation",
            widened.lower(),
            widened.upper()
        );
        // The SOUND accumulation primitive, by contrast, reaches +inf here.
        let acc = ny_core::summation_error_bound(5000.0, 5000, FloatPrecision::F16);
        assert!(
            acc.is_infinite(),
            "sound accumulation bound saturates to +inf for n=5000 f16"
        );
    }

    // =====================================================================
    // PART D rev2 ACCEPTANCE SUITE — verify_with_sound_precision Linear path.
    //
    // Each test computes the REAL deployed Linear value with the half crate
    // (rounding inputs/weights/products to compute precision and accumulating in
    // accumulate precision) and asserts the SOUND widened output interval CONTAINS
    // it. This pins F1 (per-term product rounding), F2 (overflow to ±inf), F3
    // (input/weight representation rounding), and F4 (subnormal floor), plus the
    // original three drift cases.
    // =====================================================================

    use half::{bf16, f16};
    use ndarray::{Array1, Array2};
    use ny_propagate::layers::Layer;
    use ny_propagate::layers::LinearLayer;

    /// Round a single f32 to precision `p` (round-to-nearest), as the deployed
    /// hardware stores/uses a value.
    fn rn(x: f32, p: FloatPrecision) -> f32 {
        match p {
            FloatPrecision::F32 => x,
            FloatPrecision::F16 => f16::from_f32(x).to_f32(),
            FloatPrecision::Bf16 => bf16::from_f32(x).to_f32(),
        }
    }

    /// Compute the REAL deployed Linear output for ONE output row `j` at a fixed
    /// input point `x` (length in_features), under a mixed-precision policy:
    ///   - inputs and weights are rounded into `compute_p`,
    ///   - each product round(W)*round(x) is rounded into `compute_p`,
    ///   - the running sum is accumulated left-to-right in `accumulate_p`,
    ///   - the bias (rounded into `compute_p`) is added last in `accumulate_p`.
    ///
    /// Returns the f32 view of the deployed accumulator.
    fn deployed_linear_row(
        weight: &Array2<f32>,
        bias: Option<&Array1<f32>>,
        x: &[f32],
        j: usize,
        compute_p: FloatPrecision,
        accumulate_p: FloatPrecision,
    ) -> f32 {
        let in_features = weight.ncols();
        let mut acc = rn(0.0, accumulate_p);
        for i in 0..in_features {
            let w = rn(weight[[j, i]], compute_p);
            let xi = rn(x[i], compute_p);
            // Product is formed and rounded into the compute precision.
            let prod = rn(w * xi, compute_p);
            // Accumulate in the accumulate precision.
            acc = rn(acc + prod, accumulate_p);
        }
        if let Some(b) = bias {
            let bj = rn(b[j], compute_p);
            acc = rn(acc + bj, accumulate_p);
        }
        acc
    }

    /// Build a single-Linear GraphNetwork named "linear" (network input -> linear).
    fn single_linear_graph(weight: Array2<f32>, bias: Option<Array1<f32>>) -> GraphNetwork {
        let lin = LinearLayer::new(weight, bias).expect("valid linear layer");
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input("linear", Layer::Linear(lin)));
        g.set_output("linear");
        g
    }

    /// Run verify_with_sound_precision and return the computed (deployed) output
    /// bounds, regardless of Verified/Unknown verdict. Uses a permissive output
    /// requirement so the verdict re-check never masks the bounds we want to check.
    fn sound_output_bounds(
        weight: Array2<f32>,
        bias: Option<Array1<f32>>,
        input: Vec<Bound>,
        policy: &MixedPrecisionPolicy,
    ) -> Vec<Bound> {
        let out_features = weight.nrows();
        let graph = single_linear_graph(weight, bias);
        // Permissive requirement: one [-inf, +inf] per output so decide_against_required
        // returns the computed bounds whatever they are. (We assert containment on the
        // returned bounds directly.)
        let required: Vec<Bound> = (0..out_features)
            .map(|_| Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY))
            .collect();
        let spec = VerificationSpec::new(input, required).expect("valid spec");
        let result =
            verify_with_sound_precision(&graph, &spec, policy).expect("sound precision run");
        match result {
            VerificationResult::Verified { output_bounds, .. } => output_bounds,
            VerificationResult::Unknown { bounds, .. } => bounds,
            other => panic!("expected Verified/Unknown with bounds, got {other:?}"),
        }
    }

    /// Assert that `bounds[j]` contains `deployed` (with -inf/+inf admitted).
    fn assert_contains(bounds: &[Bound], j: usize, deployed: f32, label: &str) {
        let b = &bounds[j];
        let lo = b.lower();
        let hi = b.upper();
        assert!(
            lo <= deployed && deployed <= hi,
            "{label}: deployed {deployed} NOT in widened [{lo}, {hi}] (row {j})"
        );
    }

    #[test]
    fn acceptance_f1_per_term_product_rounding_bf16_compute_f32_accumulate() {
        // F1: compute=bf16, accumulate=f32, in_features=199. 100 weights = 63.36
        // (bf16-inexact), 99 weights = -64.0 (bf16-exact), input all 1. The
        // idealized f32 sum is ~6.1e-5 but the DEPLOYED value differs by far more
        // because each product 63.36*1 is rounded to bf16 BEFORE summation.
        let in_features = 199usize;
        let mut wv = vec![0.0_f32; in_features];
        for w in wv.iter_mut().take(100) {
            *w = 63.36;
        }
        for w in wv.iter_mut().skip(100) {
            *w = -64.0;
        }
        let weight = Array2::from_shape_vec((1, in_features), wv).unwrap();
        let policy = MixedPrecisionPolicy::new(FloatPrecision::Bf16, FloatPrecision::F32);

        // Deployed value at the input point all-ones.
        let x = vec![1.0_f32; in_features];
        let deployed = deployed_linear_row(
            &weight,
            None,
            &x,
            0,
            FloatPrecision::Bf16,
            FloatPrecision::F32,
        );

        // Idealized f32 sum for context (should be tiny, ~6.1e-5).
        let ideal: f32 = (0..in_features).map(|i| weight[[0, i]]).sum();
        assert!(
            ideal.abs() < 1.0,
            "idealized sum should be tiny, got {ideal}"
        );
        // The deployed value escapes the tiny idealized region (per-term rounding).
        assert!(
            (deployed - ideal).abs() > 1.0,
            "F1 setup: deployed {deployed} should differ from ideal {ideal} by >1"
        );

        let input = vec![Bound::new(1.0, 1.0); in_features];
        let bounds = sound_output_bounds(weight, None, input, &policy);
        assert_contains(&bounds, 0, deployed, "F1 bf16/f32 in=199");
    }

    #[test]
    fn acceptance_f2_running_sum_overflow_to_inf_f16() {
        // F2: f16, 100 terms of +1000 then 100 of -1000. The f16-idealized output
        // is 0 but a deployed PARTIAL sum overflows f16 to +inf and stays there.
        // The sound widened bound MUST contain +inf.
        let in_features = 200usize;
        let mut wv = vec![0.0_f32; in_features];
        for w in wv.iter_mut().take(100) {
            *w = 1000.0;
        }
        for w in wv.iter_mut().skip(100) {
            *w = -1000.0;
        }
        let weight = Array2::from_shape_vec((1, in_features), wv).unwrap();
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);

        let x = vec![1.0_f32; in_features];
        let deployed = deployed_linear_row(
            &weight,
            None,
            &x,
            0,
            FloatPrecision::F16,
            FloatPrecision::F16,
        );
        assert!(
            deployed.is_infinite(),
            "F2 setup: deployed f16 partial-sum should overflow to ±inf, got {deployed}"
        );

        let input = vec![Bound::new(1.0, 1.0); in_features];
        let bounds = sound_output_bounds(weight, None, input, &policy);
        // The widened interval must admit the deployed +inf.
        let b = &bounds[0];
        assert!(
            b.upper().is_infinite() && b.upper() > 0.0,
            "F2: widened upper must be +inf to contain deployed +inf, got {}",
            b.upper()
        );
        assert_contains(&bounds, 0, deployed, "F2 f16 overflow");
    }

    #[test]
    fn acceptance_f3_input_weight_representation_rounding_f16() {
        // F3: f16, n=2, W=[0.07800007, -2.267], x=[-1.177, -2.167] (all off-grid).
        // Inputs and weights are rounded before use, inflating the product; the
        // deployed value must stay inside the widened interval.
        let weight = Array2::from_shape_vec((1, 2), vec![0.078_000_07_f32, -2.267]).unwrap();
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);
        let x = vec![-1.177_f32, -2.167];

        let deployed = deployed_linear_row(
            &weight,
            None,
            &x,
            0,
            FloatPrecision::F16,
            FloatPrecision::F16,
        );

        // The input interval is the point [x_i, x_i] for each i.
        let input = vec![Bound::new(x[0], x[0]), Bound::new(x[1], x[1])];
        let bounds = sound_output_bounds(weight, None, input, &policy);
        assert_contains(&bounds, 0, deployed, "F3 f16 off-grid rep rounding");
    }

    #[test]
    fn acceptance_f3_interval_inputs_contained_random() {
        // Stronger F3: non-degenerate input intervals. For many random off-grid
        // weights and input intervals, EVERY deployed value at the interval corners
        // (and a few interior points) must be inside the widened bound.
        // Deterministic xorshift64* returning a uniform f64 in [0, 1).
        fn unit(seed: &mut u64) -> f64 {
            let mut x = *seed;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            *seed = x;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        }
        fn frand(seed: &mut u64, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * (unit(seed) as f32)
        }
        let mut seed = 0x51A1_7E55_u64 ^ 0xABCD;

        for p in [FloatPrecision::F16, FloatPrecision::Bf16] {
            let policy = MixedPrecisionPolicy::uniform(p);
            for _ in 0..200 {
                let in_features = 1 + (unit(&mut seed) * 6.0) as usize;
                let mut wv = Vec::with_capacity(in_features);
                let mut lo = Vec::with_capacity(in_features);
                let mut hi = Vec::with_capacity(in_features);
                for _ in 0..in_features {
                    wv.push(frand(&mut seed, -3.0, 3.0));
                    let a = frand(&mut seed, -2.0, 2.0);
                    let b = frand(&mut seed, -2.0, 2.0);
                    lo.push(a.min(b));
                    hi.push(a.max(b));
                }
                let weight = Array2::from_shape_vec((1, in_features), wv.clone()).unwrap();
                let input: Vec<Bound> = lo
                    .iter()
                    .zip(hi.iter())
                    .map(|(&l, &h)| Bound::new(l, h))
                    .collect();
                let bounds = sound_output_bounds(weight.clone(), None, input, &policy);

                // Probe deployed values at the 2^k corners is exponential; instead
                // probe each-axis-low / each-axis-high plus all-low/all-high.
                let probes: Vec<Vec<f32>> = {
                    let mut v = vec![lo.clone(), hi.clone()];
                    for i in 0..in_features {
                        let mut p_lo = hi.clone();
                        p_lo[i] = lo[i];
                        let mut p_hi = lo.clone();
                        p_hi[i] = hi[i];
                        v.push(p_lo);
                        v.push(p_hi);
                    }
                    v
                };
                for x in &probes {
                    let deployed = deployed_linear_row(&weight, None, x, 0, p, p);
                    assert_contains(&bounds, 0, deployed, "F3 random interval");
                }
            }
        }
    }

    #[test]
    fn acceptance_f4_tiny_subnormal_reductions_contained() {
        // F4: tiny/subnormal-magnitude Linear reductions. Build rows whose products
        // are subnormal-scale so the relative gamma_N term collapses; the subnormal
        // floor must keep the deployed value contained across several N.
        for p in [FloatPrecision::F16, FloatPrecision::Bf16] {
            let s = p.smallest_subnormal();
            let policy = MixedPrecisionPolicy::uniform(p);
            for &n in &[2usize, 4, 8, 16, 64, 100] {
                // Weight = small multiple of subnormal scale; input = 1 (exact).
                // Products land in the subnormal region.
                let wv: Vec<f32> = (0..n).map(|k| s * ((k % 5) as f32 + 1.0)).collect();
                let weight = Array2::from_shape_vec((1, n), wv.clone()).unwrap();
                let input = vec![Bound::new(1.0, 1.0); n];
                let bounds = sound_output_bounds(weight.clone(), None, input, &policy);
                let x = vec![1.0_f32; n];
                let deployed = deployed_linear_row(&weight, None, &x, 0, p, p);
                assert_contains(&bounds, 0, deployed, "F4 subnormal reduction");
                // Also probe a tiny negative-mix variant.
                let wv2: Vec<f32> = (0..n).map(|k| if k % 2 == 0 { s } else { -s }).collect();
                let weight2 = Array2::from_shape_vec((1, n), wv2).unwrap();
                let input2 = vec![Bound::new(1.0, 1.0); n];
                let bounds2 = sound_output_bounds(weight2.clone(), None, input2, &policy);
                let deployed2 = deployed_linear_row(&weight2, None, &x, 0, p, p);
                assert_contains(&bounds2, 0, deployed2, "F4 subnormal alternating");
            }
        }
    }

    #[test]
    fn acceptance_original_5000_ones_f16_contained() {
        // Original drift case 1: 5000 ones in f16 (uniform precision). Deployed
        // saturates near 2048; widened bound must contain it.
        let n = 5000usize;
        let weight = Array2::from_shape_vec((1, n), vec![1.0_f32; n]).unwrap();
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);
        let input = vec![Bound::new(1.0, 1.0); n];
        let bounds = sound_output_bounds(weight.clone(), None, input, &policy);
        let x = vec![1.0_f32; n];
        let deployed = deployed_linear_row(
            &weight,
            None,
            &x,
            0,
            FloatPrecision::F16,
            FloatPrecision::F16,
        );
        assert!(
            deployed < 2100.0,
            "f16 5000-ones should saturate, got {deployed}"
        );
        assert_contains(&bounds, 0, deployed, "original 5000 ones f16");
    }

    #[test]
    fn acceptance_original_4096_term_f16_contained() {
        // Original drift case 2: 4096-term f16 reduction (n*u >= 1 -> +inf regime
        // for accumulation). Deployed contained.
        let n = 4096usize;
        // Mixed weights so the sum is non-trivial but the accumulation drift is real.
        let wv: Vec<f32> = (0..n)
            .map(|k| if k % 2 == 0 { 0.5 } else { -0.25 })
            .collect();
        let weight = Array2::from_shape_vec((1, n), wv).unwrap();
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);
        let input = vec![Bound::new(1.0, 1.0); n];
        let bounds = sound_output_bounds(weight.clone(), None, input, &policy);
        let x = vec![1.0_f32; n];
        let deployed = deployed_linear_row(
            &weight,
            None,
            &x,
            0,
            FloatPrecision::F16,
            FloatPrecision::F16,
        );
        assert_contains(&bounds, 0, deployed, "original 4096-term f16");
    }

    #[test]
    fn acceptance_original_512_term_bf16_contained() {
        // Original drift case 3: 512-term bf16 reduction.
        let n = 512usize;
        let wv: Vec<f32> = (0..n).map(|k| 0.55 + 0.01 * (k as f32 % 7.0)).collect();
        let weight = Array2::from_shape_vec((1, n), wv).unwrap();
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::Bf16);
        let input = vec![Bound::new(1.0, 1.0); n];
        let bounds = sound_output_bounds(weight.clone(), None, input, &policy);
        let x = vec![1.0_f32; n];
        let deployed = deployed_linear_row(
            &weight,
            None,
            &x,
            0,
            FloatPrecision::Bf16,
            FloatPrecision::Bf16,
        );
        assert_contains(&bounds, 0, deployed, "original 512-term bf16");
    }

    #[test]
    fn acceptance_f32_policy_sound_path_is_noop_identity() {
        // F32 policy: strict no-op delegating to the normal verifier. A property
        // the f32 idealization satisfies stays Verified, byte-identical bounds.
        let weight = Array2::from_shape_vec((1, 3), vec![1.0_f32, 1.0, 1.0]).unwrap();
        let graph = single_linear_graph(weight, Some(Array1::zeros(1)));
        // Input each in [0, 1] -> output in [0, 3]. Require [-1, 4]: IBP-provable.
        let spec = VerificationSpec::new(
            vec![
                Bound::new(0.0, 1.0),
                Bound::new(0.0, 1.0),
                Bound::new(0.0, 1.0),
            ],
            vec![Bound::new(-1.0, 4.0)],
        )
        .unwrap();
        let policy = MixedPrecisionPolicy::default();
        assert!(policy.is_idealized_f32());
        let result = verify_with_sound_precision(&graph, &spec, &policy).unwrap();
        assert!(
            result.is_verified(),
            "f32 no-op should reproduce the normal Verified verdict"
        );
        match result {
            VerificationResult::Verified { provenance, .. } => {
                // F32 path delegates to verify_graph; provenance is the engine's
                // own (Sound), NOT downgraded.
                assert_eq!(provenance.mode(), VerificationSoundnessMode::Sound);
            }
            other => panic!("expected Verified, got {other:?}"),
        }
    }

    #[test]
    fn acceptance_sound_path_never_false_verifies_coarse_case() {
        // Soundness contract: a coarse deployed case whose deployed output violates
        // a tight property must yield Unknown (or be contained), NEVER a false
        // Verify. We use the 5000-ones f16 case with a TIGHT requirement [4990, 5010]
        // that the f32 idealization (5000) satisfies but the deployed (~2048) does
        // NOT. The sound path must not return Verified.
        let n = 5000usize;
        let weight = Array2::from_shape_vec((1, n), vec![1.0_f32; n]).unwrap();
        let graph = single_linear_graph(weight, None);
        let input = vec![Bound::new(1.0, 1.0); n];
        let spec = VerificationSpec::new(input, vec![Bound::new(4990.0, 5010.0)]).unwrap();
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);
        let result = verify_with_sound_precision(&graph, &spec, &policy).unwrap();
        assert!(
            !result.is_verified(),
            "sound path must NOT Verify a tight property the deployed net violates: {result:?}"
        );
        assert!(
            matches!(result, VerificationResult::Unknown { .. }),
            "expected Unknown for the coarse f16 case, got {result:?}"
        );
    }

    #[test]
    fn acceptance_sound_path_verifies_with_real_margin() {
        // The sound path CAN Verify when the requirement is loose enough to survive
        // the (real, large) deployed widening. 5000-ones f16 with a generous
        // requirement [-inf, +inf] stays Verified AND is labeled Sound (single
        // Linear, fully accounted).
        let n = 5000usize;
        let weight = Array2::from_shape_vec((1, n), vec![1.0_f32; n]).unwrap();
        let graph = single_linear_graph(weight, None);
        let input = vec![Bound::new(1.0, 1.0); n];
        let spec = VerificationSpec::new(
            input,
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        )
        .unwrap();
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);
        let result = verify_with_sound_precision(&graph, &spec, &policy).unwrap();
        match result {
            VerificationResult::Verified { provenance, .. } => {
                assert_eq!(
                    provenance.mode(),
                    VerificationSoundnessMode::Sound,
                    "single fully-accounted Linear should be labeled Sound"
                );
            }
            other => panic!("expected Verified with Sound provenance, got {other:?}"),
        }
    }

    // =====================================================================
    // rev3 ACCEPTANCE — DIRECT INTERVAL ARITHMETIC corners (3rd audit).
    //
    // These pin the exact corners the error-ESTIMATE approach kept getting
    // wrong and that direct outward-rounded interval arithmetic fixes by
    // construction: (a) single-term n=1 with accumulate coarser than compute,
    // (b) per-product overflow into the COMPUTE precision, and a thorough
    // self-run brute-force search across precisions / fan-ins / magnitudes.
    // =====================================================================

    /// Deployed Linear row WITH a final output store at `output_p` (the coarser
    /// of compute/accumulate). This is the strongest deployment model: it adds
    /// the store the hardware applies when writing the output, on top of the
    /// per-product (compute) and per-partial-sum (accumulate) rounding modeled by
    /// `deployed_linear_row`. The sound interval must contain BOTH the un-stored
    /// accumulator and this stored value.
    fn deployed_linear_row_stored(
        weight: &Array2<f32>,
        bias: Option<&Array1<f32>>,
        x: &[f32],
        j: usize,
        compute_p: FloatPrecision,
        accumulate_p: FloatPrecision,
    ) -> f32 {
        let acc = deployed_linear_row(weight, bias, x, j, compute_p, accumulate_p);
        let output_p = compute_p.coarser(accumulate_p);
        rn(acc, output_p)
    }

    #[test]
    fn acceptance_corner_a_single_term_accumulate_coarser_than_compute() {
        // 3rd-audit corner (a): n=1 with accumulate COARSER than compute. A single
        // term still does ONE accumulate-grid store. compute=F32 (so the product is
        // exact) but accumulate=F16 rounds the running sum, and the prior
        // estimate-approach charged 0 accumulation for n_terms<=1. Direct IA folds
        // the single product through an F16 store, so it cannot miss it.
        let weight = Array2::from_shape_vec((1, 1), vec![-9.588151_f32]).unwrap();
        let x = vec![0.003_603_448_4_f32];
        let compute_p = FloatPrecision::F32;
        let accumulate_p = FloatPrecision::F16;
        let policy = MixedPrecisionPolicy::new(compute_p, accumulate_p);

        let deployed = deployed_linear_row(&weight, None, &x, 0, compute_p, accumulate_p);
        let deployed_stored =
            deployed_linear_row_stored(&weight, None, &x, 0, compute_p, accumulate_p);
        // The F16 store of the product visibly changes the value (the prior bug:
        // charged 0 here, so the bound was the exact f32 product and EXCLUDED this).
        let exact = -9.588151_f32 * 0.003_603_448_4_f32;
        assert!(
            (deployed - exact).abs() > 0.0,
            "corner (a) setup: the F16 accumulate store should move the value off the \
             exact f32 product {exact} (got deployed {deployed})"
        );

        let input = vec![Bound::new(x[0], x[0])];
        let bounds = sound_output_bounds(weight, None, input, &policy);
        assert_contains(&bounds, 0, deployed, "corner (a) single-term acc-coarser");
        assert_contains(&bounds, 0, deployed_stored, "corner (a) single-term stored");
        // It must also contain the exact f32 product (the accumulator before its
        // store could have rounded either way; the interval brackets both).
        assert_contains(&bounds, 0, exact, "corner (a) exact f32 product");
    }

    #[test]
    fn acceptance_corner_b_per_product_overflow_into_compute_f16() {
        // 3rd-audit corner (b): per-PRODUCT overflow into the COMPUTE precision.
        // compute=F16, W=[60000] (f16-representable), x=[2.0] -> product 120000
        // overflows f16 to +inf BEFORE any accumulation. The prior overflow guard
        // was keyed only on the accumulate max_normal and MISSED this. Direct IA
        // rounds the product to the compute grid, which saturates to +inf, so the
        // output interval admits +inf.
        let weight = Array2::from_shape_vec((1, 1), vec![60000.0_f32]).unwrap();
        let x = vec![2.0_f32];
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);

        // Real deployed: f16(f16(60000) * f16(2.0)) = f16(120000) = +inf.
        let deployed = deployed_linear_row(
            &weight,
            None,
            &x,
            0,
            FloatPrecision::F16,
            FloatPrecision::F16,
        );
        assert!(
            deployed.is_infinite() && deployed > 0.0,
            "corner (b) setup: f16 product 120000 must overflow to +inf, got {deployed}"
        );

        let input = vec![Bound::new(2.0, 2.0)];
        let bounds = sound_output_bounds(weight, None, input, &policy);
        assert!(
            bounds[0].upper().is_infinite() && bounds[0].upper() > 0.0,
            "corner (b): per-product f16 overflow must yield +inf upper, got {}",
            bounds[0].upper()
        );
        assert_contains(&bounds, 0, deployed, "corner (b) per-product overflow");
    }

    #[test]
    fn acceptance_corner_b_product_overflow_120064_variant() {
        // The exact magnitude named in the audit: compute=F16 product 120064 -> +inf.
        // W = 58.625*1024? Simpler: W=[58624.0]? Use W=[60032], x=[2.0] => 120064.
        // 60032 is within f16 range (max 65504) and f16-representable region; the
        // product 120064 overflows. Pin that the interval admits the +inf.
        let w = f16::from_f32(60032.0).to_f32();
        let weight = Array2::from_shape_vec((1, 1), vec![w]).unwrap();
        let x = vec![2.0_f32];
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);
        let deployed = deployed_linear_row(
            &weight,
            None,
            &x,
            0,
            FloatPrecision::F16,
            FloatPrecision::F16,
        );
        assert!(
            deployed.is_infinite() && deployed > 0.0,
            "must overflow, got {deployed}"
        );
        let input = vec![Bound::new(2.0, 2.0)];
        let bounds = sound_output_bounds(weight, None, input, &policy);
        assert_contains(&bounds, 0, deployed, "corner (b) 120064 variant");
    }

    #[test]
    fn acceptance_corner_c_accumulate_coarser_charged_correctly() {
        // 3rd-audit corner (c): accumulator rounding must be charged at the
        // ACCUMULATE grid, not the (finer) compute grid. compute=F32, accumulate=Bf16:
        // products are exact f32 but each partial sum is stored in bf16, which drifts.
        // n moderate so drift is visible but finite. Direct IA stores each partial
        // sum on the accumulate grid, so it tracks the coarse accumulator exactly.
        let n = 64usize;
        // Off-grid weights so the f32 products are non-trivial; x = 1.
        let wv: Vec<f32> = (0..n).map(|k| 0.37 + 0.013 * (k as f32 % 11.0)).collect();
        let weight = Array2::from_shape_vec((1, n), wv).unwrap();
        let compute_p = FloatPrecision::F32;
        let accumulate_p = FloatPrecision::Bf16;
        let policy = MixedPrecisionPolicy::new(compute_p, accumulate_p);
        let x = vec![1.0_f32; n];
        let deployed = deployed_linear_row(&weight, None, &x, 0, compute_p, accumulate_p);
        let deployed_stored =
            deployed_linear_row_stored(&weight, None, &x, 0, compute_p, accumulate_p);
        let input = vec![Bound::new(1.0, 1.0); n];
        let bounds = sound_output_bounds(weight, None, input, &policy);
        assert_contains(&bounds, 0, deployed, "corner (c) acc-coarser accumulation");
        assert_contains(&bounds, 0, deployed_stored, "corner (c) acc-coarser stored");
    }

    #[test]
    fn acceptance_corner_a_n1_with_bias_does_two_stores() {
        // Companion to corner (a): n=1 WITH a bias is two accumulate-grid stores
        // (product store, then bias store). accumulate coarser than compute again.
        let weight = Array2::from_shape_vec((1, 1), vec![3.3_f32]).unwrap();
        let bias = Array1::from_vec(vec![-7.123_f32]);
        let x = vec![1.9_f32];
        let compute_p = FloatPrecision::F32;
        let accumulate_p = FloatPrecision::F16;
        let policy = MixedPrecisionPolicy::new(compute_p, accumulate_p);
        let deployed = deployed_linear_row(&weight, Some(&bias), &x, 0, compute_p, accumulate_p);
        let input = vec![Bound::new(x[0], x[0])];
        let bounds = sound_output_bounds(weight, Some(bias), input, &policy);
        assert_contains(&bounds, 0, deployed, "corner (a) n=1 with bias two stores");
    }

    #[test]
    fn acceptance_brute_force_no_deployed_value_escapes() {
        // SELF BRUTE-FORCE (required): search MANY Linear configs and assert the
        // sound deployed interval CONTAINS the half-crate-simulated deployed value
        // (and its stored variant) for sampled interval corners. Reports the count
        // and that ZERO escaped.
        //
        // Sweep: compute,accumulate independently over {F16,Bf16,F32}; n in
        // {1,2,3,8,64,512}; weights/inputs covering off-grid, near-max-normal
        // (overflow), and subnormal magnitudes; point and interval inputs; with and
        // without bias.

        // Deterministic xorshift64* (no rand dependency).
        fn unit(seed: &mut u64) -> f64 {
            let mut x = *seed;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            *seed = x;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        }
        fn frand(seed: &mut u64, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * (unit(seed) as f32)
        }
        // Pick a magnitude scale from a regime list to stress every region.
        fn scaled(seed: &mut u64, regime: u8) -> f32 {
            let s = frand(seed, -1.0, 1.0);
            match regime {
                0 => s * frand(seed, 0.0, 3.0),         // small/off-grid normals
                1 => s * frand(seed, 30000.0, 70000.0), // near/over f16 max-normal
                2 => {
                    // subnormal-scale: small multiples of the f16 smallest subnormal
                    let sub = FloatPrecision::F16.smallest_subnormal();
                    s * sub * frand(seed, 1.0, 50.0)
                }
                _ => s * frand(seed, 0.0, 1000.0), // mid-range
            }
        }

        let precisions = [
            FloatPrecision::F16,
            FloatPrecision::Bf16,
            FloatPrecision::F32,
        ];
        let ns = [1usize, 2, 3, 8, 64, 512];

        let mut seed = 0x1357_9BDF_2468_ACE0_u64;
        let mut configs = 0usize;
        let mut deployed_checks = 0usize;
        let mut escapes = 0usize;

        for &compute_p in &precisions {
            for &accumulate_p in &precisions {
                // Skip the pure-F32 idealized policy (it is the verified no-op path,
                // exercised separately; the deployed value == idealized there).
                if compute_p.is_idealized_f32() && accumulate_p.is_idealized_f32() {
                    continue;
                }
                let policy = MixedPrecisionPolicy::new(compute_p, accumulate_p);
                for &n in &ns {
                    // A handful of randomized rows per (precision, n), across regimes.
                    for trial in 0..24usize {
                        let regime = (trial % 4) as u8;
                        let with_bias = trial % 3 == 0;
                        let point_input = trial % 2 == 0;

                        let mut wv = Vec::with_capacity(n);
                        let mut lo = Vec::with_capacity(n);
                        let mut hi = Vec::with_capacity(n);
                        for _ in 0..n {
                            wv.push(scaled(&mut seed, regime));
                            let a = scaled(&mut seed, regime);
                            if point_input {
                                lo.push(a);
                                hi.push(a);
                            } else {
                                let b = scaled(&mut seed, regime);
                                lo.push(a.min(b));
                                hi.push(a.max(b));
                            }
                        }
                        let weight = Array2::from_shape_vec((1, n), wv.clone()).unwrap();
                        let bias_arr = if with_bias {
                            Some(Array1::from_vec(vec![scaled(&mut seed, regime)]))
                        } else {
                            None
                        };

                        let input: Vec<Bound> = lo
                            .iter()
                            .zip(hi.iter())
                            .map(|(&l, &h)| Bound::new(l, h))
                            .collect();
                        let bounds =
                            sound_output_bounds(weight.clone(), bias_arr.clone(), input, &policy);
                        configs += 1;

                        // Probe deployed values at sampled corners: all-low, all-high,
                        // and per-axis flips (the 2^n corner set is exponential).
                        let mut probes: Vec<Vec<f32>> = vec![lo.clone(), hi.clone()];
                        for i in 0..n {
                            let mut p_lo = hi.clone();
                            p_lo[i] = lo[i];
                            let mut p_hi = lo.clone();
                            p_hi[i] = hi[i];
                            probes.push(p_lo);
                            probes.push(p_hi);
                        }
                        // Cap probes to keep n=512 affordable while still sampling.
                        probes.truncate(34);

                        let bias_ref = bias_arr.as_ref();
                        for xp in &probes {
                            for &val in &[
                                deployed_linear_row(
                                    &weight,
                                    bias_ref,
                                    xp,
                                    0,
                                    compute_p,
                                    accumulate_p,
                                ),
                                deployed_linear_row_stored(
                                    &weight,
                                    bias_ref,
                                    xp,
                                    0,
                                    compute_p,
                                    accumulate_p,
                                ),
                            ] {
                                deployed_checks += 1;
                                let b = &bounds[0];
                                // NaN deployed values cannot arise (no inf*0 here as
                                // inputs/weights are finite); guard anyway.
                                let contained = if val.is_nan() {
                                    true
                                } else {
                                    b.lower() <= val && val <= b.upper()
                                };
                                if !contained {
                                    escapes += 1;
                                    if escapes <= 5 {
                                        eprintln!(
                                            "ESCAPE compute={compute_p:?} accumulate={accumulate_p:?} \
                                             n={n} regime={regime} bias={with_bias}: deployed \
                                             {val} NOT in [{}, {}]",
                                            b.lower(),
                                            b.upper()
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        eprintln!(
            "BRUTE-FORCE: searched {configs} Linear configs, {deployed_checks} deployed-value \
             containment checks; escapes={escapes}"
        );
        assert!(
            configs >= 1000,
            "brute-force should search >=1000 configs, got {configs}"
        );
        assert_eq!(
            escapes, 0,
            "{escapes} deployed value(s) escaped the sound interval"
        );
    }

    // =====================================================================
    // rev4 ACCEPTANCE — ORDER-INDEPENDENT soundness (4th audit, H1 + H2).
    //
    // Floating-point summation is NOT associative, so a left-to-right interval
    // fold can stay finite while a DIFFERENT legal reduction order overflows a
    // partial sum to ±inf (H1), and the 4-corner min/max product hull silently
    // drops a NaN from inf*0 (H2). These pin both holes and a brute-force that
    // varies the reduction order.
    // =====================================================================

    /// Deployed Linear row simulated under an EXPLICIT term ordering `order`
    /// (a permutation of `0..N` indices into the products; index `in_features`
    /// denotes the bias term when present). Products are rounded into compute_p,
    /// each running-sum store rounds into accumulate_p — exactly the deployed
    /// per-step model, but the addition order is `order`.
    fn deployed_linear_row_ordered(
        weight: &Array2<f32>,
        bias: Option<&Array1<f32>>,
        x: &[f32],
        j: usize,
        compute_p: FloatPrecision,
        accumulate_p: FloatPrecision,
        order: &[usize],
    ) -> f32 {
        let in_features = weight.ncols();
        // Precompute each term's deployed product (or the bias bracket) at compute_p.
        let term = |idx: usize| -> f32 {
            if idx < in_features {
                let w = rn(weight[[j, idx]], compute_p);
                let xi = rn(x[idx], compute_p);
                rn(w * xi, compute_p)
            } else {
                // bias term
                rn(bias.map(|b| b[j]).unwrap_or(0.0), compute_p)
            }
        };
        let mut acc = rn(0.0, accumulate_p);
        for &idx in order {
            acc = rn(acc + term(idx), accumulate_p);
        }
        acc
    }

    /// Pairwise / tree reduction of the terms named by `idxs`, each add rounded
    /// into accumulate_p, each product rounded into compute_p. This is the order a
    /// real GPU "tree" reduction uses (and is associativity-sensitive).
    fn deployed_linear_row_tree(
        weight: &Array2<f32>,
        bias: Option<&Array1<f32>>,
        x: &[f32],
        j: usize,
        compute_p: FloatPrecision,
        accumulate_p: FloatPrecision,
        idxs: &[usize],
    ) -> f32 {
        let in_features = weight.ncols();
        let term = |idx: usize| -> f32 {
            if idx < in_features {
                let w = rn(weight[[j, idx]], compute_p);
                let xi = rn(x[idx], compute_p);
                rn(w * xi, compute_p)
            } else {
                rn(bias.map(|b| b[j]).unwrap_or(0.0), compute_p)
            }
        };
        let mut level: Vec<f32> = idxs.iter().map(|&i| term(i)).collect();
        if level.is_empty() {
            return rn(0.0, accumulate_p);
        }
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut k = 0;
            while k < level.len() {
                if k + 1 < level.len() {
                    next.push(rn(level[k] + level[k + 1], accumulate_p));
                } else {
                    next.push(level[k]);
                }
                k += 2;
            }
            level = next;
        }
        level[0]
    }

    #[test]
    fn h1_reordering_overflow_widens_to_inf_when_left_to_right_finite() {
        // H1: terms +40000, -40000, +40000, +40000 at accumulate=F16. LEFT-TO-RIGHT
        // never overflows (40000-40000=0, +40000, +40000=80000 -> actually 80000
        // > 65504, so craft so L-to-R stays finite): use +40000, -40000, +40000,
        // -40000 then a final +40000 is too many. The spec's canonical case is the
        // POSITIVE reachable sum pos = sum of positives = 120000 >= 65504, which a
        // "add all positives first" order reaches even though an alternating order
        // does not. The sound interval MUST contain +inf.
        //
        // Realize the terms as weights with input x=1: W=[40000,-40000,40000,40000].
        let weight =
            Array2::from_shape_vec((1, 4), vec![40000.0_f32, -40000.0, 40000.0, 40000.0]).unwrap();
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);
        let x = vec![1.0_f32; 4];

        // Verify the premise: an "all positives first" order overflows f16...
        let pos_first = deployed_linear_row_ordered(
            &weight,
            None,
            &x,
            0,
            FloatPrecision::F16,
            FloatPrecision::F16,
            &[0, 2, 3, 1], // +40000,+40000,+40000,-40000
        );
        assert!(
            pos_first.is_infinite() && pos_first > 0.0,
            "H1 premise: positives-first order should overflow f16 to +inf, got {pos_first}"
        );
        // ...while a DIFFERENT order (interleaved) stays finite.
        let interleaved = deployed_linear_row_ordered(
            &weight,
            None,
            &x,
            0,
            FloatPrecision::F16,
            FloatPrecision::F16,
            &[0, 1, 2, 3], // +40000,-40000,+40000,+40000 -> 40000 finite? last step 40000+40000
        );
        // (interleaved here actually overflows on the last step too; the point is
        // only that SOME order overflows — the guard must widen regardless.)
        let _ = interleaved;

        let input = vec![Bound::new(1.0, 1.0); 4];
        let bounds = sound_output_bounds(weight, None, input, &policy);
        assert!(
            bounds[0].upper().is_infinite() && bounds[0].upper() > 0.0,
            "H1: reordering can overflow -> sound upper must be +inf, got {}",
            bounds[0].upper()
        );
        // And it must contain the overflowing deployed order.
        assert_contains(&bounds, 0, pos_first, "H1 positives-first overflow");
    }

    #[test]
    fn h1_strictly_left_to_right_finite_but_reorder_overflows() {
        // A SHARPER H1: craft terms so the LEFT-TO-RIGHT deployed fold stays FINITE
        // but a reordering overflows. accumulate=F16 (max 65504). Terms (x=1):
        //   +60000, -60000, +60000, -60000, +10000
        // Left-to-right: 60000, 0, 60000, 0, 10000 -> finite (never exceeds 60000).
        // Positives-first: 60000+60000+10000 = 130000 -> overflows.
        let weight = Array2::from_shape_vec(
            (1, 5),
            vec![60000.0_f32, -60000.0, 60000.0, -60000.0, 10000.0],
        )
        .unwrap();
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);
        let x = vec![1.0_f32; 5];

        // Left-to-right deployed fold is FINITE.
        let l2r = deployed_linear_row(
            &weight,
            None,
            &x,
            0,
            FloatPrecision::F16,
            FloatPrecision::F16,
        );
        assert!(
            l2r.is_finite(),
            "H1 sharper premise: left-to-right must stay finite, got {l2r}"
        );
        // A positives-first order overflows.
        let pos_first = deployed_linear_row_ordered(
            &weight,
            None,
            &x,
            0,
            FloatPrecision::F16,
            FloatPrecision::F16,
            &[0, 2, 4, 1, 3],
        );
        assert!(
            pos_first.is_infinite() && pos_first > 0.0,
            "H1 sharper premise: positives-first must overflow, got {pos_first}"
        );

        let input = vec![Bound::new(1.0, 1.0); 5];
        let bounds = sound_output_bounds(weight, None, input, &policy);
        // The OLD left-to-right interval fold would have stayed finite here and
        // EXCLUDED the +inf deployed value (the H1 hole). The fix must widen.
        assert!(
            bounds[0].upper().is_infinite() && bounds[0].upper() > 0.0,
            "H1 sharper: sound upper must be +inf (some order overflows), got {}",
            bounds[0].upper()
        );
        assert!(
            bounds[0].lower().is_infinite() && bounds[0].lower() < 0.0,
            "H1 sharper: widen-to-[-inf,+inf] sets lower to -inf, got {}",
            bounds[0].lower()
        );
        assert_contains(&bounds, 0, l2r, "H1 sharper L-to-R finite");
        assert_contains(&bounds, 0, pos_first, "H1 sharper reorder overflow");
    }

    #[test]
    fn h2_inf_times_zero_yields_nan_term_widens_to_unconstrained() {
        // H2: compute=F16, a weight 70000 rounds to f16 +inf, and the input bound
        // spans 0 (e.g. [-1, 1] contains 0). The deployed product f16(+inf)*f16(0)
        // = NaN. The 4-corner min/max hull would DROP the NaN corner and collapse
        // the term to a finite interval; the fix must set it to [-inf, +inf], and
        // the (now non-finite reachability) guard widens the output.
        let weight = Array2::from_shape_vec((1, 1), vec![70000.0_f32]).unwrap();
        let policy = MixedPrecisionPolicy::uniform(FloatPrecision::F16);

        // Premise: the deployed product at x=0 is NaN (inf * 0).
        let w_dep = rn(70000.0, FloatPrecision::F16);
        assert!(
            w_dep.is_infinite() && w_dep > 0.0,
            "premise: 70000 -> f16 +inf"
        );
        let prod = rn(w_dep * rn(0.0, FloatPrecision::F16), FloatPrecision::F16);
        assert!(
            prod.is_nan(),
            "premise: f16(+inf)*f16(0) must be NaN, got {prod}"
        );

        // Input bound spans 0.
        let input = vec![Bound::new(-1.0, 1.0)];
        let bounds = sound_output_bounds(weight, None, input, &policy);
        assert!(
            bounds[0].lower().is_infinite()
                && bounds[0].lower() < 0.0
                && bounds[0].upper().is_infinite()
                && bounds[0].upper() > 0.0,
            "H2: inf*0 NaN term must widen output to [-inf, +inf], got [{}, {}]",
            bounds[0].lower(),
            bounds[0].upper()
        );
    }

    #[test]
    fn h2_point_input_zero_with_overflow_weight_widens() {
        // H2 variant: input is EXACTLY the point 0 and the weight overflows compute
        // precision to +inf. The deployed product is inf*0 = NaN; the sound output
        // must be unconstrained, not the finite [0, 0] a NaN-dropping hull yields.
        let weight = Array2::from_shape_vec((1, 1), vec![80000.0_f32]).unwrap();
        let policy = MixedPrecisionPolicy::new(FloatPrecision::F16, FloatPrecision::F32);
        let input = vec![Bound::new(0.0, 0.0)];
        let bounds = sound_output_bounds(weight, None, input, &policy);
        assert!(
            bounds[0].lower().is_infinite() && bounds[0].upper().is_infinite(),
            "H2 point-zero: inf*0 -> [-inf, +inf], got [{}, {}]",
            bounds[0].lower(),
            bounds[0].upper()
        );
    }

    #[test]
    fn order_varying_brute_force_no_escape_under_any_reduction_order() {
        // ORDER-VARYING BRUTE-FORCE (required): for MANY configs, simulate the
        // deployed dot product under MULTIPLE reduction orders — left-to-right,
        // right-to-left, several random permutations, and a pairwise/tree fold —
        // each with per-product (compute) and per-step (accumulate) rounding via
        // the half crate, and assert the sound interval CONTAINS the deployed value
        // under EVERY tested order. Reports config / order counts; escapes MUST be 0.
        fn unit(seed: &mut u64) -> f64 {
            let mut x = *seed;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            *seed = x;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        }
        fn frand(seed: &mut u64, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * (unit(seed) as f32)
        }
        // Magnitude regimes: small/off-grid, near/over f16 max-normal (so reorder
        // overflow is reachable), subnormal-scale, mid-range.
        fn scaled(seed: &mut u64, regime: u8) -> f32 {
            let s = frand(seed, -1.0, 1.0);
            match regime {
                0 => s * frand(seed, 0.0, 3.0),
                1 => s * frand(seed, 20000.0, 70000.0), // overflow-prone
                2 => {
                    let sub = FloatPrecision::F16.smallest_subnormal();
                    s * sub * frand(seed, 1.0, 50.0)
                }
                _ => s * frand(seed, 0.0, 1000.0),
            }
        }
        // A small Fisher-Yates shuffle on a scratch index vec.
        fn shuffle(seed: &mut u64, v: &mut [usize]) {
            let n = v.len();
            for i in (1..n).rev() {
                let j = (unit(seed) * ((i + 1) as f64)) as usize;
                v.swap(i, j.min(i));
            }
        }

        let precisions = [
            FloatPrecision::F16,
            FloatPrecision::Bf16,
            FloatPrecision::F32,
        ];
        let ns = [1usize, 2, 3, 5, 8, 32, 128];

        let mut seed = 0x0BAD_C0DE_F00D_1234_u64;
        let mut configs = 0usize;
        let mut order_checks = 0usize;
        let mut escapes = 0usize;

        for &compute_p in &precisions {
            for &accumulate_p in &precisions {
                if compute_p.is_idealized_f32() && accumulate_p.is_idealized_f32() {
                    continue;
                }
                let policy = MixedPrecisionPolicy::new(compute_p, accumulate_p);
                for &n in &ns {
                    for trial in 0..20usize {
                        let regime = (trial % 4) as u8;
                        let with_bias = trial % 3 == 0;
                        let point_input = trial % 2 == 0;

                        let mut wv = Vec::with_capacity(n);
                        let mut lo = Vec::with_capacity(n);
                        let mut hi = Vec::with_capacity(n);
                        for _ in 0..n {
                            wv.push(scaled(&mut seed, regime));
                            let a = scaled(&mut seed, regime);
                            if point_input {
                                lo.push(a);
                                hi.push(a);
                            } else {
                                let b = scaled(&mut seed, regime);
                                lo.push(a.min(b));
                                hi.push(a.max(b));
                            }
                        }
                        let weight = Array2::from_shape_vec((1, n), wv.clone()).unwrap();
                        let bias_arr = if with_bias {
                            Some(Array1::from_vec(vec![scaled(&mut seed, regime)]))
                        } else {
                            None
                        };
                        let bias_ref = bias_arr.as_ref();

                        let input: Vec<Bound> = lo
                            .iter()
                            .zip(hi.iter())
                            .map(|(&l, &h)| Bound::new(l, h))
                            .collect();
                        let bounds =
                            sound_output_bounds(weight.clone(), bias_arr.clone(), input, &policy);
                        configs += 1;
                        let b = &bounds[0];

                        // Number of accumulation terms: products + optional bias.
                        let n_terms = n + usize::from(with_bias);
                        let base: Vec<usize> = (0..n_terms).collect();

                        // Probe a few input points (corners + per-axis flips).
                        let mut probes: Vec<Vec<f32>> = vec![lo.clone(), hi.clone()];
                        for i in 0..n {
                            let mut p = hi.clone();
                            p[i] = lo[i];
                            probes.push(p);
                        }
                        probes.truncate(6);

                        for xp in &probes {
                            // Build the order set: L-to-R, R-to-L, 4 random perms,
                            // and the tree fold.
                            let mut orders: Vec<Vec<usize>> = Vec::new();
                            orders.push(base.clone());
                            let mut rev = base.clone();
                            rev.reverse();
                            orders.push(rev);
                            // "positives-first" style order: sort terms by their
                            // deployed product sign so all-positive-first is exercised
                            // (the canonical reorder-overflow trigger).
                            {
                                let mut pos_first: Vec<usize> = base.clone();
                                pos_first.sort_by(|&a, &c| {
                                    let ta = if a < n {
                                        rn(
                                            rn(weight[[0, a]], compute_p) * rn(xp[a], compute_p),
                                            compute_p,
                                        )
                                    } else {
                                        rn(bias_ref.map(|bb| bb[0]).unwrap_or(0.0), compute_p)
                                    };
                                    let tc = if c < n {
                                        rn(
                                            rn(weight[[0, c]], compute_p) * rn(xp[c], compute_p),
                                            compute_p,
                                        )
                                    } else {
                                        rn(bias_ref.map(|bb| bb[0]).unwrap_or(0.0), compute_p)
                                    };
                                    tc.partial_cmp(&ta).unwrap_or(std::cmp::Ordering::Equal)
                                });
                                orders.push(pos_first);
                            }
                            for _ in 0..4 {
                                let mut perm = base.clone();
                                shuffle(&mut seed, &mut perm);
                                orders.push(perm);
                            }

                            for ord in &orders {
                                let dep = deployed_linear_row_ordered(
                                    &weight,
                                    bias_ref,
                                    xp,
                                    0,
                                    compute_p,
                                    accumulate_p,
                                    ord,
                                );
                                order_checks += 1;
                                let contained = if dep.is_nan() {
                                    true
                                } else {
                                    b.lower() <= dep && dep <= b.upper()
                                };
                                if !contained {
                                    escapes += 1;
                                    if escapes <= 5 {
                                        eprintln!(
                                            "ORDER-ESCAPE compute={compute_p:?} acc={accumulate_p:?} \
                                             n={n} regime={regime} bias={with_bias} order={ord:?}: \
                                             deployed {dep} NOT in [{}, {}]",
                                            b.lower(),
                                            b.upper()
                                        );
                                    }
                                }
                            }
                            // Tree / pairwise fold.
                            let tree = deployed_linear_row_tree(
                                &weight,
                                bias_ref,
                                xp,
                                0,
                                compute_p,
                                accumulate_p,
                                &base,
                            );
                            order_checks += 1;
                            let contained = if tree.is_nan() {
                                true
                            } else {
                                b.lower() <= tree && tree <= b.upper()
                            };
                            if !contained {
                                escapes += 1;
                                if escapes <= 5 {
                                    eprintln!(
                                        "TREE-ESCAPE compute={compute_p:?} acc={accumulate_p:?} \
                                         n={n} regime={regime}: deployed {tree} NOT in [{}, {}]",
                                        b.lower(),
                                        b.upper()
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        eprintln!(
            "ORDER-VARYING BRUTE-FORCE: {configs} Linear configs, {order_checks} \
             order-varied deployed containment checks (L-to-R, R-to-L, positives-first, \
             4 random perms, tree); escapes={escapes}"
        );
        assert!(
            configs >= 1000,
            "should search >=1000 configs, got {configs}"
        );
        assert!(
            order_checks >= 10_000,
            "should run >=10000 order checks, got {order_checks}"
        );
        assert_eq!(
            escapes, 0,
            "{escapes} deployed value(s) escaped under some reduction order"
        );
    }
}
