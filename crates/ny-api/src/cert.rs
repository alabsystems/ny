// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proof-carrying verification certificates.
//!
//! Exposes ny's exact-rational, Clean-checkable certificate surface: a CROWN
//! verdict is turned into an entailment/Farkas certificate over exact rationals
//! that an external kernel-backed checker can replay, making a verified verdict
//! a machine-checkable proof rather than a floating-point claim.
//!
//! Covers the certificate types, the self-contained checkers, the shallow
//! ([`Relu1Problem`]) and deep ([`DeepReluProblem`]) CROWN problem/result types,
//! the JSON schema emitters used to serialize certificates for Clean, and the
//! Alethe proof emitters ([`entailment_to_alethe`] / [`farkas_to_alethe`] /
//! [`branch_tree_to_alethe`]) whose SMT-LIB/Alethe pairs the standard external
//! `carcara` checker replays. [`certify_graph_with_alethe`] threads that
//! emission through the [`certify_graph`] certificate build itself.
//!
//! The [`certify_graph`] entry point wires this surface into the verifier path:
//! it runs the normal graph verifier and, when the network and property are
//! eligible (sequential fully-connected ReLU net, conjunctive output property),
//! re-derives an exact-rational certificate, self-checks it, and attaches it to
//! the verdict's previously-dead [`VerificationProof`] channel. It NEVER changes
//! the runtime verdict and NEVER emits a certificate for an unverified or
//! ineligible network (see [`certify_graph`] for the soundness invariants).
//!
//! ```rust
//! use ny_api::cert::{Rat, Relu1Problem, check_entailment};
//!
//! let r = |n: i128, d: i128| Rat::new(n, d).unwrap();
//! let problem = Relu1Problem {
//!     w1: vec![vec![r(1, 1), r(1, 1)], vec![r(1, 1), r(-1, 1)]],
//!     b1: vec![Rat::ZERO, Rat::ZERO],
//!     w2: vec![r(1, 1), r(-1, 1)],
//!     b2: r(5, 2),
//!     input_lower: vec![r(-1, 1), r(-1, 1)],
//!     input_upper: vec![r(1, 1), r(1, 1)],
//!     alpha: Some(vec![r(1, 2), r(1, 2)]),
//! };
//! let certified = problem.certify(Rat::ZERO).unwrap();
//! check_entailment(&certified.entailment).unwrap();
//! ```

/// Deep `k`-hidden-layer CROWN problem, certified result, and supporting types.
pub use ny_cert::crown_deep::{CertifiedDeep, DeepCrownError, DeepReluProblem, PreactBounds};
/// Certificate types and JSON schema emitters for Clean serialization.
pub use ny_cert::{
    chain_to_json, entailment_to_json, farkas_to_json, ConstraintKind, EntailmentCertificate,
    FarkasCertificate, LinearConstraint,
};
/// Self-contained certificate checkers (entailment / Farkas / chain).
pub use ny_cert::{check_chain, check_entailment, check_farkas, CheckError};
/// Deployed-f32 rounding margin: the exact-rational bound `delta` with
/// `|fl32(net(x)) - net(x)| <= delta` over the whole input box, which folds a
/// real-semantics certificate onto the network as it actually executes in f32.
/// [`certify_graph_with_policy`] folds it into the strict-dominance gate when a
/// [`DeployedPrecision::DeployedF32`] policy is active.
pub use ny_cert::{deployed_fp_margin, deployed_fp_output_margin, FpMargin, FpMarginError};
/// Shallow (one ReLU hidden layer) CROWN problem and certified result.
pub use ny_cert::{CertifiedRelu1, CrownError, Relu1Problem};
/// Exact-rational scalar arithmetic underlying every certificate.
pub use ny_cert::{Rat, RatError};

/// Alethe proof emission: turn a self-checked entailment / Farkas /
/// branch-tree certificate into an SMT-LIB problem + Alethe proof pair
/// ([`AletheEmission`]) checkable by the standard external `carcara` checker
/// (`carcara check proof.alethe problem.smt2`). Fail-closed: every emitter
/// replays the certificate through its in-tree checker first and refuses
/// ([`EmitError`]) any certificate — or unsupported shape (equalities,
/// non-SMT-LIB symbols) — rather than emit wrong proof text.
pub use ny_cert::{
    branch_tree_to_alethe, entailment_to_alethe, farkas_to_alethe, AletheEmission, EmitError,
};
/// Branch-tree (case-split) certificate surface: the [`branch_tree_to_alethe`]
/// input type, its components, and the in-tree checker its emission replays.
pub use ny_cert::{
    check_branch_tree, AxisPartition, BranchError, BranchLeaf, BranchTreeCertificate, ThreshDir,
};

use crate::constraints::{augment_for_constraint, verify_with_constraints};
use ndarray::{Array1, Array2};
use ny_cert::{check_entailment as cert_check_entailment, check_farkas as cert_check_farkas};
use ny_core::{
    Bound, OutputConstraint, ProofFormat, Result, SoundnessProvenance, VerificationProof,
    VerificationResult, VerificationSpec,
};
use ny_propagate::layers::Layer;
use ny_propagate::{GraphNetwork, PropagationConfig, PropagationMethod, Verifier, NETWORK_INPUT};

/// Outcome of an attempt to attach a proof-carrying certificate to a verdict.
///
/// `result` is ALWAYS the verifier's verdict — the certificate is a pure
/// additive artifact and never alters it. `certificate_json` is `Some` only when
/// the network/property were eligible AND an exact-rational certificate was
/// built, self-checked, and serialized; in that case `result` is the `Verified`
/// verdict with its [`VerificationProof`] channel populated. `eligible` reflects
/// only the static architecture/property gate (it can be `true` while
/// `certificate_json` is `None` when the verdict was not `Verified`, or exact
/// CROWN could not close the property). `note` explains the outcome.
#[derive(Debug, Clone)]
pub struct CertifiedResult {
    /// The verifier's verdict (unchanged by certification).
    pub result: VerificationResult,
    /// Clean-canonical certificate JSON, present only when emitted.
    pub certificate_json: Option<String>,
    /// Whether the network + property passed the static eligibility gate.
    pub eligible: bool,
    /// Human-readable explanation of the certification outcome.
    pub note: String,
}

/// One discharged conjunct's Alethe emission: the SMT-LIB problem / Alethe
/// proof pair for its entailment certificate and for its Farkas certificate,
/// each independently checkable with `carcara check proof.alethe problem.smt2`.
///
/// The pair proves exactly what the conjunct's self-checked certificates
/// state. For a STRICT (`ArgmaxMargin`) conjunct the strict relation
/// `margin > 0` additionally rests on the conjunct's exact lower bound being
/// strictly positive — a gate [`certify_graph_with_alethe`] enforces
/// fail-closed BEFORE any emission, so no Alethe artifact ever accompanies an
/// undischarged strict conjunct.
#[derive(Debug, Clone)]
pub struct ConjunctAlethe {
    /// The conjunct's label — identical to the `discharged_conjunct` field of
    /// the corresponding entry in [`CertifiedResult::certificate_json`].
    pub label: String,
    /// Alethe emission of the conjunct's entailment certificate.
    pub entailment: AletheEmission,
    /// Alethe emission of the conjunct's Farkas certificate.
    pub farkas: AletheEmission,
}

/// Outcome of [`certify_graph_with_alethe`]: the plain certification outcome
/// plus, when every discharged conjunct also emitted, its Alethe proofs.
///
/// The Alethe set is WHOLE-OR-NOTHING, mirroring the certificate discipline:
/// `alethe == Some(v)` only when the exact-rational certificate was emitted
/// AND every one of its conjuncts produced a checkable [`ConjunctAlethe`] —
/// never a partial set that could be mistaken for full coverage. An emission
/// refusal (an [`EmitError`], e.g. an unsupported shape) withholds ONLY the
/// Alethe artifact; the JSON certificate and the verdict stand unchanged.
#[derive(Debug, Clone)]
pub struct AletheCertifiedResult {
    /// The plain certification outcome — identical semantics to
    /// [`certify_graph`] (verdict never altered, certificate never partial).
    pub certified: CertifiedResult,
    /// Per-conjunct Alethe emissions, in certificate-conjunct order. `Some`
    /// only when the certificate was emitted and EVERY conjunct emitted.
    pub alethe: Option<Vec<ConjunctAlethe>>,
    /// Human-readable explanation of the Alethe outcome.
    pub alethe_note: String,
}

/// Internal whole-or-nothing accumulator for per-conjunct Alethe emissions,
/// threaded through the certificate build only by
/// [`certify_graph_with_alethe`]. The first emission refusal clears every
/// collected conjunct and records the reason — the certificate build itself
/// continues unaffected (the Alethe artifact is additive on top of the
/// already-additive certificate).
#[derive(Debug, Default)]
struct AletheAcc {
    conjuncts: Vec<ConjunctAlethe>,
    refusal: Option<String>,
}

/// Which arithmetic a certificate's STRICT (argmax dominance) conjuncts are
/// required to hold in.
///
/// Every certificate this surface emits proves a property of the network's
/// IDEAL real-valued semantics: the f32 weights enter as their exact rational
/// values and all arithmetic is real. The network that ships, however, executes
/// in f32 round-to-nearest, and a strict dominance `y_class > y_j` proven with
/// an exact lower bound `L` as small as one ULP does NOT survive that rounding.
/// This policy chooses whether the strict gate accounts for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[non_exhaustive]
pub enum DeployedPrecision {
    /// **Default.** Strict conjuncts require only `L > 0` — the certificate
    /// attests the property of the ideal real-valued network, exactly as
    /// before this policy existed. Certificate bytes under this variant are
    /// bit-identical to the pre-policy emitter's.
    #[default]
    RealSemantics,
    /// Fold the deployed-f32 rounding margin into the strict gate: a strict
    /// conjunct is discharged only when `L > delta`, where `delta` bounds the
    /// f32 rounding error of the deployed outputs the conjunct compares
    /// ([`deployed_fp_output_margin`], computed in exact rationals). The
    /// conjunct then also attests the dominance for the DEPLOYED network, and
    /// records `delta` as its folded threshold.
    ///
    /// Fail-closed: if `delta` cannot be computed for any output the strict
    /// conjunct reads (a parameter that is not the exact value of a finite
    /// f32, an overflow-capable magnitude envelope, an affine row outside the
    /// error model's domain), the WHOLE certificate is refused rather than
    /// emitted with a real-semantics-only strict gate.
    DeployedF32,
}

/// Certification policy: the opt-in knobs that change what a certificate
/// attests.
///
/// [`CertifyPolicy::default`] is the historical behaviour in every field, so
/// [`certify_graph`] and [`certify_graph_with_policy`] with the default policy
/// emit byte-identical certificates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[non_exhaustive]
pub struct CertifyPolicy {
    /// Which arithmetic strict conjuncts must hold in. Default:
    /// [`DeployedPrecision::RealSemantics`].
    pub deployed_precision: DeployedPrecision,
}

impl CertifyPolicy {
    /// The default policy: strict conjuncts attest the ideal real-valued
    /// semantics (`L > 0`). Certificate bytes are bit-stable with
    /// [`certify_graph`].
    #[must_use]
    pub const fn real_semantics() -> Self {
        Self {
            deployed_precision: DeployedPrecision::RealSemantics,
        }
    }

    /// The deployed-f32 policy: strict conjuncts must clear the deployed
    /// rounding margin (`L > delta`), fail-closed when `delta` is not
    /// computable. See [`DeployedPrecision::DeployedF32`].
    #[must_use]
    pub const fn deployed_f32() -> Self {
        Self {
            deployed_precision: DeployedPrecision::DeployedF32,
        }
    }

    /// Whether the deployed-f32 fold is active.
    #[must_use]
    pub const fn folds_deployed_fp_margin(&self) -> bool {
        matches!(self.deployed_precision, DeployedPrecision::DeployedF32)
    }
}

/// Per-output-index cache of deployed-f32 error bounds. Every entry is computed
/// from the SAME hidden stack and input box, so one exact-rational interval
/// propagation per distinct read-out row is all an argmax property needs.
type DeployedErrorCache = std::collections::HashMap<usize, Rat>;

/// Upper bound on the total hidden-neuron count we attempt to certify. Exact
/// rational CROWN over a large net can create very large intermediate values
/// and consume excessive time/memory; above this we skip (non-fatal, no cert).
/// Mirrors the CLI adapter's gate.
const MAX_CERT_NEURONS: usize = 8192;

/// Convert an `f32` to its EXACT dyadic rational value (no rounding).
///
/// IEEE-754 binary32 values are exactly dyadic (`m · 2^e`), so this is lossless.
/// Returns `None` for non-finite inputs or a poisoned exact-rational arena; the
/// caller then skips certification (sound: no cert emitted).
fn f32_to_rat(f: f32) -> Option<Rat> {
    Rat::from_f32_exact(f)
}

/// Convert a slice of `f32` to exact rationals, or `None` if any does not fit.
fn f32s_to_rats(xs: &[f32]) -> Option<Vec<Rat>> {
    xs.iter().map(|&x| f32_to_rat(x)).collect()
}

/// A purely-sequential FC-ReLU network decomposed into its Linear layers, in
/// order. `hidden` is `(weight, bias)` for each hidden Linear (each followed by a
/// ReLU); `readout` is the final Linear (NOT followed by ReLU).
struct FcReluStack<'a> {
    hidden: Vec<(&'a Array2<f32>, Option<&'a Array1<f32>>)>,
    readout: (&'a Array2<f32>, Option<&'a Array1<f32>>),
}

/// Walk a `GraphNetwork` from output back to the network input along its single
/// linear chain and, if it is a clean FC-ReLU stack (`Linear, ReLU, …, Linear`),
/// return its decomposition in input→output order. Returns `Err(reason)` for any
/// branching/unsupported shape — used only to explain why a network is
/// ineligible, never to abort a run.
fn extract_fc_relu_stack(graph: &GraphNetwork) -> std::result::Result<FcReluStack<'_>, String> {
    // Follow the unique input-side predecessor from the output node back to the
    // network input. Any node with != 1 inputs (binary op / fan-in) or a
    // non-Linear/ReLU layer disqualifies the net.
    let mut chain: Vec<&Layer> = Vec::new();
    let mut current = graph.output_name().to_string();
    if current.is_empty() {
        return Err("graph has no output node".to_string());
    }
    // Bound the walk by the node count to avoid looping on a malformed graph.
    let max_steps = graph.num_nodes().saturating_add(1);
    let mut steps = 0usize;
    loop {
        if steps > max_steps {
            return Err("graph chain did not terminate at the network input".to_string());
        }
        steps += 1;
        let node = graph
            .node(&current)
            .ok_or_else(|| format!("dangling node reference '{current}'"))?;
        // A provable identity is transparent to the chain.
        //
        // `translate_input` lowers every `TraceOp::Input` to `Add(input, 0)` so
        // the traced input node has a layer to be (`ny-trace-bridge`,
        // `translate/ops_core.rs`). That is NY's own construct, and rejecting it
        // here made every trace-lowered graph — i.e. every model that reaches NY
        // through the NN bridge rather than through ONNX — permanently
        // ineligible for a certificate, with the confusing reason
        // "unsupported layer for cert: AddConstant(...)".
        //
        // Only an EXACTLY-zero offset is skipped, checked elementwise. Adding
        // zero maps `[l, u]` to `[l, u]` with no rounding at any finite
        // endpoint, so the layer contributes nothing to the entailment and
        // dropping it certifies the same function. A non-zero offset still
        // disqualifies: it is a real affine term the Farkas rows would have to
        // carry.
        if let Layer::AddConstant(add) = node.layer() {
            if add.constant().iter().all(|&c| c == 0.0) {
                let inputs = node.inputs();
                if inputs.len() != 1 {
                    return Err(format!(
                        "node '{current}' has {} inputs (cert needs a single linear chain)",
                        inputs.len()
                    ));
                }
                let pred = inputs[0].clone();
                if pred == NETWORK_INPUT {
                    break;
                }
                current = pred;
                continue;
            }
        }
        match node.layer() {
            Layer::Linear(_) | Layer::ReLU(_) => {}
            other => return Err(format!("unsupported layer for cert: {other:?}")),
        }
        chain.push(node.layer());
        let inputs = node.inputs();
        if inputs.len() != 1 {
            return Err(format!(
                "node '{current}' has {} inputs (cert needs a single linear chain)",
                inputs.len()
            ));
        }
        let pred = inputs[0].clone();
        if pred == NETWORK_INPUT {
            break;
        }
        current = pred;
    }
    // `chain` is output→input; reverse to input→output.
    chain.reverse();

    // Apply the same alternation logic as the CLI adapter: collect Linear layers
    // in order, requiring a ReLU between consecutive ones, reject anything else.
    let mut linears: Vec<(&Array2<f32>, Option<&Array1<f32>>)> = Vec::new();
    let mut expect_relu = false;
    for layer in chain {
        match layer {
            Layer::Linear(lin) => {
                if expect_relu {
                    return Err("two Linear layers with no ReLU between them".to_string());
                }
                linears.push((lin.weight(), lin.bias()));
                expect_relu = true;
            }
            Layer::ReLU(_) => {
                if !expect_relu {
                    return Err("ReLU with no preceding Linear".to_string());
                }
                expect_relu = false;
            }
            // Unreachable: the walk above already rejected other layer kinds.
            other => return Err(format!("unsupported layer for cert: {other:?}")),
        }
    }
    if linears.len() < 2 {
        return Err(format!(
            "need >=2 Linear layers (>=1 hidden + readout), found {}",
            linears.len()
        ));
    }
    if !expect_relu {
        return Err("final Linear is followed by a ReLU (not an affine read-out)".to_string());
    }
    let readout = linears.pop().expect("len>=2 checked above");
    Ok(FcReluStack {
        hidden: linears,
        readout,
    })
}

/// A single linear output inequality `margin(x) >= 0` whose truth over the whole
/// input box discharges one conjunct of the verified property. `coeff` scales the
/// readout output index `idx` (`+1` lower-bound conjunct `Y_idx >= lo`, `-1`
/// upper-bound conjunct `Y_idx <= hi`); `const_term` folds in the threshold.
struct Margin {
    idx: usize,
    coeff: i32,
    const_term: Rat,
    label: String,
}

/// Derive every per-output-interval conjunct of `bounds` as a safety margin,
/// labelling each with `label_prefix` (e.g. `"Y"` for the legacy output-bounds
/// property, or a constraint-specific tag). For each output `i` with a finite
/// bound the verifier proves: `Y_i >= lower_i` (margin `Y_i - lower_i >= 0`)
/// and/or `Y_i <= upper_i` (margin `upper_i - Y_i >= 0`). Returns `Err(reason)`
/// if any finite bound does not convert to an exact rational. Returns an EMPTY
/// vec (not an error) when `bounds` carries no finite endpoint — that interval
/// property is vacuously true and contributes no conjunct; the caller decides
/// whether the overall property still has at least one conjunct to certify.
/// SOUND: the interval property is the CONJUNCTION of all these constraints, so
/// certifying ALL of them is required and sufficient.
fn margins_for_bounds(
    bounds: &[Bound],
    label_prefix: &str,
) -> std::result::Result<Vec<Margin>, String> {
    let mut margins = Vec::new();
    for (i, b) in bounds.iter().enumerate() {
        let lo = b.lower();
        let hi = b.upper();
        if lo.is_finite() {
            let thr =
                f32_to_rat(lo).ok_or_else(|| format!("output[{i}] lower not representable"))?;
            // Y_i >= lo  ->  margin = Y_i - lo
            margins.push(Margin {
                idx: i,
                coeff: 1,
                const_term: thr.neg(),
                label: format!("{label_prefix}_{i} >= {lo}"),
            });
        }
        if hi.is_finite() {
            let thr =
                f32_to_rat(hi).ok_or_else(|| format!("output[{i}] upper not representable"))?;
            // Y_i <= hi  ->  margin = hi - Y_i
            margins.push(Margin {
                idx: i,
                coeff: -1,
                const_term: thr,
                label: format!("{label_prefix}_{i} <= {hi}"),
            });
        }
    }
    Ok(margins)
}

/// Build the exact-rational [`DeepReluProblem`] whose scalar read-out is the
/// given `margin` over the FC-ReLU stack and the input box. Returns
/// `Err(reason)` (non-fatal) on any conversion miss.
fn build_problem_for_margin(
    stack: &FcReluStack<'_>,
    input_lower: &[f32],
    input_upper: &[f32],
    margin: &Margin,
) -> std::result::Result<DeepReluProblem, String> {
    let neurons: usize = stack.hidden.iter().map(|(w, _)| w.nrows()).sum();
    if neurons > MAX_CERT_NEURONS {
        return Err(format!(
            "net too large for exact cert ({neurons} hidden neurons)"
        ));
    }

    let mut weights: Vec<Vec<Vec<Rat>>> = Vec::with_capacity(stack.hidden.len());
    let mut biases: Vec<Vec<Rat>> = Vec::with_capacity(stack.hidden.len());
    for (w, b) in &stack.hidden {
        let mut rows = Vec::with_capacity(w.nrows());
        for row in w.rows() {
            rows.push(f32s_to_rats(&row.to_vec()).ok_or("weight not representable as rational")?);
        }
        let bias = match b {
            Some(bias) => f32s_to_rats(bias.as_slice().ok_or("non-contiguous bias")?)
                .ok_or("bias not representable as rational")?,
            None => vec![Rat::ZERO; w.nrows()],
        };
        weights.push(rows);
        biases.push(bias);
    }

    // Read-out: out_weight[h] = coeff · finalW[idx][h];
    //           out_bias      = coeff · finalB[idx] + const_term.
    let (rw, rb) = stack.readout;
    let hidden_width = rw.ncols();
    let mut out_weight = vec![Rat::ZERO; hidden_width];
    let mut out_bias = margin.const_term;
    let badrat = |e: RatError| format!("exact arithmetic: {e}");
    if margin.idx >= rw.nrows() {
        return Err(format!(
            "readout has {} rows, output idx {}",
            rw.nrows(),
            margin.idx
        ));
    }
    let s = Rat::from_int(i128::from(margin.coeff));
    for (h, &wv) in rw.row(margin.idx).iter().enumerate() {
        let wr = f32_to_rat(wv).ok_or("readout weight not representable")?;
        out_weight[h] = s.mul(wr).map_err(badrat)?;
    }
    if let Some(b) = rb {
        let br = f32_to_rat(b[margin.idx]).ok_or("readout bias not representable")?;
        out_bias = out_bias.add(s.mul(br).map_err(badrat)?).map_err(badrat)?;
    }

    let lower = f32s_to_rats(input_lower).ok_or("input lower not representable")?;
    let upper = f32s_to_rats(input_upper).ok_or("input upper not representable")?;

    Ok(DeepReluProblem {
        weights,
        biases,
        out_weight,
        out_bias,
        input_lower: lower,
        input_upper: upper,
        alpha: None,
        interm_round: false,
    })
}

/// Certify a single `DeepReluProblem`'s margin, self-check it, and append one
/// Clean-canonical JSON entry to `entries`. Returns `Err(reason)` (non-fatal)
/// the moment exact CROWN cannot close it or its self-check fails — the caller
/// then omits the WHOLE certificate (never a partial one).
///
/// `strict` selects what relation the conjunct attests:
///  * `strict == false` (NON-strict conjunct: output_bounds intervals, Linear
///    halfspaces `a·y <= b` / `a·y >= b`): the property is `margin >= 0`, which
///    the proven exact lower bound `L >= 0` discharges exactly. The conjunct is
///    labelled `>= 0`.
///  * `strict == true` (STRICT conjunct: `ArgmaxMargin` requires `margin > 0`):
///    `L >= 0` is NOT sufficient (a tie `L == 0` would be admitted by a kernel
///    checker that only verifies the `>= 0` inequality). We REQUIRE `L > 0`
///    (exact strict-positivity via [`Rat::is_positive`]); if `L` is not strictly
///    positive the conjunct is NOT discharged and we return `Err` => the WHOLE
///    certificate is refused (fail-closed, no overclaim). The conjunct is then
///    labelled `> 0`, honestly stating the strict relation `L > 0` justifies.
///
/// `fp_margin` is `Some(delta)` ONLY for a strict conjunct under an active
/// [`DeployedPrecision::DeployedF32`] policy: the gate then tightens from
/// `L > 0` to `L > delta`, and the discharged conjunct additionally records the
/// folded threshold so the artifact states the deployed claim it now supports.
/// It is always `None` under the default policy, which keeps the emitted bytes
/// bit-identical to the pre-policy emitter.
fn certify_problem_into(
    problem: &DeepReluProblem,
    label: &str,
    strict: bool,
    fp_margin: Option<Rat>,
    entries: &mut Vec<serde_json::Value>,
    alethe: &mut Option<AletheAcc>,
) -> std::result::Result<(), String> {
    // certify(0): prove margin >= 0 exactly. Err means exact plain CROWN could
    // not close this conjunct (the verdict may rely on tighter alpha/beta/BaB
    // bounds); we then omit the whole cert (sound).
    let certified = problem
        .certify(Rat::ZERO)
        .map_err(|e| format!("exact CROWN did not close '{label}': {e}"))?;
    // Preserve the v1 wire shape (`n/d`, including `/1`) while using the
    // checked arena read rather than the legacy unchecked accessors.
    let (lower_num, lower_den) = certified
        .lower_bound
        .checked_parts()
        .map_err(|e| format!("exact lower bound for '{label}' is not serialisable: {e}"))?;
    let lower_bound = format!("{lower_num}/{lower_den}");

    // FAIL-CLOSED strictness gate: for a STRICT conjunct (argmax) the real
    // property is `margin > 0`. The kernel only checks the `>= 0` inequality, so
    // a tie (`L == 0`) would be silently accepted as if it proved strict argmax.
    // Require the proven exact lower bound `L` to be STRICTLY POSITIVE (not just
    // non-negative); otherwise the conjunct is undischarged and the whole cert is
    // refused. This makes strictness kernel-checkable: the emitted conjunct
    // attests `exact_lower_bound = L` with `L > 0`, which entails `margin > 0`.
    //
    // Under a DEPLOYED-PRECISION policy the same gate is folded with the
    // deployed f32 rounding margin `delta`: the deployed comparison reads
    // `fl32` outputs, which sit within `delta` of the real ones, so `L > 0` no
    // longer implies deployed dominance while `L > delta` does. The threshold
    // is recorded on the conjunct below.
    let mut deployed_record: Option<serde_json::Value> = None;
    if strict {
        match fp_margin {
            None => {
                if !certified.lower_bound.is_positive() {
                    return Err(format!(
                        "strict conjunct '{label}' not discharged: exact lower bound \
                         {lower_bound} is not > 0 (a tie does not establish strict argmax)"
                    ));
                }
            }
            Some(delta) => {
                let threshold = delta
                    .to_clean_string()
                    .map_err(|e| format!("deployed-f32 margin for '{label}' unprintable: {e}"))?;
                if certified.lower_bound <= delta {
                    return Err(format!(
                        "strict conjunct '{label}' not discharged under the deployed-f32 policy: \
                         exact lower bound {lower_bound} does not exceed the deployed rounding \
                         margin {threshold} (the deployed f32 comparison could tie or invert)"
                    ));
                }
                deployed_record = Some(serde_json::json!({
                    "policy": "deployed-f32",
                    "fp_margin": threshold,
                    "folded_threshold": threshold,
                    "attests": "the exact lower bound L exceeds the deployed f32 rounding margin \
                                delta of the compared outputs, so the DEPLOYED (f32) comparison \
                                is strictly satisfied: margin(fl32 outputs) >= L - delta > 0",
                }));
            }
        }
    }

    // In-tree self-check (entailment + Farkas replay): refuse to emit a
    // certificate that is not internally valid.
    cert_check_entailment(&certified.entailment)
        .map_err(|e| format!("entailment self-check failed for '{label}': {e}"))?;
    cert_check_farkas(&certified.farkas)
        .map_err(|e| format!("farkas self-check failed for '{label}': {e}"))?;

    let entailment_json = entailment_to_json(&certified.entailment)
        .map_err(|e| format!("entailment not serialisable for '{label}': {e}"))?;
    let farkas_json = farkas_to_json(&certified.farkas)
        .map_err(|e| format!("farkas not serialisable for '{label}': {e}"))?;
    // BRIDGE: ny-cert's bounded `Json` carries no serde impls by design; this
    // entry is still assembled with `serde_json::json!`, so re-parse the
    // canonical rendering. Removable once this file migrates to
    // `ny_cert::Json` end-to-end.
    let entailment_json: serde_json::Value = serde_json::from_str(&entailment_json.to_string())
        .map_err(|e| format!("entailment JSON bridge for '{label}': {e}"))?;
    let farkas_json: serde_json::Value = serde_json::from_str(&farkas_json.to_string())
        .map_err(|e| format!("farkas JSON bridge for '{label}': {e}"))?;
    // Be honest about what L attests: a strict conjunct proves `margin > 0`
    // (justified by the just-checked `L > 0`); a non-strict one proves
    // `margin >= 0`. A Clean kernel re-checking only `>= 0` together with the
    // attested `L > 0` can re-derive the strict relation itself.
    let relation = if strict { "margin > 0" } else { "margin >= 0" };
    let mut entry = serde_json::json!({
        "discharged_conjunct": label,
        "relation": relation,
        "strict": strict,
        "exact_lower_bound": lower_bound,
        "depth": problem.depth(),
        "entailment": entailment_json,
        "farkas": farkas_json,
    });
    // ADDITIVE, and only under an active deployed-precision policy: `relation`
    // and every field above keep their exact pre-policy meaning (the real
    // relation `L` proves), so a checker that ignores the extra object still
    // checks a true statement. Under the default policy the entry is
    // byte-identical to the pre-policy emitter's.
    if let Some(record) = deployed_record {
        entry["deployed_precision"] = record;
    }
    entries.push(entry);

    // OPTIONAL Alethe emission for this (fully discharged, self-checked)
    // conjunct. Whole-or-nothing: the first refusal clears the collected set
    // and stops collecting; it NEVER withholds the JSON certificate — the
    // Alethe pair is an additive artifact on top of an additive artifact, and
    // its emitters can refuse shapes (equalities, non-SMT-LIB symbols) the
    // certificate path soundly accepts.
    if let Some(acc) = alethe.as_mut() {
        if acc.refusal.is_none() {
            let emitted = entailment_to_alethe(&certified.entailment)
                .map_err(|e| format!("entailment Alethe emission refused for '{label}': {e}"))
                .and_then(|entailment| {
                    farkas_to_alethe(&certified.farkas)
                        .map_err(|e| format!("farkas Alethe emission refused for '{label}': {e}"))
                        .map(|farkas| ConjunctAlethe {
                            label: label.to_string(),
                            entailment,
                            farkas,
                        })
                });
            match emitted {
                Ok(conjunct) => acc.conjuncts.push(conjunct),
                Err(reason) => {
                    // Fail-closed for the Alethe SET: never leave a partial
                    // collection around that a caller could mistake for full
                    // property coverage.
                    acc.conjuncts.clear();
                    acc.refusal = Some(reason);
                }
            }
        }
    }
    Ok(())
}

/// Certify EVERY `margin >= 0` over the FC-ReLU `stack` and the input box
/// `[lo, hi]`, appending one self-checked Clean-canonical JSON entry per margin
/// to `entries`. Returns `Err(reason)` (non-fatal) the moment any conjunct
/// cannot be built, exact CROWN cannot close it, or its self-check fails — the
/// caller then omits the WHOLE certificate (never a partial one).
fn certify_margins_into(
    stack: &FcReluStack<'_>,
    lo: &[f32],
    hi: &[f32],
    margins: &[Margin],
    entries: &mut Vec<serde_json::Value>,
    alethe: &mut Option<AletheAcc>,
) -> std::result::Result<(), String> {
    for margin in margins {
        let problem = build_problem_for_margin(stack, lo, hi, margin)?;
        // Interval/bounds margins are NON-strict (`margin >= 0` is exactly the
        // property); `L >= 0` discharges them, and the deployed-precision fold
        // is a STRICT-gate policy (see `DeployedPrecision`), so `None` here.
        certify_problem_into(&problem, &margin.label, false, None, entries, alethe)?;
    }
    Ok(())
}

/// Exact-rational bound on the deployed-f32 rounding error of ORIGINAL network
/// output `out_idx` over the input box `[lo, hi]`, memoized in `cache`.
///
/// The problem handed to [`deployed_fp_output_margin`] is the deployed
/// computation verbatim: the original FC-ReLU hidden stack, and the scalar
/// read-out row `(Wr[out_idx], br[out_idx])` with NO folded threshold and unit
/// coefficient — so the returned `e` satisfies
/// `|fl32(y_out_idx(x)) - y_out_idx(x)| <= e` for every `x` in the box, where
/// `fl32` is the deployed f32 inference of the same weights.
///
/// This is deliberately NOT computed on the folded margin row: a margin row
/// like `Wr[class] - Wr[j]` is a different (and possibly much smaller) weight
/// vector than either deployed row, so its error model would describe a network
/// that fuses the comparison into the read-out — not the one that ships, which
/// rounds each logit and only then compares them.
///
/// Returns `Err(reason)` (non-fatal at this level, fatal for the certificate)
/// when the margin is unrepresentable — fail-closed by construction.
fn deployed_output_error(
    stack: &FcReluStack<'_>,
    lo: &[f32],
    hi: &[f32],
    out_idx: usize,
    cache: &mut DeployedErrorCache,
) -> std::result::Result<Rat, String> {
    if let Some(cached) = cache.get(&out_idx) {
        return Ok(*cached);
    }
    let readout = Margin {
        idx: out_idx,
        coeff: 1,
        const_term: Rat::ZERO,
        label: format!("deployed read-out row {out_idx}"),
    };
    let problem = build_problem_for_margin(stack, lo, hi, &readout)?;
    let error = deployed_fp_output_margin(&problem).map_err(|e| {
        format!("deployed-f32 rounding margin for network output {out_idx} is unavailable: {e}")
    })?;
    cache.insert(out_idx, error);
    Ok(error)
}

/// The deployed-f32 threshold row `k` of a margin map `M` must clear:
/// `delta_k = sum_o |M[k][o]| * e_o`, where `e_o` is output `o`'s deployed
/// rounding error ([`deployed_output_error`]).
///
/// SOUNDNESS. The deployed comparison evaluates `sum_o M[k][o] * fl32(y_o) +
/// c[k]`, which differs from the certified real margin `sum_o M[k][o] * y_o +
/// c[k]` by at most `sum_o |M[k][o]| * |fl32(y_o) - y_o| <= delta_k` (the
/// constant `c[k]` is exact and contributes nothing). So a proven real lower
/// bound `L > delta_k` gives `deployed margin >= L - delta_k > 0`. For an
/// `ArgmaxMargin` row (`M[k] = e_class - e_j`, `c[k] = 0`) that is exactly
/// "the deployed f32 logits still rank `class` strictly above `j`".
///
/// Every arithmetic step is exact rational: no float rounding enters the
/// threshold itself.
fn deployed_margin_threshold(
    stack: &FcReluStack<'_>,
    lo: &[f32],
    hi: &[f32],
    m: &Array2<f32>,
    k: usize,
    label: &str,
    cache: &mut DeployedErrorCache,
) -> std::result::Result<Rat, String> {
    let badrat = |e: RatError| format!("exact arithmetic while folding '{label}': {e}");
    if k >= m.nrows() {
        return Err(format!(
            "deployed-f32 fold: margin row {k} out of range ({} rows)",
            m.nrows()
        ));
    }
    let mut threshold = Rat::ZERO;
    for (o, &coeff) in m.row(k).iter().enumerate() {
        let coeff_rat = f32_to_rat(coeff).ok_or_else(|| {
            format!("deployed-f32 fold: margin coeff not representable ({label})")
        })?;
        if coeff_rat == Rat::ZERO {
            continue;
        }
        let error = deployed_output_error(stack, lo, hi, o, cache)?;
        threshold = threshold
            .add(coeff_rat.abs().mul(error).map_err(badrat)?)
            .map_err(badrat)?;
    }
    Ok(threshold)
}

/// Recover the appended affine margin read-out `(M, c)` from an augmented margin
/// network produced by [`augment_for_constraint`]: `M` is the margin layer's
/// weight `(margin_out, orig_out_width)` and `c` its bias. Returns the exact f32
/// matrix/vector (the encoder's margin coefficients are small exact values such
/// as `+/-1` and the user's `coeffs`/`bias`).
fn margin_layer_from_augmented(
    augmented: &GraphNetwork,
) -> std::result::Result<(Array2<f32>, Array1<f32>), String> {
    let out = augmented.output_name();
    let node = augmented
        .node(out)
        .ok_or_else(|| format!("augmented net has no output node '{out}'"))?;
    match node.layer() {
        Layer::Linear(lin) => {
            let m = lin.weight().clone();
            let c = lin
                .bias()
                .cloned()
                .unwrap_or_else(|| Array1::zeros(m.nrows()));
            if c.len() != m.nrows() {
                return Err(format!(
                    "margin layer bias len {} != rows {}",
                    c.len(),
                    m.nrows()
                ));
            }
            Ok((m, c))
        }
        other => Err(format!(
            "augmented output is not a Linear margin layer: {other:?}"
        )),
    }
}

/// Build the exact-rational [`DeepReluProblem`] whose scalar read-out is row `k`
/// of a margin map `M·y + c` COMPOSED with the original FC-ReLU stack's affine
/// read-out `y = Wr·h + br`, i.e. `margin_k(h) = (M[k]·Wr)·h + (M[k]·br + c[k])`.
///
/// The composition is performed ENTIRELY in exact rationals, so the certified
/// problem is the true network's margin (no f32 round-off is introduced by
/// folding the two consecutive affine layers). Hidden layers are the original
/// stack's, verbatim. Returns `Err(reason)` (non-fatal) on any conversion miss.
#[allow(clippy::too_many_arguments)]
fn build_problem_for_margin_net_row(
    stack: &FcReluStack<'_>,
    input_lower: &[f32],
    input_upper: &[f32],
    m: &Array2<f32>,
    c: &Array1<f32>,
    k: usize,
    label: &str,
) -> std::result::Result<DeepReluProblem, String> {
    let neurons: usize = stack.hidden.iter().map(|(w, _)| w.nrows()).sum();
    if neurons > MAX_CERT_NEURONS {
        return Err(format!(
            "net too large for exact cert ({neurons} hidden neurons)"
        ));
    }
    let badrat = |e: RatError| format!("exact arithmetic: {e}");

    // Hidden layers (Linear+ReLU), exactly as the original stack.
    let mut weights: Vec<Vec<Vec<Rat>>> = Vec::with_capacity(stack.hidden.len());
    let mut biases: Vec<Vec<Rat>> = Vec::with_capacity(stack.hidden.len());
    for (w, b) in &stack.hidden {
        let mut rows = Vec::with_capacity(w.nrows());
        for row in w.rows() {
            rows.push(f32s_to_rats(&row.to_vec()).ok_or("weight not representable as rational")?);
        }
        let bias = match b {
            Some(bias) => f32s_to_rats(bias.as_slice().ok_or("non-contiguous bias")?)
                .ok_or("bias not representable as rational")?,
            None => vec![Rat::ZERO; w.nrows()],
        };
        weights.push(rows);
        biases.push(bias);
    }

    // Compose row k of the margin map with the original readout, in rationals.
    let (rw, rb) = stack.readout; // rw: (orig_out_width, hidden_width)
    let orig_width = rw.nrows();
    let hidden_width = rw.ncols();
    if k >= m.nrows() {
        return Err(format!("margin row {k} out of range ({} rows)", m.nrows()));
    }
    if m.ncols() != orig_width {
        return Err(format!(
            "margin layer expects {} inputs but original readout emits {} ({label})",
            m.ncols(),
            orig_width
        ));
    }

    // out_weight[h] = sum_o M[k][o] * Wr[o][h]
    let mut out_weight = vec![Rat::ZERO; hidden_width];
    // out_bias = c[k] + sum_o M[k][o] * br[o]
    let mut out_bias = f32_to_rat(c[k]).ok_or("margin bias not representable")?;
    for o in 0..orig_width {
        let mko = f32_to_rat(m[[k, o]]).ok_or("margin coeff not representable")?;
        if mko == Rat::ZERO {
            continue;
        }
        for h in 0..hidden_width {
            let wr = f32_to_rat(rw[[o, h]]).ok_or("readout weight not representable")?;
            out_weight[h] = out_weight[h]
                .add(mko.mul(wr).map_err(badrat)?)
                .map_err(badrat)?;
        }
        if let Some(br) = rb {
            let bro = f32_to_rat(br[o]).ok_or("readout bias not representable")?;
            out_bias = out_bias
                .add(mko.mul(bro).map_err(badrat)?)
                .map_err(badrat)?;
        }
    }

    let lower = f32s_to_rats(input_lower).ok_or("input lower not representable")?;
    let upper = f32s_to_rats(input_upper).ok_or("input upper not representable")?;

    Ok(DeepReluProblem {
        weights,
        biases,
        out_weight,
        out_bias,
        input_lower: lower,
        input_upper: upper,
        alpha: None,
        interm_round: false,
    })
}

/// Attempt to build, self-check, and serialize an exact-rational certificate for
/// EVERY conjunct of `spec`'s output property over the eligible FC-ReLU `stack`.
///
/// The property is the CONJUNCTION of:
///  * every finite `spec.output_bounds()` interval conjunct (legacy behaviour),
///    certified against the original-network `stack`; AND
///  * every `spec.output_constraints()` conjunct:
///      - `Bounds(v)` -> per-output interval conjuncts over the original stack;
///      - `Linear` / `ArgmaxMargin` -> the margin network produced by
///        [`augment_for_constraint`] (original FC-ReLU stack + an appended affine
///        margin read-out, still a sequential FC-ReLU net), each of whose margin
///        outputs is certified `>= 0` via the SAME [`DeepReluProblem`] path.
///        `ArgmaxMargin` yields one margin per competitor class; ALL must close.
///
/// Returns `Ok(json)` only if EVERY requested conjunct (bounds AND constraints)
/// certifies AND its entailment + Farkas certificates pass the in-tree
/// self-check; otherwise `Err(reason)` (sound: the extra artifact is simply
/// omitted, the verdict stands). NEVER emits a partial certificate while a
/// requested conjunct is uncertified. The combined JSON is the Clean-canonical
/// certificate set, one entry per discharged conjunct.
///
/// Under a [`DeployedPrecision::DeployedF32`] `policy` every STRICT conjunct's
/// gate is folded with that conjunct's deployed rounding margin
/// ([`deployed_margin_threshold`]) and the whole certificate is refused if any
/// such margin cannot be computed. Non-strict conjuncts are untouched — the fold
/// is a strict-dominance policy, and the emitted top-level `precision_policy`
/// object says so verbatim so the artifact never implies wider coverage.
fn try_build_certificate(
    stack: &FcReluStack<'_>,
    net: &GraphNetwork,
    spec: &VerificationSpec,
    alethe: &mut Option<AletheAcc>,
    policy: CertifyPolicy,
) -> std::result::Result<String, String> {
    let lo: Vec<f32> = spec.input_bounds().iter().map(Bound::lower).collect();
    let hi: Vec<f32> = spec.input_bounds().iter().map(Bound::upper).collect();

    let mut entries: Vec<serde_json::Value> = Vec::new();
    let mut deployed_errors = DeployedErrorCache::new();

    // (a) Legacy output-bounds conjuncts over the original stack. An empty result
    // is fine ONLY because the constraints below may carry the actual conjuncts;
    // we enforce "at least one conjunct overall" after collecting everything.
    let bound_margins = margins_for_bounds(spec.output_bounds(), "Y")?;
    certify_margins_into(stack, &lo, &hi, &bound_margins, &mut entries, alethe)?;

    // (b) Rich output constraints. Each must be reduced and fully discharged, or
    // we refuse the whole certificate (no overclaim).
    for (ci, constraint) in spec.output_constraints().iter().enumerate() {
        match constraint {
            OutputConstraint::Bounds(bounds) => {
                // Per-output interval property over the ORIGINAL network outputs.
                let label = format!("constraint[{ci}].Y");
                let margins = margins_for_bounds(bounds, &label)?;
                if margins.is_empty() {
                    return Err(format!(
                        "output_constraint[{ci}] Bounds has no finite endpoint to certify"
                    ));
                }
                certify_margins_into(stack, &lo, &hi, &margins, &mut entries, alethe)?;
            }
            OutputConstraint::Linear { .. } | OutputConstraint::ArgmaxMargin { .. } => {
                // Reduce to a margin network via the P7 encoder. The encoder
                // appends a single affine read-out `M·y + c` (one row per margin:
                // 1 for Linear, one-per-competitor for ArgmaxMargin) onto the
                // ORIGINAL net's output, yielding `... -> Linear(readout) ->
                // Linear(margin)`. We do NOT collapse those two affine layers in
                // f32 (that would certify a rounded net); instead we COMPOSE them
                // EXACTLY in rationals — `margin_k = (M[k]·Wr)·h + (M[k]·br + c[k])`
                // — reusing the original FC-ReLU hidden stack verbatim and
                // certifying each composed margin row `>= 0`.
                let augmented = augment_for_constraint(net, constraint).map_err(|e| {
                    format!("output_constraint[{ci}] could not be reduced to a margin net: {e}")
                })?;
                let (m, c) = margin_layer_from_augmented(&augmented).map_err(|e| {
                    format!("output_constraint[{ci}] margin layer not recoverable: {e}")
                })?;
                let width = m.nrows();
                if width == 0 {
                    return Err(format!(
                        "output_constraint[{ci}] margin net produced zero margin outputs"
                    ));
                }
                // `ArgmaxMargin` is a STRICT property: `y[class] > y[j]` for every
                // competitor `j` (margin > 0). A tie (margin == 0) does NOT
                // establish strict argmax, so we require each competitor margin's
                // exact lower bound to be `> 0` (see `certify_problem_into`).
                // `Linear` halfspaces (`a·y <= b` / `a·y >= b`) are non-strict.
                let strict = matches!(constraint, OutputConstraint::ArgmaxMargin { .. });
                let rel = if strict { "> 0" } else { ">= 0" };
                for k in 0..width {
                    let label = if strict {
                        format!("constraint[{ci}].margin[{k}] {rel} (exact_lower_bound L > 0)")
                    } else {
                        format!("constraint[{ci}].margin[{k}] {rel}")
                    };
                    // Fail-closed: an unavailable deployed margin aborts the
                    // WHOLE certificate rather than silently reverting this
                    // conjunct to the real-semantics gate.
                    let fp_margin = if strict && policy.folds_deployed_fp_margin() {
                        Some(deployed_margin_threshold(
                            stack,
                            &lo,
                            &hi,
                            &m,
                            k,
                            &label,
                            &mut deployed_errors,
                        )?)
                    } else {
                        None
                    };
                    let problem =
                        build_problem_for_margin_net_row(stack, &lo, &hi, &m, &c, k, &label)?;
                    certify_problem_into(
                        &problem,
                        &label,
                        strict,
                        fp_margin,
                        &mut entries,
                        alethe,
                    )?;
                }
            }
        }
    }

    if entries.is_empty() {
        return Err("property has no finite conjunct to certify".to_string());
    }

    let has_constraints = !spec.output_constraints().is_empty();
    let claim = if has_constraints {
        "exact CROWN discharges every conjunct of the output property (output_bounds AND \
         output_constraints) over the whole input box"
    } else {
        "exact CROWN discharges every conjunct of the output property (output_bounds) over the \
         whole input box"
    };
    let mut payload = serde_json::json!({
        "format": "ny-cert/crown-deep/v1",
        "claim": claim,
        "covers_output_bounds": true,
        "covers_output_constraints": has_constraints,
        "conjuncts": entries,
    });
    // ADDITIVE and opt-in: absent (byte-for-byte) under the default policy. Its
    // `scope` states the exact reach of the fold so the artifact never reads as
    // "this whole certificate is deployment-covered".
    if policy.folds_deployed_fp_margin() {
        payload["precision_policy"] = serde_json::json!({
            "mode": "deployed-f32",
            "scope": "strict (argmax dominance) conjuncts only: each such conjunct attests \
                      L > delta, where delta bounds the deployed f32 rounding error of the \
                      outputs it compares, hence the dominance holds for the deployed network. \
                      Non-strict conjuncts attest the ideal real-valued semantics, exactly as \
                      under the default policy.",
            "model": "layer-wise forward error analysis in exact rationals (ny-cert::fp_margin); \
                      assumes the f32 weights ARE the deployed weights, per-operation \
                      round-to-nearest, and exact f32 inputs (input quantization error is out of \
                      scope and must be added by the caller).",
        });
    }
    serde_json::to_string(&payload).map_err(|e| format!("certificate serialisation failed: {e}"))
}

/// Populate the (otherwise dead) [`VerificationProof`] channel on a `Verified`
/// result with `proof`. No-op for any non-`Verified` verdict.
///
/// # ENSURES
/// - If `result` is `Verified`, the returned result is `Verified` with
///   `proof == Some(Box::new(proof))`.
/// - Otherwise the result is returned unchanged.
#[must_use]
pub fn attach_proof(
    mut result: VerificationResult,
    proof: VerificationProof,
) -> VerificationResult {
    if let VerificationResult::Verified { proof: slot, .. } = &mut result {
        *slot = Some(Box::new(proof));
    }
    result
}

/// Verify a spec on a `GraphNetwork` and, when sound to do so, attach an
/// exact-rational, Clean-checkable proof-carrying certificate to the verdict.
///
/// The verified property — and the property the certificate covers — is the
/// CONJUNCTION of `spec.output_bounds()` (legacy per-output intervals) AND every
/// `spec.output_constraints()` conjunct (P7 halfspace [`OutputConstraint::Linear`]
/// and robustness [`OutputConstraint::ArgmaxMargin`], plus [`OutputConstraint::Bounds`]).
/// The verdict is computed over that same conjunction (see [`combined_verdict`]),
/// so `result` and the certificate never disagree about what was proven.
///
/// The certificate is a purely additive artifact and never alters the verdict:
///
/// * **Ineligible** (the network — or, for a `Linear`/`ArgmaxMargin` constraint,
///   the augmented margin network — is not a sequential FC-ReLU net, or has a
///   fan-in/binary op, or the property has no finite conjunct): returns the plain
///   verdict with `eligible = false`, `certificate_json = None`, and a `note`
///   explaining why. No certificate is built and no false claim is made.
/// * **Eligible but not `Verified`**: a non-verified verdict NEVER receives a
///   certificate — `certificate_json = None`.
/// * **Eligible and `Verified`**: re-derives the exact-rational certificate for
///   EVERY requested conjunct (bounds AND constraints) — reducing each `Linear` /
///   `ArgmaxMargin` constraint to a margin network via
///   [`augment_for_constraint`] and certifying every margin output `>= 0` — then
///   self-checks each via [`check_entailment`]/[`check_farkas`], and — only if
///   ALL pass — emits the Clean-canonical JSON, wraps it in a
///   [`VerificationProof`] carried as bytes, and attaches it to the verdict's
///   proof channel. If ANY requested conjunct cannot be eligibly built, cannot be
///   closed by exact CROWN, or fails self-check, NO certificate is emitted
///   (`certificate_json = None`) — never a partial certificate that overclaims.
///
/// The verifier runs in `Crown` mode by default. The certificate is replayable
/// independently of the verifier's floating-point bounds.
pub fn certify_graph(net: &GraphNetwork, spec: &VerificationSpec) -> Result<CertifiedResult> {
    certify_graph_inner(net, spec, &mut None, CertifyPolicy::real_semantics())
}

/// [`certify_graph`] under an explicit [`CertifyPolicy`].
///
/// `certify_graph(net, spec)` is exactly
/// `certify_graph_with_policy(net, spec, CertifyPolicy::default())`, and the
/// default policy emits BYTE-IDENTICAL certificates — the policy adds fields
/// only when it is actually engaged.
///
/// With [`CertifyPolicy::deployed_f32`] the STRICT (argmax dominance)
/// conjuncts are held to `L > delta` instead of `L > 0`, where `delta` is the
/// exact-rational bound on the f32 rounding error of the deployed outputs that
/// conjunct compares (see [`DeployedPrecision::DeployedF32`] and
/// [`deployed_fp_output_margin`]). Each discharged strict conjunct then carries
/// a `deployed_precision` object recording the folded threshold, and the
/// certificate carries a top-level `precision_policy` object stating the fold's
/// exact scope.
///
/// FAIL-CLOSED, twice over: a strict conjunct whose `L` does not clear `delta`
/// is not discharged and the whole certificate is refused (never emitted with
/// the weaker gate); and a `delta` that cannot be computed at all — a parameter
/// that is not the exact value of a finite f32, an overflow-capable magnitude
/// envelope, an affine row outside the error model's domain — likewise refuses
/// the whole certificate rather than silently reverting to real semantics. In
/// both cases the VERDICT is untouched, exactly as for any other certificate
/// refusal.
pub fn certify_graph_with_policy(
    net: &GraphNetwork,
    spec: &VerificationSpec,
    policy: CertifyPolicy,
) -> Result<CertifiedResult> {
    certify_graph_inner(net, spec, &mut None, policy)
}

/// [`certify_graph`] plus per-conjunct Alethe proof emission, threaded through
/// the SAME certificate build (one exact-rational certification per conjunct —
/// the Alethe pairs are emitted from the very [`DeepReluProblem`] results the
/// JSON certificate serializes, never from a re-derivation that could drift).
///
/// The certification outcome ([`AletheCertifiedResult::certified`]) is
/// bit-for-bit the [`certify_graph`] contract: the verdict is never altered,
/// and the JSON certificate is emitted or withheld on exactly the same
/// conditions — an Alethe emission refusal NEVER withholds the certificate.
///
/// On top of that, `alethe` is `Some(pairs)` — one [`ConjunctAlethe`]
/// (entailment + Farkas SMT-LIB/Alethe pair, checkable by the external
/// `carcara` checker) per certificate conjunct, in certificate order — ONLY
/// when the certificate was emitted AND every conjunct emitted. Whole or
/// nothing: if any conjunct's emission is refused (an [`EmitError`], e.g. an
/// unsupported shape), or the collected set does not match the certificate
/// one-to-one, NO Alethe artifact is returned and `alethe_note` explains why —
/// never a partial set that overclaims coverage.
///
/// STRICTNESS is inherited from the certificate build: a strict
/// (`ArgmaxMargin`) conjunct is only ever certified — and therefore only ever
/// emitted — after its exact lower bound is proven strictly positive, so an
/// Alethe pair for a strict conjunct always accompanies a genuinely
/// discharged `margin > 0`.
pub fn certify_graph_with_alethe(
    net: &GraphNetwork,
    spec: &VerificationSpec,
) -> Result<AletheCertifiedResult> {
    certify_graph_with_alethe_and_policy(net, spec, CertifyPolicy::real_semantics())
}

/// [`certify_graph_with_alethe`] under an explicit [`CertifyPolicy`].
///
/// The Alethe surface is unchanged by the policy: a strict conjunct is emitted
/// only after it is discharged, so under [`CertifyPolicy::deployed_f32`] an
/// Alethe pair for a strict conjunct accompanies a gate that already cleared
/// the deployed rounding margin. The emitted SMT-LIB/Alethe pair still proves
/// the certificate's own `>= 0` inequality; the strict, deployed claim rests on
/// the conjunct's recorded `exact_lower_bound` and `deployed_precision`
/// threshold, exactly as the real-semantics strict claim rests on
/// `exact_lower_bound > 0`.
pub fn certify_graph_with_alethe_and_policy(
    net: &GraphNetwork,
    spec: &VerificationSpec,
    policy: CertifyPolicy,
) -> Result<AletheCertifiedResult> {
    let mut acc = Some(AletheAcc::default());
    let certified = certify_graph_inner(net, spec, &mut acc, policy)?;
    let acc = acc.expect("accumulator is Some for the whole build");

    // No certificate -> no Alethe artifact, whatever the accumulator holds
    // (it is cleared on refusal, but a withheld certificate can also leave
    // successfully-emitted conjuncts from before the failing one behind).
    let Some(json) = certified.certificate_json.as_deref() else {
        return Ok(AletheCertifiedResult {
            certified,
            alethe: None,
            alethe_note: "no certificate emitted - no Alethe artifact".to_string(),
        });
    };

    if let Some(reason) = acc.refusal {
        return Ok(AletheCertifiedResult {
            certified,
            alethe: None,
            alethe_note: format!(
                "certificate emitted, but the Alethe artifact is withheld (whole-or-nothing): \
                 {reason}"
            ),
        });
    }

    // FAIL-CLOSED cross-check: the Alethe set must cover the certificate
    // one-to-one (same count, same labels in order). A mismatch means a code
    // bug; withhold the artifact rather than return a set that could be
    // mistaken for full coverage.
    let covered = parse_certificate_conjunct_labels(json);
    let labels: Vec<&str> = acc.conjuncts.iter().map(|c| c.label.as_str()).collect();
    let matches = covered
        .as_ref()
        .is_some_and(|c| c.len() == labels.len() && c.iter().zip(&labels).all(|(a, b)| a == b));
    if !matches {
        return Ok(AletheCertifiedResult {
            certified,
            alethe: None,
            alethe_note: format!(
                "certificate emitted, but the Alethe artifact is withheld: emitted conjuncts \
                 {labels:?} do not match the certificate's {covered:?} (internal invariant \
                 violated; fail-closed)"
            ),
        });
    }

    let n = acc.conjuncts.len();
    Ok(AletheCertifiedResult {
        certified,
        alethe: Some(acc.conjuncts),
        alethe_note: format!(
            "Alethe proofs emitted for all {n} certificate conjunct(s); check each with `carcara \
             check proof.alethe problem.smt2`"
        ),
    })
}

/// Extract the `discharged_conjunct` labels of an emitted certificate, in
/// order. `None` if the JSON does not parse into the expected shape (treated
/// fail-closed by the caller).
fn parse_certificate_conjunct_labels(json: &str) -> Option<Vec<String>> {
    let parsed: serde_json::Value = serde_json::from_str(json).ok()?;
    parsed["conjuncts"]
        .as_array()?
        .iter()
        .map(|c| c["discharged_conjunct"].as_str().map(str::to_string))
        .collect()
}

/// Shared implementation of [`certify_graph`] / [`certify_graph_with_alethe`]:
/// identical verdict and certificate behaviour; `alethe` is `Some` only for
/// the Alethe-threading entry point.
fn certify_graph_inner(
    net: &GraphNetwork,
    spec: &VerificationSpec,
    alethe: &mut Option<AletheAcc>,
    policy: CertifyPolicy,
) -> Result<CertifiedResult> {
    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Crown,
        ..Default::default()
    });
    // The verdict must reflect the SAME property the certificate covers: the
    // conjunction of output_bounds AND output_constraints. `verify_graph` checks
    // only output_bounds; `verify_with_constraints` checks only output_constraints
    // (it ignores output_bounds when constraints are present). When both are
    // present we verify each and conjoin: Verified iff BOTH are Verified, else the
    // first non-Verified verdict (with provenance combined across both runs). This
    // keeps `result` and the certificate honest about the same property.
    let result = combined_verdict(&verifier, net, spec)?;

    // Static eligibility gate (architecture only).
    let stack = match extract_fc_relu_stack(net) {
        Ok(s) => s,
        Err(reason) => {
            return Ok(CertifiedResult {
                result,
                certificate_json: None,
                eligible: false,
                note: format!("ineligible: {reason}"),
            });
        }
    };

    // SOUNDNESS: a certificate is only ever emitted for a Verified verdict.
    if !result.is_verified() {
        return Ok(CertifiedResult {
            result,
            certificate_json: None,
            eligible: true,
            note: "eligible network, but verdict is not Verified — no certificate".to_string(),
        });
    }

    match try_build_certificate(&stack, net, spec, alethe, policy) {
        Ok(json) => {
            let proof = VerificationProof::from_parts(
                ProofFormat::BoundTrace,
                json.clone().into_bytes(),
                None,
                None,
            );
            let result = attach_proof(result, proof);
            let scope = if spec.output_constraints().is_empty() {
                "output_bounds"
            } else {
                "output_bounds AND output_constraints"
            };
            let precision = if policy.folds_deployed_fp_margin() {
                "; strict conjuncts cleared the deployed-f32 rounding margin (policy: deployed-f32)"
            } else {
                ""
            };
            Ok(CertifiedResult {
                result,
                certificate_json: Some(json),
                eligible: true,
                note: format!(
                    "exact-rational certificate covering {scope} emitted and attached to the \
                     proof channel{precision}"
                ),
            })
        }
        Err(reason) => Ok(CertifiedResult {
            result,
            certificate_json: None,
            eligible: true,
            note: format!("verdict stands; certificate omitted: {reason}"),
        }),
    }
}

/// Compute the verifier verdict for the FULL property = conjunction of
/// `spec.output_bounds()` AND `spec.output_constraints()`.
///
/// `Verifier::verify_graph` covers only `output_bounds`;
/// [`verify_with_constraints`] covers only `output_constraints` (it ignores
/// `output_bounds` when constraints are present). When the spec carries
/// constraints we run BOTH and conjoin so the verdict reflects exactly what the
/// certificate claims:
///  * `Verified` iff both the bounds run and the constraints run are `Verified`
///    (provenance combined across both);
///  * otherwise the first non-`Verified` verdict, with provenance combined over
///    both runs — never reporting success while some conjunct is open.
///
/// With no constraints this is exactly the legacy `verify_graph(net, spec)`.
fn combined_verdict(
    verifier: &Verifier,
    net: &GraphNetwork,
    spec: &VerificationSpec,
) -> Result<VerificationResult> {
    // Legacy path: bounds only.
    if spec.output_constraints().is_empty() {
        return verifier.verify_graph(net, spec);
    }

    // Conjunction: bounds AND constraints.
    let bounds_result = verifier.verify_graph(net, spec)?;
    let constraints_result = verify_with_constraints(verifier, net, spec)?;

    let combined_prov =
        SoundnessProvenance::combine(bounds_result.provenance(), constraints_result.provenance());

    // Fail-closed conjunction: if either run is not Verified, surface a
    // non-Verified verdict (prefer the bounds failure first for a stable order),
    // carrying the combined provenance.
    if !bounds_result.is_verified() {
        return Ok(bounds_result.with_provenance(combined_prov));
    }
    if !constraints_result.is_verified() {
        return Ok(constraints_result.with_provenance(combined_prov));
    }

    // Both Verified: report Verified with combined provenance. The certificate
    // (built separately over the same property) carries the machine-checkable
    // evidence; the result's output_bounds are not meaningful for the
    // heterogeneous conjunction, so we leave them empty.
    Ok(VerificationResult::Verified {
        provenance: combined_prov,
        output_bounds: Vec::new(),
        proof: None,
        actual_method: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_propagate::layers::{LinearLayer, ReLULayer, SiLULayer};
    use ny_propagate::GraphNode;

    fn linear(weight: Array2<f32>, bias: Vec<f32>) -> Layer {
        Layer::Linear(LinearLayer::new(weight, Some(Array1::from(bias))).expect("valid linear"))
    }

    /// y = 2 * relu(x0) + 1/2 as a 1->1->1 FC-ReLU graph.
    /// For x0 in [1, 2] (ReLU active) -> y in [5/2, 9/2].
    fn eligible_graph() -> GraphNetwork {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "lin1",
            linear(
                Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
                vec![0.0],
            ),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            linear(
                Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(),
                vec![0.5],
            ),
            vec!["relu".to_string()],
        ));
        g.set_output("lin2");
        g
    }

    /// The trace-lowered shape: NY's own `Add(input, 0)` identity in front of
    /// the FC-ReLU stack, exactly as `translate_input` emits it. Same function
    /// as [`eligible_graph`], so it must certify identically.
    fn trace_lowered_graph() -> GraphNetwork {
        use ny_propagate::layers::AddConstantLayer;
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "input",
            Layer::AddConstant(AddConstantLayer::new(ndarray::arr1(&[0.0f32]).into_dyn())),
        ));
        g.add_node(GraphNode::new(
            "lin1",
            linear(
                Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
                vec![0.0],
            ),
            vec!["input".to_string()],
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            linear(
                Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(),
                vec![0.5],
            ),
            vec!["relu".to_string()],
        ));
        g.set_output("lin2");
        g
    }

    /// A trace-lowered graph must be eligible: NY lowers every `TraceOp::Input`
    /// to `Add(input, 0)`, so rejecting that identity made EVERY model arriving
    /// through the NN bridge permanently uncertifiable, with the misleading
    /// reason "unsupported layer for cert: AddConstant(...)".
    #[test]
    fn identity_add_constant_prefix_does_not_block_eligibility() {
        let lowered = trace_lowered_graph();
        let eligible = eligible_graph();
        let stack = extract_fc_relu_stack(&lowered).expect("identity prefix is transparent");
        let plain = extract_fc_relu_stack(&eligible).expect("plain stack is eligible");
        assert_eq!(
            stack.hidden.len(),
            plain.hidden.len(),
            "the identity must not appear in the extracted chain"
        );
    }

    /// ...but only when it is EXACTLY zero. A real offset is an affine term the
    /// Farkas rows would have to carry, so it must still disqualify rather than
    /// be silently dropped — dropping it would certify a different function.
    #[test]
    fn nonzero_add_constant_prefix_still_disqualifies() {
        use ny_propagate::layers::AddConstantLayer;
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "offset",
            Layer::AddConstant(AddConstantLayer::new(ndarray::arr1(&[0.25f32]).into_dyn())),
        ));
        g.add_node(GraphNode::new(
            "lin1",
            linear(
                Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
                vec![0.0],
            ),
            vec!["offset".to_string()],
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            linear(
                Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(),
                vec![0.5],
            ),
            vec!["relu".to_string()],
        ));
        g.set_output("lin2");
        let err = match extract_fc_relu_stack(&g) {
            Err(reason) => reason,
            Ok(_) => panic!("a real offset must not be skipped"),
        };
        assert!(
            err.contains("AddConstant"),
            "refusal should name the layer, got: {err}"
        );
    }

    /// Same shape but with a SiLU activation instead of ReLU -> ineligible.
    fn ineligible_graph() -> GraphNetwork {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "lin1",
            linear(
                Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
                vec![0.0],
            ),
        ));
        g.add_node(GraphNode::new(
            "silu",
            Layer::SiLU(SiLULayer::new()),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            linear(
                Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(),
                vec![0.5],
            ),
            vec!["silu".to_string()],
        ));
        g.set_output("lin2");
        g
    }

    #[test]
    fn eligible_verified_net_gets_certificate_and_proof_channel() {
        let g = eligible_graph();
        // x0 in [1, 2] -> y in [5/2, 9/2]. Spec asserts y >= 0 (lower bound), a
        // conjunct exact CROWN closes; upper +inf so no upper conjunct.
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(0.0, f32::INFINITY)],
        )
        .expect("valid spec");

        let out = certify_graph(&g, &spec).expect("certify_graph runs");
        assert!(out.eligible, "FC-ReLU net must be eligible: {}", out.note);
        assert!(
            out.result.is_verified(),
            "y in [5/2, 9/2] satisfies y >= 0: {:?}",
            out.result
        );
        let json = out
            .certificate_json
            .as_ref()
            .unwrap_or_else(|| panic!("expected a certificate; note: {}", out.note));
        assert!(json.contains("ny-cert/crown-deep/v1"), "canonical format");

        // The previously-dead proof channel must now be populated.
        let VerificationResult::Verified { proof, .. } = &out.result else {
            panic!("verified result expected");
        };
        let proof = proof.as_ref().expect("proof channel populated");
        assert_eq!(proof.format(), ProofFormat::BoundTrace);
        assert_eq!(proof.as_bytes(), json.as_bytes());

        // ny_cert self-check must pass on the emitted certificate set.
        let parsed: serde_json::Value =
            serde_json::from_str(json).expect("certificate JSON parses");
        let conjuncts = parsed["conjuncts"].as_array().expect("conjuncts array");
        assert!(!conjuncts.is_empty(), "at least one discharged conjunct");
        for c in conjuncts {
            // Re-run a structural self-check: each conjunct carries an exact
            // lower bound that is positive/non-negative for the proven margin.
            assert!(c["entailment"].is_object(), "entailment present");
            assert!(c["farkas"].is_object(), "farkas present");
            assert!(
                c["exact_lower_bound"].is_string(),
                "exact rational lower bound present"
            );
        }
    }

    #[test]
    fn certificate_replays_through_selfcheck() {
        // Rebuild the exact certificate the way certify_graph does and assert the
        // public self-checkers accept it — proving the emitted artifact is sound.
        let g = eligible_graph();
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(0.0, f32::INFINITY)],
        )
        .expect("valid spec");
        let stack = extract_fc_relu_stack(&g).expect("eligible");
        let lo = vec![1.0f32];
        let hi = vec![2.0f32];
        let margins = margins_for_bounds(spec.output_bounds(), "Y").expect("margins");
        assert_eq!(margins.len(), 1, "single finite lower-bound conjunct");
        let problem =
            build_problem_for_margin(&stack, &lo, &hi, &margins[0]).expect("build problem");
        let cert = problem.certify(Rat::ZERO).expect("certify");
        assert_eq!(
            (cert.lower_bound.num(), cert.lower_bound.den()),
            (5.into(), 2.into()),
            "exact CROWN lower bound on a fully-active net is 5/2"
        );
        check_entailment(&cert.entailment).expect("entailment self-check");
        check_farkas(&cert.farkas).expect("farkas self-check");
    }

    #[test]
    fn ineligible_net_gets_no_certificate() {
        let g = ineligible_graph();
        // Loose spec so the verdict (whatever it is) is well-defined; the point
        // is that no certificate is emitted regardless.
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        )
        .expect("valid spec");
        let out = certify_graph(&g, &spec).expect("certify_graph runs");
        assert!(!out.eligible, "SiLU net must be ineligible: {}", out.note);
        assert!(
            out.certificate_json.is_none(),
            "ineligible net must not get a certificate"
        );
        // The proof channel must remain empty even if the verdict is Verified.
        if let VerificationResult::Verified { proof, .. } = &out.result {
            assert!(proof.is_none(), "no proof attached for ineligible net");
        }
    }

    #[test]
    fn unverified_eligible_net_gets_no_certificate() {
        let g = eligible_graph();
        // x0 in [1, 2] -> y in [5/2, 9/2]; demand y <= 1 (FALSE) so the verdict is
        // NOT Verified. An eligible-but-unverified net must NOT get a certificate.
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, 1.0)],
        )
        .expect("valid spec");
        let out = certify_graph(&g, &spec).expect("certify_graph runs");
        assert!(out.eligible, "still an FC-ReLU net");
        assert!(
            !out.result.is_verified(),
            "y >= 5/2 cannot satisfy y <= 1: {:?}",
            out.result
        );
        assert!(
            out.certificate_json.is_none(),
            "unverified net must NOT get a certificate"
        );
    }

    #[test]
    fn attach_proof_is_noop_for_non_verified() {
        let proof =
            VerificationProof::from_parts(ProofFormat::BoundTrace, vec![1, 2, 3], None, None);
        let unknown = VerificationResult::Unknown {
            provenance: Default::default(),
            bounds: vec![Bound::new(0.0, 1.0)],
            reason: ny_core::UnknownReason::PotentialViolation,
            actual_method: None,
        };
        let out = attach_proof(unknown, proof);
        assert!(!out.is_verified(), "non-verified result unchanged");
    }

    #[test]
    fn conjunctive_property_emits_per_conjunct_certs() {
        // Spec with BOTH a finite lower and finite upper bound on the single
        // output -> two conjuncts; both must certify for a cert to be emitted.
        let g = eligible_graph();
        // y in [5/2, 9/2]; assert 2 <= y <= 5 (both true with slack).
        let spec = VerificationSpec::new(vec![Bound::new(1.0, 2.0)], vec![Bound::new(2.0, 5.0)])
            .expect("valid spec");
        let out = certify_graph(&g, &spec).expect("certify_graph runs");
        assert!(
            out.result.is_verified(),
            "2 <= y <= 5 holds: {:?}",
            out.result
        );
        let json = out
            .certificate_json
            .unwrap_or_else(|| panic!("expected cert; note: {}", out.note));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed["conjuncts"].as_array().unwrap().len(),
            2,
            "both lower and upper conjuncts discharged"
        );
    }

    // --- Output-constraint coverage (P7) regression suite ---------------------
    //
    // These guard the certificate-overclaim fix: a certificate must cover the
    // FULL property (output_bounds AND output_constraints) or not be emitted.

    use ny_core::ConstraintKind;

    /// y = 2*relu(x0) + 1/2 with x0 in [1,2] -> y in [5/2, 9/2]. Single output.
    /// (Same shape as `eligible_graph`, reused here for constraint specs.)
    fn classifier_1to3() -> GraphNetwork {
        // Hidden: h = relu([x0, x0]) (active over [1,2]) -> h = [x0, x0].
        // Readout (3 classes): y0 = 10*h0, y1 = h1, y2 = 0.5*h1.
        //   over x0 in [1,2]: y0 in [10,20], y1 in [1,2], y2 in [0.5,1].
        // Class 0 strictly dominates -> ArgmaxMargin{0} holds.
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "lin1",
            linear(
                Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap(),
                vec![0.0, 0.0],
            ),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            linear(
                Array2::from_shape_vec((3, 2), vec![10.0, 0.0, 0.0, 1.0, 0.0, 0.5]).unwrap(),
                vec![0.0, 0.0, 0.0],
            ),
            vec!["relu".to_string()],
        ));
        g.set_output("lin2");
        g
    }

    /// A `Linear` output-constraint that the eligible net satisfies must yield a
    /// certificate that COVERS the constraint and replays through the self-check.
    #[test]
    fn linear_constraint_satisfied_is_certified_and_covers_constraint() {
        let g = eligible_graph(); // y0 = 2*x0 + 0.5 in [2.5, 4.5] over x0 in [1,2].
                                  // Constraint 1*y0 <= 5  =>  margin = 5 - y0 in [0.5, 2.5] > 0  (holds).
                                  // output_bounds vacuous so the property is exactly the constraint.
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        )
        .expect("valid spec")
        .with_output_constraints(vec![OutputConstraint::Linear {
            coeffs: vec![1.0],
            bias: 5.0,
            kind: ConstraintKind::Le,
        }])
        .expect("valid constraints");

        let out = certify_graph(&g, &spec).expect("certify_graph runs");
        assert!(out.eligible, "FC-ReLU net is eligible: {}", out.note);
        assert!(
            out.result.is_verified(),
            "1*y0 <= 5 holds: {:?}",
            out.result
        );
        let json = out
            .certificate_json
            .as_ref()
            .unwrap_or_else(|| panic!("expected a certificate; note: {}", out.note));

        let parsed: serde_json::Value = serde_json::from_str(json).expect("cert JSON parses");
        assert_eq!(
            parsed["covers_output_constraints"],
            serde_json::Value::Bool(true),
            "cert must declare it covers the output constraint"
        );
        let conjuncts = parsed["conjuncts"].as_array().expect("conjuncts array");
        // One conjunct, and it is the constraint margin (NOT a bounds conjunct).
        assert_eq!(conjuncts.len(), 1, "single margin conjunct");
        let label = conjuncts[0]["discharged_conjunct"].as_str().unwrap();
        assert!(
            label.contains("constraint[0].margin"),
            "conjunct must be the constraint margin, got '{label}'"
        );
        // Each conjunct carries serialized entailment + Farkas certificates.
        assert!(conjuncts[0]["entailment"].is_object(), "entailment present");
        assert!(conjuncts[0]["farkas"].is_object(), "farkas present");

        // Proof channel populated and equal to the JSON.
        let VerificationResult::Verified { proof, .. } = &out.result else {
            panic!("verified result expected");
        };
        let proof = proof.as_ref().expect("proof channel populated");
        assert_eq!(proof.as_bytes(), json.as_bytes());

        // REPLAY: rebuild the exact certificate for the constraint margin the way
        // certify_graph does and assert the public self-checkers accept it —
        // proving the emitted artifact genuinely covers the linear constraint.
        let c = OutputConstraint::Linear {
            coeffs: vec![1.0],
            bias: 5.0,
            kind: ConstraintKind::Le,
        };
        let stack = extract_fc_relu_stack(&g).expect("eligible");
        let augmented = augment_for_constraint(&g, &c).expect("augment");
        let (m, cc) = margin_layer_from_augmented(&augmented).expect("margin layer");
        let lo = vec![1.0f32];
        let hi = vec![2.0f32];
        let problem = build_problem_for_margin_net_row(&stack, &lo, &hi, &m, &cc, 0, "replay")
            .expect("problem");
        let cert = problem.certify(Rat::ZERO).expect("certify");
        // margin = 5 - y0 = 4.5 - 2*x0; min over x0=2 is 0.5 = 1/2.
        assert_eq!(
            (cert.lower_bound.num(), cert.lower_bound.den()),
            (1.into(), 2.into()),
            "exact margin lower bound is 1/2"
        );
        check_entailment(&cert.entailment).expect("entailment replays");
        check_farkas(&cert.farkas).expect("farkas replays");
    }

    /// An `ArgmaxMargin` spec on a small eligible classifier the net satisfies is
    /// certified per-competitor (or cleanly refused) — assert NO overclaim either
    /// way: if a cert is emitted it covers every competitor margin and replays.
    #[test]
    fn argmax_margin_satisfied_is_certified_per_competitor_or_refused() {
        let g = classifier_1to3();
        // Class 0 (y0 in [10,20]) strictly dominates y1 in [1,2] and y2 in [0.5,1].
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        )
        .expect("valid spec")
        .with_output_constraints(vec![OutputConstraint::ArgmaxMargin { class: 0 }])
        .expect("valid constraints");

        let out = certify_graph(&g, &spec).expect("certify_graph runs");
        assert!(out.eligible, "FC-ReLU classifier is eligible: {}", out.note);
        assert!(
            out.result.is_verified(),
            "class 0 dominates -> argmax holds: {:?}",
            out.result
        );

        match out.certificate_json.as_ref() {
            Some(json) => {
                let parsed: serde_json::Value =
                    serde_json::from_str(json).expect("cert JSON parses");
                assert_eq!(
                    parsed["covers_output_constraints"],
                    serde_json::Value::Bool(true)
                );
                let conjuncts = parsed["conjuncts"].as_array().expect("conjuncts");
                // 3-class argmax -> 2 competitor margins, both discharged.
                assert_eq!(conjuncts.len(), 2, "one margin per competitor class");
                for c in conjuncts {
                    assert!(c["entailment"].is_object(), "entailment present");
                    assert!(c["farkas"].is_object(), "farkas present");
                    let label = c["discharged_conjunct"].as_str().unwrap();
                    assert!(
                        label.contains("constraint[0].margin"),
                        "argmax conjunct labelled, got '{label}'"
                    );
                }
                // REPLAY: rebuild each competitor margin the way certify_graph
                // does and self-check it — proving the argmax property is covered.
                let stack = extract_fc_relu_stack(&g).expect("eligible");
                let augmented =
                    augment_for_constraint(&g, &OutputConstraint::ArgmaxMargin { class: 0 })
                        .expect("augment");
                let (m, cc) = margin_layer_from_augmented(&augmented).expect("margin layer");
                assert_eq!(m.nrows(), 2, "two competitor margins");
                let lo = vec![1.0f32];
                let hi = vec![2.0f32];
                for k in 0..m.nrows() {
                    let problem =
                        build_problem_for_margin_net_row(&stack, &lo, &hi, &m, &cc, k, "replay")
                            .expect("problem");
                    let cert = problem.certify(Rat::ZERO).expect("certify");
                    assert!(
                        cert.lower_bound.is_positive(),
                        "competitor {k} margin lower bound must be > 0"
                    );
                    check_entailment(&cert.entailment).expect("entailment replays");
                    check_farkas(&cert.farkas).expect("farkas replays");
                }
                // Proof channel populated.
                let VerificationResult::Verified { proof, .. } = &out.result else {
                    panic!("verified");
                };
                assert!(proof.is_some(), "proof attached when cert emitted");
            }
            None => {
                // Clean refusal is also sound: assert NO overclaim — no proof.
                if let VerificationResult::Verified { proof, .. } = &out.result {
                    assert!(proof.is_none(), "no proof attached when cert refused");
                }
            }
        }
    }

    /// Regression guard against the overclaim: a spec WITH an output_constraint
    /// that CANNOT be eligibly certified (ineligible augmented architecture)
    /// must NOT emit a certificate.
    #[test]
    fn uncertifiable_constraint_yields_no_certificate() {
        // SiLU net is ineligible; a Linear output_constraint must not sneak a
        // certificate past the architecture gate.
        let g = ineligible_graph();
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        )
        .expect("valid spec")
        .with_output_constraints(vec![OutputConstraint::Linear {
            coeffs: vec![1.0],
            bias: 100.0,
            kind: ConstraintKind::Le,
        }])
        .expect("valid constraints");

        let out = certify_graph(&g, &spec).expect("certify_graph runs");
        assert!(!out.eligible, "SiLU net must be ineligible: {}", out.note);
        assert!(
            out.certificate_json.is_none(),
            "ineligible net with a constraint must NOT get a certificate"
        );
        if let VerificationResult::Verified { proof, .. } = &out.result {
            assert!(proof.is_none(), "no proof attached for ineligible net");
        }
    }

    /// The output_bounds hold but an output_constraint is VIOLATED: no cert AND
    /// the verdict is NOT Verified — cert and verdict agree on the SAME property.
    #[test]
    fn violated_constraint_blocks_cert_and_verdict() {
        let g = eligible_graph(); // y0 in [2.5, 4.5].
                                  // output_bounds: y0 >= 0 (HOLDS).  constraint: 1*y0 <= 3 (Le) is
                                  // VIOLATED (margin = 3 - y0 in [-1.5, 0.5], not strictly positive).
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(0.0, f32::INFINITY)],
        )
        .expect("valid spec")
        .with_output_constraints(vec![OutputConstraint::Linear {
            coeffs: vec![1.0],
            bias: 3.0,
            kind: ConstraintKind::Le,
        }])
        .expect("valid constraints");

        let out = certify_graph(&g, &spec).expect("certify_graph runs");
        assert!(out.eligible, "still an FC-ReLU net: {}", out.note);
        assert!(
            !out.result.is_verified(),
            "violated constraint must make the verdict NOT Verified: {:?}",
            out.result
        );
        assert!(
            out.certificate_json.is_none(),
            "no certificate while a requested constraint is uncertified"
        );
        if let VerificationResult::Verified { proof, .. } = &out.result {
            assert!(proof.is_none(), "no proof attached");
        }
    }

    /// A satisfied output_bounds conjunct AND a satisfied Linear constraint both
    /// appear in the emitted certificate (covers BOTH, not just bounds).
    #[test]
    fn bounds_and_constraint_both_covered() {
        let g = eligible_graph(); // y0 in [2.5, 4.5].
                                  // output_bounds: 2 <= y0 (lower only, HOLDS).  constraint: y0 <= 5 HOLDS.
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(2.0, f32::INFINITY)],
        )
        .expect("valid spec")
        .with_output_constraints(vec![OutputConstraint::Linear {
            coeffs: vec![1.0],
            bias: 5.0,
            kind: ConstraintKind::Le,
        }])
        .expect("valid constraints");

        let out = certify_graph(&g, &spec).expect("certify_graph runs");
        assert!(out.result.is_verified(), "both hold: {:?}", out.result);
        let json = out
            .certificate_json
            .unwrap_or_else(|| panic!("expected cert; note: {}", out.note));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let conjuncts = parsed["conjuncts"].as_array().unwrap();
        assert_eq!(
            conjuncts.len(),
            2,
            "one bounds conjunct + one constraint margin"
        );
        let labels: Vec<&str> = conjuncts
            .iter()
            .map(|c| c["discharged_conjunct"].as_str().unwrap())
            .collect();
        assert!(
            labels.iter().any(|l| l.starts_with("Y_0 >=")),
            "bounds conjunct present: {labels:?}"
        );
        assert!(
            labels.iter().any(|l| l.contains("constraint[0].margin")),
            "constraint margin present: {labels:?}"
        );
    }

    /// A 1->2->2 classifier where, over x0 in [1,2], class 0 STRICTLY dominates
    /// class 1 by a positive margin: y0 = 2*h, y1 = h with h = relu(x0) = x0 in
    /// [1,2] -> margin m = y0 - y1 = h in [1,2] (lower bound 1 > 0). Used to assert
    /// the strict ArgmaxMargin cert reflects strictness.
    fn classifier_strict_1to2() -> GraphNetwork {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "lin1",
            linear(
                Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
                vec![0.0],
            ),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            linear(
                Array2::from_shape_vec((2, 1), vec![2.0, 1.0]).unwrap(),
                vec![0.0, 0.0],
            ),
            vec!["relu".to_string()],
        ));
        g.set_output("lin2");
        g
    }

    /// A 1->2->2 classifier whose two class outputs are IDENTICAL: y0 = y1 = h
    /// with h = relu(x0). The argmax margin m = y0 - y1 == 0 EVERYWHERE, so the
    /// exact lower bound is exactly 0 (a tie) — strict argmax does NOT hold.
    fn classifier_tie_1to2() -> GraphNetwork {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "lin1",
            linear(
                Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
                vec![0.0],
            ),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            linear(
                Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap(),
                vec![0.0, 0.0],
            ),
            vec!["relu".to_string()],
        ));
        g.set_output("lin2");
        g
    }

    /// STRICTNESS: an emitted ArgmaxMargin certificate must (a) only be emitted
    /// when every competitor margin's exact lower bound is STRICTLY positive, and
    /// (b) label/relate each margin conjunct as the STRICT relation `> 0`.
    #[test]
    fn argmax_cert_attests_strict_relation() {
        let g = classifier_strict_1to2(); // class 0 dominates class 1 by margin in [1,2].
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        )
        .expect("valid spec")
        .with_output_constraints(vec![OutputConstraint::ArgmaxMargin { class: 0 }])
        .expect("valid constraints");

        let out = certify_graph(&g, &spec).expect("certify_graph runs");
        assert!(
            out.result.is_verified(),
            "strict argmax holds: {:?}",
            out.result
        );
        let json = out
            .certificate_json
            .unwrap_or_else(|| panic!("expected strict argmax cert; note: {}", out.note));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let conjuncts = parsed["conjuncts"].as_array().unwrap();
        assert_eq!(conjuncts.len(), 1, "single competitor margin (2-class)");
        let c = &conjuncts[0];
        // The conjunct must honestly attest the STRICT relation and strictness flag.
        assert_eq!(
            c["relation"],
            serde_json::Value::String("margin > 0".into())
        );
        assert_eq!(c["strict"], serde_json::Value::Bool(true));
        let label = c["discharged_conjunct"].as_str().unwrap();
        assert!(
            label.contains("> 0") && label.contains("exact_lower_bound L > 0"),
            "strict label states `> 0` justified by L > 0, got '{label}'"
        );
        // And the attested exact lower bound is itself strictly positive.
        let lb = c["exact_lower_bound"].as_str().unwrap();
        let (n, d) = lb.split_once('/').expect("v1 exact bound uses n/d");
        let num: i128 = n.parse().unwrap();
        let den: i128 = d.parse().unwrap();
        assert!(
            num.signum() * den.signum() > 0,
            "exact lower bound L > 0, got {lb}"
        );
    }

    /// TIE / NO-OVERCLAIM: a borderline argmax case whose exact margin lower
    /// bound is exactly 0 must yield certificate_json == None. A kernel checking
    /// only `>= 0` would accept the tie, so the cert is fail-closed refused.
    #[test]
    fn argmax_tie_yields_no_certificate() {
        let g = classifier_tie_1to2(); // y0 == y1 -> margin == 0 everywhere (a tie).
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        )
        .expect("valid spec")
        .with_output_constraints(vec![OutputConstraint::ArgmaxMargin { class: 0 }])
        .expect("valid constraints");

        let out = certify_graph(&g, &spec).expect("certify_graph runs");
        assert!(out.eligible, "FC-ReLU classifier is eligible: {}", out.note);
        // Whatever the runtime verdict, NO strict-argmax certificate may be emitted
        // on a tie (margin == 0 does not establish strict argmax).
        assert!(
            out.certificate_json.is_none(),
            "tie (margin == 0) must NOT yield an argmax certificate (no overclaim)"
        );
        if let VerificationResult::Verified { proof, .. } = &out.result {
            assert!(proof.is_none(), "no proof attached on a tie");
        }
    }

    /// UNIT: the strictness gate lives in `certify_problem_into`. On a problem
    /// whose exact lower bound is exactly 0, `strict == true` refuses (Err) while
    /// `strict == false` accepts (the non-strict `>= 0` conjunct is fine).
    #[test]
    fn strict_gate_refuses_zero_lower_bound_unit() {
        let g = classifier_tie_1to2();
        let stack = extract_fc_relu_stack(&g).expect("eligible");
        let augmented = augment_for_constraint(&g, &OutputConstraint::ArgmaxMargin { class: 0 })
            .expect("augment");
        let (m, cc) = margin_layer_from_augmented(&augmented).expect("margin layer");
        assert_eq!(m.nrows(), 1, "single competitor margin");
        let lo = vec![1.0f32];
        let hi = vec![2.0f32];
        let problem =
            build_problem_for_margin_net_row(&stack, &lo, &hi, &m, &cc, 0, "tie").expect("problem");
        // The exact lower bound of m = y0 - y1 is exactly 0.
        let cert = problem.certify(Rat::ZERO).expect("certify");
        assert!(
            cert.lower_bound.is_zero(),
            "tie margin lower bound is exactly 0"
        );

        // strict=false: a `>= 0` conjunct is correct and accepted.
        let mut entries = Vec::new();
        certify_problem_into(&problem, "tie >= 0", false, None, &mut entries, &mut None)
            .expect("non-strict conjunct accepted at L == 0");
        assert_eq!(entries.len(), 1, "non-strict conjunct recorded");
        assert_eq!(entries[0]["strict"], serde_json::Value::Bool(false));

        // strict=true: a tie does NOT discharge `> 0` -> refuse (fail-closed).
        let mut entries = Vec::new();
        let err = certify_problem_into(&problem, "tie > 0", true, None, &mut entries, &mut None)
            .expect_err("strict conjunct must be refused at L == 0");
        assert!(
            err.contains("not > 0"),
            "fail-closed reason mentions strictness: {err}"
        );
        assert!(
            entries.is_empty(),
            "no conjunct recorded when strict gate refuses"
        );
    }

    /// `OutputConstraint::Bounds` is treated as a per-output interval property
    /// over the original net and certified alongside (covers the constraint).
    #[test]
    fn bounds_output_constraint_is_certified() {
        let g = eligible_graph(); // y0 in [2.5, 4.5].
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        )
        .expect("valid spec")
        .with_output_constraints(vec![OutputConstraint::Bounds(vec![Bound::new(2.0, 5.0)])])
        .expect("valid constraints");

        let out = certify_graph(&g, &spec).expect("certify_graph runs");
        assert!(
            out.result.is_verified(),
            "2 <= y0 <= 5 holds: {:?}",
            out.result
        );
        let json = out
            .certificate_json
            .unwrap_or_else(|| panic!("expected cert; note: {}", out.note));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let conjuncts = parsed["conjuncts"].as_array().unwrap();
        // Constraint Bounds(2..5) -> two interval conjuncts (lower & upper).
        assert_eq!(
            conjuncts.len(),
            2,
            "lower and upper of the Bounds constraint"
        );
        for c in conjuncts {
            let label = c["discharged_conjunct"].as_str().unwrap();
            assert!(
                label.contains("constraint[0].Y"),
                "Bounds-constraint conjunct labelled, got '{label}'"
            );
        }
    }

    // --- certify_graph_with_alethe (Alethe emission threading) ----------------

    /// E2E: an eligible, verified net emits BOTH the exact-rational certificate
    /// and a non-empty Alethe pair per conjunct, with labels matching the
    /// certificate one-to-one, and the certification outcome identical to
    /// [`certify_graph`]'s contract.
    #[test]
    fn certify_graph_with_alethe_emits_checkable_pairs() {
        let g = eligible_graph();
        // Two conjuncts (finite lower AND upper) so per-conjunct order matters.
        let spec = VerificationSpec::new(vec![Bound::new(1.0, 2.0)], vec![Bound::new(2.0, 5.0)])
            .expect("valid spec");

        let out = certify_graph_with_alethe(&g, &spec).expect("runs");
        assert!(out.certified.eligible, "eligible: {}", out.certified.note);
        assert!(
            out.certified.result.is_verified(),
            "2 <= y <= 5 holds: {:?}",
            out.certified.result
        );
        let json = out
            .certified
            .certificate_json
            .as_ref()
            .unwrap_or_else(|| panic!("expected certificate; note: {}", out.certified.note));

        let alethe = out
            .alethe
            .as_ref()
            .unwrap_or_else(|| panic!("expected Alethe set; note: {}", out.alethe_note));
        let parsed: serde_json::Value = serde_json::from_str(json).expect("cert JSON parses");
        let conjuncts = parsed["conjuncts"].as_array().expect("conjuncts");
        assert_eq!(
            alethe.len(),
            conjuncts.len(),
            "one Alethe pair per certificate conjunct (whole-or-nothing)"
        );
        for (pair, entry) in alethe.iter().zip(conjuncts) {
            assert_eq!(
                pair.label,
                entry["discharged_conjunct"].as_str().unwrap(),
                "Alethe pair labelled with its certificate conjunct"
            );
            // Non-empty, recognisably SMT-LIB / Alethe text for BOTH artifacts.
            for emission in [&pair.entailment, &pair.farkas] {
                assert!(
                    emission.problem.contains("assert"),
                    "SMT-LIB problem has asserts: {}",
                    emission.problem
                );
                assert!(
                    emission.proof.contains("la_generic"),
                    "Alethe proof has the la_generic step: {}",
                    emission.proof
                );
                assert!(
                    emission.proof.contains("(cl)"),
                    "Alethe proof derives the empty clause: {}",
                    emission.proof
                );
            }
        }

        // The certification outcome must match the plain entry point's.
        let plain = certify_graph(&g, &spec).expect("plain runs");
        assert_eq!(
            plain.certificate_json.as_deref(),
            Some(json.as_str()),
            "Alethe threading must not perturb the certificate build"
        );
    }

    /// NO OVERCLAIM: when no certificate is emitted (ineligible net), no Alethe
    /// artifact may be returned either.
    #[test]
    fn certify_graph_with_alethe_none_without_certificate() {
        let g = ineligible_graph();
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        )
        .expect("valid spec");
        let out = certify_graph_with_alethe(&g, &spec).expect("runs");
        assert!(!out.certified.eligible, "SiLU net ineligible");
        assert!(out.certified.certificate_json.is_none(), "no certificate");
        assert!(
            out.alethe.is_none(),
            "no certificate -> no Alethe artifact: {}",
            out.alethe_note
        );
    }

    /// NO OVERCLAIM: an eligible net whose verdict is NOT Verified gets neither
    /// a certificate nor an Alethe artifact.
    #[test]
    fn certify_graph_with_alethe_none_when_unverified() {
        let g = eligible_graph();
        // y in [5/2, 9/2]; demand y <= 1 (FALSE).
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, 1.0)],
        )
        .expect("valid spec");
        let out = certify_graph_with_alethe(&g, &spec).expect("runs");
        assert!(out.certified.eligible, "still FC-ReLU");
        assert!(!out.certified.result.is_verified(), "property is false");
        assert!(out.certified.certificate_json.is_none(), "no certificate");
        assert!(out.alethe.is_none(), "no Alethe artifact");
    }

    /// STRICT conjuncts: an ArgmaxMargin certificate's Alethe set covers every
    /// competitor margin, and is emitted only alongside the (already strictly
    /// discharged) certificate.
    #[test]
    fn certify_graph_with_alethe_covers_strict_argmax() {
        let g = classifier_strict_1to2();
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        )
        .expect("valid spec")
        .with_output_constraints(vec![OutputConstraint::ArgmaxMargin { class: 0 }])
        .expect("valid constraints");

        let out = certify_graph_with_alethe(&g, &spec).expect("runs");
        assert!(
            out.certified.result.is_verified(),
            "strict argmax holds: {:?}",
            out.certified.result
        );
        assert!(
            out.certified.certificate_json.is_some(),
            "strict cert emitted; note: {}",
            out.certified.note
        );
        let alethe = out
            .alethe
            .as_ref()
            .unwrap_or_else(|| panic!("expected Alethe set; note: {}", out.alethe_note));
        assert_eq!(alethe.len(), 1, "one competitor margin (2-class)");
        assert!(
            alethe[0].label.contains("constraint[0].margin"),
            "strict margin conjunct labelled: {}",
            alethe[0].label
        );
        assert!(!alethe[0].entailment.proof.is_empty(), "non-empty proof");
        assert!(!alethe[0].farkas.proof.is_empty(), "non-empty proof");
    }

    /// The facade's re-exported Alethe emitters work standalone on a certified
    /// problem's certificates (the direct, non-threaded path).
    #[test]
    fn standalone_alethe_emitters_accept_selfchecked_certificates() {
        let g = eligible_graph();
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(0.0, f32::INFINITY)],
        )
        .expect("valid spec");
        let stack = extract_fc_relu_stack(&g).expect("eligible");
        let margins = margins_for_bounds(spec.output_bounds(), "Y").expect("margins");
        let problem =
            build_problem_for_margin(&stack, &[1.0], &[2.0], &margins[0]).expect("build problem");
        let cert = problem.certify(Rat::ZERO).expect("certify");

        let ent = entailment_to_alethe(&cert.entailment).expect("entailment emits");
        assert!(ent.problem.contains("set-logic"), "SMT-LIB header present");
        assert!(ent.proof.contains("la_generic"), "la_generic step present");
        let far = farkas_to_alethe(&cert.farkas).expect("farkas emits");
        assert!(far.problem.contains("assert"), "asserts present");
        assert!(far.proof.contains("(cl)"), "empty clause derived");
    }

    // ---------------------------------------------------------------------
    // Deployed-precision policy (fp_margin fold into the strict gate).
    // ---------------------------------------------------------------------

    /// A 1->2->2 classifier whose argmax margin is real but TINY: y0 =
    /// (1 + 2^-23)*h, y1 = h with h = relu(x0) in [1, 2], so the exact margin
    /// lower bound is L = 2^-23 ~= 1.19e-7. That clears `L > 0` comfortably in
    /// real semantics but sits an order of magnitude BELOW the deployed f32
    /// rounding error of the two logits, so the deployed comparison could tie
    /// or invert. `1 + 2^-23` is the float immediately above 1.0, hence an
    /// exact f32 (and an exact rational lift).
    fn classifier_hairline_1to2() -> GraphNetwork {
        let epsilon = f32::EPSILON; // exactly 2^-23: the gap above 1.0
        assert_eq!(epsilon, 2.0_f32.powi(-23), "hairline epsilon is 2^-23");
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "lin1",
            linear(
                Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
                vec![0.0],
            ),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            linear(
                Array2::from_shape_vec((2, 1), vec![1.0 + epsilon, 1.0]).unwrap(),
                vec![0.0, 0.0],
            ),
            vec!["relu".to_string()],
        ));
        g.set_output("lin2");
        g
    }

    fn argmax_spec_over_1_to_2() -> VerificationSpec {
        VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        )
        .expect("valid spec")
        .with_output_constraints(vec![OutputConstraint::ArgmaxMargin { class: 0 }])
        .expect("valid constraints")
    }

    /// BIT-STABILITY PIN: the default policy must not perturb a single byte of
    /// an emitted certificate, and must not introduce either policy field.
    /// `certify_graph` and `certify_graph_with_policy(.., default)` are the same
    /// emitter.
    #[test]
    fn default_policy_certificates_are_bit_stable() {
        let cases: Vec<(GraphNetwork, VerificationSpec)> = vec![
            (
                eligible_graph(),
                VerificationSpec::new(
                    vec![Bound::new(1.0, 2.0)],
                    vec![Bound::new_allow_infinite(0.0, f32::INFINITY)],
                )
                .expect("valid spec"),
            ),
            (classifier_strict_1to2(), argmax_spec_over_1_to_2()),
        ];

        for (net, spec) in cases {
            let legacy = certify_graph(&net, &spec).expect("certify_graph runs");
            let explicit = certify_graph_with_policy(&net, &spec, CertifyPolicy::default())
                .expect("policy entry runs");
            let json = legacy
                .certificate_json
                .as_ref()
                .unwrap_or_else(|| panic!("expected a certificate; note: {}", legacy.note));

            assert_eq!(
                legacy.certificate_json, explicit.certificate_json,
                "the default policy must emit byte-identical certificate JSON"
            );
            assert_eq!(
                legacy.note, explicit.note,
                "the default policy must not change the outcome note"
            );
            assert_eq!(legacy.eligible, explicit.eligible);
            assert!(
                !json.contains("precision_policy"),
                "default certificates carry no precision_policy field: {json}"
            );
            assert!(
                !json.contains("deployed_precision"),
                "default certificates carry no deployed_precision field: {json}"
            );
            // And the default policy really is RealSemantics.
            assert!(
                !CertifyPolicy::default().folds_deployed_fp_margin(),
                "the deployed fold must be OFF by default"
            );
        }
    }

    /// A comfortable strict margin (L = 1) survives the deployed fold, and the
    /// discharged conjunct records the folded threshold it cleared.
    #[test]
    fn deployed_policy_records_the_folded_threshold_it_cleared() {
        let g = classifier_strict_1to2(); // margin = h in [1, 2] -> L = 1.
        let spec = argmax_spec_over_1_to_2();

        let out = certify_graph_with_policy(&g, &spec, CertifyPolicy::deployed_f32())
            .expect("certify_graph_with_policy runs");
        let json = out
            .certificate_json
            .as_ref()
            .unwrap_or_else(|| panic!("L = 1 must clear the fp margin; note: {}", out.note));
        assert!(
            out.note.contains("deployed-f32"),
            "the outcome note must state the active policy: {}",
            out.note
        );

        let parsed: serde_json::Value = serde_json::from_str(json).expect("certificate parses");
        assert_eq!(
            parsed["precision_policy"]["mode"],
            serde_json::Value::String("deployed-f32".into()),
            "top-level policy recorded"
        );
        let scope = parsed["precision_policy"]["scope"]
            .as_str()
            .expect("scope recorded");
        assert!(
            scope.contains("strict") && scope.contains("Non-strict"),
            "the recorded scope must state exactly which conjuncts are covered: {scope}"
        );

        let conjuncts = parsed["conjuncts"].as_array().expect("conjuncts");
        assert_eq!(conjuncts.len(), 1, "single competitor margin (2-class)");
        let c = &conjuncts[0];
        assert_eq!(c["strict"], serde_json::Value::Bool(true));
        let deployed = &c["deployed_precision"];
        assert_eq!(
            deployed["policy"],
            serde_json::Value::String("deployed-f32".into())
        );
        let fp_margin = deployed["fp_margin"].as_str().expect("fp_margin recorded");
        assert_eq!(
            deployed["folded_threshold"].as_str(),
            Some(fp_margin),
            "the folded threshold IS the fp margin"
        );

        // The recorded threshold must be exactly the fold this net's deployed
        // read-out rows produce, strictly positive, and strictly below the
        // attested exact lower bound (L = 1 for this net).
        let stack = extract_fc_relu_stack(&g).expect("eligible");
        let augmented = augment_for_constraint(&g, &OutputConstraint::ArgmaxMargin { class: 0 })
            .expect("augment");
        let (m, _c) = margin_layer_from_augmented(&augmented).expect("margin layer");
        let mut cache = DeployedErrorCache::new();
        let expected = deployed_margin_threshold(&stack, &[1.0], &[2.0], &m, 0, "row0", &mut cache)
            .expect("fold recomputes");
        assert_eq!(
            fp_margin,
            expected.to_clean_string().expect("threshold prints"),
            "the recorded threshold must be the exact fold, verbatim"
        );
        assert!(expected.is_positive(), "the fp margin is strictly positive");
        assert!(
            expected < Rat::ONE,
            "the fp margin must be strictly below L = 1 for this net"
        );
        assert_eq!(
            c["exact_lower_bound"].as_str(),
            Some("1/1"),
            "the attested exact lower bound for this net is 1"
        );
    }

    /// FAIL-CLOSED: a strict margin that is real but smaller than the deployed
    /// rounding error is discharged under the default policy and REFUSED under
    /// the deployed policy — the differential is the whole point of the fold.
    #[test]
    fn deployed_policy_refuses_a_margin_below_the_rounding_error() {
        let g = classifier_hairline_1to2(); // L = 2^-23, delta ~ 8x larger.
        let spec = argmax_spec_over_1_to_2();

        let real = certify_graph(&g, &spec).expect("certify_graph runs");
        assert!(
            real.certificate_json.is_some(),
            "the real-semantics gate (L > 0) discharges a 2^-23 margin; note: {}",
            real.note
        );

        let deployed = certify_graph_with_policy(&g, &spec, CertifyPolicy::deployed_f32())
            .expect("certify_graph_with_policy runs");
        assert!(
            deployed.certificate_json.is_none(),
            "a margin below the deployed f32 rounding error must NOT be certified: {}",
            deployed.note
        );
        assert!(
            deployed.note.contains("deployed rounding margin"),
            "the refusal must name the deployed margin: {}",
            deployed.note
        );
        // The VERDICT is never touched by a certificate refusal.
        assert_eq!(
            real.result.is_verified(),
            deployed.result.is_verified(),
            "the fold refuses the certificate, never the verdict"
        );
        if let VerificationResult::Verified { proof, .. } = &deployed.result {
            assert!(proof.is_none(), "no proof attached on a refused fold");
        }
    }

    /// SCOPE: the fold is a STRICT-gate policy. A property with only non-strict
    /// (interval) conjuncts emits conjunct entries that are byte-identical to
    /// the default policy's; only the top-level scope note is added.
    #[test]
    fn deployed_policy_leaves_non_strict_conjuncts_untouched() {
        let g = eligible_graph();
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(0.0, f32::INFINITY)],
        )
        .expect("valid spec");

        let default_json = certify_graph(&g, &spec)
            .expect("certify_graph runs")
            .certificate_json
            .expect("certificate emitted");
        let deployed_json = certify_graph_with_policy(&g, &spec, CertifyPolicy::deployed_f32())
            .expect("policy entry runs")
            .certificate_json
            .expect("certificate emitted");

        let default_parsed: serde_json::Value = serde_json::from_str(&default_json).unwrap();
        let deployed_parsed: serde_json::Value = serde_json::from_str(&deployed_json).unwrap();
        assert_eq!(
            default_parsed["conjuncts"], deployed_parsed["conjuncts"],
            "non-strict conjuncts must be identical under both policies"
        );
        assert!(
            default_parsed.get("precision_policy").is_none(),
            "no policy object under the default policy"
        );
        assert!(
            deployed_parsed.get("precision_policy").is_some(),
            "the active policy is recorded even when it changes no conjunct"
        );
    }

    /// The folded threshold is the sum of the DEPLOYED read-out rows' errors,
    /// NOT the error of the fused margin row — and the fused row's error is
    /// strictly smaller, which is exactly why folding on it would understate
    /// the deployed error of a network that rounds each logit before comparing.
    #[test]
    fn folded_threshold_sums_deployed_row_errors_not_the_fused_row() {
        let g = classifier_strict_1to2(); // y0 = 2h, y1 = h.
        let stack = extract_fc_relu_stack(&g).expect("eligible");
        let augmented = augment_for_constraint(&g, &OutputConstraint::ArgmaxMargin { class: 0 })
            .expect("augment");
        let (m, c) = margin_layer_from_augmented(&augmented).expect("margin layer");
        let lo = vec![1.0f32];
        let hi = vec![2.0f32];

        let mut cache = DeployedErrorCache::new();
        let threshold =
            deployed_margin_threshold(&stack, &lo, &hi, &m, 0, "row0", &mut cache).expect("fold");

        // Independent hand-built deployed read-out rows: y0 = 2*h and y1 = 1*h
        // over the SAME hidden stack (w = 1, b = 0) and box.
        let deployed_row = |w: Rat| DeepReluProblem {
            weights: vec![vec![vec![Rat::ONE]]],
            biases: vec![vec![Rat::ZERO]],
            out_weight: vec![w],
            out_bias: Rat::ZERO,
            input_lower: vec![Rat::from_int(1)],
            input_upper: vec![Rat::from_int(2)],
            alpha: None,
            interm_round: false,
        };
        let e0 = deployed_fp_output_margin(&deployed_row(Rat::from_int(2))).expect("e0");
        let e1 = deployed_fp_output_margin(&deployed_row(Rat::ONE)).expect("e1");
        let expected = e0.add(e1).expect("e0 + e1");
        assert_eq!(
            threshold, expected,
            "the folded threshold must be |M[k][0]|*e0 + |M[k][1]|*e1 exactly"
        );

        // The FUSED margin row (out_weight = 2 - 1 = 1) has a strictly smaller
        // error bound — the underestimate the per-row fold deliberately avoids.
        let fused = build_problem_for_margin_net_row(&stack, &lo, &hi, &m, &c, 0, "fused")
            .expect("fused margin problem");
        let fused_error = deployed_fp_output_margin(&fused).expect("fused row error");
        assert!(
            fused_error < threshold,
            "folding on the fused margin row would understate the deployed error"
        );

        // And the cache really memoizes: two outputs, two entries.
        assert_eq!(cache.len(), 2, "one cached error per deployed read-out row");
    }

    /// The Alethe surface composes with the policy: a strict conjunct that
    /// cleared the deployed fold still emits a checkable pair.
    #[test]
    fn alethe_entry_composes_with_the_deployed_policy() {
        let out = certify_graph_with_alethe_and_policy(
            &classifier_strict_1to2(),
            &argmax_spec_over_1_to_2(),
            CertifyPolicy::deployed_f32(),
        )
        .expect("alethe policy entry runs");
        let json = out
            .certified
            .certificate_json
            .as_ref()
            .expect("certificate emitted under the deployed policy");
        assert!(json.contains("deployed_precision"), "fold recorded");
        let alethe = out
            .alethe
            .as_ref()
            .unwrap_or_else(|| panic!("expected Alethe pairs; note: {}", out.alethe_note));
        assert_eq!(alethe.len(), 1, "one pair per certificate conjunct");
        assert!(!alethe[0].entailment.proof.is_empty());
    }
}
