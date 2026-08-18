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
//! and the JSON schema emitters used to serialize certificates for Clean.
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
/// Shallow (one ReLU hidden layer) CROWN problem and certified result.
pub use ny_cert::{CertifiedRelu1, CrownError, Relu1Problem};
/// Exact-rational scalar arithmetic underlying every certificate.
pub use ny_cert::{Rat, RatError};

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
fn certify_problem_into(
    problem: &DeepReluProblem,
    label: &str,
    strict: bool,
    entries: &mut Vec<serde_json::Value>,
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
    if strict && !certified.lower_bound.is_positive() {
        return Err(format!(
            "strict conjunct '{label}' not discharged: exact lower bound {lower_bound} is not > 0 \
             (a tie does not establish strict argmax)"
        ));
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
    // Be honest about what L attests: a strict conjunct proves `margin > 0`
    // (justified by the just-checked `L > 0`); a non-strict one proves
    // `margin >= 0`. A Clean kernel re-checking only `>= 0` together with the
    // attested `L > 0` can re-derive the strict relation itself.
    let relation = if strict { "margin > 0" } else { "margin >= 0" };
    entries.push(serde_json::json!({
        "discharged_conjunct": label,
        "relation": relation,
        "strict": strict,
        "exact_lower_bound": lower_bound,
        "depth": problem.depth(),
        "entailment": entailment_json,
        "farkas": farkas_json,
    }));
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
) -> std::result::Result<(), String> {
    for margin in margins {
        let problem = build_problem_for_margin(stack, lo, hi, margin)?;
        // Interval/bounds margins are NON-strict (`margin >= 0` is exactly the
        // property); `L >= 0` discharges them.
        certify_problem_into(&problem, &margin.label, false, entries)?;
    }
    Ok(())
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
fn try_build_certificate(
    stack: &FcReluStack<'_>,
    net: &GraphNetwork,
    spec: &VerificationSpec,
) -> std::result::Result<String, String> {
    let lo: Vec<f32> = spec.input_bounds().iter().map(Bound::lower).collect();
    let hi: Vec<f32> = spec.input_bounds().iter().map(Bound::upper).collect();

    let mut entries: Vec<serde_json::Value> = Vec::new();

    // (a) Legacy output-bounds conjuncts over the original stack. An empty result
    // is fine ONLY because the constraints below may carry the actual conjuncts;
    // we enforce "at least one conjunct overall" after collecting everything.
    let bound_margins = margins_for_bounds(spec.output_bounds(), "Y")?;
    certify_margins_into(stack, &lo, &hi, &bound_margins, &mut entries)?;

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
                certify_margins_into(stack, &lo, &hi, &margins, &mut entries)?;
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
                    let problem =
                        build_problem_for_margin_net_row(stack, &lo, &hi, &m, &c, k, &label)?;
                    certify_problem_into(&problem, &label, strict, &mut entries)?;
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
    let payload = serde_json::json!({
        "format": "ny-cert/crown-deep/v1",
        "claim": claim,
        "covers_output_bounds": true,
        "covers_output_constraints": has_constraints,
        "conjuncts": entries,
    });
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

    match try_build_certificate(&stack, net, spec) {
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
            Ok(CertifiedResult {
                result,
                certificate_json: Some(json),
                eligible: true,
                note: format!(
                    "exact-rational certificate covering {scope} emitted and attached to the \
                     proof channel"
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
        certify_problem_into(&problem, "tie >= 0", false, &mut entries)
            .expect("non-strict conjunct accepted at L == 0");
        assert_eq!(entries.len(), 1, "non-strict conjunct recorded");
        assert_eq!(entries[0]["strict"], serde_json::Value::Bool(false));

        // strict=true: a tie does NOT discharge `> 0` -> refuse (fail-closed).
        let mut entries = Vec::new();
        let err = certify_problem_into(&problem, "tie > 0", true, &mut entries)
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
}
