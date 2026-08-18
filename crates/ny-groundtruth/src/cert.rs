// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact-rational **dominance certificates** for ground-truth verification —
//! the M3 certification half of `docs/GEOMETRIC_GROUND_TRUTH_PLAN.md` §4.
//!
//! [`certify_dominance`] turns a *Verified* `f(x) ≥ g(x)` verdict into a
//! machine-checkable proof artifact: an exact-rational entailment + Farkas
//! certificate (Clean's external-certificate JSON, via
//! [`ny_cert::crown_deep::DeepReluProblem::certify_difference_linear`]) that
//! is self-checked by the **unchanged** `ny-cert` checkers before it is
//! emitted. The consumer is a3d's `.a3dcert` evidence bundle.
//!
//! # Eligibility — what is certified today, and the documented gap
//!
//! `ny-cert` certifies sequential FC-ReLU networks against conjunctive linear
//! output properties. This module extends that eligibility to **difference
//! networks `h = f − g`** whose subtracted side `g` is a ground-truth graph in
//! the M1 builders' layer set (`Linear` / `PowConstant` / `Add` / `Sub` over
//! exact constants):
//!
//! * **Certified: pure-`Linear` g — the PLANE builder** (including poses,
//!   which are one more `Linear`). `h` is then `f`'s FC-ReLU stack with the
//!   affine functional `g_coeffs·x + g_offset` subtracted from its scalar
//!   read-out, and the g-side contributes **exact rational rows** to the
//!   certificate: the read-out premise pair gains `+gᵢ·xᵢ` terms whose
//!   coefficients are the plane's exactly-representable constants. Every
//!   premise is a plain linear constraint; the certificate is validated by the
//!   same `check_entailment`/`check_farkas` Farkas-combination check and rests
//!   on the same kernel-checked `farkas_premise_combination` theorem —
//!   provably within the existing certificate theorem, no new checker surface.
//!
//! * **Certified: QUADRATIC g — the sphere / cylinder / cone builders**
//!   (single-level `PowConstant(2)` over affine pre-squares, combined by
//!   `Linear`/`Add`/`Sub`). The graph is folded exactly to
//!   `g(x) = g_lin·x + g₀ + Σⱼ qⱼ·tⱼ(x)²` with affine `tⱼ`, and the
//!   certificate ([`ny_cert::crown_deep::DeepReluProblem::certify_difference_quadratic`])
//!   introduces per square the fresh variables `tⱼ`/`sⱼ` and the *quadratic
//!   envelope* premises — the tangent `sⱼ ≥ 2c·tⱼ − c²` and the secant
//!   `sⱼ ≤ (l+u)·tⱼ − l·u` on `tⱼ ∈ [l, u]`. Their validity is a nonlinear
//!   fact about the square function, and it is **grounded in the
//!   kernel-checked corpus theorems `pow2_tangent` / `pow2_secant`** in the
//!   exact Lake-pinned Clean module `Crownproof.Pow2Envelope`
//!   (`#print axioms` = the standard three). The checker is UNCHANGED: every
//!   premise is still a plain linear constraint combined with non-negative
//!   multipliers under `farkas_premise_combination`.
//!
//! * **Recognized but NOT yet certifiable: nested squares — the TORUS
//!   builder** (`PowConstant(2)` applied to a value that is itself already
//!   quadratic; the residual is quartic). The pow2 envelope theorems ground
//!   single-level squares of *affine* pre-squares only; certifying a square of
//!   a quadratic needs the same envelopes chained through an interval-bounded
//!   intermediate square variable — mechanically the same premise classes, but
//!   the chained construction is not implemented. This module fails closed
//!   with [`DominanceCertError::NestedSquareNotYetCertifiable`] (the plan's
//!   Route B — SMT escalation with an Alethe certificate, where polynomial `g`
//!   is native NRA — is the alternative path).
//!
//! `min`/`max` compositions are outside the polynomial-constant layer set
//! (piecewise, not polynomial) and are reported as ineligible outright.

use ndarray::{Array1, Array2};
use ny_cert::crown_deep::{DeepCrownError, DeepReluProblem, QuadTerm};
use ny_cert::fp_margin::deployed_fp_output_margin;
use ny_cert::{check_entailment, check_farkas, entailment_to_json, farkas_to_json, Rat, RatError};
use ny_core::Bound;
use ny_propagate::{GraphNetwork, Layer, NETWORK_INPUT};
use std::collections::BTreeMap;

/// A successfully emitted, self-checked dominance certificate.
#[derive(Debug, Clone)]
pub struct DominanceCertificate {
    /// The `.cert.json` body (Clean-canonical entailment + Farkas payloads).
    pub certificate_json: String,
    /// The exact certified lower bound on `f − g`, as `"n/d"`.
    pub lower_bound: String,
    /// The deployed-f32 rounding margin folded into the certificate
    /// threshold, as `"n/d"` (same encoding as [`Self::lower_bound`]), when
    /// [`DominanceCertOptions::deployed_fp_margin`] was requested. `None` for
    /// the default real-semantics-only certificate (threshold `0`).
    pub fp_margin: Option<String>,
}

/// Options for [`certify_dominance_with`].
#[derive(Debug, Clone, Copy, Default)]
pub struct DominanceCertOptions {
    /// Fold the deployed network's f32 rounding error into the certificate.
    ///
    /// The certificates prove properties of `f`'s IDEAL real-valued
    /// semantics; the deployed network executes in f32. When set, the
    /// certificate threshold becomes `delta =`
    /// Calling [`ny_cert::fp_margin::deployed_fp_output_margin`] with `f` computes an exact
    /// rational bound on `|fl32(f(x)) − f(x)|` over the box — instead of
    /// `0`, so the certified `f(x) − g(x) ≥ delta` implies
    /// `fl32(f(x)) − g(x) ≥ 0` for the deployed network (`g` is the exact
    /// analytic ground truth and contributes no rounding). The margin used
    /// is recorded in [`DominanceCertificate::fp_margin`]. Default `false`:
    /// byte-identical to the historical threshold-`0` certificate.
    pub deployed_fp_margin: bool,
}

/// Why a dominance certificate could not be emitted. Fail-closed and honest:
/// every variant names the gate that refused, and
/// [`DominanceCertError::NestedSquareNotYetCertifiable`] documents the open
/// nested-square (quartic, torus-class) gap — single-level quadratic `g` is
/// certified via the kernel-checked pow2 envelope theorems (see the module
/// docs).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DominanceCertError {
    /// `f` is not a certifiable sequential FC-ReLU network with scalar output.
    #[error("network `f` is not certifiable: {0}")]
    IneligibleNetwork(String),

    /// `g` is in the M1 polynomial-constant layer set but squares a value that
    /// is itself already quadratic (the TORUS builder's quartic residual) —
    /// the documented not-yet-certifiable nested-square case. Single-level
    /// squares ARE certified (grounded in the kernel-checked `pow2_tangent` /
    /// `pow2_secant` corpus theorems); chaining those envelopes through an
    /// interval-bounded intermediate square variable is the remaining step.
    #[error(
        "ground-truth side squares an already-quadratic value ({layer}): degree-4 \
         (torus-class) residuals are not yet certifiable — the kernel-checked pow2 \
         envelope theorems (pow2_tangent / pow2_secant) ground single-level squares \
         of affine pre-squares only; chaining them through nested squares is the \
         remaining step (plan §4; Route B SMT escalation is the alternative path)"
    )]
    NestedSquareNotYetCertifiable {
        /// Description of the offending squaring site.
        layer: String,
    },

    /// `g` is not a ground-truth graph in the M1 builders' layer set at all.
    #[error("ground-truth side is not in the M1 polynomial-constant layer set: {0}")]
    IneligibleGroundTruth(String),

    /// The input box has a non-finite endpoint (box premises need exact
    /// rational constants).
    #[error("input bound {index} is not finite: [{lower}, {upper}]")]
    NonFiniteBounds {
        /// Input dimension index.
        index: usize,
        /// Lower endpoint.
        lower: f32,
        /// Upper endpoint.
        upper: f32,
    },

    /// Exact CROWN could not close `f − g ≥ 0` (the verdict may rely on
    /// tighter bounds than plain exact CROWN), or exact arithmetic failed.
    #[error("exact certificate construction failed: {0}")]
    Construction(String),

    /// The freshly built certificate failed the in-tree self-check. This
    /// indicates a producer bug; the artifact is refused rather than emitted.
    #[error("certificate self-check failed: {0}")]
    SelfCheck(String),

    /// The deployed-f32 rounding margin was requested
    /// ([`DominanceCertOptions::deployed_fp_margin`]) but exceeds the best
    /// certified real-semantics lower bound on `f − g`: the real margin is
    /// too thin to absorb the deployed network's rounding error, so no
    /// deployed-sound certificate can be emitted.
    #[error(
        "deployed-f32 rounding margin {margin} exceeds the certified lower bound {bound}: \
         the real-semantics margin cannot absorb the deployed network's rounding error"
    )]
    FpMarginAboveBound {
        /// The requested margin `delta`, as `"n/d"`.
        margin: String,
        /// The best certified lower bound, as `"n/d"`.
        bound: String,
    },
}

/// Result alias for this module.
pub type CertResult<T> = Result<T, DominanceCertError>;

/// Certify `f(x) − g(x) ≥ 0` over the box `input_bounds` with an
/// exact-rational, self-checked entailment + Farkas certificate.
///
/// Eligibility (see the module docs): `f` must be a sequential FC-ReLU
/// `GraphNetwork` with a scalar affine read-out; `g` must fold to an exact
/// rational scalar of the form `g_lin·x + g₀ + Σⱼ qⱼ·tⱼ(x)²` with affine
/// `tⱼ` — a `Linear`/`PowConstant(2)`/`Add`/`Sub` graph with at most
/// single-level squaring. That covers the PLANE builder (posed or not,
/// squares empty → the pure-linear certificate) and the SPHERE / CYLINDER /
/// CONE quadric builders (squares present → the pow2-envelope certificate).
///
/// # Errors
/// See [`DominanceCertError`]; in particular the nested-square (torus-class
/// quartic) builder is refused with the documented obstruction rather than a
/// weakened artifact.
pub fn certify_dominance(
    f: &GraphNetwork,
    g: &GraphNetwork,
    input_bounds: &[Bound],
) -> CertResult<DominanceCertificate> {
    certify_dominance_with(f, g, input_bounds, &DominanceCertOptions::default())
}

/// Like [`certify_dominance`], with options. With
/// [`DominanceCertOptions::deployed_fp_margin`] set, the certificate
/// threshold is the deployed-f32 rounding margin `delta` of `f` instead of
/// `0`, so the emitted proof covers the DEPLOYED (f32-executing) network,
/// not only its ideal real-valued semantics; the margin used is recorded in
/// [`DominanceCertificate::fp_margin`]. Default options reproduce
/// [`certify_dominance`] byte-identically.
///
/// # Errors
/// As [`certify_dominance`]; additionally
/// [`DominanceCertError::FpMarginAboveBound`] when the margin was requested
/// but the certified real-semantics lower bound cannot absorb it.
pub fn certify_dominance_with(
    f: &GraphNetwork,
    g: &GraphNetwork,
    input_bounds: &[Bound],
    options: &DominanceCertOptions,
) -> CertResult<DominanceCertificate> {
    let folded = fold_ground_truth(g)?;
    let g_coeffs = folded.lin.clone();
    let g_offset = folded.offset;
    let stack = extract_fc_relu(f)?;

    if stack.input_dim != g_coeffs.len() {
        return Err(DominanceCertError::IneligibleNetwork(format!(
            "f input dim {} != ground-truth input dim {}",
            stack.input_dim,
            g_coeffs.len()
        )));
    }
    if input_bounds.len() != stack.input_dim {
        return Err(DominanceCertError::IneligibleNetwork(format!(
            "input box has {} dims, network expects {}",
            input_bounds.len(),
            stack.input_dim
        )));
    }
    let mut input_lower = Vec::with_capacity(input_bounds.len());
    let mut input_upper = Vec::with_capacity(input_bounds.len());
    for (i, b) in input_bounds.iter().enumerate() {
        let (Some(lo), Some(hi)) = (
            Rat::from_f32_exact(b.lower()),
            Rat::from_f32_exact(b.upper()),
        ) else {
            return Err(DominanceCertError::NonFiniteBounds {
                index: i,
                lower: b.lower(),
                upper: b.upper(),
            });
        };
        input_lower.push(lo);
        input_upper.push(hi);
    }

    let mut problem = DeepReluProblem {
        weights: stack.weights,
        biases: stack.biases,
        out_weight: stack.out_weight,
        out_bias: stack.out_bias,
        input_lower,
        input_upper,
        alpha: None,
        interm_round: false,
    };
    // Opt-in deployed-FP fold: the threshold becomes the exact rational bound
    // on |fl32(f(x)) − f(x)| over the box (g is exact analytic ground truth —
    // only f executes in f32), so `f − g ≥ delta` in real semantics implies
    // `fl32(f)(x) ≥ g(x)` for the deployed network. Default: threshold 0,
    // byte-identical to the historical certificate.
    let fp_margin =
        if options.deployed_fp_margin {
            Some(deployed_fp_output_margin(&problem).map_err(|e| {
                DominanceCertError::Construction(format!("deployed-FP margin: {e}"))
            })?)
        } else {
            None
        };
    let threshold = fp_margin.unwrap_or(Rat::ZERO);
    // The CROWN lower-envelope slope α is a free parameter: EVERY α ∈ [0, 1]
    // yields sound premises (the checker validates the combination either
    // way). The adaptive default can be a poor choice for nets with paired
    // ±units (|x| readouts) on slightly asymmetric boxes, so sweep a small,
    // deterministic set of policies and keep the first that closes.
    let uniform_alpha = |value: Rat, layers: &[Vec<Vec<Rat>>]| {
        layers
            .iter()
            .map(|layer| vec![value; layer.len()])
            .collect::<Vec<_>>()
    };
    let alpha_policies = [
        None,
        Some(uniform_alpha(Rat::ONE, &problem.weights)),
        Some(uniform_alpha(Rat::ZERO, &problem.weights)),
    ];
    let mut certified = None;
    let mut first_error: Option<String> = None;
    // Best (largest) certified bound refused only because it sat below the
    // requested FP margin — surfaced as the dedicated obstruction below.
    let mut threshold_refusal: Option<(String, String)> = None;
    for alpha in alpha_policies {
        problem.alpha = alpha;
        // Squares present => the quadratic producer (pow2 envelope premises,
        // grounded in the kernel-checked pow2_tangent / pow2_secant theorems);
        // otherwise the pure-linear producer, exactly as before.
        let attempt = if folded.squares.is_empty() {
            problem.certify_difference_linear(&g_coeffs, g_offset, threshold)
        } else {
            problem.certify_difference_quadratic(&g_coeffs, g_offset, &folded.squares, threshold)
        };
        match attempt {
            Ok(cert) => {
                certified = Some(cert);
                break;
            }
            Err(e) => {
                if let DeepCrownError::ThresholdAboveBound {
                    threshold: t,
                    bound,
                } = &e
                {
                    threshold_refusal.get_or_insert_with(|| (t.clone(), bound.clone()));
                }
                first_error.get_or_insert_with(|| e.to_string());
            }
        }
    }
    let Some(certified) = certified else {
        // A ThresholdAboveBound refusal under the FP-margin option means the
        // proof machinery worked — the real-semantics margin is simply too
        // thin to absorb deployed rounding. Name that precisely.
        if fp_margin.is_some() {
            if let Some((margin, bound)) = threshold_refusal {
                return Err(DominanceCertError::FpMarginAboveBound { margin, bound });
            }
        }
        return Err(DominanceCertError::Construction(
            first_error.unwrap_or_else(|| "exact CROWN did not close the property".to_string()),
        ));
    };

    // Refuse to emit anything that does not replay through the UNCHANGED
    // in-tree mirror of Clean's checker.
    check_entailment(&certified.entailment)
        .map_err(|e| DominanceCertError::SelfCheck(format!("entailment: {e}")))?;
    check_farkas(&certified.farkas)
        .map_err(|e| DominanceCertError::SelfCheck(format!("farkas: {e}")))?;

    let ratstr = |r: Rat| -> CertResult<String> {
        r.to_clean_string()
            .map_err(|e: RatError| DominanceCertError::Construction(e.to_string()))
    };
    // Preserve the public v1 `n/d` representation (including denominator 1)
    // while failing closed on an unhealthy rational arena.
    let (lower_num, lower_den) = certified
        .lower_bound
        .checked_parts()
        .map_err(|e| DominanceCertError::Construction(e.to_string()))?;
    let lower_bound = format!("{lower_num}/{lower_den}");
    let g_coeff_strings = g_coeffs
        .iter()
        .map(|c| ratstr(*c))
        .collect::<CertResult<Vec<_>>>()?;
    let entailment = entailment_to_json(&certified.entailment)
        .map_err(|e| DominanceCertError::Construction(format!("entailment JSON: {e}")))?;
    let farkas = farkas_to_json(&certified.farkas)
        .map_err(|e| DominanceCertError::Construction(format!("farkas JSON: {e}")))?;
    let squares_json = folded
        .squares
        .iter()
        .map(|t| -> CertResult<serde_json::Value> {
            Ok(serde_json::json!({
                "coeff": ratstr(t.coeff)?,
                "lin": t.lin.iter().map(|c| ratstr(*c)).collect::<CertResult<Vec<_>>>()?,
                "offset": ratstr(t.offset)?,
            }))
        })
        .collect::<CertResult<Vec<_>>>()?;
    let claim = if folded.squares.is_empty() {
        "exact CROWN discharges f(x) - g(x) >= 0 over the whole input box; \
         g is an exact rational affine ground truth entering the certificate \
         as read-out premise rows"
    } else {
        "exact CROWN discharges f(x) - g(x) >= 0 over the whole input box; \
         g is an exact rational quadratic ground truth (affine part as read-out \
         premise rows; each square via definitional t-rows plus pow2 tangent/secant \
         envelope premises grounded in the kernel-checked pow2_tangent / pow2_secant \
         corpus theorems)"
    };
    let fp_margin_string = fp_margin.map(|delta| format!("{}/{}", delta.num(), delta.den()));
    let mut payload = serde_json::json!({
        "format": "ny-cert/ground-truth-dominance/v1",
        "claim": claim,
        "ground_truth": {
            "coeffs": g_coeff_strings,
            "offset": ratstr(g_offset)?,
            "squares": squares_json,
        },
        "exact_lower_bound": lower_bound,
        "entailment": entailment,
        "farkas": farkas,
    });
    // Only inserted when the option is on: the default payload stays
    // byte-identical to the historical (threshold-0) certificate.
    if let Some(margin) = &fp_margin_string {
        payload["fp_margin"] = serde_json::json!(margin);
    }
    let certificate_json = serde_json::to_string_pretty(&payload)
        .map_err(|e| DominanceCertError::Construction(format!("serialization: {e}")))?;
    Ok(DominanceCertificate {
        certificate_json,
        lower_bound,
        fp_margin: fp_margin_string,
    })
}

/// The EXACT dyadic rational of a finite f32 graph constant, or an
/// ineligibility error naming it.
fn exact(context: &str, v: f32) -> CertResult<Rat> {
    Rat::from_f32_exact(v).ok_or_else(|| {
        DominanceCertError::IneligibleNetwork(format!("{context} is not finite: {v}"))
    })
}

/// Walk a single-input chain from the output node back to `NETWORK_INPUT`,
/// returning the layers in input→output order. `describe_reject` renders the
/// error for a disqualifying node.
fn walk_chain<'g>(graph: &'g GraphNetwork, who: &str) -> Result<Vec<&'g Layer>, String> {
    let mut chain: Vec<&Layer> = Vec::new();
    let mut current = graph.output_name().to_string();
    if current.is_empty() {
        return Err(format!("{who} has no output node set"));
    }
    let max_steps = graph.num_nodes().saturating_add(1);
    let mut steps = 0usize;
    loop {
        if steps > max_steps {
            return Err(format!(
                "{who} chain did not terminate at the network input"
            ));
        }
        steps += 1;
        let node = graph
            .node(&current)
            .ok_or_else(|| format!("{who} has a dangling node reference '{current}'"))?;
        chain.push(node.layer());
        let inputs = node.inputs();
        if inputs.len() != 1 {
            return Err(format!(
                "{who} node '{current}' has {} inputs (need a single linear chain)",
                inputs.len()
            ));
        }
        let pred = inputs[0].clone();
        if pred == NETWORK_INPUT {
            break;
        }
        current = pred;
    }
    chain.reverse();
    Ok(chain)
}

/// The ground-truth side folded to the exact rational canonical form
/// `g(x) = lin·x + offset + Σⱼ squares[j].coeff · (squares[j].lin·x + squares[j].offset)²`.
struct FoldedGroundTruth {
    lin: Vec<Rat>,
    offset: Rat,
    squares: Vec<QuadTerm>,
}

/// One scalar value during folding: an affine part over the inputs plus a
/// weighted sparse sum over the shared pre-square table (`(table index,
/// coefficient)` pairs, index-sorted, no zero coefficients).
#[derive(Clone)]
struct ScalarForm {
    lin: Vec<Rat>,
    offset: Rat,
    squares: Vec<(usize, Rat)>,
}

impl ScalarForm {
    fn zero(input_dim: usize) -> Self {
        ScalarForm {
            lin: vec![Rat::ZERO; input_dim],
            offset: Rat::ZERO,
            squares: Vec::new(),
        }
    }

    /// `self += c · other` (exact; drops cancelled square coefficients).
    fn add_scaled(&mut self, c: Rat, other: &ScalarForm) -> Result<(), RatError> {
        if c.is_zero() {
            return Ok(());
        }
        for (dst, src) in self.lin.iter_mut().zip(&other.lin) {
            *dst = dst.add(c.mul(*src)?)?;
        }
        self.offset = self.offset.add(c.mul(other.offset)?)?;
        for (id, q) in &other.squares {
            let scaled = c.mul(*q)?;
            match self.squares.binary_search_by_key(id, |&(i, _)| i) {
                Ok(pos) => {
                    let next = self.squares[pos].1.add(scaled)?;
                    if next.is_zero() {
                        self.squares.remove(pos);
                    } else {
                        self.squares[pos].1 = next;
                    }
                }
                Err(pos) => {
                    if !scaled.is_zero() {
                        self.squares.insert(pos, (*id, scaled));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Fold the ground-truth graph (a `Linear`/`PowConstant(2)`/`Add`/`Sub` DAG
/// over exact constants) to [`FoldedGroundTruth`]. Pure-`Linear` graphs (the
/// plane builder, posed or not) fold with `squares` empty; the quadric
/// builders (sphere / cylinder / cone) fold to single-level squares of affine
/// pre-squares. Squaring an already-quadratic value (the torus quartic) is
/// refused with the documented [`DominanceCertError::NestedSquareNotYetCertifiable`].
fn fold_ground_truth(g: &GraphNetwork) -> CertResult<FoldedGroundTruth> {
    let ineligible = DominanceCertError::IneligibleGroundTruth;
    let output = g.output_name().to_string();
    if output.is_empty() {
        return Err(ineligible(
            "ground-truth graph has no output node set".into(),
        ));
    }
    // The input dimension: read off any Linear node fed by the network input
    // (every M1 builder starts with one; a graph squaring the raw input would
    // still need a summing Linear, but be conservative and require it).
    let mut input_dim = None;
    for name in g.node_names() {
        let node = g
            .node(name)
            .ok_or_else(|| ineligible(format!("dangling node reference '{name}'")))?;
        if node.inputs().iter().any(|i| i == NETWORK_INPUT) {
            if let Layer::Linear(lin) = node.layer() {
                let w: &Array2<f32> = lin.weight();
                input_dim = Some(w.ncols());
                break;
            }
        }
    }
    let Some(input_dim) = input_dim else {
        return Err(ineligible(
            "ground-truth graph has no Linear node reading the network input \
             (cannot determine the input dimension)"
                .into(),
        ));
    };

    let mut memo: BTreeMap<String, Vec<ScalarForm>> = BTreeMap::new();
    let mut presquares: Vec<(Vec<Rat>, Rat)> = Vec::new();
    let forms = fold_node(g, &output, input_dim, &mut memo, &mut presquares, 0)?;
    if forms.len() != 1 {
        return Err(ineligible(format!(
            "ground-truth output must be scalar, got {} outputs",
            forms.len()
        )));
    }
    let scalar = &forms[0];
    let squares = scalar
        .squares
        .iter()
        .map(|&(id, coeff)| QuadTerm {
            coeff,
            lin: presquares[id].0.clone(),
            offset: presquares[id].1,
        })
        .collect();
    Ok(FoldedGroundTruth {
        lin: scalar.lin.clone(),
        offset: scalar.offset,
        squares,
    })
}

/// Recursively fold one node of the ground-truth DAG (memoized). `depth`
/// fail-closes on cycles (a well-formed `GraphNetwork` is acyclic, but a
/// dangling/cyclic reference must not hang the certifier).
fn fold_node(
    g: &GraphNetwork,
    name: &str,
    input_dim: usize,
    memo: &mut BTreeMap<String, Vec<ScalarForm>>,
    presquares: &mut Vec<(Vec<Rat>, Rat)>,
    depth: usize,
) -> CertResult<Vec<ScalarForm>> {
    let ineligible = DominanceCertError::IneligibleGroundTruth;
    let rat_err = |e: RatError| DominanceCertError::Construction(e.to_string());
    if depth > g.num_nodes() {
        return Err(ineligible(
            "ground-truth graph did not terminate at the network input".into(),
        ));
    }
    if name == NETWORK_INPUT {
        // The identity vector of input coordinates.
        let mut basis = Vec::with_capacity(input_dim);
        for i in 0..input_dim {
            let mut f = ScalarForm::zero(input_dim);
            f.lin[i] = Rat::ONE;
            basis.push(f);
        }
        return Ok(basis);
    }
    if let Some(cached) = memo.get(name) {
        return Ok(cached.clone());
    }
    let node = g
        .node(name)
        .ok_or_else(|| ineligible(format!("dangling node reference '{name}'")))?;
    let inputs = node.inputs();
    let out = match node.layer() {
        Layer::Linear(lin) => {
            if inputs.len() != 1 {
                return Err(ineligible(format!(
                    "Linear node '{name}' has {} inputs",
                    inputs.len()
                )));
            }
            let child = fold_node(g, &inputs[0], input_dim, memo, presquares, depth + 1)?;
            let (w, bias) = linear_to_rats(lin)?;
            let mut out = Vec::with_capacity(w.len());
            for (row, b) in w.iter().zip(&bias) {
                if row.len() != child.len() {
                    return Err(ineligible(format!(
                        "Linear node '{name}' expects {} inputs, child has {}",
                        row.len(),
                        child.len()
                    )));
                }
                let mut f = ScalarForm::zero(input_dim);
                f.offset = *b;
                for (w_ij, cf) in row.iter().zip(&child) {
                    f.add_scaled(*w_ij, cf).map_err(rat_err)?;
                }
                out.push(f);
            }
            out
        }
        Layer::PowConstant(p) => {
            if inputs.len() != 1 {
                return Err(ineligible(format!(
                    "PowConstant node '{name}' has {} inputs",
                    inputs.len()
                )));
            }
            // Only the exact square is in the M1 squared/residual form.
            #[allow(clippy::float_cmp)] // exact representable constant 2.0
            if p.exponent() != 2.0 {
                return Err(ineligible(format!(
                    "PowConstant exponent {} is outside the squared/residual form (only 2)",
                    p.exponent()
                )));
            }
            let child = fold_node(g, &inputs[0], input_dim, memo, presquares, depth + 1)?;
            let mut out = Vec::with_capacity(child.len());
            for (idx, cf) in child.iter().enumerate() {
                if !cf.squares.is_empty() {
                    // Square of a quadratic: degree 4 — the torus builder.
                    return Err(DominanceCertError::NestedSquareNotYetCertifiable {
                        layer: format!("PowConstant(2) node '{name}' component {idx}"),
                    });
                }
                // Deduplicate identical affine pre-squares (the cone builder
                // squares the same centered coordinates along two paths).
                let key = (cf.lin.clone(), cf.offset);
                let id = presquares
                    .iter()
                    .position(|entry| *entry == key)
                    .unwrap_or_else(|| {
                        presquares.push(key);
                        presquares.len() - 1
                    });
                let mut f = ScalarForm::zero(input_dim);
                f.squares.push((id, Rat::ONE));
                out.push(f);
            }
            out
        }
        Layer::Add(_) | Layer::Sub(_) => {
            if inputs.len() != 2 {
                return Err(ineligible(format!(
                    "binary node '{name}' has {} inputs",
                    inputs.len()
                )));
            }
            let lhs = fold_node(g, &inputs[0], input_dim, memo, presquares, depth + 1)?;
            let rhs = fold_node(g, &inputs[1], input_dim, memo, presquares, depth + 1)?;
            if lhs.len() != rhs.len() {
                return Err(ineligible(format!(
                    "binary node '{name}' input widths differ: {} vs {}",
                    lhs.len(),
                    rhs.len()
                )));
            }
            let sign = if matches!(node.layer(), Layer::Sub(_)) {
                Rat::ONE.neg()
            } else {
                Rat::ONE
            };
            let mut out = Vec::with_capacity(lhs.len());
            for (a, b) in lhs.iter().zip(&rhs) {
                let mut f = a.clone();
                f.add_scaled(sign, b).map_err(rat_err)?;
                out.push(f);
            }
            out
        }
        other => {
            return Err(ineligible(format!(
                "layer {other:?} is outside the Linear/PowConstant/Add/Sub set"
            )));
        }
    };
    memo.insert(name.to_string(), out.clone());
    Ok(out)
}

/// Exact rational rows/bias of one Linear layer.
fn linear_to_rats(
    lin: &ny_propagate::layers::LinearLayer,
) -> CertResult<(Vec<Vec<Rat>>, Vec<Rat>)> {
    let w: &Array2<f32> = lin.weight();
    let mut rows = Vec::with_capacity(w.nrows());
    for row in w.rows() {
        let mut r = Vec::with_capacity(row.len());
        for &v in &row {
            r.push(exact("ground-truth weight", v)?);
        }
        rows.push(r);
    }
    let bias = match lin.bias() {
        Some(b) => {
            let b: &Array1<f32> = b;
            b.iter()
                .map(|&v| exact("ground-truth bias", v))
                .collect::<CertResult<Vec<_>>>()?
        }
        None => vec![Rat::ZERO; w.nrows()],
    };
    Ok((rows, bias))
}

/// `f` decomposed into exact rational FC-ReLU pieces with a scalar read-out.
struct RatStack {
    input_dim: usize,
    weights: Vec<Vec<Vec<Rat>>>,
    biases: Vec<Vec<Rat>>,
    out_weight: Vec<Rat>,
    out_bias: Rat,
}

/// Extract `f` as a sequential FC-ReLU stack (`Linear, ReLU, …, Linear`) with
/// a SCALAR affine read-out, in exact rationals.
fn extract_fc_relu(f: &GraphNetwork) -> CertResult<RatStack> {
    let chain = walk_chain(f, "network `f`").map_err(DominanceCertError::IneligibleNetwork)?;

    let mut linears: Vec<(Vec<Vec<Rat>>, Vec<Rat>)> = Vec::new();
    let mut expect_relu = false;
    for layer in chain {
        match layer {
            Layer::Linear(lin) => {
                if expect_relu {
                    return Err(DominanceCertError::IneligibleNetwork(
                        "two Linear layers with no ReLU between them".to_string(),
                    ));
                }
                linears.push(linear_to_rats(lin).map_err(|e| {
                    DominanceCertError::IneligibleNetwork(format!("f weight conversion: {e}"))
                })?);
                expect_relu = true;
            }
            Layer::ReLU(_) => {
                if !expect_relu {
                    return Err(DominanceCertError::IneligibleNetwork(
                        "ReLU with no preceding Linear".to_string(),
                    ));
                }
                expect_relu = false;
            }
            other => {
                return Err(DominanceCertError::IneligibleNetwork(format!(
                    "unsupported layer for cert: {other:?}"
                )));
            }
        }
    }
    if linears.len() < 2 {
        return Err(DominanceCertError::IneligibleNetwork(format!(
            "need >=2 Linear layers (>=1 hidden + readout), found {}",
            linears.len()
        )));
    }
    if !expect_relu {
        return Err(DominanceCertError::IneligibleNetwork(
            "final Linear is followed by a ReLU (not an affine read-out)".to_string(),
        ));
    }
    let (readout_w, readout_b) = linears.pop().unwrap_or_default();
    if readout_w.len() != 1 {
        return Err(DominanceCertError::IneligibleNetwork(format!(
            "read-out must be scalar for the dominance property, got {} outputs",
            readout_w.len()
        )));
    }
    let input_dim = linears
        .first()
        .and_then(|(w, _)| w.first())
        .map_or(0, Vec::len);
    let (weights, biases) = linears.into_iter().unzip();
    let out_weight = readout_w.into_iter().next().unwrap_or_default();
    let out_bias = readout_b.first().copied().unwrap_or(Rat::ZERO);
    Ok(RatStack {
        input_dim,
        weights,
        biases,
        out_weight,
        out_bias,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::{
        cone_residual, cylinder_residual, signed_plane_distance, sphere_residual, torus_residual,
    };
    use crate::compose::min_of;
    use crate::sidecar::GroundTruthSpec;
    use ny_propagate::layers::{LinearLayer, ReLULayer};
    use ny_propagate::GraphNode;

    fn linear(rows: usize, cols: usize, w: Vec<f32>, b: Vec<f32>) -> Layer {
        Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((rows, cols), w).expect("shape"),
                Some(Array1::from(b)),
            )
            .expect("valid linear"),
        )
    }

    /// f(x) = relu(x0) + relu(-x0) + relu(x1) + relu(-x1) + 3
    ///      = |x0| + |x1| + 3, a genuine 3->4->1 FC-ReLU net.
    fn abs_sum_net() -> GraphNetwork {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "lin1",
            linear(
                4,
                3,
                vec![
                    1.0, 0.0, 0.0, //
                    -1.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, //
                    0.0, -1.0, 0.0,
                ],
                vec![0.0; 4],
            ),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "readout",
            linear(1, 4, vec![1.0, 1.0, 1.0, 1.0], vec![3.0]),
            vec!["relu".to_string()],
        ));
        g.set_output("readout");
        g
    }

    fn unit_box() -> Vec<Bound> {
        vec![
            Bound::new(-1.0, 1.0),
            Bound::new(-1.0, 1.0),
            Bound::new(-1.0, 1.0),
        ]
    }

    #[test]
    fn plane_dominance_certifies_end_to_end() {
        // g(x) = x2 − 0.5; f − g = |x0| + |x1| − x2 + 3.5 ≥ 2.5 on the box.
        let f = abs_sum_net();
        let g = signed_plane_distance([0.0, 0.0, 1.0], -0.5).expect("plane builds");
        let cert = certify_dominance(&f, &g, &unit_box()).expect("plane case certifies");

        let parsed: serde_json::Value =
            serde_json::from_str(&cert.certificate_json).expect("cert JSON parses");
        assert_eq!(parsed["format"], "ny-cert/ground-truth-dominance/v1");
        assert_eq!(parsed["exact_lower_bound"], cert.lower_bound);
        assert_eq!(parsed["ground_truth"]["coeffs"][2], "1");
        assert_eq!(parsed["ground_truth"]["offset"], "-1/2");
        assert!(parsed["entailment"].is_object());
        assert!(parsed["farkas"].is_object());

        // The certified bound must be a genuine non-negative rational, and
        // (soundness) must not exceed the true minimum 5/2 of f − g.
        let (num, den) = cert
            .lower_bound
            .split_once('/')
            .map_or((cert.lower_bound.as_str(), "1"), |(n, d)| (n, d));
        let lb: f64 = num.parse::<f64>().unwrap() / den.parse::<f64>().unwrap();
        assert!((0.0..=2.5).contains(&lb), "lower bound {lb} out of range");
    }

    #[test]
    fn plane_via_sidecar_builder_certifies() {
        // End-to-end through the .gt.json loader: the sanctioned M2->M3 path.
        let f = abs_sum_net();
        let spec = GroundTruthSpec::plane([0.0, 0.0, 1.0], -0.5);
        let g = spec.build().expect("sidecar plane builds");
        certify_dominance(&f, &g, &unit_box()).expect("sidecar plane certifies");
    }

    #[test]
    fn sphere_dominance_certifies_end_to_end() {
        // g(x) = ‖x‖² − 9/4; f − g = |x0| + |x1| + 21/4 − ‖x‖² ≥ 13/4 on the
        // unit box. The certificate uses the pow2 secant premises (all square
        // coefficients are +1), and must self-check and stay sound.
        let f = abs_sum_net();
        let g = sphere_residual([0.0, 0.0, 0.0], 1.5).expect("sphere builds");
        let cert = certify_dominance(&f, &g, &unit_box()).expect("sphere case certifies");

        let parsed: serde_json::Value =
            serde_json::from_str(&cert.certificate_json).expect("cert JSON parses");
        assert_eq!(parsed["format"], "ny-cert/ground-truth-dominance/v1");
        assert_eq!(parsed["ground_truth"]["offset"], "-9/4");
        let squares = parsed["ground_truth"]["squares"]
            .as_array()
            .expect("squares array");
        assert_eq!(squares.len(), 3, "three coordinate squares");
        for sq in squares {
            assert_eq!(sq["coeff"], "1");
        }
        assert!(
            parsed["claim"]
                .as_str()
                .unwrap()
                .contains("pow2_tangent / pow2_secant"),
            "claim names the kernel-checked grounding"
        );
        // The entailment mentions the square variables (the envelope premises).
        assert!(cert.certificate_json.contains("s0"));

        // Soundness: the certified bound cannot exceed the true minimum 13/4.
        let (num, den) = cert
            .lower_bound
            .split_once('/')
            .map_or((cert.lower_bound.as_str(), "1"), |(n, d)| (n, d));
        let lb: f64 = num.parse::<f64>().unwrap() / den.parse::<f64>().unwrap();
        assert!(
            (0.0..=3.25).contains(&lb),
            "lower bound {lb} out of range (true min 13/4)"
        );
    }

    #[test]
    fn cylinder_dominance_certifies_end_to_end() {
        // g(x) = x0² + x1² − 9 (axis z); f − g ≥ 10 closes trivially, but the
        // certificate must be constructed, self-checked, and emitted.
        let f = abs_sum_net();
        let g = cylinder_residual([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 3.0).expect("cylinder");
        let cert = certify_dominance(&f, &g, &unit_box()).expect("cylinder case certifies");
        let parsed: serde_json::Value = serde_json::from_str(&cert.certificate_json).unwrap();
        assert_eq!(parsed["ground_truth"]["offset"], "-9");
        // True min of f − g on the box is 10 (at |x0|=|x1|=1): |t|−t² ≥ 0.
        let (num, den) = cert
            .lower_bound
            .split_once('/')
            .map_or((cert.lower_bound.as_str(), "1"), |(n, d)| (n, d));
        let lb: f64 = num.parse::<f64>().unwrap() / den.parse::<f64>().unwrap();
        assert!((0.0..=10.0).contains(&lb), "lower bound {lb} out of range");
    }

    #[test]
    fn cone_dominance_exercises_negative_squares_and_dedupe() {
        // g(x) = ½‖x‖² − x2² (axis z, apex 0, cos²α = ½). The axial square
        // x2² DEDUPES against the centered coordinate square, leaving three
        // pre-squares with coefficients (½, ½, −½) — the negative one takes
        // the pow2 TANGENT premise. f − g = |x0|+|x1|+3 − ½(x0²+x1²) + ½x2² ≥ 3.
        let f = abs_sum_net();
        let g = cone_residual([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 0.5).expect("cone builds");
        let cert = certify_dominance(&f, &g, &unit_box()).expect("cone case certifies");
        let parsed: serde_json::Value = serde_json::from_str(&cert.certificate_json).unwrap();
        let squares = parsed["ground_truth"]["squares"]
            .as_array()
            .expect("squares array");
        assert_eq!(squares.len(), 3, "axial square dedupes into coordinates");
        let coeffs: Vec<&str> = squares
            .iter()
            .map(|s| s["coeff"].as_str().unwrap())
            .collect();
        assert!(
            coeffs.contains(&"-1/2"),
            "one square carries the negative (tangent-side) coefficient: {coeffs:?}"
        );
        let (num, den) = cert
            .lower_bound
            .split_once('/')
            .map_or((cert.lower_bound.as_str(), "1"), |(n, d)| (n, d));
        let lb: f64 = num.parse::<f64>().unwrap() / den.parse::<f64>().unwrap();
        assert!((0.0..=3.0).contains(&lb), "lower bound {lb} out of range");
    }

    #[test]
    fn fp_margin_option_certifies_and_records_the_margin() {
        // Same plane case as `plane_dominance_certifies_end_to_end` (real
        // margin 2.5, vastly above the ~1e-6 deployed-FP delta): the
        // certificate must still close with `threshold = delta`, record the
        // margin, and self-check.
        let f = abs_sum_net();
        let g = signed_plane_distance([0.0, 0.0, 1.0], -0.5).expect("plane builds");
        let opts = DominanceCertOptions {
            deployed_fp_margin: true,
        };
        let cert =
            certify_dominance_with(&f, &g, &unit_box(), &opts).expect("margin case certifies");
        let margin = cert.fp_margin.as_deref().expect("margin is recorded");
        let (num, den) = margin.split_once('/').expect("margin is n/d");
        let m: f64 = num.parse::<f64>().unwrap() / den.parse::<f64>().unwrap();
        assert!(
            m > 0.0 && m < 1e-3,
            "deployed-FP margin should be tiny and positive, got {m}"
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&cert.certificate_json).expect("cert JSON parses");
        assert_eq!(parsed["fp_margin"], margin, "payload carries the margin");
        // The proved conclusion is now `y >= delta`, not `y >= 0`.
        assert_eq!(parsed["entailment"]["conclusion"]["constant"], margin);

        // Default behavior is untouched: no field, no payload key, and the
        // payload is byte-identical to the historical certificate.
        let base = certify_dominance(&f, &g, &unit_box()).expect("baseline certifies");
        assert!(base.fp_margin.is_none(), "default records no margin");
        assert!(
            !base.certificate_json.contains("fp_margin"),
            "default payload has no fp_margin key"
        );
        let base_with =
            certify_dominance_with(&f, &g, &unit_box(), &DominanceCertOptions::default())
                .expect("default options certify");
        assert_eq!(
            base.certificate_json, base_with.certificate_json,
            "default options reproduce certify_dominance byte-identically"
        );
    }

    #[test]
    fn fp_margin_exceeding_the_bound_is_surfaced_as_the_dedicated_error() {
        // g(x) = x2 + 2: f − g = |x0| + |x1| + 1 − x2 has true minimum 0 on
        // the box and exact CROWN certifies the bound EXACTLY 0 — so the
        // threshold-0 baseline certifies, while any positive deployed-FP
        // margin cannot be absorbed and must surface as FpMarginAboveBound.
        let f = abs_sum_net();
        let g = signed_plane_distance([0.0, 0.0, 1.0], 2.0).expect("plane builds");
        certify_dominance(&f, &g, &unit_box()).expect("threshold-0 baseline certifies");

        let opts = DominanceCertOptions {
            deployed_fp_margin: true,
        };
        let err = certify_dominance_with(&f, &g, &unit_box(), &opts)
            .expect_err("zero real margin cannot absorb deployed rounding");
        let DominanceCertError::FpMarginAboveBound { margin, bound } = err else {
            panic!("expected FpMarginAboveBound, got: {err:?}");
        };
        assert_eq!(bound, "0/1", "the certified bound is exactly zero");
        // The refused margin is the positive delta. Its numerator is a
        // bignum (far beyond i64), so check sign/nonzero structurally.
        let (num, den) = margin.split_once('/').expect("margin is n/d");
        assert!(
            num.chars().all(|c| c.is_ascii_digit()) && num != "0",
            "the refused margin numerator is a positive integer: {margin}"
        );
        assert!(
            den.chars().all(|c| c.is_ascii_digit()) && den != "0",
            "the refused margin denominator is a positive integer: {margin}"
        );
    }

    #[test]
    fn torus_nested_square_is_refused_with_the_documented_obstruction() {
        // The torus residual squares (‖x−p‖² + R² − r²) — a quadratic — so it
        // is quartic and stays refused, honestly and precisely.
        let f = abs_sum_net();
        let g = torus_residual([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 2.0, 1.0).expect("torus builds");
        let err = certify_dominance(&f, &g, &unit_box()).expect_err("quartic must be refused");
        assert!(matches!(
            err,
            DominanceCertError::NestedSquareNotYetCertifiable { .. }
        ));
        let msg = err.to_string();
        assert!(
            msg.contains("pow2_tangent") && msg.contains("nested squares"),
            "obstruction message must state the remaining step: {msg}"
        );
    }

    #[test]
    fn min_composition_is_ineligible_not_quadratic() {
        let f = abs_sum_net();
        let p1 = signed_plane_distance([0.0, 0.0, 1.0], 0.0).unwrap();
        let p2 = signed_plane_distance([0.0, 1.0, 0.0], 0.0).unwrap();
        let g = min_of(&[p1, p2]).unwrap();
        assert!(matches!(
            certify_dominance(&f, &g, &unit_box()),
            Err(DominanceCertError::IneligibleGroundTruth(_))
        ));
    }

    #[test]
    fn non_fc_relu_f_is_ineligible() {
        // Using a quadric residual AS f (contains PowConstant) must be refused
        // on the f-side gate.
        let f = cylinder_residual([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 3.0).unwrap();
        let g = signed_plane_distance([0.0, 0.0, 1.0], 0.0).unwrap();
        assert!(matches!(
            certify_dominance(&f, &g, &unit_box()),
            Err(DominanceCertError::IneligibleNetwork(_))
        ));
    }

    #[test]
    fn infinite_box_is_refused() {
        let f = abs_sum_net();
        let g = signed_plane_distance([0.0, 0.0, 1.0], 0.0).unwrap();
        let bounds = vec![
            Bound::new(-1.0, 1.0),
            Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY),
            Bound::new(-1.0, 1.0),
        ];
        assert!(matches!(
            certify_dominance(&f, &g, &bounds),
            Err(DominanceCertError::NonFiniteBounds { index: 1, .. })
        ));
    }

    #[test]
    fn posed_plane_folds_and_certifies() {
        // Pose (a second Linear in the chain): swap x0/x2 then the x2-plane
        // reads x0. g(x) = x0 − 0.5.
        let spec =
            GroundTruthSpec::plane([0.0, 0.0, 1.0], -0.5).with_pose(crate::sidecar::PoseSpec {
                linear: [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                translation: [0.0, 0.0, 0.0],
            });
        let g = spec.build().expect("posed plane builds");
        let f = abs_sum_net();
        let cert = certify_dominance(&f, &g, &unit_box()).expect("posed plane certifies");
        let parsed: serde_json::Value = serde_json::from_str(&cert.certificate_json).unwrap();
        assert_eq!(parsed["ground_truth"]["coeffs"][0], "1");
        assert_eq!(parsed["ground_truth"]["coeffs"][2], "0");
    }
}
