// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proof-carrying certificate adapter.
//!
//! After β-CROWN returns a `Verified` verdict, this re-derives an
//! **exact-rational** CROWN certificate from the verified network and the
//! property's input box using the `ny-cert` crate, runs the in-tree self-check
//! (entailment + Farkas replay), and writes a `.cert.json` sidecar that Clean's
//! trusted, kernel-backed external-certificate verifier can check. This turns a
//! `Verified` verdict into a machine-checkable proof rather than a
//! floating-point claim — the headline of NY's *Proof-Carrying Verification*
//! program (see `crates/ny-cert/src/lib.rs`).
//!
//! ## Soundness / scope invariants
//!
//! * **Certificate emission NEVER upgrades the runtime verdict.** If we cannot
//!   certify (ineligible architecture, a rational-arena failure, or the
//!   property is above the exact plain-CROWN bound), we omit the sidecar and
//!   log. The typed pre-verification artifact request may conservatively
//!   disable a verdict-only proof lane whose proof cannot be exported.
//!   Emission is wrapped so an adapter bug can only *fail to emit*, never abort
//!   a run.
//! * **The reconstructed claim is "safe ⟺ margin ≥ 0".** We fold the property's
//!   threshold and direction into a scalar margin, then `certify(0)`. If we
//!   picked the direction wrong, `certify` simply fails on a genuinely-safe
//!   instance, so a direction mistake degrades to a *skipped* cert — it can
//!   never emit a cert that asserts the wrong inequality.
//! * **Weights are converted to rationals EXACTLY** via their dyadic IEEE-754
//!   value (no rounding), so the certificate attests the f32-realised network
//!   the verifier actually ran on — it is faithful to the verified object.
//! * **Emission is gated by `ProofOpts`:** ON by default, OFF in competition
//!   mode (the VNN-COMP scored entry point), where the exact-arithmetic pass is
//!   pure overhead.

use std::path::PathBuf;

use ny_cert::crown_deep::{DeepCrownError, DeepReluProblem};
use ny_cert::{check_entailment, check_farkas, schema, Rat};
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::{BabVerificationStatus, BetaCrownResult, Layer};
use tracing::{info, warn};

use super::dispatch::DispatchContext;
use super::BetaCrownModel;

/// Upper bound on the total neuron count we attempt to certify. Exact rational
/// CROWN over a large net can create very large intermediate values and consume
/// excessive time/memory; above this we skip (non-fatal). Pure-FC eligible
/// categories (sat_relu, soundnessbench, small ACAS-style nets) sit well under
/// this.
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
/// order. `hidden` is `(weight_rows, bias)` for each hidden Linear (each
/// followed by a ReLU); `readout` is the final Linear (NOT followed by ReLU).
struct FcReluStack<'a> {
    hidden: Vec<(&'a ndarray::Array2<f32>, Option<&'a ndarray::Array1<f32>>)>,
    readout: (&'a ndarray::Array2<f32>, Option<&'a ndarray::Array1<f32>>),
}

/// Walk a sequential network and, if it is a clean FC-ReLU stack
/// (`Linear, ReLU, Linear, ReLU, …, Linear`), return its decomposition.
/// Returns `Err(reason)` for any other shape — used only to log why a network
/// is ineligible, never to fail a run.
fn extract_fc_relu_stack(network: &ny_propagate::Network) -> Result<FcReluStack<'_>, String> {
    let layers = network.layers();
    // Collect Linear layers in order; require a ReLU between consecutive ones
    // and reject every other layer kind (Conv, MatMul/Add-fused affine,
    // Softmax, normalisation, …). `expect_relu` tracks the alternation.
    let mut linears: Vec<(&ndarray::Array2<f32>, Option<&ndarray::Array1<f32>>)> = Vec::new();
    let mut expect_relu = false;
    for layer in layers {
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
            other => return Err(format!("unsupported layer for cert: {other:?}")),
        }
    }
    if linears.len() < 2 {
        return Err(format!(
            "need >=2 Linear layers (>=1 hidden + readout), found {}",
            linears.len()
        ));
    }
    // The last collected Linear is the readout; it must NOT be followed by a
    // ReLU (a trailing ReLU would mean the output passes through ReLU, which is
    // not the affine read-out DeepReluProblem expects).
    if !expect_relu {
        return Err("final Linear is followed by a ReLU (not an affine read-out)".to_string());
    }
    let readout = linears.pop().expect("len>=2 checked above");
    Ok(FcReluStack {
        hidden: linears,
        readout,
    })
}

/// Convert an `f64` to its EXACT dyadic rational value. VNN-LIB constants are
/// parsed as `f64`; non-finite values and a poisoned arena fail closed.
fn f64_to_rat(f: f64) -> Option<Rat> {
    Rat::from_f64_exact(f)
}

/// A single linear output inequality `margin(x) > 0` whose truth over the whole
/// input box discharges (empties) the unsafe region. `coeffs[k] = (idx, sign)`
/// contributes `sign · Y_idx`; `const_term` is added. When `strict`, the
/// certified bound must be strictly positive (refuting a possibly-closed unsafe
/// conjunct); otherwise `>= 0` suffices (the direct epsilon-robustness margin).
struct Margin {
    coeffs: Vec<(usize, i32)>,
    const_term: Rat,
    strict: bool,
    label: String,
}

/// Build the candidate margins whose individual truth proves the property safe.
///
/// VNN-LIB semantics ([`ny_onnx::vnnlib::VnnLibSpec`]): when
/// `is_disjunction == false` the unsafe region is the CONJUNCTION of the output
/// constraints, so it is empty if ANY ONE constraint's complement holds over
/// the whole box — we return one margin per conjunct and emit on the first that
/// certifies. A disjunctive unsafe region needs ALL clauses refuted at once and
/// is deferred (`None`). With no spec (ε-ball robustness) we form the single
/// direct safe-margin from the threshold/direction the verifier used.
fn unsafe_complement_margins(ctx: &DispatchContext<'_>) -> Option<Vec<Margin>> {
    if let Some(spec) = ctx.vnnlib_spec {
        if spec.is_disjunction {
            return None;
        }
        // Keep the conjuncts we can form a margin for. SOUND: the unsafe region
        // is the intersection of ALL conjuncts, so proving ANY ONE supported
        // conjunct's complement over the box empties it — an unsupported sibling
        // conjunct cannot make a refuted intersection non-empty.
        let margins: Vec<Margin> = spec
            .output_constraints
            .iter()
            .filter_map(margin_for_constraint)
            .collect();
        if margins.is_empty() {
            return None;
        }
        Some(margins)
    } else {
        let idx = ctx.const_output_idx.unwrap_or(0);
        if idx >= ctx.output_dim {
            return None;
        }
        let thr = f32_to_rat(ctx.effective_threshold)?;
        // verify_upper_bound: verifier proved Y_idx <= thr -> margin = thr - Y_idx
        // else:               verifier proved Y_idx >= thr -> margin = Y_idx - thr
        let (sign, const_term) = if ctx.config.verify_upper_bound {
            (-1, thr)
        } else {
            (1, thr.neg())
        };
        Some(vec![Margin {
            coeffs: vec![(idx, sign)],
            const_term,
            strict: false,
            label: format!("epsilon-robustness output[{idx}] vs threshold"),
        }])
    }
}

/// The complement margin for one unsafe output constraint: `margin > 0` ⟹ the
/// constraint cannot hold anywhere in the box ⟹ this conjunct of the unsafe
/// region is empty.
fn margin_for_constraint(c: &OutputConstraint) -> Option<Margin> {
    let (coeffs, const_term, label): (Vec<(usize, i32)>, Rat, String) = match *c {
        // unsafe Y_i >= k  -> prove Y_i < k : margin = k - Y_i
        OutputConstraint::GreaterEqConst(i, k) | OutputConstraint::GreaterThanConst(i, k) => {
            (vec![(i, -1)], f64_to_rat(k)?, format!("Y_{i} < {k}"))
        }
        // unsafe Y_i <= k  -> prove Y_i > k : margin = Y_i - k
        OutputConstraint::LessEqConst(i, k) | OutputConstraint::LessThanConst(i, k) => {
            (vec![(i, 1)], f64_to_rat(k)?.neg(), format!("Y_{i} > {k}"))
        }
        // unsafe Y_i <= Y_j -> prove Y_i > Y_j : margin = Y_i - Y_j
        OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
            (vec![(i, 1), (j, -1)], Rat::ZERO, format!("Y_{i} > Y_{j}"))
        }
        // unsafe Y_i >= Y_j -> prove Y_i < Y_j : margin = Y_j - Y_i
        OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
            (vec![(j, 1), (i, -1)], Rat::ZERO, format!("Y_{i} < Y_{j}"))
        }
        // #[non_exhaustive]: any future constraint kind is conservatively skipped.
        _ => return None,
    };
    Some(Margin {
        coeffs,
        const_term,
        strict: true,
        label,
    })
}

/// Build the exact-rational [`DeepReluProblem`] whose scalar read-out is the
/// given `margin` (a signed linear combination of the network's outputs plus a
/// constant). Returns `Err(reason)` (non-fatal) on any conversion miss.
fn build_problem_for_margin(
    stack: &FcReluStack<'_>,
    input_lower: &[f32],
    input_upper: &[f32],
    margin: &Margin,
) -> Result<DeepReluProblem, String> {
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

    // Read-out: out_weight[h] = Σ_k sign_k · finalW[idx_k][h];
    //           out_bias      = Σ_k sign_k · finalB[idx_k] + const_term.
    let (rw, rb) = stack.readout;
    let hidden_width = rw.ncols();
    let mut out_weight = vec![Rat::ZERO; hidden_width];
    let mut out_bias = margin.const_term;
    let badrat = |e: ny_cert::RatError| format!("exact arithmetic: {e}");
    for &(idx, sign) in &margin.coeffs {
        if idx >= rw.nrows() {
            return Err(format!(
                "readout has {} rows, output idx {}",
                rw.nrows(),
                idx
            ));
        }
        let s = Rat::from_int(i128::from(sign));
        for (h, &wv) in rw.row(idx).iter().enumerate() {
            let wr = f32_to_rat(wv).ok_or("readout weight not representable")?;
            out_weight[h] = out_weight[h]
                .add(s.mul(wr).map_err(badrat)?)
                .map_err(badrat)?;
        }
        if let Some(b) = rb {
            let br = f32_to_rat(b[idx]).ok_or("readout bias not representable")?;
            out_bias = out_bias.add(s.mul(br).map_err(badrat)?).map_err(badrat)?;
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

/// Emit a proof-carrying certificate sidecar for a `Verified` verdict, if
/// enabled by `ProofOpts` and the network/property are eligible.
///
/// This is intentionally **infallible from the caller's perspective**: every
/// failure path logs and returns, so it can never abort a verification run or
/// change the verdict.
pub(super) fn maybe_emit_certificate(
    ctx: &DispatchContext<'_>,
    result: &BetaCrownResult,
    overall_deadline: Option<std::time::Instant>,
) {
    // Gate 1: only for a Verified verdict.
    if !matches!(result.result, BabVerificationStatus::Verified) {
        return;
    }
    // Gate 2: proof/cert features must be enabled (default on; off in
    // competition mode).
    if !ctx.proof_opts.should_emit_certificate() {
        return;
    }
    // Gate 2b (#3328): the certificate is an OPTIONAL post-verdict sidecar, but
    // exact-rational deep-CROWN can run for tens of seconds on adversarial nets
    // (e.g. acasxu's 6-layer unstable-ReLU stack, whose interned rationals grow
    // to thousands of bits). It must NEVER push the CLI past the caller's overall
    // `--timeout`. If too little of that budget remains to finish, skip emission
    // outright — the verdict is already decided and unaffected. When some budget
    // does remain, the emission deadline below is additionally clamped to it.
    const CERT_MIN_REMAINING: std::time::Duration = std::time::Duration::from_secs(2);
    if let Some(dl) = overall_deadline {
        let remaining = dl.checked_duration_since(std::time::Instant::now());
        if remaining.map_or(true, |r| r < CERT_MIN_REMAINING) {
            info!(
                "cert: skipped (under {}s of the overall --timeout budget remains; \
                 verdict unaffected)",
                CERT_MIN_REMAINING.as_secs()
            );
            return;
        }
    }
    // Gate 3: sequential FC-ReLU only. `model_net` is a `&mut` field behind a
    // shared `&DispatchContext`, so reborrow it as shared (`&*`) to read it.
    let network = match &*ctx.model_net {
        BetaCrownModel::Sequential(net) => net.as_ref(),
        BetaCrownModel::Graph(_) => {
            info!("cert: skipped (DAG/graph model not yet supported by the exact-cert adapter)");
            return;
        }
    };
    let stack = match extract_fc_relu_stack(network) {
        Ok(s) => s,
        Err(reason) => {
            info!("cert: skipped (ineligible network: {reason})");
            return;
        }
    };
    let input_lower = ctx.input.lower();
    let input_upper = ctx.input.upper();
    let (Some(lo), Some(hi)) = (input_lower.as_slice(), input_upper.as_slice()) else {
        info!("cert: skipped (non-contiguous input bounds)");
        return;
    };

    // Gate 4: derive the candidate safety margins from the property. For a
    // conjunctive unsafe region, certifying ANY ONE margin > 0 proves safety.
    let Some(margins) = unsafe_complement_margins(ctx) else {
        info!("cert: skipped (disjunctive / unsupported property for the Phase-A adapter)");
        return;
    };

    // Wall-clock budget for the WHOLE emission pass (all margins). The verdict
    // is already decided and a certificate is an optional sidecar, but exact
    // rational deep-CROWN can blow up on adversarial magnitudes (observed:
    // acasxu max-diff margins accumulating million-bit rationals stalled the
    // CLI indefinitely AFTER verification finished — the verdict never
    // printed). Bound the post-verdict work and fail open to "no certificate".
    // NY_CERT_BUDGET_SECS overrides the default 60s; 0 disables the cap.
    // An unparseable value falls back to the default WITH a warning (silent
    // fallback would mask an operator typo); a huge value that would overflow
    // `Instant + Duration` degrades to uncapped via `checked_add` — never a
    // panic, which (pre-verdict-print ordering aside) must not exist on an
    // infallible-by-contract path.
    let budget_env = std::env::var("NY_CERT_BUDGET_SECS").ok();
    let budget_secs = match budget_env.as_deref() {
        None => 60,
        Some(raw) => match raw.parse::<u64>() {
            Ok(v) => v,
            Err(_) => {
                warn!(
                    "cert: invalid NY_CERT_BUDGET_SECS '{raw}' (want a whole number of \
                     seconds; 0 = uncapped); using the default 60s"
                );
                60
            }
        },
    };
    // The emission deadline is the SOONER of the cert-specific budget
    // (NY_CERT_BUDGET_SECS, default 60s; 0 = uncapped) and the caller's overall
    // --timeout deadline (Gate 2b already skipped when < CERT_MIN_REMAINING
    // remained). The sidecar must never overrun the competition budget, so the
    // overall deadline binds even when the cert budget is uncapped.
    let cert_budget_deadline = if budget_secs > 0 {
        match std::time::Instant::now().checked_add(std::time::Duration::from_secs(budget_secs)) {
            Some(d) => Some(d),
            None => {
                warn!(
                    "cert: NY_CERT_BUDGET_SECS={budget_secs} overflows a deadline; \
                     bounding emission by the overall --timeout only"
                );
                None
            }
        }
    } else {
        None
    };
    let effective_deadline = match (cert_budget_deadline, overall_deadline) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };
    let _budget_guard = effective_deadline.map(ny_cert::budget::DeadlineGuard::install);

    for margin in &margins {
        let problem = match build_problem_for_margin(&stack, lo, hi, margin) {
            Ok(p) => p,
            Err(reason) => {
                info!("cert: margin '{}' skipped ({reason})", margin.label);
                continue;
            }
        };
        // certify(0): prove margin >= 0 exactly. Err means exact plain CROWN does
        // not close this conjunct (the verdict may rely on tighter alpha/beta/BaB
        // bounds) — try the next conjunct. Budget expiry is different: it bounds
        // the WHOLE emission pass, so stop entirely rather than burn the same
        // exhausted budget on every remaining margin.
        let certified = match problem.certify(Rat::ZERO) {
            Ok(c) => c,
            Err(DeepCrownError::BudgetExceeded) => {
                info!(
                    "cert: skipped (emission budget of {budget_secs}s exhausted on margin '{}'; \
                     verdict unaffected — raise NY_CERT_BUDGET_SECS or set 0 to disable the cap)",
                    margin.label
                );
                return;
            }
            // Like budget expiry, arena poisoning is a whole-thread condition:
            // every remaining margin would hit the same poisoned arena, and any
            // certificate built from it must not be trusted. Fail CLOSED for
            // the entire emission pass (the verdict itself is unaffected — a
            // certificate is an optional sidecar).
            Err(DeepCrownError::ArenaPoisoned) => {
                warn!(
                    "cert: rational arena poisoned on margin '{}' — stopping certificate \
                     emission for all margins (a fallback arm was reached; any certificate \
                     would be untrustworthy; verdict unaffected)",
                    margin.label
                );
                return;
            }
            Err(_) => continue,
        };
        // A possibly-closed unsafe conjunct (>=, <=) is only refuted by a STRICT
        // margin > 0; the denominator is always positive, so num > 0 iff > 0.
        if margin.strict && !certified.lower_bound.is_positive() {
            continue;
        }

        // In-tree self-check (entailment + Farkas replay): a cheap safety net
        // that the emitted certificate is internally valid before we write it.
        if let Err(e) = check_entailment(&certified.entailment) {
            warn!("cert: SELF-CHECK FAILED on entailment, not emitting ({e})");
            return;
        }
        if let Err(e) = check_farkas(&certified.farkas) {
            warn!("cert: SELF-CHECK FAILED on farkas, not emitting ({e})");
            return;
        }

        let entailment_json = match schema::entailment_to_json(&certified.entailment) {
            Ok(j) => j,
            Err(e) => {
                info!("cert: skipped (entailment not serialisable: {e})");
                return;
            }
        };
        let farkas_json = match schema::farkas_to_json(&certified.farkas) {
            Ok(j) => j,
            Err(e) => {
                info!("cert: skipped (farkas not serialisable: {e})");
                return;
            }
        };

        let lower_bound = match certified.lower_bound.checked_parts() {
            Ok((num, den)) => format!("{num}/{den}"),
            Err(e) => {
                warn!("cert: skipped (exact lower bound not serialisable: {e})");
                return;
            }
        };
        let payload = serde_json::json!({
            "format": "ny-cert/crown-deep/v1",
            "claim": format!(
                "exact CROWN proves '{}' over the whole input box, emptying the unsafe region",
                margin.label
            ),
            "model": ctx.model_path.display().to_string(),
            "property": ctx.property.as_ref().map(|p| p.display().to_string()),
            "discharged_margin": margin.label,
            "exact_lower_bound": lower_bound,
            "depth": problem.depth(),
            "entailment": entailment_json,
            "farkas": farkas_json,
        });

        let out_path: PathBuf = ctx
            .proof_opts
            .certificate_path
            .clone()
            .unwrap_or_else(|| ctx.model_path.with_extension("cert.json"));
        let text = match serde_json::to_string_pretty(&payload) {
            Ok(t) => t,
            Err(e) => {
                warn!("cert: failed to serialise sidecar ({e})");
                return;
            }
        };
        match std::fs::write(&out_path, text) {
            Ok(()) => info!(
                "cert: wrote machine-checkable certificate (proves {}; {} hidden layers; exact bound {}) to {}",
                margin.label,
                problem.depth(),
                lower_bound,
                out_path.display()
            ),
            Err(e) => warn!("cert: failed to write sidecar to {}: {e}", out_path.display()),
        }
        return;
    }
    info!("cert: verdict not cert-backed by exact plain CROWN (no single conjunct closed)");
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn f32_to_rat_is_exact_dyadic() {
        assert_eq!(f32_to_rat(0.0), Some(Rat::ZERO));
        assert_eq!(f32_to_rat(0.5), Rat::new(1, 2).ok());
        assert_eq!(f32_to_rat(-0.25), Rat::new(-1, 4).ok());
        assert_eq!(f32_to_rat(3.0), Rat::new(3, 1).ok());
        // f32(0.1) = 13421773 / 2^27 exactly (the nearest binary32 to 0.1).
        let r = f32_to_rat(0.1f32).expect("finite binary32 is exactly representable");
        // f32(0.1) is the already-reduced dyadic 13421773 / 2^27.
        assert_eq!(r, Rat::new(13_421_773, 134_217_728).expect("valid dyadic"));
        assert!(f32_to_rat(f32::INFINITY).is_none());
        assert!(f32_to_rat(f32::NAN).is_none());
    }

    #[test]
    fn f64_to_rat_is_exact_dyadic() {
        assert_eq!(f64_to_rat(1.0), Rat::new(1, 1).ok());
        assert_eq!(f64_to_rat(0.0), Some(Rat::ZERO));
        assert_eq!(f64_to_rat(-2.5), Rat::new(-5, 2).ok());
    }

    // End-to-end exercise of the certificate math the adapter runs: build the
    // exact-rational DeepReluProblem for a margin, certify it, and self-check
    // the entailment + Farkas certificates — on a fully-active net where CROWN
    // is exact, so the bound is deterministic.
    #[test]
    fn certifies_margin_on_active_net() {
        // y = 2 * relu(x0) + 1/2, with x0 in [1, 2] (ReLU always active) -> y in
        // [5/2, 9/2]; the exact CROWN lower bound on y is 5/2 > 0.
        let w1 = arr2(&[[1.0f32]]);
        let b1 = arr1(&[0.0f32]);
        let w2 = arr2(&[[2.0f32]]);
        let b2 = arr1(&[0.5f32]);
        let stack = FcReluStack {
            hidden: vec![(&w1, Some(&b1))],
            readout: (&w2, Some(&b2)),
        };
        // unsafe "Y_0 <= 0" -> complement margin = Y_0 (coeff +1, const 0).
        let margin = Margin {
            coeffs: vec![(0, 1)],
            const_term: Rat::ZERO,
            strict: true,
            label: "Y_0 > 0".to_string(),
        };
        let problem =
            build_problem_for_margin(&stack, &[1.0], &[2.0], &margin).expect("build problem");
        let cert = problem.certify(Rat::ZERO).expect("certify margin >= 0");
        assert_eq!(
            cert.lower_bound,
            Rat::new(5, 2).expect("5/2"),
            "exact CROWN lower bound on a fully-active net is 5/2"
        );
        assert!(cert.lower_bound.is_positive(), "margin strictly positive");
        check_entailment(&cert.entailment).expect("entailment self-check passes");
        check_farkas(&cert.farkas).expect("farkas self-check passes");
    }

    #[test]
    fn margin_for_each_constraint_kind() {
        // const + relational constraints all map to a single linear margin.
        assert!(margin_for_constraint(&OutputConstraint::GreaterEqConst(0, 1.0)).is_some());
        assert!(margin_for_constraint(&OutputConstraint::LessEqConst(1, 0.0)).is_some());
        assert!(margin_for_constraint(&OutputConstraint::LessEq(0, 1)).is_some());
        assert!(margin_for_constraint(&OutputConstraint::GreaterEq(2, 3)).is_some());
    }

    #[test]
    fn proof_opts_resolves_competition_mode() {
        use crate::commands::beta_crown::ProofOpts;
        use ny_propagate::VerificationArtifactAuthority;

        // Default: proof ON.
        let default = ProofOpts::default();
        assert!(default.should_emit_certificate());
        assert_eq!(
            default.verification_artifact_authority(),
            VerificationArtifactAuthority::CertificateExport
        );
        // Competition mode: proof OFF.
        let competition = ProofOpts {
            competition_mode: true,
            ..Default::default()
        };
        assert!(!competition.should_emit_certificate());
        assert_eq!(
            competition.verification_artifact_authority(),
            VerificationArtifactAuthority::VerdictOnly
        );
        // Explicit --emit-certificate wins over competition mode.
        let forced_export = ProofOpts {
            competition_mode: true,
            emit_certificate: Some(true),
            ..Default::default()
        };
        assert!(forced_export.should_emit_certificate());
        assert_eq!(
            forced_export.verification_artifact_authority(),
            VerificationArtifactAuthority::CertificateExport
        );
        // Explicit --no-certificate wins outside competition mode.
        let forced_verdict = ProofOpts {
            emit_certificate: Some(false),
            ..Default::default()
        };
        assert!(!forced_verdict.should_emit_certificate());
        assert_eq!(
            forced_verdict.verification_artifact_authority(),
            VerificationArtifactAuthority::VerdictOnly
        );
    }

    /// The GPU CROWN soundness gate is unconditional; compatibility flags and
    /// competition mode cannot disengage it.
    #[test]
    fn gpu_crown_gate_defaults_to_sound() {
        use crate::commands::beta_crown::ProofOpts;
        // Default interactive run: gate ON (sound).
        assert!(ProofOpts::default().sound_gpu_crown_required());
        // The removed compatibility flag cannot weaken the gate.
        assert!(ProofOpts {
            allow_unsound_gpu_crown: true,
            ..Default::default()
        }
        .sound_gpu_crown_required());
        // Competition mode is sound under the same unconditional policy.
        assert!(ProofOpts {
            competition_mode: true,
            allow_unsound_gpu_crown: true,
            ..Default::default()
        }
        .sound_gpu_crown_required());
        let error = ProofOpts {
            allow_unsound_gpu_crown: true,
            ..Default::default()
        }
        .validate()
        .expect_err("the removed unsound opt-out must fail before verification work");
        assert!(error.to_string().contains("is disabled"));
    }
}
