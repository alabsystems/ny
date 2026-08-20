// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Relational-ACAS formula-implication hardening: flip the relational `unsat`
//! soundness gate from fragile shape matching to a true formula-implication
//! check.
//!
//! The difference-network verifier proves a region `E` empty (no reachable
//! pair violates the checked bound). Authorizing `unsat` additionally needs
//! `parsed_unsafe ⊆ E` — historically attempted by fragile SHAPE matching.
//! This module proves the implication SEMANTICALLY:
//!
//! 1. The parser supplies the FULL asserted formula as an exact-rational DNF
//!    (`ny_onnx::vnnlib::dual_formula`, fail-closed extraction).
//! 2. `E` is constructed from the LITERALLY-verified claim: the exact f32
//!    box the verifier ran on, the inward-rounded `ε̂` / margin bound it
//!    enforced, exact strictness.
//! 3. `parsed ⇒ E` decomposes into finitely many QFLRA infeasibility
//!    obligations: for every parsed DNF clause `C` and every disjunct `N` of
//!    `¬E`, prove `C ∧ N` infeasible. Each obligation is an exact rational
//!    LP: ay-milp answers `Infeasible` with a Farkas certificate (verified
//!    against its own model), whose multipliers are RE-CHECKED here by the
//!    independent `ny_cert::check_farkas` over the original constraints with
//!    their TRUE strictness. Inside the LP, strict atoms are closed with
//!    their bound pulled inward by δ so strictness-only contradictions are
//!    witnessable; the ay run only PROPOSES multipliers — `check_farkas`
//!    over the true strict system is the sole soundness authority.
//! 4. A structural spot check confirms the difference network computes
//!    `f − g` at sampled points (fail-closed).
//!
//! ANY failure — missing DNF, inexact value, LP feasible/timeout, certificate
//! rejected, spot check failed — withholds the authorization token and the
//! gate stays down (`Verified → unknown`), exactly today's behavior. A wrong
//! `unsat` therefore requires BOTH a wrong α-CROWN emptiness proof AND a
//! wrong Farkas chain — the 0-wrong discipline is preserved.

use std::path::Path;

#[cfg(feature = "mip")]
use ny_cert::check_farkas;
use ny_cert::schema::{farkas_to_json, ConstraintKind, FarkasCertificate, LinearConstraint};
use ny_cert::Rat;
use ny_core::Bound;
use ny_onnx::vnnlib::dual_formula::{
    DualAtomRelation, DualFormulaDnf, DualLinearAtom, DualVar, DualVarRole, Dyadic,
};
use ny_propagate::GraphNetwork;

/// Master gate for the relational `unsat` flip. **Default ON**, tri-state:
/// `NY_RELATIONAL_UNSAT=0` is an explicit kill-switch, `=1` is explicit-on,
/// and unset defaults ON.
///
/// Category-scoped by construction: every caller of this gate lives inside
/// `run_relational_vnncomp`, which the vnncomp dispatch enters ONLY for a
/// relational category (`is_relational_category`, vnncomp.rs). Non-relational
/// categories never reach this code, so defaulting ON is byte-identical for
/// every other benchmark.
///
/// Sound by construction: this gate does NOT relax any bound — it only PERMITS
/// the certified implication lane to authorize `unsat`, and
/// `try_authorize_relational_unsat` still requires a complete, self-checked
/// Farkas proof per (clause, ¬E) pair (returns `None` otherwise). The
/// certificate is the guard; the env was only ever a staged-rollout switch.
///
/// BANKED-POINTS FIX (2026-07-19): the scored path (`run_instance.sh` →
/// `ny vnncomp`, no preset in `configs/vnncomp26/`) never set the env, so with
/// the old default-OFF every certified difference-net `Verified` was mapped to
/// `unknown` — all 36 banked isomorphic UNSATs would have scored `unknown` in
/// competition. Default-ON closes that hole with no scored-env dependency.
pub(super) fn relational_unsat_enabled() -> bool {
    !matches!(
        std::env::var("NY_RELATIONAL_UNSAT").ok().as_deref(),
        Some("0")
    )
}

/// The authorization token: produced ONLY by a complete, self-checked
/// implication proof + structural spot check. Carried into
/// `verify_difference_bounds`, where a difference-network `Verified` may
/// become `unsat` while the default-on lane is enabled. Also carries everything
/// the sidecar-v2 audit trail needs.
pub(super) struct RelationalUnsatAuth {
    /// One verified Farkas certificate per (parsed clause, ¬E disjunct) pair.
    pub pair_certs: Vec<PairCert>,
    /// JSON description of the checked region `E` (exact values, as strings).
    pub checked_region: serde_json::Value,
    /// Spot-check summary (points checked, max deviation observed).
    pub spot_check: serde_json::Value,
}

/// One implication obligation's verified certificate.
pub(super) struct PairCert {
    /// Index of the parsed DNF clause.
    pub clause: usize,
    /// Index of the `¬E` disjunct.
    pub neg_disjunct: usize,
    /// The self-checked Farkas certificate over the pair's constraints.
    pub cert: FarkasCertificate,
}

/// The region `E` the verifier LITERALLY proved empty of network-consistent
/// points: `conj ∧ (disj₁ ∨ disj₂ ∨ …)` over the formula variables. All
/// constraints are inequalities (equalities are split into `Le`+`Ge` pairs at
/// construction so negation and Farkas multipliers stay well-defined).
pub(super) struct CheckedRegion {
    conj: Vec<LinearConstraint>,
    disj: Vec<LinearConstraint>,
    description: serde_json::Value,
}

/// Canonical variable name for the Farkas layer.
fn var_name(v: DualVar) -> String {
    let prefix = match v.role {
        DualVarRole::FInput => "xf",
        DualVarRole::GInput => "xg",
        DualVarRole::FOutput => "yf",
        DualVarRole::GOutput => "yg",
    };
    format!("{prefix}_{}", v.index)
}

/// Exact conversion of the extractor's dyadic to ny-cert's `Rat`. `Rat` is
/// BigRational-backed, so arbitrarily small dyadics convert exactly — the
/// outward-rounded verifier boxes produce f32 SUBNORMAL endpoints (down to
/// 2⁻¹⁴⁹, e.g. a `0.0` bound stepped to `-1e-45`) which must survive this
/// conversion or the whole monotonic arm fails closed. The power of two is
/// applied in i128-safe chunks.
fn dyadic_to_rat(d: Dyadic) -> Option<Rat> {
    if d.mant == 0 {
        return Some(Rat::ZERO);
    }
    let mut r = Rat::new(d.mant, 1).ok()?;
    let mut remaining = i64::from(d.exp2);
    while remaining > 0 {
        let chunk = remaining.min(62);
        r = r.mul(Rat::new(1i128 << chunk, 1).ok()?).ok()?;
        remaining -= chunk;
    }
    while remaining < 0 {
        let chunk = (-remaining).min(62);
        r = r.mul(Rat::new(1, 1i128 << chunk).ok()?).ok()?;
        remaining += chunk;
    }
    Some(r)
}

/// Convert one parsed atom into inequality `LinearConstraint`s (an `Eq`
/// yields the `Le`+`Ge` pair — ny-cert's Farkas `combine` self-cancels an
/// `Eq` under a single multiplier, so equalities MUST be split to be usable).
fn atom_to_constraints(atom: &DualLinearAtom) -> Option<Vec<LinearConstraint>> {
    let mut pairs = Vec::with_capacity(atom.coeffs.len());
    for (v, c) in &atom.coeffs {
        pairs.push((var_name(*v), dyadic_to_rat(*c)?));
    }
    let refs: Vec<(&str, Rat)> = pairs.iter().map(|(n, r)| (n.as_str(), *r)).collect();
    let constant = dyadic_to_rat(atom.constant)?;
    let make = |kind| LinearConstraint::with_kind(kind, &refs, constant);
    Some(match atom.relation {
        DualAtomRelation::Le => vec![make(ConstraintKind::Le)],
        DualAtomRelation::Lt => vec![make(ConstraintKind::Lt)],
        DualAtomRelation::Ge => vec![make(ConstraintKind::Ge)],
        DualAtomRelation::Gt => vec![make(ConstraintKind::Gt)],
        DualAtomRelation::Eq => vec![make(ConstraintKind::Le), make(ConstraintKind::Ge)],
    })
}

/// Negation of an inequality (`Eq` never reaches here — split at build time).
fn negate(c: &LinearConstraint) -> Option<LinearConstraint> {
    let kind = match c.kind {
        ConstraintKind::Le => ConstraintKind::Gt,
        ConstraintKind::Lt => ConstraintKind::Ge,
        ConstraintKind::Ge => ConstraintKind::Lt,
        ConstraintKind::Gt => ConstraintKind::Le,
        ConstraintKind::Eq => return None,
    };
    Some(LinearConstraint {
        kind,
        coefficients: c.coefficients.clone(),
        constant: c.constant,
    })
}

/// Helper: `Σ coeffs·vars ⋈ k` from `(name, ±1)` pairs and an exact-f32 `k`.
fn simple_constraint(
    kind: ConstraintKind,
    terms: &[(&str, i8)],
    k: f32,
) -> Option<LinearConstraint> {
    let constant = Rat::from_f32_exact(k)?;
    let pairs: Vec<(&str, Rat)> = terms
        .iter()
        .map(|(n, s)| (*n, if *s >= 0 { Rat::ONE } else { Rat::ONE.neg() }))
        .collect();
    Some(LinearConstraint::with_kind(kind, &pairs, constant))
}

/// `E` for the ISOMORPHIC arm — the literally-verified claim:
/// the shared-input difference network was run over the exact f32 box
/// `input_bounds` and proved `|Y_f[i] − Y_g[i]| ≤ ε̂` for every output. The
/// region proved empty of network-consistent points is therefore
///   conj:  `X_f[i] ∈ box_i` and `X_f[i] == X_g[i]` (the shared input), and
///   disj:  `Y_g[i] − Y_f[i] > ε̂` or `Y_g[i] − Y_f[i] < −ε̂` for some `i`.
pub(super) fn checked_region_isomorphic(
    input_bounds: &[Bound],
    eps_hat: f32,
    output_dim: usize,
) -> Option<CheckedRegion> {
    if !eps_hat.is_finite() || eps_hat < 0.0 || output_dim == 0 {
        return None;
    }
    let mut conj = Vec::new();
    for (i, b) in input_bounds.iter().enumerate() {
        let xf = format!("xf_{i}");
        let xg = format!("xg_{i}");
        conj.push(simple_constraint(
            ConstraintKind::Ge,
            &[(xf.as_str(), 1)],
            b.lower(),
        )?);
        conj.push(simple_constraint(
            ConstraintKind::Le,
            &[(xf.as_str(), 1)],
            b.upper(),
        )?);
        // Shared input: X_f == X_g, split into the Le+Ge pair.
        conj.push(simple_constraint(
            ConstraintKind::Le,
            &[(xf.as_str(), 1), (xg.as_str(), -1)],
            0.0,
        )?);
        conj.push(simple_constraint(
            ConstraintKind::Ge,
            &[(xf.as_str(), 1), (xg.as_str(), -1)],
            0.0,
        )?);
    }
    let mut disj = Vec::new();
    for i in 0..output_dim {
        let yg = format!("yg_{i}");
        let yf = format!("yf_{i}");
        // Verified |Y_g − Y_f| ≤ ε̂ ⇒ impossible: (Y_g − Y_f) > ε̂ or < −ε̂.
        disj.push(simple_constraint(
            ConstraintKind::Gt,
            &[(yg.as_str(), 1), (yf.as_str(), -1)],
            eps_hat,
        )?);
        disj.push(simple_constraint(
            ConstraintKind::Lt,
            &[(yg.as_str(), 1), (yf.as_str(), -1)],
            -eps_hat,
        )?);
    }
    let description = serde_json::json!({
        "kind": "isomorphic-shared-input",
        "eps_hat": format!("{eps_hat:?}"),
        "input_box": input_bounds.iter().map(|b| [format!("{:?}", b.lower()), format!("{:?}", b.upper())]).collect::<Vec<_>>(),
        "output_dim": output_dim,
    });
    Some(CheckedRegion {
        conj,
        disj,
        description,
    })
}

/// `E` for the MONOTONIC arm — the literally-verified claim: the coupled
/// difference network (inputs `[xg0, delta, x1..x4]` over the exact f32 box
/// `diff_input_bounds`, `xf0 = xg0 + delta`) proved
/// `Y_f[out] − Y_g[out] ≥ lb`. The region proved empty is therefore
///   conj:  `X_g[0] ∈ box₀`, `X_f[0] − X_g[0] ∈ delta-box` (LITERAL, the
///          outward-rounded box the net ran on),
///          `X_f[k] ∈ boxₖ` and `X_f[k] == X_g[k]` for the shared `k ≥ 1`,
///   disj:  `Y_f[out] − Y_g[out] < lb` (a single disjunct).
pub(super) fn checked_region_monotonic(
    diff_input_bounds: &[Bound],
    output: usize,
    lb: f32,
) -> Option<CheckedRegion> {
    // Layout per build_monotonic_difference_network: [xg0, delta, x1, x2, x3, x4].
    if diff_input_bounds.len() < 3 || !lb.is_finite() {
        return None;
    }
    let xg0_box = diff_input_bounds[0];
    let delta_box = diff_input_bounds[1];
    let mut conj = vec![
        simple_constraint(ConstraintKind::Ge, &[("xg_0", 1)], xg0_box.lower())?,
        simple_constraint(ConstraintKind::Le, &[("xg_0", 1)], xg0_box.upper())?,
        // xf0 − xg0 = delta ∈ the LITERAL delta box the verifier ran on. The
        // production box is OUTWARD-rounded (its lower is typically the f32
        // subnormal below 0, not 0 itself); using the literal endpoints keeps E
        // exactly the verified region. A lower > 0 would merely shrink E and the
        // implication proof would refuse on its own — no shape guard needed.
        simple_constraint(
            ConstraintKind::Ge,
            &[("xf_0", 1), ("xg_0", -1)],
            delta_box.lower(),
        )?,
        simple_constraint(
            ConstraintKind::Le,
            &[("xf_0", 1), ("xg_0", -1)],
            delta_box.upper(),
        )?,
    ];
    for (k, b) in diff_input_bounds.iter().enumerate().skip(2) {
        let idx = k - 1; // diff input k maps to shared formula index k-1
        let xf = format!("xf_{idx}");
        let xg = format!("xg_{idx}");
        conj.push(simple_constraint(
            ConstraintKind::Ge,
            &[(xf.as_str(), 1)],
            b.lower(),
        )?);
        conj.push(simple_constraint(
            ConstraintKind::Le,
            &[(xf.as_str(), 1)],
            b.upper(),
        )?);
        conj.push(simple_constraint(
            ConstraintKind::Le,
            &[(xf.as_str(), 1), (xg.as_str(), -1)],
            0.0,
        )?);
        conj.push(simple_constraint(
            ConstraintKind::Ge,
            &[(xf.as_str(), 1), (xg.as_str(), -1)],
            0.0,
        )?);
    }
    let yf = format!("yf_{output}");
    let yg = format!("yg_{output}");
    // Verified Y_f − Y_g ≥ lb ⇒ impossible: Y_f − Y_g < lb.
    let disj = vec![simple_constraint(
        ConstraintKind::Lt,
        &[(yf.as_str(), 1), (yg.as_str(), -1)],
        lb,
    )?];
    let description = serde_json::json!({
        "kind": "monotonic-coupled",
        "lb": format!("{lb:?}"),
        "output": output,
        "diff_input_box": diff_input_bounds.iter().map(|b| [format!("{:?}", b.lower()), format!("{:?}", b.upper())]).collect::<Vec<_>>(),
    });
    Some(CheckedRegion {
        conj,
        disj,
        description,
    })
}

/// The disjuncts of `¬E`: one per negated conjunct atom, plus (when `disj`
/// is non-empty) the single conjunction of all negated disjunct atoms.
fn negated_region(e: &CheckedRegion) -> Option<Vec<Vec<LinearConstraint>>> {
    let mut out = Vec::with_capacity(e.conj.len() + 1);
    for c in &e.conj {
        out.push(vec![negate(c)?]);
    }
    if !e.disj.is_empty() {
        let mut all = Vec::with_capacity(e.disj.len());
        for d in &e.disj {
            all.push(negate(d)?);
        }
        out.push(all);
    }
    Some(out)
}

/// Prove `parsed ⇒ E`: every (clause, ¬E-disjunct) pair infeasible, each
/// backed by an ay-produced, `check_farkas`-re-checked certificate. `None`
/// on ANY miss (gate stays down). Only compiled with the `mip` feature (the
/// ay lane); without it the authorization is never granted.
pub(super) fn prove_parsed_implies_checked(
    dnf: &DualFormulaDnf,
    e: &CheckedRegion,
    per_pair_timeout_secs: f64,
) -> Option<Vec<PairCert>> {
    let neg = negated_region(e)?;
    let mut certs = Vec::with_capacity(dnf.clauses.len() * neg.len());
    for (ci, clause) in dnf.clauses.iter().enumerate() {
        // Parsed clause → inequality constraints (Eq split into Le+Ge).
        let mut base: Vec<LinearConstraint> = Vec::new();
        for atom in clause {
            base.extend(atom_to_constraints(atom)?);
        }
        if base.is_empty() {
            return None; // an unconstrained clause can never imply E
        }
        for (ni, n) in neg.iter().enumerate() {
            let mut system = base.clone();
            system.extend(n.iter().cloned());
            let cert = prove_system_infeasible(&system, per_pair_timeout_secs)?;
            certs.push(PairCert {
                clause: ci,
                neg_disjunct: ni,
                cert,
            });
        }
    }
    Some(certs)
}

/// Prove one conjunction of linear inequalities infeasible: exact LP via
/// ay-milp (strict atoms δ-tightened to closed rows — a multiplier
/// PROPOSAL), then mapped back and RE-CHECKED by `ny_cert::check_farkas`
/// over the TRUE (strictness-preserving) constraints, which alone decides.
#[cfg(feature = "mip")]
fn prove_system_infeasible(
    system: &[LinearConstraint],
    timeout_secs: f64,
) -> Option<FarkasCertificate> {
    use std::collections::BTreeMap;

    use ny_mip::ir::MilpProblem;
    use ny_mip::RowSide;

    // Variable name → free LP column.
    let mut problem = MilpProblem::new();
    let mut cols: BTreeMap<String, ny_mip::ir::Col> = BTreeMap::new();
    for c in system {
        for name in c.coefficients.keys() {
            cols.entry(name.clone())
                .or_insert_with(|| problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY));
        }
    }
    // One one-sided row per constraint, in order (row i == system[i]).
    // Le/Lt → upper-bounded row; Ge/Gt → lower-bounded row.
    //
    // STRICT rows are TIGHTENED by one f64 ulp (`next_down`/`next_up` of the
    // exact bound — always representable, even for the subnormal endpoints
    // the outward-rounded verifier boxes produce) so the closed LP can
    // witness infeasibilities that hold ONLY by strictness — e.g. the
    // ubiquitous boundary pair `x ≥ b ∧ x < b`, whose pure closure has the
    // feasible touching point `x = b`. The tightened system is a SUBSET of
    // the strict region, so ay's verdict on it is NOT itself the proof: it
    // is a multiplier PROPOSAL. The sole soundness authority is
    // `check_farkas` below, which re-validates the combination against the
    // TRUE constraints with their TRUE strictness — a genuinely strict
    // boundary contradiction combines to constant exactly 0 with a strict
    // row positively weighted (accepted), while a thin-but-FEASIBLE system
    // combines to a strictly positive constant (rejected).
    for c in system {
        let mut coeffs = Vec::with_capacity(c.coefficients.len());
        for (name, r) in &c.coefficients {
            let f = rat_to_exact_f64(*r)?;
            coeffs.push((cols[name], f));
        }
        let k = rat_to_exact_f64(c.constant)?;
        match c.kind {
            ConstraintKind::Le => {
                problem.add_row(f64::NEG_INFINITY, k, coeffs);
            }
            ConstraintKind::Lt => {
                let k = k.next_down();
                if !k.is_finite() {
                    return None;
                }
                problem.add_row(f64::NEG_INFINITY, k, coeffs);
            }
            ConstraintKind::Ge => {
                problem.add_row(k, f64::INFINITY, coeffs);
            }
            ConstraintKind::Gt => {
                let k = k.next_up();
                if !k.is_finite() {
                    return None;
                }
                problem.add_row(k, f64::INFINITY, coeffs);
            }
            ConstraintKind::Eq => return None, // split upstream; never here
        }
    }

    let rows = ny_mip::prove_infeasible_with_row_farkas(&problem, timeout_secs).ok()??;

    // Map row multipliers back to constraints. Only the ACTIVE side of each
    // one-sided row is legitimate; anything else fails closed.
    let mut multipliers: Vec<Rat> = vec![Rat::ZERO; system.len()];
    for (row, side, coeff) in rows {
        let c = system.get(row)?;
        let side_ok = matches!(
            (c.kind, side),
            (ConstraintKind::Le | ConstraintKind::Lt, RowSide::Upper)
                | (ConstraintKind::Ge | ConstraintKind::Gt, RowSide::Lower)
        );
        if !side_ok {
            return None;
        }
        let m = Rat::from_bigints(coeff.numer().clone(), coeff.denom().clone()).ok()?;
        multipliers[row] = multipliers[row].add(m).ok()?;
    }
    // Keep only the positively-multiplied constraints (zero-multiplied entries
    // are inert for check_farkas but dropping them keeps the sidecar minimal).
    let mut constraints = Vec::new();
    let mut kept_multipliers = Vec::new();
    for (c, m) in system.iter().zip(multipliers) {
        if m.is_positive() {
            constraints.push(c.clone());
            kept_multipliers.push(m);
        }
    }
    if constraints.is_empty() {
        return None;
    }
    let cert = FarkasCertificate {
        constraints,
        multipliers: kept_multipliers,
    };
    // INDEPENDENT re-check with the true strictness. Any rejection → decline.
    check_farkas(&cert).ok()?;
    Some(cert)
}

/// Without the `mip` feature the ay exact-LP lane is unavailable: the
/// implication can never be proven and the gate stays down.
#[cfg(not(feature = "mip"))]
fn prove_system_infeasible(
    _system: &[LinearConstraint],
    _timeout_secs: f64,
) -> Option<FarkasCertificate> {
    None
}

/// A `Rat` that is EXACTLY representable as f64 (so the LP model is the
/// exact system). `None` otherwise (fail-closed).
#[cfg(feature = "mip")]
fn rat_to_exact_f64(r: Rat) -> Option<f64> {
    let f = r.to_f64_approx();
    if !f.is_finite() {
        return None;
    }
    // Round-trip: the f64's exact dyadic value must equal the rational.
    let back = Dyadic::from_f64(f)?;
    let back_rat = dyadic_to_rat(back)?;
    if back_rat == r {
        Some(f)
    } else {
        None
    }
}

// ===========================================================================
// Difference-network structural spot check
// ===========================================================================

/// How the difference network wires the two originals, for the spot check.
pub(super) enum DiffWiring {
    /// Shared input: `h(x) = f(x) − g(x)`.
    SharedInput,
    /// Monotonic coupling: diff input `[xg0, delta, x1..]`,
    /// `xf = (xg0+delta, x1..)`, `xg = (xg0, x1..)`, `h = f(xf) − g(xg)`.
    MonotonicCoupled,
}

/// Forward a concrete point through a graph (point-IBP center).
fn forward_point(graph: &GraphNetwork, values: &[f32]) -> Option<Vec<f32>> {
    let arr =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[values.len()]), values.to_vec()).ok()?;
    let pt = ny_tensor::BoundedTensor::concrete(arr).ok()?;
    let out = graph.propagate_concrete_point(&pt, None, None).ok()?;
    Some(out.center().iter().copied().collect())
}

/// Confirm the difference network computes `f − g` at `n` deterministic
/// sample points of its input box (fail-closed on any deviation beyond a
/// small f32-noise tolerance, or any forward failure). This closes the
/// remaining structural assumption of the implication proof: that the
/// α-CROWN claim was about the function `f − g` under the stated coupling.
pub(super) fn spot_check_difference_net(
    diff: &GraphNetwork,
    f_net: &GraphNetwork,
    g_net: &GraphNetwork,
    input_bounds: &[Bound],
    wiring: &DiffWiring,
    n: usize,
) -> Option<serde_json::Value> {
    const TOL: f32 = 1e-3;
    let mut rng: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut rand01 = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as f32) / ((1u32 << 24) as f32)
    };
    let mut max_dev = 0.0f32;
    for _ in 0..n {
        let x: Vec<f32> = input_bounds
            .iter()
            .map(|b| b.lower() + (b.upper() - b.lower()) * rand01())
            .collect();
        let h = forward_point(diff, &x)?;
        let (xf, xg): (Vec<f32>, Vec<f32>) = match wiring {
            DiffWiring::SharedInput => (x.clone(), x.clone()),
            DiffWiring::MonotonicCoupled => {
                if x.len() < 3 {
                    return None;
                }
                let mut xf = Vec::with_capacity(x.len() - 1);
                let mut xg = Vec::with_capacity(x.len() - 1);
                xf.push(x[0] + x[1]);
                xg.push(x[0]);
                xf.extend_from_slice(&x[2..]);
                xg.extend_from_slice(&x[2..]);
                (xf, xg)
            }
        };
        let yf = forward_point(f_net, &xf)?;
        let yg = forward_point(g_net, &xg)?;
        if h.len() != yf.len() || h.len() != yg.len() {
            return None;
        }
        for i in 0..h.len() {
            let dev = (h[i] - (yf[i] - yg[i])).abs();
            if !dev.is_finite() {
                return None;
            }
            max_dev = max_dev.max(dev);
        }
    }
    if max_dev > TOL {
        return None;
    }
    Some(serde_json::json!({
        "points": n,
        "max_abs_deviation": format!("{max_dev:e}"),
        "tolerance": format!("{TOL:e}"),
    }))
}

/// Which literally-verified claim `E` should be built from (the caller
/// supplies the EXACT values it passed to the verifier).
pub(super) enum CheckedKind {
    /// Shared-input epsilon equivalence: `|Y_f − Y_g| ≤ eps_hat` proved on
    /// every of `output_dim` outputs.
    Isomorphic { eps_hat: f32, output_dim: usize },
    /// Coupled monotonicity: `Y_f[output] − Y_g[output] ≥ lb` proved.
    Monotonic { output: usize, lb: f32 },
}

/// Attempt the FULL authorization: env gate → parsed DNF present →
/// checked-region construction from the literal claim → difference-net
/// structural spot check → clause-by-clause implication proof with verified
/// Farkas certificates. `None` on ANY miss (the caller's gate stays down).
pub(super) fn try_authorize_relational_unsat(
    dnf: Option<&DualFormulaDnf>,
    kind: CheckedKind,
    diff: &GraphNetwork,
    f_net: &GraphNetwork,
    g_net: &GraphNetwork,
    input_bounds: &[Bound],
) -> Option<RelationalUnsatAuth> {
    if !relational_unsat_enabled() {
        return None;
    }
    let Some(dnf) = dnf else {
        eprintln!(
            "relational unsat: parsed formula DNF unavailable (extraction failed closed); \
             gate stays down"
        );
        return None;
    };
    let (e, wiring) = match kind {
        CheckedKind::Isomorphic {
            eps_hat,
            output_dim,
        } => (
            checked_region_isomorphic(input_bounds, eps_hat, output_dim)?,
            DiffWiring::SharedInput,
        ),
        CheckedKind::Monotonic { output, lb } => (
            checked_region_monotonic(input_bounds, output, lb)?,
            DiffWiring::MonotonicCoupled,
        ),
    };
    let Some(spot_check) = spot_check_difference_net(diff, f_net, g_net, input_bounds, &wiring, 8)
    else {
        eprintln!("relational unsat: difference-net structural spot check FAILED; gate stays down");
        return None;
    };
    const PER_PAIR_TIMEOUT_SECS: f64 = 5.0;
    let Some(pair_certs) = prove_parsed_implies_checked(dnf, &e, PER_PAIR_TIMEOUT_SECS) else {
        eprintln!(
            "relational unsat: formula implication NOT proven ({} clauses); gate stays down",
            dnf.clauses.len()
        );
        return None;
    };
    println!(
        "relational unsat AUTHORIZED: parsed formula ({} clauses, {} asserts) implies the \
         verified region — {} Farkas pair-certificates self-checked",
        dnf.clauses.len(),
        dnf.num_asserts,
        pair_certs.len()
    );
    Some(RelationalUnsatAuth {
        pair_certs,
        checked_region: e.description,
        spot_check,
    })
}

// ===========================================================================
// Sidecar v2
// ===========================================================================

/// Serialize the implication proof + checked-region claim + spot check to the
/// v2 sidecar next to the VNN-LIB file. Best-effort: failure never changes
/// the (already-authorized) verdict.
pub(super) fn write_implication_sidecar(
    auth: &RelationalUnsatAuth,
    vnnlib: &Path,
    sidecar_path: &Path,
) -> Result<(), String> {
    let mut entries = Vec::with_capacity(auth.pair_certs.len());
    for pc in &auth.pair_certs {
        let json = farkas_to_json(&pc.cert).map_err(|e| {
            format!(
                "pair (clause {}, neg-disjunct {}) not serialisable: {e}",
                pc.clause, pc.neg_disjunct
            )
        })?;
        entries.push(serde_json::json!({
            "clause": pc.clause,
            "neg_disjunct": pc.neg_disjunct,
            "farkas": json,
        }));
    }
    let payload = serde_json::json!({
        "format": "ny-cert/relational-formula-implication/v2",
        "claim":
            "every clause of the parsed VNN-LIB unsafe-region DNF implies the region E the \
             difference-network verifier proved empty of network-consistent points: for every \
             (clause, ¬E-disjunct) pair, a non-negative Farkas combination of the pair's exact \
             linear atoms yields a contradiction, so parsed_unsafe ⊆ E and the verified \
             emptiness of E entails unsat",
        "vnnlib": vnnlib.display().to_string(),
        "checked_region": auth.checked_region,
        "difference_net_spot_check": auth.spot_check,
        "implication_certificates": entries,
    });
    let text = serde_json::to_string_pretty(&payload)
        .map_err(|e| format!("failed to serialise sidecar: {e}"))?;
    std::fs::write(sidecar_path, text)
        .map_err(|e| format!("failed to write sidecar to {}: {e}", sidecar_path.display()))?;
    println!(
        "Wrote relational formula-implication certificate ({} pair proofs) to {}",
        auth.pair_certs.len(),
        sidecar_path.display()
    );
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only atom builder: `Σ coeff·var ⋈ k` from f64 literals (each an
    /// exact dyadic, same semantics the extractor gives real files).
    #[cfg(feature = "mip")]
    fn atom(
        relation: DualAtomRelation,
        terms: &[(DualVarRole, usize, f64)],
        k: f64,
    ) -> DualLinearAtom {
        DualLinearAtom {
            relation,
            coeffs: terms
                .iter()
                .map(|(role, index, c)| {
                    (
                        DualVar {
                            role: *role,
                            index: *index,
                        },
                        Dyadic::from_f64(*c).expect("finite test literal"),
                    )
                })
                .collect(),
            constant: Dyadic::from_f64(k).expect("finite test literal"),
        }
    }

    /// The largest f32 that is <= the f64 `x` (the inward rounding the real
    /// pipeline applies to epsilon before building E).
    #[cfg(feature = "mip")]
    fn inward_f32(x: f64) -> f32 {
        let mut e = x as f32;
        // Exact ULP bit-walk: the f64>f64 compare is the intended exact
        // termination test (decrement e until f64::from(e) <= x), not an
        // approximate/epsilon compare.
        #[allow(clippy::while_float)]
        while f64::from(e) > x {
            e = f32::from_bits(e.to_bits() - 1);
        }
        e
    }

    /// The clauses of the real isomorphic-ACAS formula shape, shrunk to
    /// `in_dim` inputs / `out_dim` outputs: box on X_f, X_f == X_g coupling,
    /// and per output the two one-sided deviation atoms (strict, eps = 0.05
    /// as an f64 literal — exactly what the 2026 files assert).
    #[cfg(feature = "mip")]
    fn iso_dnf(in_dim: usize, out_dim: usize, eps: f64) -> DualFormulaDnf {
        use DualAtomRelation::{Eq, Ge, Gt, Le, Lt};
        use DualVarRole::{FInput, FOutput, GInput, GOutput};
        let mut shared = Vec::new();
        for i in 0..in_dim {
            shared.push(atom(Ge, &[(FInput, i, 1.0)], -1.0));
            shared.push(atom(Le, &[(FInput, i, 1.0)], 1.0));
            shared.push(atom(Eq, &[(FInput, i, 1.0), (GInput, i, -1.0)], 0.0));
        }
        let mut clauses = Vec::new();
        for i in 0..out_dim {
            for (rel, sign) in [(Gt, 1.0), (Lt, -1.0)] {
                let mut clause = shared.clone();
                clause.push(atom(
                    rel,
                    &[(GOutput, i, 1.0), (FOutput, i, -1.0)],
                    sign * eps,
                ));
                clauses.push(clause);
            }
        }
        DualFormulaDnf {
            num_asserts: 2 * in_dim + 1,
            clauses,
        }
    }

    #[test]
    fn negate_flips_and_rejects_eq() {
        let le = simple_constraint(ConstraintKind::Le, &[("x", 1)], 1.0).unwrap();
        assert_eq!(negate(&le).unwrap().kind, ConstraintKind::Gt);
        let ge = simple_constraint(ConstraintKind::Ge, &[("x", 1)], 1.0).unwrap();
        assert_eq!(negate(&ge).unwrap().kind, ConstraintKind::Lt);
        let eq = simple_constraint(ConstraintKind::Eq, &[("x", 1)], 1.0).unwrap();
        assert!(negate(&eq).is_none(), "Eq must be split before negation");
    }

    #[test]
    fn dyadic_to_rat_is_exact() {
        let d = Dyadic::from_f64(0.75).unwrap();
        assert_eq!(dyadic_to_rat(d).unwrap(), Rat::new(3, 4).unwrap());
        let d = Dyadic::from_f64(-0.05).unwrap();
        // -0.05 as f64 is NOT -1/20; the conversion must preserve the dyadic.
        assert_ne!(dyadic_to_rat(d).unwrap(), Rat::new(-1, 20).unwrap());
        // f32 subnormals (outward-rounded 0.0 endpoints) must convert: the
        // denominator 2^149 exceeds i128 and needs the chunked BigRational
        // path on BOTH conversion lanes (constraint build + f64 round-trip).
        let sub = f32::from_bits(1); // 2^-149 == 1e-45
        let via_dyadic = dyadic_to_rat(Dyadic::from_f64(f64::from(sub)).unwrap()).unwrap();
        assert_eq!(via_dyadic, Rat::from_f32_exact(sub).unwrap());
    }

    #[test]
    fn checked_region_guards_fail_closed() {
        let b = [Bound::new(-1.0, 1.0)];
        assert!(checked_region_isomorphic(&b, -0.01, 1).is_none());
        assert!(checked_region_isomorphic(&b, f32::NAN, 1).is_none());
        assert!(checked_region_isomorphic(&b, 0.05, 0).is_none());
        // Monotonic layout requires [xg0, delta, shared..] and a finite lb.
        let short = [Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
        assert!(checked_region_monotonic(&short, 0, 0.0).is_none());
        let ok = [
            Bound::new(0.0, 1.0),
            Bound::new(0.0, 1.0),
            Bound::new(-1.0, 1.0),
        ];
        assert!(checked_region_monotonic(&ok, 0, f32::NAN).is_none());
        // Delta endpoints are used LITERALLY (production boxes are outward-
        // rounded to subnormals around 0); a shifted delta box still builds —
        // the implication proof is what refuses a mismatched region.
        let shifted = [
            Bound::new(0.0, 1.0),
            Bound::new(0.5, 1.0),
            Bound::new(-1.0, 1.0),
        ];
        assert!(checked_region_monotonic(&shifted, 0, 0.0).is_some());
    }

    #[test]
    fn negated_region_shape() {
        let b = [Bound::new(-1.0, 1.0)];
        let e = checked_region_isomorphic(&b, 0.05, 2).unwrap();
        // conj: 2 box + 2 coupling atoms; disj: 4 deviation atoms.
        let neg = negated_region(&e).unwrap();
        assert_eq!(neg.len(), 4 + 1);
        for d in &neg[..4] {
            assert_eq!(d.len(), 1);
        }
        assert_eq!(neg[4].len(), 4, "all negated disjunct atoms conjoined");
    }

    #[cfg(feature = "mip")]
    #[test]
    fn iso_implication_proved_with_inward_eps() {
        // eps_hat = largest f32 <= the parsed f64 0.05 (the real pipeline's
        // inward rounding) — the implication must hold and every pair must
        // come back with a self-checked Farkas certificate.
        let dnf = iso_dnf(2, 2, 0.05);
        let bounds = [Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)];
        let e = checked_region_isomorphic(&bounds, inward_f32(0.05), 2).unwrap();
        let certs = prove_parsed_implies_checked(&dnf, &e, 5.0)
            .expect("implication must be provable with inward-rounded eps");
        // 4 clauses x (8 negated conj atoms + 1 negated-disj conjunction).
        assert_eq!(certs.len(), 4 * 9);
    }

    #[cfg(feature = "mip")]
    #[test]
    fn iso_implication_rejects_outward_eps() {
        // ADVERSARIAL: plain `0.05f32` is strictly ABOVE the f64 literal 0.05
        // the formula asserts, so E is smaller than the parsed unsafe region
        // and the implication is genuinely false (points with deviation
        // between 0.05f64 and 0.05f32 are unsafe but not in E). The driver
        // must refuse — this is exactly the outward-rounding soundness trap.
        let dnf = iso_dnf(1, 1, 0.05);
        let bounds = [Bound::new(-1.0, 1.0)];
        assert!(f64::from(0.05f32) > 0.05);
        let e = checked_region_isomorphic(&bounds, 0.05f32, 1).unwrap();
        assert!(
            prove_parsed_implies_checked(&dnf, &e, 5.0).is_none(),
            "outward-rounded eps_hat must not authorize"
        );
    }

    #[cfg(feature = "mip")]
    #[test]
    fn iso_implication_rejects_missing_coupling() {
        // ADVERSARIAL: a formula whose clauses do NOT couple X_f == X_g (the
        // shape six review rounds worried about). E asserts the coupling, so
        // parsed does not imply E and the driver must refuse.
        let mut dnf = iso_dnf(1, 1, 0.05);
        for clause in &mut dnf.clauses {
            clause.retain(|a| a.relation != DualAtomRelation::Eq);
        }
        let bounds = [Bound::new(-1.0, 1.0)];
        let e = checked_region_isomorphic(&bounds, inward_f32(0.05), 1).unwrap();
        assert!(
            prove_parsed_implies_checked(&dnf, &e, 5.0).is_none(),
            "uncoupled formula must not authorize"
        );
    }

    #[cfg(feature = "mip")]
    #[test]
    fn iso_implication_rejects_wider_parsed_box() {
        // ADVERSARIAL: the parsed box is WIDER than the box the verifier ran
        // on — the verified claim does not cover the whole unsafe region.
        let dnf = iso_dnf(1, 1, 0.05); // parsed box [-1, 1]
        let bounds = [Bound::new(-0.5, 0.5)]; // verified box smaller
        let e = checked_region_isomorphic(&bounds, inward_f32(0.05), 1).unwrap();
        assert!(
            prove_parsed_implies_checked(&dnf, &e, 5.0).is_none(),
            "under-covered input box must not authorize"
        );
    }

    /// The real monotonic-ACAS clause shape, shrunk to 3 formula inputs
    /// (varying index 0 + shared 1..2) and 2 outputs: boxes on X_f AND X_g,
    /// the `X_f[0] >= X_g[0]` ordering, equality coupling elsewhere, and the
    /// strict unsafe atom `Y_f[out] < Y_g[out]`.
    #[cfg(feature = "mip")]
    fn mono_clause(out: usize) -> DualFormulaDnf {
        use DualAtomRelation::{Eq, Ge, Le, Lt};
        use DualVarRole::{FInput, FOutput, GInput, GOutput};
        let mut clause = vec![
            atom(Ge, &[(FInput, 0, 1.0)], 0.0),
            atom(Le, &[(FInput, 0, 1.0)], 2.0),
            atom(Ge, &[(GInput, 0, 1.0)], 0.0),
            atom(Le, &[(GInput, 0, 1.0)], 1.0),
            atom(Ge, &[(FInput, 0, 1.0), (GInput, 0, -1.0)], 0.0),
        ];
        for k in 1..3 {
            clause.push(atom(Ge, &[(FInput, k, 1.0)], -1.0));
            clause.push(atom(Le, &[(FInput, k, 1.0)], 1.0));
            clause.push(atom(Eq, &[(FInput, k, 1.0), (GInput, k, -1.0)], 0.0));
        }
        clause.push(atom(Lt, &[(FOutput, out, 1.0), (GOutput, out, -1.0)], 0.0));
        DualFormulaDnf {
            clauses: vec![clause],
            num_asserts: 12,
        }
    }

    /// Difference-net input layout matching `mono_clause`:
    /// [xg0 in [0,1], delta in [0,2], x1, x2 in [-1,1]].
    #[cfg(feature = "mip")]
    fn mono_diff_bounds() -> [Bound; 4] {
        [
            Bound::new(0.0, 1.0),
            Bound::new(0.0, 2.0),
            Bound::new(-1.0, 1.0),
            Bound::new(-1.0, 1.0),
        ]
    }

    #[cfg(feature = "mip")]
    #[test]
    fn mono_implication_proved_with_positive_lb() {
        // Verified `Y_f[1] - Y_g[1] >= 0.25` over the coupled box; the parsed
        // strict unsafe atom `Y_f[1] < Y_g[1]` lands strictly inside the
        // refuted region, so every obligation is infeasible.
        let dnf = mono_clause(1);
        let e = checked_region_monotonic(&mono_diff_bounds(), 1, 0.25).unwrap();
        let certs = prove_parsed_implies_checked(&dnf, &e, 5.0)
            .expect("monotonic implication must be provable");
        // 1 clause x (12 negated conj atoms + 1 negated disjunct):
        // conj = xg0 box (2) + delta window (2) + per shared k in {1,2} the
        // xf box (2) and the == coupling split (2).
        assert_eq!(certs.len(), 13);
    }

    #[cfg(feature = "mip")]
    #[test]
    fn mono_implication_rejects_wrong_output() {
        // ADVERSARIAL: the claim was verified for output 1 but the formula's
        // unsafe atom is about output 0 — must refuse.
        let dnf = mono_clause(0);
        let e = checked_region_monotonic(&mono_diff_bounds(), 1, 0.25).unwrap();
        assert!(
            prove_parsed_implies_checked(&dnf, &e, 5.0).is_none(),
            "output mismatch must not authorize"
        );
    }

    #[cfg(feature = "mip")]
    #[test]
    fn mono_implication_rejects_reversed_ordering() {
        // ADVERSARIAL: the formula orders X_g[0] >= X_f[0] (delta <= 0) while
        // the verified coupling is X_f[0] - X_g[0] in [0, 2] — the parsed
        // region pokes out of E wherever xf0 < xg0, so no authorization.
        // (The parse-level ordering-shape rejection used to catch a cousin of
        // this; the semantic layer now refuses it directly.)
        let mut dnf = mono_clause(1);
        let ord = &mut dnf.clauses[0][4];
        assert_eq!(ord.relation, DualAtomRelation::Ge);
        ord.relation = DualAtomRelation::Le; // xf0 - xg0 <= 0
        let e = checked_region_monotonic(&mono_diff_bounds(), 1, 0.25).unwrap();
        assert!(
            prove_parsed_implies_checked(&dnf, &e, 5.0).is_none(),
            "reversed ordering must not authorize"
        );
    }

    #[cfg(feature = "mip")]
    #[test]
    fn strict_boundary_eps_is_certified() {
        // Boundary case: eps_hat EXACTLY equal to the asserted eps with a
        // strict parsed atom. The implication is genuinely true (yg-yf > e
        // contradicts yg-yf <= e) but holds ONLY by strictness — the pure
        // closure touches at yg-yf == e. The δ-tightened LP proposes the
        // multipliers and check_farkas accepts them over the true strict
        // constraints (combined constant exactly 0, strict row weighted).
        let dnf = iso_dnf(1, 1, 0.03125); // exact in f32 and f64
        let bounds = [Bound::new(-1.0, 1.0)];
        let e = checked_region_isomorphic(&bounds, 0.03125, 1).unwrap();
        assert!(prove_parsed_implies_checked(&dnf, &e, 5.0).is_some());
    }

    #[cfg(feature = "mip")]
    #[test]
    fn minimal_infeasible_system_is_certified() {
        let system = vec![
            simple_constraint(ConstraintKind::Le, &[("x", 1)], 0.0).unwrap(),
            simple_constraint(ConstraintKind::Ge, &[("x", 1)], 1.0).unwrap(),
        ];
        assert!(
            prove_system_infeasible(&system, 5.0).is_some(),
            "x<=0 && x>=1 must be provably infeasible"
        );
    }

    #[test]
    fn mono_e_builds_from_production_outward_rounded_bounds() {
        // Regression: the PRODUCTION monotonic diff box is outward-rounded,
        // while the FTZ-safe conversion preserves an exactly representable
        // zero endpoint as +0 instead of publishing a subnormal. E
        // construction must accept that literal box (real
        // monotonic_acasxu_2026 instance_0 values).
        use super::super::vnncomp::finite_bound_from_f64;
        let f: [(f64, f64); 5] = [
            (-0.16247807, 0.667245963),
            (-0.25, 0.0),
            (0.25, 0.5),
            (0.227272727, 0.227272727),
            (0.25, 0.25),
        ];
        let (g0_lower, f0_upper) = (f[0].0, f[0].1);
        let mut bounds = vec![
            finite_bound_from_f64(g0_lower, f0_upper).unwrap(),
            finite_bound_from_f64(0.0, f0_upper - g0_lower).unwrap(),
        ];
        for &(lo, hi) in f.iter().skip(1) {
            bounds.push(finite_bound_from_f64(lo, hi).unwrap());
        }
        assert_eq!(
            bounds[1].lower().to_bits(),
            0,
            "an exact zero lower endpoint stays +0"
        );
        assert!(checked_region_monotonic(&bounds, 3, 0.0).is_some());
    }

    #[cfg(feature = "mip")]
    #[test]
    fn thin_feasible_system_is_never_certified() {
        // A genuinely FEASIBLE but very thin system (x > 0 && x <= 2^-50)
        // must never come back certified, however the internal lanes route
        // it (here: the ulp-tightened LP is still feasible, ay answers Sat).
        let tiny = (2.0f32).powi(-50);
        let system = vec![
            simple_constraint(ConstraintKind::Gt, &[("x", 1)], 0.0).unwrap(),
            simple_constraint(ConstraintKind::Le, &[("x", 1)], tiny).unwrap(),
        ];
        assert!(prove_system_infeasible(&system, 5.0).is_none());
    }

    #[cfg(feature = "mip")]
    #[test]
    fn adjacent_f64_feasible_system_is_rejected_by_check_farkas() {
        // THE soundness pin for the ulp-tightening: constants one f64 ulp
        // apart, both strict (x > 1 && x < 1+ulp). Over the REALS this is
        // genuinely FEASIBLE, but the ulp-tightened closed LP is INFEASIBLE
        // (bounds cross), so ay PROPOSES a Farkas combination — and
        // check_farkas must reject it over the true constraints (combined
        // constant +ulp > 0). If this test ever fails, the proposal lane has
        // been promoted to an authority and the 0-wrong discipline is broken.
        let rat_of_f64 =
            |v: f64| dyadic_to_rat(Dyadic::from_f64(v).expect("finite")).expect("in range");
        let a = 1.0_f64;
        let b = f64::from_bits(a.to_bits() + 1);
        let terms = [("x", Rat::ONE)];
        let system = vec![
            LinearConstraint::with_kind(ConstraintKind::Gt, &terms, rat_of_f64(a)),
            LinearConstraint::with_kind(ConstraintKind::Lt, &terms, rat_of_f64(b)),
        ];
        assert!(prove_system_infeasible(&system, 5.0).is_none());
    }

    // -- difference-net structural spot check ------------------------------

    /// A 1-in/1-out graph computing `w * x`.
    fn scale_graph(w: f32) -> GraphNetwork {
        use ndarray::arr2;
        use ny_propagate::layers::LinearLayer;
        use ny_propagate::GraphNode;
        let mut g = GraphNetwork::new();
        g.try_add_node(GraphNode::from_input(
            "out",
            ny_propagate::Layer::Linear(LinearLayer::new(arr2(&[[w]]), None).unwrap()),
        ))
        .unwrap();
        g.set_output("out");
        g
    }

    #[test]
    fn spot_check_accepts_true_difference_and_rejects_wrong_one() {
        let f = scale_graph(2.0);
        let g = scale_graph(0.5);
        let good_diff = scale_graph(1.5); // 2x - 0.5x
        let bad_diff = scale_graph(2.5);
        let bounds = [Bound::new(-1.0, 1.0)];
        assert!(spot_check_difference_net(
            &good_diff,
            &f,
            &g,
            &bounds,
            &DiffWiring::SharedInput,
            8
        )
        .is_some());
        assert!(
            spot_check_difference_net(&bad_diff, &f, &g, &bounds, &DiffWiring::SharedInput, 8)
                .is_none()
        );
    }

    /// The gate is now DEFAULT-ON, tri-state: `=0` kill-switch, `=1` on,
    /// unset → on. Serialized (env is process-global) + save/restored.
    #[test]
    fn relational_unsat_gate_tri_state() {
        // Serialized + restored via the blessed env choke point (clippy env
        // wall).
        ny_test_utils::env::with_env_edits(|env| {
            env.set("NY_RELATIONAL_UNSAT", "0");
            assert!(!relational_unsat_enabled(), "=0 must be the kill-switch");
            env.set("NY_RELATIONAL_UNSAT", "1");
            assert!(relational_unsat_enabled(), "=1 must be on");
            env.remove("NY_RELATIONAL_UNSAT");
            assert!(
                relational_unsat_enabled(),
                "unset must DEFAULT ON (the scored-path banked-points fix)"
            );
        });
    }

    #[cfg(feature = "mip")]
    #[test]
    fn authorization_succeeds_on_valid_symbolic_containment() {
        // `try_authorize_relational_unsat` proves the SYMBOLIC polyhedral
        // containment parsed-DNF ⊆ checked-region E over FREE variables (yf/yg
        // are unconstrained — it never references f/g's output VALUES). Here that
        // containment genuinely HOLDS: the parsed deviation atom |yg-yf| > 0.05
        // implies E's disjunct |yg-yf| > eps_hat because eps_hat = inward_f32(0.05)
        // <= 0.05, and the boxes/coupling atoms are identical — so `try_authorize`
        // correctly returns Some with valid, exactly-`check_farkas`-verified Farkas
        // pair-certificates. (A previous version of this test asserted `.is_none()`
        // on the false premise that f-g = 1.5x >> eps means "no proof exists" — but
        // detecting that f-g EXCEEDS eps is α-CROWN's separate job, not this
        // symbolic step. In production the token is consumed only when α-CROWN
        // already returned Verified, which for this violated net it never does.)
        // The genuine soundness boundaries are covered by the adversarial pins
        // `iso_implication_rejects_outward_eps`,
        // `iso_implication_rejects_missing_coupling`,
        // `iso_implication_rejects_wider_parsed_box`, and
        // `adjacent_f64_feasible_system_is_rejected_by_check_farkas`.
        ny_test_utils::env::with_env_edits(|env| {
            // The test owns a serialized, scoped enabled state. An unrelated
            // process-level kill switch must not turn a proof regression into
            // a passing no-op, and the caller's value is restored afterwards.
            env.set("NY_RELATIONAL_UNSAT", "1");
            let f = scale_graph(2.0);
            let g = scale_graph(0.5);
            let diff = scale_graph(1.5);
            let bounds = [Bound::new(-1.0, 1.0)];
            let dnf = iso_dnf(1, 1, 0.05);
            assert!(try_authorize_relational_unsat(
                Some(&dnf),
                CheckedKind::Isomorphic {
                    eps_hat: inward_f32(0.05),
                    output_dim: 1
                },
                &diff,
                &f,
                &g,
                &bounds,
            )
            .is_some());
        });
    }
}

/// End-to-end validation against the REAL VNN-COMP 2026 relational ACAS
/// files: for every instance whose formula the extractor covers, the
/// implication `parsed ⇒ E` must be provable with the EXACT bounds the
/// production arms derive (`bounds_from_f64` / the monotonic diff layout)
/// and the inward-rounded ε̂ / a boundary `lb = 0`. These real-corpus tests
/// live only in the explicit `external-vnncomp` lane and fail actionably when
/// selected without their benchmark checkout. This is the formula half of the gate flip measured
/// on all 100 scored instances; only the α-CROWN emptiness proof remains for the
/// coordinator's live runs.
// Every item in this module — both tests and their helpers — is
// `external-vnncomp` content. Gating the MODULE rather than each import keeps
// the two in step; with `mip` alone the imports and helpers were all dead, and
// the crate failed `-D warnings` in that combination.
#[cfg(all(test, feature = "mip", feature = "external-vnncomp"))]
mod real_benchmark_e2e {
    use std::path::{Path, PathBuf};

    use ny_onnx::vnnlib::{load_vnnlib, DualNetworkProperty};

    use super::super::vnncomp::{
        bounds_from_f64, dual_difference_soundness_gate, finite_bound_from_f64,
        inward_nonnegative_f32,
    };
    use super::*;

    fn benchmark_root() -> Option<PathBuf> {
        crate::commands::vnncomp2026_benchmarks_root()
    }

    fn vnnlib_files(root: &Path, category: &str) -> Vec<PathBuf> {
        let dir = root.join(category).join("2.0/vnnlib");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Vec::new();
        };
        let mut files: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "vnnlib"))
            .collect();
        files.sort();
        files
    }

    /// SMOKE the full relational-BaB wiring on ONE real isomorphic instance
    /// with a tiny budget: the run must complete without error and return a
    /// sound verdict (never an unauthorized `unsat`). This exercises the
    /// falsifier slice, the band→clause conversion, the input-split BaB call,
    /// and the verdict mapping end-to-end on real networks.
    #[test]
    #[cfg(feature = "external-vnncomp")]
    fn relational_bab_smoke_one_real_instance() {
        let root = benchmark_root().unwrap_or_else(|| {
            panic!(
                "external VNN-COMP 2026 relational fixtures missing; run \
                 benchmarks/vnncomp2026_benchmarks/setup.sh"
            )
        });
        let base = root.join("isomorphic_acasxu_2026/2.0");
        let vnnlib = base.join("vnnlib/instance_0.vnnlib");
        let onnx_field = "[('f', 'onnx/original/ACASXU_run2a_2_4_batch_2000.onnx'), \
                          ('g', 'onnx/perturbed/ACASXU_run2a_2_4_batch_2000_perturbed_0.onnx')]";
        assert!(
            vnnlib.is_file(),
            "external relational fixture missing at {}; run \
             benchmarks/vnncomp2026_benchmarks/setup.sh",
            vnnlib.display()
        );
        let verdict = super::super::vnncomp::run_relational_vnncomp(
            "isomorphic_acasxu_2026",
            Path::new(onnx_field),
            &vnnlib,
            10,
        )
        .expect("relational BaB run must not error");
        eprintln!("relational BaB smoke verdict: {verdict:?}");
        assert!(
            !matches!(verdict, super::super::vnncomp::VnncompResult::Error),
            "smoke run must produce a sound verdict, not Error"
        );
        // The gate is DEFAULT-ON now, so an `unsat` here is legitimate — it can
        // only arise via the certified implication token. Only under the
        // explicit `=0` kill-switch is `unsat` impossible; assert that there.
        if std::env::var("NY_RELATIONAL_UNSAT").ok().as_deref() == Some("0") {
            assert!(
                !matches!(verdict, super::super::vnncomp::VnncompResult::Unsat),
                "kill-switch (=0) must forbid the unsat flip"
            );
        }
    }

    #[test]
    #[cfg(feature = "external-vnncomp")]
    fn real_2026_formulas_imply_their_checked_regions() {
        let root = benchmark_root().unwrap_or_else(|| {
            panic!(
                "external VNN-COMP 2026 relational fixtures missing; run \
                 benchmarks/vnncomp2026_benchmarks/setup.sh"
            )
        });
        let mut proven = 0usize;
        let mut total = 0usize;
        let mut failures: Vec<String> = Vec::new();
        for category in ["isomorphic_acasxu_2026", "monotonic_acasxu_2026"] {
            for file in vnnlib_files(&root, category) {
                total += 1;
                let label = format!("{category}/{}", file.file_name().unwrap().to_string_lossy());
                let spec = match load_vnnlib(&file) {
                    Ok(spec) => spec,
                    Err(e) => {
                        failures.push(format!("{label}: parse error {e}"));
                        continue;
                    }
                };
                let Some(dual) = spec.dual_network.as_ref() else {
                    failures.push(format!("{label}: no dual spec"));
                    continue;
                };
                let Some(dnf) = dual.formula_dnf.as_ref() else {
                    failures.push(format!("{label}: no formula DNF"));
                    continue;
                };
                let e = match dual.property {
                    DualNetworkProperty::EpsilonEquivalence { epsilon } => {
                        // The real files spell the complement as nested
                        // or-of-ors with arithmetic rhs — the canonical shape
                        // gate must now accept it (the live 0/8 blocker), so
                        // every iso instance reaches the verify step.
                        if let Err(reason) =
                            dual_difference_soundness_gate("isomorphic_acasxu_2026", dual)
                        {
                            failures.push(format!("{label}: shape gate declined: {reason}"));
                            continue;
                        }
                        // Mirror the isomorphic arm exactly.
                        let Ok(bounds) = bounds_from_f64(&dual.f_input_bounds) else {
                            failures.push(format!("{label}: bounds"));
                            continue;
                        };
                        checked_region_isomorphic(&bounds, inward_nonnegative_f32(epsilon), 5)
                    }
                    DualNetworkProperty::MonotonicGreaterEq { output, .. } => {
                        // Mirror build_monotonic_difference_network's layout:
                        // [xg0 in [g0_lower, f0_upper], delta in [0, f0u - g0l],
                        //  f box for 1..]. `lb = 0.0` is the WEAKEST claim a
                        // verified run can make against the strict unsafe atom
                        // (any positive proven bound only strengthens it).
                        let (_, f0_upper) = dual.f_input_bounds[0];
                        let (g0_lower, _) = dual.g_input_bounds[0];
                        let mut bounds = vec![
                            finite_bound_from_f64(g0_lower, f0_upper).unwrap(),
                            finite_bound_from_f64(0.0, f0_upper - g0_lower).unwrap(),
                        ];
                        for &(lo, hi) in dual.f_input_bounds.iter().skip(1) {
                            bounds.push(finite_bound_from_f64(lo, hi).unwrap());
                        }
                        checked_region_monotonic(&bounds, output, 0.0)
                    }
                    _ => {
                        failures.push(format!("{label}: unexpected dual property"));
                        continue;
                    }
                };
                let Some(e) = e else {
                    failures.push(format!("{label}: E construction failed"));
                    continue;
                };
                match prove_parsed_implies_checked(dnf, &e, 5.0) {
                    Some(certs) => {
                        proven += 1;
                        assert!(!certs.is_empty());
                    }
                    None => failures.push(format!("{label}: implication not proven")),
                }
            }
        }
        eprintln!("formula-implication proven: {proven}/{total} real 2026 relational instances");
        assert!(total > 0, "benchmark dir present but no instances found");
        assert!(
            failures.is_empty(),
            "implication failures:\n{}",
            failures.join("\n")
        );
    }
}

/// DIAGNOSTIC (temporary): why does the relational BaB gap stall? Measures the
/// deep-subdomain band bounds under IBP vs per-node CROWN-IBP intermediates on
/// the REAL instance_0 difference network.
// Sole test is `external-vnncomp`; gate the module so its imports cannot go
// dead under `mip` alone.
#[cfg(all(test, feature = "mip", feature = "external-vnncomp"))]
mod bab_gap_probe {

    #[test]
    #[cfg(feature = "external-vnncomp")]
    fn probe_diffnet_intermediates_on_deep_box() {
        let root = crate::commands::vnncomp2026_benchmarks_root()
            .expect("VNN-COMP 2026 corpus missing under either checkout name");
        let base = root.join("isomorphic_acasxu_2026/2.0");
        assert!(
            base.is_dir(),
            "external VNN-COMP 2026 relational fixtures missing at {}; run \
             benchmarks/vnncomp2026_benchmarks/setup.sh",
            base.display()
        );
        let f = base.join("onnx/original/ACASXU_run2a_2_4_batch_2000.onnx");
        let g = base.join("onnx/perturbed/ACASXU_run2a_2_4_batch_2000_perturbed_0.onnx");
        let vnnlib = base.join("vnnlib/instance_0.vnnlib");
        let graph_f = super::super::vnncomp::load_graph_network(&f).expect("load f");
        let graph_g = super::super::vnncomp::load_graph_network(&g).expect("load g");
        let diff = ny_propagate::build_difference_network(&graph_f, &graph_g).expect("diff");
        let n_nodes = diff.exec_order().expect("exec").len();
        let mut kinds: std::collections::BTreeMap<&'static str, usize> =
            std::collections::BTreeMap::new();
        for name in diff.exec_order().expect("exec") {
            if let Some(node) = diff.node(name) {
                *kinds.entry(node.layer().layer_type()).or_default() += 1;
            }
        }
        eprintln!(
            "diff net: {n_nodes} nodes (per-node CROWN-IBP threshold = 50); layers = {kinds:?}"
        );
        let spec = ny_onnx::vnnlib::load_vnnlib(&vnnlib).expect("vnnlib");
        let dual = spec.dual_network.expect("dual");
        let bounds = super::super::vnncomp::bounds_from_f64(&dual.f_input_bounds).expect("bounds");

        // Deep sub-box: shrink each dim to 1/64 width at its center (depth ~30).
        let deep: Vec<ny_core::Bound> = bounds
            .iter()
            .map(|b| {
                let c = f32::midpoint(b.lower(), b.upper());
                let w = (b.upper() - b.lower()) / 128.0;
                ny_core::Bound::new(c - w, c + w)
            })
            .collect();
        let input = ny_propagate::Verifier::bounds_to_tensor(&deep, None).expect("tensor");

        // Band spec: rows ±e_i over 5 outputs.
        let n_out = 5usize;
        let mut rows = Vec::new();
        for i in 0..n_out {
            let mut r = vec![0.0f32; n_out];
            r[i] = 1.0;
            rows.push(r.clone());
            r[i] = -1.0;
            rows.push(r);
        }
        let spec_matrix =
            ndarray::Array2::from_shape_vec((rows.len(), n_out), rows.concat()).unwrap();

        // (A) IBP intermediates.
        let t = std::time::Instant::now();
        let ibp_nb = diff.collect_node_bounds(&input).expect("ibp nb");
        let t_ibp = t.elapsed().as_secs_f64();
        let (b_ibp, _) = diff
            .propagate_crown_with_specs_and_node_bounds_and_linear(
                &input,
                &spec_matrix,
                None,
                &ibp_nb,
            )
            .expect("crown w/ ibp nb");
        let flat = b_ibp.flatten();
        let min_lower_ibp = (0..flat.len())
            .map(|i| flat.lower()[[i]])
            .fold(f32::INFINITY, f32::min);

        // (B) per-node CROWN-IBP intermediates.
        let t = std::time::Instant::now();
        let cibp_nb = diff
            .collect_crown_ibp_bounds_dag_with_engine(&input, None)
            .expect("crown-ibp nb");
        let t_cibp = t.elapsed().as_secs_f64();
        let (b_cibp, _) = diff
            .propagate_crown_with_specs_and_node_bounds_and_linear(
                &input,
                &spec_matrix,
                None,
                &cibp_nb,
            )
            .expect("crown w/ crown-ibp nb");
        let flat = b_cibp.flatten();
        let min_lower_cibp = (0..flat.len())
            .map(|i| flat.lower()[[i]])
            .fold(f32::INFINITY, f32::min);

        // LEVER-1 CHECK: is the collection's OUTPUT entry equal to the spec
        // backward's +/-e_i band bounds (the redundancy claim)?
        let out_name = diff.output_name();
        let out_entry = cibp_nb.get(out_name).expect("output entry").flatten();
        let entry_min = (0..out_entry.len())
            .map(|i| out_entry.lower()[[i]].min(-out_entry.upper()[[i]]))
            .fold(f32::INFINITY, f32::min);
        assert!(
            [min_lower_ibp, min_lower_cibp, entry_min]
                .into_iter()
                .all(f32::is_finite),
            "deep-box comparison must publish finite IBP/CROWN-IBP bounds"
        );
        assert!(
            min_lower_cibp + 1e-5 >= min_lower_ibp,
            "CROWN-IBP intermediates must not loosen the binding lower bound: IBP={min_lower_ibp}, CROWN-IBP={min_lower_cibp}"
        );
        eprintln!(
            "lever-1 parity: collection output entry band min={entry_min:.6} vs spec backward min={min_lower_cibp:.6}"
        );

        eprintln!(
            "deep 1/64 CENTER box: IBP-interm min_lower={min_lower_ibp:.4} ({t_ibp:.3}s) | CROWN-IBP-interm min_lower={min_lower_cibp:.4} ({t_cibp:.3}s) | need > -0.05"
        );

        // CORNER boxes at several depths, CROWN-IBP intermediates: is the
        // stall genuine corner hardness or a refresh failure?
        let mut corner_cases = 0usize;
        for frac in [16.0f32, 64.0, 256.0, 1024.0] {
            let corner: Vec<ny_core::Bound> = bounds
                .iter()
                .map(|b| {
                    let w = (b.upper() - b.lower()) / frac;
                    ny_core::Bound::new(b.lower(), b.lower() + w)
                })
                .collect();
            let cin = ny_propagate::Verifier::bounds_to_tensor(&corner, None).expect("tensor");
            let t = std::time::Instant::now();
            let nb = diff
                .collect_crown_ibp_bounds_dag_with_engine(&cin, None)
                .expect("crown-ibp nb");
            let t_nb = t.elapsed().as_secs_f64();
            let (b, _) = diff
                .propagate_crown_with_specs_and_node_bounds_and_linear(
                    &cin,
                    &spec_matrix,
                    None,
                    &nb,
                )
                .expect("crown");
            let flat = b.flatten();
            let ml = (0..flat.len())
                .map(|i| flat.lower()[[i]])
                .fold(f32::INFINITY, f32::min);
            // Same corner with plain IBP intermediates for contrast.
            let ibp_nb = diff.collect_node_bounds(&cin).expect("ibp nb");
            let (bi, _) = diff
                .propagate_crown_with_specs_and_node_bounds_and_linear(
                    &cin,
                    &spec_matrix,
                    None,
                    &ibp_nb,
                )
                .expect("crown ibp");
            let flati = bi.flatten();
            let mli = (0..flati.len())
                .map(|i| flati.lower()[[i]])
                .fold(f32::INFINITY, f32::min);
            assert!(
                ml.is_finite() && mli.is_finite(),
                "corner 1/{frac} must publish finite comparison bounds"
            );
            assert!(
                ml + 1e-5 >= mli,
                "corner 1/{frac}: CROWN-IBP must not loosen IBP intermediates ({ml} < {mli})"
            );
            corner_cases += 1;
            eprintln!(
                "corner 1/{frac} box: CROWN-IBP min_lower={ml:.4} ({t_nb:.3}s) | IBP min_lower={mli:.4}"
            );
        }
        assert_eq!(corner_cases, 4, "the complete corner-depth corpus must run");
    }
}

/// F32-TAX PROBE (#relational-bab goal escalation): same-algorithm CROWN in
/// f64 vs f32-STORAGE on the real instance_0 difference net's stuck geometry
/// (the −0.0001 head class). Isolates the f32 per-node bound-storage tax from
/// relaxation/slope differences: both variants share every algorithmic choice
/// (adaptive lower slope `α = u > -l`, chord upper, per-node CROWN-IBP
/// intermediates, identical fold order); the f32 variant merely rounds every
/// stored per-node bound through f32.
// Sole test is `external-vnncomp`; gate the module so its imports cannot go
// dead under `mip` alone.
#[cfg(all(test, feature = "mip", feature = "external-vnncomp"))]
mod f32_tax_probe {
    use std::collections::HashMap;

    use ny_propagate::{GraphNetwork, Layer};

    /// Backward CROWN lower bound of `row · output` over `box` with the given
    /// per-node pre-activation bounds (f64). Supports the difference-net op
    /// class: Linear / AddConstant / SubConstant / Flatten / ReLU / Sub.
    fn crown_lower_f64(
        graph: &GraphNetwork,
        input_lo: &[f64],
        input_hi: &[f64],
        node_bounds: &HashMap<String, (Vec<f64>, Vec<f64>)>,
        target: &str,
        row: &[f64],
    ) -> Option<f64> {
        let exec = graph.exec_order().ok()?;
        // coefficient map: node -> (A row over its outputs, accumulated bias)
        let mut coeffs: HashMap<String, Vec<f64>> = HashMap::new();
        let mut bias = 0.0f64;
        coeffs.insert(target.to_string(), row.to_vec());
        let mut input_acc: Vec<f64> = vec![0.0; input_lo.len()];
        let pos = |name: &str| exec.iter().position(|n| n == name);
        let target_pos = pos(target)?;
        for name in exec.iter().take(target_pos + 1).rev() {
            let Some(a) = coeffs.remove(name.as_str()) else {
                continue;
            };
            let node = graph.node(name)?;
            let inputs = node.inputs();
            let push = |coeffs: &mut HashMap<String, Vec<f64>>,
                        input_acc: &mut Vec<f64>,
                        dst: &str,
                        add: Vec<f64>| {
                if dst == ny_propagate::NETWORK_INPUT {
                    for (t, v) in input_acc.iter_mut().zip(add) {
                        *t += v;
                    }
                } else {
                    let entry = coeffs
                        .entry(dst.to_string())
                        .or_insert_with(|| vec![0.0; add.len()]);
                    for (t, v) in entry.iter_mut().zip(add) {
                        *t += v;
                    }
                }
            };
            match node.layer() {
                Layer::Linear(lin) => {
                    let (out_dim, in_dim) = lin.weight().dim();
                    if a.len() != out_dim {
                        return None;
                    }
                    let mut back = vec![0.0f64; in_dim];
                    for (i, &ai) in a.iter().enumerate() {
                        if ai == 0.0 {
                            continue;
                        }
                        for j in 0..in_dim {
                            back[j] += ai * f64::from(lin.weight()[[i, j]]);
                        }
                        if let Some(b) = lin.bias() {
                            bias += ai * f64::from(b[i]);
                        }
                    }
                    push(&mut coeffs, &mut input_acc, &inputs[0], back);
                }
                Layer::AddConstant(ac) => {
                    let c = ac.constant();
                    for (i, &ai) in a.iter().enumerate() {
                        let cv = c
                            .iter()
                            .nth(if c.len() == 1 { 0 } else { i })
                            .copied()
                            .unwrap_or(0.0);
                        bias += ai * f64::from(cv);
                    }
                    push(&mut coeffs, &mut input_acc, &inputs[0], a);
                }
                Layer::SubConstant(sc) => {
                    let c = sc.constant();
                    let mut fwd = a.clone();
                    for (i, ai) in fwd.iter_mut().enumerate() {
                        let cv = c
                            .iter()
                            .nth(if c.len() == 1 { 0 } else { i })
                            .copied()
                            .unwrap_or(0.0);
                        if sc.reverse {
                            // y = c - x
                            bias += *ai * f64::from(cv);
                            *ai = -*ai;
                        } else {
                            bias -= *ai * f64::from(cv);
                        }
                    }
                    push(&mut coeffs, &mut input_acc, &inputs[0], fwd);
                }
                Layer::Flatten(_) | Layer::Reshape(_) => {
                    push(&mut coeffs, &mut input_acc, &inputs[0], a);
                }
                Layer::Sub(_) => {
                    let neg: Vec<f64> = a.iter().map(|v| -v).collect();
                    push(&mut coeffs, &mut input_acc, &inputs[0], a);
                    push(&mut coeffs, &mut input_acc, &inputs[1], neg);
                }
                Layer::ReLU(_) => {
                    let (pl, pu) = node_bounds.get(inputs[0].as_str())?;
                    let mut back = vec![0.0f64; a.len()];
                    for (j, &aj) in a.iter().enumerate() {
                        let (l, u) = (pl[j], pu[j]);
                        if l >= 0.0 {
                            back[j] = aj;
                        } else if u <= 0.0 {
                            back[j] = 0.0;
                        } else if aj >= 0.0 {
                            // lower relax: slope α (adaptive), zero intercept.
                            let alpha = if u > -l { 1.0 } else { 0.0 };
                            back[j] = aj * alpha;
                        } else {
                            // upper relax (chord): slope u/(u-l), intercept -l*u/(u-l).
                            let s = u / (u - l);
                            back[j] = aj * s;
                            bias += aj * (-l * s);
                        }
                    }
                    push(&mut coeffs, &mut input_acc, &inputs[0], back);
                }
                _ => return None,
            }
        }
        // Concretize at the input box.
        let mut lo = bias;
        for (j, &c) in input_acc.iter().enumerate() {
            lo += if c >= 0.0 {
                c * input_lo[j]
            } else {
                c * input_hi[j]
            };
        }
        Some(lo)
    }

    /// Per-node bounds by f64 per-node CROWN-IBP (identity rows per node),
    /// optionally rounding every stored bound through f32 (the storage tax).
    fn collect_crown_ibp_bounds(
        graph: &GraphNetwork,
        input_lo: &[f64],
        input_hi: &[f64],
        f32_storage: bool,
    ) -> Option<HashMap<String, (Vec<f64>, Vec<f64>)>> {
        let exec = graph.exec_order().ok()?;
        let mut out: HashMap<String, (Vec<f64>, Vec<f64>)> = HashMap::new();
        for name in exec {
            // Node output width: forward one identity row per output elem
            // requires knowing the width; derive from the layer.
            let node = graph.node(name)?;
            let width = match node.layer() {
                Layer::Linear(lin) => lin.weight().dim().0,
                Layer::ReLU(_)
                | Layer::AddConstant(_)
                | Layer::SubConstant(_)
                | Layer::Flatten(_)
                | Layer::Reshape(_)
                | Layer::Sub(_) => {
                    let src = node.inputs().first()?;
                    if src == ny_propagate::NETWORK_INPUT {
                        input_lo.len()
                    } else {
                        out.get(src.as_str())?.0.len()
                    }
                }
                _ => return None,
            };
            let mut lows = Vec::with_capacity(width);
            let mut highs = Vec::with_capacity(width);
            for j in 0..width {
                let mut row = vec![0.0f64; width];
                row[j] = 1.0;
                let lo = crown_lower_f64(graph, input_lo, input_hi, &out, name, &row)?;
                row[j] = -1.0;
                let hi = -crown_lower_f64(graph, input_lo, input_hi, &out, name, &row)?;
                if f32_storage {
                    lows.push(f64::from(lo as f32));
                    highs.push(f64::from(hi as f32));
                } else {
                    lows.push(lo);
                    highs.push(hi);
                }
            }
            out.insert(name.clone(), (lows, highs));
        }
        Some(out)
    }

    #[test]
    #[cfg(feature = "external-vnncomp")]
    fn measure_f32_storage_tax_on_stuck_geometry() {
        let base = crate::commands::vnncomp2026_benchmarks_root()
            .expect("VNN-COMP 2026 corpus missing under either checkout name")
            .join("isomorphic_acasxu_2026/2.0");
        assert!(
            base.is_dir(),
            "external VNN-COMP 2026 relational fixtures missing at {}; run \
             benchmarks/vnncomp2026_benchmarks/setup.sh",
            base.display()
        );
        let f = base.join("onnx/original/ACASXU_run2a_2_4_batch_2000.onnx");
        let g = base.join("onnx/perturbed/ACASXU_run2a_2_4_batch_2000_perturbed_0.onnx");
        let graph_f = super::super::vnncomp::load_graph_network(&f).expect("load f");
        let graph_g = super::super::vnncomp::load_graph_network(&g).expect("load g");
        let diff = ny_propagate::build_difference_network(&graph_f, &graph_g).expect("diff");
        let spec = ny_onnx::vnnlib::load_vnnlib(&base.join("vnnlib/instance_0.vnnlib")).unwrap();
        let dual = spec.dual_network.expect("dual");
        let bounds = super::super::vnncomp::bounds_from_f64(&dual.f_input_bounds).unwrap();
        let out_dim = 5usize;

        // Stuck-geometry boxes: corners at 1/16..1/48 (the −0.001..−0.0001
        // class in the earlier probe) + a 1/64 center.
        let mut cases: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();
        for frac in [16.0f64, 24.0, 32.0, 48.0] {
            let lo: Vec<f64> = bounds.iter().map(|b| f64::from(b.lower())).collect();
            let hi: Vec<f64> = bounds
                .iter()
                .zip(&lo)
                .map(|(b, &l)| l + (f64::from(b.upper()) - l) / frac)
                .collect();
            cases.push((format!("corner 1/{frac}"), lo, hi));
        }
        {
            let lo: Vec<f64> = bounds
                .iter()
                .map(|b| {
                    let c = f64::midpoint(f64::from(b.lower()), f64::from(b.upper()));
                    c - (f64::from(b.upper()) - f64::from(b.lower())) / 128.0
                })
                .collect();
            let hi: Vec<f64> = bounds
                .iter()
                .map(|b| {
                    let c = f64::midpoint(f64::from(b.lower()), f64::from(b.upper()));
                    c + (f64::from(b.upper()) - f64::from(b.lower())) / 128.0
                })
                .collect();
            cases.push(("center 1/64".to_string(), lo, hi));
        }

        for (label, lo, hi) in cases {
            let nb32 = collect_crown_ibp_bounds(&diff, &lo, &hi, true).expect("f32-storage nb");
            let nb64 = collect_crown_ibp_bounds(&diff, &lo, &hi, false).expect("f64 nb");
            let out_name = diff.output_name().to_string();
            // Worst band row min over ±e_i (the verified-band objective).
            let mut worst32 = f64::INFINITY;
            let mut worst64 = f64::INFINITY;
            for i in 0..out_dim {
                for sign in [1.0f64, -1.0] {
                    let mut row = vec![0.0f64; out_dim];
                    row[i] = sign;
                    let b32 =
                        crown_lower_f64(&diff, &lo, &hi, &nb32, &out_name, &row).expect("b32");
                    let b64 =
                        crown_lower_f64(&diff, &lo, &hi, &nb64, &out_name, &row).expect("b64");
                    worst32 = worst32.min(b32);
                    worst64 = worst64.min(b64);
                }
            }
            assert!(
                worst32.is_finite() && worst64.is_finite(),
                "{label}: both storage modes must publish finite bounds"
            );
            assert!(
                worst64 + 1e-6 >= worst32,
                "{label}: f64 storage unexpectedly underperformed f32 storage: f32={worst32}, f64={worst64}"
            );
            eprintln!(
                "[f32-tax] {label}: f32-storage worst_row={worst32:.7} | f64 worst_row={worst64:.7} | tax={:.3e}",
                worst64 - worst32
            );
        }
    }
}

/// SECOND-LEVER PROBE (#relational-bab goal escalation): does α-optimizing
/// the INTERMEDIATE collection (rather than the final row) buy tightness at
/// the stuck geometry? Measures, on the real instance_0 diff net:
///   A = default per-node CROWN-IBP intermediates → row backward;
///   B = A-intermediates ∩ α-collection intermediates → row backward;
///   C = B + row-retargeted α slopes (the full stack).
// Sole test is `external-vnncomp`; gate the module so its imports cannot go
// dead under `mip` alone.
#[cfg(all(test, feature = "mip", feature = "external-vnncomp"))]
mod interm_alpha_probe {

    use ny_tensor::BoundedTensor;

    fn min_band_row(bounds: &BoundedTensor) -> f32 {
        let f = bounds.flatten();
        (0..f.len())
            .map(|i| f.lower()[[i]].min(-f.upper()[[i]]))
            .fold(f32::INFINITY, f32::min)
    }

    #[test]
    #[cfg(feature = "external-vnncomp")]
    fn measure_interm_alpha_gain_on_stuck_geometry() {
        let base = crate::commands::vnncomp2026_benchmarks_root()
            .expect("VNN-COMP 2026 corpus missing under either checkout name")
            .join("isomorphic_acasxu_2026/2.0");
        assert!(
            base.is_dir(),
            "external VNN-COMP 2026 relational fixtures missing at {}; run \
             benchmarks/vnncomp2026_benchmarks/setup.sh",
            base.display()
        );
        let f = base.join("onnx/original/ACASXU_run2a_2_4_batch_2000.onnx");
        let g = base.join("onnx/perturbed/ACASXU_run2a_2_4_batch_2000_perturbed_0.onnx");
        let graph_f = super::super::vnncomp::load_graph_network(&f).expect("load f");
        let graph_g = super::super::vnncomp::load_graph_network(&g).expect("load g");
        let diff = ny_propagate::build_difference_network(&graph_f, &graph_g).expect("diff");
        let spec = ny_onnx::vnnlib::load_vnnlib(&base.join("vnnlib/instance_0.vnnlib")).unwrap();
        let dual = spec.dual_network.expect("dual");
        let bounds = super::super::vnncomp::bounds_from_f64(&dual.f_input_bounds).unwrap();
        let n_out = 5usize;
        let mut rows = Vec::new();
        for i in 0..n_out {
            let mut r = vec![0.0f32; n_out];
            r[i] = 1.0;
            rows.push(r.clone());
            r[i] = -1.0;
            rows.push(r);
        }
        let spec_matrix =
            ndarray::Array2::from_shape_vec((rows.len(), n_out), rows.concat()).unwrap();

        let mut compared = 0usize;
        for frac in [64.0f32, 128.0, 256.0] {
            let lo: Vec<f32> = bounds
                .iter()
                .map(|b| {
                    let c = f32::midpoint(b.lower(), b.upper());
                    c - (b.upper() - b.lower()) / (2.0 * frac)
                })
                .collect();
            let hi: Vec<f32> = bounds
                .iter()
                .map(|b| {
                    let c = f32::midpoint(b.lower(), b.upper());
                    c + (b.upper() - b.lower()) / (2.0 * frac)
                })
                .collect();
            let input = BoundedTensor::new(
                ndarray::Array1::from(lo).into_dyn(),
                ndarray::Array1::from(hi).into_dyn(),
            )
            .unwrap();

            // A: default per-node CROWN-IBP.
            let nb = diff
                .collect_crown_ibp_bounds_dag_with_engine(&input, None)
                .expect("crown-ibp");
            let (a_bounds, _) = diff
                .propagate_crown_with_specs_and_node_bounds_and_linear(
                    &input,
                    &spec_matrix,
                    None,
                    &nb,
                )
                .expect("A");
            let a = min_band_row(&a_bounds);

            // B: intersect with a 30-iter α collection's node bounds.
            let acfg = ny_propagate::bounds::AlphaCrownConfig {
                iterations: 30,
                ..Default::default()
            };
            let (alpha_nb, init_alpha) = diff
                .collect_alpha_crown_bounds_dag_with_engine(&input, &acfg, None)
                .expect("alpha collection");
            let mut merged = nb.clone();
            for (name, bt) in &alpha_nb {
                if let Some(cur) = merged.get_mut(name) {
                    if cur.lower().shape() == bt.lower().shape() {
                        let l = ndarray::Zip::from(cur.lower())
                            .and(bt.lower())
                            .map_collect(|&x, &y| x.max(y));
                        let u = ndarray::Zip::from(cur.upper())
                            .and(bt.upper())
                            .map_collect(|&x, &y| x.min(y));
                        if let Ok(t) = BoundedTensor::new(l, u) {
                            *cur = t;
                        }
                    }
                }
            }
            let _ = &init_alpha;
            let (b_bounds, _) = diff
                .propagate_crown_with_specs_and_node_bounds_and_linear(
                    &input,
                    &spec_matrix,
                    None,
                    &merged,
                )
                .expect("B");
            let b = min_band_row(&b_bounds);

            assert!(
                a.is_finite() && b.is_finite(),
                "center 1/{frac}: both intermediate-bound routes must publish finite margins"
            );
            assert!(
                b + 1e-5 >= a,
                "center 1/{frac}: intersecting tighter alpha intermediates must not lower the certified margin ({b} < {a})"
            );
            compared += 1;

            eprintln!(
                "[interm-alpha] center 1/{frac}: A(default CROWN-IBP)={a:.6} B(∩ 30-iter α-collection)={b:.6} | gain B−A={:.3e}",
                b - a
            );
        }
        assert_eq!(
            compared, 3,
            "the complete intermediate-alpha corpus must run"
        );
    }
}
