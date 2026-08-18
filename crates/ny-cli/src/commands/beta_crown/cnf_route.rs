// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CNF-recovery verification driver for SAT-encoded ReLU nets (sat_relu):
//! decompile the compiled k-SAT gadget back to its source CNF and decide it
//! exactly with a certificate-producing SAT solver.
//!
//! # Why this exists
//!
//! The sat_relu benchmark compiles k-SAT instances into Gemm→ReLU→Gemm nets;
//! CROWN's LP relaxation is structurally hopeless on them (BaB: 0/100) and the
//! float-MIP fallback gives neither exact arithmetic nor a certificate. But the
//! compilation is mechanically invertible, and the inversion is EXACT:
//!
//! * hidden CLAUSE rows  `h_i = ReLU(Σ_j w_ij x_j + b_i)` with `w ∈ {−1,+1}`
//!   (`+1` ⇔ the literal is NEGATED, `−1` ⇔ positive) and `b_i = 1 − #negated`;
//!   for boolean `x` the row is `1` iff the clause is FALSIFIED, else `0`;
//! * an IDENTITY block `ReLU(x_j)` and a BOOLEANIZATION block `ReLU(2x_j − 1)`;
//! * outputs `Y_0 = 1 − Σ(clause rows)`, `Y_1 = Σ x_j − Σ ReLU(2x_j − 1)`;
//! * every spec asserts `x ∈ [0,1]^n` with unsafe region `Y_0 ≥ 1 ∧ Y_1 ≤ 0`.
//!
//! On `[0,1]`, `x − ReLU(2x−1) ≥ 0` with equality iff `x ∈ {0,1}`, so
//! `Y_1 ≤ 0` forces boolean inputs, and then `Y_0 ≥ 1` holds iff every clause
//! is satisfied. The unsafe region is nonempty **iff the recovered CNF is
//! satisfiable** — an exact equivalence, not a relaxation.
//!
//! # Strategy
//!
//! Detect the gadget BIT-EXACTLY (all weights are small integers, exactly
//! representable in f32 — any deviation falls through fail-closed), decompile
//! to a CNF, and decide it with the `ay-sat` CDCL solver **in process**:
//! - UNSAT ⇒ property holds ⇒ `Verified`, but ONLY after ay's own refutation
//!   has been re-derived as an in-memory resolution DAG and replayed by
//!   [`ay_sat::ResolutionDag::validate`] (see the soundness contract below);
//! - SAT   ⇒ boolean model ⇒ exact `{0,1}` witness, CONFIRMED in-process by a
//!   concrete forward before claiming (and re-confirmed by the vnncomp
//!   ONNX-Runtime trusted-oracle gate downstream);
//! - anything else (deadline / UNKNOWN / certificate trouble) ⇒ `None` ⇒ the
//!   caller continues the normal pipeline when budget remains; an exhausted
//!   caller deadline may instead terminate as `Timeout`.
//!
//! # Why in-process (and not a subprocess)
//!
//! This route used to shell out to an `ay` binary discovered at runtime via
//! `$NY_AY`, then `ay` on `$PATH`, then the rustup `trust` toolchain. When none
//! of the three resolved — the common case on a clean checkout, since nothing
//! in `cargo build` or `vnncomp_scripts/` ever produces that binary — the route
//! silently returned `None` and all 100 sat_relu instances timed out. The
//! ledger's sat_relu row was therefore not reproducible from the repo: it
//! depended on an artifact the build never emitted.
//!
//! `ay-sat` is now a direct rev-pinned dependency of `ny-cli` (the SAME ay
//! commit `ny-mip` already pins for `ay-milp`, and already in the build graph
//! transitively via ay-milp -> ay-dpll -> ay-sat). There is no environment
//! seam left to misconfigure and no per-solve process spawn, DIMACS temp file,
//! or stdout parsing. The old 8 GiB `--memory` envelope is moot in process:
//! ay-sat sizes its arena from `num_vars` alone (no machine-memory probing),
//! and a recovered sat_relu CNF is a few hundred clauses over a few dozen
//! variables.
//!
//! # Soundness contract
//!
//! A wrong UNSAT is a false `Verified` and costs −150, so the solver's *status*
//! is never trusted on its own:
//!
//! 1. **Detection** is bit-exact and fail-closed, and establishes the exact
//!    equivalence above: the unsafe region is nonempty **iff** the recovered CNF
//!    is satisfiable. It is a decompilation, not a relaxation.
//! 2. **UNSAT ⇒ Verified** requires a *checked certificate*.
//!    [`ay_sat::prove_cnf_unsat_dimacs`] re-solves the CNF with proof logging
//!    and returns the solver's own refutation as an in-memory
//!    [`ay_sat::ResolutionDag`] (original clauses + LRAT-style RUP steps ending
//!    in the empty clause). We then re-run [`ay_sat::ResolutionDag::validate`]
//!    ourselves — an independent, hint-driven RUP replay with CaDiCaL
//!    `lratchecker.cpp` semantics — and, critically, confirm the DAG's
//!    `original_clauses` are EXACTLY the clauses we handed in. A valid
//!    refutation of some *other* clause set would prove nothing about ours.
//! 3. **SAT ⇒ Violated** is confirmed by a concrete in-process forward through
//!    the real network (`concrete_violates`), then re-confirmed downstream by
//!    the vnncomp ORT trusted-oracle gate. The SAT direction cannot produce a
//!    false `Verified` at all.
//! 4. Every failure mode — deadline expiry, `Unknown`, a certificate that does
//!    not replay, a certificate over the wrong clauses — returns `None`. The
//!    route can decline to produce a verdict; it can never produce a wrong one.
//!
//! Disable the whole driver with `NY_NO_CNF_ROUTE=1` (batteries-included:
//! default ON; detection costs microseconds and only fires on the exact gadget).

use std::time::Instant;

use ay_sat::{Literal, ResolutionDag, SatResult, Solver};
use ndarray::{ArrayD, IxDyn};
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::{
    layers::LinearLayer, BabVerificationStatus, BetaCrownResult, GraphNetwork, Layer, Network,
    NETWORK_INPUT,
};
use tracing::{debug, info, warn};

use super::cell_enum::concrete_violates;
use super::BetaCrownModel;

/// Wall-clock headroom the certification pass must have before it is attempted.
///
/// The gate solve (which runs under the real deadline) has already refuted the
/// formula by the time we get here, so certification re-treads a search we know
/// terminates; but it re-solves with proof logging and is not itself
/// deadline-interruptible, so we only start it with room to spare.
/// Below this floor we decline rather than risk overrunning the harness
/// watchdog. See [`certify_headroom`].
const CERTIFY_MIN_REMAINING: std::time::Duration = std::time::Duration::from_secs(1);

/// Multiple of the observed gate-solve cost the certification pass must have
/// available before it is started. Certification re-treads the same search with
/// preprocessing likewise disabled, so the gate's own cost is the natural scale;
/// the multiple absorbs proof recording and the RUP replay.
const CERTIFY_COST_MULTIPLE: u32 = 4;

/// Wall-clock the certification pass must have before it is worth starting:
/// the floor, or `CERTIFY_COST_MULTIPLE` times what the gate solve just cost,
/// whichever is larger.
///
/// The gate runs under the real deadline and has already refuted the formula, so
/// its elapsed time is a live, instance-specific measurement of how hard this
/// search is — a far better bound than any fixed constant. A gadget that the
/// gate cracked in 0.1s needs only the floor; one that took 20s is required to
/// show 80s before we commit to re-solving it uninterruptibly.
fn certify_headroom(gate_cost: std::time::Duration) -> std::time::Duration {
    CERTIFY_MIN_REMAINING.max(gate_cost * CERTIFY_COST_MULTIPLE)
}

/// A recovered CNF: variables are `1..=n_vars` (DIMACS convention), each clause
/// a list of nonzero literals (`+v` positive, `−v` negated).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RecoveredCnf {
    pub n_vars: usize,
    pub clauses: Vec<Vec<i64>>,
}

/// Try the CNF-recovery driver. `None` means not applicable or undecided. The
/// caller preserves normal fallthrough when budget remains; an exhausted
/// caller deadline may terminate as `Timeout`.
pub(super) fn try_cnf_recovery(
    model_net: &BetaCrownModel,
    input_shape: &[usize],
    vnnlib: &VnnLibSpec,
    deadline: Option<Instant>,
) -> Option<BetaCrownResult> {
    if std::env::var_os("NY_NO_CNF_ROUTE").is_some_and(|v| v == "1") {
        return None;
    }
    let start = Instant::now();

    let cnf = detect(model_net, input_shape, vnnlib)?;
    info!(
        "CNF recovery qualifies: {} vars, {} clauses (SAT-encoded ReLU gadget)",
        cnf.n_vars,
        cnf.clauses.len()
    );

    // DIMACS literals are `i32` at the ay-sat boundary; a formula that does not
    // fit is not a sat_relu gadget. Fail closed.
    let clauses = to_i32_clauses(&cnf)?;

    match solve_gate(&cnf, &clauses, deadline)? {
        SatResult::Sat(model) => {
            // `model[i]` is variable `i + 1` (ay-sat's DIMACS convention, the
            // same one the old `v `-line parser used). Fail closed if the
            // solver returned fewer assignments than we have variables.
            if model.len() < cnf.n_vars {
                warn!(
                    "CNF recovery: SAT model covers {} of {} variables; falling through (sound)",
                    model.len(),
                    cnf.n_vars
                );
                return None;
            }
            let result = confirm_boolean_witness(
                model_net,
                input_shape,
                vnnlib,
                &model[..cnf.n_vars],
                start,
            );
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                warn!(
                    "CNF recovery: SAT witness confirmation finished after the caller's \
                     deadline; falling through without publishing a verdict"
                );
                return None;
            }
            result
        }
        SatResult::Unsat(_) => {
            // Do NOT conclude Verified from the solver's status. Re-derive the
            // refutation as a certificate and CHECK it; a wrong UNSAT here is a
            // −150 false-VERIFIED. See the module-level soundness contract.
            let needed = certify_headroom(start.elapsed());
            if let Some(deadline) = deadline {
                let remaining = deadline.checked_duration_since(Instant::now())?;
                if remaining < needed {
                    warn!(
                        "CNF recovery: refuted but only {:.2}s left to certify (need {:.2}s); \
                         falling through (sound — never Verified on an unchecked refutation)",
                        remaining.as_secs_f64(),
                        needed.as_secs_f64()
                    );
                    return None;
                }
            }
            if !certify_refutation(&cnf, &clauses) {
                return None;
            }
            // The proof API is not interruptible. The headroom gate above
            // bounds the risk, and this second gate ensures a proof that
            // nevertheless finishes late can never publish a verdict.
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                warn!(
                    "CNF recovery: refutation certificate validated after the caller's \
                     deadline; falling through without publishing a verdict"
                );
                return None;
            }
            info!(
                "CNF recovery: ay UNSAT + resolution-DAG certificate VALIDATED in {:.2}s \
                 -> property VERIFIED",
                start.elapsed().as_secs_f64()
            );
            Some(BetaCrownResult {
                result: BabVerificationStatus::Verified,
                domains_explored: cnf.clauses.len(),
                domains_verified: cnf.clauses.len(),
                cuts_generated: 0,
                max_depth_reached: 0,
                time_elapsed: start.elapsed(),
                output_bounds: None,
            })
        }
        // `SatResult` is `#[non_exhaustive]`, so this arm covers both today's
        // `Unknown` and any status ay-sat adds later. Both decline: only the
        // two arms above — each with its own confirmation — may yield a
        // verdict, and an unrecognized future status must never acquire one by
        // default.
        _ => {
            // Deadline expiry lands here; ay-sat's contract is that an expired
            // deadline can only produce Unknown, never a verdict.
            info!(
                "CNF recovery: ay returned no decision after {:.2}s (deadline or \
                 incompleteness); falling through to the normal pipeline",
                start.elapsed().as_secs_f64()
            );
            None
        }
    }
}

/// Narrow the recovered clauses to the `i32` DIMACS literals ay-sat consumes.
///
/// Also the panic guard for the ay-sat boundary: `Literal::from_dimacs` panics
/// on literal `0` and on a variable outside the solver's range. `extract_cnf`
/// cannot emit either, but a verifier must not be one refactor away from
/// aborting mid-run, so every literal is range-checked here and anything
/// unrepresentable declines the route instead.
fn to_i32_clauses(cnf: &RecoveredCnf) -> Option<Vec<Vec<i32>>> {
    let n_vars = i32::try_from(cnf.n_vars).ok()?;
    cnf.clauses
        .iter()
        .map(|clause| {
            clause
                .iter()
                .map(|&lit| {
                    let lit = i32::try_from(lit).ok()?;
                    // Nonzero, and |lit| names a variable we declared.
                    (lit != 0 && lit.unsigned_abs() <= n_vars.unsigned_abs()).then_some(lit)
                })
                .collect::<Option<Vec<i32>>>()
        })
        .collect()
}

/// The deadline-bounded gate solve: decide the CNF within the caller's budget.
///
/// This is the only phase that can block for the whole budget, so it is the one
/// that carries the deadline. Preprocessing is disabled to match the
/// certification pass exactly (see [`certify_refutation`]), which makes this
/// solve a faithful predictor of that pass's cost and avoids the
/// `Unknown`-under-proof-logging gap that equisatisfiable preprocessing
/// transforms can otherwise cause.
fn solve_gate(
    cnf: &RecoveredCnf,
    clauses: &[Vec<i32>],
    deadline: Option<Instant>,
) -> Option<SatResult> {
    // A deadline already in the past leaves nothing to spend.
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return None;
    }

    let mut solver = Solver::new(cnf.n_vars);
    solver.set_preprocess_enabled(false);
    solver.set_solve_deadline(deadline);
    for clause in clauses {
        solver.add_clause(clause.iter().copied().map(Literal::from_dimacs).collect());
    }
    Some(solver.solve().into_inner())
}

/// Produce and CHECK a refutation certificate for `cnf`.
///
/// [`ay_sat::prove_cnf_unsat_dimacs`] re-solves with LRAT proof logging and
/// preprocessing disabled, returning the solver's own refutation as an
/// in-memory [`ResolutionDag`]. Two independent checks must then pass:
///
/// * [`ResolutionDag::validate`] — a hint-driven RUP replay of every derived
///   step, ending at the empty clause (CaDiCaL `lratchecker.cpp` semantics).
///   `prove_cnf_unsat_dimacs` runs this too; we re-run it so the guarantee is
///   enforced HERE and cannot be lost to an upstream refactor.
/// * [`certificate_covers`] — the DAG's original clauses are exactly the ones
///   we handed in. A perfectly valid refutation of a *different* clause set
///   would prove nothing about this network.
///
/// Returns `false` on any failure; the caller then declines to produce a
/// verdict.
fn certify_refutation(cnf: &RecoveredCnf, clauses: &[Vec<i32>]) -> bool {
    let dag = match ay_sat::prove_cnf_unsat_dimacs(cnf.n_vars, clauses) {
        Ok(dag) => dag,
        Err(e) => {
            warn!(
                "CNF recovery: ay reported UNSAT but no refutation certificate could be \
                 produced ({e}); falling through (sound — never Verified on an unchecked \
                 refutation)"
            );
            return false;
        }
    };
    if let Err(e) = dag.validate() {
        warn!(
            "CNF recovery: refutation certificate did not replay ({e}); falling through \
             (sound — never Verified on an unchecked refutation)"
        );
        return false;
    }
    if !certificate_covers(&dag, cnf.n_vars, clauses) {
        warn!(
            "CNF recovery: refutation certificate is over a different clause set than the \
             recovered CNF; falling through (sound)"
        );
        return false;
    }
    true
}

/// Confirm `dag` refutes EXACTLY the clause set we submitted: same variable
/// count, same clause count, and every original clause identical (literal for
/// literal, in order) with the canonical LRAT id `index + 1`.
fn certificate_covers(dag: &ResolutionDag, n_vars: usize, clauses: &[Vec<i32>]) -> bool {
    if dag.num_vars != n_vars || dag.original_clauses.len() != clauses.len() {
        return false;
    }
    dag.original_clauses
        .iter()
        .zip(clauses)
        .enumerate()
        .all(|(index, ((id, dag_clause), ours))| {
            *id == index as u64 + 1
                && dag_clause.len() == ours.len()
                && dag_clause
                    .iter()
                    .zip(ours)
                    .all(|(lit, &ours)| lit.to_dimacs() == ours)
        })
}

/// Build the exact `{0,1}` witness from the boolean model, confirm it with a
/// concrete in-process forward, and package the `Violated` result (the vnncomp
/// ORT trusted-oracle gate re-confirms downstream).
fn confirm_boolean_witness(
    model_net: &BetaCrownModel,
    input_shape: &[usize],
    vnnlib: &VnnLibSpec,
    model: &[bool],
    start: Instant,
) -> Option<BetaCrownResult> {
    let values_f32: Vec<f32> = model.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    let point = ArrayD::from_shape_vec(IxDyn(input_shape), values_f32.clone()).ok()?;
    let input = ny_tensor::BoundedTensor::concrete(point).ok()?;
    let output = match model_net {
        BetaCrownModel::Sequential(network) => {
            network.propagate_concrete_point(&input, None).ok()?
        }
        BetaCrownModel::Graph(graph) => graph.propagate_concrete_point(&input, None, None).ok()?,
    };
    let output_center = output.center();
    if !concrete_violates(&output_center, vnnlib) {
        warn!(
            "CNF recovery: SAT model failed concrete confirmation (Y = {:?}); \
             falling through (sound)",
            output_center.iter().collect::<Vec<_>>()
        );
        return None;
    }
    info!(
        "CNF recovery: ay SAT in {:.1}s -> confirmed boolean counterexample",
        start.elapsed().as_secs_f64()
    );
    Some(BetaCrownResult {
        result: BabVerificationStatus::Violated {
            counterexample: values_f32,
            output: output_center.iter().copied().collect(),
        },
        domains_explored: 1,
        domains_verified: 0,
        cuts_generated: 0,
        max_depth_reached: 0,
        time_elapsed: start.elapsed(),
        output_bounds: None,
    })
}

/// Structural + spec detection. `None` => not applicable (fall through).
///
/// FAIL-CLOSED: every weight, bias, block count, and spec constraint must match
/// the gadget BIT-EXACTLY (all constants are small integers, exact in f32).
/// The detector is the soundness gate for the UNSAT direction until the Lean
/// gadget-equivalence theorem lands, so any deviation aborts.
fn detect(
    model_net: &BetaCrownModel,
    input_shape: &[usize],
    vnnlib: &VnnLibSpec,
) -> Option<RecoveredCnf> {
    // ---- Spec shape: x ∈ [0,1]^n, unsafe = {Y_0 ≥ 1 ∧ Y_1 ≤ 0}, nothing else.
    if vnnlib.dual_network.is_some() {
        return None;
    }
    if vnnlib
        .per_clause_input_bounds
        .iter()
        .any(|bounds| !bounds.is_empty())
    {
        return None;
    }
    let n: usize = input_shape.iter().product();
    if n == 0 || vnnlib.input_bounds.len() != n || vnnlib.num_outputs != 2 {
        return None;
    }
    if !vnnlib
        .input_bounds
        .iter()
        .all(|&(lo, hi)| lo == 0.0 && hi == 1.0)
    {
        return None;
    }
    let unsafe_ok = |cs: &[OutputConstraint]| -> bool {
        cs.len() == 2
            && cs
                .iter()
                .any(|c| matches!(c, OutputConstraint::GreaterEqConst(0, v) if *v == 1.0))
            && cs
                .iter()
                .any(|c| matches!(c, OutputConstraint::LessEqConst(1, v) if *v == 0.0))
    };
    // The parser may populate the flat conjunction, the single-clause form, or
    // BOTH (the clause mirroring the conjunction). Whatever is present must be
    // exactly the sat_relu unsafe region.
    let spec_ok = match vnnlib.output_constraint_clauses.len() {
        0 => unsafe_ok(&vnnlib.output_constraints),
        1 => {
            unsafe_ok(&vnnlib.output_constraint_clauses[0])
                && (vnnlib.output_constraints.is_empty() || unsafe_ok(&vnnlib.output_constraints))
        }
        _ => false,
    };
    if !spec_ok {
        debug!(
            "CNF route: spec is not the sat_relu unsafe region (constraints={:?}, clauses={}); \
             falling through",
            vnnlib.output_constraints,
            vnnlib.output_constraint_clauses.len()
        );
        return None;
    }

    // ---- Structure: input → Linear(W1,b1) → ReLU → Linear(W2,b2) = output.
    let (l1, l2) = match model_net {
        BetaCrownModel::Sequential(network) => sequential_chain(network)?,
        BetaCrownModel::Graph(graph) => graph_chain(graph)?,
    };

    let Some(b1) = l1.bias() else {
        debug!("CNF route: L1 has no bias; falling through");
        return None;
    };
    let Some(b2) = l2.bias() else {
        debug!("CNF route: L2 has no bias; falling through");
        return None;
    };
    let cnf = extract_cnf(
        l1.weight().view(),
        b1.view(),
        l2.weight().view(),
        b2.view(),
        n,
    );
    if cnf.is_none() {
        debug!("CNF route: chain matched but gadget extraction failed (bit-exact mismatch)");
    }
    cnf
}

/// The sequential form: exactly `[Linear, ReLU, Linear]`.
fn sequential_chain(network: &Network) -> Option<(&LinearLayer, &LinearLayer)> {
    match network.layers() {
        [Layer::Linear(l1), Layer::ReLU(_), Layer::Linear(l2)] => Some((l1, l2)),
        layers => {
            debug!(
                "CNF route: sequential chain is {:?}, not [Linear, ReLU, Linear]; falling through",
                layers.iter().map(Layer::layer_type).collect::<Vec<_>>()
            );
            None
        }
    }
}

/// The graph form: `NETWORK_INPUT → Linear → ReLU → Linear = output`, each node
/// single-input and the sole consumer of its predecessor.
fn graph_chain(graph: &GraphNetwork) -> Option<(&LinearLayer, &LinearLayer)> {
    let mut consumers: std::collections::HashMap<&str, Vec<&str>> =
        std::collections::HashMap::new();
    for name in graph.node_names() {
        let node = graph.node(name)?;
        for input in node.inputs() {
            consumers.entry(input.as_str()).or_default().push(name);
        }
    }
    let sole = |key: &str| -> Option<&str> {
        match consumers.get(key).map(Vec::as_slice) {
            Some([one]) => Some(*one),
            _ => None,
        }
    };
    let l1_name = sole(NETWORK_INPUT)?;
    let l1_node = graph.node(l1_name)?;
    if l1_node.inputs().len() != 1 {
        return None;
    }
    let Layer::Linear(l1) = l1_node.layer() else {
        return None;
    };
    let relu_name = sole(l1_name)?;
    let relu_node = graph.node(relu_name)?;
    if !matches!(relu_node.layer(), Layer::ReLU(_)) || relu_node.inputs().len() != 1 {
        return None;
    }
    let l2_name = sole(relu_name)?;
    let l2_node = graph.node(l2_name)?;
    if l2_node.inputs().len() != 1 || graph.output_name() != l2_name {
        return None;
    }
    let Layer::Linear(l2) = l2_node.layer() else {
        return None;
    };
    Some((l1, l2))
}

/// The pure extraction core (unit-tested without graph construction): classify
/// hidden rows by their W2 columns, validate every entry bit-exactly, and
/// recover the clauses.
pub(super) fn extract_cnf(
    w1: ndarray::ArrayView2<'_, f32>,
    b1: ndarray::ArrayView1<'_, f32>,
    w2: ndarray::ArrayView2<'_, f32>,
    b2: ndarray::ArrayView1<'_, f32>,
    n: usize,
) -> Option<RecoveredCnf> {
    let h = w1.nrows();
    if w1.ncols() != n || b1.len() != h || w2.nrows() != 2 || w2.ncols() != h || b2.len() != 2 {
        return None;
    }
    // Output biases: Y_0 = 1 − Σ clauses;  Y_1 = Σ ident − Σ bool.
    if b2[0] != 1.0 || b2[1] != 0.0 {
        return None;
    }

    let mut ident_seen = vec![false; n];
    let mut bool_seen = vec![false; n];
    let mut clauses: Vec<Vec<i64>> = Vec::new();

    for i in 0..h {
        let (c0, c1) = (w2[(0, i)], w2[(1, i)]);
        if c0 == -1.0 && c1 == 0.0 {
            // CLAUSE row: entries in {−1, 0, +1}; b = 1 − #(+1 entries).
            let mut lits: Vec<i64> = Vec::new();
            let mut pos_entries: usize = 0; // +1 entries = negated literals
            for j in 0..n {
                match w1[(i, j)] {
                    0.0 => {}
                    1.0 => {
                        pos_entries += 1;
                        lits.push(-((j as i64) + 1)); // +1 ⇔ ¬x_j
                    }
                    -1.0 => {
                        lits.push((j as i64) + 1); // −1 ⇔ x_j
                    }
                    _ => return None,
                }
            }
            if lits.is_empty() {
                return None;
            }
            if b1[i] != 1.0 - pos_entries as f32 {
                return None;
            }
            clauses.push(lits);
        } else if c0 == 0.0 && c1 == 1.0 {
            // IDENTITY row: single +1 at j, zero bias.
            let j = single_nonzero(w1.row(i), 1.0)?;
            if b1[i] != 0.0 || ident_seen[j] {
                return None;
            }
            ident_seen[j] = true;
        } else if c0 == 0.0 && c1 == -1.0 {
            // BOOLEANIZATION row: single +2 at j, bias −1.
            let j = single_nonzero(w1.row(i), 2.0)?;
            if b1[i] != -1.0 || bool_seen[j] {
                return None;
            }
            bool_seen[j] = true;
        } else {
            return None;
        }
    }

    if clauses.is_empty()
        || !ident_seen.iter().all(|&s| s)
        || !bool_seen.iter().all(|&s| s)
        || h != clauses.len() + 2 * n
    {
        return None;
    }
    Some(RecoveredCnf { n_vars: n, clauses })
}

/// Index of the single nonzero entry of `row`, required to equal `value`
/// bit-exactly; `None` on zero or multiple nonzeros or a mismatched value.
fn single_nonzero(row: ndarray::ArrayView1<'_, f32>, value: f32) -> Option<usize> {
    let mut found: Option<usize> = None;
    for (j, &x) in row.iter().enumerate() {
        if x == 0.0 {
            continue;
        }
        if x != value || found.is_some() {
            return None;
        }
        found = Some(j);
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    use std::time::Duration;

    fn cnf(n_vars: usize, clauses: &[&[i64]]) -> RecoveredCnf {
        RecoveredCnf {
            n_vars,
            clauses: clauses.iter().map(|c| c.to_vec()).collect(),
        }
    }

    /// The in-process solver decides an UNSAT formula AND its refutation
    /// certificate replays against exactly the clauses we submitted. This is
    /// the whole UNSAT ⇒ `Verified` soundness path, minus the network layer.
    #[test]
    fn unsat_formula_is_refuted_and_certified() {
        // (x1) ∧ (¬x1) — unsatisfiable.
        let formula = cnf(1, &[&[1], &[-1]]);
        let clauses = to_i32_clauses(&formula).expect("representable");
        let deadline = Instant::now() + Duration::from_secs(30);
        assert!(matches!(
            solve_gate(&formula, &clauses, Some(deadline)),
            Some(SatResult::Unsat(_))
        ));
        assert!(certify_refutation(&formula, &clauses));
    }

    /// A satisfiable formula yields a model covering every variable, and is
    /// never certified as refuted.
    #[test]
    fn sat_formula_yields_a_full_model_and_no_certificate() {
        // (x1 ∨ x2) ∧ (¬x1) ⇒ x1=false, x2=true.
        let formula = cnf(2, &[&[1, 2], &[-1]]);
        let clauses = to_i32_clauses(&formula).expect("representable");
        let deadline = Instant::now() + Duration::from_secs(30);
        let Some(SatResult::Sat(model)) = solve_gate(&formula, &clauses, Some(deadline)) else {
            panic!("expected SAT");
        };
        assert!(model.len() >= formula.n_vars);
        assert!(!model[0], "x1 must be false");
        assert!(model[1], "x2 must be true");
        // No refutation exists, so certification must fail closed.
        assert!(!certify_refutation(&formula, &clauses));
    }

    /// A certificate is only accepted for the EXACT clause set it refutes.
    /// Re-pointing a valid refutation at a different formula must be rejected,
    /// otherwise a verified proof of some other problem could admit a verdict.
    #[test]
    fn certificate_must_cover_the_submitted_clause_set() {
        let refuted = cnf(1, &[&[1], &[-1]]);
        let refuted_clauses = to_i32_clauses(&refuted).expect("representable");
        let dag =
            ay_sat::prove_cnf_unsat_dimacs(refuted.n_vars, &refuted_clauses).expect("refutable");
        assert!(certificate_covers(&dag, refuted.n_vars, &refuted_clauses));

        // Same shape, different literals.
        let other = to_i32_clauses(&cnf(1, &[&[-1], &[1]])).expect("representable");
        assert!(!certificate_covers(&dag, 1, &other));
        // Fewer clauses than the DAG refutes.
        let short = to_i32_clauses(&cnf(1, &[&[1]])).expect("representable");
        assert!(!certificate_covers(&dag, 1, &short));
        // Variable count mismatch.
        assert!(!certificate_covers(&dag, 2, &refuted_clauses));
    }

    /// Certification is uninterruptible, so its go/no-go budget must scale with
    /// how expensive the (deadline-bounded) gate solve just proved to be, never
    /// sitting at a fixed constant.
    #[test]
    fn certify_headroom_scales_with_gate_cost() {
        // A cheap gate only has to clear the floor.
        assert_eq!(
            certify_headroom(Duration::from_millis(10)),
            CERTIFY_MIN_REMAINING
        );
        // An expensive gate demands a multiple of its own cost.
        assert_eq!(
            certify_headroom(Duration::from_secs(20)),
            Duration::from_secs(80)
        );
        // Monotone: a harder gate never asks for less.
        assert!(
            certify_headroom(Duration::from_secs(5)) >= certify_headroom(Duration::from_secs(1))
        );
    }

    /// An already-expired deadline must decline rather than start a solve.
    #[test]
    fn expired_deadline_declines() {
        let formula = cnf(1, &[&[1], &[-1]]);
        let clauses = to_i32_clauses(&formula).expect("representable");
        let expired = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("a one-second-old instant is representable");
        assert!(solve_gate(&formula, &clauses, Some(expired)).is_none());
    }

    #[test]
    fn unbounded_gate_solve_has_no_synthetic_deadline() {
        let formula = cnf(1, &[&[1], &[-1]]);
        let clauses = to_i32_clauses(&formula).expect("representable");
        assert!(matches!(
            solve_gate(&formula, &clauses, None),
            Some(SatResult::Unsat(_))
        ));
    }

    /// The ay-sat boundary panics on literal 0 / out-of-range variables, so the
    /// conversion must reject them instead of handing them over.
    #[test]
    fn unrepresentable_literals_decline_instead_of_panicking() {
        assert!(
            to_i32_clauses(&cnf(2, &[&[1, 0]])).is_none(),
            "zero literal"
        );
        assert!(
            to_i32_clauses(&cnf(2, &[&[3]])).is_none(),
            "variable above n_vars"
        );
        assert!(
            to_i32_clauses(&cnf(2, &[&[i64::from(i32::MAX) + 1]])).is_none(),
            "literal beyond i32"
        );
        assert!(to_i32_clauses(&cnf(2, &[&[1, -2]])).is_some(), "in range");
    }

    /// Gadget for the 2-var CNF  (x1 ∨ ¬x2) ∧ (¬x1):
    ///   clause rows:  [−1, +1] b=0   (x1 ∨ ¬x2: −1@1 ⇒ x1, +1@2 ⇒ ¬x2, #neg=1)
    ///                 [+1,  0] b=0   (¬x1: #neg=1 ⇒ b=1−1=0)
    ///   ident rows:   [1,0] b=0 ; [0,1] b=0
    ///   bool rows:    [2,0] b=−1 ; [0,2] b=−1
    fn gadget() -> (
        ndarray::Array2<f32>,
        ndarray::Array1<f32>,
        ndarray::Array2<f32>,
        ndarray::Array1<f32>,
    ) {
        let w1 = arr2(&[
            [-1.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]);
        let b1 = arr1(&[0.0, 0.0, 0.0, 0.0, -1.0, -1.0]);
        let w2 = arr2(&[
            [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, -1.0, -1.0],
        ]);
        let b2 = arr1(&[1.0, 0.0]);
        (w1, b1, w2, b2)
    }

    #[test]
    fn extracts_the_exact_cnf() {
        let (w1, b1, w2, b2) = gadget();
        let recovered = extract_cnf(w1.view(), b1.view(), w2.view(), b2.view(), 2).expect("gadget");
        assert_eq!(recovered.n_vars, 2);
        assert_eq!(recovered.clauses, vec![vec![1, -2], vec![-1]]);
        // Literal polarity survives the ay-sat round trip: −1 ⇒ x_j, +1 ⇒ ¬x_j.
        let clauses = to_i32_clauses(&recovered).expect("representable");
        assert_eq!(clauses, vec![vec![1, -2], vec![-1]]);
    }

    /// End-to-end on the decompiled gadget: (x1 ∨ ¬x2) ∧ (¬x1) is satisfiable
    /// (x1=false, x2=false), so the route must find a model rather than refute.
    #[test]
    fn gadget_cnf_round_trips_through_the_solver() {
        let (w1, b1, w2, b2) = gadget();
        let recovered = extract_cnf(w1.view(), b1.view(), w2.view(), b2.view(), 2).expect("gadget");
        let clauses = to_i32_clauses(&recovered).expect("representable");
        let deadline = Instant::now() + Duration::from_secs(30);
        let Some(SatResult::Sat(model)) = solve_gate(&recovered, &clauses, Some(deadline)) else {
            panic!("expected SAT");
        };
        assert!(!model[0], "x1 must be false to satisfy (¬x1)");
    }

    /// FAIL-CLOSED: any single perturbed constant must abort extraction — the
    /// detector is the soundness gate for the UNSAT direction.
    #[test]
    fn rejects_any_deviation() {
        // clause bias off by one
        let (w1, mut b1, w2, b2) = gadget();
        b1[0] = 1.0;
        assert!(extract_cnf(w1.view(), b1.view(), w2.view(), b2.view(), 2).is_none());
        // non-integer clause weight
        let (mut w1, b1, w2, b2) = gadget();
        w1[(0, 0)] = -1.0000001;
        assert!(extract_cnf(w1.view(), b1.view(), w2.view(), b2.view(), 2).is_none());
        // booleanization coefficient not 2
        let (mut w1, b1, w2, b2) = gadget();
        w1[(4, 0)] = 1.5;
        assert!(extract_cnf(w1.view(), b1.view(), w2.view(), b2.view(), 2).is_none());
        // missing identity row for var 2 (retag its W2 column as a clause row)
        let (w1, b1, mut w2, b2) = gadget();
        w2[(0, 3)] = -1.0;
        w2[(1, 3)] = 0.0;
        assert!(extract_cnf(w1.view(), b1.view(), w2.view(), b2.view(), 2).is_none());
        // Y_0 bias wrong
        let (w1, b1, w2, mut b2) = gadget();
        b2[0] = 2.0;
        assert!(extract_cnf(w1.view(), b1.view(), w2.view(), b2.view(), 2).is_none());
        // an unclassifiable W2 column
        let (w1, b1, mut w2, b2) = gadget();
        w2[(0, 2)] = 0.5;
        assert!(extract_cnf(w1.view(), b1.view(), w2.view(), b2.view(), 2).is_none());
    }

    /// Deciding a formula must not require anything outside the built binary.
    ///
    /// This is the regression guard for the defect that made sat_relu score 0
    /// from a clean build: the route used to shell out to an `ay` binary
    /// discovered through `$NY_AY` / `$PATH` / the rustup `trust` toolchain, and
    /// silently declined when none resolved. This test — like every other test
    /// here — runs on machines where no such binary exists, so a return to any
    /// out-of-process transport makes the suite fail rather than quietly
    /// forfeit a category.
    #[test]
    fn deciding_needs_no_external_binary() {
        let formula = cnf(3, &[&[1, 2], &[-1], &[-2, 3]]);
        let clauses = to_i32_clauses(&formula).expect("representable");
        let deadline = Instant::now() + Duration::from_secs(30);
        let Some(SatResult::Sat(model)) = solve_gate(&formula, &clauses, Some(deadline)) else {
            panic!("expected SAT");
        };
        // x1=false forces x2=true, which forces x3=true.
        assert!(!model[0] && model[1] && model[2]);
    }
}
