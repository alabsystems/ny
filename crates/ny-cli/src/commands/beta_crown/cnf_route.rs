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
//! to DIMACS, and hand the CNF to the `ay` CDCL SAT solver:
//! - `s UNSATISFIABLE` ⇒ property holds ⇒ `Verified`; ay writes a DRAT proof
//!   artifact next to the CNF by default (certificate-grade follow-up: check it
//!   with ay's DRAT/LRAT checkers + the Lean gadget-equivalence theorem);
//! - `s SATISFIABLE`  ⇒ boolean model ⇒ exact `{0,1}` witness, CONFIRMED
//!   in-process by a concrete forward before claiming (and re-confirmed by the
//!   vnncomp ONNX-Runtime trusted-oracle gate downstream);
//! - anything else (timeout / UNKNOWN / missing binary / parse trouble)
//!   ⇒ `None` ⇒ the normal pipeline continues unchanged.
//!
//! Binary discovery: `$NY_AY` (path to the `ay` binary), then `ay` on `$PATH`,
//! then the rustup-linked `trust` toolchain.
//! SAT-variant pin: `--sat-variant probe` by default (`NY_AY_SAT_VARIANT`
//! overrides; `default` omits the flag) — the default variant has a known
//! `s UNKNOWN`-under-proof-logging completeness gap on some UNSAT instances.
//! Disable the whole driver with `NY_NO_CNF_ROUTE=1` (batteries-included:
//! default ON; detection costs microseconds and only fires on the exact gadget).

use std::io::Write as _;
use std::process::Command;
use std::time::Instant;

use ndarray::{ArrayD, IxDyn};
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::{
    layers::LinearLayer, BabVerificationStatus, BetaCrownResult, GraphNetwork, Layer, Network,
    NETWORK_INPUT,
};
use tracing::{debug, info, warn};

use super::cell_enum::concrete_violates;
use super::BetaCrownModel;

/// A recovered CNF: variables are `1..=n_vars` (DIMACS convention), each clause
/// a list of nonzero literals (`+v` positive, `−v` negated).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RecoveredCnf {
    pub n_vars: usize,
    pub clauses: Vec<Vec<i64>>,
}

/// Try the CNF-recovery driver. `None` => not applicable / undecided: the
/// caller MUST continue with the normal pipeline unchanged.
pub(super) fn try_cnf_recovery(
    model_net: &BetaCrownModel,
    input_shape: &[usize],
    vnnlib: &VnnLibSpec,
    deadline: Instant,
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

    let ay = resolve_ay_binary()?;
    let budget = deadline.checked_duration_since(Instant::now())?;

    // Write the DIMACS next to a unique temp stem so ay's default proof artifact
    // (`<input>.drat`) lands somewhere predictable and loggable.
    let dir = std::env::temp_dir();
    let stem = format!(
        "ny_cnf_{}_{}",
        std::process::id(),
        start.elapsed().as_nanos()
    );
    let cnf_path = dir.join(format!("{stem}.cnf"));
    {
        let mut f = std::fs::File::create(&cnf_path).ok()?;
        write!(f, "{}", to_dimacs(&cnf)).ok()?;
    }

    let mut cmd = Command::new(&ay);
    cmd.arg("solve").arg(&cnf_path);
    // Leave ≥1s to write the verdict; ay's -t is milliseconds.
    let ms = budget.as_millis().saturating_sub(1000).max(1000) as u64;
    cmd.arg("-t").arg(ms.to_string());
    let variant = std::env::var("NY_AY_SAT_VARIANT").unwrap_or_else(|_| "probe".to_string());
    if variant != "default" {
        cmd.arg("--sat-variant").arg(&variant);
    }
    debug!("CNF recovery: invoking {:?}", cmd);
    let out = match cmd.output() {
        Ok(o) => o,
        Err(e) => {
            warn!("CNF recovery: failed to run ay ({e}); falling through");
            return None;
        }
    };
    let stdout = String::from_utf8_lossy(&out.stdout);
    let status_line = stdout
        .lines()
        .find(|l| l.starts_with("s "))
        .map(str::trim)
        .unwrap_or("");

    match status_line {
        "s UNSATISFIABLE" => {
            // Do NOT trust ay's UNSAT on faith: check the DRAT proof it just
            // wrote with ay's own independent DRAT checker before concluding
            // Verified. A wrong UNSAT would be a −150 false-VERIFIED; the proof
            // check closes that gap (certificate-CHECKED, not verdict-grade).
            let drat = cnf_path.with_extension("cnf.drat");
            if !check_drat(&ay, &cnf_path, &drat) {
                warn!(
                    "CNF recovery: ay reported UNSAT but its DRAT proof did not verify \
                     (or is missing); falling through (sound — never Verified on an \
                     unchecked refutation)"
                );
                return None;
            }
            info!(
                "CNF recovery: ay UNSAT + DRAT VERIFIED in {:.1}s -> property VERIFIED ({})",
                start.elapsed().as_secs_f64(),
                drat.display()
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
        "s SATISFIABLE" => {
            let model = parse_dimacs_model(&stdout, cnf.n_vars)?;
            confirm_boolean_witness(model_net, input_shape, vnnlib, &model, start)
        }
        other => {
            debug!(
                "CNF recovery: ay returned '{}' (not decisive); falling through",
                other
            );
            None
        }
    }
}

/// Verify a DRAT refutation with ay's own checker: `ay check drat <cnf> <drat>`
/// prints `s VERIFIED` / exit 0 on success. Returns `true` ONLY on a verified
/// proof — a missing artifact, a checker error, or `s NOT VERIFIED` all return
/// `false` (fail-closed: an unverified refutation is never trusted).
fn check_drat(ay: &std::path::Path, cnf: &std::path::Path, drat: &std::path::Path) -> bool {
    if !drat.is_file() {
        return false;
    }
    match Command::new(ay)
        .arg("check")
        .arg("drat")
        .arg(cnf)
        .arg(drat)
        .output()
    {
        Ok(o) => o.status.success() && String::from_utf8_lossy(&o.stdout).contains("s VERIFIED"),
        Err(e) => {
            warn!("CNF recovery: `ay check drat` failed to run ({e})");
            false
        }
    }
}

/// `$NY_AY` (explicit path), `ay` on PATH, then the rustup-linked `trust`
/// toolchain. Returns `None` when no binary is resolvable (driver silently
/// disabled).
fn resolve_ay_binary() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("NY_AY") {
        let p = std::path::PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
        warn!("NY_AY is set but not a file; ignoring");
    }
    // PATH probe: cheap `which`-alike.
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let cand = dir.join("ay");
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    let rustup_home = std::env::var_os("RUSTUP_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".rustup"))
        });
    if let Some(candidate) = rustup_home.map(|root| root.join("toolchains/trust/bin/ay")) {
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    debug!("CNF recovery: no ay binary (NY_AY, PATH, or rustup trust toolchain)");
    None
}

/// Render DIMACS. Variables 1..=n, `0`-terminated clause lines.
fn to_dimacs(cnf: &RecoveredCnf) -> String {
    let mut s = format!("p cnf {} {}\n", cnf.n_vars, cnf.clauses.len());
    for clause in &cnf.clauses {
        for lit in clause {
            s.push_str(&lit.to_string());
            s.push(' ');
        }
        s.push_str("0\n");
    }
    s
}

/// Parse the `v `-line model (DIMACS conventions: signed ints, 0 terminator).
/// Returns per-variable booleans indexed `0..n_vars`. Fails closed on any
/// missing variable.
fn parse_dimacs_model(stdout: &str, n_vars: usize) -> Option<Vec<bool>> {
    let mut assigned: Vec<Option<bool>> = vec![None; n_vars];
    for line in stdout.lines() {
        let Some(rest) = line.strip_prefix("v ") else {
            continue;
        };
        for tok in rest.split_whitespace() {
            let lit: i64 = tok.parse().ok()?;
            if lit == 0 {
                continue;
            }
            let var = lit.unsigned_abs() as usize;
            if var == 0 || var > n_vars {
                return None; // out-of-range literal: fail closed
            }
            assigned[var - 1] = Some(lit > 0);
        }
    }
    assigned.into_iter().collect()
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

    let Some(b1) = l1.bias.as_ref() else {
        debug!("CNF route: L1 has no bias; falling through");
        return None;
    };
    let Some(b2) = l2.bias.as_ref() else {
        debug!("CNF route: L2 has no bias; falling through");
        return None;
    };
    let cnf = extract_cnf(l1.weight.view(), b1.view(), l2.weight.view(), b2.view(), n);
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
        let cnf = extract_cnf(w1.view(), b1.view(), w2.view(), b2.view(), 2).expect("gadget");
        assert_eq!(cnf.n_vars, 2);
        assert_eq!(cnf.clauses, vec![vec![1, -2], vec![-1]]);
        assert_eq!(to_dimacs(&cnf), "p cnf 2 2\n1 -2 0\n-1 0\n");
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

    #[test]
    fn model_parse_roundtrip() {
        let out = "c comment\ns SATISFIABLE\nv 1 -2 0\n";
        assert_eq!(parse_dimacs_model(out, 2), Some(vec![true, false]));
        // missing var 2 fails closed
        assert_eq!(parse_dimacs_model("s SATISFIABLE\nv 1 0\n", 2), None);
        // out-of-range literal fails closed
        assert_eq!(parse_dimacs_model("v 3 0\n", 2), None);
    }
}
