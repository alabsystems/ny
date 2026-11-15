// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! AY proposes multipliers for the complete source relaxation. It never owns
//! neural verdict authority. All public solver evidence is checked against the
//! exact AY model and its rational primal is checked again against ORIGINAL
//! source rows. Rounded multipliers are replayed by NY's source checker.

use ay_milp::{
    BoundSide, Col, FactRef, LpSession, Model, OptimalityCertificate, Outcome, Sense, SolveOpts,
};
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[path = "paired_difference_ay_reduced.rs"]
mod reduced;
pub use reduced::{
    diagnose_reduced_dual_text, solve_reduced_float, solve_reduced_float_with_policy,
    NumericalReductionReplay, ReducedProposalPolicy, ReducedProposalResult,
    ReducedProposalStatistics,
};

const MAX_VARIABLES: usize = 8192;
const MAX_ROWS: usize = 100_000;
const MAX_TERMS: usize = 5_000_000;
const MAX_MAGNITUDE: f64 = 1e100;
const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_RATIONAL_BITS: u64 = 32_768;
const MAX_TOTAL_BITS: u64 = 16_777_216;
const MAX_ACTIVE_WORKERS: usize = 8;
static ACTIVE_WORKERS: AtomicUsize = AtomicUsize::new(0);

struct WorkerPermit<'a> {
    counter: &'a AtomicUsize,
}

impl Drop for WorkerPermit<'_> {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

fn acquire_worker(counter: &AtomicUsize, limit: usize) -> Result<WorkerPermit<'_>, String> {
    let mut active = counter.load(Ordering::Acquire);
    loop {
        if active >= limit {
            return Err("AY provider busy: active-worker cap".into());
        }
        match counter.compare_exchange_weak(active, active + 1, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => return Ok(WorkerPermit { counter }),
            Err(now) => active = now,
        }
    }
}

#[derive(Clone, Debug)]
struct Row {
    terms: Vec<(usize, f64)>,
    rhs: f64,
}

/// Created from the immutable dump produced by the source engine, never from
/// the candidate file. The final checker reconstructs the source independently.
#[derive(Clone, Debug)]
pub struct SourceModel {
    bounds: Vec<(f64, f64)>,
    equalities: Vec<Row>,
    pivots: Vec<usize>,
    inequalities: Vec<Row>,
    outputs: Vec<(usize, usize)>,
}

impl SourceModel {
    pub fn parse(text: &str) -> Result<Self, String> {
        if text.len() > MAX_BYTES {
            return Err("LP byte cap".into());
        }
        let mut t = text.split_whitespace();
        if t.next() != Some("PAIRED_LP_GEQ_V1") {
            return Err("bad LP sense/version".into());
        }
        fn read<T: std::str::FromStr>(t: &mut std::str::SplitWhitespace<'_>) -> Result<T, String> {
            t.next()
                .ok_or("missing LP token")?
                .parse()
                .map_err(|_| "bad LP token".into())
        }
        fn number(t: &mut std::str::SplitWhitespace<'_>) -> Result<f64, String> {
            let x: f64 = read(t)?;
            if !x.is_finite() || x.abs() > MAX_MAGNITUDE {
                return Err("LP number cap".into());
            }
            Ok(x)
        }
        fn row(
            t: &mut std::str::SplitWhitespace<'_>,
            nv: usize,
            charged: &mut usize,
        ) -> Result<Row, String> {
            let n: usize = read(t)?;
            *charged = charged.checked_add(n).ok_or("LP term overflow")?;
            if *charged > MAX_TERMS {
                return Err("LP term cap".into());
            }
            let mut terms = Vec::with_capacity(n);
            for _ in 0..n {
                let j: usize = read(t)?;
                let a = number(t)?;
                if j >= nv {
                    return Err("LP row index".into());
                }
                terms.push((j, a));
            }
            Ok(Row {
                terms,
                rhs: number(t)?,
            })
        }
        let nv: usize = read(&mut t)?;
        let ne: usize = read(&mut t)?;
        let ni: usize = read(&mut t)?;
        let no: usize = read(&mut t)?;
        if nv == 0
            || nv > MAX_VARIABLES
            || ne.checked_add(ni).ok_or("LP row overflow")? > MAX_ROWS
            || no == 0
            || no > 128
            || ne > nv
        {
            return Err("LP dimension cap".into());
        }
        let mut bounds = Vec::with_capacity(nv);
        for _ in 0..nv {
            let l = number(&mut t)?;
            let h = number(&mut t)?;
            if l > h {
                return Err("inverted LP box".into());
            }
            bounds.push((l, h));
        }
        let mut equalities = Vec::with_capacity(ne);
        let mut pivots = BTreeSet::new();
        let mut authored_pivots = Vec::with_capacity(ne);
        let mut charged = 0;
        for _ in 0..ne {
            let p: usize = read(&mut t)?;
            let r = row(&mut t, nv, &mut charged)?;
            if p >= nv
                || !pivots.insert(p)
                || r.terms.iter().filter(|&&(j, _)| j == p).count() != 1
                || !r.terms.iter().any(|&(j, a)| j == p && a == 1.0)
                || r.terms.iter().any(|&(j, _)| j > p)
            {
                return Err("invalid authored LP pivot".into());
            }
            equalities.push(r);
            authored_pivots.push(p);
        }
        let inequalities = (0..ni)
            .map(|_| row(&mut t, nv, &mut charged))
            .collect::<Result<Vec<_>, _>>()?;
        let mut outputs = Vec::with_capacity(no);
        for _ in 0..no {
            let f: usize = read(&mut t)?;
            let g: usize = read(&mut t)?;
            if f >= nv || g >= nv {
                return Err("LP output index".into());
            }
            outputs.push((f, g));
        }
        if t.next().is_some() {
            return Err("trailing LP data".into());
        }
        Ok(Self {
            bounds,
            equalities,
            pivots: authored_pivots,
            inequalities,
            outputs,
        })
    }

    fn zero_dual(&self) -> RoundedDual {
        RoundedDual {
            equalities: vec![0.0; self.equalities.len()],
            inequalities: vec![0.0; self.inequalities.len()],
        }
    }

    fn dual_text(&self, candidates: &[RoundedDual]) -> String {
        let mut out = format!(
            "PAIRED_LP_DUAL_GEQ_V1 {} {} {}\n",
            self.equalities.len(),
            self.inequalities.len(),
            self.outputs.len()
        );
        for d in candidates {
            for &x in d.equalities.iter().chain(&d.inequalities) {
                write!(&mut out, "{x:.17e} ").unwrap();
            }
            out.push('\n');
        }
        out
    }
}

#[derive(Clone, Debug)]
struct RoundedDual {
    equalities: Vec<f64>,
    inequalities: Vec<f64>,
}

#[derive(Clone, Copy)]
enum Role {
    Equality(usize),
    Inequality(usize),
    Alias,
}

struct Prepared {
    model: Model,
    cols: Vec<Col>,
    roles: Vec<Role>,
    aliases: usize,
}

fn rational(x: f64) -> BigRational {
    BigRational::from_float(x).expect("admitted finite source number")
}
fn expired(deadline: Instant) -> Result<(), String> {
    if Instant::now() >= deadline {
        Err("deadline".into())
    } else {
        Ok(())
    }
}

fn prepare(source: &SourceModel, deadline: Instant) -> Result<Prepared, String> {
    let mut model = Model::new();
    let mut cols: Vec<_> = source
        .bounds
        .iter()
        .map(|&(l, h)| model.add_col(l, h))
        .collect();
    let mut aliases = Vec::new();
    let mut roles = Vec::new();
    for (kind, rows) in [(true, &source.equalities), (false, &source.inequalities)] {
        for (i, row) in rows.iter().enumerate() {
            expired(deadline)?;
            // Model::add_row sums duplicate columns in floating point. Exact
            // coalescing is legal only when the sum is EXACTLY representable.
            let mut exact = BTreeMap::<usize, BigRational>::new();
            for &(j, a) in &row.terms {
                *exact.entry(j).or_insert_with(BigRational::zero) += rational(a);
            }
            let merged: Option<Vec<_>> = exact
                .iter()
                .filter(|(_, a)| !a.is_zero())
                .map(|(&j, a)| {
                    let f = a.to_f64()?;
                    (f.is_finite() && rational(f) == *a).then_some((cols[j], f))
                })
                .collect();
            let terms = if let Some(merged) = merged {
                merged
            } else {
                // An exact alias lifting preserves each authored coefficient
                // without changing the feasible set, even for huge cancellation.
                let mut seen = BTreeSet::new();
                let mut lifted = Vec::new();
                for &(j, a) in &row.terms {
                    if a == 0.0 {
                        continue;
                    }
                    let col = if seen.insert(j) {
                        cols[j]
                    } else {
                        if cols.len() >= 2 * MAX_VARIABLES || aliases.len() >= MAX_VARIABLES {
                            return Err("alias variable cap".into());
                        }
                        let (l, h) = source.bounds[j];
                        let alias = model.add_col(l, h);
                        aliases.push((alias, cols[j]));
                        cols.push(alias);
                        alias
                    };
                    lifted.push((col, a));
                }
                lifted
            };
            let ay_row = model.add_row(row.rhs, if kind { row.rhs } else { f64::INFINITY }, &terms);
            if ay_row.index() != roles.len() {
                return Err("AY source row order".into());
            }
            roles.push(if kind {
                Role::Equality(i)
            } else {
                Role::Inequality(i)
            });
        }
    }
    for &(alias, original) in &aliases {
        expired(deadline)?;
        let row = model.add_row(0.0, 0.0, &[(alias, 1.0), (original, -1.0)]);
        if row.index() != roles.len() {
            return Err("AY alias row order".into());
        }
        roles.push(Role::Alias);
    }
    Ok(Prepared {
        model,
        cols,
        roles,
        aliases: aliases.len(),
    })
}

fn objective(source: &SourceModel, output: usize, sign: f64) -> Vec<(u32, BigRational)> {
    let (f, g) = source.outputs[output];
    if f == g {
        return Vec::new();
    }
    let mut result = vec![(f as u32, rational(sign)), (g as u32, rational(-sign))];
    result.sort_by_key(|&(j, _)| j);
    result
}

fn charge(value: &BigRational, total: &mut u64) -> Result<(), String> {
    let bits = value.numer().bits().max(value.denom().bits());
    *total = total.checked_add(bits).ok_or("rational size overflow")?;
    if bits > MAX_RATIONAL_BITS || *total > MAX_TOTAL_BITS {
        return Err("rational certificate cap".into());
    }
    Ok(())
}

fn validate_certificate(
    cert: &OptimalityCertificate,
    prepared: &Prepared,
    expected: &[(u32, BigRational)],
    primal: &[BigRational],
    value: &BigRational,
    source: &SourceModel,
    deadline: Instant,
) -> Result<(), String> {
    expired(deadline)?;
    if cert.sense != Sense::Minimize
        || cert.objective != expected
        || primal.len() != prepared.cols.len()
        || cert.multipliers.len() > 4 * MAX_ROWS
        || &cert.bound != value
    {
        return Err("AY artifact objective/arity/value mismatch".into());
    }
    let mut total = 0;
    for x in std::iter::once(&cert.bound)
        .chain(primal)
        .chain(cert.objective.iter().map(|(_, a)| a))
        .chain(cert.multipliers.iter().map(|m| &m.coeff))
    {
        expired(deadline)?;
        charge(x, &mut total)?;
    }
    // Public AY verification is not cooperative; the complete caller worker
    // is deadline-isolated. These calls contain no numerical tolerance.
    cert.verify(&prepared.model)
        .map_err(|e| format!("AY exact dual verification: {e}"))?;
    prepared
        .model
        .check_point(primal)
        .map_err(|e| format!("AY exact primal verification: {e:?}"))?;
    if prepared.model.objective_value_at(primal) != *value {
        return Err("AY exact primal objective mismatch".into());
    }
    check_original_primal(source, &primal[..source.bounds.len()], deadline)?;
    expired(deadline)
}

fn check_original_primal(
    source: &SourceModel,
    primal: &[BigRational],
    deadline: Instant,
) -> Result<(), String> {
    if primal.len() != source.bounds.len() {
        return Err("source primal arity".into());
    }
    for (x, &(l, h)) in primal.iter().zip(&source.bounds) {
        if *x < rational(l) || *x > rational(h) {
            return Err("source exact primal box violation".into());
        }
    }
    for (equality, rows) in [(true, &source.equalities), (false, &source.inequalities)] {
        for r in rows {
            expired(deadline)?;
            let lhs: BigRational = r.terms.iter().map(|&(j, a)| rational(a) * &primal[j]).sum();
            let rhs = rational(r.rhs);
            if (equality && lhs != rhs) || (!equality && lhs < rhs) {
                return Err("source exact primal row violation".into());
            }
        }
    }
    Ok(())
}

fn extract(
    cert: &OptimalityCertificate,
    prepared: &Prepared,
    source: &SourceModel,
    deadline: Instant,
) -> Result<RoundedDual, String> {
    let mut eq = vec![BigRational::zero(); source.equalities.len()];
    let mut geq = vec![BigRational::zero(); source.inequalities.len()];
    for m in &cert.multipliers {
        expired(deadline)?;
        if !m.coeff.is_positive() {
            return Err("nonpositive AY multiplier".into());
        }
        match m.fact {
            FactRef::RowBound { row, side } => {
                match *prepared.roles.get(row.index()).ok_or("AY row fact index")? {
                    Role::Equality(i) => match side {
                        BoundSide::Lower => eq[i] += &m.coeff,
                        BoundSide::Upper => eq[i] -= &m.coeff,
                    },
                    Role::Inequality(i) => match side {
                        BoundSide::Lower => geq[i] += &m.coeff,
                        BoundSide::Upper => return Err("AY infinite GEQ upper fact".into()),
                    },
                    Role::Alias => {}
                }
            }
            // NY re-prices the full residual on the original input/state box.
            // Alias equations have no original source rows and are omitted.
            FactRef::ColBound { col, .. } if col.index() < prepared.cols.len() => {}
            _ => return Err("unknown AY certificate fact".into()),
        }
    }
    fn convert(x: BigRational) -> Result<f64, String> {
        let x = x.to_f64().ok_or("AY multiplier conversion")?;
        if !x.is_finite() || x.abs() > MAX_MAGNITUDE {
            return Err("NY multiplier cap".into());
        }
        Ok(x)
    }
    Ok(RoundedDual {
        equalities: eq.into_iter().map(convert).collect::<Result<_, _>>()?,
        inequalities: geq.into_iter().map(convert).collect::<Result<_, _>>()?,
    })
}

fn exact_replay(
    source: &SourceModel,
    dual: &RoundedDual,
    output: usize,
    sign: f64,
    deadline: Instant,
) -> Result<BigRational, String> {
    let mut residual = vec![BigRational::zero(); source.bounds.len()];
    for (j, a) in objective(source, output, sign) {
        residual[j as usize] += a;
    }
    let mut rhs = BigRational::zero();
    for (rows, weights) in [
        (&source.equalities, &dual.equalities),
        (&source.inequalities, &dual.inequalities),
    ] {
        for (r, &w) in rows.iter().zip(weights) {
            expired(deadline)?;
            if w == 0.0 {
                continue;
            }
            let w = rational(w);
            rhs += &w * rational(r.rhs);
            for &(j, a) in &r.terms {
                residual[j] -= &w * rational(a);
            }
        }
    }
    for (a, &(l, h)) in residual.iter().zip(&source.bounds) {
        rhs += a * rational(if a.is_negative() { h } else { l });
    }
    Ok(rhs)
}

#[derive(Clone, Debug, Serialize)]
pub struct ObjectiveDiagnostic {
    pub output: usize,
    pub sign: i32,
    pub status: String,
    pub elapsed_s: f64,
    pub ay_exact_optimum: Option<String>,
    pub ay_optimum_f64: Option<f64>,
    pub original_rounded_dual_exact_lower: Option<String>,
    pub original_rounded_dual_lower_f64: Option<f64>,
    pub exact_primal_checked: bool,
    pub primal_source_values: Vec<String>,
    /// Untrusted proposal-ranking score; never a certified or optimal bound.
    pub numerical_proposal_lower: Option<f64>,
    /// Untrusted reduced GEQ/box score before lifting; not an LP optimum.
    pub reduced_proposal_lower: Option<f64>,
    /// Source numerical score minus reduced numerical score. This is a
    /// diagnostic of substitution/lifting loss, not a proof-error bound.
    pub reduction_lift_gap: Option<f64>,
}

#[derive(Debug)]
pub struct AyResult {
    pub dual_text: String,
    /// Alternate sign-convention proposals for independent source replay.
    pub alternate_dual_text: Option<String>,
    pub diagnostics: Vec<ObjectiveDiagnostic>,
    pub variables: usize,
    pub equalities: usize,
    pub inequalities: usize,
    pub alias_variables: usize,
    pub finished_objectives: usize,
    pub reduced_free_variables: Option<usize>,
    pub reduced_nonzeros: Option<usize>,
    pub termination_reason: Option<String>,
}

/// One worker message. The per-objective payloads are boxed: unboxed they made
/// `Objective` 300+ bytes against `Termination`'s 24, and each is sent once per
/// objective (clippy::large_enum_variant, workspace threshold 64).
enum Event {
    Objective(Box<(usize, RoundedDual, Option<RoundedDual>, ObjectiveDiagnostic)>),
    Failure(Box<ObjectiveDiagnostic>),
    Ready(usize),
    ReducedReady(usize, usize),
    Termination(String),
    Finished,
}

/// Run one deadline-isolated, single-threaded AY continuous-LP worker. Late
/// objectives cannot emit artifacts. Completed partial candidates are retained;
/// missing objectives get zero proposals and still undergo full source replay.
pub fn solve(
    source: &SourceModel,
    deadline: Instant,
    memory_bytes: usize,
) -> Result<AyResult, String> {
    workspace_admission(source, deadline, memory_bytes)?;
    let permit = acquire_worker(&ACTIVE_WORKERS, MAX_ACTIVE_WORKERS)?;
    let owned = source.clone();
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("ny-paired-ay-lp".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            // A timed-out caller never releases this permit. It remains held
            // until the actual worker exits, including after a caught panic.
            let _permit = permit;
            let work = || -> Result<(), String> {
                let mut prepared = prepare(&owned, deadline)?;
                let _ = sender.send(Event::Ready(prepared.aliases));
                for output in 0..owned.outputs.len() {
                    for (side, sign) in [1.0, -1.0].into_iter().enumerate() {
                        expired(deadline)?;
                        let start = Instant::now();
                        let expected = objective(&owned, output, sign);
                        let terms: Vec<_> = expected
                            .iter()
                            .map(|&(j, ref a)| (prepared.cols[j as usize], a.to_f64().unwrap()))
                            .collect();
                        prepared.model.set_objective(&terms, Sense::Minimize);
                        let options = solver_options(deadline, memory_bytes);
                        let outcome = LpSession::new(&prepared.model, &options)
                            .and_then(|mut s| s.optimize_model_objective());
                        let result = match outcome {
                            Ok(Outcome::Optimal {
                                value,
                                model_values,
                                cert: Some(cert),
                            }) => validate_certificate(
                                &cert,
                                &prepared,
                                &expected,
                                &model_values,
                                &value,
                                &owned,
                                deadline,
                            )
                            .and_then(|()| extract(&cert, &prepared, &owned, deadline))
                            .and_then(|dual| {
                                let score = exact_replay(&owned, &dual, output, sign, deadline)?;
                                Ok((dual, value, score, model_values))
                            }),
                            Ok(other) => Err(format!(
                                "AY declined checked optimum: {}",
                                match other {
                                    Outcome::Optimal { .. } => "missing_certificate",
                                    Outcome::Infeasible { .. } => "infeasible",
                                    Outcome::Unknown { .. } => "unknown",
                                    Outcome::Unbounded => "unbounded",
                                    _ => "partial",
                                }
                            )),
                            Err(e) => Err(format!("AY solve: {e}")),
                        };
                        let mut d = diagnostic(output, sign, start);
                        match result {
                            Ok((dual, value, score, primal)) => {
                                d.status = "exact_optimal_artifact_checked".into();
                                d.ay_optimum_f64 = value.to_f64();
                                d.ay_exact_optimum = Some(value.to_string());
                                d.original_rounded_dual_lower_f64 = score.to_f64();
                                d.original_rounded_dual_exact_lower = Some(score.to_string());
                                d.exact_primal_checked = true;
                                d.primal_source_values = primal[..owned.bounds.len()]
                                    .iter()
                                    .map(ToString::to_string)
                                    .collect();
                                expired(deadline)?;
                                sender
                                    .send(Event::Objective(Box::new((
                                        2 * output + side,
                                        dual,
                                        None,
                                        d,
                                    ))))
                                    .map_err(|_| "caller expired".to_owned())?;
                            }
                            Err(e) => {
                                d.status = e;
                                let _ = sender.send(Event::Failure(Box::new(d)));
                            }
                        }
                    }
                }
                Ok(())
            };
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) {
                Ok(Err(reason)) => {
                    let _ = sender.send(Event::Termination(reason));
                }
                Err(_) => {
                    let _ = sender.send(Event::Termination("AY worker panic".into()));
                }
                Ok(Ok(())) => {}
            }
            let _ = sender.send(Event::Finished);
        })
        .map_err(|e| e.to_string())?;
    Ok(collect(source, deadline, receiver))
}

fn workspace_admission(
    source: &SourceModel,
    deadline: Instant,
    memory_bytes: usize,
) -> Result<(), String> {
    if !(16 * 1024 * 1024..=2 * 1024 * 1024 * 1024).contains(&memory_bytes) {
        return Err("AY memory budget out of range".into());
    }
    expired(deadline)?;
    // AY's LP memory option is advisory in this revision. A caller-side
    // deterministic admission covers its source matrix/tableau mirrors too.
    let nr = source.equalities.len() + source.inequalities.len();
    let nnz: usize = source
        .equalities
        .iter()
        .chain(&source.inequalities)
        .map(|r| r.terms.len())
        .sum();
    // Count the maximum alias lifting required by repeated coordinates.
    // Exactly representable sums may need fewer; the estimate remains safe.
    let mut duplicates = 0usize;
    for row in source.equalities.iter().chain(&source.inequalities) {
        expired(deadline)?;
        let mut seen = BTreeSet::new();
        for &(j, a) in &row.terms {
            if a != 0.0 && !seen.insert(j) {
                duplicates = duplicates.checked_add(1).ok_or("alias estimate overflow")?;
            }
        }
    }
    let aliases = duplicates.min(MAX_VARIABLES);
    let estimate = source
        .bounds
        .len()
        .checked_add(aliases)
        .and_then(|n| n.checked_mul(nr.checked_add(aliases)?))
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_add(nnz.checked_add(2 * aliases)?.checked_mul(384)?))
        .and_then(|n| n.checked_add(32 * 1024 * 1024))
        .ok_or("AY workspace estimate overflow")?;
    if estimate > memory_bytes {
        return Err(format!(
            "AY workspace admission {estimate} > {memory_bytes}"
        ));
    }
    expired(deadline)
}

fn solver_options(deadline: Instant, memory_bytes: usize) -> SolveOpts {
    SolveOpts::new()
        .with_deadline(deadline)
        .with_threads(1)
        .with_determinism(true)
        .with_require_certificates(true)
        .with_memory_budget(Some(memory_bytes))
        .with_tree_cert_leaves(0)
}

fn diagnostic(output: usize, sign: f64, start: Instant) -> ObjectiveDiagnostic {
    ObjectiveDiagnostic {
        output,
        sign: sign as i32,
        status: "declined".into(),
        elapsed_s: start.elapsed().as_secs_f64(),
        ay_exact_optimum: None,
        ay_optimum_f64: None,
        original_rounded_dual_exact_lower: None,
        original_rounded_dual_lower_f64: None,
        exact_primal_checked: false,
        primal_source_values: Vec::new(),
        numerical_proposal_lower: None,
        reduced_proposal_lower: None,
        reduction_lift_gap: None,
    }
}

fn extract_float(
    duals: &[f64],
    prepared: &Prepared,
    source: &SourceModel,
    convention: f64,
) -> Result<RoundedDual, String> {
    if duals.len() != prepared.roles.len() || (convention != 1.0 && convention != -1.0) {
        return Err("AY float-dual arity/convention".into());
    }
    let mut result = source.zero_dual();
    for (&raw, &role) in duals.iter().zip(&prepared.roles) {
        if !raw.is_finite() || raw.abs() > MAX_MAGNITUDE {
            return Err("AY float-dual number cap".into());
        }
        let y = raw * convention;
        match role {
            Role::Equality(i) => result.equalities[i] = y,
            // Dropping a wrong-signed row multiplier is allowed; NY prices
            // the resulting nonzero residual on its own full source box.
            Role::Inequality(i) => result.inequalities[i] = y.max(0.0),
            Role::Alias => {}
        }
    }
    Ok(result)
}

/// A ranking heuristic over ORIGINAL rows, including every variable box.
/// Ordinary floating-point sums have no authority; both conventions are
/// returned to the independent outward source checker.
fn numerical_score_until(
    source: &SourceModel,
    dual: &RoundedDual,
    output: usize,
    sign: f64,
    deadline: Option<Instant>,
) -> Option<f64> {
    let expired = || deadline.is_some_and(|limit| Instant::now() >= limit);
    if expired() {
        return None;
    }
    let mut residual = vec![0.0; source.bounds.len()];
    let (f, g) = source.outputs[output];
    residual[f] += sign;
    residual[g] -= sign;
    let mut value = 0.0;
    for (rows, weights) in [
        (&source.equalities, &dual.equalities),
        (&source.inequalities, &dual.inequalities),
    ] {
        for (row, &w) in rows.iter().zip(weights) {
            if expired() {
                return None;
            }
            if w == 0.0 {
                continue;
            }
            value += w * row.rhs;
            for (k, &(j, a)) in row.terms.iter().enumerate() {
                if k & 0x3ff == 0 && expired() {
                    return None;
                }
                residual[j] -= w * a;
            }
        }
    }
    for (j, (&a, &(l, h))) in residual.iter().zip(&source.bounds).enumerate() {
        if j & 0x3ff == 0 && expired() {
            return None;
        }
        value += a * if a < 0.0 { h } else { l };
    }
    value.is_finite().then_some(value)
}

/// AY's float simplex supplies explicitly untrusted row multipliers. The
/// provider returns both sign conventions, with GEQ weights clamped >=0.
/// Exact primal/optimality fields remain absent. Only independent NY source
/// replay can turn any returned proposal into mathematical band evidence.
pub fn solve_float(
    source: &SourceModel,
    deadline: Instant,
    memory_bytes: usize,
) -> Result<AyResult, String> {
    workspace_admission(source, deadline, memory_bytes)?;
    let permit = acquire_worker(&ACTIVE_WORKERS, MAX_ACTIVE_WORKERS)?;
    let owned = source.clone();
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("ny-paired-ay-float".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let _permit = permit;
            let work = || -> Result<(), String> {
                let prepared = prepare(&owned, deadline)?;
                let _ = sender.send(Event::Ready(prepared.aliases));
                let options = solver_options(deadline, memory_bytes)
                    .with_engine(ay_milp::EngineEconomics::new().with_float_lane(true));
                let mut session =
                    LpSession::new(&prepared.model, &options).map_err(|e| e.to_string())?;
                for output in 0..owned.outputs.len() {
                    for (side, sign) in [1.0, -1.0].into_iter().enumerate() {
                        expired(deadline)?;
                        let start = Instant::now();
                        let (f, g) = owned.outputs[output];
                        let expr = if f == g {
                            Vec::new()
                        } else {
                            vec![(prepared.cols[f], sign), (prepared.cols[g], -sign)]
                        };
                        let raw = if expr.is_empty() {
                            Ok(Some(vec![0.0; prepared.roles.len()]))
                        } else {
                            session.float_dual_for_expr(&expr, Sense::Minimize)
                        };
                        let result = raw.map_err(|e| e.to_string()).and_then(|raw| {
                            let raw = raw.ok_or("AY float simplex declined row-dual proposal")?;
                            let positive = extract_float(&raw, &prepared, &owned, 1.0)?;
                            let negative = extract_float(&raw, &prepared, &owned, -1.0)?;
                            let ps = numerical_score_until(
                                &owned,
                                &positive,
                                output,
                                sign,
                                Some(deadline),
                            );
                            let ns = numerical_score_until(
                                &owned,
                                &negative,
                                output,
                                sign,
                                Some(deadline),
                            );
                            let prefer_negative = ns.is_some_and(|n| ps.is_none_or(|p| n > p));
                            let (first, second, score) = if prefer_negative {
                                (negative, positive, ns)
                            } else {
                                (positive, negative, ps)
                            };
                            Ok((first, second, score))
                        });
                        let mut diagnostic = diagnostic(output, sign, start);
                        match result {
                            Ok((first, second, score)) => {
                                diagnostic.status = "untrusted_float_duals".into();
                                diagnostic.numerical_proposal_lower = score;
                                expired(deadline)?;
                                sender
                                    .send(Event::Objective(Box::new((
                                        2 * output + side,
                                        first,
                                        Some(second),
                                        diagnostic,
                                    ))))
                                    .map_err(|_| "caller expired".to_owned())?;
                            }
                            Err(reason) => {
                                diagnostic.status = reason;
                                let _ = sender.send(Event::Failure(Box::new(diagnostic)));
                            }
                        }
                    }
                }
                Ok(())
            };
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) {
                Ok(Err(reason)) => {
                    let _ = sender.send(Event::Termination(reason));
                }
                Err(_) => {
                    let _ = sender.send(Event::Termination("AY worker panic".into()));
                }
                Ok(Ok(())) => {}
            }
            let _ = sender.send(Event::Finished);
        })
        .map_err(|e| e.to_string())?;
    Ok(collect(source, deadline, receiver))
}

fn collect(
    source: &SourceModel,
    deadline: Instant,
    receiver: std::sync::mpsc::Receiver<Event>,
) -> AyResult {
    let mut candidates = vec![source.zero_dual(); 2 * source.outputs.len()];
    let mut diagnostics = Vec::new();
    let mut aliases = 0;
    let mut completed = 0;
    let mut alternate = candidates.clone();
    let mut has_alternate = false;
    let mut reduced_free_variables = None;
    let mut reduced_nonzeros = None;
    let mut termination_reason = None;
    loop {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .unwrap_or(Duration::ZERO);
        match receiver.recv_timeout(remaining) {
            Ok(Event::Objective(objective)) => {
                let (index, dual, other, diagnostic) = *objective;
                has_alternate |= other.is_some();
                alternate[index] = other.unwrap_or_else(|| dual.clone());
                candidates[index] = dual;
                diagnostics.push(diagnostic);
                completed += 1;
            }
            Ok(Event::Failure(d)) => diagnostics.push(*d),
            Ok(Event::Ready(n)) => aliases = n,
            Ok(Event::ReducedReady(free, nnz)) => {
                reduced_free_variables = Some(free);
                reduced_nonzeros = Some(nnz);
            }
            Ok(Event::Termination(reason)) => termination_reason = Some(reason),
            Ok(Event::Finished) | Err(_) => break,
        }
    }
    diagnostics.sort_by_key(|d| (d.output, -d.sign));
    AyResult {
        dual_text: source.dual_text(&candidates),
        alternate_dual_text: has_alternate.then(|| source.dual_text(&alternate)),
        diagnostics,
        variables: source.bounds.len(),
        equalities: source.equalities.len(),
        inequalities: source.inequalities.len(),
        alias_variables: aliases,
        finished_objectives: completed,
        reduced_free_variables,
        reduced_nonzeros,
        termination_reason,
    }
}

#[cfg(test)]
#[path = "paired_difference_ay_tests.rs"]
mod tests;
