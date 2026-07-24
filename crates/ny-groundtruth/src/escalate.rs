// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SMT escalation for ground-truth verification — Route B of
//! `docs/GEOMETRIC_GROUND_TRUTH_PLAN.md`.
//!
//! When the CROWN path of [`crate::verify`] answers *Unknown* (its linear
//! relaxation of `PowConstant`/ReLU is too loose for the margin), the **same
//! artifact** — the difference network `h(x) = f(x) − g(x)` built by
//! [`build_difference_network`] — is encoded as one *exact* satisfiability
//! query to the published first-party ay solver:
//!
//! ```text
//! exists x in box:  h(x) < 0        (Dominates; <= 0 with a strict margin)
//! exists x in box:  |h(x)| > eps    (AbsBound)
//! ```
//!
//! `unsat` refutes the violation, i.e. **proves the relation on the whole
//! box** — with an Alethe certificate written next to the query file that an
//! independent checker can replay without trusting AY. `sat` yields a model
//! that is *re-validated in exact rational arithmetic* before being reported
//! as a counterexample (AY prints placeholder models for Krawczyk-certified
//! irrational solutions — a placeholder that fails validation is downgraded
//! to the honest [`SmtVerdict::ViolationExists`]). Timeout and solver
//! `unknown` are reported as [`SmtVerdict::Unknown`], never guessed.
//!
//! # What exactly is proved
//!
//! The encoding denotes the **ideal real-valued semantics** of the difference
//! network: every f32 weight/bias/constant enters as its exact rational value
//! and all arithmetic is real. This is the same object `crate::cert` certifies
//! (exact-rational CROWN), and the float verify path proves the same property
//! through sound FP enclosures — the engines answer the same question at
//! different completeness levels.
//!
//! # Encoding (the layer set of the M1 builders + FC-ReLU `f`)
//!
//! * `Linear` / `Add` / `Sub` / `PowConstant(k)` are **inlined as terms**
//!   (affine combinations, sums, differences, `k`-fold products). No fresh
//!   variables: AY's QF_NRA interval branch-and-prune contracts markedly
//!   better without redundant affine equalities (measured: the same
//!   3-D quadratic query flips from instant `unknown` to `unsat` when
//!   pre-activation equalities are inlined).
//! * `ReLU` / `MinBinary` / `MaxBinary` get **named variables defined by
//!   `ite`** (`(assert (= v (ite (>= t 0) t 0)))`): the piecewise structure
//!   is what the solver case-splits on, and naming it is likewise measurably
//!   friendlier than inline `ite` terms.
//! * The logic is `QF_NRA` when any `PowConstant` exponent ≥ 2 appears,
//!   else `QF_LRA`.
//!
//! # Subprocess lane
//!
//! AY runs as a subprocess (`ay solve -t <ms> query.smt2`), the pattern
//! a3d-solve/a3d-sketch ship: certificates land on disk next to the query,
//! `-t` yields a sound `unknown` on budget exhaustion, and ny's build stays
//! decoupled from the ay workspace. The binary is located via `$NY_AY`, then
//! `ay` on `PATH`, then the rustup-linked `trust` toolchain.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

use ny_core::Bound;
use ny_propagate::{build_difference_network, GraphNetwork, Layer, NETWORK_INPUT};
use ny_tensor::next_down_f32;

use crate::exact::rational;
use crate::verify::Relation;

/// Why an SMT escalation could not run (as opposed to running and answering
/// [`SmtVerdict::Unknown`], which is a sound non-answer from the solver).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EscalateError {
    /// The difference network contains a layer outside the encodable set
    /// (`Linear`/`ReLU`/`PowConstant`/`Add`/`Sub`/`MinBinary`/`MaxBinary`).
    #[error(
        "layer `{layer}` is not encodable for SMT escalation (supported: Linear, ReLU, \
         PowConstant with small integer exponent, Add, Sub, MinBinary, MaxBinary)"
    )]
    UnsupportedLayer {
        /// Short name of the offending layer.
        layer: String,
    },

    /// A `PowConstant` exponent is not an integer in `1..=4` — the encoding
    /// expands powers to products, and the M1 builders only emit squares.
    #[error("PowConstant exponent {exponent} is not an integer in 1..=4")]
    UnsupportedExponent {
        /// Offending exponent.
        exponent: f32,
    },

    /// A graph constant is NaN or infinite and has no exact rational value.
    #[error("{context} is not finite: {value}")]
    NonFiniteConstant {
        /// Which constant (layer/row description).
        context: String,
        /// Offending value.
        value: f64,
    },

    /// The input box has a non-finite endpoint; the query needs a compact box.
    #[error("input bound {index} is not finite: [{lower}, {upper}]")]
    NonFiniteBounds {
        /// Input dimension index.
        index: usize,
        /// Lower endpoint.
        lower: f32,
        /// Upper endpoint.
        upper: f32,
    },

    /// The `AbsBound` epsilon is not a strictly positive finite value (after
    /// the same sound f32 round-down the float verify path applies).
    #[error("absbound epsilon {value} is invalid: {reason}")]
    InvalidEpsilon {
        /// Requested epsilon.
        value: f64,
        /// Why it was rejected.
        reason: String,
    },

    /// The difference network is structurally unusable (dangling node
    /// reference, arity mismatch, missing output).
    #[error("difference network shape error: {0}")]
    GraphShape(String),

    /// Writing the query or spawning the solver failed.
    #[error("solver io: {0}")]
    Io(#[from] std::io::Error),

    /// An underlying ny error (difference-network construction, topo sort).
    #[error(transparent)]
    Ny(#[from] ny_core::NyError),
}

/// Result alias for this module.
pub type EscalateResult<T> = Result<T, EscalateError>;

/// Options for [`SmtEscalation::escalate`].
#[derive(Debug, Clone)]
pub struct EscalateOptions {
    /// Wall-clock budget passed to AY as `-t <ms>` (AY answers a sound
    /// `unknown` on exhaustion). `None` runs without a budget.
    pub timeout_ms: Option<u64>,
    /// For [`Relation::Dominates`]: encode the violation as `h(x) <= 0` so
    /// that `unsat` proves the *strict* margin `f(x) > g(x)` (needed when the
    /// unsafe clause is non-strict, e.g. VNN-LIB `(<= Y_f Y_g)`). Ignored for
    /// [`Relation::AbsBound`].
    pub require_strict_margin: bool,
}

impl Default for EscalateOptions {
    fn default() -> Self {
        Self {
            timeout_ms: Some(60_000),
            require_strict_margin: false,
        }
    }
}

/// Verdict of one SMT escalation.
#[derive(Debug)]
#[non_exhaustive]
pub enum SmtVerdict {
    /// `unsat`: the relation holds on the whole box (exact real semantics).
    Proved {
        /// The `.smt2` query file (kept for audit).
        query: PathBuf,
        /// Alethe certificate AY wrote next to the query, when present
        /// (AY emits proofs by default; absence is reported, not hidden).
        certificate: Option<PathBuf>,
    },
    /// `sat` with a model that **re-validated in exact rational arithmetic**:
    /// a genuine counterexample.
    Falsified {
        /// Witness point, rounded once to f64 for display.
        witness: Vec<f64>,
        /// Exact rational witness coordinates (`"n/d"` or `"n"`).
        witness_exact: Vec<String>,
        /// Index of the violating output of `h`.
        output_index: usize,
        /// Exact rational value of the violating `h` output at the witness.
        difference_exact: String,
        /// The `.smt2` query file.
        query: PathBuf,
    },
    /// `sat`, but the printed model does not check out under exact rational
    /// evaluation (AY prints placeholder models for Krawczyk-certified
    /// irrational solutions). A violation *exists* by the solver's existence
    /// certificate, but no concrete validated witness can be reported.
    ViolationExists {
        /// What failed to validate (missing variables, out-of-box point, or
        /// a non-violating evaluation).
        detail: String,
        /// The `.smt2` query file.
        query: PathBuf,
    },
    /// Solver `unknown` (including timeout) or unparsable output — a sound
    /// non-answer.
    Unknown {
        /// Solver-reported reason when available.
        reason: String,
        /// The `.smt2` query file.
        query: PathBuf,
    },
}

/// Handle to a located `ay` binary plus a working directory where query and
/// certificate files are written (and kept, for audit).
#[derive(Debug)]
pub struct SmtEscalation {
    ay: PathBuf,
    workdir: PathBuf,
}

impl SmtEscalation {
    /// Locate `ay`: `$NY_AY` explicitly, then `ay` on `PATH`, then the
    /// rustup-linked `trust` toolchain. Returns `None` (rather than erroring)
    /// so callers and tests can degrade gracefully on machines without AY.
    pub fn locate() -> Option<Self> {
        let rustup_trust_ay = std::env::var_os("RUSTUP_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".rustup")))
            .map(|root| root.join("toolchains/trust/bin/ay"));
        let candidates = [
            std::env::var_os("NY_AY").map(PathBuf::from),
            which("ay"),
            rustup_trust_ay,
        ];
        let ay = candidates.into_iter().flatten().find(|p| p.is_file())?;
        let workdir = std::env::temp_dir().join(format!("ny-gt-smt-{}", std::process::id()));
        std::fs::create_dir_all(&workdir).ok()?;
        Some(Self { ay, workdir })
    }

    /// Use an explicit solver binary and working directory (no probing).
    pub fn with_solver(ay: PathBuf, workdir: PathBuf) -> Self {
        Self { ay, workdir }
    }

    /// Where query and certificate files are written.
    pub fn workdir(&self) -> &Path {
        &self.workdir
    }

    /// Escalate `f ⋈ g` on `input_bounds` to one exact AY query over the
    /// difference network (module docs: encoding, semantics, verdicts).
    ///
    /// # Errors
    /// [`EscalateError`] when the query cannot be *built or run* (unsupported
    /// layer, non-finite constant/box, io). Solver non-answers are not errors;
    /// they come back as [`SmtVerdict::Unknown`].
    pub fn escalate(
        &self,
        f: &GraphNetwork,
        g: &GraphNetwork,
        relation: Relation,
        input_bounds: &[Bound],
        options: &EscalateOptions,
    ) -> EscalateResult<SmtVerdict> {
        let violation = Violation::for_relation(relation, options.require_strict_margin)?;
        let h = build_difference_network(f, g)?;
        let encoded = encode_violation_query(&h, input_bounds, &violation)?;

        let (answer, query, stdout) = self.run_solver(&encoded.text, options.timeout_ms)?;
        match answer {
            SolverAnswer::Unsat => {
                let cert = PathBuf::from(format!("{}.alethe", query.display()));
                Ok(SmtVerdict::Proved {
                    certificate: cert.is_file().then_some(cert),
                    query,
                })
            }
            SolverAnswer::Sat => Ok(validate_sat_model(
                &h,
                input_bounds,
                &violation,
                &stdout,
                query,
            )),
            SolverAnswer::Unknown { reason } => Ok(SmtVerdict::Unknown { reason, query }),
        }
    }

    /// Write the query to a fresh file, run `ay solve` and parse the verdict
    /// line (`sat`/`unsat`/`unknown`; AY prefixes diagnostics with `c `).
    fn run_solver(
        &self,
        query_text: &str,
        timeout_ms: Option<u64>,
    ) -> EscalateResult<(SolverAnswer, PathBuf, String)> {
        static QUERY_SEQ: AtomicU64 = AtomicU64::new(0);
        let n = QUERY_SEQ.fetch_add(1, Ordering::Relaxed);
        let query = self.workdir.join(format!("gt-escalate-q{n}.smt2"));
        std::fs::write(&query, query_text)?;

        let mut cmd = Command::new(&self.ay);
        cmd.arg("solve");
        if let Some(ms) = timeout_ms {
            cmd.arg("-t").arg(ms.to_string());
        }
        let out = cmd.arg(&query).output()?;
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();

        let verdict = stdout.lines().find_map(|l| match l.trim() {
            "sat" => Some(SolverAnswer::Sat),
            "unsat" => Some(SolverAnswer::Unsat),
            "unknown" => Some(SolverAnswer::Unknown {
                reason: parse_unknown_reason(&stdout),
            }),
            _ => None,
        });
        let answer = verdict.unwrap_or_else(|| SolverAnswer::Unknown {
            reason: format!(
                "no verdict line in solver output (exit {:?}; stderr: {})",
                out.status.code(),
                String::from_utf8_lossy(&out.stderr)
                    .lines()
                    .next()
                    .unwrap_or("")
            ),
        });
        Ok((answer, query, stdout))
    }
}

/// Parsed solver verdict.
#[derive(Debug)]
enum SolverAnswer {
    Sat,
    Unsat,
    Unknown { reason: String },
}

/// The violation being asserted (whose refutation proves the relation).
#[derive(Debug, Clone)]
enum Violation {
    /// `h(x) < 0` — refuting it proves `f >= g`.
    DominatesBelowZero,
    /// `h(x) <= 0` — refuting it proves the strict `f > g`.
    DominatesAtOrBelowZero,
    /// `|h(x)| > eps` — refuting it proves `|f − g| <= eps`.
    OutsideAbsBound(BigRational),
}

impl Violation {
    fn for_relation(relation: Relation, require_strict_margin: bool) -> EscalateResult<Self> {
        match relation {
            Relation::Dominates => Ok(if require_strict_margin {
                Self::DominatesAtOrBelowZero
            } else {
                Self::DominatesBelowZero
            }),
            Relation::AbsBound(eps) => {
                if !eps.is_finite() {
                    return Err(EscalateError::InvalidEpsilon {
                        value: eps,
                        reason: "must be finite".to_string(),
                    });
                }
                // Same sound round-down as the float verify path: the checked
                // property is never weaker than requested.
                let mut eps32 = eps as f32;
                if f64::from(eps32) > eps {
                    eps32 = next_down_f32(eps32);
                }
                if eps32 <= 0.0 {
                    return Err(EscalateError::InvalidEpsilon {
                        value: eps,
                        reason: "must be strictly positive after sound f32 rounding".to_string(),
                    });
                }
                Ok(Self::OutsideAbsBound(rational(eps32)))
            }
        }
    }

    /// One disjunct of the violation clause for output term `t`.
    fn disjunct(&self, t: &str) -> String {
        match self {
            Self::DominatesBelowZero => format!("(< {t} 0.0)"),
            Self::DominatesAtOrBelowZero => format!("(<= {t} 0.0)"),
            Self::OutsideAbsBound(eps) => {
                let e = smt_rat(eps);
                format!("(or (> {t} {e}) (< {t} (- {e})))")
            }
        }
    }

    /// Does the exact output vector violate the relation? Returns the first
    /// violating index and value.
    fn violated_by(&self, outputs: &[BigRational]) -> Option<(usize, BigRational)> {
        outputs.iter().enumerate().find_map(|(i, v)| {
            let hit = match self {
                Self::DominatesBelowZero => v.is_negative(),
                Self::DominatesAtOrBelowZero => !v.is_positive(),
                Self::OutsideAbsBound(eps) => v.abs() > *eps,
            };
            hit.then(|| (i, v.clone()))
        })
    }
}

/// An encoded query plus the facts tests care about.
struct EncodedQuery {
    text: String,
    /// True when a `PowConstant` exponent >= 2 forced `QF_NRA`.
    #[cfg_attr(not(test), allow(dead_code))]
    nonlinear: bool,
}

/// Encode `exists x in box: violation(h(x))` as one SMT-LIB2 query.
///
/// Affine/polynomial layers are inlined as terms; ReLU/min/max get named
/// `ite`-defined variables (module docs: measured solver friendliness).
fn encode_violation_query(
    h: &GraphNetwork,
    input_bounds: &[Bound],
    violation: &Violation,
) -> EscalateResult<EncodedQuery> {
    if input_bounds.is_empty() {
        return Err(EscalateError::GraphShape(
            "input box has zero dimensions".to_string(),
        ));
    }
    let mut decls = String::new();
    let mut defs = String::new();
    for (i, b) in input_bounds.iter().enumerate() {
        if !b.lower().is_finite() || !b.upper().is_finite() {
            return Err(EscalateError::NonFiniteBounds {
                index: i,
                lower: b.lower(),
                upper: b.upper(),
            });
        }
        let lo = smt_rat(&rational(b.lower()));
        let hi = smt_rat(&rational(b.upper()));
        let _ = writeln!(decls, "(declare-const x{i} Real)");
        let _ = writeln!(defs, "(assert (and (>= x{i} {lo}) (<= x{i} {hi})))");
    }

    let mut fresh = Fresh::default();
    let mut nonlinear = false;
    let mut terms: HashMap<String, Vec<String>> = HashMap::new();
    terms.insert(
        NETWORK_INPUT.to_string(),
        (0..input_bounds.len()).map(|i| format!("x{i}")).collect(),
    );

    let order = h.exec_order()?.to_vec();
    for name in &order {
        let node = h
            .node(name)
            .ok_or_else(|| EscalateError::GraphShape(format!("dangling node '{name}'")))?;
        let ins: Vec<&Vec<String>> = node
            .inputs()
            .iter()
            .map(|input| {
                terms.get(input).ok_or_else(|| {
                    EscalateError::GraphShape(format!(
                        "node '{name}' reads '{input}' before it is defined"
                    ))
                })
            })
            .collect::<EscalateResult<_>>()?;

        let out = match node.layer() {
            Layer::Linear(lin) => encode_linear(lin, single_input(name, &ins)?)?,
            Layer::ReLU(_) => single_input(name, &ins)?
                .iter()
                .map(|t| {
                    define_fresh(
                        &mut decls,
                        &mut defs,
                        &mut fresh,
                        &format!("(ite (>= {t} 0.0) {t} 0.0)"),
                    )
                })
                .collect(),
            Layer::PowConstant(p) => {
                let (exp, poly) = validated_exponent(p.exponent())?;
                nonlinear |= poly;
                single_input(name, &ins)?
                    .iter()
                    .map(|t| power_term(t, exp))
                    .collect()
            }
            Layer::Add(_) => elementwise(name, &ins, |a, b| format!("(+ {a} {b})"))?,
            Layer::Sub(_) => elementwise(name, &ins, |a, b| format!("(- {a} {b})"))?,
            Layer::MinBinary(_) => {
                elementwise(name, &ins, |a, b| format!("(ite (<= {a} {b}) {a} {b})"))?
                    .iter()
                    .map(|t| define_fresh(&mut decls, &mut defs, &mut fresh, t))
                    .collect()
            }
            Layer::MaxBinary(_) => {
                elementwise(name, &ins, |a, b| format!("(ite (>= {a} {b}) {a} {b})"))?
                    .iter()
                    .map(|t| define_fresh(&mut decls, &mut defs, &mut fresh, t))
                    .collect()
            }
            other => {
                return Err(EscalateError::UnsupportedLayer {
                    layer: layer_name(other).to_string(),
                })
            }
        };
        terms.insert(name.clone(), out);
    }

    let outputs = terms.get(h.output_name()).ok_or_else(|| {
        EscalateError::GraphShape(format!(
            "output node '{}' produced no terms",
            h.output_name()
        ))
    })?;
    let disjuncts: Vec<String> = outputs.iter().map(|t| violation.disjunct(t)).collect();
    let clause = match disjuncts.as_slice() {
        [only] => only.clone(),
        many => format!("(or {})", many.join(" ")),
    };

    let logic = if nonlinear { "QF_NRA" } else { "QF_LRA" };
    let text =
        format!("(set-logic {logic})\n{decls}{defs}(assert {clause})\n(check-sat)\n(get-model)\n");
    Ok(EncodedQuery { text, nonlinear })
}

/// The single input vector of a unary node.
fn single_input<'t, T>(name: &str, ins: &[&'t Vec<T>]) -> EscalateResult<&'t Vec<T>> {
    match ins {
        [only] => Ok(only),
        _ => Err(EscalateError::GraphShape(format!(
            "node '{name}' has {} inputs (expected 1)",
            ins.len()
        ))),
    }
}

/// `which(1)` without a dependency: first `PATH` entry containing `name`.
fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(name))
        .find(|p| p.is_file())
}

/// Element-wise combination of a binary node's two input term vectors.
fn elementwise(
    name: &str,
    ins: &[&Vec<String>],
    op: impl Fn(&str, &str) -> String,
) -> EscalateResult<Vec<String>> {
    let [a, b] = ins else {
        return Err(EscalateError::GraphShape(format!(
            "node '{name}' has {} inputs (expected 2)",
            ins.len()
        )));
    };
    if a.len() != b.len() {
        return Err(EscalateError::GraphShape(format!(
            "node '{name}' input arities differ: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| op(x, y)).collect())
}

/// Fresh-variable allocator with common-subexpression elimination: identical
/// definitions share one variable. (Duplicate neurons are common in
/// difference networks, and every redundant variable measurably hurts AY's
/// interval contraction — see the module docs.)
#[derive(Default)]
struct Fresh {
    next: usize,
    interned: HashMap<String, String>,
}

/// Declare a fresh variable `v<k>` defined equal to `term` (or return the
/// existing variable already defined equal to the same term).
fn define_fresh(decls: &mut String, defs: &mut String, fresh: &mut Fresh, term: &str) -> String {
    if let Some(existing) = fresh.interned.get(term) {
        return existing.clone();
    }
    let v = format!("v{}", fresh.next);
    fresh.next += 1;
    let _ = writeln!(decls, "(declare-const {v} Real)");
    let _ = writeln!(defs, "(assert (= {v} {term}))");
    fresh.interned.insert(term.to_string(), v.clone());
    v
}

/// Encode one Linear layer as inlined affine terms (exact rational constants;
/// `1.0`/`-1.0` coefficients stay bare for solver-friendly shapes).
fn encode_linear(
    lin: &ny_propagate::layers::LinearLayer,
    input: &[String],
) -> EscalateResult<Vec<String>> {
    let w = &lin.weight;
    if w.ncols() != input.len() {
        return Err(EscalateError::GraphShape(format!(
            "Linear expects {} inputs, got {}",
            w.ncols(),
            input.len()
        )));
    }
    let mut out = Vec::with_capacity(w.nrows());
    for (i, row) in w.rows().into_iter().enumerate() {
        let mut addends: Vec<String> = Vec::new();
        for (v, term) in row.iter().zip(input) {
            let c = finite("Linear weight", *v)?;
            if c == 0.0 {
                continue;
            }
            addends.push(if c == 1.0 {
                term.clone()
            } else if c == -1.0 {
                format!("(- {term})")
            } else {
                format!("(* {} {term})", smt_rat(&rational(c)))
            });
        }
        let bias = match &lin.bias {
            Some(b) => finite("Linear bias", b[i])?,
            None => 0.0,
        };
        if bias != 0.0 || addends.is_empty() {
            addends.push(smt_rat(&rational(bias)));
        }
        out.push(match addends.as_slice() {
            [only] => only.clone(),
            many => format!("(+ {})", many.join(" ")),
        });
    }
    Ok(out)
}

/// Validate a `PowConstant` exponent: integer in `1..=4`. Returns the integer
/// and whether it makes the query nonlinear.
fn validated_exponent(exponent: f32) -> EscalateResult<(u32, bool)> {
    let is_small_int = exponent.fract() == 0.0 && (1.0..=4.0).contains(&exponent);
    if !is_small_int {
        return Err(EscalateError::UnsupportedExponent { exponent });
    }
    let e = exponent as u32;
    Ok((e, e >= 2))
}

/// `t^e` as an e-fold product (`e` validated small).
fn power_term(t: &str, e: u32) -> String {
    if e == 1 {
        t.to_string()
    } else {
        let copies = vec![t; e as usize];
        format!("(* {})", copies.join(" "))
    }
}

/// Reject non-finite f32 graph constants (no exact rational value).
fn finite(context: &str, v: f32) -> EscalateResult<f32> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(EscalateError::NonFiniteConstant {
            context: context.to_string(),
            value: f64::from(v),
        })
    }
}

/// Short display name of a layer (its `Debug` form prints whole weight
/// arrays; the error message only needs the variant).
fn layer_name(layer: &Layer) -> &'static str {
    match layer {
        Layer::Linear(_) => "Linear",
        Layer::ReLU(_) => "ReLU",
        Layer::PowConstant(_) => "PowConstant",
        Layer::Add(_) => "Add",
        Layer::Sub(_) => "Sub",
        Layer::MinBinary(_) => "MinBinary",
        Layer::MaxBinary(_) => "MaxBinary",
        _ => "unsupported",
    }
}

/// One SMT-LIB2 rational literal: `n.0`, `(/ n.0 d.0)`, negatives wrapped as
/// `(- …)` (SMT-LIB has no negative numerals).
fn smt_rat(q: &BigRational) -> String {
    let mag_num = q.numer().magnitude();
    let den = q.denom().magnitude(); // canonical form: denominator > 0
    let core = if q.denom().is_one() {
        format!("{mag_num}.0")
    } else {
        format!("(/ {mag_num}.0 {den}.0)")
    };
    if q.numer().is_negative() {
        format!("(- {core})")
    } else {
        core
    }
}

/// `(:reason-unknown …)` payload when AY printed one.
fn parse_unknown_reason(stdout: &str) -> String {
    stdout
        .lines()
        .find_map(|l| {
            let l = l.trim();
            l.strip_prefix("(:reason-unknown")
                .map(|rest| rest.trim_end_matches(')').trim().to_string())
        })
        .unwrap_or_else(|| "solver returned unknown".to_string())
}

/// Validate a `sat` model against exact rational evaluation of `h`; produce
/// [`SmtVerdict::Falsified`] only when the witness checks out end to end.
fn validate_sat_model(
    h: &GraphNetwork,
    input_bounds: &[Bound],
    violation: &Violation,
    stdout: &str,
    query: PathBuf,
) -> SmtVerdict {
    let model = parse_model(stdout);
    let mut point = Vec::with_capacity(input_bounds.len());
    for i in 0..input_bounds.len() {
        match model.get(&format!("x{i}")) {
            Some(v) => point.push(v.clone()),
            None => {
                return SmtVerdict::ViolationExists {
                    detail: format!("model does not assign input x{i}"),
                    query,
                }
            }
        }
    }
    for (i, (v, b)) in point.iter().zip(input_bounds).enumerate() {
        if *v < rational(b.lower()) || *v > rational(b.upper()) {
            return SmtVerdict::ViolationExists {
                detail: format!("model point leaves the box at dimension {i}"),
                query,
            };
        }
    }
    let outputs = match eval_exact(h, &point) {
        Ok(outputs) => outputs,
        Err(e) => {
            return SmtVerdict::ViolationExists {
                detail: format!("exact evaluation of the model failed: {e}"),
                query,
            }
        }
    };
    match violation.violated_by(&outputs) {
        Some((output_index, value)) => SmtVerdict::Falsified {
            witness: point
                .iter()
                .map(|v| v.to_f64().unwrap_or(f64::NAN))
                .collect(),
            witness_exact: point.iter().map(rat_string).collect(),
            output_index,
            difference_exact: rat_string(&value),
            query,
        },
        None => SmtVerdict::ViolationExists {
            detail: format!(
                "model point does not violate the relation under exact evaluation \
                 (placeholder model for a certified-existence answer); h = [{}]",
                outputs
                    .iter()
                    .map(rat_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            query,
        },
    }
}

/// `"n"` / `"n/d"` rendering of an exact rational.
fn rat_string(q: &BigRational) -> String {
    if q.denom().is_one() {
        q.numer().to_string()
    } else {
        format!("{}/{}", q.numer(), q.denom())
    }
}

/// Evaluate the difference network at an exact rational point (same layer set
/// as the encoder; this is the validation oracle for `sat` models).
fn eval_exact(h: &GraphNetwork, point: &[BigRational]) -> EscalateResult<Vec<BigRational>> {
    let mut values: HashMap<String, Vec<BigRational>> = HashMap::new();
    values.insert(NETWORK_INPUT.to_string(), point.to_vec());

    let order = h.exec_order()?.to_vec();
    for name in &order {
        let node = h
            .node(name)
            .ok_or_else(|| EscalateError::GraphShape(format!("dangling node '{name}'")))?;
        let ins: Vec<&Vec<BigRational>> = node
            .inputs()
            .iter()
            .map(|input| {
                values.get(input).ok_or_else(|| {
                    EscalateError::GraphShape(format!(
                        "node '{name}' reads '{input}' before it is defined"
                    ))
                })
            })
            .collect::<EscalateResult<_>>()?;

        let out: Vec<BigRational> = match node.layer() {
            Layer::Linear(lin) => {
                let input = single_input(name, &ins)?;
                let w = &lin.weight;
                if w.ncols() != input.len() {
                    return Err(EscalateError::GraphShape(format!(
                        "Linear expects {} inputs, got {}",
                        w.ncols(),
                        input.len()
                    )));
                }
                w.rows()
                    .into_iter()
                    .enumerate()
                    .map(|(i, row)| {
                        let mut acc = match &lin.bias {
                            Some(b) => rational(finite("Linear bias", b[i])?),
                            None => BigRational::zero(),
                        };
                        for (v, x) in row.iter().zip(input) {
                            acc += rational(finite("Linear weight", *v)?) * x;
                        }
                        Ok(acc)
                    })
                    .collect::<EscalateResult<_>>()?
            }
            Layer::ReLU(_) => single_input(name, &ins)?
                .iter()
                .map(|v| {
                    if v.is_negative() {
                        BigRational::zero()
                    } else {
                        v.clone()
                    }
                })
                .collect(),
            Layer::PowConstant(p) => {
                let (exp, _) = validated_exponent(p.exponent())?;
                single_input(name, &ins)?
                    .iter()
                    .map(|v| {
                        let mut acc = v.clone();
                        for _ in 1..exp {
                            acc *= v;
                        }
                        acc
                    })
                    .collect()
            }
            Layer::Add(_) => zip_exact(name, &ins, |a, b| a + b)?,
            Layer::Sub(_) => zip_exact(name, &ins, |a, b| a - b)?,
            Layer::MinBinary(_) => zip_exact(name, &ins, |a, b| if a <= b { a } else { b })?,
            Layer::MaxBinary(_) => zip_exact(name, &ins, |a, b| if a >= b { a } else { b })?,
            other => {
                return Err(EscalateError::UnsupportedLayer {
                    layer: layer_name(other).to_string(),
                })
            }
        };
        values.insert(name.clone(), out);
    }
    values
        .remove(h.output_name())
        .ok_or_else(|| EscalateError::GraphShape("output node produced no values".to_string()))
}

/// Element-wise exact combination of a binary node's inputs.
fn zip_exact(
    name: &str,
    ins: &[&Vec<BigRational>],
    op: impl Fn(BigRational, BigRational) -> BigRational,
) -> EscalateResult<Vec<BigRational>> {
    let [a, b] = ins else {
        return Err(EscalateError::GraphShape(format!(
            "node '{name}' has {} inputs (expected 2)",
            ins.len()
        )));
    };
    if a.len() != b.len() {
        return Err(EscalateError::GraphShape(format!(
            "node '{name}' input arities differ: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| op(x.clone(), y.clone()))
        .collect())
}

/// Parse `(define-fun NAME () Real VALUE)` bindings out of solver stdout.
/// Values: numerals (`3`, `1.5`), `(/ a b)`, `(- v)` — anything else (e.g.
/// algebraic-number placeholders) is skipped, and validation fails honestly.
fn parse_model(stdout: &str) -> HashMap<String, BigRational> {
    let tokens = tokenize(stdout);
    let mut model = HashMap::new();
    let mut i = 0;
    while i < tokens.len() {
        // Look for: ( define-fun NAME ( ) Real VALUE )
        if tokens[i] == "("
            && tokens.get(i + 1).map(String::as_str) == Some("define-fun")
            && tokens.get(i + 3).map(String::as_str) == Some("(")
            && tokens.get(i + 4).map(String::as_str) == Some(")")
            && tokens.get(i + 5).map(String::as_str) == Some("Real")
        {
            let name = tokens[i + 2].clone();
            if let Some((value, used)) = parse_value(&tokens[i + 6..]) {
                model.insert(name, value);
                i += 6 + used;
                continue;
            }
        }
        i += 1;
    }
    model
}

/// Split into parens and atoms.
fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        match c {
            '(' | ')' => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                tokens.push(c.to_string());
            }
            c if c.is_whitespace() => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            c => current.push(c),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Parse one rational value at the head of `tokens`; returns the value and
/// how many tokens it consumed.
fn parse_value(tokens: &[String]) -> Option<(BigRational, usize)> {
    match tokens.first().map(String::as_str) {
        Some("(") => match tokens.get(1).map(String::as_str) {
            Some("-") => {
                let (inner, used) = parse_value(&tokens[2..])?;
                (tokens.get(2 + used).map(String::as_str) == Some(")")).then(|| (-inner, used + 3))
            }
            Some("/") => {
                let (num, used_n) = parse_value(&tokens[2..])?;
                let (den, used_d) = parse_value(&tokens[2 + used_n..])?;
                if den.is_zero() {
                    return None;
                }
                (tokens.get(2 + used_n + used_d).map(String::as_str) == Some(")"))
                    .then(|| (num / den, used_n + used_d + 3))
            }
            _ => None,
        },
        Some(atom) => parse_numeral(atom).map(|q| (q, 1)),
        None => None,
    }
}

/// Parse a decimal numeral (`"3"`, `"1.5"`) as an exact rational (a decimal
/// literal denotes digits/10^k exactly — no float rounding).
fn parse_numeral(atom: &str) -> Option<BigRational> {
    let (int_part, frac_part) = match atom.split_once('.') {
        Some((i, f)) => (i, f),
        None => (atom, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }
    let all_digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !(int_part.is_empty() || all_digits(int_part))
        || !(frac_part.is_empty() || all_digits(frac_part))
    {
        return None;
    }
    let digits = format!("{int_part}{frac_part}");
    let numer: BigInt = digits.parse().ok()?;
    let denom = BigInt::from(10u32).pow(u32::try_from(frac_part.len()).ok()?);
    Some(BigRational::new(numer, denom))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::{signed_plane_distance, sphere_residual};
    use ndarray::{Array1, Array2};
    use ny_propagate::layers::{LinearLayer, ReLULayer};
    use ny_propagate::GraphNode;

    fn rat(n: i64, d: i64) -> BigRational {
        BigRational::new(BigInt::from(n), BigInt::from(d))
    }

    #[test]
    fn smt_rat_literals_are_exact_and_wrapped() {
        assert_eq!(smt_rat(&rat(3, 1)), "3.0");
        assert_eq!(smt_rat(&rat(-3, 1)), "(- 3.0)");
        assert_eq!(smt_rat(&rat(3, 4)), "(/ 3.0 4.0)");
        assert_eq!(smt_rat(&rat(-3, 4)), "(- (/ 3.0 4.0))");
        assert_eq!(smt_rat(&BigRational::zero()), "0.0");
    }

    #[test]
    fn model_parser_handles_ay_shapes() {
        let stdout = r#"
c writing Alethe proof to q.smt2.alethe
sat
(model
  (define-fun x0 () Real 0.0)
  (define-fun x1 () Real (- 1.0))
  (define-fun x2 () Real (/ 3.0 4.0))
  (define-fun v0 () Real (- (/ 1.0 3.0)))
  (define-fun n7 () Real 2)
)
"#;
        let model = parse_model(stdout);
        assert_eq!(model["x0"], BigRational::zero());
        assert_eq!(model["x1"], rat(-1, 1));
        assert_eq!(model["x2"], rat(3, 4));
        assert_eq!(model["v0"], rat(-1, 3));
        assert_eq!(model["n7"], rat(2, 1));
    }

    #[test]
    fn numeral_parser_is_exact_for_decimals() {
        // 0.1 must be exactly 1/10, not the f64 nearest to it.
        assert_eq!(parse_numeral("0.1"), Some(rat(1, 10)));
        assert_eq!(parse_numeral("1.5"), Some(rat(3, 2)));
        assert_eq!(parse_numeral("12"), Some(rat(12, 1)));
        assert_eq!(parse_numeral("."), None);
        assert_eq!(parse_numeral("abc"), None);
        assert_eq!(parse_numeral("1e5"), None);
    }

    #[test]
    fn unknown_reason_is_extracted() {
        assert_eq!(
            parse_unknown_reason("unknown\n(:reason-unknown incomplete)\n"),
            "incomplete"
        );
        assert_eq!(parse_unknown_reason("unknown\n"), "solver returned unknown");
    }

    /// f(x) = w * (relu(x0) + relu(-x0) + relu(x1) + relu(-x1) + relu(x2) +
    /// relu(-x2)) + bias = w * (|x0| + |x1| + |x2|) + bias as a genuine
    /// 3 -> 6 -> 1 FC-ReLU GraphNetwork.
    fn abs_net(weight: f32, bias: f32) -> GraphNetwork {
        let w1 = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 0.0, 0.0, //
                -1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, -1.0, 0.0, //
                0.0, 0.0, 1.0, //
                0.0, 0.0, -1.0,
            ],
        )
        .expect("shape");
        let w2 = Array2::from_shape_vec((1, 6), vec![weight; 6]).expect("shape");
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "lin1",
            Layer::Linear(LinearLayer::new(w1, None).expect("valid linear")),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "readout",
            Layer::Linear(
                LinearLayer::new(w2, Some(Array1::from(vec![bias]))).expect("valid linear"),
            ),
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
    fn encoding_inlines_affine_and_names_relu() {
        let f = abs_net(2.0, -0.75);
        let g = sphere_residual([0.0, 0.0, 0.0], 1.0).expect("sphere");
        let h = build_difference_network(&f, &g).expect("difference network");
        let q = encode_violation_query(&h, &unit_box(), &Violation::DominatesBelowZero)
            .expect("encodes");
        assert!(q.nonlinear, "sphere side has squares");
        assert!(q.text.starts_with("(set-logic QF_NRA)"));
        // ReLU neurons become named ite-defined variables...
        assert_eq!(q.text.matches("(declare-const v").count(), 6);
        assert!(q.text.contains("(ite (>= "));
        // ...while Linear pre-activations are inlined: no equality whose rhs
        // is a bare input variable (the shape that broke ICP contraction).
        assert!(!q.text.contains("(= v0 x0)"));
        assert!(q.text.contains("(assert (< "), "violation clause present");
        assert!(q.text.contains("(get-model)"));
    }

    #[test]
    fn encoding_is_linear_logic_without_powers() {
        let f = abs_net(1.0, 3.0);
        let g = signed_plane_distance([0.0, 0.0, 1.0], -0.5).expect("plane");
        let h = build_difference_network(&f, &g).expect("difference network");
        let q = encode_violation_query(&h, &unit_box(), &Violation::DominatesBelowZero)
            .expect("encodes");
        assert!(!q.nonlinear);
        assert!(q.text.starts_with("(set-logic QF_LRA)"));
    }

    #[test]
    fn encoding_rejects_unsupported_exponent_and_infinite_box() {
        let g = sphere_residual([0.0, 0.0, 0.0], 1.0).expect("sphere");
        let h = build_difference_network(&g, &g).expect("difference network");
        let bad_box = vec![
            Bound::new(-1.0, 1.0),
            Bound::new_allow_infinite(f32::NEG_INFINITY, 1.0),
            Bound::new(-1.0, 1.0),
        ];
        assert!(matches!(
            encode_violation_query(&h, &bad_box, &Violation::DominatesBelowZero),
            Err(EscalateError::NonFiniteBounds { index: 1, .. })
        ));
        assert!(matches!(
            validated_exponent(2.5),
            Err(EscalateError::UnsupportedExponent { .. })
        ));
        assert!(matches!(
            validated_exponent(7.0),
            Err(EscalateError::UnsupportedExponent { .. })
        ));
        assert_eq!(validated_exponent(2.0).expect("square"), (2, true));
        assert_eq!(validated_exponent(1.0).expect("identity"), (1, false));
    }

    #[test]
    fn exact_eval_matches_reference_on_difference_network() {
        // h = f - g with f = 2*sum|xi| - 3/4, g = ||x||^2 - 1.
        let f = abs_net(2.0, -0.75);
        let g = sphere_residual([0.0, 0.0, 0.0], 1.0).expect("sphere");
        let h = build_difference_network(&f, &g).expect("difference network");
        let point = vec![rat(1, 2), rat(-1, 4), BigRational::zero()];
        // f = 2*(1/2 + 1/4 + 0) - 3/4 = 3/4; g = 1/4 + 1/16 - 1 = -11/16.
        // h = 3/4 + 11/16 = 23/16.
        let out = eval_exact(&h, &point).expect("evaluates");
        assert_eq!(out, vec![rat(23, 16)]);
    }

    #[test]
    fn placeholder_model_is_downgraded_not_reported_as_witness() {
        // A model that does NOT violate: validation must refuse the witness.
        let f = abs_net(2.0, -0.75);
        let g = sphere_residual([0.0, 0.0, 0.0], 1.0).expect("sphere");
        let h = build_difference_network(&f, &g).expect("difference network");
        let stdout = "sat\n(model\n  (define-fun x0 () Real 0.0)\n  \
                      (define-fun x1 () Real 0.0)\n  (define-fun x2 () Real 0.0)\n)\n";
        let verdict = validate_sat_model(
            &h,
            &unit_box(),
            &Violation::DominatesBelowZero,
            stdout,
            PathBuf::from("q.smt2"),
        );
        assert!(
            matches!(verdict, SmtVerdict::ViolationExists { ref detail, .. }
                if detail.contains("does not violate")),
            "got {verdict:?}"
        );

        // Out-of-box and missing-variable models are refused likewise.
        let out_of_box = "sat\n(model\n  (define-fun x0 () Real 7.0)\n  \
                          (define-fun x1 () Real 0.0)\n  (define-fun x2 () Real 0.0)\n)\n";
        assert!(matches!(
            validate_sat_model(
                &h,
                &unit_box(),
                &Violation::DominatesBelowZero,
                out_of_box,
                PathBuf::from("q.smt2"),
            ),
            SmtVerdict::ViolationExists { .. }
        ));
        let missing = "sat\n(model\n  (define-fun x0 () Real 0.0)\n)\n";
        assert!(matches!(
            validate_sat_model(
                &h,
                &unit_box(),
                &Violation::DominatesBelowZero,
                missing,
                PathBuf::from("q.smt2"),
            ),
            SmtVerdict::ViolationExists { .. }
        ));
    }

    #[test]
    fn genuine_violating_model_is_accepted_as_witness() {
        // f = 2*sum|xi| - 5/4 vs g = sphere: h(0) = -5/4 + 1 = -1/4 < 0.
        let f = abs_net(2.0, -1.25);
        let g = sphere_residual([0.0, 0.0, 0.0], 1.0).expect("sphere");
        let h = build_difference_network(&f, &g).expect("difference network");
        let stdout = "sat\n(model\n  (define-fun x0 () Real 0.0)\n  \
                      (define-fun x1 () Real 0.0)\n  (define-fun x2 () Real 0.0)\n)\n";
        let verdict = validate_sat_model(
            &h,
            &unit_box(),
            &Violation::DominatesBelowZero,
            stdout,
            PathBuf::from("q.smt2"),
        );
        match verdict {
            SmtVerdict::Falsified {
                witness,
                witness_exact,
                output_index,
                difference_exact,
                ..
            } => {
                assert_eq!(witness, vec![0.0, 0.0, 0.0]);
                assert_eq!(witness_exact, vec!["0", "0", "0"]);
                assert_eq!(output_index, 0);
                assert_eq!(difference_exact, "-1/4");
            }
            other => panic!("expected Falsified, got {other:?}"),
        }
    }

    #[test]
    fn violation_spec_matches_relations() {
        let dom = Violation::for_relation(Relation::Dominates, false).expect("dominates");
        assert!(dom.violated_by(&[rat(-1, 8)]).is_some());
        assert!(dom.violated_by(&[BigRational::zero()]).is_none());

        let strict = Violation::for_relation(Relation::Dominates, true).expect("strict");
        assert!(strict.violated_by(&[BigRational::zero()]).is_some());
        assert!(strict.violated_by(&[rat(1, 8)]).is_none());

        let abs = Violation::for_relation(Relation::AbsBound(0.5), false).expect("absbound");
        assert!(abs.violated_by(&[rat(3, 4)]).is_some());
        assert!(abs.violated_by(&[rat(-3, 4)]).is_some());
        assert!(abs.violated_by(&[rat(1, 2)]).is_none());
        assert_eq!(
            abs.disjunct("t"),
            "(or (> t (/ 1.0 2.0)) (< t (- (/ 1.0 2.0))))"
        );

        assert!(matches!(
            Violation::for_relation(Relation::AbsBound(0.0), false),
            Err(EscalateError::InvalidEpsilon { .. })
        ));
        assert!(matches!(
            Violation::for_relation(Relation::AbsBound(f64::NAN), false),
            Err(EscalateError::InvalidEpsilon { .. })
        ));
    }
}
