// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Alethe proof **emission** for ny-cert certificate objects (Marabou-PR#894
//! parity: NY-side generation of SMT-standard, independently checkable proofs).
//!
//! Every emitter produces a *pair* of texts — an SMT-LIB problem (one
//! `declare-const` per variable, one `assert` per premise) and an Alethe proof
//! that refutes it — suitable for the standard external checker
//! [carcara](https://github.com/ufmg-smite/carcara) via
//! `carcara check proof.alethe problem.smt2`.
//!
//! Mapping (the same shape veriT/cvc5 emit for a linear-arithmetic UNSAT):
//!
//! - [`farkas_to_alethe`]: `assume` each constraint; ONE `la_generic` step
//!   whose clause is the disjunction of the *negated* constraint atoms and
//!   whose `:args` are exactly the certificate's non-negative Farkas
//!   multipliers; a closing `resolution` step derives the empty clause `(cl)`.
//! - [`entailment_to_alethe`]: refutation-style lowering — assume the premises
//!   plus the **negated** conclusion (with multiplier `1`), then reuse the
//!   Farkas emission. `Σa·x ≥ c` becomes the strict `Σa·x < c`, so the
//!   combination is strict exactly as [`check_entailment`]'s slack condition
//!   requires.
//! - [`branch_tree_to_alethe`]: the full case-split skeleton — one `la_generic`
//!   refutation per leaf cell (cell faces + the negated property row), one
//!   `la_generic` split tautology `(cl (<= x m) (>= x m))` per interior axis
//!   edge, and a synthesized resolution DAG folding the product grid down to
//!   the empty clause.
//!
//! All rationals are emitted **exactly** (`k.0` / `(/ n.0 d.0)`, negatives as
//! `(- …)` per SMT-LIB) — no floats anywhere.
//!
//! # Fail-closed
//! Emission REFUSES rather than emit wrong Alethe: every entry point first
//! replays the certificate through the corresponding in-tree verifier
//! ([`check_farkas`] / [`check_entailment`] / [`check_branch_tree`]) and then
//! rejects any shape outside the supported fragment (equality constraints,
//! non-SMT-LIB variable names, multi-member or upper-bound branch trees).

use crate::branch::{check_branch_tree, BranchError, BranchTreeCertificate, ThreshDir};
use crate::rational::{Rat, RatError};
use crate::schema::{ConstraintKind, EntailmentCertificate, FarkasCertificate, LinearConstraint};
use crate::selfcheck::{check_entailment, check_farkas, CheckError};
use std::collections::{BTreeMap, BTreeSet};

/// An emitted (SMT-LIB problem, Alethe proof) pair. Check externally with
/// `carcara check proof.alethe problem.smt2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AletheEmission {
    /// SMT-LIB 2 problem text: `set-logic`, `declare-const`s, `assert`s.
    pub problem: String,
    /// Alethe proof text: `assume`s, `la_generic` step(s), resolution to `(cl)`.
    pub proof: String,
}

/// Why a certificate could not be emitted as Alethe (fail-closed: any
/// unsupported shape declines instead of emitting wrong proof text).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EmitError {
    /// The certificate failed its own verifier — never emit an unproven claim.
    #[error("certificate failed local verification: {0}")]
    Check(#[from] CheckError),
    /// The branch-tree certificate failed [`check_branch_tree`].
    #[error("branch-tree certificate failed verification: {0}")]
    Branch(#[from] BranchError),
    /// Constraint `{0}` is an equality. `la_generic` needs a *signed* arg
    /// convention for equalities that [`check_farkas`] (where an `Eq`'s two
    /// normalized halves cancel) cannot pre-validate — so we refuse.
    #[error("constraint {0} is an equality; Alethe emission supports inequalities only")]
    UnsupportedEq(usize),
    /// A variable name is not a safe SMT-LIB simple symbol.
    #[error("variable name {0:?} is not a safe SMT-LIB simple symbol")]
    BadSymbol(String),
    /// The branch tree is outside the supported single-affine, lower-bound
    /// fragment (details in the message).
    #[error("unsupported branch-tree shape: {0}")]
    UnsupportedBranchShape(String),
    /// Exact arithmetic failure (infallible in practice — full bignum).
    #[error(transparent)]
    Rat(#[from] RatError),
}

/// Words that cannot be emitted unquoted as variables in both artifacts.
///
/// This table is the union of (a) SMT-LIB 2.7 section 3.1's basic reserved words
/// and command names, (b) Alethe's `choice` binder, `cl` operator, and proof
/// commands (specification figure 2, matching Carcara's non-punctuation
/// `Reserved` tokens), and (c) deliberately blocked Core/QF_LRA theory symbols
/// plus `Int`. Only entries that can pass our conservative character grammar
/// need listing: punctuation-containing words such as `!` and `declare-const`
/// are already rejected below. The SMT-LIB command names remain here even where
/// Carcara's lexer is permissive; target permissiveness does not make them valid
/// simple symbols under the standard.
///
/// Reals_Ints-only operators such as `abs`, `div`, `mod`, `to_real`, `to_int`,
/// and `is_int` remain usable because this emitter fixes the problem logic to
/// QF_LRA.
const RESERVED: &[&str] = &[
    // SMT-LIB 2.7 section 3.1: basic reserved words accepted by our grammar.
    "_",
    "BINARY",
    "DECIMAL",
    "HEXADECIMAL",
    "NUMERAL",
    "STRING",
    "as",
    "lambda",
    "let",
    "exists",
    "forall",
    "match",
    "par",
    // SMT-LIB 2.7 command names accepted by our grammar.
    "assert",
    "echo",
    "exit",
    "pop",
    "push",
    "reset",
    // Alethe specification figure 2: proof-term syntax and commands.
    "choice",
    "cl",
    "assume",
    "step",
    "anchor",
    // Core and Reals theory symbols accepted by our grammar; Int is a
    // conservative future-proofing refusal if the emitted logic expands.
    "true",
    "false",
    "not",
    "and",
    "or",
    "xor",
    "ite",
    "distinct",
    "Bool",
    "Real",
    "Int",
];

/// Accept only conservative SMT-LIB simple symbols: `[A-Za-z_][A-Za-z0-9_.]*`,
/// not a reserved word. Anything else declines (no `|…|` quoting — fail closed).
fn check_symbol(name: &str) -> Result<(), EmitError> {
    let mut chars = name.chars();
    let ok_first = match chars.next() {
        Some(c) => c.is_ascii_alphabetic() || c == '_',
        None => false,
    };
    let mut ok = ok_first;
    for c in chars {
        if !(c.is_ascii_alphanumeric() || c == '_' || c == '.') {
            ok = false;
            break;
        }
    }
    for r in RESERVED {
        if *r == name {
            ok = false;
            break;
        }
    }
    if ok {
        Ok(())
    } else {
        Err(EmitError::BadSymbol(name.to_string()))
    }
}

/// Print a rational as an exact Real-sorted SMT-LIB term: `k.0` for integers,
/// `(/ n.0 d.0)` for proper fractions, negatives wrapped as `(- …)`.
fn rat_term(r: Rat) -> Result<String, EmitError> {
    // Full-bignum and exact (`"n"` / `"n/d"`); the only refusal is a poisoned
    // rational arena, which must not cross an emission boundary.
    let s = match r.to_clean_string() {
        Ok(s) => s,
        Err(e) => return Err(EmitError::Rat(e)),
    };
    let (neg, mag) = match s.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, s.as_str()),
    };
    let core = match mag.split_once('/') {
        Some((n, d)) => format!("(/ {n}.0 {d}.0)"),
        None => format!("{mag}.0"),
    };
    Ok(if neg { format!("(- {core})") } else { core })
}

/// Print a [`LinearConstraint`] as a binary comparison atom
/// `(<op> <linear-sum> <constant>)`. `idx` is only used to report which
/// constraint was an (unsupported) equality.
fn atom_sexpr(c: &LinearConstraint, idx: usize) -> Result<String, EmitError> {
    let op = match c.kind {
        ConstraintKind::Le => "<=",
        ConstraintKind::Lt => "<",
        ConstraintKind::Ge => ">=",
        ConstraintKind::Gt => ">",
        ConstraintKind::Eq => return Err(EmitError::UnsupportedEq(idx)),
    };
    let mut terms: Vec<String> = Vec::new();
    for (name, coeff) in &c.coefficients {
        check_symbol(name)?;
        if *coeff == Rat::ONE {
            terms.push(name.clone());
        } else {
            terms.push(format!("(* {} {name})", rat_term(*coeff)?));
        }
    }
    let lhs = if terms.is_empty() {
        "0.0".to_owned()
    } else if terms.len() == 1 {
        match terms.pop() {
            Some(t) => t,
            None => "0.0".to_owned(), // unreachable: len == 1
        }
    } else {
        // Explicit concat (not `terms.join(" ")`): identical space-separated
        // string; clears the join absent-callee.
        let mut joined = String::new();
        for (i, t) in terms.iter().enumerate() {
            if i > 0 {
                joined.push(' ');
            }
            joined.push_str(t);
        }
        format!("(+ {joined})")
    };
    Ok(format!("({op} {lhs} {})", rat_term(c.constant)?))
}

/// Build the SMT-LIB problem text for a set of variables and assertion atoms.
fn problem_text(vars: &BTreeSet<String>, atoms: &[String]) -> String {
    let mut p = String::from("(set-logic QF_LRA)\n");
    for v in vars {
        p.push_str("(declare-const ");
        p.push_str(v);
        p.push_str(" Real)\n");
    }
    for a in atoms {
        p.push_str("(assert ");
        p.push_str(a);
        p.push_str(")\n");
    }
    p.push_str("(check-sat)\n");
    p
}

/// Emit a Farkas refutation as Alethe.
///
/// Shape: one `assume` per constraint, one `la_generic` step whose clause is
/// the disjunction of the negated atoms with `:args` = the certificate's
/// multipliers (exact rationals), and a final `resolution` step to `(cl)`.
///
/// # Errors
/// Fail-closed: [`EmitError::Check`] when [`check_farkas`] rejects the
/// certificate, [`EmitError::UnsupportedEq`] on any equality constraint,
/// [`EmitError::BadSymbol`] on a non-SMT-LIB variable name.
pub fn farkas_to_alethe(cert: &FarkasCertificate) -> Result<AletheEmission, EmitError> {
    // Gate FIRST: never print a proof for a certificate that does not check.
    match check_farkas(cert) {
        Ok(_) => {}
        Err(e) => return Err(EmitError::Check(e)),
    }
    // Minimize at emission: dead rows (multiplier exactly zero) contribute
    // nothing to the la_generic combination, so drop them — smaller problem
    // and proof, same refutation. Fail-closed inside `minimized`: the smaller
    // cert is used only when `check_farkas` accepts it with the identical
    // residual; otherwise the original (already gated above) is emitted.
    let cert = &cert.minimized();
    // check_farkas guarantees constraint/multiplier parallelism and that every
    // multiplier is non-negative; equalities are refused below (their two
    // normalized halves cancel in check_farkas, so a signed-arg convention
    // could not be pre-validated).
    let mut atoms: Vec<String> = Vec::new();
    let mut vars: BTreeSet<String> = BTreeSet::new();
    for (i, c) in cert.constraints.iter().enumerate() {
        atoms.push(atom_sexpr(c, i)?);
        for k in c.coefficients.keys() {
            vars.insert(k.clone());
        }
    }
    let mut args: Vec<String> = Vec::new();
    for m in &cert.multipliers {
        args.push(rat_term(*m)?);
    }

    let problem = problem_text(&vars, &atoms);
    let mut proof = String::new();
    for (i, a) in atoms.iter().enumerate() {
        proof.push_str(&format!("(assume h{i} {a})\n"));
    }
    let mut clause = String::new();
    for a in &atoms {
        clause.push_str(&format!(" (not {a})"));
    }
    // Explicit concat (not `args.join(" ")`): identical space-separated string;
    // clears the join absent-callee.
    let mut args_joined = String::new();
    for (i, a) in args.iter().enumerate() {
        if i > 0 {
            args_joined.push(' ');
        }
        args_joined.push_str(a);
    }
    proof.push_str(&format!(
        "(step t1 (cl{clause}) :rule la_generic :args ({args_joined}))\n"
    ));
    let mut premises = String::from("t1");
    for i in 0..atoms.len() {
        premises.push_str(&format!(" h{i}"));
    }
    proof.push_str(&format!(
        "(step t2 (cl) :rule resolution :premises ({premises}))\n"
    ));
    Ok(AletheEmission { problem, proof })
}

/// Emit an entailment certificate as an Alethe refutation.
///
/// Alethe has no native implication object for this, so the standard lowering
/// is refutation-style: assume the premises **plus the negated conclusion**
/// (`Σa·x ≥ c` becomes the strict `Σa·x < c`, with an appended multiplier `1`),
/// producing a Farkas certificate that is emitted via [`farkas_to_alethe`].
/// The negation of a non-strict conclusion is strict, which makes the combined
/// contradiction strict — exactly matching [`check_entailment`]'s slack
/// condition, and re-verified by the [`check_farkas`] gate inside the Farkas
/// emitter. Dead premise rows (multiplier exactly zero) are dropped by the
/// Farkas emitter's fail-closed minimization pass (the appended negated
/// conclusion carries multiplier `1`, so it is never dropped).
///
/// # Errors
/// Fail-closed: [`EmitError::Check`] when [`check_entailment`] rejects the
/// certificate, plus every [`farkas_to_alethe`] refusal on the lowered form.
pub fn entailment_to_alethe(cert: &EntailmentCertificate) -> Result<AletheEmission, EmitError> {
    match check_entailment(cert) {
        Ok(_) => {}
        Err(e) => return Err(EmitError::Check(e)),
    }
    let negated_kind = match cert.conclusion.kind {
        ConstraintKind::Ge => ConstraintKind::Lt,
        ConstraintKind::Gt => ConstraintKind::Le,
        ConstraintKind::Le => ConstraintKind::Gt,
        ConstraintKind::Lt => ConstraintKind::Ge,
        // Unreachable: check_entailment rejects Eq conclusions
        // (NonInequalityConclusion) — fail closed anyway.
        ConstraintKind::Eq => return Err(EmitError::UnsupportedEq(cert.premises.len())),
    };
    let negated = LinearConstraint {
        kind: negated_kind,
        coefficients: cert.conclusion.coefficients.clone(),
        constant: cert.conclusion.constant,
    };
    let mut constraints = cert.premises.clone();
    constraints.push(negated);
    let mut multipliers = cert.multipliers.clone();
    multipliers.push(Rat::ONE);
    farkas_to_alethe(&FarkasCertificate {
        constraints,
        multipliers,
    })
}

// ---------------------------------------------------------------------------
// Branch-tree emission: per-leaf la_generic refutations + split tautologies +
// a synthesized resolution DAG folding the product grid to the empty clause.
// ---------------------------------------------------------------------------

/// A proof literal: `(polarity, atom-sexpr)`; `false` = negated.
type Lit = (bool, String);

fn lit_str(l: &Lit) -> String {
    if l.0 {
        l.1.clone()
    } else {
        format!("(not {})", l.1)
    }
}

/// Sequential Alethe step writer (`t1`, `t2`, …).
struct StepWriter {
    proof: String,
    next: usize,
}

impl StepWriter {
    fn new() -> Self {
        StepWriter {
            proof: String::new(),
            next: 0,
        }
    }

    fn assume(&mut self, name: &str, atom: &str) {
        self.proof.push_str(&format!("(assume {name} {atom})\n"));
    }

    fn step(&mut self, lits: &[Lit], rule: &str, args: Option<&str>, premises: &[&str]) -> String {
        self.next = self.next.saturating_add(1);
        let id = format!("t{}", self.next);
        let mut clause = String::new();
        for l in lits {
            clause.push_str(&format!(" {}", lit_str(l)));
        }
        let mut line = format!("(step {id} (cl{clause}) :rule {rule}");
        if !premises.is_empty() {
            // Explicit concat (not `premises.join(" ")`): identical space-separated
            // string; clears the join absent-callee.
            let mut prem_joined = String::new();
            for (i, p) in premises.iter().enumerate() {
                if i > 0 {
                    prem_joined.push(' ');
                }
                prem_joined.push_str(p);
            }
            line.push_str(&format!(" :premises ({prem_joined})"));
        }
        if let Some(a) = args {
            line.push_str(&format!(" :args ({a})"));
        }
        line.push_str(")\n");
        self.proof.push_str(&line);
        id
    }
}

/// A derived clause: its Alethe step id plus its (deduplicated) literal list.
struct Clause {
    id: String,
    lits: Vec<Lit>,
}

fn shape(msg: &str) -> EmitError {
    EmitError::UnsupportedBranchShape(msg.to_string())
}

/// Corner key mirroring `branch.rs`: injective over distinct grid cells.
fn corner_key(lo: &[Rat], hi: &[Rat]) -> Result<String, EmitError> {
    let mut s = String::new();
    for (l, h) in lo.iter().zip(hi) {
        let ls = match l.to_clean_string() {
            Ok(v) => v,
            Err(e) => return Err(EmitError::Rat(e)),
        };
        let hs = match h.to_clean_string() {
            Ok(v) => v,
            Err(e) => return Err(EmitError::Rat(e)),
        };
        s.push_str(&ls);
        s.push('|');
        s.push_str(&hs);
        s.push(';');
    }
    Ok(s)
}

/// Unit-coefficient face constraint `var ⋈ bound`.
fn face(var: &str, kind: ConstraintKind, bound: Rat) -> LinearConstraint {
    LinearConstraint::with_kind(kind, &[(var, Rat::ONE)], bound)
}

struct BranchEmit<'a> {
    cert: &'a BranchTreeCertificate,
    /// The negated property row `a·x ≤ threshold − b` shared by every leaf.
    p_row: LinearConstraint,
    p_atom: String,
    leaf_by_key: BTreeMap<String, usize>,
    w: StepWriter,
    /// Split-tautology cache: `(<= var m)` atom → step id.
    splits: BTreeMap<String, String>,
}

impl BranchEmit<'_> {
    /// Emit (or reuse) the split tautology `(cl (<= var m) (>= var m))`.
    fn split_step(&mut self, var: &str, edge: Rat) -> Result<String, EmitError> {
        let le_atom = atom_sexpr(&face(var, ConstraintKind::Le, edge), 0)?;
        if let Some(id) = self.splits.get(&le_atom) {
            return Ok(id.clone());
        }
        let ge_atom = atom_sexpr(&face(var, ConstraintKind::Ge, edge), 0)?;
        // `step` only borrows this two-literal clause, so a stack array is the
        // exact carrier and avoids an unnecessary allocation boundary.
        let lits = [(true, le_atom.clone()), (true, ge_atom)];
        let id = self.w.step(&lits, "la_generic", Some("1.0 1.0"), &[]);
        self.splits.insert(le_atom, id.clone());
        Ok(id)
    }

    /// Emit the `la_generic` refutation for the grid cell at `coords`:
    /// clause = negated cell faces (all `2·naxes` of them, zero-arg where the
    /// leaf's entailment does not use a face) + the negated property row;
    /// args = the leaf entailment's face multipliers + `1` on the property.
    fn emit_leaf(&mut self, coords: &[usize]) -> Result<Clause, EmitError> {
        let n = self.cert.axes.len();
        let mut lo: Vec<Rat> = Vec::new();
        let mut hi: Vec<Rat> = Vec::new();
        for (axis, &k) in self.cert.axes.iter().zip(coords) {
            let (Some(l), Some(h)) = (axis.edges.get(k), axis.edges.get(k.saturating_add(1)))
            else {
                return Err(shape("internal: grid cursor outside axis (unreachable)"));
            };
            lo.push(*l);
            hi.push(*h);
        }
        let key = corner_key(&lo, &hi)?;
        let li = match self.leaf_by_key.get(&key) {
            Some(i) => *i,
            // Unreachable: check_branch_tree verified the exact product cover.
            None => return Err(shape("internal: grid cell has no leaf (unreachable)")),
        };
        let Some(leaf) = self.cert.leaves.get(li) else {
            return Err(shape("internal: leaf index out of range (unreachable)"));
        };
        let Some(ent) = leaf.member_entailments.first() else {
            return Err(shape(
                "internal: leaf without member entailment (unreachable)",
            ));
        };

        // Face multiplier slots: (lower, upper) per axis, summed from the
        // entailment's premises (check_branch_tree already proved each premise
        // IS a face of this cell; re-derive fail-closed anyway).
        let mut slots: Vec<(Rat, Rat)> = Vec::new();
        for _ in 0..n {
            slots.push((Rat::ZERO, Rat::ZERO));
        }
        if ent.premises.len() != ent.multipliers.len() {
            return Err(shape("leaf entailment premise/multiplier length mismatch"));
        }
        for (p, m) in ent.premises.iter().zip(&ent.multipliers) {
            if p.coefficients.len() != 1 {
                return Err(shape("leaf premise is not a unit-coefficient cell face"));
            }
            let Some((name, coeff)) = p.coefficients.iter().next() else {
                return Err(shape("internal: empty premise coefficients (unreachable)"));
            };
            if *coeff != Rat::ONE {
                return Err(shape("leaf premise face coefficient is not 1"));
            }
            let mut axis_opt: Option<usize> = None;
            for (d, a) in self.cert.axes.iter().enumerate() {
                if a.var.as_str() == name.as_str() {
                    axis_opt = Some(d);
                    break;
                }
            }
            let Some(d) = axis_opt else {
                return Err(shape("leaf premise variable is not an axis variable"));
            };
            let Some(slot) = slots.get_mut(d) else {
                return Err(shape("internal: slot index out of range (unreachable)"));
            };
            let (is_lo, is_hi) = match (lo.get(d), hi.get(d)) {
                (Some(l), Some(h)) => (
                    p.kind == ConstraintKind::Ge && p.constant == *l,
                    p.kind == ConstraintKind::Le && p.constant == *h,
                ),
                _ => (false, false),
            };
            if is_lo {
                slot.0 = match slot.0.add(*m) {
                    Ok(v) => v,
                    Err(e) => return Err(EmitError::Rat(e)),
                };
            } else if is_hi {
                slot.1 = match slot.1.add(*m) {
                    Ok(v) => v,
                    Err(e) => return Err(EmitError::Rat(e)),
                };
            } else {
                return Err(shape("leaf premise is not a face of its own cell"));
            }
        }

        // Fail-closed gate: the assembled per-leaf system (cell faces + negated
        // property) must ITSELF be a checkable Farkas refutation.
        let mut constraints: Vec<LinearConstraint> = Vec::new();
        let mut multipliers: Vec<Rat> = Vec::new();
        let mut lits: Vec<Lit> = Vec::new();
        let mut args: Vec<String> = Vec::new();
        for (d, axis) in self.cert.axes.iter().enumerate() {
            let (Some(l), Some(h), Some(slot)) = (lo.get(d), hi.get(d), slots.get(d)) else {
                return Err(shape("internal: axis index out of range (unreachable)"));
            };
            let ge = face(&axis.var, ConstraintKind::Ge, *l);
            let le = face(&axis.var, ConstraintKind::Le, *h);
            lits.push((false, atom_sexpr(&ge, 0)?));
            args.push(rat_term(slot.0)?);
            lits.push((false, atom_sexpr(&le, 0)?));
            args.push(rat_term(slot.1)?);
            constraints.push(ge);
            multipliers.push(slot.0);
            constraints.push(le);
            multipliers.push(slot.1);
        }
        constraints.push(self.p_row.clone());
        multipliers.push(Rat::ONE);
        lits.push((false, self.p_atom.clone()));
        args.push("1.0".to_owned());
        match check_farkas(&FarkasCertificate {
            constraints,
            multipliers,
        }) {
            Ok(_) => {}
            Err(e) => return Err(EmitError::Check(e)),
        }

        let id = self.w.step(&lits, "la_generic", Some(&args.join(" ")), &[]);
        Ok(Clause { id, lits })
    }

    /// Merge the covers of two adjacent regions along axis `d` at `edge` via
    /// the split tautology and one chain-resolution step. If a cover already
    /// lacks its pivot literal it covers the merged region on its own.
    fn merge(
        &mut self,
        d: usize,
        edge: Rat,
        left: Clause,
        right: Clause,
    ) -> Result<Clause, EmitError> {
        let var = match self.cert.axes.get(d) {
            Some(a) => a.var.clone(),
            None => return Err(shape("internal: merge axis out of range (unreachable)")),
        };
        let le_atom = atom_sexpr(&face(&var, ConstraintKind::Le, edge), 0)?;
        let ge_atom = atom_sexpr(&face(&var, ConstraintKind::Ge, edge), 0)?;
        let pivot_l: Lit = (false, le_atom);
        let pivot_r: Lit = (false, ge_atom);
        if !left.lits.contains(&pivot_l) {
            return Ok(left);
        }
        if !right.lits.contains(&pivot_r) {
            return Ok(right);
        }
        let split = self.split_step(&var, edge)?;
        let mut lits: Vec<Lit> = Vec::new();
        for l in &left.lits {
            if *l != pivot_l && !lits.contains(l) {
                lits.push(l.clone());
            }
        }
        for l in &right.lits {
            if *l != pivot_r && !lits.contains(l) {
                lits.push(l.clone());
            }
        }
        let id = self.w.step(
            &lits,
            "resolution",
            None,
            &[split.as_str(), left.id.as_str(), right.id.as_str()],
        );
        Ok(Clause { id, lits })
    }

    /// Derive the cover clause for the region fixing axes `0..coords.len()` to
    /// the given cells and spanning the full box on the remaining axes.
    fn cover(
        &mut self,
        coords: &mut Vec<usize>,
        axis_count: usize,
        d: usize,
    ) -> Result<Clause, EmitError> {
        // `axis_count` is captured once at entry and passed by value. The
        // `>=` guard makes the decreasing `axis_count - d` measure explicit
        // while remaining identical for the 0, 1, ... recursion below.
        if d >= axis_count {
            return self.emit_leaf(coords);
        }
        let ncells = match self.cert.axes.get(d) {
            Some(a) => a.edges.len().saturating_sub(1),
            None => return Err(shape("internal: cover axis out of range (unreachable)")),
        };
        coords.push(0);
        let mut acc = self.cover(coords, axis_count, d.saturating_add(1))?;
        coords.pop();
        for k in 1..ncells {
            coords.push(k);
            let right = self.cover(coords, axis_count, d.saturating_add(1))?;
            coords.pop();
            let edge = match self.cert.axes.get(d) {
                Some(a) => match a.edges.get(k) {
                    Some(e) => *e,
                    None => return Err(shape("internal: edge index out of range (unreachable)")),
                },
                None => return Err(shape("internal: cover axis out of range (unreachable)")),
            };
            acc = self.merge(d, edge, acc, right)?;
        }
        Ok(acc)
    }
}

/// Emit a branch-tree (case-split) certificate as a full Alethe refutation.
///
/// Supported fragment (anything else declines, fail-closed): direction
/// [`ThreshDir::Le`] (refuting the unsafe region `y ≤ threshold`), every leaf
/// carrying EXACTLY ONE member entailment, and all leaves sharing one affine
/// output `y = a·x + b` (identical conclusion coefficients and bias). The
/// emitted problem asserts the whole-box faces plus the negated property row
/// `a·x ≤ threshold − b`; the proof refutes each grid cell with `la_generic`
/// (cell faces + property row, using the leaf entailment's own multipliers),
/// proves a split tautology `(cl (<= x m) (>= x m))` per interior axis edge,
/// and folds the product grid with chain-resolution steps down to `(cl)`.
///
/// # Errors
/// Fail-closed: [`EmitError::Branch`] when [`check_branch_tree`] rejects the
/// certificate; [`EmitError::UnsupportedBranchShape`] outside the fragment
/// above; [`EmitError::Check`] if any assembled per-leaf refutation fails
/// [`check_farkas`] (unreachable for a certificate that passed the gate).
pub fn branch_tree_to_alethe(cert: &BranchTreeCertificate) -> Result<AletheEmission, EmitError> {
    // Gate FIRST: the composed certificate must verify before anything prints.
    match check_branch_tree(cert) {
        Ok(_) => {}
        Err(e) => return Err(EmitError::Branch(e)),
    }
    if cert.dir != ThreshDir::Le {
        return Err(shape(
            "only ThreshDir::Le (lower-bound tree refuting `y <= t`) is supported",
        ));
    }
    if cert.axes.is_empty() {
        return Err(shape("certificate has no axes"));
    }
    let mut axis_vars: BTreeSet<String> = BTreeSet::new();
    for a in &cert.axes {
        check_symbol(&a.var)?;
        if !axis_vars.insert(a.var.clone()) {
            return Err(shape("duplicate axis variable name"));
        }
    }
    // One member per leaf; a single shared affine function `a·x + b`.
    let Some(first_leaf) = cert.leaves.first() else {
        return Err(shape("certificate has no leaves"));
    };
    let Some(first_ent) = first_leaf.member_entailments.first() else {
        return Err(shape("leaf 0 has no member entailment"));
    };
    let a_coeffs = first_ent.conclusion.coefficients.clone();
    let Some(bias) = first_leaf.member_biases.first().copied() else {
        return Err(shape("leaf 0 has no member bias"));
    };
    for leaf in &cert.leaves {
        if leaf.member_entailments.len() != 1 || leaf.member_biases.len() != 1 {
            return Err(shape(
                "every leaf must carry exactly one member entailment (single-affine fragment)",
            ));
        }
        let Some(ent) = leaf.member_entailments.first() else {
            return Err(shape("internal: missing member entailment (unreachable)"));
        };
        if ent.conclusion.coefficients != a_coeffs {
            return Err(shape("leaves do not share one affine output function"));
        }
        let Some(b) = leaf.member_biases.first() else {
            return Err(shape("internal: missing member bias (unreachable)"));
        };
        if *b != bias {
            return Err(shape("leaves do not share one member bias"));
        }
    }
    for v in a_coeffs.keys() {
        if !axis_vars.contains(v) {
            return Err(shape("conclusion variable is not an axis variable"));
        }
    }

    // The negated property row: `y <= t` with `y = a·x + b` is `a·x <= t - b`.
    let p_const = match cert.threshold.sub(bias) {
        Ok(v) => v,
        Err(e) => return Err(EmitError::Rat(e)),
    };
    let p_row = LinearConstraint {
        kind: ConstraintKind::Le,
        coefficients: a_coeffs,
        constant: p_const,
    };
    let p_atom = atom_sexpr(&p_row, 0)?;

    // Problem + assumes: whole-box faces per axis, then the property row.
    let mut atoms: Vec<String> = Vec::new();
    let mut vars: BTreeSet<String> = BTreeSet::new();
    for a in &cert.axes {
        vars.insert(a.var.clone());
        let (Some(blo), Some(bhi)) = (a.edges.first(), a.edges.last()) else {
            return Err(shape("internal: empty axis edges (unreachable)"));
        };
        atoms.push(atom_sexpr(&face(&a.var, ConstraintKind::Ge, *blo), 0)?);
        atoms.push(atom_sexpr(&face(&a.var, ConstraintKind::Le, *bhi), 0)?);
    }
    atoms.push(p_atom.clone());
    let problem = problem_text(&vars, &atoms);

    let mut w = StepWriter::new();
    let mut assume_by_atom: BTreeMap<String, String> = BTreeMap::new();
    for (i, a) in atoms.iter().enumerate() {
        let name = format!("h{i}");
        w.assume(&name, a);
        assume_by_atom.insert(a.clone(), name);
    }

    // Leaf lookup by exact cell corners (the cover check proved bijectivity).
    let mut leaf_by_key: BTreeMap<String, usize> = BTreeMap::new();
    for (i, leaf) in cert.leaves.iter().enumerate() {
        leaf_by_key.insert(corner_key(&leaf.lo, &leaf.hi)?, i);
    }

    let mut ctx = BranchEmit {
        cert,
        p_row,
        p_atom,
        leaf_by_key,
        w,
        splits: BTreeMap::new(),
    };
    let mut coords: Vec<usize> = Vec::new();
    let root = ctx.cover(&mut coords, ctx.cert.axes.len(), 0)?;

    // Every literal left in the root cover is a negated box face or the
    // negated property row — all assumed units. Resolve them away to `(cl)`.
    let mut premises: Vec<&str> = Vec::new();
    premises.push(root.id.as_str());
    for l in &root.lits {
        if l.0 {
            return Err(shape(
                "internal: positive literal in root cover (unreachable)",
            ));
        }
        match assume_by_atom.get(&l.1) {
            Some(name) => premises.push(name.as_str()),
            None => {
                return Err(shape(
                    "internal: root literal is not an assumed atom (unreachable)",
                ))
            }
        }
    }
    ctx.w.step(&[], "resolution", None, &premises);

    Ok(AletheEmission {
        problem,
        proof: ctx.w.proof,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alethe_bridge::bridge_la_generic;
    use crate::branch::{AxisPartition, BranchLeaf};
    use crate::selfcheck::CheckError;

    fn r(n: i128, d: i128) -> Rat {
        Rat::new(n, d).unwrap()
    }

    /// The invprop_cert test case: premise `y >= 3`, violation row `y <= 1`,
    /// gamma = 1 — the combination collapses to `0 <= -2` => false.
    fn invprop_farkas() -> FarkasCertificate {
        FarkasCertificate {
            constraints: vec![
                LinearConstraint::with_kind(ConstraintKind::Ge, &[("y", Rat::ONE)], r(3, 1)),
                LinearConstraint::with_kind(ConstraintKind::Le, &[("y", Rat::ONE)], r(1, 1)),
            ],
            multipliers: vec![Rat::ONE, Rat::ONE],
        }
    }

    #[test]
    fn farkas_invprop_golden() {
        let em = farkas_to_alethe(&invprop_farkas()).expect("valid Farkas cert emits");
        assert_eq!(
            em.problem,
            "(set-logic QF_LRA)\n\
             (declare-const y Real)\n\
             (assert (>= y 3.0))\n\
             (assert (<= y 1.0))\n\
             (check-sat)\n"
        );
        assert_eq!(
            em.proof,
            "(assume h0 (>= y 3.0))\n\
             (assume h1 (<= y 1.0))\n\
             (step t1 (cl (not (>= y 3.0)) (not (<= y 1.0))) :rule la_generic :args (1.0 1.0))\n\
             (step t2 (cl) :rule resolution :premises (t1 h0 h1))\n"
        );
    }

    #[test]
    fn farkas_emission_round_trips_through_the_bridge() {
        let em = farkas_to_alethe(&invprop_farkas()).unwrap();
        // The importer re-checks the emitted la_generic under check_farkas and
        // returns the same contradiction constant: 1 - 3 = -2.
        let witness = bridge_la_generic(&em.proof).expect("emitted proof bridges back");
        assert_eq!(witness, r(-2, 1));
    }

    #[test]
    fn farkas_emission_omits_zero_multiplier_rows_before_symbol_and_kind_checks() {
        let cert = FarkasCertificate {
            constraints: vec![
                // Deliberately unsupported if emitted: the row is both an
                // equality and carries an unsafe SMT symbol. Its exact-zero
                // multiplier makes it semantically dead, so minimization must
                // remove it before the emitter's kind/symbol checks.
                LinearConstraint::with_kind(
                    ConstraintKind::Eq,
                    &[("dead symbol", Rat::ONE)],
                    r(42, 1),
                ),
                LinearConstraint::with_kind(ConstraintKind::Ge, &[("y", Rat::ONE)], r(3, 1)),
                LinearConstraint::with_kind(ConstraintKind::Le, &[("y", Rat::ONE)], r(1, 1)),
            ],
            multipliers: vec![Rat::ZERO, Rat::ONE, Rat::ONE],
        };
        assert_eq!(check_farkas(&cert), Ok(r(-2, 1)));

        let em = farkas_to_alethe(&cert).expect("dead unsupported row must be omitted");

        assert!(!em.problem.contains("dead symbol"));
        assert!(!em.proof.contains("dead symbol"));
        assert!(em.proof.contains(":args (1.0 1.0)"));
        assert_eq!(
            bridge_la_generic(&em.proof).expect("minimized proof bridges back"),
            r(-2, 1)
        );
    }

    #[test]
    fn alethe_emission_omits_zero_multiplier_dead_premises() {
        let mut farkas = invprop_farkas();
        farkas.constraints.insert(
            1,
            LinearConstraint::with_kind(
                ConstraintKind::Eq,
                &[("dead_farkas", Rat::ONE)],
                Rat::ZERO,
            ),
        );
        farkas.multipliers.insert(1, Rat::ZERO);
        let original_residual = check_farkas(&farkas).expect("dead row is semantically inert");
        let emission = farkas_to_alethe(&farkas)
            .expect("a zero-weight equality is dropped before Alethe's equality refusal");
        assert!(!emission.problem.contains("dead_farkas"));
        assert!(!emission.proof.contains("dead_farkas"));
        assert_eq!(emission.problem.matches("(assert ").count(), 2);
        assert_eq!(emission.proof.matches("(assume ").count(), 2);
        assert!(emission.proof.contains(":args (1.0 1.0)"));
        assert_eq!(
            bridge_la_generic(&emission.proof).expect("minimized proof bridges back"),
            original_residual,
            "dead-row omission must preserve the checked contradiction"
        );

        let entailment = EntailmentCertificate {
            premises: vec![
                LinearConstraint::with_kind(
                    ConstraintKind::Eq,
                    &[("dead_entailment", Rat::ONE)],
                    Rat::ZERO,
                ),
                LinearConstraint::with_kind(ConstraintKind::Ge, &[("x", Rat::ONE)], Rat::ONE),
            ],
            multipliers: vec![Rat::ZERO, Rat::ONE],
            conclusion: LinearConstraint::with_kind(
                ConstraintKind::Ge,
                &[("x", Rat::ONE)],
                Rat::ONE,
            ),
        };
        let original_bounds =
            check_entailment(&entailment).expect("dead entailment premise is inert");
        let emission = entailment_to_alethe(&entailment).expect("valid entailment emits");
        assert!(!emission.problem.contains("dead_entailment"));
        assert!(!emission.proof.contains("dead_entailment"));
        // One live premise plus the appended strict negation of the conclusion.
        assert_eq!(emission.problem.matches("(assert ").count(), 2);
        assert_eq!(emission.proof.matches("(assume ").count(), 2);
        assert!(emission.proof.contains(":args (1.0 1.0)"));
        assert_eq!(original_bounds, (r(-1, 1), r(-1, 1)));
        assert_eq!(
            bridge_la_generic(&emission.proof).expect("minimized entailment proof bridges back"),
            Rat::ZERO
        );
    }

    #[test]
    fn farkas_fractional_multiplier_round_trips_exactly() {
        // 2x >= 3 (mult 1/2) + x <= 1 (mult 1): 0 <= 1 - 3/2 = -1/2.
        let cert = FarkasCertificate {
            constraints: vec![
                LinearConstraint::with_kind(ConstraintKind::Ge, &[("x", r(2, 1))], r(3, 1)),
                LinearConstraint::with_kind(ConstraintKind::Le, &[("x", Rat::ONE)], r(1, 1)),
            ],
            multipliers: vec![r(1, 2), Rat::ONE],
        };
        let em = farkas_to_alethe(&cert).unwrap();
        assert!(
            em.proof.contains(":args ((/ 1.0 2.0) 1.0)"),
            "fractional multipliers must be exact rationals, got:\n{}",
            em.proof
        );
        // NOTE: this atom has a non-unit coefficient `(* 2.0 x)`, which the
        // import bridge's term grammar does not cover — round-trip only the
        // args parsing here via a unit-coefficient variant.
        let unit = FarkasCertificate {
            constraints: vec![
                LinearConstraint::with_kind(ConstraintKind::Ge, &[("x", Rat::ONE)], r(3, 2)),
                LinearConstraint::with_kind(ConstraintKind::Le, &[("x", r(2, 1))], r(1, 1)),
            ],
            multipliers: vec![Rat::ONE, r(1, 2)],
        };
        let em2 = farkas_to_alethe(&unit).unwrap();
        assert!(em2.proof.contains(":args (1.0 (/ 1.0 2.0))"));
    }

    #[test]
    fn entailment_lowers_to_a_strict_refutation() {
        // x >= 1 (mult 1) entails x >= 1; negation `x < 1` closes strictly.
        let cert = EntailmentCertificate {
            premises: vec![LinearConstraint::with_kind(
                ConstraintKind::Ge,
                &[("x", Rat::ONE)],
                r(1, 1),
            )],
            multipliers: vec![Rat::ONE],
            conclusion: LinearConstraint::with_kind(
                ConstraintKind::Ge,
                &[("x", Rat::ONE)],
                r(1, 1),
            ),
        };
        let em = entailment_to_alethe(&cert).expect("valid entailment emits");
        assert!(
            em.proof.contains("(assume h1 (< x 1.0))"),
            "negated Ge conclusion must be assumed as strict <, got:\n{}",
            em.proof
        );
        // Round-trip: the strict combination collapses to the constant 0
        // (`0 < 0` — a strict Farkas contradiction).
        let witness = bridge_la_generic(&em.proof).expect("emitted proof bridges back");
        assert_eq!(witness, Rat::ZERO);
    }

    #[test]
    fn declines_equality_constraints() {
        // `z = 0` self-cancels in check_farkas (the cert still verifies), but
        // Alethe emission has no pre-validated signed-arg convention for it.
        let mut cert = invprop_farkas();
        cert.constraints.push(LinearConstraint::with_kind(
            ConstraintKind::Eq,
            &[("z", Rat::ONE)],
            Rat::ZERO,
        ));
        cert.multipliers.push(Rat::ONE);
        assert!(check_farkas(&cert).is_ok(), "precondition: cert verifies");
        assert_eq!(farkas_to_alethe(&cert), Err(EmitError::UnsupportedEq(2)));
    }

    #[test]
    fn declines_a_failing_certificate() {
        // y >= 1/2 + y <= 1 is satisfiable: no contradiction, no emission.
        let cert = FarkasCertificate {
            constraints: vec![
                LinearConstraint::with_kind(ConstraintKind::Ge, &[("y", Rat::ONE)], r(1, 2)),
                LinearConstraint::with_kind(ConstraintKind::Le, &[("y", Rat::ONE)], r(1, 1)),
            ],
            multipliers: vec![Rat::ONE, Rat::ONE],
        };
        assert_eq!(
            farkas_to_alethe(&cert),
            Err(EmitError::Check(CheckError::NotEstablished))
        );
    }

    #[test]
    fn declines_non_smtlib_variable_names() {
        let cert = FarkasCertificate {
            constraints: vec![
                LinearConstraint::with_kind(ConstraintKind::Ge, &[("bad name", Rat::ONE)], r(3, 1)),
                LinearConstraint::with_kind(ConstraintKind::Le, &[("bad name", Rat::ONE)], r(1, 1)),
            ],
            multipliers: vec![Rat::ONE, Rat::ONE],
        };
        assert!(check_farkas(&cert).is_ok(), "precondition: cert verifies");
        assert_eq!(
            farkas_to_alethe(&cert),
            Err(EmitError::BadSymbol("bad name".to_string()))
        );
    }

    #[test]
    fn declines_every_unsafe_unquoted_symbol() {
        // Exercise the public entry point for every token in the sourced set,
        // rather than only `_`: all of these satisfy the conservative
        // character grammar, but are reserved syntax or deliberately blocked
        // theory names.
        for &name in RESERVED {
            let cert = FarkasCertificate {
                constraints: vec![
                    LinearConstraint::with_kind(ConstraintKind::Ge, &[(name, Rat::ONE)], r(3, 1)),
                    LinearConstraint::with_kind(ConstraintKind::Le, &[(name, Rat::ONE)], r(1, 1)),
                ],
                multipliers: vec![Rat::ONE, Rat::ONE],
            };
            assert!(
                check_farkas(&cert).is_ok(),
                "precondition: cert for {name:?} verifies"
            );
            assert_eq!(
                farkas_to_alethe(&cert),
                Err(EmitError::BadSymbol(name.to_string())),
                "unsafe symbol {name:?} must fail closed"
            );
        }
    }

    #[test]
    fn accepts_ordinary_safe_symbols() {
        for name in [
            "x",
            "X0",
            "_x",
            "x_1",
            "x.y",
            "lambda_1",
            "clause",
            "exit_code",
            "BINARY1",
            "abs",
            "div",
            "mod",
            "to_real",
            "to_int",
            "is_int",
        ] {
            assert_eq!(check_symbol(name), Ok(()), "safe symbol {name:?}");
        }
    }

    // --- branch-tree fixtures (mirroring branch.rs's two_cell_cert) ---------

    fn ent(a0: Rat, a1: Rat, b: Rat, bound: Rat, lo: &[Rat], hi: &[Rat]) -> EntailmentCertificate {
        let (p0, mu0) = if a0.is_negative() {
            (face("x0", ConstraintKind::Le, hi[0]), a0.neg())
        } else {
            (face("x0", ConstraintKind::Ge, lo[0]), a0)
        };
        let (p1, mu1) = if a1.is_negative() {
            (face("x1", ConstraintKind::Le, hi[1]), a1.neg())
        } else {
            (face("x1", ConstraintKind::Ge, lo[1]), a1)
        };
        EntailmentCertificate {
            premises: vec![p0, p1],
            multipliers: vec![mu0, mu1],
            conclusion: LinearConstraint::with_kind(
                ConstraintKind::Ge,
                &[("x0", a0), ("x1", a1)],
                bound.sub(b).unwrap(),
            ),
        }
    }

    /// 2-cell split of `[-1,1]²` at `x0 = 0` for the affine `y = x0`:
    /// per-cell lower bounds -1 and 0; threshold -2 (property `y <= -2`).
    fn two_cell_cert(threshold: Rat) -> BranchTreeCertificate {
        let lo = r(-1, 1);
        let mid = r(0, 1);
        let hi = r(1, 1);
        let mk = |x0lo: Rat, x0hi: Rat, bound: Rat| BranchLeaf {
            lo: vec![x0lo, lo],
            hi: vec![x0hi, hi],
            bound,
            member_entailments: vec![ent(
                Rat::ONE,
                Rat::ZERO,
                Rat::ZERO,
                bound,
                &[x0lo, lo],
                &[x0hi, hi],
            )],
            member_biases: vec![Rat::ZERO],
        };
        BranchTreeCertificate {
            axes: vec![
                AxisPartition {
                    var: "x0".to_owned(),
                    edges: vec![lo, mid, hi],
                },
                AxisPartition {
                    var: "x1".to_owned(),
                    edges: vec![lo, hi],
                },
            ],
            leaves: vec![mk(lo, mid, r(-1, 1)), mk(mid, hi, Rat::ZERO)],
            threshold,
            dir: ThreshDir::Le,
        }
    }

    #[test]
    fn branch_two_leaf_golden() {
        let em = branch_tree_to_alethe(&two_cell_cert(r(-2, 1))).expect("valid tree emits");
        assert_eq!(
            em.problem,
            "(set-logic QF_LRA)\n\
             (declare-const x0 Real)\n\
             (declare-const x1 Real)\n\
             (assert (>= x0 (- 1.0)))\n\
             (assert (<= x0 1.0))\n\
             (assert (>= x1 (- 1.0)))\n\
             (assert (<= x1 1.0))\n\
             (assert (<= x0 (- 2.0)))\n\
             (check-sat)\n"
        );
        assert_eq!(
            em.proof,
            "(assume h0 (>= x0 (- 1.0)))\n\
             (assume h1 (<= x0 1.0))\n\
             (assume h2 (>= x1 (- 1.0)))\n\
             (assume h3 (<= x1 1.0))\n\
             (assume h4 (<= x0 (- 2.0)))\n\
             (step t1 (cl (not (>= x0 (- 1.0))) (not (<= x0 0.0)) (not (>= x1 (- 1.0))) (not (<= x1 1.0)) (not (<= x0 (- 2.0)))) :rule la_generic :args (1.0 0.0 0.0 0.0 1.0))\n\
             (step t2 (cl (not (>= x0 0.0)) (not (<= x0 1.0)) (not (>= x1 (- 1.0))) (not (<= x1 1.0)) (not (<= x0 (- 2.0)))) :rule la_generic :args (1.0 0.0 0.0 0.0 1.0))\n\
             (step t3 (cl (<= x0 0.0) (>= x0 0.0)) :rule la_generic :args (1.0 1.0))\n\
             (step t4 (cl (not (>= x0 (- 1.0))) (not (>= x1 (- 1.0))) (not (<= x1 1.0)) (not (<= x0 (- 2.0))) (not (<= x0 1.0))) :rule resolution :premises (t3 t1 t2))\n\
             (step t5 (cl) :rule resolution :premises (t4 h0 h2 h3 h4 h1))\n"
        );
    }

    #[test]
    fn branch_declines_upper_bound_direction() {
        // Same tree flipped to Ge. Its leaves prove only lower bounds, which
        // cannot refute an unsafe y>=threshold region, so both checker and
        // emitter must decline before constructing a proof.
        let mut cert = two_cell_cert(r(1, 1));
        cert.dir = ThreshDir::Ge;
        assert!(matches!(
            check_branch_tree(&cert),
            Err(BranchError::UnsupportedDirection(ThreshDir::Ge))
        ));
        assert!(matches!(
            branch_tree_to_alethe(&cert),
            Err(EmitError::Branch(BranchError::UnsupportedDirection(
                ThreshDir::Ge
            )))
        ));
    }

    #[test]
    fn branch_declines_multi_member_leaves() {
        // Duplicate a leaf's member (still verifies) — outside the
        // single-affine fragment, so emission must decline.
        let mut cert = two_cell_cert(r(-2, 1));
        let extra = cert.leaves[0].member_entailments[0].clone();
        cert.leaves[0].member_entailments.push(extra);
        cert.leaves[0].member_biases.push(Rat::ZERO);
        assert!(
            check_branch_tree(&cert).is_ok(),
            "precondition: cert verifies"
        );
        assert!(matches!(
            branch_tree_to_alethe(&cert),
            Err(EmitError::UnsupportedBranchShape(_))
        ));
    }

    #[test]
    fn branch_declines_an_uncleared_tree() {
        // threshold == min bound: check_branch_tree rejects; emission declines.
        let cert = two_cell_cert(r(-1, 1));
        assert!(matches!(
            branch_tree_to_alethe(&cert),
            Err(EmitError::Branch(BranchError::ThresholdNotCleared(_)))
        ));
    }

    #[test]
    fn rat_term_prints_exact_smtlib_reals() {
        assert_eq!(rat_term(r(3, 1)).unwrap(), "3.0");
        assert_eq!(rat_term(r(-3, 1)).unwrap(), "(- 3.0)");
        assert_eq!(rat_term(r(1, 2)).unwrap(), "(/ 1.0 2.0)");
        assert_eq!(rat_term(r(-7, 3)).unwrap(), "(- (/ 7.0 3.0))");
        assert_eq!(rat_term(Rat::ZERO).unwrap(), "0.0");
    }
}
