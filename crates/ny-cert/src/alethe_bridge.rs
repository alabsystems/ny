// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Prototype bridge: an Alethe `la_generic` linear-arithmetic refutation step
//! (as emitted by the `ay` solver on a QF_LRA / ReLU-as-disjunction UNSAT
//! query) reduced to a ny-cert [`FarkasCertificate`] obligation, so the same
//! Clean-kernel-checked `farkas_premise_combination` theorem that grounds every
//! CROWN verdict also grounds an `ay` UNSAT.
//!
//! # What this closes and what it does not (READINESS, honest)
//! An `ay` Alethe proof of an NN-MILP subdomain UNSAT is, in shape, a Boolean
//! resolution skeleton (`or_pos` / `and_pos` / `resolution` / — in the current
//! smoke build — a final `trust` glue step) whose *arithmetic leaves* are
//! `la_generic` steps. Each `la_generic` step carries exactly the Farkas
//! multipliers (`:args`) of a non-negative combination that refutes a conjunction
//! of linear atoms — the identical obligation ny-cert's [`check_farkas`] already
//! discharges for CROWN. **This module bridges one `la_generic` leaf.** The
//! smallest remaining piece for an end-to-end `ay`-UNSAT → MipCert verdict is the
//! Boolean glue: replaying the resolution DAG that composes the leaves (and
//! eliminating `ay`'s `trust` step), which maps onto the corpus'
//! `MipCert.pattern_tree_cover` (case-split cover) rather than the Farkas core.
//! See `docs/AY_UNSAT_NY_CERT_LOOP.md`.
//!
//! Experimental / dark: not wired into any verdict path; exercised only by the
//! gated bridge test. Fail-closed on any shape it does not fully understand.

use crate::rational::Rat;
use crate::schema::{ConstraintKind, FarkasCertificate, LinearConstraint};
use crate::selfcheck::{check_farkas, CheckError};
use std::collections::BTreeMap;

/// Why an Alethe step could not be bridged to a Farkas obligation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    /// No `la_generic` step found in the input.
    NoLaGeneric,
    /// The clause literal count and `:args` coefficient count disagree.
    ArgCountMismatch(usize, usize),
    /// A literal was not the supported `(not <atom>)` shape.
    UnsupportedLiteral(String),
    /// An atom was not a supported `(<op> <term> <term>)` comparison.
    UnsupportedAtom(String),
    /// A term was neither a variable, a bare rational, nor `(/ n d)`.
    UnsupportedTerm(String),
    /// An *inequality* atom carried a negative multiplier — flipping an
    /// inequality is unsound, so we refuse it (only equalities may take any sign).
    NegativeIneqMultiplier(usize),
    /// The reconstructed Farkas certificate did not check out.
    Check(CheckError),
}

/// One parsed comparison atom `Σ cᵢ·vᵢ  ⋈  K`.
struct Atom {
    kind: ConstraintKind,
    coeffs: BTreeMap<String, Rat>,
    constant: Rat,
}

/// A term reduced to `(variable coefficients, constant)`.
struct Term {
    coeffs: BTreeMap<String, Rat>,
    constant: Rat,
}

/// Bridge the first `la_generic` step of an Alethe proof to a Farkas
/// certificate and check it exactly as Clean's kernel-side verifier does.
///
/// On success returns the collapsed contradiction constant (`≤ 0`, `< 0` if
/// strict) — the same witness [`check_farkas`] returns.
///
/// # Soundness
/// The bridge only *weakens* premises: an equality `E = K` used with multiplier
/// sign `s` is emitted as the one-sided inequality it implies (`E ≤ K` or
/// `E ≥ K`); inequalities are passed through unflipped with their non-negative
/// multiplier, and a negative multiplier on an inequality is refused. Thus any
/// Farkas contradiction [`check_farkas`] certifies over the emitted constraints
/// is a fortiori a contradiction over the original atoms — so `ay`'s
/// `la_generic` refutation is soundly re-established under the corpus'
/// kernel-checked `farkas_premise_combination`.
///
/// # Errors
/// [`BridgeError`] on any unsupported shape or a failed Farkas check.
pub fn bridge_la_generic(alethe: &str) -> Result<Rat, BridgeError> {
    let (literals, args) = find_la_generic(alethe).ok_or(BridgeError::NoLaGeneric)?;
    if literals.len() != args.len() {
        return Err(BridgeError::ArgCountMismatch(literals.len(), args.len()));
    }

    let mut constraints = Vec::new();
    let mut multipliers = Vec::new();
    for (idx, (lit, &arg)) in literals.iter().zip(&args).enumerate() {
        // Literals are `(not <atom>)`; the negation is the atom asserted in the
        // refuted conjunction (`la_generic` proves that conjunction unsat).
        // `match` not `ok_or_else(|| …)`: keeps the Ok/Err mapping in verified
        // code (no absent `<{closure} as Fn>::call` adapter), matching this
        // file's convention (parse_atom's split_sexpr handling).
        let atom_src = match strip_not(lit) {
            Some(inner) => inner,
            None => return Err(BridgeError::UnsupportedLiteral(lit.clone())),
        };
        let atom = parse_atom(atom_src)?;

        let (kind, mult) = match atom.kind {
            ConstraintKind::Eq => {
                // Equalities may take any-sign multiplier: emit the one-sided
                // inequality the equality implies, orienting by the sign.
                let kind = if arg.is_negative() {
                    ConstraintKind::Ge
                } else {
                    ConstraintKind::Le
                };
                (kind, if arg.is_negative() { arg.neg() } else { arg })
            }
            other => {
                if arg.is_negative() {
                    return Err(BridgeError::NegativeIneqMultiplier(idx));
                }
                (other, arg)
            }
        };
        // Explicit Vec::new()+push (not `.collect()`): the length is the
        // input-derived atom coefficient count, so a bulk `.collect()` raises
        // an unbounded-alloc obligation; identical elements and order.
        let mut terms: Vec<(&str, Rat)> = Vec::new();
        for (k, v) in &atom.coeffs {
            terms.push((k.as_str(), *v));
        }
        constraints.push(LinearConstraint::with_kind(kind, &terms, atom.constant));
        multipliers.push(mult);
    }

    let cert = FarkasCertificate {
        constraints,
        multipliers,
    };
    // match instead of `.map_err(closure)`: keeps the Ok/Err mapping in
    // verified code (no absent-adapter `Result::map_err` obligation).
    match check_farkas(&cert) {
        Ok(r) => Ok(r),
        Err(e) => Err(BridgeError::Check(e)),
    }
}

/// Strip a leading `(not …)`, returning the inner atom source.
fn strip_not(lit: &str) -> Option<&str> {
    let t = lit.trim();
    let inner = t.strip_prefix("(not")?;
    let inner = inner.strip_suffix(')')?;
    Some(inner.trim())
}

/// Parse a comparison atom `(<op> <term> <term>)` into `Σ cᵢ·vᵢ ⋈ K`.
fn parse_atom(src: &str) -> Result<Atom, BridgeError> {
    let items = split_sexpr(src).ok_or_else(|| BridgeError::UnsupportedAtom(src.to_string()))?;
    if items.len() != 3 {
        return Err(BridgeError::UnsupportedAtom(src.to_string()));
    }
    // total: `items.len() == 3` was checked above, so `first`/`get(1)`/`get(2)`
    // are always `Some` — the fallbacks are unreachable and fail closed
    // (reject as unsupported) rather than index.
    // match instead of `.map(String::as_str)`: keeps the head-token deref in
    // verified code (no absent-adapter `Option::map` obligation).
    let head: Option<&str> = match items.first() {
        Some(s) => Some(s.as_str()),
        None => None,
    };
    let kind = match head {
        Some("<=") => ConstraintKind::Le,
        Some("<") => ConstraintKind::Lt,
        Some("=") => ConstraintKind::Eq,
        Some(">=") => ConstraintKind::Ge,
        Some(">") => ConstraintKind::Gt,
        _ => return Err(BridgeError::UnsupportedAtom(src.to_string())),
    };
    let bad = || BridgeError::UnsupportedAtom(src.to_string());
    let lhs = parse_term(items.get(1).ok_or_else(bad)?)?;
    let rhs = parse_term(items.get(2).ok_or_else(bad)?)?;
    // Move variables left, constants right: (L.coeffs − R.coeffs) ⋈ (R.const − L.const).
    // By-ref iteration + `get`/`insert` (not by-value `for … in rhs.coeffs`
    // with `entry(..).or_insert_with(closure)`): keeps the merge in verified
    // code (no absent `BTreeMap::into_iter` or closure-Fn obligation).
    // `insert` on a present key replaces the value and keeps the existing key
    // — identical entries, same ascending visit order, same early-`Err`.
    // match instead of `.map_err(closure)?`: keeps the Ok/Err mapping in
    // verified code (no absent-adapter `Result::map_err` obligation).
    let mut coeffs = lhs.coeffs;
    for (v, c) in &rhs.coeffs {
        let cur = match coeffs.get(v) {
            Some(e) => *e,
            None => Rat::from_int(0),
        };
        // `add` is arbitrary-precision and documented infallible.
        let next = match cur.add(c.neg()) {
            Ok(r) => r,
            Err(_) => return Err(BridgeError::UnsupportedTerm(src.to_string())),
        };
        coeffs.insert(v.clone(), next);
    }
    coeffs.retain(|_, c| !c.is_zero());
    let constant = rhs
        .constant
        .add(lhs.constant.neg())
        .map_err(|_| BridgeError::UnsupportedTerm(src.to_string()))?;
    Ok(Atom {
        kind,
        coeffs,
        constant,
    })
}

/// Parse a term: a variable, a bare rational, or `(/ n d)`.
fn parse_term(src: &str) -> Result<Term, BridgeError> {
    let t = src.trim();
    if let Some(items) = split_sexpr(t) {
        // `(/ num den)` division form.
        // total: guarded by `items.len() == 3`, so the `get`s are always
        // `Some` — the fallbacks are unreachable and fail closed (reject as
        // unsupported) rather than index.
        // match instead of `.map(String::as_str)`: keeps the head-token compare
        // in verified code (no absent-adapter `Option::map` obligation).
        if items.len() == 3 && matches!(items.first(), Some(s) if s.as_str() == "/") {
            let bad = || BridgeError::UnsupportedTerm(t.to_string());
            let n = parse_rat_atom(items.get(1).ok_or_else(bad)?)?;
            let d = parse_rat_atom(items.get(2).ok_or_else(bad)?)?;
            let d_inv = d
                .inv()
                .map_err(|_| BridgeError::UnsupportedTerm(t.to_string()))?;
            let q = n
                .mul(d_inv)
                .map_err(|_| BridgeError::UnsupportedTerm(t.to_string()))?;
            return Ok(Term {
                coeffs: BTreeMap::new(),
                constant: q,
            });
        }
        return Err(BridgeError::UnsupportedTerm(t.to_string()));
    }
    // Atom: numeric literal or a variable name.
    if let Ok(r) = parse_rat_atom(t) {
        return Ok(Term {
            coeffs: BTreeMap::new(),
            constant: r,
        });
    }
    let mut coeffs = BTreeMap::new();
    coeffs.insert(t.to_string(), Rat::from_int(1));
    Ok(Term {
        coeffs,
        constant: Rat::from_int(0),
    })
}

/// Parse a bare numeric atom: `N`, `N.M`, or `-N` into an exact rational.
fn parse_rat_atom(s: &str) -> Result<Rat, BridgeError> {
    let s = s.trim();
    if let Some((int, frac)) = s.split_once('.') {
        // Decimal `int.frac` → (int·10^k + frac) / 10^k, exact.
        let neg = int.starts_with('-');
        let int_digits: String = int.chars().filter(|c| c.is_ascii_digit()).collect();
        let scale = 10i128
            .checked_pow(frac.len() as u32)
            .ok_or_else(|| BridgeError::UnsupportedTerm(s.to_string()))?;
        let ip: i128 = int_digits
            .parse()
            .map_err(|_| BridgeError::UnsupportedTerm(s.to_string()))?;
        let fp: i128 = if frac.is_empty() {
            0
        } else {
            frac.parse()
                .map_err(|_| BridgeError::UnsupportedTerm(s.to_string()))?
        };
        let mag = ip
            .checked_mul(scale)
            .and_then(|v| v.checked_add(fp))
            .ok_or_else(|| BridgeError::UnsupportedTerm(s.to_string()))?;
        // total: `checked_neg` (not `-mag`): `-i128::MIN` would overflow. `mag`
        // is in practice `> i128::MIN` (`ip·scale >= 0` and `|fp| < scale <=
        // 10^38`), so the `Err` arm is unreachable — and a pathological input
        // still fails closed as UnsupportedTerm instead of wrapping/panicking.
        let num = if neg {
            mag.checked_neg()
                .ok_or_else(|| BridgeError::UnsupportedTerm(s.to_string()))?
        } else {
            mag
        };
        return Rat::new(num, scale).map_err(|_| BridgeError::UnsupportedTerm(s.to_string()));
    }
    let n: i128 = s
        .parse()
        .map_err(|_| BridgeError::UnsupportedTerm(s.to_string()))?;
    Ok(Rat::from_int(n))
}

/// Locate the first `la_generic` step; return its clause literals and `:args`.
fn find_la_generic(alethe: &str) -> Option<(Vec<String>, Vec<Rat>)> {
    for line in alethe.lines() {
        let line = line.trim();
        if !line.contains(":rule la_generic") {
            continue;
        }
        // Extract the `(cl …)` clause and the `:args (…)` list.
        let cl_body = between(line, "(cl", ") :rule")?;
        // Manual construction (not `format!`): a `format!` lowers to
        // `std::fmt::format`, an extern dispatch the verifier flags as possibly
        // running a user Display/Debug. `cl_body` is a `&str` whose Display is
        // the identity, so `"(cl" + cl_body + ")"` is byte-identical and total.
        let mut cl_src = String::from("(cl");
        cl_src.push_str(cl_body);
        cl_src.push(')');
        let cl_items = split_sexpr(&cl_src)?;
        // Explicit Vec::new()+push (not `.collect()`): input-derived counts
        // raise unbounded bulk-alloc obligations; identical Vecs, same order
        // (and the same early-`None` on a bad coefficient).
        let mut literals = Vec::new();
        for lit in cl_items.into_iter().skip(1) {
            literals.push(lit); // drop "cl"
        }
        let args_src = args_group(line)?;
        let arg_items = split_sexpr(&args_src)?;
        let mut args = Vec::new();
        for tok in &arg_items {
            args.push(parse_arg_rat(tok)?);
        }
        return Some((literals, args));
    }
    None
}

/// Extract the balanced `(…)` group following `:args` (coefficients may be
/// parenthesized terms like `(/ 1.0 2.0)` or `(- 1)`, so a naive
/// substring-to-`)` scan would truncate them).
// Explicit matches (not `.map`): keeps the conversions in verified code (no
// absent-adapter `Option::map` obligation) — this file's convention.
#[allow(clippy::manual_map)]
fn args_group(line: &str) -> Option<String> {
    // total: `checked_add` (not `+`): both operands are `find`-derived offsets
    // `<= isize::MAX`, so the sum never overflows `usize` — the `None` arm is
    // unreachable and fails closed instead of wrapping.
    let start = line.find(":args")?.checked_add(":args".len())?;
    let rest = line.get(start..)?;
    let open = rest.find('(')?;
    let body = rest.get(open..)?;
    let mut depth = 0i32;
    for (i, ch) in body.char_indices() {
        match ch {
            '(' => {
                // total: `checked_add`/`checked_sub` (not `+= 1`/`-= 1`): the
                // nesting counter over/underflows only beyond ±2³¹ parens —
                // unreachable in practice; a pathological input fails closed.
                depth = depth.checked_add(1)?;
            }
            ')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    // `)` is one byte, so `..=i` is boundary-aligned.
                    // match instead of `.map(str::to_string)`: keeps the
                    // conversion in verified code (no absent-adapter
                    // `Option::map` obligation).
                    return match body.get(..=i) {
                        Some(g) => Some(g.to_string()),
                        None => None,
                    };
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse one `:args` coefficient into an exact rational. Accepts bare
/// numerals (`1`, `-1`, `1.5`), SMT-LIB negation `(- t)`, and division
/// `(/ n d)` (each side itself a bare numeral).
// Explicit matches (not `?`/`.map`/`.ok()`): keeps the Ok/Err and Option
// mappings in verified code (no absent-adapter `Option::map`/`Result::ok`
// obligation) — this file's convention.
#[allow(clippy::manual_map, clippy::manual_ok_err, clippy::question_mark)]
fn parse_arg_rat(s: &str) -> Option<Rat> {
    let t = s.trim();
    if let Some(items) = split_sexpr(t) {
        // match instead of `.map(String::as_str)`: keeps the head-token compare
        // in verified code (no absent-adapter `Option::map` obligation).
        if items.len() == 2 && matches!(items.first(), Some(h) if h.as_str() == "-") {
            let inner = match items.get(1) {
                Some(x) => x,
                None => return None, // unreachable: len == 2
            };
            return match parse_arg_rat(inner) {
                Some(r) => Some(r.neg()),
                None => None,
            };
        }
        if items.len() == 3 && matches!(items.first(), Some(h) if h.as_str() == "/") {
            let n = match items.get(1) {
                Some(x) => match parse_rat_atom(x) {
                    Ok(r) => r,
                    Err(_) => return None,
                },
                None => return None, // unreachable: len == 3
            };
            let d = match items.get(2) {
                Some(x) => match parse_rat_atom(x) {
                    Ok(r) => r,
                    Err(_) => return None,
                },
                None => return None, // unreachable: len == 3
            };
            let d_inv = match d.inv() {
                Ok(v) => v,
                Err(_) => return None,
            };
            return match n.mul(d_inv) {
                Ok(v) => Some(v),
                Err(_) => None,
            };
        }
        return None;
    }
    match parse_rat_atom(t) {
        Ok(r) => Some(r),
        Err(_) => None,
    }
}

/// Substring strictly between the first `start` and the following `end`.
fn between<'a>(hay: &'a str, start: &str, end: &str) -> Option<&'a str> {
    // total: `checked_add` (not `+`): both operands are string lengths/offsets
    // `<= isize::MAX`, so the sum never overflows `usize` — the `None` arm is
    // unreachable and fails closed instead of wrapping.
    let s = hay.find(start)?.checked_add(start.len())?;
    // total: both offsets are `find`-derived and in-bounds/boundary-aligned by
    // contract, which the intraprocedural verifier cannot see; `get` + `?` is
    // behavior-identical and fails closed (`None`) instead of slicing.
    let rest = hay.get(s..)?;
    let e = rest.find(end)?;
    rest.get(..e)
}

/// Split the top-level children of a single s-expression `(a b c)` into their
/// source strings (nested parens kept intact as one child). Returns `None` if
/// the input is not a parenthesized list.
fn split_sexpr(src: &str) -> Option<Vec<String>> {
    let t = src.trim();
    let inner = t.strip_prefix('(')?.strip_suffix(')')?;
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in inner.chars() {
        match ch {
            '(' => {
                // total: `checked_add`/`checked_sub` (not `+= 1`/`-= 1`): the
                // nesting counter over input text over/underflows only beyond
                // ±2³¹ parens — unreachable in practice, and a pathological
                // input fails closed (`None` = not a parseable s-expression)
                // instead of wrapping.
                depth = depth.checked_add(1)?;
                cur.push(ch);
            }
            ')' => {
                depth = depth.checked_sub(1)?;
                cur.push(ch);
            }
            c if c.is_whitespace() && depth == 0 => {
                if !cur.is_empty() {
                    // Move + fresh `String::new()` (not `std::mem::take(&mut cur)`):
                    // `mem::take` is an absent-callee for the panic-freedom checker;
                    // this is its exact effect — push the old string, leave `cur`
                    // holding a fresh empty `String` (`String::default()`).
                    out.push(cur);
                    cur = String::new();
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact `la_generic` step `ay` emits for `smoke/relu1_unsat.smt2`.
    const RELU1_UNSAT_LA_GENERIC: &str = "(step t4 (cl (not (>= x (/ 1.0 1.0))) (not (< y (/ 1.0 2.0))) (not (<= x z)) (not (= y z))) :rule la_generic :args (1 1 1 -1))";

    #[test]
    fn bridges_relu1_unsat_la_generic_to_farkas_contradiction() {
        let c = bridge_la_generic(RELU1_UNSAT_LA_GENERIC).expect("bridge + Farkas check");
        // The refutation collapses to 0 < −1/2 (strict), so the witness constant
        // is −1/2 ≤ 0: a genuine Farkas contradiction, kernel-schema-checked.
        assert!(
            !c.is_positive(),
            "contradiction constant must be ≤ 0, got {c:?}"
        );
        assert_eq!(c, Rat::new(-1, 2).unwrap());
    }

    #[test]
    fn refuses_negative_multiplier_on_an_inequality() {
        // Same clause but the first (inequality) literal gets a negative coeff:
        // flipping an inequality is unsound, so the bridge must fail closed.
        let bad = "(step t (cl (not (>= x (/ 1.0 1.0))) (not (< y (/ 1.0 2.0))) (not (<= x z)) (not (= y z))) :rule la_generic :args (-1 1 1 -1))";
        assert!(matches!(
            bridge_la_generic(bad),
            Err(BridgeError::NegativeIneqMultiplier(0))
        ));
    }

    #[test]
    fn parses_decimal_and_division_terms() {
        let a = parse_atom("(>= x (/ 3.0 2.0))").unwrap();
        assert_eq!(a.kind, ConstraintKind::Ge);
        assert_eq!(a.constant, Rat::new(3, 2).unwrap());
    }
}
