// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The checker's own VNN-LIB reader and clause canonicalizer (no code shared
//! with any producer's spec parser; the producer's `clauses` records are
//! compared against THIS parse, so a U1 spec-translation disagreement is a
//! reported finding, not a silent pass).
//!
//! # Admitted forms
//!
//! * `(declare-const X_i Real)` / `(declare-const Y_j Real)`, indices
//!   contiguous from 0;
//! * `(assert A)` where `A` is an atom, an `(and ...)` of atoms, or ONE
//!   top-level `(or ...)` whose children are atoms or `(and ...)` of atoms;
//! * atoms `(op P Q)`, `op` in `<= >= < >`, over `{X_i, Y_j, decimal}` with
//!   no atom mixing `X` and `Y`; an `X` atom is a bound `X_i op c`.
//!
//! Anything else (another command, `=`, arithmetic terms, bracket-indexed
//! VNN-LIB 2.0 names, nested disjunctions under a conjunction, two top-level
//! disjunctions, an unbounded input, a gzip file) is `unsupported`.
//!
//! # Regions and disjuncts
//!
//! The asserted set is `X ∧ (D_1 ∨ … ∨ D_K)`. When the top-level disjunction
//! carries `X` atoms, each child is its own REGION (its own box) with one
//! disjunct; otherwise there is one region with `K` disjuncts. A property is
//! unsat iff for every region `r` and disjunct `k` no `x ∈ X_r` has
//! `N(x) ∈ D_{r,k}`.
//!
//! # Canonical form (the bundle's `clauses` strings)
//!
//! Each atom is rewritten `Σ c_i Y_i <= d` (`(<= A B)` → `A − B <= 0`,
//! `(>= A B)` → `B − A <= 0`, constants moved right; `<` for strict atoms),
//! coefficients and `d` exact rationals in lowest terms (`n` or `n/d`), zero
//! coefficients dropped, sorted by index, written `Yi:c,…<=d`. A disjunct is
//! its distinct atom strings sorted and joined by `&`.
//!
//! # Rounding
//!
//! Box bounds are parsed with Rust's correctly rounded `f64::from_str` and
//! widened one ulp OUTWARD for the bound computations (`lo_out`, `hi_out` ⊇
//! the exact box) and one ulp INWARD for refutation probes (`lo_in`, `hi_in`
//! ⊆ the exact box), unless the decimal is exactly representable. An atom's
//! right-hand side is kept as `d_up >= d >= d_down`.

use std::collections::BTreeMap;

use crate::arith::{down, up};

/// Why the reader refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecError(pub String);

fn unsup<T>(what: impl Into<String>) -> Result<T, SpecError> {
    Err(SpecError(what.into()))
}

/// An exact rational in lowest terms, `den > 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q {
    /// Numerator.
    pub num: i128,
    /// Denominator (> 0).
    pub den: i128,
}

const fn gcd(mut a: i128, mut b: i128) -> i128 {
    if a < 0 {
        a = -a;
    }
    if b < 0 {
        b = -b;
    }
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

impl Q {
    /// `num / den`, reduced; `None` on overflow or zero denominator.
    #[must_use]
    pub fn new(num: i128, den: i128) -> Option<Self> {
        if den == 0 {
            return None;
        }
        let (mut n, mut d) = if den < 0 {
            (num.checked_neg()?, den.checked_neg()?)
        } else {
            (num, den)
        };
        let g = gcd(n, d);
        if g > 1 {
            n /= g;
            d /= g;
        }
        Some(Self { num: n, den: d })
    }

    /// Zero.
    pub const ZERO: Self = Self { num: 0, den: 1 };

    /// Sum; `None` on overflow (named like `i128::checked_add`).
    #[must_use]
    pub fn checked_add(self, o: Self) -> Option<Self> {
        let n = self
            .num
            .checked_mul(o.den)?
            .checked_add(o.num.checked_mul(self.den)?)?;
        Self::new(n, self.den.checked_mul(o.den)?)
    }

    /// Negation; `None` on overflow (named like `i128::checked_neg`).
    #[must_use]
    pub fn checked_neg(self) -> Option<Self> {
        Some(Self {
            num: self.num.checked_neg()?,
            den: self.den,
        })
    }

    /// Canonical text: `n` or `n/d`.
    #[must_use]
    pub fn text(self) -> String {
        if self.den == 1 {
            format!("{}", self.num)
        } else {
            format!("{}/{}", self.num, self.den)
        }
    }

    /// Whether the value is exactly an f64 (dyadic with a 53-bit numerator).
    fn exact_f64(self) -> bool {
        let d = self.den as u128;
        d.is_power_of_two() && self.num.unsigned_abs() <= (1u128 << 53)
    }

    /// `(down, up)` f64 neighbours enclosing the exact value.
    #[must_use]
    pub fn f64_bounds(self) -> (f64, f64) {
        let v = self.num as f64 / self.den as f64;
        if self.exact_f64() {
            return (v, v);
        }
        // fl(n), fl(d) and the quotient each round by at most half an ulp:
        // four one-ulp steps outward dominate the three roundings.
        let (mut lo, mut hi) = (v, v);
        for _ in 0..4 {
            lo = down(lo);
            hi = up(hi);
        }
        (lo, hi)
    }
}

/// Parse a decimal literal (`-12.5`, `3`, `1e-3`, `.5`) exactly.
#[must_use]
pub fn parse_decimal(text: &str) -> Option<Q> {
    let s = text.trim();
    let (neg, body) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    let (mant, exp) = match body.find(['e', 'E']) {
        Some(p) => (&body[..p], body[p + 1..].parse::<i32>().ok()?),
        None => (body, 0),
    };
    let (ip, fp) = match mant.find('.') {
        Some(p) => (&mant[..p], &mant[p + 1..]),
        None => (mant, ""),
    };
    if ip.is_empty() && fp.is_empty() {
        return None;
    }
    if !ip.bytes().all(|b| b.is_ascii_digit()) || !fp.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut num: i128 = 0;
    for b in ip.bytes().chain(fp.bytes()) {
        num = num.checked_mul(10)?.checked_add(i128::from(b - b'0'))?;
    }
    let scale = exp.checked_sub(i32::try_from(fp.len()).ok()?)?;
    let mut den: i128 = 1;
    if scale >= 0 {
        for _ in 0..scale {
            num = num.checked_mul(10)?;
        }
    } else {
        for _ in 0..(-scale) {
            den = den.checked_mul(10)?;
        }
    }
    if neg {
        num = num.checked_neg()?;
    }
    Q::new(num, den)
}

/// One output atom `Σ c_i Y_i <= d` (or `<` when strict).
#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    /// `(Y index, coefficient)`, sorted, nonzero, exact small integers.
    pub coeffs: Vec<(usize, f64)>,
    /// Strict comparison.
    pub strict: bool,
    /// Exact right-hand side.
    pub d: Q,
    /// `d_down <= d`.
    pub d_down: f64,
    /// `d_up >= d`.
    pub d_up: f64,
    /// Canonical text.
    pub canonical: String,
}

/// One disjunct: a conjunction of atoms.
#[derive(Debug, Clone, PartialEq)]
pub struct Disjunct {
    /// Atoms (distinct canonical strings, sorted).
    pub atoms: Vec<Atom>,
    /// Canonical text (`&`-joined sorted atom strings).
    pub canonical: String,
}

/// One input region with its disjuncts.
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    /// Outward lower bounds.
    pub lo_out: Vec<f64>,
    /// Outward upper bounds.
    pub hi_out: Vec<f64>,
    /// Inward lower bounds (probe points live in `[lo_in, hi_in]`).
    pub lo_in: Vec<f64>,
    /// Inward upper bounds.
    pub hi_in: Vec<f64>,
    /// Disjuncts of this region.
    pub disjuncts: Vec<Disjunct>,
}

/// A parsed property.
#[derive(Debug, Clone, PartialEq)]
pub struct Spec {
    /// Number of `X` variables.
    pub n_x: usize,
    /// Number of `Y` variables.
    pub n_y: usize,
    /// Regions.
    pub regions: Vec<Region>,
}

#[derive(Debug, Clone, PartialEq)]
enum Sx {
    List(Vec<Sx>),
    Sym(String),
}

fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.split(';').next().unwrap_or("");
        let mut cur = String::new();
        for ch in line.chars() {
            match ch {
                '(' | ')' => {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                    out.push(ch.to_string());
                }
                c if c.is_whitespace() => {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                }
                c => cur.push(c),
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
    }
    out
}

fn parse_sx(tokens: &[String]) -> Result<Vec<Sx>, SpecError> {
    let mut stack: Vec<Vec<Sx>> = vec![Vec::new()];
    for t in tokens {
        match t.as_str() {
            "(" => stack.push(Vec::new()),
            ")" => {
                let done = stack
                    .pop()
                    .ok_or_else(|| SpecError("unbalanced )".into()))?;
                let parent = stack
                    .last_mut()
                    .ok_or_else(|| SpecError("unbalanced )".into()))?;
                parent.push(Sx::List(done));
            }
            s => stack
                .last_mut()
                .ok_or_else(|| SpecError("unbalanced".into()))?
                .push(Sx::Sym(s.to_string())),
        }
    }
    if stack.len() != 1 {
        return unsup("unbalanced parentheses");
    }
    Ok(stack.pop().unwrap_or_default())
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Term {
    X(usize),
    Y(usize),
    C(Q),
}

fn var_index(s: &str, prefix: &str) -> Option<usize> {
    let rest = s.strip_prefix(prefix)?;
    if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if rest.len() > 1 && rest.starts_with('0') {
        return None;
    }
    rest.parse().ok()
}

fn term(sx: &Sx) -> Result<(Term, String), SpecError> {
    let Sx::Sym(s) = sx else {
        return unsup("arithmetic terms are not supported");
    };
    if let Some(i) = var_index(s, "X_") {
        return Ok((Term::X(i), s.clone()));
    }
    if let Some(j) = var_index(s, "Y_") {
        return Ok((Term::Y(j), s.clone()));
    }
    match parse_decimal(s) {
        Some(q) => Ok((Term::C(q), s.clone())),
        None => unsup(format!("unsupported term `{s}`")),
    }
}

/// One bound on an input variable.
#[derive(Debug, Clone)]
struct XBound {
    var: usize,
    lower: bool,
    strict: bool,
    text: String,
    q: Q,
}

#[derive(Debug, Clone)]
enum Lit {
    X(XBound),
    Y(Atom),
}

fn make_atom(op: &str, a: Term, b: Term) -> Result<Option<Atom>, SpecError> {
    // A - B <= 0 for <=/<; B - A <= 0 for >=/>.
    let (strict, pos, neg) = match op {
        "<=" => (false, a, b),
        "<" => (true, a, b),
        ">=" => (false, b, a),
        ">" => (true, b, a),
        _ => return unsup(format!("comparison `{op}`")),
    };
    let mut coeffs: BTreeMap<usize, i128> = BTreeMap::new();
    let mut d = Q::ZERO;
    let overflow = || SpecError("constant overflow".into());
    match pos {
        Term::Y(j) => *coeffs.entry(j).or_insert(0) += 1,
        Term::C(q) => {
            d = d
                .checked_add(q.checked_neg().ok_or_else(overflow)?)
                .ok_or_else(overflow)?
        }
        Term::X(_) => return unsup("atom mixes X and Y"),
    }
    match neg {
        Term::Y(j) => *coeffs.entry(j).or_insert(0) -= 1,
        Term::C(q) => d = d.checked_add(q).ok_or_else(overflow)?,
        Term::X(_) => return unsup("atom mixes X and Y"),
    }
    let coeffs: Vec<(usize, i128)> = coeffs.into_iter().filter(|(_, c)| *c != 0).collect();
    if coeffs.is_empty() {
        // Constant truth value: 0 <= d (or 0 < d).
        let holds = if strict { d.num > 0 } else { d.num >= 0 };
        if holds {
            return Ok(None);
        }
        return unsup("constant-false atom");
    }
    let lhs: Vec<String> = coeffs.iter().map(|(j, c)| format!("Y{j}:{c}")).collect();
    let canonical = format!(
        "{}{}{}",
        lhs.join(","),
        if strict { "<" } else { "<=" },
        d.text()
    );
    let (d_down, d_up) = d.f64_bounds();
    Ok(Some(Atom {
        coeffs: coeffs.iter().map(|&(j, c)| (j, c as f64)).collect(),
        strict,
        d,
        d_down,
        d_up,
        canonical,
    }))
}

fn literal(sx: &Sx) -> Result<Option<Lit>, SpecError> {
    let Sx::List(items) = sx else {
        return unsup("bare symbol where an atom was expected");
    };
    let [Sx::Sym(op), a, b] = items.as_slice() else {
        return unsup("atom must be (op A B)");
    };
    if !matches!(op.as_str(), "<=" | ">=" | "<" | ">") {
        return unsup(format!("operator `{op}`"));
    }
    let (ta, sa) = term(a)?;
    let (tb, sb) = term(b)?;
    let strict = op == "<" || op == ">";
    let le = op == "<=" || op == "<";
    match (ta, tb) {
        (Term::X(i), Term::C(q)) => Ok(Some(Lit::X(XBound {
            var: i,
            lower: !le,
            strict,
            text: sb,
            q,
        }))),
        (Term::C(q), Term::X(i)) => Ok(Some(Lit::X(XBound {
            var: i,
            lower: le,
            strict,
            text: sa,
            q,
        }))),
        (Term::X(_), _) | (_, Term::X(_)) => unsup("input atom that is not a bound"),
        _ => Ok(make_atom(op, ta, tb)?.map(Lit::Y)),
    }
}

fn conjunction(sx: &Sx, out: &mut Vec<Lit>) -> Result<(), SpecError> {
    if let Sx::List(items) = sx {
        if let Some(Sx::Sym(h)) = items.first() {
            if h == "and" {
                for c in &items[1..] {
                    conjunction(c, out)?;
                }
                return Ok(());
            }
            if h == "or" {
                return unsup("disjunction nested under a conjunction");
            }
        }
    }
    if let Some(l) = literal(sx)? {
        out.push(l);
    }
    Ok(())
}

fn disjunction(sx: &Sx, out: &mut Vec<Vec<Lit>>) -> Result<(), SpecError> {
    if let Sx::List(items) = sx {
        if let Some(Sx::Sym(h)) = items.first() {
            if h == "or" {
                for c in &items[1..] {
                    disjunction(c, out)?;
                }
                return Ok(());
            }
        }
    }
    let mut conj = Vec::new();
    conjunction(sx, &mut conj)?;
    out.push(conj);
    Ok(())
}

fn build_box(
    n_x: usize,
    bounds: &[&XBound],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), SpecError> {
    let mut lo_out = vec![f64::NEG_INFINITY; n_x];
    let mut hi_out = vec![f64::INFINITY; n_x];
    let mut lo_in = vec![f64::NEG_INFINITY; n_x];
    let mut hi_in = vec![f64::INFINITY; n_x];
    for b in bounds {
        if b.var >= n_x {
            return unsup(format!("X_{} is not declared", b.var));
        }
        let v: f64 = b
            .text
            .parse()
            .map_err(|_| SpecError(format!("bound `{}`", b.text)))?;
        if !v.is_finite() {
            return unsup("non-finite bound");
        }
        let exact = b.q.exact_f64();
        let (d, u) = if exact { (v, v) } else { (down(v), up(v)) };
        // Inward: strictly inside for a strict bound.
        let (din, uin) = (
            if exact && !b.strict { v } else { up(v) },
            if exact && !b.strict { v } else { down(v) },
        );
        if b.lower {
            lo_out[b.var] = lo_out[b.var].max(d);
            lo_in[b.var] = lo_in[b.var].max(din);
        } else {
            hi_out[b.var] = hi_out[b.var].min(u);
            hi_in[b.var] = hi_in[b.var].min(uin);
        }
    }
    for i in 0..n_x {
        if !(lo_out[i].is_finite() && hi_out[i].is_finite()) {
            return unsup(format!("X_{i} is not bounded on both sides"));
        }
        if lo_out[i] > hi_out[i] {
            return unsup(format!("X_{i} has an empty range"));
        }
    }
    Ok((lo_out, hi_out, lo_in, hi_in))
}

fn make_disjunct(atoms: Vec<Atom>) -> Result<Disjunct, SpecError> {
    let mut by: BTreeMap<String, Atom> = BTreeMap::new();
    for a in atoms {
        by.entry(a.canonical.clone()).or_insert(a);
    }
    if by.is_empty() {
        return unsup("a disjunct with no output constraint");
    }
    let canonical = by.keys().cloned().collect::<Vec<_>>().join("&");
    Ok(Disjunct {
        atoms: by.into_values().collect(),
        canonical,
    })
}

/// Parse a VNN-LIB property.
pub fn parse(text: &str) -> Result<Spec, SpecError> {
    let top = parse_sx(&tokenize(text))?;
    let mut xs: Vec<usize> = Vec::new();
    let mut ys: Vec<usize> = Vec::new();
    let mut global: Vec<Lit> = Vec::new();
    let mut parts: Option<Vec<Vec<Lit>>> = None;
    for cmd in &top {
        let Sx::List(items) = cmd else {
            return unsup("stray symbol at top level");
        };
        let Some(Sx::Sym(head)) = items.first() else {
            return unsup("empty command");
        };
        match head.as_str() {
            "declare-const" => {
                let [_, Sx::Sym(name), Sx::Sym(ty)] = items.as_slice() else {
                    return unsup("declare-const shape");
                };
                if ty != "Real" {
                    return unsup(format!("declare-const type `{ty}`"));
                }
                if let Some(i) = var_index(name, "X_") {
                    xs.push(i);
                } else if let Some(j) = var_index(name, "Y_") {
                    ys.push(j);
                } else {
                    return unsup(format!("variable `{name}` (only X_i / Y_j)"));
                }
            }
            "assert" => {
                let [_, body] = items.as_slice() else {
                    return unsup("assert shape");
                };
                let is_or = matches!(body, Sx::List(v) if matches!(v.first(), Some(Sx::Sym(h)) if h == "or"));
                if is_or {
                    if parts.is_some() {
                        return unsup("two top-level disjunctions");
                    }
                    let mut p = Vec::new();
                    disjunction(body, &mut p)?;
                    parts = Some(p);
                } else {
                    conjunction(body, &mut global)?;
                }
            }
            other => return unsup(format!("command `{other}`")),
        }
    }
    let contiguous = |v: &mut Vec<usize>| {
        v.sort_unstable();
        v.iter().enumerate().all(|(k, &i)| k == i)
    };
    if xs.is_empty() || ys.is_empty() || !contiguous(&mut xs) || !contiguous(&mut ys) {
        return unsup("X_ and Y_ declarations must be contiguous from 0");
    }
    let (n_x, n_y) = (xs.len(), ys.len());
    let check_y = |a: &Atom| -> Result<(), SpecError> {
        if a.coeffs.iter().any(|&(j, _)| j >= n_y) {
            return unsup("Y index is not declared");
        }
        Ok(())
    };
    let global_x: Vec<&XBound> = global
        .iter()
        .filter_map(|l| match l {
            Lit::X(b) => Some(b),
            Lit::Y(_) => None,
        })
        .collect();
    let global_y: Vec<Atom> = global
        .iter()
        .filter_map(|l| match l {
            Lit::Y(a) => Some(a.clone()),
            Lit::X(_) => None,
        })
        .collect();
    for a in &global_y {
        check_y(a)?;
    }
    let mut regions = Vec::new();
    match parts {
        None => {
            let (lo_out, hi_out, lo_in, hi_in) = build_box(n_x, &global_x)?;
            regions.push(Region {
                lo_out,
                hi_out,
                lo_in,
                hi_in,
                disjuncts: vec![make_disjunct(global_y)?],
            });
        }
        Some(parts) => {
            let per_region = parts
                .iter()
                .any(|p| p.iter().any(|l| matches!(l, Lit::X(_))));
            if per_region {
                for p in &parts {
                    let mut xb = global_x.clone();
                    let mut ya = global_y.clone();
                    for l in p {
                        match l {
                            Lit::X(b) => xb.push(b),
                            Lit::Y(a) => {
                                check_y(a)?;
                                ya.push(a.clone());
                            }
                        }
                    }
                    let (lo_out, hi_out, lo_in, hi_in) = build_box(n_x, &xb)?;
                    regions.push(Region {
                        lo_out,
                        hi_out,
                        lo_in,
                        hi_in,
                        disjuncts: vec![make_disjunct(ya)?],
                    });
                }
            } else {
                let (lo_out, hi_out, lo_in, hi_in) = build_box(n_x, &global_x)?;
                let mut disjuncts = Vec::new();
                for p in &parts {
                    let mut ya = global_y.clone();
                    for l in p {
                        if let Lit::Y(a) = l {
                            check_y(a)?;
                            ya.push(a.clone());
                        }
                    }
                    disjuncts.push(make_disjunct(ya)?);
                }
                regions.push(Region {
                    lo_out,
                    hi_out,
                    lo_in,
                    hi_in,
                    disjuncts,
                });
            }
        }
    }
    Ok(Spec { n_x, n_y, regions })
}

/// The canonical atom string of `Y_t - Y_j <= 0` (a margin-row class `j`
/// against the true class `t`): the form the margin-row emitter must write.
#[must_use]
pub fn margin_clause(t: usize, j: usize) -> String {
    let mut c: BTreeMap<usize, i128> = BTreeMap::new();
    *c.entry(t).or_insert(0) += 1;
    *c.entry(j).or_insert(0) -= 1;
    let lhs: Vec<String> = c
        .iter()
        .filter(|(_, v)| **v != 0)
        .map(|(k, v)| format!("Y{k}:{v}"))
        .collect();
    format!("{}<=0", lhs.join(","))
}

#[cfg(test)]
mod tests {
    use super::{margin_clause, parse, parse_decimal, Q};

    #[test]
    fn decimals_are_exact() {
        assert_eq!(parse_decimal("0.5"), Q::new(1, 2));
        assert_eq!(parse_decimal("-1e-3"), Q::new(-1, 1000));
        assert_eq!(parse_decimal("12"), Q::new(12, 1));
        assert_eq!(parse_decimal(".25"), Q::new(1, 4));
        assert!(parse_decimal("1.2.3").is_none());
    }

    #[test]
    fn robustness_form_parses_to_one_region() {
        let text = "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n\
                    (assert (<= X_0 0.5))\n(assert (>= X_0 -0.5))\n\
                    (assert (or (and (>= Y_1 Y_0))))\n";
        let s = parse(text).expect("parses");
        assert_eq!(s.regions.len(), 1);
        assert_eq!(s.regions[0].disjuncts[0].canonical, "Y0:1,Y1:-1<=0");
        assert_eq!(margin_clause(0, 1), "Y0:1,Y1:-1<=0");
        assert_eq!(margin_clause(3, 1), "Y1:-1,Y3:1<=0");
    }
}
