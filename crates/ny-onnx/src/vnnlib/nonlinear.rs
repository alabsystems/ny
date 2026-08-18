// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact evaluation of VNN-LIB 2.0 formulas containing NON-LINEAR arithmetic
//! (`#nonlinear-vnnlib`).
//!
//! # Why this exists
//!
//! The linear parser refuses any constraint whose arithmetic is not affine
//! (`parser.rs`, "Non-linear arithmetic expressions are not supported in
//! constraints"). `adaptive_cruise_control_non_linear_2026` is exactly such a
//! benchmark: its input region is a parabola
//! (`(>= (* X[0,0] 200.0) (* X[0,1] X[0,1]))`) and its output property mixes
//! input×output products (`(* (* X[0,0] 2.0) Y[0,0])`). The CLI now routes
//! this supported fragment through the exact-point and interval paths below;
//! formulas outside that fragment still fail closed to `unknown`.
//!
//! # What this module does
//!
//! For `sat`, it evaluates the ORIGINAL, UNRELAXED formula at a concrete `(x, y)`
//! point. A counterexample is accepted only if the full formula is definitely
//! true in exact rational arithmetic, including the exact source decimals.
//!
//! For `unsat`, it evaluates that same complete formula with outward-rounded
//! interval arithmetic over each input subbox and a certified enclosure of the
//! network outputs. A subbox is discharged only when the formula is definitely
//! false throughout it; undecided boxes must be split. The CLI's interval
//! branch-and-bound path may conclude `unsat` only after every subbox has been
//! discharged. A supported but interval-undecided expression returns
//! [`Tri::Unknown`]; an unsupported expression or unsupported floating-point
//! environment returns `None`. Neither result can become a proof.
//!
//! The tempting shortcut — dropping non-linear input constraints and calling
//! the rest a "relaxation" — is not used: it would be wrong for `sat`, and the
//! complete formula is cheap enough to retain in the interval proof.
//!
//! # The input box is a SUPERSET, on purpose
//!
//! [`NonLinearFormula::input_box`] returns the hull implied by the purely
//! interval constraints, ignoring the non-linear ones. That is a superset of the
//! true (non-convex) region, which is the right thing for driving a SEARCH: every
//! candidate it produces is re-checked against the complete formula by
//! [`NonLinearFormula::holds_at`], so a point from outside the real region is
//! rejected there rather than believed.

use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
use ny_core::{has_f64_interval_proof_environment, NyError, Result};

use super::{
    certified_input_box::{parse_exact_decimal, rational_to_lower_f64, rational_to_upper_f64},
    syntax::{parse_shape_string, strip_vnnlib_comments, tokenize},
};

const MAX_EXPRESSION_NESTING: usize = 256;
const MAX_EXPRESSION_TOKENS: usize = 16 * 1024 * 1024;

/// A decimal literal in the formula, retained both as the exact SMT-LIB
/// rational and as its narrowest outward `f64` enclosure.
///
/// The ordinary VNN-LIB AST stores only nearest `f64`. That is not sufficient
/// here: for example, nearest-f64 `0.1` is greater than exact decimal `0.1`,
/// so treating the rounded value as exact can turn a boundary comparison into
/// a false counterexample.
#[derive(Debug, Clone)]
struct Number {
    exact: BigRational,
    lower: f64,
    upper: f64,
}

/// Proof-sensitive S-expression representation used only by the nonlinear
/// evaluator. Numeric spellings must not pass through `f64` before their exact
/// meaning has been captured.
#[derive(Debug, Clone)]
enum Expr {
    Symbol(String),
    Number(Number),
    List(Vec<Self>),
}

/// Three-valued truth over a box: `Unknown` means "split further", never
/// "refuted".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tri {
    True,
    False,
    Unknown,
}

/// Build a `Tri` from a definitely-true and a definitely-false witness. The two
/// can never both hold; if neither does the answer is `Unknown`.
fn tri(definitely_true: bool, definitely_false: bool) -> Tri {
    debug_assert!(!(definitely_true && definitely_false));
    if definitely_true {
        Tri::True
    } else if definitely_false {
        Tri::False
    } else {
        Tri::Unknown
    }
}

/// Widen an interval one ulp OUTWARD on each side, covering the rounding of the
/// f64 operation that produced it. Non-finite or reversed intermediates refuse
/// evaluation: `f64::min`/`max` can otherwise discard a NaN corner and fabricate
/// an apparently finite proof bound.
fn out((lo, hi): (f64, f64)) -> Option<(f64, f64)> {
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return None;
    }
    let widened = (lo.next_down(), hi.next_up());
    (widened.0.is_finite() && widened.1.is_finite()).then_some(widened)
}

/// Which declared tensor a scalar variable belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VarKind {
    /// A network input coordinate.
    Input,
    /// A network output coordinate.
    Output,
}

#[derive(Debug, Clone)]
struct Decl {
    name: String,
    kind: VarKind,
    shape: Vec<usize>,
}

fn parse_formula_expressions(tokens: &[String]) -> Result<Vec<Expr>> {
    if tokens.len() > MAX_EXPRESSION_TOKENS {
        return Err(NyError::InvalidSpec(format!(
            "non-linear formula exceeds the {MAX_EXPRESSION_TOKENS}-token cap"
        )));
    }
    let mut expressions = Vec::new();
    let mut position = 0;
    while position < tokens.len() {
        let (expression, next) = parse_formula_expression(tokens, position, 0)?;
        expressions.push(expression);
        position = next;
    }
    Ok(expressions)
}

fn parse_formula_expression(
    tokens: &[String],
    position: usize,
    depth: usize,
) -> Result<(Expr, usize)> {
    if depth > MAX_EXPRESSION_NESTING {
        return Err(NyError::InvalidSpec(format!(
            "non-linear formula nesting exceeds {MAX_EXPRESSION_NESTING}"
        )));
    }
    let token = tokens
        .get(position)
        .ok_or_else(|| NyError::InvalidSpec("unexpected end of non-linear formula".to_string()))?;
    if token == "(" {
        let mut items = Vec::new();
        let mut next = position + 1;
        while tokens.get(next).is_some_and(|token| token != ")") {
            let (item, item_next) = parse_formula_expression(tokens, next, depth + 1)?;
            items.push(item);
            next = item_next;
        }
        if tokens.get(next).is_none() {
            return Err(NyError::InvalidSpec(
                "unmatched opening parenthesis in non-linear formula".to_string(),
            ));
        }
        return Ok((Expr::List(items), next + 1));
    }
    if token == ")" {
        return Err(NyError::InvalidSpec(
            "unexpected closing parenthesis in non-linear formula".to_string(),
        ));
    }
    // Match the ordinary parser's numeric-token classification, but retain the
    // exact decimal before doing anything with the nearest binary64 value.
    if token.parse::<f64>().is_ok() {
        let exact = parse_exact_decimal(token)?;
        let lower = rational_to_lower_f64(&exact, 0).map_err(|_| {
            NyError::InvalidSpec(format!(
                "non-linear numeric literal has no finite outward f64 enclosure: {token}"
            ))
        })?;
        let upper = rational_to_upper_f64(&exact, 0).map_err(|_| {
            NyError::InvalidSpec(format!(
                "non-linear numeric literal has no finite outward f64 enclosure: {token}"
            ))
        })?;
        return Ok((
            Expr::Number(Number {
                exact,
                lower,
                upper,
            }),
            position + 1,
        ));
    }
    Ok((Expr::Symbol(token.clone()), position + 1))
}

/// A parsed VNN-LIB formula that may contain non-linear arithmetic.
#[derive(Debug, Clone)]
pub struct NonLinearFormula {
    decls: Vec<Decl>,
    asserts: Vec<Expr>,
    num_inputs: usize,
    num_outputs: usize,
}

impl NonLinearFormula {
    /// Number of scalar input coordinates.
    #[must_use]
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Number of scalar output coordinates.
    #[must_use]
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// True when at least one assertion uses non-affine arithmetic, i.e. this
    /// file is one the linear parser would refuse.
    #[must_use]
    pub fn is_nonlinear(&self) -> bool {
        self.asserts.iter().any(expr_is_nonlinear)
    }

    /// Parse a VNN-LIB 2.0 source without rejecting non-linear arithmetic.
    pub fn parse(content: &str) -> Result<Self> {
        let cleaned = strip_vnnlib_comments(content);
        let tokens = tokenize(&cleaned)?;
        let exprs = parse_formula_expressions(&tokens)?;

        let mut decls: Vec<Decl> = Vec::new();
        let mut asserts: Vec<Expr> = Vec::new();
        for expr in &exprs {
            let Expr::List(items) = expr else { continue };
            let Some(Expr::Symbol(head)) = items.first() else {
                continue;
            };
            match head.as_str() {
                "declare-network" => collect_network_decls(items, &mut decls)?,
                "assert" => {
                    if items.len() != 2 {
                        return Err(NyError::InvalidSpec(
                            "non-linear parser: assert requires exactly one expression".into(),
                        ));
                    }
                    asserts.push(items[1].clone());
                }
                _ => {}
            }
        }
        if decls.is_empty() {
            return Err(NyError::InvalidSpec(
                "non-linear parser: no (declare-network ...) block found".into(),
            ));
        }
        if asserts.is_empty() {
            return Err(NyError::InvalidSpec(
                "non-linear parser: no assertions found".into(),
            ));
        }
        let numel = |k: VarKind| -> Result<usize> {
            decls
                .iter()
                .filter(|d| d.kind == k)
                .try_fold(0usize, |total, declaration| {
                    let elements = declaration
                        .shape
                        .iter()
                        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension));
                    total
                        .checked_add(elements.ok_or_else(|| {
                            NyError::InvalidSpec(
                                "non-linear tensor declaration size overflows usize".into(),
                            )
                        })?)
                        .ok_or_else(|| {
                            NyError::InvalidSpec(
                                "non-linear tensor declaration count overflows usize".into(),
                            )
                        })
                })
        };
        let num_inputs = numel(VarKind::Input)?;
        let num_outputs = numel(VarKind::Output)?;
        Ok(Self {
            decls,
            asserts,
            num_inputs,
            num_outputs,
        })
    }

    /// Interval hull of the input box implied by the purely-interval
    /// constraints. **A SUPERSET of the true region** — see the module doc.
    ///
    /// Returns `None` when some coordinate is left unbounded, since an unbounded
    /// search domain is not something to silently invent a default for.
    #[must_use]
    pub fn input_box(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        if !has_f64_interval_proof_environment() {
            return None;
        }
        let mut lo = vec![f64::NEG_INFINITY; self.num_inputs];
        let mut hi = vec![f64::INFINITY; self.num_inputs];
        for a in &self.asserts {
            self.tighten_from(a, &mut lo, &mut hi);
        }
        if lo.iter().chain(hi.iter()).any(|v| !v.is_finite()) {
            return None;
        }
        for (l, u) in lo.iter().zip(hi.iter()) {
            if l > u {
                return None;
            }
        }
        Some((lo, hi))
    }

    /// Collect simple `X[i] <= c` / `X[i] >= c` facts, descending only through
    /// `and` (an `or` says nothing about any single branch).
    fn tighten_from(&self, e: &Expr, lo: &mut [f64], hi: &mut [f64]) {
        let Expr::List(items) = e else { return };
        let Some(Expr::Symbol(op)) = items.first() else {
            return;
        };
        match op.as_str() {
            "and" => {
                for sub in &items[1..] {
                    self.tighten_from(sub, lo, hi);
                }
            }
            "<=" | "<" | ">=" | ">" | "=" | "==" => {
                if items.len() != 3 {
                    return;
                }
                let (a, b) = (&items[1], &items[2]);
                // var OP const
                if let (Some((VarKind::Input, i)), Expr::Number(c)) = (self.var_of(a), b) {
                    match op.as_str() {
                        "<=" | "<" => hi[i] = hi[i].min(c.upper),
                        ">=" | ">" => lo[i] = lo[i].max(c.lower),
                        _ => {
                            lo[i] = lo[i].max(c.lower);
                            hi[i] = hi[i].min(c.upper);
                        }
                    }
                }
                // const OP var
                if let (Expr::Number(c), Some((VarKind::Input, i))) = (a, self.var_of(b)) {
                    match op.as_str() {
                        "<=" | "<" => lo[i] = lo[i].max(c.lower),
                        ">=" | ">" => hi[i] = hi[i].min(c.upper),
                        _ => {
                            lo[i] = lo[i].max(c.lower);
                            hi[i] = hi[i].min(c.upper);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Resolve `Name[i,j,...]` to `(kind, flat_index)`.
    ///
    /// The index is into the CONCATENATION of every declaration of that kind, in
    /// declaration order — which is how `holds_at`/`holds_over_box` receive their
    /// `x` and `y` slices. Without the per-declaration offset a spec declaring
    /// two input tensors would alias `X_g[0]` onto `X_f[0]` and silently evaluate
    /// the wrong variable, which can produce a wrong verdict in either direction.
    fn var_of(&self, e: &Expr) -> Option<(VarKind, usize)> {
        let Expr::Symbol(sym) = e else { return None };
        let open = sym.find('[')?;
        let name = &sym[..open];
        let idx_part = sym[open..].strip_prefix('[')?.strip_suffix(']')?;
        let decl = self.decls.iter().find(|d| d.name == name)?;
        // Offset past every earlier declaration of the SAME kind.
        let mut offset = 0usize;
        for d in &self.decls {
            if d.name == name {
                break;
            }
            if d.kind == decl.kind {
                let elements = d
                    .shape
                    .iter()
                    .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))?;
                offset = offset.checked_add(elements)?;
            }
        }
        let indices: Vec<usize> = idx_part
            .split(',')
            .map(|p| p.trim().parse::<usize>().ok())
            .collect::<Option<_>>()?;
        if indices.len() != decl.shape.len() {
            return None;
        }
        let mut flat = 0usize;
        for (&idx, &dim) in indices.iter().zip(decl.shape.iter()) {
            if idx >= dim {
                return None;
            }
            flat = flat.checked_mul(dim)?.checked_add(idx)?;
        }
        Some((decl.kind, offset.checked_add(flat)?))
    }

    /// Evaluate the FULL conjunction of assertions at a concrete point.
    ///
    /// Every finite `f64` coordinate is converted to its exact dyadic rational
    /// and every source decimal retains its exact SMT-LIB rational. Therefore
    /// `Some(true)` is a definite truth, not a nearest-`f64` guess at a boundary.
    ///
    /// `None` means "this formula uses something this evaluator does not
    /// implement" — never a silent `false`, so an unsupported construct can
    /// never be mistaken for a refuted property.
    #[must_use]
    pub fn holds_at(&self, x: &[f64], y: &[f64]) -> Option<bool> {
        if x.len() != self.num_inputs || y.len() != self.num_outputs {
            return None;
        }
        for a in &self.asserts {
            if !self.eval_bool_exact(a, x, y)? {
                return Some(false);
            }
        }
        Some(true)
    }

    /// Three-valued evaluation of the FULL conjunction over a BOX of inputs and
    /// a sound enclosure of the outputs (`#nonlinear-vnnlib` unsat half).
    ///
    /// `Some(Tri::False)` means: NO point in this box can satisfy the formula,
    /// i.e. no counterexample lives here. That is the fact an `unsat` proof is
    /// built from, and it is sound because every arithmetic step widens OUTWARD
    /// and every comparison only commits when the intervals are disjoint.
    ///
    /// `Some(Tri::True)` means every point in the box satisfies it, while
    /// `Some(Tri::Unknown)` is undetermined at this resolution (split further).
    /// `None` means malformed dimensions, an unsupported construct, or a
    /// floating-point environment that cannot support sound outward arithmetic,
    /// so none of those cases can be mistaken for a refutation.
    #[must_use]
    pub fn holds_over_box(&self, xl: &[f64], xu: &[f64], yl: &[f64], yu: &[f64]) -> Option<Tri> {
        if !has_f64_interval_proof_environment() {
            return None;
        }
        if xl.len() != self.num_inputs
            || xu.len() != self.num_inputs
            || yl.len() != self.num_outputs
            || yu.len() != self.num_outputs
        {
            return None;
        }
        let mut all_true = true;
        for a in &self.asserts {
            match self.eval_bool_iv(a, xl, xu, yl, yu)? {
                Tri::False => return Some(Tri::False),
                Tri::Unknown => all_true = false,
                Tri::True => {}
            }
        }
        Some(if all_true { Tri::True } else { Tri::Unknown })
    }

    fn eval_bool_iv(
        &self,
        e: &Expr,
        xl: &[f64],
        xu: &[f64],
        yl: &[f64],
        yu: &[f64],
    ) -> Option<Tri> {
        let Expr::List(items) = e else { return None };
        let Some(Expr::Symbol(op)) = items.first() else {
            return None;
        };
        let args = &items[1..];
        match op.as_str() {
            "and" => {
                let mut all_true = true;
                for a in args {
                    match self.eval_bool_iv(a, xl, xu, yl, yu)? {
                        Tri::False => return Some(Tri::False),
                        Tri::Unknown => all_true = false,
                        Tri::True => {}
                    }
                }
                Some(if all_true { Tri::True } else { Tri::Unknown })
            }
            "or" => {
                let mut all_false = true;
                for a in args {
                    match self.eval_bool_iv(a, xl, xu, yl, yu)? {
                        Tri::True => return Some(Tri::True),
                        Tri::Unknown => all_false = false,
                        Tri::False => {}
                    }
                }
                Some(if all_false { Tri::False } else { Tri::Unknown })
            }
            "not" => {
                let [a] = args else { return None };
                Some(match self.eval_bool_iv(a, xl, xu, yl, yu)? {
                    Tri::True => Tri::False,
                    Tri::False => Tri::True,
                    Tri::Unknown => Tri::Unknown,
                })
            }
            "<=" | "<" | ">=" | ">" | "=" | "==" | "!=" => {
                if args.len() != 2 {
                    return None;
                }
                let (al, au) = self.eval_num_iv(&args[0], xl, xu, yl, yu)?;
                let (bl, bu) = self.eval_num_iv(&args[1], xl, xu, yl, yu)?;
                if [al, au, bl, bu].iter().any(|v| !v.is_finite()) {
                    return None;
                }
                // Commit only when the intervals settle the comparison for EVERY
                // pair in them; otherwise report Unknown and let the caller split.
                Some(match op.as_str() {
                    "<=" => tri(au <= bl, al > bu),
                    "<" => tri(au < bl, al >= bu),
                    ">=" => tri(al >= bu, au < bl),
                    ">" => tri(al > bu, au <= bl),
                    // Equality can only be CERTAIN on two degenerate, equal points.
                    "=" | "==" => tri(al == au && bl == bu && al == bl, au < bl || al > bu),
                    "!=" => tri(au < bl || al > bu, al == au && bl == bu && al == bl),
                    _ => return None,
                })
            }
            _ => None,
        }
    }

    /// Outward-rounded interval evaluation of a numeric term.
    fn eval_num_iv(
        &self,
        e: &Expr,
        xl: &[f64],
        xu: &[f64],
        yl: &[f64],
        yu: &[f64],
    ) -> Option<(f64, f64)> {
        match e {
            Expr::Number(value) => Some((value.lower, value.upper)),
            Expr::Symbol(_) => {
                let (kind, i) = self.var_of(e)?;
                let interval = match kind {
                    VarKind::Input => Some((*xl.get(i)?, *xu.get(i)?)),
                    VarKind::Output => Some((*yl.get(i)?, *yu.get(i)?)),
                }?;
                (interval.0.is_finite() && interval.1.is_finite() && interval.0 <= interval.1)
                    .then_some(interval)
            }
            Expr::List(items) => {
                let Some(Expr::Symbol(op)) = items.first() else {
                    return None;
                };
                let args = &items[1..];
                if args.is_empty() {
                    return None;
                }
                match op.as_str() {
                    "+" => args.iter().try_fold((0.0, 0.0), |(l, u), a| {
                        let (al, au) = self.eval_num_iv(a, xl, xu, yl, yu)?;
                        out((l + al, u + au))
                    }),
                    "-" => {
                        let (fl, fu) = self.eval_num_iv(&args[0], xl, xu, yl, yu)?;
                        if args.len() == 1 {
                            return out((-fu, -fl));
                        }
                        args[1..].iter().try_fold((fl, fu), |(l, u), a| {
                            let (al, au) = self.eval_num_iv(a, xl, xu, yl, yu)?;
                            out((l - au, u - al))
                        })
                    }
                    "*" => args.iter().try_fold((1.0, 1.0), |(l, u), a| {
                        let (al, au) = self.eval_num_iv(a, xl, xu, yl, yu)?;
                        // The extremes of a product of intervals are attained at
                        // the corners.
                        let c = [l * al, l * au, u * al, u * au];
                        if c.iter().any(|value| !value.is_finite()) {
                            return None;
                        }
                        let lo = c.iter().copied().fold(f64::INFINITY, f64::min);
                        let hi = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        out((lo, hi))
                    }),
                    "/" => {
                        let first = self.eval_num_iv(&args[0], xl, xu, yl, yu)?;
                        args[1..].iter().try_fold(first, |(l, u), a| {
                            let (al, au) = self.eval_num_iv(a, xl, xu, yl, yu)?;
                            // A divisor interval straddling zero makes the result
                            // unbounded; refuse rather than invent a bound.
                            if al <= 0.0 && au >= 0.0 {
                                return None;
                            }
                            let c = [l / al, l / au, u / al, u / au];
                            if c.iter().any(|value| !value.is_finite()) {
                                return None;
                            }
                            let lo = c.iter().copied().fold(f64::INFINITY, f64::min);
                            let hi = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            out((lo, hi))
                        })
                    }
                    _ => None,
                }
            }
        }
    }

    fn eval_bool_exact(&self, e: &Expr, x: &[f64], y: &[f64]) -> Option<bool> {
        let Expr::List(items) = e else { return None };
        let Some(Expr::Symbol(op)) = items.first() else {
            return None;
        };
        let args = &items[1..];
        match op.as_str() {
            "and" => {
                for a in args {
                    if !self.eval_bool_exact(a, x, y)? {
                        return Some(false);
                    }
                }
                Some(true)
            }
            "or" => {
                for a in args {
                    if self.eval_bool_exact(a, x, y)? {
                        return Some(true);
                    }
                }
                Some(false)
            }
            "not" => {
                let [a] = args else { return None };
                Some(!self.eval_bool_exact(a, x, y)?)
            }
            "<=" | "<" | ">=" | ">" | "=" | "==" | "!=" => {
                if args.len() != 2 {
                    return None;
                }
                let l = self.eval_num_exact(&args[0], x, y)?;
                let r = self.eval_num_exact(&args[1], x, y)?;
                Some(match op.as_str() {
                    "<=" => l <= r,
                    "<" => l < r,
                    ">=" => l >= r,
                    ">" => l > r,
                    "=" | "==" => l == r,
                    _ => l != r,
                })
            }
            _ => None,
        }
    }

    fn eval_num_exact(&self, e: &Expr, x: &[f64], y: &[f64]) -> Option<BigRational> {
        match e {
            Expr::Number(value) => Some(value.exact.clone()),
            Expr::Symbol(_) => {
                let (kind, i) = self.var_of(e)?;
                let value = match kind {
                    VarKind::Input => x.get(i).copied(),
                    VarKind::Output => y.get(i).copied(),
                }?;
                BigRational::from_float(value)
            }
            Expr::List(items) => {
                let Some(Expr::Symbol(op)) = items.first() else {
                    return None;
                };
                let args = &items[1..];
                if args.is_empty() {
                    return None;
                }
                match op.as_str() {
                    "+" => args.iter().try_fold(BigRational::zero(), |acc, a| {
                        Some(acc + self.eval_num_exact(a, x, y)?)
                    }),
                    "*" => args
                        .iter()
                        .try_fold(BigRational::from_integer(1.into()), |acc, a| {
                            Some(acc * self.eval_num_exact(a, x, y)?)
                        }),
                    "-" => {
                        let first = self.eval_num_exact(&args[0], x, y)?;
                        if args.len() == 1 {
                            return Some(-first);
                        }
                        args[1..]
                            .iter()
                            .try_fold(first, |acc, a| Some(acc - self.eval_num_exact(a, x, y)?))
                    }
                    "/" => {
                        let first = self.eval_num_exact(&args[0], x, y)?;
                        args[1..].iter().try_fold(first, |acc, a| {
                            let divisor = self.eval_num_exact(a, x, y)?;
                            if divisor.is_zero() {
                                return None;
                            }
                            Some(acc / divisor)
                        })
                    }
                    _ => None,
                }
            }
        }
    }
}

/// True when `e` multiplies or divides by anything that is not a literal, i.e.
/// the affine parser would refuse it.
fn expr_is_nonlinear(e: &Expr) -> bool {
    let Expr::List(items) = e else { return false };
    let Some(Expr::Symbol(op)) = items.first() else {
        return items.iter().any(expr_is_nonlinear);
    };
    let args = &items[1..];
    if op == "*" {
        let non_literal = args
            .iter()
            .filter(|argument| !matches!(argument, Expr::Number(_)))
            .count();
        if non_literal > 1 {
            return true;
        }
    }
    if op == "/"
        && args.get(1..).is_some_and(|denominators| {
            denominators
                .iter()
                .any(|argument| !matches!(argument, Expr::Number(_)))
        })
    {
        return true;
    }
    args.iter().any(expr_is_nonlinear)
}

fn collect_network_decls(items: &[Expr], out: &mut Vec<Decl>) -> Result<()> {
    if !matches!(items.get(1), Some(Expr::Symbol(_))) {
        return Err(NyError::InvalidSpec(
            "non-linear parser: declare-network requires a network name".into(),
        ));
    }
    for item in &items[1..] {
        let Expr::List(inner) = item else { continue };
        let Some(Expr::Symbol(kw)) = inner.first() else {
            continue;
        };
        let kind = match kw.as_str() {
            "declare-input" => VarKind::Input,
            "declare-output" => VarKind::Output,
            _ => continue,
        };
        let Some(Expr::Symbol(name)) = inner.get(1) else {
            return Err(NyError::InvalidSpec(format!(
                "non-linear parser: {kw} requires a tensor name"
            )));
        };
        if out.iter().any(|declaration| declaration.name == *name) {
            return Err(NyError::InvalidSpec(format!(
                "non-linear parser: duplicate tensor declaration '{name}'"
            )));
        }
        let shape = parse_decl_shape(&inner[2..], kw)?;
        out.push(Decl {
            name: name.clone(),
            kind,
            shape,
        });
    }
    Ok(())
}

fn parse_decl_shape(items: &[Expr], declaration: &str) -> Result<Vec<usize>> {
    let mut bracket_tokens = Vec::new();
    let mut in_brackets = false;
    for item in items {
        match item {
            Expr::Symbol(token) => {
                if !in_brackets && token.contains('[') {
                    in_brackets = true;
                }
                if in_brackets {
                    bracket_tokens.push(token.clone());
                    if token.contains(']') {
                        let shape = parse_shape_string(&bracket_tokens.join(" "))?;
                        if shape.contains(&0) {
                            return Err(NyError::InvalidSpec(
                                "non-linear tensor dimensions must be non-zero".into(),
                            ));
                        }
                        return Ok(shape);
                    }
                }
            }
            Expr::Number(number) if in_brackets => {
                if !number.exact.is_integer() {
                    return Err(NyError::InvalidSpec(format!(
                        "non-linear {declaration} shape contains a fractional dimension"
                    )));
                }
                bracket_tokens.push(number.exact.to_integer().to_string());
            }
            Expr::List(dimensions) if !in_brackets => {
                let mut shape = Vec::with_capacity(dimensions.len());
                for dimension in dimensions {
                    let Expr::Number(number) = dimension else {
                        return Err(NyError::InvalidSpec(format!(
                            "non-linear {declaration} shape contains a non-numeric dimension"
                        )));
                    };
                    let value = number
                        .exact
                        .to_usize()
                        .filter(|&value| value > 0 && number.exact.is_integer());
                    shape.push(value.ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "non-linear {declaration} shape contains an invalid dimension"
                        ))
                    })?);
                }
                return Ok(shape);
            }
            _ => {}
        }
    }
    if in_brackets {
        return Err(NyError::InvalidSpec(format!(
            "non-linear {declaration} has an unterminated tensor shape"
        )));
    }
    Err(NyError::InvalidSpec(format!(
        "non-linear {declaration} is missing a tensor shape"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ACC_SHAPED: &str = r#"
(vnnlib-version <2.0>)
(declare-network f
    (declare-input X real [1,2])
    (declare-output Y real [1,1])
)
(assert (and (>= X[0,0] 20.0) (<= X[0,0] 40.0)))
(assert (and (>= X[0,1] -40.0) (<= X[0,1] 0.0)))
(assert (>= (* X[0,0] 200.0) (* X[0,1] X[0,1])))
(assert (or (< Y[0,0] -100.001) (> Y[0,0] 100.001)))
"#;

    #[test]
    fn interval_proof_environment_is_qualified() {
        assert!(
            has_f64_interval_proof_environment(),
            "test host must use round-to-nearest and preserve binary64 subnormal operands/results"
        );
    }

    #[test]
    fn parses_a_file_the_affine_parser_refuses() {
        let f = NonLinearFormula::parse(ACC_SHAPED).expect("parse");
        assert_eq!(f.num_inputs(), 2);
        assert_eq!(f.num_outputs(), 1);
        assert!(f.is_nonlinear(), "X[0,1]*X[0,1] must be detected");
    }

    #[test]
    fn variable_denominator_is_nonlinear_even_with_literal_numerator() {
        let src = r#"
(declare-network f
    (declare-input X real [1,1])
    (declare-output Y real [1,1])
)
(assert (>= (/ 1.0 X[0,0]) Y[0,0]))
"#;
        let formula = NonLinearFormula::parse(src).expect("parse");
        assert!(
            formula.is_nonlinear(),
            "a variable denominator cannot enter the affine pipeline"
        );
    }

    #[test]
    fn input_box_is_the_interval_hull_and_ignores_the_parabola() {
        let f = NonLinearFormula::parse(ACC_SHAPED).expect("parse");
        let (lo, hi) = f.input_box().expect("box");
        assert_eq!(lo, vec![20.0, -40.0]);
        assert_eq!(hi, vec![40.0, 0.0]);
    }

    #[test]
    fn holds_at_enforces_the_nonlinear_input_constraint_exactly() {
        let f = NonLinearFormula::parse(ACC_SHAPED).expect("parse");
        // Inside the box AND inside the parabola (200*30 = 6000 >= (-10)^2),
        // with an output that violates the property.
        assert_eq!(f.holds_at(&[30.0, -10.0], &[-200.0]), Some(true));
        // Same point, but an output that does NOT violate the property: the
        // output clause is what rejects it.
        assert_eq!(f.holds_at(&[30.0, -10.0], &[0.0]), Some(false));
    }

    #[test]
    fn a_point_outside_the_declared_box_is_still_evaluated_exactly() {
        // The searcher may propose a point from the SUPERSET hull; `holds_at`
        // is what rejects it. X[0,0]=10 violates `>= 20`.
        let f = NonLinearFormula::parse(ACC_SHAPED).expect("parse");
        assert_eq!(f.holds_at(&[10.0, -10.0], &[-200.0]), Some(false));
    }

    #[test]
    fn unsupported_construct_returns_none_not_false() {
        // `xor` is not implemented; it must not be silently treated as refuted.
        let src = ACC_SHAPED.replace("(or (< Y[0,0] -100.001)", "(xor (< Y[0,0] -100.001)");
        let f = NonLinearFormula::parse(&src).expect("parse");
        assert_eq!(f.holds_at(&[30.0, -10.0], &[-200.0]), None);
    }

    #[test]
    fn multiple_declarations_of_the_same_kind_do_not_alias() {
        // Regression: var_of returned the index WITHIN a declaration, so with two
        // declared input tensors `X_g[0]` aliased onto `X_f[0]` and the evaluator
        // silently read the wrong variable — a wrong verdict in either direction.
        let src = r#"
(declare-network f
    (declare-input X_f real [1,2])
    (declare-output Y_f real [1,1])
)
(declare-network g
    (declare-input X_g real [1,2])
    (declare-output Y_g real [1,1])
)
(assert (>= X_f[0,0] 100.0))
(assert (<= X_g[0,0] 1.0))
(assert (>= Y_g[0,0] 7.0))
"#;
        let f = NonLinearFormula::parse(src).expect("parse");
        assert_eq!(f.num_inputs(), 4, "2 tensors x 2 elements");
        assert_eq!(f.num_outputs(), 2);
        // x = [X_f[0,0], X_f[0,1], X_g[0,0], X_g[0,1]], y = [Y_f, Y_g].
        // X_f[0,0]=200 >= 100 ok; X_g[0,0]=0 <= 1 ok; Y_g=9 >= 7 ok.
        assert_eq!(
            f.holds_at(&[200.0, 0.0, 0.0, 0.0], &[0.0, 9.0]),
            Some(true),
            "second-tensor coordinates must resolve past the first tensor"
        );
        // Now make ONLY X_g[0,0] violate: if it aliased X_f[0,0] this would still pass.
        assert_eq!(
            f.holds_at(&[200.0, 0.0, 5.0, 0.0], &[0.0, 9.0]),
            Some(false)
        );
        // And only Y_g violate, proving outputs offset independently of inputs.
        assert_eq!(
            f.holds_at(&[200.0, 0.0, 0.0, 0.0], &[9.0, 1.0]),
            Some(false)
        );
    }

    #[test]
    fn division_by_zero_refuses_instead_of_deciding_a_comparison() {
        let src = r#"
(declare-network f
    (declare-input X real [1,1])
    (declare-output Y real [1,1])
)
(assert (>= (/ Y[0,0] X[0,0]) 1.0))
"#;
        let f = NonLinearFormula::parse(src).expect("parse");
        assert_eq!(f.holds_at(&[0.0], &[5.0]), None);
        assert_eq!(f.holds_at(&[2.0], &[5.0]), Some(true));
    }

    #[test]
    fn source_decimal_is_not_replaced_by_its_nearest_f64() {
        let src = r#"
(declare-network f
    (declare-input X real [1,1])
    (declare-output Y real [1,1])
)
(assert (>= (* X[0,0] X[0,0]) 0.0))
(assert (<= Y[0,0] 0.1))
"#;
        let formula = NonLinearFormula::parse(src).expect("parse");
        let nearest = 0.1_f64;
        // nearest-f64 0.1 is strictly ABOVE exact decimal 0.1. The old
        // evaluator rounded the source literal first and accepted equality,
        // creating a false SAT witness.
        assert_eq!(formula.holds_at(&[0.0], &[nearest]), Some(false));
        assert_eq!(
            formula.holds_over_box(&[0.0], &[0.0], &[nearest], &[nearest]),
            Some(Tri::Unknown),
            "an outward literal enclosure must not claim this boundary true"
        );

        let strict = src.replace("(<= Y[0,0] 0.1)", "(> Y[0,0] 0.1)");
        let strict = NonLinearFormula::parse(&strict).expect("parse");
        assert_eq!(
            strict.holds_at(&[0.0], &[nearest]),
            Some(true),
            "exact rational evaluation should retain definite boundary facts"
        );
    }

    #[test]
    fn each_rounded_equal_decimal_keeps_its_own_exact_value() {
        let src = r#"
(declare-network f
    (declare-input X real [1,1])
    (declare-output Y real [1,1])
)
(assert (>= (* X[0,0] X[0,0]) 0.0))
(assert (and
    (>= Y[0,0] 0.1)
    (<= Y[0,0] 0.100000000000000001)
))
"#;
        assert_eq!(
            0.1_f64,
            "0.100000000000000001"
                .parse::<f64>()
                .expect("finite decimal"),
            "regression requires two distinct decimals with one f64 image"
        );
        let formula = NonLinearFormula::parse(src).expect("parse");
        assert_eq!(
            formula.holds_at(&[0.0], &[0.1]),
            Some(false),
            "literal identity cannot be keyed by its rounded f64 value"
        );
    }

    #[test]
    fn input_box_rounds_exact_decimal_endpoints_outward() {
        let src = r#"
(declare-network f
    (declare-input X real [1,1])
    (declare-output Y real [1,1])
)
(assert (and (>= X[0,0] 0.1) (<= X[0,0] 0.1)))
(assert (>= (* X[0,0] X[0,0]) 0.0))
(assert (>= Y[0,0] 0.0))
"#;
        let formula = NonLinearFormula::parse(src).expect("parse");
        let (lower, upper) = formula.input_box().expect("bounded input");
        assert_eq!(lower, vec![0.1_f64.next_down()]);
        assert_eq!(upper, vec![0.1_f64]);
    }

    #[test]
    fn non_finite_literal_enclosure_is_rejected() {
        let src = r#"
(declare-network f
    (declare-input X real [1,1])
    (declare-output Y real [1,1])
)
(assert (>= (* X[0,0] X[0,0]) 0.0))
(assert (>= Y[0,0] 1e400))
"#;
        let error = NonLinearFormula::parse(src).expect_err("overflowing literal");
        assert!(error
            .to_string()
            .contains("has no finite outward f64 enclosure"));
    }

    #[test]
    fn malformed_assertion_fails_closed() {
        let src = r#"
(declare-network f
    (declare-input X real [1,1])
    (declare-output Y real [1,1])
)
(assert (>= (* X[0,0] X[0,0]) 0.0) (>= Y[0,0] 0.0))
"#;
        assert!(NonLinearFormula::parse(src).is_err());
    }
}
