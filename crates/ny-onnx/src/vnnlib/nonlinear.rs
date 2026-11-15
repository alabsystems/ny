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
//!
//! # Strict atoms and the pivot case-split (`#acc-strict-intervals`)
//!
//! [`NonLinearFormula::holds_over_box`] works on CLOSED intervals, so it can
//! never refute a cell whose falsity margin is exactly zero somewhere — an
//! atom like `(!= Y 0)` against `Y ∈ [-ε, ε]`, or `(<= (* 2 X Y) (* Z Z))` at
//! `Y → 0⁺, Z = 0`. That is the measured residual on
//! `adaptive_cruise_control_non_linear_2026` rows 15 and 38: the formula is
//! false for EVERY real point, but only because of strict inequalities, so
//! bisection never terminates.
//!
//! [`NonLinearFormula::holds_over_box_split`] closes that gap with two exact
//! facts of real arithmetic and nothing else:
//!
//! 1. **An exact partition.** Every variable whose cell meets a constant it
//!    is compared against in a single-variable affine atom (a *pivot*) is cut
//!    into `[lo, c)`, `{c}` and `(c, hi]`. The pieces cover every real of the
//!    cell (see `partition_at_pivots`), so refuting every piece refutes the
//!    cell. No epsilon is involved — unlike a δ-satisfiability encoding, no
//!    point with `0 < |Y| < ε` is dropped.
//! 2. **Open endpoints.** Each piece carries per-endpoint strictness
//!    (`OpenIv`). Only a piece boundary sets it; arithmetic propagates it
//!    only where a real-arithmetic identity says so (negation, addition, and
//!    the sign of a product or quotient of strictly signed factors).
//!
//! The network enclosure stays the caller's closed, sound per-cell box: an
//! output piece only RESTRICTS it, and every real point `(x, f(x))` of the
//! cell lies in exactly the product piece that contains it.

use std::sync::OnceLock;

use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};
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

/// Upper bound on the product-piece combinations ONE
/// [`NonLinearFormula::holds_over_box_split`] call may evaluate. ACC needs at
/// most 3 (Y) x 3 (X1) x 3 (X0) = 27 on a straddling cell; 729 = 3^6 leaves
/// room for up to six split variables before the split declines (a decline is
/// `Unknown`, never a refutation).
pub const MAX_CASE_SPLIT_PIECES: usize = 729;

/// A variable with more distinct pivot constants than this is not split at
/// all. Dropping pivots is always sound (any pivot set yields an exact cover);
/// it only bounds the per-cell piece count on formulas with many literals.
const MAX_PIVOTS_PER_VARIABLE: usize = 32;

/// Pivot collection stops after this many comparison atoms. Again, fewer
/// pivots can only cost refutations, never produce a wrong one.
const MAX_PIVOT_ATOMS: usize = 1 << 16;

/// A set of reals `{ v : lo ⊴ v ⊴ hi }` where `⊴` is `<` at an OPEN endpoint
/// and `<=` at a closed one (`#acc-strict-intervals`).
///
/// Every value this module constructs is finite and NON-EMPTY
/// ([`OpenIv::checked`]); the comparison rules below rely on that.
///
/// Soundness convention: an `OpenIv` produced by arithmetic is an ENCLOSURE —
/// it contains every value the expression takes over the piece. Openness is
/// set only by a piece boundary (a pivot or a strict-sign fact) and is
/// propagated only by real-arithmetic identities; it is never inferred from a
/// rounding artifact. Outward rounding moves a bound strictly away from the
/// true range, so an open flag remains valid after widening.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OpenIv {
    lo: f64,
    hi: f64,
    lo_open: bool,
    hi_open: bool,
}

impl OpenIv {
    fn closed(lo: f64, hi: f64) -> Self {
        Self {
            lo,
            hi,
            lo_open: false,
            hi_open: false,
        }
    }

    /// Finite and non-empty, or refused.
    fn checked(self) -> Option<Self> {
        (self.lo.is_finite() && self.hi.is_finite() && !self.is_empty()).then_some(self)
    }

    fn is_empty(&self) -> bool {
        self.lo > self.hi || (self.lo == self.hi && (self.lo_open || self.hi_open))
    }

    /// A single real (non-empty and degenerate implies both ends closed).
    fn is_point(&self) -> bool {
        self.lo == self.hi && !self.lo_open && !self.hi_open
    }

    fn is_exact_zero(&self) -> bool {
        self.is_point() && self.lo == 0.0
    }

    /// The mutation used by the openness pin: the same bounds, both closed.
    fn closure(self) -> Self {
        Self::closed(self.lo, self.hi)
    }

    /// Every element is `>= 0`.
    fn nonnegative(&self) -> bool {
        self.lo >= 0.0
    }

    /// Every element is `> 0`.
    fn positive(&self) -> bool {
        self.lo > 0.0 || (self.lo == 0.0 && self.lo_open)
    }

    /// Every element is `<= 0`.
    fn nonpositive(&self) -> bool {
        self.hi <= 0.0
    }

    /// Every element is `< 0`.
    fn negative(&self) -> bool {
        self.hi < 0.0 || (self.hi == 0.0 && self.hi_open)
    }

    /// Intersection with the (possibly unbounded) range `lo ⊴ v ⊴ hi`, or
    /// `None` when that intersection is empty.
    fn restrict(self, lo: f64, lo_open: bool, hi: f64, hi_open: bool) -> Option<Self> {
        let (new_lo, new_lo_open) = if lo > self.lo {
            (lo, lo_open)
        } else if lo < self.lo {
            (self.lo, self.lo_open)
        } else {
            (self.lo, self.lo_open || lo_open)
        };
        let (new_hi, new_hi_open) = if hi < self.hi {
            (hi, hi_open)
        } else if hi > self.hi {
            (self.hi, self.hi_open)
        } else {
            (self.hi, self.hi_open || hi_open)
        };
        Self {
            lo: new_lo,
            hi: new_hi,
            lo_open: new_lo_open,
            hi_open: new_hi_open,
        }
        .checked()
    }

    /// `-a` is exact in IEEE arithmetic, so no widening is needed and the
    /// endpoint strictness simply swaps sides.
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
            lo_open: self.hi_open,
            hi_open: self.lo_open,
        }
    }

    /// `x + y > a.lo + b.lo` whenever either lower bound is strict (and
    /// likewise above); the outward widening only moves the bound further out.
    ///
    /// Two further real-arithmetic facts keep a sign that widening would
    /// otherwise blur into a subnormal of the wrong sign: `a + 0 = a` exactly,
    /// and a sum of non-negatives is non-negative — strictly positive when
    /// either summand is (mirror image for non-positives).
    fn add(self, other: Self) -> Option<Self> {
        if other.is_exact_zero() {
            return Some(self);
        }
        if self.is_exact_zero() {
            return Some(other);
        }
        let (lo, hi) = out((self.lo + other.lo, self.hi + other.hi))?;
        let mut result = Self {
            lo,
            hi,
            lo_open: self.lo_open || other.lo_open,
            hi_open: self.hi_open || other.hi_open,
        };
        if self.nonnegative() && other.nonnegative() {
            let strict = self.positive() || other.positive();
            if result.lo < 0.0 {
                result.lo = 0.0;
                result.lo_open = strict;
            } else if result.lo == 0.0 {
                result.lo = 0.0;
                result.lo_open = result.lo_open || strict;
            }
        }
        if self.nonpositive() && other.nonpositive() {
            let strict = self.negative() || other.negative();
            if result.hi > 0.0 {
                result.hi = 0.0;
                result.hi_open = strict;
            } else if result.hi == 0.0 {
                result.hi = 0.0;
                result.hi_open = result.hi_open || strict;
            }
        }
        result.checked()
    }

    fn sub(self, other: Self) -> Option<Self> {
        self.add(other.neg())
    }

    /// Today's closed corner hull, outward-widened, then refined ONLY by the
    /// sign of the factors.
    fn mul(self, other: Self) -> Option<Self> {
        // `0 * b = 0` exactly for every finite real `b`.
        if self.is_exact_zero() || other.is_exact_zero() {
            return Some(Self::closed(0.0, 0.0));
        }
        let corners = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        if corners.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let lo = corners.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (lo, hi) = out((lo, hi))?;
        Self::closed(lo, hi).refined_by_sign(self, other)
    }

    /// Division refuses any divisor that touches zero, exactly like the closed
    /// evaluator (an open-at-zero divisor is refused too), then applies the
    /// same sign refinement: `sign(a / b) = sign(a) * sign(b)` for `b != 0`.
    fn div(self, other: Self) -> Option<Self> {
        if other.lo <= 0.0 && other.hi >= 0.0 {
            return None;
        }
        if self.is_exact_zero() {
            return Some(Self::closed(0.0, 0.0));
        }
        let corners = [
            self.lo / other.lo,
            self.lo / other.hi,
            self.hi / other.lo,
            self.hi / other.hi,
        ];
        if corners.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let lo = corners.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (lo, hi) = out((lo, hi))?;
        Self::closed(lo, hi).refined_by_sign(self, other)
    }

    /// Intersect a product/quotient enclosure with the half-line its factors'
    /// signs force. Both facts are real arithmetic:
    ///
    /// * same-sign factors give a result `>= 0`, and `> 0` when both factors
    ///   are STRICTLY signed;
    /// * opposite-sign factors give a result `<= 0`, and `< 0` when both are
    ///   strictly signed.
    ///
    /// The intersection of two sets that both contain the true range still
    /// contains it, so this never loses a point. An exact-zero factor is
    /// handled by the callers before this is reached.
    fn refined_by_sign(self, a: Self, b: Self) -> Option<Self> {
        let mut result = self;
        if (a.nonnegative() && b.nonnegative()) || (a.nonpositive() && b.nonpositive()) {
            let strict = (a.positive() && b.positive()) || (a.negative() && b.negative());
            if result.lo < 0.0 {
                result.lo = 0.0;
                result.lo_open = strict;
            } else if result.lo == 0.0 {
                result.lo = 0.0;
                result.lo_open = result.lo_open || strict;
            }
        }
        if (a.nonnegative() && b.nonpositive()) || (a.nonpositive() && b.nonnegative()) {
            let strict = (a.positive() && b.negative()) || (a.negative() && b.positive());
            if result.hi > 0.0 {
                result.hi = 0.0;
                result.hi_open = strict;
            } else if result.hi == 0.0 {
                result.hi = 0.0;
                result.hi_open = result.hi_open || strict;
            }
        }
        result.checked()
    }
}

/// Every element of `a` is `<=` every element of `b`.
fn definitely_le(a: OpenIv, b: OpenIv) -> bool {
    a.hi <= b.lo
}

/// Every element of `a` is `<` every element of `b`: either the hulls are
/// disjoint, or they touch at a point that at least one side EXCLUDES.
fn definitely_lt(a: OpenIv, b: OpenIv) -> bool {
    a.hi < b.lo || (a.hi == b.lo && (a.hi_open || b.lo_open))
}

/// Three-valued comparison of two open-endpoint enclosures. It commits only
/// when the comparison holds (or fails) for EVERY pair of elements; the open
/// ends are what let two hulls that merely touch commit — e.g. `Y ∈ (0, ε]`
/// is definitely `> 0` and definitely `!= 0`.
fn compare_open(op: &str, a: OpenIv, b: OpenIv) -> Option<Tri> {
    let equal_points = a.is_point() && b.is_point() && a.lo == b.lo;
    let apart = definitely_lt(a, b) || definitely_lt(b, a);
    Some(match op {
        "<=" => tri(definitely_le(a, b), definitely_lt(b, a)),
        "<" => tri(definitely_lt(a, b), definitely_le(b, a)),
        ">=" => tri(definitely_le(b, a), definitely_lt(a, b)),
        ">" => tri(definitely_lt(b, a), definitely_le(a, b)),
        "=" | "==" => tri(equal_points, apart),
        "!=" => tri(apart, equal_points),
        _ => return None,
    })
}

/// Exact constants a single scalar variable is compared against
/// (`#acc-strict-intervals`), sorted strictly ascending by exact value.
#[derive(Debug, Clone)]
struct PivotSet {
    kind: VarKind,
    index: usize,
    values: Vec<Number>,
}

/// Outcome of [`NonLinearFormula::holds_over_box_split`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaseSplitVerdict {
    /// [`Tri::False`] iff EVERY piece of the exact partition was refuted, i.e.
    /// no point of the cell can satisfy the formula. Otherwise
    /// [`Tri::Unknown`]. Never [`Tri::True`]: the split is a refutation tool.
    pub truth: Tri,
    /// Piece combinations evaluated. `0` means the split DECLINED (more than
    /// [`MAX_CASE_SPLIT_PIECES`] combinations) and evaluated nothing.
    pub pieces: usize,
}

/// Partition `cell` at the sorted exact `pivots` (`#acc-strict-intervals`).
///
/// # Why the pieces cover the cell
///
/// Let `p1 < ... < pk` be the pivots kept (any sorted subsequence works), each
/// with a sound enclosure `lower_i <= p_i <= upper_i`. The ranges
/// `(-inf, upper_1)`, `[lower_i, upper_i]`, `(lower_i, upper_(i+1))` and
/// `(lower_k, +inf)` cover the whole real line: a real `v < p1` is below
/// `upper_1`; `v = p_i` lies in `[lower_i, upper_i]`; `p_i < v < p_(i+1)`
/// gives `lower_i <= p_i < v < p_(i+1) <= upper_(i+1)`; and `v > pk` is above
/// `lower_k`. Intersecting each range with the cell therefore covers the
/// cell. Ranges may overlap when two pivots share an f64 enclosure; overlap
/// only repeats work. Empty intersections are dropped (they hold no point).
fn partition_at_pivots(cell: OpenIv, pivots: &[Number]) -> Vec<OpenIv> {
    let relevant: Vec<&Number> = pivots
        .iter()
        .filter(|pivot| pivot.upper >= cell.lo && pivot.lower <= cell.hi)
        .collect();
    let Some(first) = relevant.first() else {
        return vec![cell];
    };
    let mut pieces = Vec::with_capacity(2 * relevant.len() + 1);
    let mut push = |lo: f64, lo_open: bool, hi: f64, hi_open: bool| {
        if let Some(piece) = cell.restrict(lo, lo_open, hi, hi_open) {
            pieces.push(piece);
        }
    };
    push(f64::NEG_INFINITY, true, first.upper, true);
    for (position, pivot) in relevant.iter().enumerate() {
        push(pivot.lower, false, pivot.upper, false);
        match relevant.get(position + 1) {
            Some(next) => push(pivot.lower, true, next.upper, true),
            None => push(pivot.lower, true, f64::INFINITY, true),
        }
    }
    pieces
}

/// `coefficient * variable + constant`, exactly, for AT MOST one scalar
/// variable. Used only to find pivot constants; a term that is not of this
/// form simply contributes no pivot.
struct SingleVarAffine {
    var: Option<(VarKind, usize)>,
    coefficient: BigRational,
    constant: BigRational,
}

impl SingleVarAffine {
    fn constant(value: BigRational) -> Self {
        Self {
            var: None,
            coefficient: BigRational::zero(),
            constant: value,
        }
    }

    fn normalized(self) -> Self {
        if self.coefficient.is_zero() {
            Self {
                var: None,
                coefficient: self.coefficient,
                constant: self.constant,
            }
        } else {
            self
        }
    }

    fn add(self, other: Self) -> Option<Self> {
        let var = match (self.var, other.var) {
            (Some(left), Some(right)) if left != right => return None,
            (Some(var), _) | (_, Some(var)) => Some(var),
            (None, None) => None,
        };
        Some(
            Self {
                var,
                coefficient: self.coefficient + other.coefficient,
                constant: self.constant + other.constant,
            }
            .normalized(),
        )
    }

    fn neg(self) -> Self {
        Self {
            var: self.var,
            coefficient: -self.coefficient,
            constant: -self.constant,
        }
    }

    fn mul(self, other: Self) -> Option<Self> {
        let (with_var, scalar) = match (self.var.is_some(), other.var.is_some()) {
            (true, true) => return None,
            (true, false) => (self, other.constant),
            (false, _) => (other, self.constant),
        };
        Some(
            Self {
                var: with_var.var,
                coefficient: with_var.coefficient * scalar.clone(),
                constant: with_var.constant * scalar,
            }
            .normalized(),
        )
    }

    fn div(self, other: Self) -> Option<Self> {
        if other.var.is_some() || other.constant.is_zero() {
            return None;
        }
        Some(
            Self {
                var: self.var,
                coefficient: self.coefficient / other.constant.clone(),
                constant: self.constant / other.constant,
            }
            .normalized(),
        )
    }
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
    /// Retained for typed execution profiles; legacy real evaluators are unchanged.
    dtype: String,
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
    /// `#acc-strict-intervals` pivot constants, collected on first use by
    /// [`Self::holds_over_box_split`] so every other caller of [`Self::parse`]
    /// (the sat search, the sat gate) pays nothing for them.
    pivots: OnceLock<Vec<PivotSet>>,
    single_network: bool,
    f32_declaration_profile: bool,
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
        let mut networks = 0usize;
        let mut versions = 0usize;
        let mut f32_declaration_profile = true;
        for expr in &exprs {
            let Expr::List(items) = expr else {
                f32_declaration_profile = false;
                continue;
            };
            let Some(Expr::Symbol(head)) = items.first() else {
                f32_declaration_profile = false;
                continue;
            };
            match head.as_str() {
                "vnnlib-version" => {
                    versions += 1;
                    f32_declaration_profile &= matches!(items.as_slice(),
                        [Expr::Symbol(_), Expr::Symbol(v)] if v == "<2.0>");
                }
                "declare-network" => {
                    networks += 1;
                    f32_declaration_profile &= items.get(2..).is_some_and(|body| body.iter().all(|item| matches!(item,
                        Expr::List(v) if matches!(v.as_slice(), [Expr::Symbol(k),Expr::Symbol(_),Expr::Symbol(_),Expr::Symbol(s)] if (k == "declare-input" || k == "declare-output") && s.starts_with('[') && s.ends_with(']')))));
                    collect_network_decls(items, &mut decls)?;
                }
                "assert" => {
                    if items.len() != 2 {
                        return Err(NyError::InvalidSpec(
                            "non-linear parser: assert requires exactly one expression".into(),
                        ));
                    }
                    asserts.push(items[1].clone());
                }
                _ => f32_declaration_profile = false,
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
            pivots: OnceLock::new(),
            single_network: networks == 1,
            f32_declaration_profile: f32_declaration_profile && versions == 1,
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

    /// `#acc-strict-intervals`: refute a cell that the closed evaluator
    /// ([`Self::holds_over_box`]) leaves [`Tri::Unknown`], by an EXACT case
    /// split of every pivot variable plus open-endpoint interval logic.
    ///
    /// `(xl, xu)` is the input cell and `(yl, yu)` a SOUND closed enclosure of
    /// the network outputs over it — the same arguments `holds_over_box`
    /// takes. Every variable whose cell meets one of its pivot constants `c`
    /// (a constant it is compared against in a single-variable affine atom:
    /// `v op c`, `c op v`, `(- v) op c`, `(* k v) op c`, ...) is cut into
    /// `[lo, c)`, `{c}` and `(c, hi]`; the full conjunction is then evaluated
    /// over every combination of pieces.
    ///
    /// Returns `truth: Tri::False` only when EVERY combination is definitely
    /// false. Because the pieces cover the cell exactly (see
    /// `partition_at_pivots`) and each piece evaluation is a sound
    /// open-endpoint enclosure, that means no real point `(x, y)` with `x` in
    /// the cell and `y` in the enclosure satisfies the formula — in
    /// particular no `(x, f(x))`. Anything else is `Tri::Unknown` (split
    /// further); `None` means malformed dimensions or an unqualified
    /// floating-point environment, exactly as for `holds_over_box`.
    #[must_use]
    pub fn holds_over_box_split(
        &self,
        xl: &[f64],
        xu: &[f64],
        yl: &[f64],
        yu: &[f64],
    ) -> Option<CaseSplitVerdict> {
        self.holds_over_box_split_with(xl, xu, yl, yu, true)
    }

    /// `honor_open = false` is the MUTATION used by the openness pin: the same
    /// partition with every piece replaced by its closure. It must lose the
    /// knife-edge refutations, which proves the open endpoints are what carry
    /// them.
    fn holds_over_box_split_with(
        &self,
        xl: &[f64],
        xu: &[f64],
        yl: &[f64],
        yu: &[f64],
        honor_open: bool,
    ) -> Option<CaseSplitVerdict> {
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
        let closed_cell = |lower: &[f64], upper: &[f64]| -> Option<Vec<OpenIv>> {
            lower
                .iter()
                .zip(upper)
                .map(|(&lo, &hi)| OpenIv::closed(lo, hi).checked())
                .collect()
        };
        let mut xs = closed_cell(xl, xu)?;
        let mut ys = closed_cell(yl, yu)?;

        // One axis per pivoted variable, holding that variable's pieces.
        let mut axes: Vec<(VarKind, usize, Vec<OpenIv>)> = Vec::new();
        for set in self.case_split_pivots() {
            let cell = match set.kind {
                VarKind::Input => *xs.get(set.index)?,
                VarKind::Output => *ys.get(set.index)?,
            };
            let mut pieces = partition_at_pivots(cell, &set.values);
            if !honor_open {
                for piece in &mut pieces {
                    *piece = piece.closure();
                }
            }
            // A partition of a non-empty cell is never empty. If it ever were,
            // the product below would be vacuous and "every piece refuted"
            // would hold of nothing — refuse instead of refuting.
            if pieces.is_empty() {
                return None;
            }
            axes.push((set.kind, set.index, pieces));
        }
        let total = axes
            .iter()
            .try_fold(1usize, |product, (_, _, pieces)| {
                product.checked_mul(pieces.len())
            })
            .filter(|&total| total <= MAX_CASE_SPLIT_PIECES);
        let Some(total) = total else {
            return Some(CaseSplitVerdict {
                truth: Tri::Unknown,
                pieces: 0,
            });
        };

        // Mixed-radix walk over every combination; stop at the first piece
        // that is not definitely false.
        let mut digits = vec![0usize; axes.len()];
        let mut evaluated = 0usize;
        loop {
            for ((kind, index, pieces), &digit) in axes.iter().zip(&digits) {
                let piece = *pieces.get(digit)?;
                match kind {
                    VarKind::Input => *xs.get_mut(*index)? = piece,
                    VarKind::Output => *ys.get_mut(*index)? = piece,
                }
            }
            evaluated += 1;
            if self.holds_over_pieces(&xs, &ys) != Some(Tri::False) {
                return Some(CaseSplitVerdict {
                    truth: Tri::Unknown,
                    pieces: evaluated,
                });
            }
            let mut position = digits.len();
            loop {
                if position == 0 {
                    // Every combination was visited exactly once and refuted.
                    // Re-check the count rather than trust the counter logic:
                    // a refutation must never rest on a skipped piece.
                    let truth = if evaluated == total {
                        Tri::False
                    } else {
                        Tri::Unknown
                    };
                    return Some(CaseSplitVerdict {
                        truth,
                        pieces: evaluated,
                    });
                }
                position -= 1;
                digits[position] += 1;
                if digits[position] < axes[position].2.len() {
                    break;
                }
                digits[position] = 0;
            }
        }
    }

    /// Human-readable pivot sets for the lane's observable, e.g.
    /// `x0@{0,20,40} x1@{0,40} y0@{-100.001,-100,0,100,100.001}` (flat input
    /// index `x<i>`, flat output index `y<j>`). `none` when nothing pivots.
    #[must_use]
    pub fn case_split_pivot_summary(&self) -> String {
        let sets = self.case_split_pivots();
        if sets.is_empty() {
            return "none".to_string();
        }
        sets.iter()
            .map(|set| {
                let prefix = match set.kind {
                    VarKind::Input => 'x',
                    VarKind::Output => 'y',
                };
                let values: Vec<String> = set
                    .values
                    .iter()
                    .map(|value| format!("{}", value.exact.to_f64().unwrap_or(value.lower)))
                    .collect();
                format!("{prefix}{}@{{{}}}", set.index, values.join(","))
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn case_split_pivots(&self) -> &[PivotSet] {
        self.pivots.get_or_init(|| self.collect_pivots())
    }

    /// Walk every assertion for single-variable affine comparison atoms and
    /// record, per variable, the exact constant at which each atom flips.
    /// The choice of pivots affects only WHICH cells the split can refute,
    /// never whether a refutation is sound: any pivot set gives an exact cover.
    fn collect_pivots(&self) -> Vec<PivotSet> {
        let mut found: Vec<((VarKind, usize), BigRational)> = Vec::new();
        for assertion in &self.asserts {
            self.collect_pivot_atoms(assertion, &mut found, 0);
        }
        let kind_rank = |kind: VarKind| match kind {
            VarKind::Input => 0u8,
            VarKind::Output => 1u8,
        };
        found.sort_by(|(left_var, left), (right_var, right)| {
            (kind_rank(left_var.0), left_var.1)
                .cmp(&(kind_rank(right_var.0), right_var.1))
                .then_with(|| left.cmp(right))
        });
        found.dedup_by(|(right_var, right), (left_var, left)| {
            left_var == right_var && left == right
        });

        let mut sets: Vec<PivotSet> = Vec::new();
        for ((kind, index), exact) in found {
            let (Ok(lower), Ok(upper)) = (
                rational_to_lower_f64(&exact, 0),
                rational_to_upper_f64(&exact, 0),
            ) else {
                // No finite enclosure: omit this pivot (always sound).
                continue;
            };
            let number = Number {
                exact,
                lower,
                upper,
            };
            let extends_last = sets
                .last()
                .is_some_and(|set| set.kind == kind && set.index == index);
            if extends_last {
                if let Some(set) = sets.last_mut() {
                    set.values.push(number);
                }
            } else {
                sets.push(PivotSet {
                    kind,
                    index,
                    values: vec![number],
                });
            }
        }
        sets.retain(|set| set.values.len() <= MAX_PIVOTS_PER_VARIABLE);
        sets
    }

    fn collect_pivot_atoms(
        &self,
        e: &Expr,
        found: &mut Vec<((VarKind, usize), BigRational)>,
        depth: usize,
    ) {
        if depth > MAX_EXPRESSION_NESTING || found.len() >= MAX_PIVOT_ATOMS {
            return;
        }
        let Expr::List(items) = e else { return };
        let Some(Expr::Symbol(op)) = items.first() else {
            return;
        };
        let args = &items[1..];
        match op.as_str() {
            "and" | "or" | "not" => {
                for argument in args {
                    self.collect_pivot_atoms(argument, found, depth + 1);
                }
            }
            "<=" | "<" | ">=" | ">" | "=" | "==" | "!=" if args.len() == 2 => {
                let (Some(left), Some(right)) = (
                    self.single_var_affine(&args[0], 0),
                    self.single_var_affine(&args[1], 0),
                ) else {
                    return;
                };
                // left - right = k*v + m, so the atom flips at v = -m / k.
                let Some(difference) = left.add(right.neg()) else {
                    return;
                };
                if let Some(var) = difference.var {
                    if !difference.coefficient.is_zero() {
                        found.push((var, -difference.constant / difference.coefficient));
                    }
                }
            }
            _ => {}
        }
    }

    fn single_var_affine(&self, e: &Expr, depth: usize) -> Option<SingleVarAffine> {
        if depth > MAX_EXPRESSION_NESTING {
            return None;
        }
        match e {
            Expr::Number(value) => Some(SingleVarAffine::constant(value.exact.clone())),
            Expr::Symbol(_) => Some(SingleVarAffine {
                var: Some(self.var_of(e)?),
                coefficient: BigRational::one(),
                constant: BigRational::zero(),
            }),
            Expr::List(items) => {
                let Some(Expr::Symbol(op)) = items.first() else {
                    return None;
                };
                let args = &items[1..];
                let first = self.single_var_affine(args.first()?, depth + 1)?;
                let rest = &args[1..];
                match op.as_str() {
                    "+" => rest.iter().try_fold(first, |sum, argument| {
                        sum.add(self.single_var_affine(argument, depth + 1)?)
                    }),
                    "-" if rest.is_empty() => Some(first.neg()),
                    "-" => rest.iter().try_fold(first, |difference, argument| {
                        difference.add(self.single_var_affine(argument, depth + 1)?.neg())
                    }),
                    "*" => rest.iter().try_fold(first, |product, argument| {
                        product.mul(self.single_var_affine(argument, depth + 1)?)
                    }),
                    "/" => rest.iter().try_fold(first, |quotient, argument| {
                        quotient.div(self.single_var_affine(argument, depth + 1)?)
                    }),
                    _ => None,
                }
            }
        }
    }

    /// Three-valued truth of the FULL conjunction over one product piece,
    /// with open-endpoint semantics. A single definitely-false conjunct
    /// refutes the piece; an unsupported conjunct can only withhold a verdict.
    fn holds_over_pieces(&self, xs: &[OpenIv], ys: &[OpenIv]) -> Option<Tri> {
        let mut all_true = true;
        let mut unsupported = false;
        for assertion in &self.asserts {
            match self.eval_bool_open(assertion, xs, ys) {
                Some(Tri::False) => return Some(Tri::False),
                Some(Tri::Unknown) => all_true = false,
                Some(Tri::True) => {}
                None => unsupported = true,
            }
        }
        if unsupported {
            return None;
        }
        Some(if all_true { Tri::True } else { Tri::Unknown })
    }

    fn eval_bool_open(&self, e: &Expr, xs: &[OpenIv], ys: &[OpenIv]) -> Option<Tri> {
        let Expr::List(items) = e else { return None };
        let Some(Expr::Symbol(op)) = items.first() else {
            return None;
        };
        let args = &items[1..];
        match op.as_str() {
            "and" => {
                let mut all_true = true;
                for a in args {
                    match self.eval_bool_open(a, xs, ys)? {
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
                    match self.eval_bool_open(a, xs, ys)? {
                        Tri::True => return Some(Tri::True),
                        Tri::Unknown => all_false = false,
                        Tri::False => {}
                    }
                }
                Some(if all_false { Tri::False } else { Tri::Unknown })
            }
            "not" => {
                let [a] = args else { return None };
                Some(match self.eval_bool_open(a, xs, ys)? {
                    Tri::True => Tri::False,
                    Tri::False => Tri::True,
                    Tri::Unknown => Tri::Unknown,
                })
            }
            "<=" | "<" | ">=" | ">" | "=" | "==" | "!=" => {
                if args.len() != 2 {
                    return None;
                }
                let a = self.eval_num_open(&args[0], xs, ys)?;
                let b = self.eval_num_open(&args[1], xs, ys)?;
                compare_open(op, a, b)
            }
            _ => None,
        }
    }

    /// Open-endpoint, outward-rounded interval evaluation of a numeric term
    /// over one product piece. An n-ary `+`/`*` is seeded with its FIRST
    /// operand (the identities `0 + a = a`, `1 * a = a`), which is exact.
    fn eval_num_open(&self, e: &Expr, xs: &[OpenIv], ys: &[OpenIv]) -> Option<OpenIv> {
        match e {
            Expr::Number(value) => OpenIv::closed(value.lower, value.upper).checked(),
            Expr::Symbol(_) => {
                let (kind, i) = self.var_of(e)?;
                let interval = match kind {
                    VarKind::Input => xs.get(i).copied(),
                    VarKind::Output => ys.get(i).copied(),
                }?;
                interval.checked()
            }
            Expr::List(items) => {
                let Some(Expr::Symbol(op)) = items.first() else {
                    return None;
                };
                let args = &items[1..];
                let first = self.eval_num_open(args.first()?, xs, ys)?;
                let rest = &args[1..];
                match op.as_str() {
                    "+" => rest
                        .iter()
                        .try_fold(first, |sum, a| sum.add(self.eval_num_open(a, xs, ys)?)),
                    "-" if rest.is_empty() => Some(first.neg()),
                    "-" => rest.iter().try_fold(first, |difference, a| {
                        difference.sub(self.eval_num_open(a, xs, ys)?)
                    }),
                    "*" => rest.iter().try_fold(first, |product, a| {
                        product.mul(self.eval_num_open(a, xs, ys)?)
                    }),
                    "/" => rest.iter().try_fold(first, |quotient, a| {
                        quotient.div(self.eval_num_open(a, xs, ys)?)
                    }),
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
            dtype: match inner.get(2) {
                Some(Expr::Symbol(dtype)) => dtype.to_ascii_lowercase(),
                _ => String::new(),
            },
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

    // ---------------------------------------------------------------------
    // #acc-strict-intervals: open-endpoint logic and the pivot case-split.
    // ---------------------------------------------------------------------

    /// The output assertion shared VERBATIM by all 50
    /// `adaptive_cruise_control_non_linear_2026` instances (every instance's
    /// `(assert (or ...` line is byte-identical).
    const ACC_OUTPUT_ASSERT: &str = "(assert (or (or (< Y[0,0] -100.001) (> Y[0,0] 100.001)) (and (< Y[0,0] 100.0) (or (!= Y[0,0] 0.0) (<= (* (+ X[0,0] (* X[0,1] 0.1)) 200.0) (* X[0,1] X[0,1]))) (or (>= Y[0,0] 100.0) (< Y[0,0] -100.0) (== Y[0,0] 0.0) (<= (* 200.0 (+ X[0,0] (* X[0,1] 0.1) (* Y[0,0] 0.005))) (* (+ X[0,1] (* Y[0,0] 0.1)) (+ X[0,1] (* Y[0,0] 0.1)))) (and (and (or (<= Y[0,0] 0.0) (and (<= (- X[0,1]) (* 0.1 Y[0,0])) (>= (- X[0,1]) 0.0))) (or (>= Y[0,0] 0.0) (and (>= (- X[0,1]) (* 0.1 Y[0,0])) (<= (- X[0,1]) 0.0)))) (and (or (<= Y[0,0] 0.0) (<= (* (* X[0,0] 2.0) Y[0,0]) (* X[0,1] X[0,1]))) (or (>= Y[0,0] 0.0) (>= (* (* X[0,0] 2.0) Y[0,0]) (* X[0,1] X[0,1])))))))))";

    /// An ACC instance with the given input box and the given operator on the
    /// `X[0,0] 0.0` atom (`>` is the benchmark's strict atom; `>=` is the
    /// counterfactual used to show the strict atom is what refutes X0 = 0).
    fn acc_formula(x0: (&str, &str), x1: (&str, &str), x0_atom: &str) -> NonLinearFormula {
        let src = format!(
            "(vnnlib-version <2.0>)\n\
             (declare-network f\n\
             (declare-input X real [1,2])\n\
             (declare-output Y real [1,1])\n\
             )\n\
             (assert (and (>= X[0,0] {}) (<= X[0,0] {})))\n\
             (assert (and (>= X[0,1] {}) (<= X[0,1] {})))\n\
             (assert (and ({} X[0,0] 0.0) (>= (* X[0,0] 200.0) (* X[0,1] X[0,1]))))\n\
             {}\n",
            x0.0, x0.1, x1.0, x1.1, x0_atom, ACC_OUTPUT_ASSERT
        );
        NonLinearFormula::parse(&src).expect("ACC-shaped instance parses")
    }

    /// Instance 38: X0 in [20, 40], X1 in [0, 40]. Field-unsat; the falsity
    /// margin vanishes at (X1 = 0, Y -> 0+).
    fn acc_row38() -> NonLinearFormula {
        acc_formula(("20.0", "40.0"), ("0.0", "40.0"), ">")
    }

    /// Instance 15: X0 in [0, 20], X1 in [0, 40]. Field-unsat only because of
    /// the strict `(> X[0,0] 0.0)` at the corner X0 -> 0+, X1 = 0.
    fn acc_row15() -> NonLinearFormula {
        acc_formula(("0.0", "20.0"), ("0.0", "40.0"), ">")
    }

    /// Deterministic xorshift64 (no `rand`/`proptest` dependency in ny-onnx).
    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % (n as u64)) as usize
        }

        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / 9_007_199_254_740_992.0
        }

        fn chance(&mut self, p: f64) -> bool {
            self.unit() < p
        }
    }

    fn exact(value: f64) -> BigRational {
        BigRational::from_float(value).expect("finite f64")
    }

    /// Exact membership of a rational in an open-endpoint interval.
    fn contains_exact(iv: OpenIv, value: &BigRational) -> bool {
        let (lo, hi) = (exact(iv.lo), exact(iv.hi));
        let above = if iv.lo_open {
            lo < *value
        } else {
            lo <= *value
        };
        let below = if iv.hi_open {
            *value < hi
        } else {
            *value <= hi
        };
        above && below
    }

    /// The f64 members of `iv` worth testing: closed endpoints, the first
    /// representable values inside each end, the midpoint and zero.
    fn members(iv: OpenIv) -> Vec<f64> {
        let mut out = Vec::new();
        for v in [
            iv.lo,
            iv.hi,
            iv.lo.next_up(),
            iv.hi.next_down(),
            iv.lo.midpoint(iv.hi),
            0.0,
        ] {
            if contains_exact(iv, &exact(v)) && !out.contains(&v) {
                out.push(v);
            }
        }
        out
    }

    fn random_open_iv(rng: &mut XorShift) -> OpenIv {
        let tiny = f64::from_bits(1);
        let ends = [-3.0, -1.0, -0.5, -tiny, 0.0, tiny, 0.25, 1.0, 2.0];
        loop {
            let (a, b) = (ends[rng.below(ends.len())], ends[rng.below(ends.len())]);
            let candidate = OpenIv {
                lo: a.min(b),
                hi: a.max(b),
                lo_open: rng.chance(0.4),
                hi_open: rng.chance(0.4),
            };
            if let Some(iv) = candidate.checked() {
                return iv;
            }
        }
    }

    /// A random closed span biased to put an anchor exactly at an endpoint,
    /// strictly inside, or as a point cell — the knife edges.
    fn random_span(rng: &mut XorShift, anchors: &[f64], width: f64) -> (f64, f64) {
        let anchor = anchors[rng.below(anchors.len())];
        let (a, b) = match rng.below(4) {
            0 => (anchor, anchor + rng.unit() * width),
            1 => (anchor - rng.unit() * width, anchor),
            2 => (anchor - rng.unit() * width, anchor + rng.unit() * width),
            _ => (anchor, anchor),
        };
        (a.min(b), a.max(b))
    }

    /// An f64 point of `[lo, hi]`, biased to endpoints, their neighbours and
    /// the anchors (and the anchors' neighbours) that fall inside.
    fn pick_member(rng: &mut XorShift, lo: f64, hi: f64, anchors: &[f64]) -> f64 {
        let mut candidates = vec![
            lo,
            hi,
            lo.next_up().min(hi),
            hi.next_down().max(lo),
            lo.midpoint(hi),
            (lo + (hi - lo) * rng.unit()).clamp(lo, hi),
        ];
        for &anchor in anchors {
            for v in [anchor, anchor.next_up(), anchor.next_down()] {
                if lo <= v && v <= hi {
                    candidates.push(v);
                }
            }
        }
        candidates[rng.below(candidates.len())]
    }

    /// ACC row 38, the X1 = 0 edge: the closed evaluator cannot refute a cell
    /// whose output enclosure straddles Y = 0 there, and the case split can.
    #[test]
    fn case_split_refutes_the_row38_x1_edge_cell_that_closed_intervals_cannot() {
        let formula = acc_row38();
        let (xl, xu, yl, yu) = ([28.0, 0.0], [29.0, 0.5], [-0.5], [0.5]);
        assert_eq!(
            formula.holds_over_box(&xl, &xu, &yl, &yu),
            Some(Tri::Unknown),
            "the fixture must reproduce the measured residual: closed intervals straddle Y = 0"
        );
        let verdict = formula
            .holds_over_box_split(&xl, &xu, &yl, &yu)
            .expect("supported formula and environment");
        assert_eq!(verdict.truth, Tri::False);
        assert_eq!(
            verdict.pieces, 6,
            "X1 in {{0}} | (0, 0.5] times Y in [-0.5, 0) | {{0}} | (0, 0.5]"
        );
    }

    /// MUTATION PIN: replacing every piece by its closure (treating open as
    /// closed) must put the row-38 knife-edge cells back to Unknown — both on
    /// the X1 = 0 edge and in the interior, where Y alone straddles 0. This is
    /// what proves openness, not the partition alone, carries the refutation.
    #[test]
    fn case_split_openness_is_load_bearing_on_row38() {
        let formula = acc_row38();
        for (xl, xu) in [([28.0, 0.0], [29.0, 0.5]), ([28.0, 1.0], [29.0, 1.5])] {
            let (yl, yu) = ([-0.5], [0.5]);
            assert_eq!(
                formula.holds_over_box(&xl, &xu, &yl, &yu),
                Some(Tri::Unknown)
            );
            let honest = formula
                .holds_over_box_split_with(&xl, &xu, &yl, &yu, true)
                .expect("supported");
            let closed = formula
                .holds_over_box_split_with(&xl, &xu, &yl, &yu, false)
                .expect("supported");
            assert_eq!(honest.truth, Tri::False, "cell {xl:?}..{xu:?}");
            assert_eq!(
                closed.truth,
                Tri::Unknown,
                "cell {xl:?}..{xu:?}: closed pieces must NOT refute it, or the pin is vacuous"
            );
        }
    }

    /// ACC row 15, the corner X0 -> 0+, X1 = 0 with Y ~ 6.75: refuted only
    /// with open pieces; the closure mutation leaves it Unknown.
    #[test]
    fn case_split_refutes_the_row15_corner_and_openness_is_load_bearing() {
        let formula = acc_row15();
        let (xl, xu, yl, yu) = ([0.0, 0.0], [0.01, 0.01], [6.0], [7.5]);
        assert_eq!(
            formula.holds_over_box(&xl, &xu, &yl, &yu),
            Some(Tri::Unknown)
        );
        let verdict = formula
            .holds_over_box_split(&xl, &xu, &yl, &yu)
            .expect("supported");
        assert_eq!(verdict.truth, Tri::False);
        assert_eq!(
            verdict.pieces, 4,
            "X0 in {{0}} | (0, 0.01] times X1 in {{0}} | (0, 0.01]"
        );
        let closed = formula
            .holds_over_box_split_with(&xl, &xu, &yl, &yu, false)
            .expect("supported");
        assert_eq!(closed.truth, Tri::Unknown);
    }

    /// The X0 = {0} piece is refuted by the strict `(> X[0,0] 0.0)` and by
    /// nothing else: with `>=` in its place the same piece holds EVERYWHERE
    /// (the corner is then a genuine violation), and the split must not
    /// refute the corner cell.
    #[test]
    fn case_split_x0_zero_piece_is_refuted_only_by_the_strict_input_atom() {
        let point = |v: f64| OpenIv::closed(v, v);
        let xs = [point(0.0), point(0.0)];
        let ys = [OpenIv::closed(6.0, 7.5)];
        let strict = acc_row15();
        let non_strict = acc_formula(("0.0", "20.0"), ("0.0", "40.0"), ">=");
        assert_eq!(strict.holds_over_pieces(&xs, &ys), Some(Tri::False));
        assert_eq!(non_strict.holds_over_pieces(&xs, &ys), Some(Tri::True));
        assert_eq!(strict.holds_at(&[0.0, 0.0], &[6.75]), Some(false));
        assert_eq!(non_strict.holds_at(&[0.0, 0.0], &[6.75]), Some(true));
        let corner = non_strict
            .holds_over_box_split(&[0.0, 0.0], &[0.01, 0.01], &[6.0], &[7.5])
            .expect("supported");
        assert_eq!(corner.truth, Tri::Unknown);
    }

    #[test]
    fn case_split_never_reports_true_and_refuses_malformed_cells() {
        let formula = acc_row38();
        // Every point of this cell violates the property (Y > 100.001).
        let (xl, xu, yl, yu) = ([28.0, 1.0], [29.0, 2.0], [101.0], [102.0]);
        assert_eq!(formula.holds_over_box(&xl, &xu, &yl, &yu), Some(Tri::True));
        let verdict = formula
            .holds_over_box_split(&xl, &xu, &yl, &yu)
            .expect("supported");
        assert_eq!(
            verdict,
            CaseSplitVerdict {
                truth: Tri::Unknown,
                pieces: 1
            }
        );
        assert_eq!(
            formula.holds_over_box_split(&[28.0], &[29.0], &[0.0], &[1.0]),
            None,
            "wrong input arity"
        );
        assert_eq!(
            formula.holds_over_box_split(&[29.0, 1.0], &[28.0, 2.0], &[0.0], &[1.0]),
            None,
            "a reversed cell is refused, never refuted"
        );
        assert_eq!(
            formula.holds_over_box_split(&[28.0, f64::NAN], &[29.0, 2.0], &[0.0], &[1.0]),
            None
        );
    }

    #[test]
    fn case_split_pivots_are_the_constants_of_single_variable_affine_atoms() {
        let decimal = |token: &str| parse_exact_decimal(token).expect("decimal");
        let collected = |formula: &NonLinearFormula| -> Vec<(VarKind, usize, Vec<BigRational>)> {
            formula
                .case_split_pivots()
                .iter()
                .map(|set| {
                    (
                        set.kind,
                        set.index,
                        set.values.iter().map(|n| n.exact.clone()).collect(),
                    )
                })
                .collect()
        };
        let row38 = acc_row38();
        assert_eq!(
            collected(&row38),
            vec![
                (
                    VarKind::Input,
                    0,
                    vec![decimal("0.0"), decimal("20.0"), decimal("40.0")]
                ),
                (VarKind::Input, 1, vec![decimal("0.0"), decimal("40.0")]),
                (
                    VarKind::Output,
                    0,
                    vec![
                        decimal("-100.001"),
                        decimal("-100.0"),
                        decimal("0.0"),
                        decimal("100.0"),
                        decimal("100.001")
                    ]
                ),
            ]
        );
        assert!(row38
            .case_split_pivot_summary()
            .starts_with("x0@{0,20,40} x1@{0,40} y0@{"));

        // Scaled, negated, shifted and divided single-variable atoms; the
        // two-variable product contributes nothing.
        let src = r#"
(declare-network f
    (declare-input X real [1,1])
    (declare-output Y real [1,1])
)
(assert (>= (* -2.0 X[0,0]) 3.0))
(assert (<= (- (/ Y[0,0] 4.0) 1.0) 0.5))
(assert (>= (* X[0,0] Y[0,0]) 0.0))
"#;
        let scaled = NonLinearFormula::parse(src).expect("parse");
        assert_eq!(
            collected(&scaled),
            vec![
                (VarKind::Input, 0, vec![decimal("-1.5")]),
                (VarKind::Output, 0, vec![decimal("6.0")]),
            ]
        );
    }

    #[test]
    fn case_split_declines_rather_than_explode_above_the_piece_cap() {
        // Nine pivots per variable -> 19 pieces each -> 19^3 = 6859 combinations.
        let mut src = String::from(
            "(declare-network f (declare-input X real [1,2]) (declare-output Y real [1,1]))\n",
        );
        for k in 1..=9 {
            src +=
                &format!("(assert (or (<= X[0,0] 0.{k}) (>= X[0,1] 0.{k}) (!= Y[0,0] 0.{k})))\n");
        }
        src += "(assert (>= (* X[0,0] X[0,1]) 0.0))\n";
        let formula = NonLinearFormula::parse(&src).expect("parse");
        assert_eq!(
            formula.holds_over_box_split(&[0.0, 0.0], &[1.0, 1.0], &[0.0], &[1.0]),
            Some(CaseSplitVerdict {
                truth: Tri::Unknown,
                pieces: 0
            })
        );
    }

    /// SOUNDNESS of the interval rules themselves: every exact result of
    /// `x op y` for f64 members `x`, `y` of random open/closed intervals lies
    /// in the computed enclosure, including subnormal and zero endpoints.
    #[test]
    fn open_interval_arithmetic_contains_every_sampled_exact_result() {
        let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
        let mut checked = 0usize;
        for _ in 0..4000 {
            let (a, b) = (random_open_iv(&mut rng), random_open_iv(&mut rng));
            let (xs, ys) = (members(a), members(b));
            if xs.is_empty() || ys.is_empty() {
                continue;
            }
            let negated = a.neg();
            for &x in &xs {
                assert!(
                    contains_exact(negated, &-exact(x)),
                    "-{a:?} = {negated:?} misses -{x}"
                );
            }
            for (op, result) in [
                ("+", a.add(b)),
                ("-", a.sub(b)),
                ("*", a.mul(b)),
                ("/", a.div(b)),
            ] {
                let Some(result) = result else { continue };
                for &x in &xs {
                    for &y in &ys {
                        let (ex, ey) = (exact(x), exact(y));
                        let value = match op {
                            "+" => ex + ey,
                            "-" => ex - ey,
                            "*" => ex * ey,
                            _ => {
                                if ey.is_zero() {
                                    continue;
                                }
                                ex / ey
                            }
                        };
                        assert!(
                            contains_exact(result, &value),
                            "{a:?} {op} {b:?} = {result:?} misses {x} {op} {y}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(
            checked > 10_000,
            "the sampler must exercise the rules, got {checked}"
        );
    }

    /// SOUNDNESS of the comparison rules: a committed verdict must agree with
    /// the exact comparison of EVERY sampled pair of members.
    #[test]
    fn open_interval_comparisons_commit_only_when_every_sampled_pair_agrees() {
        let mut rng = XorShift(0x2545_f491_4f6c_dd1d);
        let mut committed = 0usize;
        for _ in 0..4000 {
            let (a, b) = (random_open_iv(&mut rng), random_open_iv(&mut rng));
            let (xs, ys) = (members(a), members(b));
            if xs.is_empty() || ys.is_empty() {
                continue;
            }
            for op in ["<=", "<", ">=", ">", "==", "!="] {
                let verdict = compare_open(op, a, b).expect("known operator");
                if verdict == Tri::Unknown {
                    continue;
                }
                committed += 1;
                for &x in &xs {
                    for &y in &ys {
                        let (ex, ey) = (exact(x), exact(y));
                        let holds = match op {
                            "<=" => ex <= ey,
                            "<" => ex < ey,
                            ">=" => ex >= ey,
                            ">" => ex > ey,
                            "==" => ex == ey,
                            _ => ex != ey,
                        };
                        assert_eq!(
                            holds,
                            verdict == Tri::True,
                            "{a:?} {op} {b:?} committed {verdict:?}, but ({x}, {y}) disagrees"
                        );
                    }
                }
            }
        }
        assert!(
            committed > 1000,
            "the sampler must commit verdicts, got {committed}"
        );
    }

    /// SOUNDNESS of the partition: every sampled real of the cell — f64
    /// points near every endpoint and pivot, each exact pivot, and the exact
    /// midpoints between pivots — lies in at least one piece, and every piece
    /// lies inside the cell.
    #[test]
    fn case_split_partition_covers_every_real_of_the_cell() {
        let mut rng = XorShift(0xd1b5_4a32_d192_ed03);
        let decimals = [
            "-100.001", "-100.0", "-0.1", "0.0", "0.1", "0.25", "1.0", "100.001",
        ];
        let pivot = |token: &str| {
            let exact = parse_exact_decimal(token).expect("decimal");
            Number {
                lower: rational_to_lower_f64(&exact, 0).expect("finite"),
                upper: rational_to_upper_f64(&exact, 0).expect("finite"),
                exact,
            }
        };
        let anchors: Vec<f64> = decimals
            .iter()
            .map(|&token| pivot(token).lower)
            .chain(decimals.iter().map(|&token| pivot(token).upper))
            .collect();
        let mut pieces_seen = 0usize;
        for _ in 0..3000 {
            let mut chosen: Vec<Number> = Vec::new();
            for token in decimals {
                if rng.chance(0.5) {
                    chosen.push(pivot(token));
                }
            }
            let (lo, hi) = random_span(&mut rng, &anchors, 0.3);
            let cell = OpenIv::closed(lo, hi);
            let pieces = partition_at_pivots(cell, &chosen);
            pieces_seen += pieces.len();
            for piece in &pieces {
                assert!(
                    piece.lo >= cell.lo && piece.hi <= cell.hi && !piece.is_empty(),
                    "piece {piece:?} escapes cell {cell:?}"
                );
            }
            let mut probes: Vec<BigRational> = Vec::new();
            for _ in 0..8 {
                probes.push(exact(pick_member(&mut rng, lo, hi, &anchors)));
            }
            for (position, number) in chosen.iter().enumerate() {
                probes.push(number.exact.clone());
                if let Some(next) = chosen.get(position + 1) {
                    probes.push(
                        (number.exact.clone() + next.exact.clone())
                            / BigRational::from_integer(2.into()),
                    );
                }
            }
            for probe in probes {
                if !contains_exact(cell, &probe) {
                    continue;
                }
                assert!(
                    pieces.iter().any(|piece| contains_exact(*piece, &probe)),
                    "{probe} in cell {cell:?} is covered by no piece of {pieces:?} (pivots {:?})",
                    chosen
                        .iter()
                        .map(|n| n.exact.to_string())
                        .collect::<Vec<_>>()
                );
            }
        }
        assert!(
            pieces_seen > 3000,
            "the partition must actually split cells"
        );
    }

    /// THE PROPTEST (seeded, dependency-free). For random ACC-shaped cells
    /// biased onto every knife edge — X0 = 0, X1 = 0, Y = 0, Y = +-100.001 —
    /// a case-split refutation implies that NO exact-rational point of the
    /// cell x enclosure satisfies the formula. The `>=` variant makes the
    /// corner X0 = X1 = 0 a genuine violation, so the sampler also has
    /// positive controls it must never refute.
    #[test]
    fn case_split_false_implies_no_sampled_exact_point_satisfies_the_formula() {
        let strict = acc_formula(("0.0", "100.0"), ("-40.0", "40.0"), ">");
        let non_strict = acc_formula(("0.0", "100.0"), ("-40.0", "40.0"), ">=");
        let tiny = f64::from_bits(1);
        let x0_anchors = [0.0, tiny, 1e-3, 20.0, 28.78, 40.0];
        let x1_anchors = [0.0, 1e-3, -1e-3, 6.55, 40.0, -40.0];
        let y_anchors = [
            0.0,
            tiny,
            -tiny,
            1e-3,
            -1e-3,
            6.75,
            100.0,
            -100.0,
            100.001,
            -100.001,
            100.001_f64.next_up(),
            (-100.001_f64).next_down(),
        ];
        let mut rng = XorShift(0x6a09_e667_f3bc_c909);
        let (mut refuted, mut knife_edge, mut violations) = (0usize, 0usize, 0usize);
        for round in 0..1200 {
            let formula = if round % 2 == 0 { &strict } else { &non_strict };
            let (x0l, x0u) = random_span(&mut rng, &x0_anchors, 0.05);
            let (x1l, x1u) = random_span(&mut rng, &x1_anchors, 0.05);
            let (yl, yu) = random_span(&mut rng, &y_anchors, 0.2);
            let (xl, xu) = ([x0l, x1l], [x0u, x1u]);
            let verdict = formula
                .holds_over_box_split(&xl, &xu, &[yl], &[yu])
                .expect("well-formed cell");
            let closed = formula
                .holds_over_box(&xl, &xu, &[yl], &[yu])
                .expect("well-formed cell");
            let split_refuted = verdict.truth == Tri::False;
            if split_refuted {
                refuted += 1;
                if closed != Tri::False {
                    knife_edge += 1;
                }
            }
            let samples = if split_refuted || closed == Tri::False {
                24
            } else {
                6
            };
            for _ in 0..samples {
                let x = [
                    pick_member(&mut rng, x0l, x0u, &x0_anchors),
                    pick_member(&mut rng, x1l, x1u, &x1_anchors),
                ];
                let y = [pick_member(&mut rng, yl, yu, &y_anchors)];
                let holds = formula.holds_at(&x, &y).expect("supported");
                if holds {
                    violations += 1;
                }
                assert!(
                    !(holds && split_refuted),
                    "case split refuted {xl:?}..{xu:?} x [{yl}, {yu}] but x={x:?}, y={y:?} \
                     satisfies the formula exactly"
                );
                assert!(
                    !(holds && closed == Tri::False),
                    "closed evaluator refuted {xl:?}..{xu:?} x [{yl}, {yu}] but x={x:?}, y={y:?} \
                     satisfies the formula exactly"
                );
            }
        }
        assert!(
            refuted > 100,
            "too few refutations to mean anything: {refuted}"
        );
        assert!(
            knife_edge > 10,
            "the split must refute cells the closed evaluator cannot, got {knife_edge}"
        );
        assert!(
            violations > 0,
            "positive control: the sampler must find genuine violations somewhere"
        );
    }
}

/// Typed binary32 execution of the original complete predicate.
#[path = "nonlinear/f32_predicate.rs"]
pub mod f32_predicate;

#[path = "nonlinear/f32_chain.rs"]
pub mod f32_chain;
