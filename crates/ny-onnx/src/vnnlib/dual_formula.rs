// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! FULL-COVERAGE DNF extraction of a dual-network VNN-LIB 2.0 formula
//! (relational-ACAS gate-flip hardening: the exact-DNF input that lets the
//! relational `unsat` gate be proved by implication rather than shape match).
//!
//! Unlike the canonical-shape validation flags in [`super::spec`], this module
//! converts the WHOLE asserted formula — every assert, every disjunct, every
//! atom — into a disjunctive normal form over exact-rational linear atoms,
//! with NO canonical-shape assumptions. The downstream formula-implication
//! check (ny-cli `relational_equiv`) then proves `parsed_unsafe ⇒ E` (the
//! region the difference-network verifier literally proved empty) clause by
//! clause with exact-LP Farkas certificates — replacing the fragile
//! shape-matching that keeps the relational `unsat` gate down.
//!
//! FAIL-CLOSED CONTRACT: any construct this extractor cannot express EXACTLY
//! as a linear atom over the declared dual-network variables (unknown
//! operator, non-linear term, inexact constant arithmetic, index out of
//! range, clause blow-up, …) yields `None` for the whole formula — the
//! caller keeps the gate down for that instance. A returned DNF is EXACTLY
//! the asserted formula under f64-literal semantics (each decimal literal
//! denotes the f64 it parses to, whose dyadic value is preserved exactly —
//! the same interpretation the existing isomorphic Farkas path uses).

use std::collections::BTreeMap;

use super::syntax::Expr;

/// Cap on DNF clauses (the AND-of-ORs cross product); beyond this the
/// extraction fails closed rather than blowing up.
const MAX_DNF_CLAUSES: usize = 512;

/// Which dual-network tensor a variable belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DualVarRole {
    /// The reference network's input (`X_f`).
    FInput,
    /// The derived network's input (`X_g`).
    GInput,
    /// The reference network's output (`Y_f`).
    FOutput,
    /// The derived network's output (`Y_g`).
    GOutput,
}

/// One dual-network scalar variable: a role + FLAT (row-major) index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DualVar {
    /// Which declared tensor.
    pub role: DualVarRole,
    /// Flat row-major element index within that tensor.
    pub index: usize,
}

/// Atom relation, exactly as parsed (strictness preserved).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DualAtomRelation {
    /// `Σ c·v <= k`
    Le,
    /// `Σ c·v < k`
    Lt,
    /// `Σ c·v >= k`
    Ge,
    /// `Σ c·v > k`
    Gt,
    /// `Σ c·v == k`
    Eq,
}

/// An EXACT dyadic rational `mant · 2^exp2` — the lossless value of a parsed
/// f64 literal and of every exact combination this extractor performs.
/// All arithmetic is checked; overflow fails closed (`None`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dyadic {
    /// Signed mantissa.
    pub mant: i128,
    /// Base-2 exponent (value = mant * 2^exp2).
    pub exp2: i32,
}

impl Dyadic {
    /// The exact dyadic value of a finite f64. `None` for NaN/infinity.
    pub fn from_f64(x: f64) -> Option<Self> {
        if !x.is_finite() {
            return None;
        }
        if x == 0.0 {
            return Some(Self { mant: 0, exp2: 0 });
        }
        let bits = x.to_bits();
        let sign: i128 = if bits >> 63 == 0 { 1 } else { -1 };
        let exp_field = ((bits >> 52) & 0x7ff) as i32;
        let frac = (bits & 0x000f_ffff_ffff_ffff) as i128;
        let (mant, exp2) = if exp_field == 0 {
            (frac, -1022 - 52)
        } else {
            ((1i128 << 52) | frac, exp_field - 1023 - 52)
        };
        Some(Self::normalized(Self {
            mant: sign * mant,
            exp2,
        }))
    }

    /// Strip trailing zero bits so equal values have equal representations.
    fn normalized(mut self) -> Self {
        if self.mant == 0 {
            self.exp2 = 0;
            return self;
        }
        while self.mant % 2 == 0 {
            self.mant /= 2;
            self.exp2 += 1;
        }
        self
    }

    /// Exact sum. `None` on overflow (alignment shift or mantissa add).
    // Inherent fallible dyadic op returning `Option`, not the infallible
    // `std::ops::Add` shape — this is the exact-arithmetic DSL.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Option<Self> {
        let min_exp = self.exp2.min(other.exp2);
        let a = self
            .mant
            .checked_shl(u32::try_from(self.exp2 - min_exp).ok()?.min(127))?;
        // A shift that large would have overflowed i128 already unless mant==0.
        let b = other
            .mant
            .checked_shl(u32::try_from(other.exp2 - min_exp).ok()?.min(127))?;
        Some(
            Self {
                mant: a.checked_add(b)?,
                exp2: min_exp,
            }
            .normalized(),
        )
    }

    /// Exact product. `None` on overflow.
    // Inherent fallible dyadic op returning `Option`, not the infallible
    // `std::ops::Mul` shape — this is the exact-arithmetic DSL.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Option<Self> {
        Some(
            Self {
                mant: self.mant.checked_mul(other.mant)?,
                exp2: self.exp2.checked_add(other.exp2)?,
            }
            .normalized(),
        )
    }

    /// Exact negation.
    // Inherent dyadic op, part of the exact-arithmetic DSL — deliberately
    // not the `std::ops::Neg` trait method.
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        Self {
            mant: -self.mant,
            exp2: self.exp2,
        }
    }

    /// Whether this is exactly zero.
    pub fn is_zero(&self) -> bool {
        self.mant == 0
    }
}

/// One linear atom `Σ coeffs·vars ⋈ constant`, all values exact dyadics,
/// relation exactly as parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DualLinearAtom {
    /// The relation, strictness preserved.
    pub relation: DualAtomRelation,
    /// Variable coefficients (deduplicated, sorted by variable, no zeros).
    pub coeffs: Vec<(DualVar, Dyadic)>,
    /// Right-hand-side constant.
    pub constant: Dyadic,
}

/// The FULL asserted formula in DNF: the unsafe region is the OR of the
/// clauses; each clause is a conjunction of atoms. Guaranteed complete: every
/// assert of the file contributed (the extraction fails closed otherwise).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DualFormulaDnf {
    /// OR of AND-clauses.
    pub clauses: Vec<Vec<DualLinearAtom>>,
    /// Number of top-level asserts folded in (diagnostics).
    pub num_asserts: usize,
}

/// A declared dual tensor: name -> (role, row-major shape).
struct TensorDecl {
    name: String,
    role: DualVarRole,
    shape: Vec<usize>,
}

/// Linear expression accumulator: `Σ coeffs·vars + constant`.
#[derive(Clone)]
struct LinExpr {
    coeffs: BTreeMap<DualVar, Dyadic>,
    constant: Dyadic,
}

impl LinExpr {
    fn constant(c: Dyadic) -> Self {
        Self {
            coeffs: BTreeMap::new(),
            constant: c,
        }
    }

    fn var(v: DualVar) -> Self {
        let mut coeffs = BTreeMap::new();
        coeffs.insert(v, Dyadic { mant: 1, exp2: 0 });
        Self {
            coeffs,
            constant: Dyadic { mant: 0, exp2: 0 },
        }
    }

    fn add(mut self, other: &Self) -> Option<Self> {
        for (v, c) in &other.coeffs {
            let entry = self.coeffs.entry(*v).or_insert(Dyadic { mant: 0, exp2: 0 });
            *entry = entry.add(*c)?;
        }
        self.constant = self.constant.add(other.constant)?;
        self.coeffs.retain(|_, c| !c.is_zero());
        Some(self)
    }

    fn neg(mut self) -> Self {
        for c in self.coeffs.values_mut() {
            *c = c.neg();
        }
        self.constant = self.constant.neg();
        self
    }

    fn scale(mut self, k: Dyadic) -> Option<Self> {
        for c in self.coeffs.values_mut() {
            *c = c.mul(k)?;
        }
        self.constant = self.constant.mul(k)?;
        self.coeffs.retain(|_, c| !c.is_zero());
        Some(self)
    }

    fn is_constant(&self) -> bool {
        self.coeffs.is_empty()
    }
}

/// Extract the FULL formula DNF from a dual-network VNN-LIB 2.0 expression
/// list. `None` (fail-closed) whenever ANY assert cannot be converted exactly.
pub(crate) fn extract_dual_formula_dnf(exprs: &[Expr]) -> Option<DualFormulaDnf> {
    let tensors = collect_tensor_decls(exprs)?;
    // AND over asserts: DNF cross product, capped.
    let mut dnf: Vec<Vec<DualLinearAtom>> = vec![Vec::new()];
    let mut num_asserts = 0usize;
    for expr in exprs {
        let Expr::List(items) = expr else { continue };
        let Some(Expr::Symbol(op)) = items.first() else {
            continue;
        };
        if op != "assert" {
            continue;
        }
        if items.len() != 2 {
            return None; // malformed assert: fail closed
        }
        num_asserts += 1;
        let assert_dnf = formula_to_dnf(&items[1], &tensors)?;
        // dnf = dnf AND assert_dnf (clause cross product).
        let mut combined = Vec::with_capacity(dnf.len().saturating_mul(assert_dnf.len()));
        for base in &dnf {
            for add in &assert_dnf {
                if combined.len() >= MAX_DNF_CLAUSES {
                    return None; // blow-up: fail closed
                }
                let mut clause = base.clone();
                clause.extend(add.iter().cloned());
                combined.push(clause);
            }
        }
        dnf = combined;
        if dnf.is_empty() {
            return None; // an assert with an empty DNF cannot be represented
        }
    }
    if num_asserts == 0 {
        return None; // nothing asserted: refuse to "cover" a vacuous formula
    }
    Some(DualFormulaDnf {
        clauses: dnf,
        num_asserts,
    })
}

/// Parse the two `declare-network` blocks into tensor declarations with FULL
/// shapes. The derived network is the one carrying `isomorphic-to`/`equal-to`
/// (or `ground-truth`); the other is the reference. `None` unless exactly two
/// networks with exactly one derived are declared.
fn collect_tensor_decls(exprs: &[Expr]) -> Option<Vec<TensorDecl>> {
    struct NetDecl {
        input: (String, Vec<usize>),
        output: (String, Vec<usize>),
        is_derived: bool,
    }
    let mut nets: Vec<NetDecl> = Vec::new();
    for expr in exprs {
        let Expr::List(items) = expr else { continue };
        let Some(Expr::Symbol(op)) = items.first() else {
            continue;
        };
        if op != "declare-network" {
            continue;
        }
        let mut input = None;
        let mut output = None;
        let mut is_derived = false;
        for nested in items.iter().skip(2) {
            let Expr::List(nested_items) = nested else {
                continue;
            };
            let Some(Expr::Symbol(nested_op)) = nested_items.first() else {
                continue;
            };
            match nested_op.as_str() {
                "declare-input" => input = parse_decl_name_shape(nested_items),
                "declare-output" => output = parse_decl_name_shape(nested_items),
                "isomorphic-to" | "equal-to" | "ground-truth" => is_derived = true,
                _ => {}
            }
        }
        nets.push(NetDecl {
            input: input?,
            output: output?,
            is_derived,
        });
    }
    if nets.len() != 2 {
        return None;
    }
    let derived_count = nets.iter().filter(|n| n.is_derived).count();
    if derived_count != 1 {
        return None;
    }
    let mut tensors = Vec::with_capacity(4);
    for net in nets {
        let (in_role, out_role) = if net.is_derived {
            (DualVarRole::GInput, DualVarRole::GOutput)
        } else {
            (DualVarRole::FInput, DualVarRole::FOutput)
        };
        tensors.push(TensorDecl {
            name: net.input.0,
            role: in_role,
            shape: net.input.1,
        });
        tensors.push(TensorDecl {
            name: net.output.0,
            role: out_role,
            shape: net.output.1,
        });
    }
    Some(tensors)
}

/// Parse `(declare-input NAME real [d0, d1, ...])` into (NAME, shape).
fn parse_decl_name_shape(items: &[Expr]) -> Option<(String, Vec<usize>)> {
    let Some(Expr::Symbol(name)) = items.get(1) else {
        return None;
    };
    // The shape token(s) follow the dtype symbol. The tokenizer keeps
    // `[1,` `1,` `5]` style fragments as symbols (commas/brackets are not
    // delimiters), possibly split on spaces — reassemble from the first
    // bracket-bearing token (dtype-agnostic: `real`, `Float32`, ...).
    let mut shape_text = String::new();
    let mut started = false;
    for item in items.iter().skip(2) {
        match item {
            Expr::Symbol(s) => {
                if !started && s.contains('[') {
                    started = true;
                }
                if started {
                    shape_text.push_str(s);
                }
            }
            Expr::Number(n) if started => {
                shape_text.push_str(&format!("{n}"));
            }
            _ => {}
        }
    }
    let inner = shape_text
        .trim()
        .strip_prefix('[')?
        .strip_suffix(']')?
        .trim();
    if inner.is_empty() {
        return Some((name.clone(), vec![]));
    }
    let mut shape = Vec::new();
    for part in inner.split(',') {
        let d: usize = part.trim().parse().ok()?;
        shape.push(d);
    }
    Some((name.clone(), shape))
}

/// Convert one asserted formula into DNF (recursion over `and`/`or`;
/// leaves are relations). Fail-closed `None` on anything else.
fn formula_to_dnf(expr: &Expr, tensors: &[TensorDecl]) -> Option<Vec<Vec<DualLinearAtom>>> {
    let Expr::List(items) = expr else {
        return None;
    };
    let Some(Expr::Symbol(op)) = items.first() else {
        return None;
    };
    match op.as_str() {
        "and" => {
            // DNF(A ∧ B ∧ …) = cross product of the operand DNFs.
            let mut dnf: Vec<Vec<DualLinearAtom>> = vec![Vec::new()];
            for sub in items.iter().skip(1) {
                let sub_dnf = formula_to_dnf(sub, tensors)?;
                let mut combined = Vec::new();
                for base in &dnf {
                    for add in &sub_dnf {
                        if combined.len() >= MAX_DNF_CLAUSES {
                            return None;
                        }
                        let mut clause = base.clone();
                        clause.extend(add.iter().cloned());
                        combined.push(clause);
                    }
                }
                dnf = combined;
            }
            Some(dnf)
        }
        "or" => {
            // DNF(A ∨ B ∨ …) = concatenation of the operand DNFs.
            let mut dnf = Vec::new();
            for sub in items.iter().skip(1) {
                let mut sub_dnf = formula_to_dnf(sub, tensors)?;
                dnf.append(&mut sub_dnf);
                if dnf.len() > MAX_DNF_CLAUSES {
                    return None;
                }
            }
            if dnf.is_empty() {
                return None; // empty `or` is `false`: unrepresentable here
            }
            Some(dnf)
        }
        "<=" | "<" | ">=" | ">" | "==" | "=" => {
            let atom = relation_to_atom(op, items, tensors)?;
            Some(vec![vec![atom]])
        }
        _ => None, // unknown operator: fail closed
    }
}

/// Convert `(op lhs rhs)` into a canonical atom `Σ c·v ⋈ k` (moving every
/// variable left and every constant right, exactly).
fn relation_to_atom(op: &str, items: &[Expr], tensors: &[TensorDecl]) -> Option<DualLinearAtom> {
    if items.len() != 3 {
        return None;
    }
    let lhs = linear_of_expr(&items[1], tensors)?;
    let rhs = linear_of_expr(&items[2], tensors)?;
    // lhs ⋈ rhs  ⇔  (lhs − rhs) ⋈ 0  ⇔  Σ c·v ⋈ (rhs.const − lhs.const)
    let diff = lhs.add(&rhs.neg())?;
    let relation = match op {
        "<=" => DualAtomRelation::Le,
        "<" => DualAtomRelation::Lt,
        ">=" => DualAtomRelation::Ge,
        ">" => DualAtomRelation::Gt,
        "==" | "=" => DualAtomRelation::Eq,
        _ => return None,
    };
    if diff.is_constant() {
        return None; // variable-free relation: nothing to prove against, refuse
    }
    Some(DualLinearAtom {
        relation,
        coeffs: diff.coeffs.into_iter().collect(),
        constant: diff.constant.neg(),
    })
}

/// Evaluate an expression to an exact linear form. Supports numbers, indexed
/// dual-tensor variables, `+`, binary/unary `-`, and `*` with one constant
/// side. Anything else — including any INEXACT arithmetic — fails closed.
fn linear_of_expr(expr: &Expr, tensors: &[TensorDecl]) -> Option<LinExpr> {
    match expr {
        Expr::Number(n) => Some(LinExpr::constant(Dyadic::from_f64(*n)?)),
        Expr::Symbol(s) => {
            // Either a numeric literal the tokenizer left symbolic, or an
            // indexed variable `NAME[i,j,...]`.
            if let Ok(n) = s.parse::<f64>() {
                return Some(LinExpr::constant(Dyadic::from_f64(n)?));
            }
            Some(LinExpr::var(parse_indexed_var(s, tensors)?))
        }
        Expr::List(items) => {
            let Some(Expr::Symbol(op)) = items.first() else {
                return None;
            };
            match op.as_str() {
                "+" => {
                    let mut acc = LinExpr::constant(Dyadic { mant: 0, exp2: 0 });
                    for sub in items.iter().skip(1) {
                        acc = acc.add(&linear_of_expr(sub, tensors)?)?;
                    }
                    Some(acc)
                }
                "-" => match items.len() {
                    2 => Some(linear_of_expr(&items[1], tensors)?.neg()),
                    n if n >= 3 => {
                        let mut acc = linear_of_expr(&items[1], tensors)?;
                        for sub in items.iter().skip(2) {
                            acc = acc.add(&linear_of_expr(sub, tensors)?.neg())?;
                        }
                        Some(acc)
                    }
                    _ => None,
                },
                "*" => {
                    if items.len() != 3 {
                        return None;
                    }
                    let a = linear_of_expr(&items[1], tensors)?;
                    let b = linear_of_expr(&items[2], tensors)?;
                    match (a.is_constant(), b.is_constant()) {
                        (true, _) => b.scale(a.constant),
                        (_, true) => a.scale(b.constant),
                        _ => None, // nonlinear product: fail closed
                    }
                }
                _ => None,
            }
        }
    }
}

/// Parse `NAME[i,j,...]` into a [`DualVar`] with a row-major flat index,
/// validated against the declared shape. `None` on unknown tensor, rank
/// mismatch, or out-of-range index.
fn parse_indexed_var(sym: &str, tensors: &[TensorDecl]) -> Option<DualVar> {
    let open = sym.find('[')?;
    let name = &sym[..open];
    let idx_part = sym[open..].strip_prefix('[')?.strip_suffix(']')?;
    let decl = tensors.iter().find(|t| t.name == name)?;
    let indices: Vec<usize> = idx_part
        .split(',')
        .map(|p| p.trim().parse::<usize>().ok())
        .collect::<Option<_>>()?;
    if indices.len() != decl.shape.len() {
        return None;
    }
    let mut flat = 0usize;
    for (i, (&idx, &dim)) in indices.iter().zip(decl.shape.iter()).enumerate() {
        if idx >= dim {
            return None;
        }
        let _ = i;
        flat = flat.checked_mul(dim)?.checked_add(idx)?;
    }
    Some(DualVar {
        role: decl.role,
        index: flat,
    })
}
