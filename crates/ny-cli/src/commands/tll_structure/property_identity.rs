// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact source authentication for the deliberately narrow TLL VNN-LIB form.
//!
//! Accepted properties use one of the exact scalar VNN-LIB 1.0 or tensor
//! VNN-LIB 2.0 declaration envelopes used by TLLVerifyBench. Each gives both
//! inputs one direct lower and upper decimal bound and contains one direct
//! scalar output comparison. For VNN-LIB 1.0, the shared certified parser also
//! supplies an independently extracted exact-rational input box. This module
//! separately proves both raw dialects, constructs their exact-decimal outward
//! boxes, and rounds the output decimal in the verdict-safe direction.
//! Arithmetic, aliases, Boolean structure, mixed dialects, and extra assertions
//! are rejected instead of being normalized into authority.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::path::Path;

use ny_onnx::vnnlib::{
    parse_vnnlib, parse_vnnlib_with_certified_input_box, OutputConstraint, VnnLibSpec,
};
use sha2::{Digest, Sha256};

use super::ThreshDir;

const MAX_PROPERTY_BYTES: u64 = 1024 * 1024;
const MAX_TOKENS: usize = 4096;
const MAX_NESTING: usize = 32;
const MAX_DECIMAL_DIGITS: usize = 4096;
const MAX_DECIMAL_EXPONENT: i32 = 4096;

/// Property proof coupled to the exact source bytes that were authenticated.
pub(super) struct AuthenticatedTllProperty {
    spec: VnnLibSpec,
    input_bounds: [(f64, f64); 2],
    direction: ThreshDir,
    directed_threshold: f64,
    source_sha256: [u8; 32],
}

impl AuthenticatedTllProperty {
    pub(super) fn spec(&self) -> &VnnLibSpec {
        &self.spec
    }

    pub(super) fn input_bounds(&self) -> [(f64, f64); 2] {
        self.input_bounds
    }

    /// Threshold rounded conservatively for proving the unsafe atom empty.
    pub(super) fn threshold(&self) -> (ThreshDir, f64) {
        (self.direction, self.directed_threshold)
    }

    pub(super) fn source_still_matches(&self, path: &Path) -> bool {
        let Ok(metadata) = std::fs::metadata(path) else {
            return false;
        };
        if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_PROPERTY_BYTES {
            return false;
        }
        let Ok(bytes) = std::fs::read(path) else {
            return false;
        };
        u64::try_from(bytes.len()).ok() == Some(metadata.len())
            && <[u8; 32]>::from(Sha256::digest(&bytes)) == self.source_sha256
    }
}

/// Authenticate a plain, materialized `.vnnlib` source. Compressed benchmark
/// archives must be materialized by the caller, as the competition wrapper
/// already does for the model and property paths passed to NY.
pub(super) fn authenticate_raw_tll_property(path: &Path) -> Option<AuthenticatedTllProperty> {
    if path.extension().and_then(|extension| extension.to_str()) != Some("vnnlib") {
        return None;
    }
    let metadata = std::fs::metadata(path).ok()?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_PROPERTY_BYTES {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    if u64::try_from(bytes.len()).ok()? != metadata.len() {
        return None;
    }
    let content = std::str::from_utf8(&bytes).ok()?;
    let raw = authenticate_raw_shape(content)?;
    // The shared certified input parser is an independent exact-rational
    // oracle for scalar VNN-LIB 1.0. Its currently published surface does not
    // recognize VNN-LIB 2.0 tensor references, so the separately authenticated
    // tensor dialect uses this module's exact decimal enclosure and the shared
    // ordinary parser as its structural cross-check.
    let (spec, independently_certified_box) = match raw.dialect {
        PropertyDialect::ScalarV1 => {
            let (spec, box_) = parse_vnnlib_with_certified_input_box(content).ok()?;
            (spec, Some(box_))
        }
        PropertyDialect::TensorV2 => (parse_vnnlib(content).ok()?, None),
    };
    let input_bounds = raw.outward_input_bounds()?;
    cross_check_ordinary_spec(&spec, &raw, input_bounds)?;
    if let Some(certified_box) = independently_certified_box {
        if certified_box.len() != 2
            || certified_box.lower().len() != 2
            || certified_box.upper().len() != 2
            || certified_box
                .lower()
                .iter()
                .zip(certified_box.upper())
                .zip(input_bounds)
                .any(
                    |((&certified_lower, &certified_upper), (raw_lower, raw_upper))| {
                        certified_lower.to_bits() != raw_lower.to_bits()
                            || certified_upper.to_bits() != raw_upper.to_bits()
                    },
                )
        {
            return None;
        }
    }

    let (lower, upper) = raw.threshold.decimal.outward_f64()?;
    let directed_threshold = match raw.threshold.direction {
        // Proving lb > ceil(c) is at least as strong as proving lb > exact c.
        ThreshDir::Le => upper,
        // Proving ub < floor(c) is at least as strong as proving ub < exact c.
        ThreshDir::Ge => lower,
    };
    if !directed_threshold.is_finite() {
        return None;
    }

    Some(AuthenticatedTllProperty {
        spec,
        input_bounds,
        direction: raw.threshold.direction,
        directed_threshold,
        source_sha256: Sha256::digest(&bytes).into(),
    })
}

#[derive(Clone)]
struct RawThreshold<'a> {
    direction: ThreshDir,
    strict: bool,
    literal: &'a str,
    decimal: ExactDecimal,
}

struct RawProperty<'a> {
    dialect: PropertyDialect,
    lower: [ExactDecimal; 2],
    upper: [ExactDecimal; 2],
    threshold: RawThreshold<'a>,
}

impl RawProperty<'_> {
    fn outward_input_bounds(&self) -> Option<[(f64, f64); 2]> {
        let endpoint = |index: usize| {
            if self.lower[index].cmp_exact(&self.upper[index])? == Ordering::Greater {
                return None;
            }
            let (lower, _) = self.lower[index].outward_f64()?;
            let (_, upper) = self.upper[index].outward_f64()?;
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return None;
            }
            Some((lower, upper))
        };
        Some([endpoint(0)?, endpoint(1)?])
    }
}

/// Prove the complete raw top-level shape and return its sole output atom.
fn authenticate_raw_shape(content: &str) -> Option<RawProperty<'_>> {
    let tokens = tokenize(content)?;
    let expressions = parse_expressions(&tokens)?;
    let dialect = authenticate_declaration_envelope(&expressions)?;
    let mut lower = [None, None];
    let mut upper = [None, None];
    let mut output = None;

    for expression in &expressions {
        let items = expression.list()?;
        match items {
            [RawExpr::Atom("assert"), asserted] => {
                let atom = DirectAtom::parse(asserted, dialect)?;
                match atom.variable {
                    DirectVariable::Input(index) => {
                        let slot = match atom.direction {
                            AtomDirection::Le => &mut upper[index],
                            AtomDirection::Ge => &mut lower[index],
                        };
                        if slot.replace(atom.decimal).is_some() {
                            return None;
                        }
                    }
                    DirectVariable::Output => {
                        if output.is_some() {
                            return None;
                        }
                        output = Some(RawThreshold {
                            direction: match atom.direction {
                                AtomDirection::Le => ThreshDir::Le,
                                AtomDirection::Ge => ThreshDir::Ge,
                            },
                            strict: atom.strict,
                            literal: atom.literal,
                            decimal: atom.decimal,
                        });
                    }
                }
            }
            _ if is_authenticated_declaration(expression, dialect) => {}
            _ => return None,
        }
    }

    let [Some(lower0), Some(lower1)] = lower else {
        return None;
    };
    let [Some(upper0), Some(upper1)] = upper else {
        return None;
    };
    Some(RawProperty {
        dialect,
        lower: [lower0, lower1],
        upper: [upper0, upper1],
        threshold: output?,
    })
}

#[derive(Clone, Copy)]
enum PropertyDialect {
    ScalarV1,
    TensorV2,
}

fn authenticate_declaration_envelope(expressions: &[RawExpr<'_>]) -> Option<PropertyDialect> {
    if expressions.len() == 8 {
        let mut declarations = HashSet::new();
        for expression in expressions {
            let Some([RawExpr::Atom("declare-const"), RawExpr::Atom(name), RawExpr::Atom("Real")]) =
                expression.list()
            else {
                continue;
            };
            if !matches!(*name, "X_0" | "X_1" | "Y_0") || !declarations.insert(*name) {
                return None;
            }
        }
        if declarations == HashSet::from(["X_0", "X_1", "Y_0"]) {
            return Some(PropertyDialect::ScalarV1);
        }
    }

    if expressions.len() == 7
        && is_v2_version(expressions.first()?)
        && is_v2_network(expressions.get(1)?)
    {
        return Some(PropertyDialect::TensorV2);
    }
    None
}

fn is_authenticated_declaration(expression: &RawExpr<'_>, dialect: PropertyDialect) -> bool {
    match dialect {
        PropertyDialect::ScalarV1 => matches!(
            expression.list(),
            Some([
                RawExpr::Atom("declare-const"),
                RawExpr::Atom("X_0" | "X_1" | "Y_0"),
                RawExpr::Atom("Real")
            ])
        ),
        PropertyDialect::TensorV2 => is_v2_version(expression) || is_v2_network(expression),
    }
}

fn is_v2_version(expression: &RawExpr<'_>) -> bool {
    matches!(
        expression.list(),
        Some([RawExpr::Atom("vnnlib-version"), RawExpr::Atom("<2.0>")])
    )
}

fn is_v2_network(expression: &RawExpr<'_>) -> bool {
    let Some([RawExpr::Atom("declare-network"), RawExpr::Atom("N"), input, output]) =
        expression.list()
    else {
        return false;
    };
    matches!(
        input.list(),
        Some([
            RawExpr::Atom("declare-input"),
            RawExpr::Atom("X"),
            RawExpr::Atom("float32"),
            RawExpr::Atom("[1,"),
            RawExpr::Atom("2]")
        ])
    ) && matches!(
        output.list(),
        Some([
            RawExpr::Atom("declare-output"),
            RawExpr::Atom("Y"),
            RawExpr::Atom("float32"),
            RawExpr::Atom("[1,"),
            RawExpr::Atom("1]")
        ])
    )
}

#[derive(Clone, Copy)]
enum DirectVariable {
    Input(usize),
    Output,
}

#[derive(Clone, Copy)]
enum AtomDirection {
    Le,
    Ge,
}

#[derive(Clone)]
struct DirectAtom<'a> {
    variable: DirectVariable,
    direction: AtomDirection,
    strict: bool,
    literal: &'a str,
    decimal: ExactDecimal,
}

impl<'a> DirectAtom<'a> {
    fn parse(expression: &RawExpr<'a>, dialect: PropertyDialect) -> Option<Self> {
        let [RawExpr::Atom(operator), RawExpr::Atom(left), RawExpr::Atom(right)] =
            expression.list()?
        else {
            return None;
        };
        let (base_direction, strict) = match *operator {
            "<=" => (AtomDirection::Le, false),
            "<" => (AtomDirection::Le, true),
            ">=" => (AtomDirection::Ge, false),
            ">" => (AtomDirection::Ge, true),
            _ => return None,
        };
        let left_variable = parse_variable(left, dialect);
        let right_variable = parse_variable(right, dialect);
        let (variable, literal, direction) = match (left_variable, right_variable) {
            (Some(variable), None) => (variable, *right, base_direction),
            (None, Some(variable)) => (
                variable,
                *left,
                match base_direction {
                    AtomDirection::Le => AtomDirection::Ge,
                    AtomDirection::Ge => AtomDirection::Le,
                },
            ),
            _ => return None,
        };
        let decimal = ExactDecimal::parse(literal)?;
        Some(Self {
            variable,
            direction,
            strict,
            literal,
            decimal,
        })
    }
}

fn parse_variable(token: &str, dialect: PropertyDialect) -> Option<DirectVariable> {
    match dialect {
        PropertyDialect::ScalarV1 => match token {
            "X_0" => Some(DirectVariable::Input(0)),
            "X_1" => Some(DirectVariable::Input(1)),
            "Y_0" => Some(DirectVariable::Output),
            _ => None,
        },
        PropertyDialect::TensorV2 => match token {
            "X[0,0]" => Some(DirectVariable::Input(0)),
            "X[0,1]" => Some(DirectVariable::Input(1)),
            "Y[0,0]" => Some(DirectVariable::Output),
            _ => None,
        },
    }
}

fn cross_check_ordinary_spec(
    spec: &VnnLibSpec,
    raw: &RawProperty<'_>,
    input_bounds: [(f64, f64); 2],
) -> Option<()> {
    if spec.num_inputs != 2
        || spec.num_outputs != 1
        || spec.is_disjunction
        || spec.dual_network.is_some()
        || spec.output_constraints.len() != 1
        || spec.output_constraint_clauses.len() != 1
        || spec.output_constraint_clauses[0].len() != 1
        || spec.output_constraint_clauses[0][0] != spec.output_constraints[0]
        || spec
            .per_clause_input_bounds
            .iter()
            .any(|bounds| !bounds.is_empty())
    {
        return None;
    }
    if spec.input_bounds.len() != 2 {
        return None;
    }
    for index in 0..2 {
        let (ordinary_lower, ordinary_upper) = spec.input_bounds[index];
        if !ordinary_lower.is_finite()
            || !ordinary_upper.is_finite()
            || ordinary_lower > ordinary_upper
            || ordinary_lower.to_bits() != raw.lower[index].nearest.to_bits()
            || ordinary_upper.to_bits() != raw.upper[index].nearest.to_bits()
            || input_bounds[index].0 > ordinary_lower
            || input_bounds[index].1 < ordinary_upper
        {
            return None;
        }
    }
    let (ordinary_direction, ordinary_strict, ordinary_value) =
        match spec.output_constraints.as_slice() {
            [OutputConstraint::LessEqConst(0, value)] => (ThreshDir::Le, false, *value),
            [OutputConstraint::LessThanConst(0, value)] => (ThreshDir::Le, true, *value),
            [OutputConstraint::GreaterEqConst(0, value)] => (ThreshDir::Ge, false, *value),
            [OutputConstraint::GreaterThanConst(0, value)] => (ThreshDir::Ge, true, *value),
            _ => return None,
        };
    let ordinary_literal = raw.threshold.literal.parse::<f64>().ok()?;
    if ordinary_direction != raw.threshold.direction
        || ordinary_strict != raw.threshold.strict
        || !ordinary_value.is_finite()
        || ordinary_value.to_bits() != ordinary_literal.to_bits()
    {
        return None;
    }
    let (lower, upper) = raw.threshold.decimal.outward_f64()?;
    if ordinary_value < lower || ordinary_value > upper {
        return None;
    }
    Some(())
}

fn parse_expressions<'a>(tokens: &[&'a str]) -> Option<Vec<RawExpr<'a>>> {
    let mut position = 0usize;
    let mut expressions = Vec::new();
    while position < tokens.len() {
        let (expression, next) = parse_expression(tokens, position, 0)?;
        expressions.push(expression);
        position = next;
    }
    Some(expressions)
}

#[derive(Clone, Debug)]
enum RawExpr<'a> {
    Atom(&'a str),
    List(Vec<RawExpr<'a>>),
}

impl RawExpr<'_> {
    fn list(&self) -> Option<&[Self]> {
        match self {
            Self::List(items) => Some(items),
            Self::Atom(_) => None,
        }
    }
}

fn parse_expression<'a>(
    tokens: &[&'a str],
    position: usize,
    depth: usize,
) -> Option<(RawExpr<'a>, usize)> {
    if depth > MAX_NESTING {
        return None;
    }
    match *tokens.get(position)? {
        "(" => {
            let mut items = Vec::new();
            let mut next = position.checked_add(1)?;
            while tokens.get(next).copied()? != ")" {
                let (item, after) = parse_expression(tokens, next, depth + 1)?;
                items.push(item);
                next = after;
            }
            Some((RawExpr::List(items), next.checked_add(1)?))
        }
        ")" => None,
        atom => Some((RawExpr::Atom(atom), position.checked_add(1)?)),
    }
}

fn tokenize(content: &str) -> Option<Vec<&str>> {
    let bytes = content.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            byte if byte.is_ascii_whitespace() => index += 1,
            b';' => {
                while index < bytes.len() && bytes[index] != b'\n' {
                    index += 1;
                }
            }
            b'(' | b')' => {
                tokens.push(&content[index..=index]);
                index += 1;
            }
            b'"' | b'|' => return None,
            _ => {
                let start = index;
                while index < bytes.len()
                    && !bytes[index].is_ascii_whitespace()
                    && !matches!(bytes[index], b'(' | b')' | b';')
                {
                    if matches!(bytes[index], b'"' | b'|') {
                        return None;
                    }
                    index += 1;
                }
                tokens.push(content.get(start..index)?);
            }
        }
        if tokens.len() > MAX_TOKENS {
            return None;
        }
    }
    Some(tokens)
}

/// Exact bounded decimal represented as `significand * 10^exponent10`.
#[derive(Clone, Debug)]
struct ExactDecimal {
    negative: bool,
    significand: BigNat,
    exponent10: i32,
    nearest: f64,
}

impl ExactDecimal {
    fn parse(token: &str) -> Option<Self> {
        if token.is_empty() || token.len() > MAX_DECIMAL_DIGITS + 16 {
            return None;
        }
        let (negative, unsigned) = if let Some(rest) = token.strip_prefix('-') {
            (true, rest)
        } else if let Some(rest) = token.strip_prefix('+') {
            (false, rest)
        } else {
            (false, token)
        };
        let mut exponent_parts = unsigned.split(['e', 'E']);
        let mantissa = exponent_parts.next()?;
        let exponent_text = exponent_parts.next();
        if exponent_parts.next().is_some() {
            return None;
        }
        let exponent = match exponent_text {
            Some(text)
                if !text.is_empty()
                    && text
                        .strip_prefix(['+', '-'])
                        .unwrap_or(text)
                        .bytes()
                        .all(|byte| byte.is_ascii_digit()) =>
            {
                text.parse::<i32>().ok()?
            }
            Some(_) => return None,
            None => 0,
        };
        if exponent.unsigned_abs() > MAX_DECIMAL_EXPONENT as u32 {
            return None;
        }
        let mut mantissa_parts = mantissa.split('.');
        let integer = mantissa_parts.next()?;
        let fraction = mantissa_parts.next().unwrap_or_default();
        if mantissa_parts.next().is_some()
            || (integer.is_empty() && fraction.is_empty())
            || !integer.bytes().all(|byte| byte.is_ascii_digit())
            || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        {
            return None;
        }
        let digits_len = integer.len().checked_add(fraction.len())?;
        if digits_len == 0 || digits_len > MAX_DECIMAL_DIGITS {
            return None;
        }
        let exponent10 = exponent.checked_sub(i32::try_from(fraction.len()).ok()?)?;
        if exponent10.unsigned_abs() > MAX_DECIMAL_EXPONENT as u32 {
            return None;
        }
        let mut significand = BigNat::zero();
        for byte in integer.bytes().chain(fraction.bytes()) {
            significand.mul_small(10)?;
            significand.add_small(u32::from(byte - b'0'))?;
        }
        let nearest = token.parse::<f64>().ok()?;
        if !nearest.is_finite() {
            return None;
        }
        Some(Self {
            negative: negative && !significand.is_zero(),
            significand,
            exponent10,
            nearest,
        })
    }

    /// Exact decimal rounded down/up respectively; one endpoint equals the
    /// nearest binary64 and the other is at most one adjacent binary64 away.
    fn outward_f64(&self) -> Option<(f64, f64)> {
        let relation = self.cmp_f64(self.nearest)?;
        let lower = if relation == Ordering::Less {
            self.nearest.next_down()
        } else {
            self.nearest
        };
        let upper = if relation == Ordering::Greater {
            self.nearest.next_up()
        } else {
            self.nearest
        };
        if !lower.is_finite()
            || !upper.is_finite()
            || lower > upper
            || self.cmp_f64(lower)? == Ordering::Less
            || self.cmp_f64(upper)? == Ordering::Greater
        {
            return None;
        }
        Some((lower, upper))
    }

    /// Compare two exact decimals without first converting either to binary64.
    fn cmp_exact(&self, other: &Self) -> Option<Ordering> {
        let self_zero = self.significand.is_zero();
        let other_zero = other.significand.is_zero();
        if self_zero || other_zero {
            return Some(match (self_zero, other_zero) {
                (true, true) => Ordering::Equal,
                (true, false) if other.negative => Ordering::Greater,
                (true, false) => Ordering::Less,
                (false, true) if self.negative => Ordering::Less,
                (false, true) => Ordering::Greater,
                (false, false) => unreachable!("one exact decimal was zero"),
            });
        }
        if self.negative != other.negative {
            return Some(if self.negative {
                Ordering::Less
            } else {
                Ordering::Greater
            });
        }

        let common_exponent = self.exponent10.min(other.exponent10);
        let mut self_magnitude = self.significand.clone();
        let mut other_magnitude = other.significand.clone();
        self_magnitude.mul_pow(10, u32::try_from(self.exponent10 - common_exponent).ok()?)?;
        other_magnitude.mul_pow(10, u32::try_from(other.exponent10 - common_exponent).ok()?)?;
        let magnitude = self_magnitude.cmp(&other_magnitude);
        Some(if self.negative {
            magnitude.reverse()
        } else {
            magnitude
        })
    }

    /// Compare the exact decimal (`self`) to one finite IEEE-754 binary64.
    fn cmp_f64(&self, value: f64) -> Option<Ordering> {
        if !value.is_finite() {
            return None;
        }
        let value_zero = value == 0.0;
        if self.significand.is_zero() {
            return Some(if value_zero {
                Ordering::Equal
            } else if value.is_sign_negative() {
                Ordering::Greater
            } else {
                Ordering::Less
            });
        }
        if value_zero {
            return Some(if self.negative {
                Ordering::Less
            } else {
                Ordering::Greater
            });
        }
        let value_negative = value.is_sign_negative();
        if self.negative != value_negative {
            return Some(if self.negative {
                Ordering::Less
            } else {
                Ordering::Greater
            });
        }
        let (mantissa, exponent2) = f64_magnitude_parts(value.abs())?;
        let mut decimal_side = self.significand.clone();
        let mut binary_side = BigNat::from_u64(mantissa);
        if self.exponent10 >= 0 {
            decimal_side.mul_pow(10, self.exponent10 as u32)?;
        } else {
            binary_side.mul_pow(10, self.exponent10.unsigned_abs())?;
        }
        if exponent2 >= 0 {
            binary_side.mul_pow(2, exponent2 as u32)?;
        } else {
            decimal_side.mul_pow(2, exponent2.unsigned_abs())?;
        }
        let magnitude = decimal_side.cmp(&binary_side);
        Some(if self.negative {
            magnitude.reverse()
        } else {
            magnitude
        })
    }
}

fn f64_magnitude_parts(value: f64) -> Option<(u64, i32)> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let bits = value.to_bits();
    let exponent_bits = i32::try_from((bits >> 52) & 0x7ff).ok()?;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent_bits == 0 {
        Some((fraction, -1074))
    } else {
        Some(((1_u64 << 52) | fraction, exponent_bits - 1023 - 52))
    }
}

/// Minimal nonnegative big integer, base 1e9, sufficient for exact bounded
/// decimal-vs-dyadic comparison without adding a production dependency.
#[derive(Clone, Debug, Eq, PartialEq)]
struct BigNat(Vec<u32>);

impl BigNat {
    const BASE: u64 = 1_000_000_000;

    fn zero() -> Self {
        Self(Vec::new())
    }

    fn from_u64(mut value: u64) -> Self {
        let mut limbs = Vec::new();
        while value != 0 {
            limbs.push((value % Self::BASE) as u32);
            value /= Self::BASE;
        }
        Self(limbs)
    }

    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    fn add_small(&mut self, value: u32) -> Option<()> {
        if value == 0 {
            return Some(());
        }
        let mut carry = u64::from(value);
        let mut index = 0usize;
        while carry != 0 {
            if index == self.0.len() {
                self.0.try_reserve(1).ok()?;
                self.0.push(0);
            }
            let sum = u64::from(self.0[index]).checked_add(carry)?;
            self.0[index] = (sum % Self::BASE) as u32;
            carry = sum / Self::BASE;
            index += 1;
        }
        Some(())
    }

    fn mul_small(&mut self, multiplier: u32) -> Option<()> {
        if self.is_zero() || multiplier == 1 {
            return Some(());
        }
        if multiplier == 0 {
            self.0.clear();
            return Some(());
        }
        let mut carry = 0_u64;
        for limb in &mut self.0 {
            let product = u64::from(*limb)
                .checked_mul(u64::from(multiplier))?
                .checked_add(carry)?;
            *limb = (product % Self::BASE) as u32;
            carry = product / Self::BASE;
        }
        while carry != 0 {
            self.0.try_reserve(1).ok()?;
            self.0.push((carry % Self::BASE) as u32);
            carry /= Self::BASE;
        }
        Some(())
    }

    fn mul_pow(&mut self, base: u32, exponent: u32) -> Option<()> {
        for _ in 0..exponent {
            self.mul_small(base)?;
        }
        Some(())
    }
}

impl Ord for BigNat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .len()
            .cmp(&other.0.len())
            .then_with(|| self.0.iter().rev().cmp(other.0.iter().rev()))
    }
}

impl PartialOrd for BigNat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "external-vnncomp")]
    use std::io::Read;

    #[cfg(feature = "external-vnncomp")]
    use flate2::read::GzDecoder;
    #[cfg(feature = "mip")]
    use num_rational::BigRational;

    use super::*;

    fn property(output: &str) -> String {
        format!(
            "(declare-const X_0 Real)\n\
             (declare-const X_1 Real)\n\
             (declare-const Y_0 Real)\n\
             (assert (<= X_0 2))\n\
             (assert (>= X_0 -2))\n\
             (assert (<= X_1 0.3))\n\
             (assert (>= X_1 -0.1))\n\
             (assert {output})\n"
        )
    }

    fn tensor_property(output: &str) -> String {
        format!(
            "(vnnlib-version <2.0>)\n\
             (declare-network N\n\
               (declare-input X float32 [1, 2])\n\
               (declare-output Y float32 [1, 1]))\n\
             (assert (<= X[0,0] 2))\n\
             (assert (>= X[0,0] -2))\n\
             (assert (<= X[0,1] 0.3))\n\
             (assert (>= X[0,1] -0.1))\n\
             (assert {output})\n"
        )
    }

    fn write_property(content: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("property.vnnlib");
        std::fs::write(&path, content).expect("write property");
        (directory, path)
    }

    #[test]
    fn exact_decimal_rounding_is_directed() {
        let tenth = ExactDecimal::parse("0.1").unwrap();
        let (lo, hi) = tenth.outward_f64().unwrap();
        assert_eq!(lo, 0.1_f64.next_down());
        assert_eq!(hi, 0.1_f64);

        let above_one = ExactDecimal::parse("1.00000000000000001").unwrap();
        assert_eq!(above_one.outward_f64().unwrap(), (1.0, 1.0_f64.next_up()));
        let below_negative_one = ExactDecimal::parse("-1.00000000000000001").unwrap();
        assert_eq!(
            below_negative_one.outward_f64().unwrap(),
            ((-1.0_f64).next_down(), -1.0)
        );
        assert_eq!(
            ExactDecimal::parse("1e-400")
                .unwrap()
                .outward_f64()
                .unwrap(),
            (0.0, 0.0_f64.next_up())
        );
    }

    #[cfg(feature = "mip")]
    #[test]
    fn decimal_comparison_matches_bigrational_oracle() {
        const NUMERATORS: [i64; 13] = [
            -999_999_999_999_999_999,
            -1_000_000_001,
            -11,
            -10,
            -3,
            -1,
            0,
            1,
            3,
            10,
            11,
            1_000_000_001,
            999_999_999_999_999_999,
        ];

        for scale in 0_u32..=18 {
            let denominator = 10_i64.pow(scale);
            for (number_index, numerator) in NUMERATORS.into_iter().enumerate() {
                let token = format!("{numerator}e-{scale}");
                let decimal = ExactDecimal::parse(&token).expect("bounded decimal");
                let exact = BigRational::new(numerator.into(), denominator.into());
                let nearest = token.parse::<f64>().expect("finite nearest decimal");
                for candidate in [nearest.next_down(), nearest, nearest.next_up()] {
                    let candidate_exact =
                        BigRational::from_float(candidate).expect("finite dyadic candidate");
                    assert_eq!(
                        decimal.cmp_f64(candidate),
                        Some(exact.cmp(&candidate_exact)),
                        "token={token}, candidate={candidate:e}"
                    );
                }

                let (lower, upper) = decimal.outward_f64().expect("finite enclosure");
                let lower_exact = BigRational::from_float(lower).expect("finite lower");
                let upper_exact = BigRational::from_float(upper).expect("finite upper");
                assert!(lower_exact <= exact, "lower missed {token}");
                assert!(upper_exact >= exact, "upper missed {token}");

                let other_scale = 18 - scale;
                let other_numerator = NUMERATORS[(number_index + 5) % NUMERATORS.len()];
                let other_token = format!("{other_numerator}e-{other_scale}");
                let other_decimal =
                    ExactDecimal::parse(&other_token).expect("other bounded decimal");
                let other_exact =
                    BigRational::new(other_numerator.into(), 10_i64.pow(other_scale).into());
                assert_eq!(
                    decimal.cmp_exact(&other_decimal),
                    Some(exact.cmp(&other_exact)),
                    "exact comparison: {token} vs {other_token}"
                );
            }
        }
    }

    #[test]
    fn authenticates_outward_box_and_directional_thresholds() {
        let (_directory, path) = write_property(&property("(<= Y_0 1.00000000000000001)"));
        let authenticated = authenticate_raw_tll_property(&path).expect("authenticate <= property");
        assert_eq!(
            authenticated.threshold(),
            (ThreshDir::Le, 1.0_f64.next_up())
        );
        assert_eq!(authenticated.input_bounds()[0], (-2.0, 2.0));
        assert_eq!(
            authenticated.input_bounds()[1],
            (-0.1_f64, 0.3_f64.next_up())
        );

        let (_directory, path) = write_property(&property("(>= Y_0 -1.00000000000000001)"));
        let authenticated = authenticate_raw_tll_property(&path).expect("authenticate >= property");
        assert_eq!(
            authenticated.threshold(),
            (ThreshDir::Ge, (-1.0_f64).next_down())
        );
    }

    #[test]
    fn authenticates_exact_tll_tensor_dialect() {
        let (_directory, path) =
            write_property(&tensor_property("(<= Y[0,0] 1.00000000000000001)"));
        let authenticated = authenticate_raw_tll_property(&path).expect("tensor property");
        assert_eq!(
            authenticated.threshold(),
            (ThreshDir::Le, 1.0_f64.next_up())
        );
        assert_eq!(authenticated.input_bounds()[0], (-2.0, 2.0));
        assert_eq!(
            authenticated.input_bounds()[1],
            (-0.1_f64, 0.3_f64.next_up())
        );
    }

    #[test]
    fn tensor_dialect_declaration_changes_and_mixing_fail_closed() {
        let valid = tensor_property("(<= Y[0,0] 1)");
        for malformed in [
            valid.replace("<2.0>", "<2.1>"),
            valid.replace("declare-network N", "declare-network Other"),
            valid.replace("float32 [1, 2]", "float64 [1, 2]"),
            valid.replace("float32 [1, 2]", "float32 [2]"),
            valid.replace("float32 [1, 1]", "float32 [1, 2]"),
            valid.replace("X[0,0]", "X_0"),
            valid.replace("Y[0,0]", "Y_0"),
            format!("{valid}(assert (<= Y[0,0] 2))\n"),
        ] {
            let (_directory, path) = write_property(&malformed);
            assert!(
                authenticate_raw_tll_property(&path).is_none(),
                "accepted:\n{malformed}"
            );
        }
    }

    #[test]
    fn reversed_and_strict_output_atoms_preserve_semantics() {
        let (_directory, path) = write_property(&property("(< 0.25 Y_0)"));
        let authenticated = authenticate_raw_tll_property(&path).expect("reversed strict atom");
        assert_eq!(authenticated.threshold(), (ThreshDir::Ge, 0.25));
    }

    #[test]
    fn arithmetic_disjunction_aliases_and_extras_fail_closed() {
        for malformed in [
            property("(<= (+ Y_0 1) 2)"),
            property("(or (<= Y_0 1) (>= Y_0 2))"),
            property("(= Y_0 1)"),
            property("(<= Y_0 Y_0)"),
            property("(<= Y_00 1)"),
            format!("{}(assert (<= Y_0 2))\n", property("(<= Y_0 1)")),
        ] {
            let (_directory, path) = write_property(&malformed);
            assert!(
                authenticate_raw_tll_property(&path).is_none(),
                "accepted:\n{malformed}"
            );
        }
    }

    #[test]
    fn duplicate_bounds_declarations_and_compound_inputs_fail_closed() {
        for malformed in [
            property("(<= Y_0 1)").replace(
                "(assert (<= X_0 2))",
                "(assert (<= X_0 2))\n(assert (<= X_0 3))",
            ),
            property("(<= Y_0 1)").replace(
                "(declare-const X_0 Real)",
                "(declare-const X_0 Real)\n(declare-const X_0 Real)",
            ),
            property("(<= Y_0 1)").replace(
                "(assert (<= X_0 2))",
                "(assert (and (<= X_0 2) (>= X_0 -2)))",
            ),
            property("(<= Y_0 1)")
                .replace("(assert (<= X_0 2))", "(assert (<= X_0 1.0))")
                .replace(
                    "(assert (>= X_0 -2))",
                    "(assert (>= X_0 1.00000000000000001))",
                ),
        ] {
            let (_directory, path) = write_property(&malformed);
            assert!(
                authenticate_raw_tll_property(&path).is_none(),
                "accepted:\n{malformed}"
            );
        }
    }

    #[test]
    fn source_seal_detects_replacement_and_compressed_path_declines() {
        let (directory, path) = write_property(&property("(<= Y_0 1)"));
        let authenticated = authenticate_raw_tll_property(&path).expect("authenticate source");
        assert!(authenticated.source_still_matches(&path));
        std::fs::write(&path, property("(<= Y_0 2)")).expect("replace property");
        assert!(!authenticated.source_still_matches(&path));
        let compressed = directory.path().join("property.vnnlib.gz");
        std::fs::write(&compressed, b"not accepted as a materialized property").unwrap();
        assert!(authenticate_raw_tll_property(&compressed).is_none());
    }

    /// Qualify every `.vnnlib`/`.vnnlib.gz` in a requested official directory.
    /// Selecting the external lane without the directory is an actionable
    /// failure, never a vacuous pass.
    #[cfg(feature = "external-vnncomp")]
    #[test]
    fn requested_real_property_directory_is_authenticated() {
        let root = std::env::var("NY_TLL_PROPERTY_FIXTURE_ROOT").expect(
            "external-vnncomp TLL property conformance requires \
             NY_TLL_PROPERTY_FIXTURE_ROOT=/path/to/vnnlib-directory",
        );
        let mut tested = 0usize;
        for entry in std::fs::read_dir(root).expect("read fixture directory") {
            let path = entry.expect("fixture entry").path();
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();
            if name.ends_with(".vnnlib") {
                assert!(
                    authenticate_raw_tll_property(&path).is_some(),
                    "{}",
                    path.display()
                );
                tested += 1;
            } else if name.ends_with(".vnnlib.gz") {
                let mut decoder = GzDecoder::new(std::fs::File::open(&path).expect("open gzip"));
                let mut content = Vec::new();
                decoder
                    .read_to_end(&mut content)
                    .expect("decompress property");
                let directory = tempfile::tempdir().expect("tempdir");
                let materialized = directory.path().join("property.vnnlib");
                std::fs::write(&materialized, content).expect("materialize property");
                assert!(
                    authenticate_raw_tll_property(&materialized).is_some(),
                    "{}",
                    path.display()
                );
                tested += 1;
            }
        }
        assert!(tested > 0, "no VNN-LIB fixtures found");
    }
}
