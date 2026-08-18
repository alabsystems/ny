// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact-decimal extraction for a fail-closed VNN-LIB input box.
//!
//! The ordinary VNN-LIB parser stores numeric literals as nearest `f64`
//! values.  That is appropriate for the existing execution paths, but a
//! proof-arithmetic constructor cannot claim that those dyadics enclose the
//! exact SMT-LIB decimal.  This module reparses the deliberately small direct
//! input-bound surface as exact rationals and rounds each endpoint outward to
//! `f64`.
//!
//! Only top-level, axis-aligned VNN-LIB 1.0 atoms are accepted:
//!
//! ```text
//! (assert (<= X_0 0.25))
//! (assert (>= X_0 0.10))
//! ```
//!
//! Reversed operands and `=`, `<`, and `>` are also recognized.  Any compound
//! assertion that mentions an input, any clause-scoped input bound, a missing
//! endpoint, or a non-finite outward endpoint fails closed.  The companion
//! affine-property API below additionally authenticates a deliberately tiny
//! output surface directly from this raw syntax tree.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use flate2::read::GzDecoder;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::ToPrimitive;
use ny_core::{NyError, Result};

use super::syntax::{strip_vnnlib_comments, tokenize};
use super::{parse_vnnlib, OutputConstraint, VnnLibSpec};

const MAX_RAW_NESTING: usize = 256;
const MAX_CERTIFIED_SOURCE_BYTES: usize = 64 * 1024 * 1024;
const MAX_RAW_TOKENS: usize = 16 * 1024 * 1024;
const MAX_DECIMAL_DIGITS: usize = 4_096;
const MAX_DECIMAL_SCALE: i64 = 4_096;

/// An exact-decimal input box represented by finite outward `f64` endpoints.
///
/// `declared_point` records exact-rational equality of the tightest lower and
/// upper atoms.  It is only a decomposition hint: consumers must retain the
/// complete `[lower, upper]` enclosure even when the bit is set.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedInputBox {
    lower: Vec<f64>,
    upper: Vec<f64>,
    declared_point: Vec<bool>,
    center_hi: Vec<f64>,
    center_lo: Vec<f64>,
    center_err: Vec<f64>,
    half_width: Vec<f64>,
}

/// Exact-decimal thresholds for a deliberately narrow scalar unsafe moat.
///
/// The admitted VNN-LIB surface has exactly five globally bounded scalar
/// inputs, one output (`Y_0`), and exactly two singleton unsafe disjuncts:
///
/// ```text
/// (or (and (>= Y_0 HIGH)) (and (<= Y_0 LOW)))
/// ```
///
/// `high_lower` is `HIGH` rounded toward negative infinity and `low_upper` is
/// `LOW` rounded toward positive infinity.  Those directions preserve the
/// closed unsafe set when a future proof lane compares an output enclosure
/// against the moat.  This type is extraction-only: it has no verdict method
/// and is not consulted by the verifier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CertifiedScalarMoat {
    high_lower: f64,
    low_upper: f64,
}

/// One source-authenticated, constant-free output comparison.
///
/// The operands were direct canonical scalar names in the VNN-LIB source. No
/// ordinary-parser normalization, decimal arithmetic, alias, or affine
/// cancellation can create a value of this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CertifiedRelationalOutputAtom {
    /// A direct `(<= Y_i Y_j)` atom.
    LessEq(usize, usize),
    /// A direct `(>= Y_i Y_j)` atom.
    GreaterEq(usize, usize),
    /// A direct `(< Y_i Y_j)` atom.
    LessThan(usize, usize),
    /// A direct `(> Y_i Y_j)` atom.
    GreaterThan(usize, usize),
}

/// A source-authenticated conjunction of direct relational output atoms.
///
/// Each output assertion must be either one direct atom or one flat `(and
/// ...)` whose children are direct atoms. Multiple assertions are themselves
/// conjunctive. Equality is intentionally not expanded in this v1 surface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertifiedRelationalOutputConjunction {
    atoms: Vec<CertifiedRelationalOutputAtom>,
}

impl CertifiedRelationalOutputConjunction {
    /// Direct output atoms, in source assertion order.
    #[must_use]
    pub fn atoms(&self) -> &[CertifiedRelationalOutputAtom] {
        &self.atoms
    }

    /// Number of authenticated output atoms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    /// Whether the conjunction contains no output atom.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }
}

impl CertifiedScalarMoat {
    /// `HIGH` rounded toward negative infinity.
    #[must_use]
    pub fn high_lower(self) -> f64 {
        self.high_lower
    }

    /// `LOW` rounded toward positive infinity.
    #[must_use]
    pub fn low_upper(self) -> f64 {
        self.low_upper
    }
}

impl CertifiedInputBox {
    /// Finite lower endpoints rounded toward negative infinity.
    #[must_use]
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    /// Finite upper endpoints rounded toward positive infinity.
    #[must_use]
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }

    /// Exact-rational equality hints, one per input coordinate.
    #[must_use]
    pub fn declared_point(&self) -> &[bool] {
        &self.declared_point
    }

    /// Number of input coordinates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lower.len()
    }

    /// Whether the box has no coordinates.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lower.is_empty()
    }

    /// Consume the box into `(lower, upper, declared_point)` vectors.
    #[must_use]
    pub fn into_parts(self) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
        (self.lower, self.upper, self.declared_point)
    }

    /// Leading word of the DOUBLE-DOUBLE exact box center, per coordinate.
    ///
    /// # Why a second word exists (`#dd-zonotope`)
    ///
    /// [`Self::lower`] / [`Self::upper`] are outward `f64` endpoints, so for a
    /// non-dyadic decimal such as `2.6399001` a declared POINT arrives as a
    /// one-ulp-wide bracket (`~4.4e-16`). Any consumer that must treat the
    /// fixed coordinates as a point therefore inherits `~2.2e-16` of interval
    /// uncertainty per coordinate.
    ///
    /// MEASURED on `vgg16-7` spec1: propagated rigorously through the network
    /// that residue reaches the logits at a certified half-width of `53.4`, on
    /// a true margin of `1.6375` — vacuous. The end-to-end amplification is
    /// `~2.4e17`, so the center must be represented to better than `~7e-20` for
    /// a 1%-of-margin certificate.
    ///
    /// `center_hi + center_lo` is a double-double (`~106` bit) approximation of
    /// the EXACT rational center, and [`Self::center_err`] bounds the residual.
    /// That residual is `~|x| * 2^-106 ~ 3e-32`, i.e. `~7.7e-15` at the logits
    /// under the same amplification — 14 orders of headroom.
    #[must_use]
    pub fn center_hi(&self) -> &[f64] {
        &self.center_hi
    }

    /// Trailing word of the double-double exact box center.
    #[must_use]
    pub fn center_lo(&self) -> &[f64] {
        &self.center_lo
    }

    /// Nonnegative bound on `|exact_center - (center_hi + center_lo)|`,
    /// rounded outward.
    #[must_use]
    pub fn center_err(&self) -> &[f64] {
        &self.center_err
    }

    /// Exact half-width `(upper - lower) / 2` rounded OUTWARD to `f64`.
    ///
    /// This is computed from the exact rationals, so a declared point has
    /// half-width exactly `0.0` — unlike `upper() - lower()`, which carries the
    /// outward-`f64` bracket of a non-dyadic decimal.
    #[must_use]
    pub fn half_width(&self) -> &[f64] {
        &self.half_width
    }
}

/// Load the ordinary VNN-LIB specification and its direct exact-decimal input
/// box from the same file contents.
///
/// This intentionally does not use the ordinary parse memo: the exact raw
/// decimal spellings are not stored in [`VnnLibSpec`].  The scored verifier is
/// not wired to this API yet.
pub fn load_vnnlib_with_certified_input_box<P: AsRef<Path>>(
    path: P,
) -> Result<(VnnLibSpec, CertifiedInputBox)> {
    let content = read_certified_source_maybe_gzip(path.as_ref())?;
    parse_vnnlib_with_certified_input_box(&content)
}

/// Parse an ordinary VNN-LIB specification and independently extract a direct
/// exact-decimal input box from `content`.
pub fn parse_vnnlib_with_certified_input_box(
    content: &str,
) -> Result<(VnnLibSpec, CertifiedInputBox)> {
    check_certified_source_size(content.len())?;
    preflight_certified_syntax(content)?;
    let spec = parse_vnnlib(content)?;
    let box_ = extract_certified_input_box(content, &spec)?;
    Ok((spec, box_))
}

/// Load the ordinary specification together with the two source-certified
/// surfaces used by the strict affine proof lane.
///
/// In addition to a direct exact-decimal input box, every output assertion must
/// be a conjunction of direct `Y_i`-to-`Y_j` comparisons. Numeric literals,
/// arithmetic expressions, equality expansion, disjunction, mixed input/output
/// assertions, and other output syntax fail closed. The extracted atoms are
/// cross-checked against the ordinary parser's complete single-clause shape.
pub fn load_vnnlib_with_certified_affine_property<P: AsRef<Path>>(
    path: P,
) -> Result<(
    VnnLibSpec,
    CertifiedInputBox,
    CertifiedRelationalOutputConjunction,
)> {
    let content = read_certified_source_maybe_gzip(path.as_ref())?;
    parse_vnnlib_with_certified_affine_property(&content)
}

/// Parse the strict source-certified affine-property surface from `content`.
pub fn parse_vnnlib_with_certified_affine_property(
    content: &str,
) -> Result<(
    VnnLibSpec,
    CertifiedInputBox,
    CertifiedRelationalOutputConjunction,
)> {
    check_certified_source_size(content.len())?;
    preflight_certified_syntax(content)?;
    let spec = parse_vnnlib(content)?;
    let box_ = extract_certified_input_box(content, &spec)?;
    let outputs = extract_certified_relational_output_conjunction(content, &spec)?;
    Ok((spec, box_, outputs))
}

/// Load the ordinary specification, its exact-decimal input box, and a narrow
/// scalar output moat from the same file contents.
///
/// This parser is intentionally unwired: constructing a
/// [`CertifiedScalarMoat`] cannot produce or alter a verifier verdict.
pub fn load_vnnlib_with_certified_scalar_moat<P: AsRef<Path>>(
    path: P,
) -> Result<(VnnLibSpec, CertifiedInputBox, CertifiedScalarMoat)> {
    let content = read_certified_source_maybe_gzip(path.as_ref())?;
    parse_vnnlib_with_certified_scalar_moat(&content)
}

/// Parse the ordinary specification, exact-decimal input box, and narrow
/// scalar output moat from `content`.
pub fn parse_vnnlib_with_certified_scalar_moat(
    content: &str,
) -> Result<(VnnLibSpec, CertifiedInputBox, CertifiedScalarMoat)> {
    check_certified_source_size(content.len())?;
    preflight_certified_syntax(content)?;
    let spec = parse_vnnlib(content)?;
    let box_ = extract_certified_input_box(content, &spec)?;
    if box_.len() != 5 || spec.num_inputs != 5 {
        return Err(NyError::InvalidSpec(
            "certified scalar moat requires exactly five scalar inputs".to_string(),
        ));
    }
    let moat = extract_certified_scalar_moat(content, &spec)?;
    Ok((spec, box_, moat))
}

#[derive(Debug)]
enum RawExpr<'a> {
    Atom(&'a str),
    List(Vec<Self>),
}

fn check_certified_source_size(source_bytes: usize) -> Result<()> {
    if source_bytes > MAX_CERTIFIED_SOURCE_BYTES {
        return Err(NyError::InvalidSpec(format!(
            "certified input source exceeds the {MAX_CERTIFIED_SOURCE_BYTES}-byte cap"
        )));
    }
    Ok(())
}

fn check_raw_token_count(token_count: usize) -> Result<()> {
    if token_count > MAX_RAW_TOKENS {
        return Err(NyError::InvalidSpec(format!(
            "certified input token count exceeds the {MAX_RAW_TOKENS}-token cap"
        )));
    }
    Ok(())
}

fn read_certified_source_maybe_gzip(path: &Path) -> Result<String> {
    let file = File::open(path).map_err(|error| {
        NyError::ModelLoad(format!(
            "Failed to open certified VNN-LIB {}: {error}",
            path.display()
        ))
    })?;
    if path.extension().and_then(|extension| extension.to_str()) == Some("gz") {
        read_certified_source(GzDecoder::new(file), path)
    } else {
        read_certified_source(file, path)
    }
}

fn read_certified_source(mut reader: impl Read, path: &Path) -> Result<String> {
    // Read through a fixed stack buffer and reserve only after checking the
    // resulting logical length. In particular, a gzip expansion can never
    // accumulate more than MAX_CERTIFIED_SOURCE_BYTES on the heap.
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 16 * 1024];
    loop {
        let count = reader.read(&mut chunk).map_err(|error| {
            NyError::ModelLoad(format!(
                "Failed to read certified VNN-LIB {}: {error}",
                path.display()
            ))
        })?;
        if count == 0 {
            break;
        }
        let next_len = bytes.len().checked_add(count).ok_or_else(|| {
            NyError::InvalidSpec("certified input source byte count overflow".to_string())
        })?;
        check_certified_source_size(next_len)?;
        bytes
            .try_reserve(count)
            .map_err(|_| allocation_error("certified source bytes"))?;
        bytes.extend_from_slice(&chunk[..count]);
    }
    String::from_utf8(bytes).map_err(|error| {
        NyError::ModelLoad(format!(
            "Failed to decode certified VNN-LIB {} as UTF-8: {error}",
            path.display()
        ))
    })
}

fn preflight_certified_syntax(content: &str) -> Result<()> {
    // Mirror strip_vnnlib_comments + tokenize without building either the
    // token vector or a recursive expression tree. This bounds the legacy
    // parser before it can recurse on adversarial input.
    let mut depth = 0_usize;
    let mut token_count = 0_usize;
    let mut in_atom = false;
    let mut in_comment = false;
    let mut in_string = false;

    for character in content.chars() {
        if in_comment {
            if character == '\n' {
                in_comment = false;
            }
            continue;
        }
        if in_string {
            if character == '"' {
                in_string = false;
                record_preflight_token(&mut token_count)?;
            }
            continue;
        }

        match character {
            ';' => {
                finish_preflight_atom(&mut in_atom, &mut token_count)?;
                in_comment = true;
            }
            '"' => {
                finish_preflight_atom(&mut in_atom, &mut token_count)?;
                in_string = true;
            }
            '(' => {
                finish_preflight_atom(&mut in_atom, &mut token_count)?;
                record_preflight_token(&mut token_count)?;
                depth = depth.checked_add(1).ok_or_else(|| {
                    NyError::InvalidSpec(
                        "certified input parser nesting depth overflow".to_string(),
                    )
                })?;
                if depth > MAX_RAW_NESTING {
                    return Err(NyError::InvalidSpec(format!(
                        "certified input parser nesting exceeds {MAX_RAW_NESTING}"
                    )));
                }
            }
            ')' => {
                finish_preflight_atom(&mut in_atom, &mut token_count)?;
                record_preflight_token(&mut token_count)?;
                depth = depth.checked_sub(1).ok_or_else(|| {
                    NyError::InvalidSpec(
                        "certified input parser found an unmatched ')'".to_string(),
                    )
                })?;
            }
            ' ' | '\t' | '\n' | '\r' => {
                finish_preflight_atom(&mut in_atom, &mut token_count)?;
            }
            _ => in_atom = true,
        }
    }
    if in_string {
        record_preflight_token(&mut token_count)?;
    } else {
        finish_preflight_atom(&mut in_atom, &mut token_count)?;
    }
    if depth != 0 {
        return Err(NyError::InvalidSpec(
            "certified input parser found an unmatched '('".to_string(),
        ));
    }
    Ok(())
}

fn finish_preflight_atom(in_atom: &mut bool, token_count: &mut usize) -> Result<()> {
    if std::mem::take(in_atom) {
        record_preflight_token(token_count)?;
    }
    Ok(())
}

fn record_preflight_token(token_count: &mut usize) -> Result<()> {
    *token_count = token_count
        .checked_add(1)
        .ok_or_else(|| NyError::InvalidSpec("certified input token count overflow".to_string()))?;
    check_raw_token_count(*token_count)
}

fn extract_certified_input_box(content: &str, spec: &VnnLibSpec) -> Result<CertifiedInputBox> {
    if spec.dual_network.is_some() {
        return Err(NyError::InvalidSpec(
            "certified direct input box does not support dual-network properties".to_string(),
        ));
    }
    if spec
        .per_clause_input_bounds
        .iter()
        .any(|bounds| !bounds.is_empty())
    {
        return Err(NyError::InvalidSpec(
            "certified direct input box does not support clause-scoped input bounds".to_string(),
        ));
    }

    let cleaned = strip_vnnlib_comments(content);
    let tokens = tokenize(&cleaned)?;
    check_raw_token_count(tokens.len())?;
    reject_noncanonical_variable_aliases(&tokens)?;
    let expressions = parse_raw_expressions(&tokens)?;
    let mut lower = none_slots(spec.num_inputs, "certified exact lower endpoints")?;
    let mut upper = none_slots(spec.num_inputs, "certified exact upper endpoints")?;

    for expression in &expressions {
        collect_top_level_assert(expression, &mut lower, &mut upper)?;
    }

    let mut lower_f64 = Vec::new();
    let mut upper_f64 = Vec::new();
    let mut declared_point = Vec::new();
    let mut center_hi = Vec::new();
    let mut center_lo = Vec::new();
    let mut center_err = Vec::new();
    let mut half_width = Vec::new();
    lower_f64
        .try_reserve_exact(spec.num_inputs)
        .map_err(|_| allocation_error("certified lower endpoints"))?;
    upper_f64
        .try_reserve_exact(spec.num_inputs)
        .map_err(|_| allocation_error("certified upper endpoints"))?;
    declared_point
        .try_reserve_exact(spec.num_inputs)
        .map_err(|_| allocation_error("certified point mask"))?;

    for index in 0..spec.num_inputs {
        let exact_lower = lower[index].as_ref().ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "certified direct input box is missing a lower bound for X_{index}"
            ))
        })?;
        let exact_upper = upper[index].as_ref().ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "certified direct input box is missing an upper bound for X_{index}"
            ))
        })?;
        if exact_lower > exact_upper {
            return Err(NyError::InvalidSpec(format!(
                "certified exact lower bound exceeds upper bound for X_{index}"
            )));
        }

        let lo = rational_to_lower_f64(exact_lower, index)?;
        let hi = rational_to_upper_f64(exact_upper, index)?;
        if lo > hi || !lo.is_finite() || !hi.is_finite() {
            return Err(NyError::InvalidSpec(format!(
                "certified outward bounds for X_{index} are not a finite interval"
            )));
        }

        // Cross-check the independently extracted enclosure against the
        // ordinary parser.  This detects a direct-atom spelling that the two
        // parsers interpreted with different input indexing or tightening.
        let Some(&(parsed_lower, parsed_upper)) = spec.input_bounds.get(index) else {
            return Err(NyError::InvalidSpec(format!(
                "ordinary parser omitted X_{index} from the input box"
            )));
        };
        if !parsed_lower.is_finite()
            || !parsed_upper.is_finite()
            || lo > parsed_lower
            || hi < parsed_upper
        {
            return Err(NyError::InvalidSpec(format!(
                "certified direct bounds disagree with ordinary bounds for X_{index}"
            )));
        }

        let two = BigRational::from(BigInt::from(2));
        let exact_center = (exact_lower.clone() + exact_upper.clone()) / two.clone();
        let exact_half = (exact_upper.clone() - exact_lower.clone()) / two;
        let (chi, clo, cerr) = rational_to_double_double(&exact_center, index)?;
        let half = rational_to_upper_f64(&exact_half, index)?;
        if !half.is_finite() || half < 0.0 {
            return Err(conversion_error(index));
        }

        lower_f64.push(lo);
        upper_f64.push(hi);
        declared_point.push(exact_lower == exact_upper);
        center_hi.push(chi);
        center_lo.push(clo);
        center_err.push(cerr);
        half_width.push(half);
    }

    Ok(CertifiedInputBox {
        lower: lower_f64,
        upper: upper_f64,
        declared_point,
        center_hi,
        center_lo,
        center_err,
        half_width,
    })
}

fn extract_certified_relational_output_conjunction(
    content: &str,
    spec: &VnnLibSpec,
) -> Result<CertifiedRelationalOutputConjunction> {
    if spec.dual_network.is_some()
        || spec.is_disjunction
        || spec
            .per_clause_input_bounds
            .iter()
            .any(|bounds| !bounds.is_empty())
    {
        return Err(NyError::InvalidSpec(
            "certified affine output requires one single-network conjunctive region".to_string(),
        ));
    }

    let cleaned = strip_vnnlib_comments(content);
    let tokens = tokenize(&cleaned)?;
    check_raw_token_count(tokens.len())?;
    reject_noncanonical_variable_aliases(&tokens)?;
    let expressions = parse_raw_expressions(&tokens)?;
    let mut atoms = Vec::new();

    for expression in &expressions {
        let RawExpr::List(items) = expression else {
            continue;
        };
        if !matches!(items.first(), Some(RawExpr::Atom(op)) if *op == "assert") {
            continue;
        }
        if items.len() != 2 {
            return Err(certified_output_shape_error());
        }
        let asserted = &items[1];
        let mentions_input = contains_input_reference(asserted);
        let mentions_output = contains_output_reference(asserted);
        match (mentions_input, mentions_output) {
            (true, false) => {
                // `extract_certified_input_box` independently requires this to
                // be a direct input atom and authenticates its exact decimal.
            }
            (false, true) => {
                collect_direct_output_assertion(asserted, spec.num_outputs, &mut atoms)?
            }
            (true, true) => {
                return Err(NyError::InvalidSpec(
                    "certified affine output rejects mixed input/output assertions".to_string(),
                ));
            }
            (false, false) => {
                return Err(NyError::InvalidSpec(
                    "certified affine output rejects assertions without a direct X or Y reference"
                        .to_string(),
                ));
            }
        }
    }

    if atoms.is_empty() {
        return Err(NyError::InvalidSpec(
            "certified affine output requires at least one direct output atom".to_string(),
        ));
    }

    // The independent ordinary parser must have preserved exactly the same
    // one-clause semantics. This catches disagreements in indexing, relation,
    // ordering, duplicate retention, and clause normalization.
    let [ordinary_clause] = spec.output_constraint_clauses.as_slice() else {
        return Err(certified_output_cross_check_error());
    };
    if ordinary_clause.len() != atoms.len()
        || spec.output_constraints.as_slice() != ordinary_clause.as_slice()
        || atoms
            .iter()
            .zip(ordinary_clause)
            .any(|(source, ordinary)| !certified_atom_matches_ordinary(*source, ordinary))
    {
        return Err(certified_output_cross_check_error());
    }

    Ok(CertifiedRelationalOutputConjunction { atoms })
}

fn collect_direct_output_assertion(
    expression: &RawExpr<'_>,
    num_outputs: usize,
    atoms: &mut Vec<CertifiedRelationalOutputAtom>,
) -> Result<()> {
    let RawExpr::List(items) = expression else {
        return Err(certified_output_shape_error());
    };
    if matches!(items.first(), Some(RawExpr::Atom(op)) if *op == "and") {
        if items.len() < 2 {
            return Err(certified_output_shape_error());
        }
        for child in &items[1..] {
            // Deliberately flat: nested Boolean structure is outside this
            // source-authenticated v1 grammar even when logically equivalent.
            atoms.push(parse_direct_output_atom(child, num_outputs)?);
        }
        return Ok(());
    }
    atoms.push(parse_direct_output_atom(expression, num_outputs)?);
    Ok(())
}

fn parse_direct_output_atom(
    expression: &RawExpr<'_>,
    num_outputs: usize,
) -> Result<CertifiedRelationalOutputAtom> {
    let RawExpr::List(items) = expression else {
        return Err(certified_output_shape_error());
    };
    if items.len() != 3 {
        return Err(certified_output_shape_error());
    }
    let operator = atom(&items[0]).ok_or_else(certified_output_shape_error)?;
    let left = atom(&items[1])
        .and_then(parse_output_index)
        .ok_or_else(certified_output_shape_error)?;
    let right = atom(&items[2])
        .and_then(parse_output_index)
        .ok_or_else(certified_output_shape_error)?;
    if left >= num_outputs || right >= num_outputs {
        return Err(NyError::InvalidSpec(format!(
            "certified affine output atom references an undeclared output (Y_{left}, Y_{right})"
        )));
    }
    match operator {
        "<=" => Ok(CertifiedRelationalOutputAtom::LessEq(left, right)),
        ">=" => Ok(CertifiedRelationalOutputAtom::GreaterEq(left, right)),
        "<" => Ok(CertifiedRelationalOutputAtom::LessThan(left, right)),
        ">" => Ok(CertifiedRelationalOutputAtom::GreaterThan(left, right)),
        _ => Err(certified_output_shape_error()),
    }
}

fn certified_atom_matches_ordinary(
    source: CertifiedRelationalOutputAtom,
    ordinary: &OutputConstraint,
) -> bool {
    matches!(
        (source, ordinary),
        (
            CertifiedRelationalOutputAtom::LessEq(i, j),
            OutputConstraint::LessEq(oi, oj)
        ) | (
            CertifiedRelationalOutputAtom::GreaterEq(i, j),
            OutputConstraint::GreaterEq(oi, oj)
        ) | (
            CertifiedRelationalOutputAtom::LessThan(i, j),
            OutputConstraint::LessThan(oi, oj)
        ) | (
            CertifiedRelationalOutputAtom::GreaterThan(i, j),
            OutputConstraint::GreaterThan(oi, oj)
        ) if i == *oi && j == *oj
    )
}

fn certified_output_shape_error() -> NyError {
    NyError::InvalidSpec(
        "certified affine output requires direct `Y_i`-to-`Y_j` comparison atoms in a flat conjunction"
            .to_string(),
    )
}

fn certified_output_cross_check_error() -> NyError {
    NyError::InvalidSpec(
        "certified affine output disagrees with the ordinary output-clause parse".to_string(),
    )
}

fn extract_certified_scalar_moat(content: &str, spec: &VnnLibSpec) -> Result<CertifiedScalarMoat> {
    if spec.dual_network.is_some() || spec.num_outputs != 1 {
        return Err(NyError::InvalidSpec(
            "certified scalar moat requires exactly one output Y_0".to_string(),
        ));
    }
    if spec
        .per_clause_input_bounds
        .iter()
        .any(|bounds| !bounds.is_empty())
    {
        return Err(NyError::InvalidSpec(
            "certified scalar moat does not support clause-scoped input bounds".to_string(),
        ));
    }

    let cleaned = strip_vnnlib_comments(content);
    let tokens = tokenize(&cleaned)?;
    check_raw_token_count(tokens.len())?;
    reject_noncanonical_variable_aliases(&tokens)?;
    let expressions = parse_raw_expressions(&tokens)?;
    let mut declared_inputs = [false; 5];
    let mut declared_output = false;
    let mut output_assertion = None;

    for expression in &expressions {
        let RawExpr::List(items) = expression else {
            return Err(NyError::InvalidSpec(
                "certified scalar moat requires list-shaped top-level forms".to_string(),
            ));
        };
        match atom(items.first().ok_or_else(scalar_moat_top_level_error)?) {
            Some("declare-const") => {
                collect_scalar_moat_declaration(items, &mut declared_inputs, &mut declared_output)?;
            }
            Some("assert") => {
                if items.len() != 2 {
                    return Err(NyError::InvalidSpec(
                        "certified scalar moat found a malformed top-level assert".to_string(),
                    ));
                }
                let asserted = &items[1];
                if contains_output_reference(asserted) {
                    if output_assertion.replace(asserted).is_some() {
                        return Err(NyError::InvalidSpec(
                            "certified scalar moat requires exactly one output assertion"
                                .to_string(),
                        ));
                    }
                } else if !contains_input_reference(asserted) {
                    return Err(NyError::InvalidSpec(
                        "certified scalar moat found an unsupported non-input assertion"
                            .to_string(),
                    ));
                }
            }
            _ => {
                return Err(NyError::InvalidSpec(
                    "certified scalar moat found an unsupported top-level form".to_string(),
                ));
            }
        }
    }
    if declared_inputs.iter().any(|declared| !declared) || !declared_output {
        return Err(NyError::InvalidSpec(
            "certified scalar moat requires declarations for exactly X_0..X_4 and Y_0".to_string(),
        ));
    }

    let asserted = output_assertion.ok_or_else(|| {
        NyError::InvalidSpec(
            "certified scalar moat requires exactly one output assertion".to_string(),
        )
    })?;
    let (high, low) = parse_exact_scalar_moat_assertion(asserted)?;
    if low >= high {
        return Err(NyError::InvalidSpec(
            "certified scalar moat requires LOW < HIGH exactly".to_string(),
        ));
    }

    cross_check_scalar_moat_with_ordinary_spec(spec, &high, &low)?;
    let high_lower = rational_to_lower_f64(&high, 0).map_err(|_| {
        NyError::InvalidSpec(
            "certified scalar moat HIGH has no finite downward f64 rounding".to_string(),
        )
    })?;
    let low_upper = rational_to_upper_f64(&low, 0).map_err(|_| {
        NyError::InvalidSpec(
            "certified scalar moat LOW has no finite upward f64 rounding".to_string(),
        )
    })?;
    if !high_lower.is_finite() || !low_upper.is_finite() || low_upper >= high_lower {
        return Err(NyError::InvalidSpec(
            "certified scalar moat has no finite, nonempty outward-f64 gap".to_string(),
        ));
    }

    Ok(CertifiedScalarMoat {
        high_lower,
        low_upper,
    })
}

fn collect_scalar_moat_declaration(
    items: &[RawExpr<'_>],
    declared_inputs: &mut [bool; 5],
    declared_output: &mut bool,
) -> Result<()> {
    if items.len() != 3 || atom(&items[2]) != Some("Real") {
        return Err(NyError::InvalidSpec(
            "certified scalar moat requires scalar Real declarations".to_string(),
        ));
    }
    let name = atom(&items[1]).ok_or_else(scalar_moat_top_level_error)?;
    if let Some(index) = parse_input_index(name) {
        let declared = declared_inputs.get_mut(index).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "certified scalar moat declaration references unexpected X_{index}"
            ))
        })?;
        if std::mem::replace(declared, true) {
            return Err(NyError::InvalidSpec(format!(
                "certified scalar moat has a duplicate X_{index} declaration"
            )));
        }
        return Ok(());
    }
    if name == "Y_0" {
        if std::mem::replace(declared_output, true) {
            return Err(NyError::InvalidSpec(
                "certified scalar moat has a duplicate Y_0 declaration".to_string(),
            ));
        }
        return Ok(());
    }
    Err(NyError::InvalidSpec(format!(
        "certified scalar moat found an unsupported declaration `{name}`"
    )))
}

fn parse_exact_scalar_moat_assertion(asserted: &RawExpr<'_>) -> Result<(BigRational, BigRational)> {
    let RawExpr::List(items) = asserted else {
        return Err(scalar_moat_shape_error());
    };
    if items.len() != 3 || !matches!(items.first(), Some(RawExpr::Atom(op)) if *op == "or") {
        return Err(scalar_moat_shape_error());
    }

    let mut high = None;
    let mut low = None;
    for disjunct in &items[1..] {
        let RawExpr::List(conjunction) = disjunct else {
            return Err(scalar_moat_shape_error());
        };
        if conjunction.len() != 2
            || !matches!(conjunction.first(), Some(RawExpr::Atom(op)) if *op == "and")
        {
            return Err(scalar_moat_shape_error());
        }
        let RawExpr::List(atom_items) = &conjunction[1] else {
            return Err(scalar_moat_shape_error());
        };
        if atom_items.len() != 3 {
            return Err(scalar_moat_shape_error());
        }
        let Some(operator) = atom(&atom_items[0]) else {
            return Err(scalar_moat_shape_error());
        };
        if matches!(operator, "<" | ">") {
            return Err(NyError::InvalidSpec(
                "certified scalar moat requires non-strict output comparisons".to_string(),
            ));
        }
        if !matches!(operator, "<=" | ">=")
            || atom(&atom_items[1]) != Some("Y_0")
            || atom(&atom_items[2]).is_none()
        {
            return Err(scalar_moat_shape_error());
        }
        let value = parse_exact_decimal(atom(&atom_items[2]).expect("checked as an atom"))?;
        let slot = if operator == ">=" {
            &mut high
        } else {
            &mut low
        };
        if slot.replace(value).is_some() {
            return Err(NyError::InvalidSpec(
                "certified scalar moat has a duplicate output disjunct".to_string(),
            ));
        }
    }

    match (high, low) {
        (Some(high), Some(low)) => Ok((high, low)),
        _ => Err(NyError::InvalidSpec(
            "certified scalar moat requires one high and one low disjunct".to_string(),
        )),
    }
}

fn cross_check_scalar_moat_with_ordinary_spec(
    spec: &VnnLibSpec,
    high: &BigRational,
    low: &BigRational,
) -> Result<()> {
    if !spec.is_disjunction
        || spec.output_constraints.len() != 2
        || spec.output_constraint_clauses.len() != 2
        || spec
            .output_constraint_clauses
            .iter()
            .any(|clause| clause.len() != 1)
    {
        return Err(NyError::InvalidSpec(
            "certified scalar moat disagrees with the ordinary output-clause shape".to_string(),
        ));
    }

    let mut ordinary_high = None;
    let mut ordinary_low = None;
    for clause in &spec.output_constraint_clauses {
        match clause.as_slice() {
            [OutputConstraint::GreaterEqConst(0, value)] => {
                if ordinary_high.replace(*value).is_some() {
                    return Err(scalar_moat_cross_check_error());
                }
            }
            [OutputConstraint::LessEqConst(0, value)] => {
                if ordinary_low.replace(*value).is_some() {
                    return Err(scalar_moat_cross_check_error());
                }
            }
            _ => return Err(scalar_moat_cross_check_error()),
        }
    }

    let ordinary_high = ordinary_high.ok_or_else(scalar_moat_cross_check_error)?;
    let ordinary_low = ordinary_low.ok_or_else(scalar_moat_cross_check_error)?;
    if !ordinary_high.is_finite() || !ordinary_low.is_finite() {
        return Err(scalar_moat_cross_check_error());
    }
    let ordinary_high =
        BigRational::from_float(ordinary_high).ok_or_else(scalar_moat_cross_check_error)?;
    let ordinary_low =
        BigRational::from_float(ordinary_low).ok_or_else(scalar_moat_cross_check_error)?;
    let high_floor = rational_to_lower_f64(high, 0).map_err(|_| scalar_moat_cross_check_error())?;
    let high_ceil = rational_to_upper_f64(high, 0).map_err(|_| scalar_moat_cross_check_error())?;
    let low_floor = rational_to_lower_f64(low, 0).map_err(|_| scalar_moat_cross_check_error())?;
    let low_ceil = rational_to_upper_f64(low, 0).map_err(|_| scalar_moat_cross_check_error())?;
    let high_floor =
        BigRational::from_float(high_floor).ok_or_else(scalar_moat_cross_check_error)?;
    let high_ceil = BigRational::from_float(high_ceil).ok_or_else(scalar_moat_cross_check_error)?;
    let low_floor = BigRational::from_float(low_floor).ok_or_else(scalar_moat_cross_check_error)?;
    let low_ceil = BigRational::from_float(low_ceil).ok_or_else(scalar_moat_cross_check_error)?;
    if ordinary_high < high_floor
        || ordinary_high > high_ceil
        || ordinary_low < low_floor
        || ordinary_low > low_ceil
    {
        return Err(scalar_moat_cross_check_error());
    }
    Ok(())
}

fn scalar_moat_shape_error() -> NyError {
    NyError::InvalidSpec(
        "certified scalar moat requires exactly two singleton `(and ...)` output disjuncts"
            .to_string(),
    )
}

fn scalar_moat_top_level_error() -> NyError {
    NyError::InvalidSpec("certified scalar moat found a malformed top-level form".to_string())
}

fn scalar_moat_cross_check_error() -> NyError {
    NyError::InvalidSpec(
        "certified scalar moat disagrees with the ordinary output parser".to_string(),
    )
}

/// Decompose an exact rational into a double-double `(hi, lo)` plus an OUTWARD
/// bound on the remaining residual.
///
/// Neither `to_f64` call has to be correctly rounded: each residual is formed
/// in exact rational arithmetic from the value actually produced, so a sloppy
/// conversion can only make `err` larger, never unsound.
fn rational_to_double_double(value: &BigRational, index: usize) -> Result<(f64, f64, f64)> {
    let hi = value.to_f64().ok_or_else(|| conversion_error(index))?;
    if !hi.is_finite() {
        return Err(conversion_error(index));
    }
    let hi_r = BigRational::from_float(hi).ok_or_else(|| conversion_error(index))?;
    let r1 = value.clone() - hi_r;
    let lo = r1.to_f64().ok_or_else(|| conversion_error(index))?;
    if !lo.is_finite() {
        return Err(conversion_error(index));
    }
    let lo_r = BigRational::from_float(lo).ok_or_else(|| conversion_error(index))?;
    let r2 = r1 - lo_r;
    let zero = BigRational::from(BigInt::from(0));
    let abs_r2 = if r2 < zero { -r2 } else { r2 };
    let err = rational_to_upper_f64(&abs_r2, index)?;
    if !err.is_finite() || err < 0.0 {
        return Err(conversion_error(index));
    }
    Ok((hi, lo, err))
}

fn allocation_error(resource: &'static str) -> NyError {
    NyError::InvalidSpec(format!("unable to reserve storage for {resource}"))
}

fn none_slots<T>(len: usize, resource: &'static str) -> Result<Vec<Option<T>>> {
    let mut slots = Vec::new();
    slots
        .try_reserve_exact(len)
        .map_err(|_| allocation_error(resource))?;
    slots.resize_with(len, || None);
    Ok(slots)
}

fn parse_raw_expressions(tokens: &[String]) -> Result<Vec<RawExpr<'_>>> {
    let mut roots = Vec::new();
    let mut stack: Vec<Vec<RawExpr>> = Vec::new();

    for token in tokens {
        match token.as_str() {
            "(" => {
                if stack.len() >= MAX_RAW_NESTING {
                    return Err(NyError::InvalidSpec(format!(
                        "certified input parser nesting exceeds {MAX_RAW_NESTING}"
                    )));
                }
                stack
                    .try_reserve(1)
                    .map_err(|_| allocation_error("certified expression stack"))?;
                stack.push(Vec::new());
            }
            ")" => {
                let items = stack.pop().ok_or_else(|| {
                    NyError::InvalidSpec(
                        "certified input parser found an unmatched ')'".to_string(),
                    )
                })?;
                append_raw(&mut roots, &mut stack, RawExpr::List(items))?;
            }
            atom => append_raw(&mut roots, &mut stack, RawExpr::Atom(atom))?,
        }
    }
    if !stack.is_empty() {
        return Err(NyError::InvalidSpec(
            "certified input parser found an unmatched '('".to_string(),
        ));
    }
    Ok(roots)
}

fn append_raw<'a>(
    roots: &mut Vec<RawExpr<'a>>,
    stack: &mut [Vec<RawExpr<'a>>],
    expression: RawExpr<'a>,
) -> Result<()> {
    let target = if let Some(parent) = stack.last_mut() {
        parent
    } else {
        roots
    };
    target
        .try_reserve(1)
        .map_err(|_| allocation_error("certified raw expressions"))?;
    target.push(expression);
    Ok(())
}

fn collect_top_level_assert(
    expression: &RawExpr,
    lower: &mut [Option<BigRational>],
    upper: &mut [Option<BigRational>],
) -> Result<()> {
    let RawExpr::List(items) = expression else {
        return Ok(());
    };
    if !matches!(items.first(), Some(RawExpr::Atom(op)) if *op == "assert") {
        return Ok(());
    }
    let Some(asserted) = items.get(1) else {
        return Err(NyError::InvalidSpec(
            "certified input parser found an empty assert".to_string(),
        ));
    };
    if items.len() != 2 {
        return Err(NyError::InvalidSpec(
            "certified input parser found a malformed assert".to_string(),
        ));
    }

    if apply_direct_bound(asserted, lower, upper)? {
        return Ok(());
    }
    if contains_input_reference(asserted) {
        return Err(NyError::InvalidSpec(
            "certified input box requires every input assertion to be a direct axis-aligned atom"
                .to_string(),
        ));
    }
    Ok(())
}

fn apply_direct_bound(
    expression: &RawExpr,
    lower: &mut [Option<BigRational>],
    upper: &mut [Option<BigRational>],
) -> Result<bool> {
    let RawExpr::List(items) = expression else {
        return Ok(false);
    };
    if items.len() != 3 {
        return Ok(false);
    }
    let Some(operator) = atom(&items[0]) else {
        return Ok(false);
    };
    if !matches!(operator, "<=" | ">=" | "<" | ">" | "=") {
        return Ok(false);
    }

    let left_input = atom(&items[1]).and_then(parse_input_index);
    let right_input = atom(&items[2]).and_then(parse_input_index);
    let (index, literal, input_on_left) = match (left_input, right_input) {
        (Some(index), None) => (index, atom(&items[2]), true),
        (None, Some(index)) => (index, atom(&items[1]), false),
        (Some(_), Some(_)) => {
            return Err(NyError::InvalidSpec(
                "certified input box does not support relational input atoms".to_string(),
            ));
        }
        (None, None) => return Ok(false),
    };
    if index >= lower.len() {
        return Err(NyError::InvalidSpec(format!(
            "certified input atom references undeclared X_{index}"
        )));
    }
    let literal = literal.ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "certified input atom for X_{index} requires one direct decimal literal"
        ))
    })?;
    let value = parse_exact_decimal(literal)?;

    match (operator, input_on_left) {
        ("<=", true) | ("<", true) | (">=", false) | (">", false) => {
            tighten_upper(&mut upper[index], value);
        }
        (">=", true) | (">", true) | ("<=", false) | ("<", false) => {
            tighten_lower(&mut lower[index], value);
        }
        ("=", _) => {
            tighten_lower(&mut lower[index], value.clone());
            tighten_upper(&mut upper[index], value);
        }
        _ => unreachable!("comparison operator checked above"),
    }
    Ok(true)
}

fn atom<'a>(expression: &RawExpr<'a>) -> Option<&'a str> {
    match expression {
        RawExpr::Atom(value) => Some(*value),
        RawExpr::List(_) => None,
    }
}

fn parse_input_index(atom: &str) -> Option<usize> {
    parse_canonical_variable_index(atom, "X_")
}

fn parse_output_index(atom: &str) -> Option<usize> {
    parse_canonical_variable_index(atom, "Y_")
}

fn parse_canonical_variable_index(atom: &str, prefix: &str) -> Option<usize> {
    let suffix = atom.strip_prefix(prefix)?;
    if suffix.is_empty()
        || (suffix.len() > 1 && suffix.starts_with('0'))
        || !suffix.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    suffix.parse().ok()
}

fn reject_noncanonical_variable_aliases(tokens: &[String]) -> Result<()> {
    for token in tokens {
        let prefix = if token.starts_with("X_") {
            Some("X_")
        } else if token.starts_with("Y_") {
            Some("Y_")
        } else {
            None
        };
        if prefix.is_some_and(|prefix| parse_canonical_variable_index(token, prefix).is_none()) {
            return Err(NyError::InvalidSpec(format!(
                "certified input parser rejects noncanonical variable alias `{token}`"
            )));
        }
    }
    Ok(())
}

fn contains_input_reference(expression: &RawExpr) -> bool {
    match expression {
        RawExpr::Atom(value) => value.starts_with("X_") || value.starts_with("X[") || *value == "X",
        RawExpr::List(items) => items.iter().any(contains_input_reference),
    }
}

fn contains_output_reference(expression: &RawExpr) -> bool {
    match expression {
        RawExpr::Atom(value) => value.starts_with("Y_") || value.starts_with("Y[") || *value == "Y",
        RawExpr::List(items) => items.iter().any(contains_output_reference),
    }
}

fn tighten_lower(slot: &mut Option<BigRational>, value: BigRational) {
    if slot.as_ref().is_none_or(|current| value > *current) {
        *slot = Some(value);
    }
}

fn tighten_upper(slot: &mut Option<BigRational>, value: BigRational) {
    if slot.as_ref().is_none_or(|current| value < *current) {
        *slot = Some(value);
    }
}

/// Parse one SMT-LIB decimal without first rounding it through `f64`.
///
/// Kept visible to sibling VNN-LIB evaluators so every proof-sensitive parser
/// uses the same bounded decimal grammar.
pub(super) fn parse_exact_decimal(token: &str) -> Result<BigRational> {
    if token.is_empty() || token.len() > MAX_DECIMAL_DIGITS {
        return Err(NyError::InvalidSpec(
            "certified input decimal is empty or exceeds the digit cap".to_string(),
        ));
    }

    let (negative, unsigned) = if let Some(rest) = token.strip_prefix('-') {
        (true, rest)
    } else if let Some(rest) = token.strip_prefix('+') {
        (false, rest)
    } else {
        (false, token)
    };
    let (mantissa, exponent) = if let Some((mantissa, exponent)) = unsigned
        .split_once('e')
        .or_else(|| unsigned.split_once('E'))
    {
        let exponent = exponent.parse::<i64>().map_err(|_| {
            NyError::InvalidSpec(format!(
                "certified input literal has an invalid exponent: {token}"
            ))
        })?;
        (mantissa, exponent)
    } else {
        (unsigned, 0)
    };
    if exponent.unsigned_abs() > u64::try_from(MAX_DECIMAL_SCALE).unwrap_or(u64::MAX) {
        return Err(NyError::InvalidSpec(format!(
            "certified input literal exponent exceeds the scale cap: {token}"
        )));
    }

    let mut parts = mantissa.split('.');
    let integer = parts.next().unwrap_or_default();
    let fraction = parts.next().unwrap_or_default();
    if parts.next().is_some()
        || (integer.is_empty() && fraction.is_empty())
        || !integer.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(NyError::InvalidSpec(format!(
            "certified input literal is not a decimal numeral: {token}"
        )));
    }

    let digit_count = integer.len().checked_add(fraction.len()).ok_or_else(|| {
        NyError::InvalidSpec("certified decimal digit count overflow".to_string())
    })?;
    if digit_count == 0 || digit_count > MAX_DECIMAL_DIGITS {
        return Err(NyError::InvalidSpec(format!(
            "certified input literal exceeds the digit cap: {token}"
        )));
    }
    let digits = format!("{integer}{fraction}");
    let mut numerator = BigInt::parse_bytes(digits.as_bytes(), 10).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "certified input literal is not a decimal numeral: {token}"
        ))
    })?;
    if negative {
        numerator = -numerator;
    }

    let scale = i64::try_from(fraction.len())
        .map_err(|_| NyError::InvalidSpec("certified decimal scale overflow".to_string()))?
        - exponent;
    if scale.unsigned_abs() > u64::try_from(MAX_DECIMAL_SCALE).unwrap_or(u64::MAX) {
        return Err(NyError::InvalidSpec(format!(
            "certified input literal exceeds the scale cap: {token}"
        )));
    }
    if scale >= 0 {
        let scale = u32::try_from(scale)
            .map_err(|_| NyError::InvalidSpec("certified decimal scale overflow".to_string()))?;
        Ok(BigRational::new(numerator, BigInt::from(10_u8).pow(scale)))
    } else {
        let magnitude = u32::try_from(-scale)
            .map_err(|_| NyError::InvalidSpec("certified decimal scale overflow".to_string()))?;
        Ok(BigRational::from_integer(
            numerator * BigInt::from(10_u8).pow(magnitude),
        ))
    }
}

pub(super) fn rational_to_lower_f64(value: &BigRational, index: usize) -> Result<f64> {
    let candidate = value.to_f64().ok_or_else(|| conversion_error(index))?;
    if !candidate.is_finite() {
        return Err(conversion_error(index));
    }
    let dyadic = BigRational::from_float(candidate).ok_or_else(|| conversion_error(index))?;
    let outward = if dyadic > *value {
        candidate.next_down()
    } else {
        candidate
    };
    if !outward.is_finite() || BigRational::from_float(outward).is_none_or(|dyadic| dyadic > *value)
    {
        return Err(conversion_error(index));
    }
    Ok(outward)
}

pub(super) fn rational_to_upper_f64(value: &BigRational, index: usize) -> Result<f64> {
    let candidate = value.to_f64().ok_or_else(|| conversion_error(index))?;
    if !candidate.is_finite() {
        return Err(conversion_error(index));
    }
    let dyadic = BigRational::from_float(candidate).ok_or_else(|| conversion_error(index))?;
    let outward = if dyadic < *value {
        candidate.next_up()
    } else {
        candidate
    };
    if !outward.is_finite() || BigRational::from_float(outward).is_none_or(|dyadic| dyadic < *value)
    {
        return Err(conversion_error(index));
    }
    Ok(outward)
}

fn conversion_error(index: usize) -> NyError {
    NyError::InvalidSpec(format!(
        "exact input bound for X_{index} has no finite outward f64 enclosure"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "external-vnncomp")]
    use std::path::PathBuf;

    fn exact(value: f64) -> BigRational {
        BigRational::from_float(value).expect("finite test value")
    }

    #[test]
    fn non_dyadic_point_is_outward_and_remains_a_hint() {
        let content = "
            (declare-const X_0 Real)
            (assert (>= X_0 0.1))
            (assert (<= X_0 0.1))
        ";
        let (_, box_) = parse_vnnlib_with_certified_input_box(content).unwrap();
        let target = parse_exact_decimal("0.1").unwrap();
        assert_eq!(box_.declared_point(), &[true]);
        assert!(exact(box_.lower()[0]) <= target);
        assert!(exact(box_.upper()[0]) >= target);
        assert!(box_.lower()[0] < box_.upper()[0]);
    }

    #[test]
    fn exact_dyadic_point_needs_no_gratuitous_width() {
        let content = "
            (declare-const X_0 Real)
            (assert (= X_0 0.5))
        ";
        let (_, box_) = parse_vnnlib_with_certified_input_box(content).unwrap();
        assert_eq!(box_.lower(), &[0.5]);
        assert_eq!(box_.upper(), &[0.5]);
        assert_eq!(box_.declared_point(), &[true]);
    }

    #[test]
    fn reversed_atoms_and_multiple_bounds_tighten_exactly() {
        let content = "
            (declare-const X_0 Real)
            (assert (<= -2e-1 X_0))
            (assert (>= X_0 -0.1))
            (assert (>= 0.4 X_0))
            (assert (<= X_0 0.3))
        ";
        let (_, box_) = parse_vnnlib_with_certified_input_box(content).unwrap();
        let lower = parse_exact_decimal("-0.1").unwrap();
        let upper = parse_exact_decimal("0.3").unwrap();
        assert!(exact(box_.lower()[0]) <= lower);
        assert!(exact(box_.upper()[0]) >= upper);
        assert_eq!(box_.declared_point(), &[false]);
    }

    #[test]
    fn compound_input_assertion_fails_closed() {
        let content = "
            (declare-const X_0 Real)
            (assert (>= X_0 0.0))
            (assert (<= (+ X_0 0.0) 1.0))
        ";
        let error = parse_vnnlib_with_certified_input_box(content).unwrap_err();
        assert!(error.to_string().contains("direct axis-aligned atom"));
    }

    #[test]
    fn direct_relational_outputs_are_source_certified() {
        let content = "
            (declare-const X_0 Real)
            (declare-const Y_0 Real)
            (declare-const Y_1 Real)
            (declare-const Y_2 Real)
            (assert (>= X_0 -1.0))
            (assert (<= X_0 1.0))
            (assert (and (<= Y_0 Y_1) (> Y_2 Y_0)))
        ";
        let (_, _, outputs) = parse_vnnlib_with_certified_affine_property(content).unwrap();
        assert_eq!(
            outputs.atoms(),
            &[
                CertifiedRelationalOutputAtom::LessEq(0, 1),
                CertifiedRelationalOutputAtom::GreaterThan(2, 0),
            ]
        );
    }

    #[test]
    fn certified_relational_output_rejects_decimal_cancellation_collision() {
        let content = "
            (declare-const X_0 Real)
            (declare-const Y_0 Real)
            (declare-const Y_1 Real)
            (assert (>= X_0 -1.0))
            (assert (<= X_0 1.0))
            (assert (<= (+ Y_0 1.0) (+ Y_1 1.00000000000000001)))
        ";
        let ordinary = parse_vnnlib(content).expect("ordinary parser normalizes the collision");
        assert_eq!(
            ordinary.output_constraints.as_slice(),
            &[OutputConstraint::LessEq(0, 1)]
        );
        let error = parse_vnnlib_with_certified_affine_property(content).unwrap_err();
        assert!(
            error.to_string().contains("direct `Y_i`-to-`Y_j`"),
            "unexpected refusal: {error}"
        );
    }

    #[test]
    fn certified_relational_output_rejects_constants_equality_and_boolean_structure() {
        let prefix = "
            (declare-const X_0 Real)
            (declare-const Y_0 Real)
            (declare-const Y_1 Real)
            (assert (>= X_0 -1.0))
            (assert (<= X_0 1.0))
        ";
        for assertion in [
            "(assert (<= Y_0 0.0))",
            "(assert (= Y_0 Y_1))",
            "(assert (or (<= Y_0 Y_1) (>= Y_0 Y_1)))",
            "(assert (and (and (<= Y_0 Y_1))))",
        ] {
            let content = format!("{prefix}\n{assertion}\n");
            assert!(
                parse_vnnlib_with_certified_affine_property(&content).is_err(),
                "unsupported output surface was accepted: {assertion}"
            );
        }
    }

    #[test]
    fn missing_endpoint_fails_closed() {
        let content = "
            (declare-const X_0 Real)
            (assert (>= X_0 0.0))
        ";
        let error = parse_vnnlib_with_certified_input_box(content).unwrap_err();
        assert!(error.to_string().contains("missing an upper bound"));
    }

    #[test]
    fn exact_decimal_parser_handles_sign_scale_and_exponent() {
        assert_eq!(
            parse_exact_decimal("-12.50e-2").unwrap(),
            BigRational::new(BigInt::from(-1), BigInt::from(8))
        );
        assert_eq!(
            parse_exact_decimal("3E2").unwrap(),
            BigRational::from_integer(BigInt::from(300))
        );
        assert!(parse_exact_decimal("1/3").is_err());
        assert!(parse_exact_decimal("NaN").is_err());
    }

    #[test]
    fn certified_parser_resource_caps_fail_closed() {
        assert!(check_certified_source_size(MAX_CERTIFIED_SOURCE_BYTES + 1).is_err());
        assert!(check_raw_token_count(MAX_RAW_TOKENS + 1).is_err());
        assert!(check_certified_source_size(MAX_CERTIFIED_SOURCE_BYTES).is_ok());
        assert!(check_raw_token_count(MAX_RAW_TOKENS).is_ok());
    }

    #[test]
    #[cfg(feature = "external-vnncomp")]
    fn real_metaroom_119_retains_161_symbols() {
        let path = std::env::var_os("NY_METAROOM_119_VNNLIB")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR")).join(
                    "../../benchmarks/vnncomp2025/benchmarks/metaroom_2023/vnnlib/\
                     spec_idx_119_eps_0.00000436.vnnlib",
                )
            });
        assert!(
            path.is_file(),
            "MetaRoom VNN-LIB fixture is missing at {}; \
             run benchmarks/download_benchmarks.sh",
            path.display()
        );
        let (spec, box_) = load_vnnlib_with_certified_input_box(path).unwrap();
        assert_eq!(spec.num_inputs, 5_376);
        assert_eq!(box_.len(), 5_376);
        assert_eq!(
            box_.declared_point().iter().filter(|&&point| point).count(),
            5_215
        );
        assert_eq!(
            box_.declared_point()
                .iter()
                .filter(|&&point| !point)
                .count(),
            161
        );
        assert!(box_
            .lower()
            .iter()
            .zip(box_.upper())
            .all(|(&lower, &upper)| lower.is_finite() && lower <= upper && upper.is_finite()));
    }
}
