// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Default-dark exact phase-cell certificates for one structurally free axis.
//!
//! The graph walk is exact over `BigRational` until the final sigmoid.  ReLU
//! roots are therefore enumerated without a tolerance or a sampled/JVP event
//! proposal.  The sigmoid and inverse-sigmoid boundary are handled only through
//! the directed interval kernel in `one_axis_directed`.
//!
//! This surface is deliberately not called by any production verifier.  Its
//! result is an observation, every certificate says `verdict_authority=false`,
//! and all malformed input, unsupported algebra, resource exhaustion, or
//! deadline expiry declines closed.

use std::collections::HashMap;
use std::str::FromStr;
use std::time::Instant;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use sha2::{Digest, Sha256};

use crate::layers::Layer;

use super::one_axis_algebra::{
    OneAxisAlgebraClass, ONE_AXIS_MAX_EDGES, ONE_AXIS_MAX_NODES, ONE_AXIS_MAX_RANK,
    ONE_AXIS_MAX_TENSOR_ELEMENTS, ONE_AXIS_MAX_TOTAL_ELEMENTS,
};
use super::one_axis_directed::{
    logit_enclosure, rational_enclosure, sigmoid_enclosure, DirectedInterval,
};
use super::{GraphNetwork, NETWORK_INPUT};

pub const ONE_AXIS_PHASE_CERTIFICATE_VERSION: &str = "ny.one-axis-phase.m1.v1";

mod grouped;
pub use grouped::{
    OneAxisGroupedContextCertificate, OneAxisGroupedMemberCertificate, OneAxisGroupedPhaseAttempt,
    OneAxisGroupedPhaseCertificate, OneAxisGroupedPhaseLimits, OneAxisGroupedReplayResult,
    ONE_AXIS_GROUPED_PHASE_CERTIFICATE_VERSION,
};

/// Exact scalar accepted at the public checker boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneAxisRational(BigRational);

impl OneAxisRational {
    pub fn new(numerator: BigInt, denominator: BigInt) -> Option<Self> {
        (!denominator.is_zero()).then(|| Self(BigRational::new(numerator, denominator)))
    }

    pub fn from_integer(value: i64) -> Self {
        Self(BigRational::from_integer(value.into()))
    }

    pub fn from_f32_exact(value: f32) -> Option<Self> {
        // Convert the binary32 representation directly. Widening through f64
        // can collapse a subnormal when the host enables DAZ; `from_float`
        // decodes the original f32 bits instead.
        BigRational::from_float(value).map(Self)
    }

    /// Parse a finite decimal/scientific literal exactly.
    ///
    /// The exponent and digit caps prevent adversarial literals from allocating
    /// an unbounded power of ten before the normal rational-bit budget runs.
    pub fn parse_decimal(text: &str) -> Option<Self> {
        const MAX_DIGITS: usize = 4096;
        const MAX_ABS_EXPONENT: i32 = 4096;

        let text = text.trim();
        if text.is_empty() {
            return None;
        }
        let (negative, unsigned) = match text.as_bytes().first()? {
            b'+' => (false, &text[1..]),
            b'-' => (true, &text[1..]),
            _ => (false, text),
        };
        let (mantissa, exponent_text) =
            unsigned.find(['e', 'E']).map_or((unsigned, None), |index| {
                (&unsigned[..index], Some(&unsigned[index + 1..]))
            });
        let exponent = match exponent_text {
            Some(raw) if !raw.is_empty() => i32::from_str(raw).ok()?,
            Some(_) => return None,
            None => 0,
        };
        if exponent.unsigned_abs() > MAX_ABS_EXPONENT.unsigned_abs() {
            return None;
        }
        let mut digits = String::with_capacity(mantissa.len());
        let mut fractional_digits = 0usize;
        let mut seen_point = false;
        for byte in mantissa.bytes() {
            match byte {
                b'0'..=b'9' => {
                    digits.push(char::from(byte));
                    fractional_digits += usize::from(seen_point);
                }
                b'.' if !seen_point => seen_point = true,
                _ => return None,
            }
        }
        if digits.is_empty() || digits.len() > MAX_DIGITS {
            return None;
        }
        let mut numerator = BigInt::from_str(&digits).ok()?;
        if negative {
            numerator = -numerator;
        }
        let decimal_exponent = exponent.checked_sub(i32::try_from(fractional_digits).ok()?)?;
        let power = decimal_exponent.unsigned_abs();
        let factor = BigInt::from(10_u8).pow(power);
        let rational = if decimal_exponent >= 0 {
            BigRational::from_integer(numerator * factor)
        } else {
            BigRational::new(numerator, factor)
        };
        Some(Self(rational))
    }

    pub fn numerator(&self) -> &BigInt {
        self.0.numer()
    }

    pub fn denominator(&self) -> &BigInt {
        self.0.denom()
    }
}

impl FromStr for OneAxisRational {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse_decimal(value).ok_or(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OneAxisConstraintRelation {
    LessEqual,
    GreaterEqual,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneAxisOutputConstraint {
    pub relation: OneAxisConstraintRelation,
    pub bound: OneAxisRational,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneAxisExactProblem {
    pub input_shape: Vec<usize>,
    /// Exact fixed coordinates.  The `free_axis` entry is ignored.
    pub fixed_inputs: Vec<OneAxisRational>,
    pub free_axis: usize,
    pub lower: OneAxisRational,
    pub upper: OneAxisRational,
    /// Conjunction of scalar output constraints.
    pub constraints: Vec<OneAxisOutputConstraint>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneAxisPhaseLimits {
    pub max_phase_cells: usize,
    pub max_relu_scalars_per_phase: usize,
    pub max_exact_operations: usize,
    pub max_rational_bits: u64,
    pub max_constraints: usize,
    pub max_tensor_elements: usize,
    pub max_total_tensor_elements: usize,
}

impl Default for OneAxisPhaseLimits {
    fn default() -> Self {
        Self {
            max_phase_cells: 4096,
            max_relu_scalars_per_phase: 65_536,
            max_exact_operations: 100_000_000,
            max_rational_bits: 16_384,
            max_constraints: 16,
            max_tensor_elements: 262_144,
            max_total_tensor_elements: 1_048_576,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OneAxisPhaseDeclineReason {
    Deadline,
    StructuralRefusal,
    InvalidProblem,
    ProblemLimit,
    ContextLimit,
    ConstraintLimit,
    TotalConstraintLimit,
    PhaseCellLimit,
    TotalPhaseCellLimit,
    HammingTraversalLimit,
    ReluScalarLimit,
    ExactOperationLimit,
    RationalBitLimit,
    GraphShape,
    UnsupportedAlgebra,
    DynamicMulOperands,
    InvalidDivisor,
    NonScalarOutput,
    DirectedArithmetic,
    CertificateMalformed,
    ProblemDigestMismatch,
    ReplayMismatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneAxisPhaseDecline {
    pub reason: OneAxisPhaseDeclineReason,
    pub node: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OneAxisCoreGuard {
    Always,
    Impossible,
    LessEqual(OneAxisRational),
    GreaterEqual(OneAxisRational),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneAxisPeeledConstraint {
    pub necessary: OneAxisCoreGuard,
    pub sufficient: OneAxisCoreGuard,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OneAxisWrapperEnclosure {
    pub offset_lower: f64,
    pub offset_upper: f64,
    /// `1` for `offset + sigmoid(core)`, `-1` for
    /// `offset - sigmoid(core)`.
    pub sigmoid_sign: i8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneAxisAffineCertificate {
    pub slope: OneAxisRational,
    pub bias: OneAxisRational,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneAxisPhaseCellCertificate {
    pub lower: OneAxisRational,
    pub upper: OneAxisRational,
    pub core: OneAxisAffineCertificate,
    pub relu_phase_digest: [u8; 32],
    pub relu_scalars: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OneAxisPhaseObservation {
    CertifiedEmpty,
    ExactWitness { free_value: OneAxisRational },
    Inconclusive,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OneAxisPhaseCertificate {
    pub version: &'static str,
    pub verdict_authority: bool,
    pub problem_digest: [u8; 32],
    /// Digest of the admitted post-loader graph semantics on this input shape.
    pub graph_digest: [u8; 32],
    pub cells: Vec<OneAxisPhaseCellCertificate>,
    pub wrapper: OneAxisWrapperEnclosure,
    pub peeled_constraints: Vec<OneAxisPeeledConstraint>,
    pub observation: OneAxisPhaseObservation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OneAxisPhaseAttempt {
    pub certificate: Option<OneAxisPhaseCertificate>,
    pub decline: Option<OneAxisPhaseDecline>,
    /// Completed phase cells before success or fail-closed refusal.
    pub phase_cells_examined: usize,
    /// Charged exact rational operations before success or refusal.
    pub exact_operations: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneAxisReplayResult {
    pub accepted: bool,
    pub observation: Option<OneAxisPhaseObservation>,
    pub decline: Option<OneAxisPhaseDecline>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExactAffine {
    slope: BigRational,
    bias: BigRational,
    /// Structural dependence, retained even when the current phase has slope
    /// zero (for example an inactive ReLU).  Sigmoid peeling must use this,
    /// not the phase-local slope, to distinguish a dynamic core from a truly
    /// static sibling.
    depends: bool,
}

impl ExactAffine {
    fn constant(value: BigRational) -> Self {
        Self {
            slope: BigRational::zero(),
            bias: value,
            depends: false,
        }
    }

    fn variable() -> Self {
        Self {
            slope: BigRational::one(),
            bias: BigRational::zero(),
            depends: true,
        }
    }

    fn value_at(&self, point: &BigRational, budget: &mut ExactBudget<'_>) -> Option<BigRational> {
        let product = budget.mul(&self.slope, point)?;
        budget.add(&product, &self.bias)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExactTensor {
    shape: Vec<usize>,
    values: Vec<ExactAffine>,
}

#[derive(Clone, Debug)]
struct LinearPhaseCache {
    input: ExactTensor,
    output: ExactTensor,
    context_epoch: usize,
}

impl LinearPhaseCache {
    fn retained_elements(&self) -> Option<usize> {
        self.input
            .values
            .len()
            .checked_add(self.output.values.len())
    }
}

#[derive(Clone, Debug)]
struct WrapperValue {
    offset: DirectedInterval,
    sign: i8,
    core: ExactAffine,
}

#[derive(Clone, Debug)]
enum PhaseValue {
    Affine(ExactTensor),
    StaticSigmoid(DirectedInterval),
    DynamicSigmoid(ExactAffine),
    Wrapper(WrapperValue),
}

struct ExactBudget<'a> {
    limits: &'a OneAxisPhaseLimits,
    deadline: Instant,
    operations: usize,
    work_items: usize,
    phase_cache_elements: usize,
    context_epoch: usize,
    cross_context_sparse_linear_updates: usize,
    failure: Option<OneAxisPhaseDeclineReason>,
}

impl<'a> ExactBudget<'a> {
    fn new(limits: &'a OneAxisPhaseLimits, deadline: Instant) -> Self {
        Self {
            limits,
            deadline,
            operations: 0,
            work_items: 0,
            phase_cache_elements: 0,
            context_epoch: 0,
            cross_context_sparse_linear_updates: 0,
            failure: None,
        }
    }

    fn begin_context(&mut self, epoch: usize) {
        self.context_epoch = epoch;
    }

    fn try_replace_phase_cache(&mut self, old: usize, new: usize) -> bool {
        let Some(retained) = self
            .phase_cache_elements
            .checked_sub(old)
            .and_then(|value| value.checked_add(new))
        else {
            return false;
        };
        if retained > self.limits.max_total_tensor_elements {
            return false;
        }
        self.phase_cache_elements = retained;
        true
    }

    fn check_deadline(&mut self) -> bool {
        if Instant::now() >= self.deadline {
            self.failure = Some(OneAxisPhaseDeclineReason::Deadline);
            false
        } else {
            true
        }
    }

    fn check_value(&mut self, value: &BigRational) -> bool {
        let bits = value.numer().bits().max(value.denom().bits());
        if bits > self.limits.max_rational_bits {
            self.failure = Some(OneAxisPhaseDeclineReason::RationalBitLimit);
            false
        } else {
            true
        }
    }

    /// Poll work that can bypass the charged-arithmetic counter, such as an
    /// exact zero shortcut or a tensor copy.  This keeps deadline latency
    /// bounded even on sparse/static graphs.
    fn poll_work(&mut self) -> bool {
        self.work_items = self.work_items.wrapping_add(1);
        !self.work_items.is_multiple_of(1024) || self.check_deadline()
    }

    fn charge(&mut self, value: BigRational) -> Option<BigRational> {
        self.operations = self.operations.checked_add(1)?;
        if self.operations > self.limits.max_exact_operations {
            self.failure = Some(OneAxisPhaseDeclineReason::ExactOperationLimit);
            return None;
        }
        if self.operations.is_multiple_of(1024) && !self.check_deadline() {
            return None;
        }
        self.check_value(&value).then_some(value)
    }

    fn add(&mut self, left: &BigRational, right: &BigRational) -> Option<BigRational> {
        if !self.poll_work() {
            return None;
        }
        if left.is_zero() {
            return Some(right.clone());
        }
        if right.is_zero() {
            return Some(left.clone());
        }
        self.charge(left + right)
    }

    fn sub(&mut self, left: &BigRational, right: &BigRational) -> Option<BigRational> {
        if !self.poll_work() {
            return None;
        }
        if right.is_zero() {
            return Some(left.clone());
        }
        if left == right {
            return Some(BigRational::zero());
        }
        self.charge(left - right)
    }

    fn mul(&mut self, left: &BigRational, right: &BigRational) -> Option<BigRational> {
        if !self.poll_work() {
            return None;
        }
        if left.is_zero() || right.is_zero() {
            return Some(BigRational::zero());
        }
        if left.is_one() {
            return Some(right.clone());
        }
        if right.is_one() {
            return Some(left.clone());
        }
        self.charge(left * right)
    }

    fn div(&mut self, left: &BigRational, right: &BigRational) -> Option<BigRational> {
        if !self.poll_work() {
            return None;
        }
        if right.is_zero() {
            self.failure = Some(OneAxisPhaseDeclineReason::InvalidDivisor);
            return None;
        }
        if left.is_zero() {
            return Some(BigRational::zero());
        }
        if right.is_one() {
            return Some(left.clone());
        }
        self.charge(left / right)
    }

    fn neg(&mut self, value: &BigRational) -> Option<BigRational> {
        if !self.poll_work() {
            return None;
        }
        if value.is_zero() {
            return Some(BigRational::zero());
        }
        self.charge(-value)
    }

    fn affine_add(&mut self, left: &ExactAffine, right: &ExactAffine) -> Option<ExactAffine> {
        Some(ExactAffine {
            slope: self.add(&left.slope, &right.slope)?,
            bias: self.add(&left.bias, &right.bias)?,
            depends: left.depends || right.depends,
        })
    }

    fn affine_sub(&mut self, left: &ExactAffine, right: &ExactAffine) -> Option<ExactAffine> {
        Some(ExactAffine {
            slope: self.sub(&left.slope, &right.slope)?,
            bias: self.sub(&left.bias, &right.bias)?,
            depends: left.depends || right.depends,
        })
    }

    fn affine_scale(&mut self, value: &ExactAffine, scale: &BigRational) -> Option<ExactAffine> {
        Some(ExactAffine {
            slope: self.mul(&value.slope, scale)?,
            bias: self.mul(&value.bias, scale)?,
            depends: value.depends,
        })
    }
}

fn decline(reason: OneAxisPhaseDeclineReason, node: Option<&str>) -> OneAxisPhaseDecline {
    OneAxisPhaseDecline {
        reason,
        node: node.map(str::to_owned),
    }
}

fn tensor_elements(shape: &[usize]) -> Option<usize> {
    if shape.len() > ONE_AXIS_MAX_RANK {
        return None;
    }
    if shape.is_empty() {
        return Some(1);
    }
    shape.iter().try_fold(1usize, |product, &dimension| {
        (dimension > 0)
            .then(|| product.checked_mul(dimension))
            .flatten()
    })
}

fn strides(shape: &[usize]) -> Option<Vec<usize>> {
    let mut result = vec![1usize; shape.len()];
    for index in (0..shape.len().saturating_sub(1)).rev() {
        result[index] = result[index + 1].checked_mul(shape[index + 1])?;
    }
    Some(result)
}

fn flat_coordinate(flat: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut remaining = flat;
    let mut coordinate = Vec::with_capacity(shape.len());
    for (&dimension, &stride) in shape.iter().zip(strides) {
        let index = remaining / stride;
        remaining %= stride;
        debug_assert!(index < dimension);
        coordinate.push(index);
    }
    coordinate
}

fn coordinate_flat(coordinate: &[usize], strides: &[usize]) -> Option<usize> {
    coordinate
        .iter()
        .zip(strides)
        .try_fold(0usize, |flat, (&index, &stride)| {
            flat.checked_add(index.checked_mul(stride)?)
        })
}

fn broadcast_flat_index(
    output_coordinate: &[usize],
    input_shape: &[usize],
    input_strides: &[usize],
) -> Option<usize> {
    if input_shape.len() > output_coordinate.len() {
        return None;
    }
    let offset = output_coordinate.len() - input_shape.len();
    input_shape.iter().zip(input_strides).enumerate().try_fold(
        0usize,
        |flat, (index, (&dimension, &stride))| {
            let output_index = output_coordinate[offset + index];
            let input_index = if dimension == 1 { 0 } else { output_index };
            (input_index < dimension)
                .then(|| flat.checked_add(input_index.checked_mul(stride)?))
                .flatten()
        },
    )
}

fn exact_f32(value: f32) -> Option<BigRational> {
    // Decode binary32 directly: widening through a floating instruction can
    // collapse a subnormal when the host enables DAZ.
    BigRational::from_float(value)
}

fn bounded_exact_f32(value: f32, budget: &mut ExactBudget<'_>) -> Option<BigRational> {
    let exact = exact_f32(value)?;
    budget.check_value(&exact).then_some(exact)
}

fn hash_usize(hasher: &mut Sha256, value: usize) {
    hasher.update(u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes());
}

fn hash_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hash_usize(hasher, bytes.len());
    hasher.update(bytes);
}

fn hash_rational(hasher: &mut Sha256, value: &BigRational) {
    let numerator = value.numer().to_signed_bytes_le();
    let denominator = value.denom().to_signed_bytes_le();
    hash_bytes(hasher, &numerator);
    hash_bytes(hasher, &denominator);
}

fn problem_digest(problem: &OneAxisExactProblem, deadline: Instant) -> Option<[u8; 32]> {
    if Instant::now() >= deadline {
        return None;
    }
    let mut hasher = Sha256::new();
    hasher.update(ONE_AXIS_PHASE_CERTIFICATE_VERSION.as_bytes());
    hash_usize(&mut hasher, problem.input_shape.len());
    for &dimension in &problem.input_shape {
        hash_usize(&mut hasher, dimension);
    }
    hash_usize(&mut hasher, problem.free_axis);
    hash_usize(&mut hasher, problem.fixed_inputs.len());
    for (index, value) in problem.fixed_inputs.iter().enumerate() {
        if index.is_multiple_of(4096) && Instant::now() >= deadline {
            return None;
        }
        hash_rational(&mut hasher, &value.0);
    }
    hash_rational(&mut hasher, &problem.lower.0);
    hash_rational(&mut hasher, &problem.upper.0);
    hash_usize(&mut hasher, problem.constraints.len());
    for constraint in &problem.constraints {
        hasher.update([match constraint.relation {
            OneAxisConstraintRelation::LessEqual => 0,
            OneAxisConstraintRelation::GreaterEqual => 1,
        }]);
        hash_rational(&mut hasher, &constraint.bound.0);
    }
    Some(hasher.finalize().into())
}

fn graph_digest(
    graph: &GraphNetwork,
    problem: &OneAxisExactProblem,
    deadline: Instant,
) -> Option<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(b"ny.one-axis-phase.graph.v1");
    hash_bytes(&mut hasher, graph.output_name().as_bytes());
    for (node_index, name) in graph.exec_order().ok()?.iter().enumerate() {
        if node_index % 8 == 0 && Instant::now() >= deadline {
            return None;
        }
        let node = graph.node(name)?;
        hash_bytes(&mut hasher, name.as_bytes());
        hash_bytes(&mut hasher, node.layer().layer_type().as_bytes());
        hash_usize(&mut hasher, node.inputs().len());
        for input in node.inputs() {
            hash_bytes(&mut hasher, input.as_bytes());
        }
        let output_shape = graph.declared_shape(name)?;
        hash_usize(&mut hasher, output_shape.len());
        for &dimension in output_shape {
            hash_usize(&mut hasher, dimension);
        }
        match node.layer() {
            Layer::Slice(layer) => {
                let [input] = node.inputs() else {
                    return None;
                };
                let input_shape = if input == NETWORK_INPUT {
                    problem.input_shape.as_slice()
                } else {
                    graph.declared_shape(input)?
                };
                let (axis, start, end) = layer.resolved_range(input_shape).ok()?;
                hash_usize(&mut hasher, axis);
                hash_usize(&mut hasher, start);
                hash_usize(&mut hasher, end);
            }
            Layer::Linear(layer) => {
                hash_usize(&mut hasher, layer.weight.nrows());
                hash_usize(&mut hasher, layer.weight.ncols());
                for (index, &weight) in layer.weight.iter().enumerate() {
                    if index % 4096 == 0 && Instant::now() >= deadline {
                        return None;
                    }
                    hasher.update(weight.to_bits().to_le_bytes());
                }
                match &layer.bias {
                    Some(bias) => {
                        hasher.update([1]);
                        hash_usize(&mut hasher, bias.len());
                        for (index, &value) in bias.iter().enumerate() {
                            if index.is_multiple_of(4096) && Instant::now() >= deadline {
                                return None;
                            }
                            hasher.update(value.to_bits().to_le_bytes());
                        }
                    }
                    None => hasher.update([0]),
                }
            }
            Layer::AddConstant(layer) => {
                hash_usize(&mut hasher, layer.constant().ndim());
                for &dimension in layer.constant().shape() {
                    hash_usize(&mut hasher, dimension);
                }
                for (index, &value) in layer.constant().iter().enumerate() {
                    if index % 4096 == 0 && Instant::now() >= deadline {
                        return None;
                    }
                    hasher.update(value.to_bits().to_le_bytes());
                }
            }
            Layer::ReduceSum(layer) => {
                hash_usize(&mut hasher, layer.axes.len());
                for &axis in &layer.axes {
                    hasher.update(axis.to_le_bytes());
                }
                hasher.update([u8::from(layer.keepdims)]);
            }
            Layer::Concat(layer) => hasher.update(layer.axis.to_le_bytes()),
            Layer::ReLU(_)
            | Layer::MulBinary(_)
            | Layer::Div(_)
            | Layer::Sigmoid(_)
            | Layer::Sub(_) => {}
            _ => return None,
        }
    }
    Some(hasher.finalize().into())
}

fn validate_problem(
    problem: &OneAxisExactProblem,
    limits: &OneAxisPhaseLimits,
    deadline: Instant,
) -> Result<usize, OneAxisPhaseDeclineReason> {
    if Instant::now() >= deadline {
        return Err(OneAxisPhaseDeclineReason::Deadline);
    }
    let elements =
        tensor_elements(&problem.input_shape).ok_or(OneAxisPhaseDeclineReason::InvalidProblem)?;
    if elements > ONE_AXIS_MAX_TENSOR_ELEMENTS
        || elements > limits.max_tensor_elements
        || problem.fixed_inputs.len() != elements
        || problem.free_axis >= elements
        || problem.lower.0 > problem.upper.0
    {
        return Err(OneAxisPhaseDeclineReason::InvalidProblem);
    }
    if problem.constraints.is_empty() {
        return Err(OneAxisPhaseDeclineReason::InvalidProblem);
    }
    if problem.constraints.len() > limits.max_constraints {
        return Err(OneAxisPhaseDeclineReason::ConstraintLimit);
    }
    for (index, value) in problem
        .fixed_inputs
        .iter()
        .map(|value| &value.0)
        .chain([&problem.lower.0, &problem.upper.0])
        .chain(
            problem
                .constraints
                .iter()
                .map(|constraint| &constraint.bound.0),
        )
        .enumerate()
    {
        if index.is_multiple_of(4096) && Instant::now() >= deadline {
            return Err(OneAxisPhaseDeclineReason::Deadline);
        }
        if value.numer().bits().max(value.denom().bits()) > limits.max_rational_bits {
            return Err(OneAxisPhaseDeclineReason::RationalBitLimit);
        }
    }
    Ok(elements)
}

fn input_tensor(
    problem: &OneAxisExactProblem,
    budget: &mut ExactBudget<'_>,
) -> Option<ExactTensor> {
    let mut values = Vec::with_capacity(problem.fixed_inputs.len());
    for (index, value) in problem.fixed_inputs.iter().enumerate() {
        if !budget.poll_work() || !budget.check_value(&value.0) {
            return None;
        }
        values.push(if index == problem.free_axis {
            ExactAffine::variable()
        } else {
            ExactAffine::constant(value.0.clone())
        });
    }
    Some(ExactTensor {
        shape: problem.input_shape.clone(),
        values,
    })
}

fn binary_affine_tensor(
    left: &ExactTensor,
    right: &ExactTensor,
    output_shape: &[usize],
    budget: &mut ExactBudget<'_>,
    mut operation: impl FnMut(&ExactAffine, &ExactAffine, &mut ExactBudget<'_>) -> Option<ExactAffine>,
) -> Option<ExactTensor> {
    if crate::shape::broadcast_shapes(&left.shape, &right.shape).as_deref() != Some(output_shape) {
        return None;
    }
    let output_elements = tensor_elements(output_shape)?;
    let output_strides = strides(output_shape)?;
    let left_strides = strides(&left.shape)?;
    let right_strides = strides(&right.shape)?;
    let mut values = Vec::with_capacity(output_elements);
    for flat in 0..output_elements {
        let coordinate = flat_coordinate(flat, output_shape, &output_strides);
        let left_index = broadcast_flat_index(&coordinate, &left.shape, &left_strides)?;
        let right_index = broadcast_flat_index(&coordinate, &right.shape, &right_strides)?;
        values.push(operation(
            left.values.get(left_index)?,
            right.values.get(right_index)?,
            budget,
        )?);
    }
    Some(ExactTensor {
        shape: output_shape.to_vec(),
        values,
    })
}

fn slice_tensor(
    input: &ExactTensor,
    layer: &crate::layers::SliceLayer,
    output_shape: &[usize],
    budget: &mut ExactBudget<'_>,
) -> Option<ExactTensor> {
    let (axis, start, end) = layer.resolved_range(&input.shape).ok()?;
    let mut expected = input.shape.clone();
    expected[axis] = end.checked_sub(start)?;
    if expected != output_shape {
        return None;
    }
    let output_elements = tensor_elements(output_shape)?;
    let output_strides = strides(output_shape)?;
    let input_strides = strides(&input.shape)?;
    let mut values = Vec::with_capacity(output_elements);
    for flat in 0..output_elements {
        if !budget.poll_work() {
            return None;
        }
        let mut coordinate = flat_coordinate(flat, output_shape, &output_strides);
        coordinate[axis] = coordinate[axis].checked_add(start)?;
        values.push(
            input
                .values
                .get(coordinate_flat(&coordinate, &input_strides)?)?
                .clone(),
        );
    }
    Some(ExactTensor {
        shape: output_shape.to_vec(),
        values,
    })
}

fn linear_tensor(
    input: &ExactTensor,
    layer: &crate::layers::LinearLayer,
    output_shape: &[usize],
    cached_static_contributions: Option<&[ExactAffine]>,
    budget: &mut ExactBudget<'_>,
) -> Option<(ExactTensor, Vec<ExactAffine>)> {
    let (&input_width, leading) = input.shape.split_last()?;
    if input_width != layer.in_features() {
        return None;
    }
    let mut expected = leading.to_vec();
    expected.push(layer.out_features());
    if expected != output_shape {
        return None;
    }
    let batches = tensor_elements(leading)?;
    let output_elements = tensor_elements(output_shape)?;
    if cached_static_contributions.is_some_and(|cache| cache.len() != output_elements) {
        return None;
    }
    let mut values = Vec::with_capacity(output_elements);
    let mut static_contributions = Vec::with_capacity(output_elements);
    for batch in 0..batches {
        let input_start = batch.checked_mul(input_width)?;
        let input_end = input_start.checked_add(input_width)?;
        let batch_inputs = input.values.get(input_start..input_end)?;
        // `depends` is structural, not phase-local: an inactive ReLU retains
        // it so a later phase can expose the path again.  Preserve that bit
        // independently, then omit only coefficients whose current affine
        // value is identically zero.  Such a coefficient contributes exactly
        // zero to every output in this phase, so multiplying it by every
        // dense weight is pure exact-rational work.
        let mut batch_depends = false;
        for value in batch_inputs {
            if !budget.poll_work() {
                return None;
            }
            batch_depends |= value.depends;
        }
        for output in 0..layer.out_features() {
            let output_index = batch
                .checked_mul(layer.out_features())?
                .checked_add(output)?;
            let mut static_accumulator = match cached_static_contributions {
                Some(cache) => cache.get(output_index)?.clone(),
                None => ExactAffine::constant(match &layer.bias {
                    Some(bias) => bounded_exact_f32(*bias.get(output)?, budget)?,
                    None => BigRational::zero(),
                }),
            };
            let mut dynamic_accumulator = ExactAffine {
                slope: BigRational::zero(),
                bias: BigRational::zero(),
                depends: batch_depends,
            };
            for (input_index, input_value) in batch_inputs.iter().enumerate() {
                if !budget.poll_work() {
                    return None;
                }
                if cached_static_contributions.is_some() && !input_value.depends {
                    continue;
                }
                if input_value.depends && input_value.slope.is_zero() && input_value.bias.is_zero()
                {
                    // The first full evaluation validates every structurally
                    // dynamic coefficient even when its current affine input
                    // is zero. Later phases can skip it because the graph is
                    // immutable and the static-contribution cache proves that
                    // this node already completed that validation pass.
                    if cached_static_contributions.is_none() {
                        bounded_exact_f32(layer.weight[[output, input_index]], budget)?;
                    }
                    continue;
                }
                let weight = bounded_exact_f32(layer.weight[[output, input_index]], budget)?;
                let scaled = budget.affine_scale(input_value, &weight)?;
                if input_value.depends {
                    dynamic_accumulator = budget.affine_add(&dynamic_accumulator, &scaled)?;
                } else {
                    static_accumulator = budget.affine_add(&static_accumulator, &scaled)?;
                }
            }
            static_contributions.push(static_accumulator.clone());
            values.push(budget.affine_add(&static_accumulator, &dynamic_accumulator)?);
        }
    }
    Some((
        ExactTensor {
            shape: output_shape.to_vec(),
            values,
        },
        static_contributions,
    ))
}

/// Re-evaluate a dense affine layer from the previous phase by applying only
/// the exact input coefficients that changed at the phase boundary.
///
/// A one-dimensional ReLU sweep usually flips a small number of scalar phases
/// at a time.  Recomputing all `out_features * in_features` products after
/// such a flip is unnecessary: linearity makes the new output exactly
/// `cached_output + W * (input - cached_input)`.
/// [`LinearPhaseDelta::Unchanged`] reuses the cached exact output,
/// [`LinearPhaseDelta::Recompute`] asks the caller to use the full path when
/// the delta is not sparse enough, and `None` is a fail-closed
/// arithmetic/deadline refusal.
enum LinearPhaseDelta {
    Unchanged,
    Recompute,
    Updated(ExactTensor),
}

fn linear_tensor_from_phase_delta(
    input: &ExactTensor,
    layer: &crate::layers::LinearLayer,
    output_shape: &[usize],
    cached: &LinearPhaseCache,
    budget: &mut ExactBudget<'_>,
) -> Option<LinearPhaseDelta> {
    const MAX_CHANGED_FRACTION_DENOMINATOR: usize = 4;

    if cached.input.shape != input.shape
        || cached.output.shape != output_shape
        || cached.input.values.len() != input.values.len()
    {
        return Some(LinearPhaseDelta::Recompute);
    }
    let (&input_width, leading) = input.shape.split_last()?;
    if input_width != layer.in_features() {
        return Some(LinearPhaseDelta::Recompute);
    }
    let mut expected = leading.to_vec();
    expected.push(layer.out_features());
    if expected != output_shape {
        return Some(LinearPhaseDelta::Recompute);
    }
    let batches = tensor_elements(leading)?;
    let mut changed_by_batch = Vec::with_capacity(batches);
    let mut any_changed = false;
    for batch in 0..batches {
        let start = batch.checked_mul(input_width)?;
        let end = start.checked_add(input_width)?;
        let current = input.values.get(start..end)?;
        let previous = cached.input.values.get(start..end)?;
        let mut changed = Vec::new();
        let mut batch_depends = false;
        for (index, (left, right)) in current.iter().zip(previous).enumerate() {
            if !budget.poll_work() {
                return None;
            }
            if left.depends != right.depends {
                return Some(LinearPhaseDelta::Recompute);
            }
            if left.slope != right.slope || left.bias != right.bias {
                changed.push(index);
                any_changed = true;
            }
            batch_depends |= left.depends;
        }
        if changed
            .len()
            .saturating_mul(MAX_CHANGED_FRACTION_DENOMINATOR)
            > input_width
        {
            return Some(LinearPhaseDelta::Recompute);
        }
        changed_by_batch.push((changed, batch_depends));
    }
    if !any_changed {
        return Some(LinearPhaseDelta::Unchanged);
    }

    let mut output = clone_exact_tensor(&cached.output, budget)?;
    for (batch, (changed, batch_depends)) in changed_by_batch.iter().enumerate() {
        if changed.is_empty() {
            continue;
        }
        let input_start = batch.checked_mul(input_width)?;
        let current = input
            .values
            .get(input_start..input_start.checked_add(input_width)?)?;
        let previous = cached
            .input
            .values
            .get(input_start..input_start.checked_add(input_width)?)?;
        let mut deltas = Vec::with_capacity(changed.len());
        for &input_index in changed {
            let new_value = current.get(input_index)?;
            let old_value = previous.get(input_index)?;
            deltas.push((
                input_index,
                ExactAffine {
                    slope: budget.sub(&new_value.slope, &old_value.slope)?,
                    bias: budget.sub(&new_value.bias, &old_value.bias)?,
                    depends: false,
                },
            ));
        }
        for output_index in 0..layer.out_features() {
            let flat = batch
                .checked_mul(layer.out_features())?
                .checked_add(output_index)?;
            let mut updated = output.values.get(flat)?.clone();
            for (input_index, delta) in &deltas {
                if !budget.poll_work() {
                    return None;
                }
                if delta.slope.is_zero() && delta.bias.is_zero() {
                    continue;
                }
                let weight = bounded_exact_f32(layer.weight[[output_index, *input_index]], budget)?;
                if weight.is_zero() {
                    continue;
                }
                let scaled = budget.affine_scale(delta, &weight)?;
                updated = budget.affine_add(&updated, &scaled)?;
            }
            updated.depends = *batch_depends;
            *output.values.get_mut(flat)? = updated;
        }
    }
    Some(LinearPhaseDelta::Updated(output))
}

fn add_constant_tensor(
    input: &ExactTensor,
    layer: &crate::layers::AddConstantLayer,
    output_shape: &[usize],
    budget: &mut ExactBudget<'_>,
) -> Option<ExactTensor> {
    // Match `AddConstantLayer::propagate_ibp`: a rank-1 `[C]` constant on a
    // rank-3 `[C,H,W]` tensor is a channel bias and is reshaped to `[C,1,1]`
    // before broadcasting.  Ordinary trailing-axis broadcasting can also
    // accept `[C,H,C]`, so using the raw shape there would silently evaluate a
    // different function.
    let raw_constant_shape = layer.constant().shape();
    let channel_bias_shape = (raw_constant_shape.len() == 1
        && input.shape.len() == 3
        && raw_constant_shape[0] == input.shape[0])
        .then(|| vec![raw_constant_shape[0], 1, 1]);
    let constant_shape = channel_bias_shape.as_deref().unwrap_or(raw_constant_shape);
    if crate::shape::broadcast_shapes(&input.shape, constant_shape).as_deref() != Some(output_shape)
    {
        return None;
    }
    let mut constant_values = Vec::with_capacity(layer.constant().len());
    for &value in layer.constant() {
        if !budget.poll_work() {
            return None;
        }
        constant_values.push(ExactAffine::constant(bounded_exact_f32(value, budget)?));
    }
    let constant = ExactTensor {
        shape: constant_shape.to_vec(),
        values: constant_values,
    };
    binary_affine_tensor(
        input,
        &constant,
        output_shape,
        budget,
        |left, right, budget| budget.affine_add(left, right),
    )
}

fn resolved_reduction_axes(axes: &[i64], rank: usize) -> Option<Vec<usize>> {
    let source: Vec<i64> = if axes.is_empty() {
        (0..rank).map(|axis| axis as i64).collect()
    } else {
        axes.to_vec()
    };
    let rank_i64 = i64::try_from(rank).ok()?;
    let mut result = Vec::with_capacity(source.len());
    for axis in source {
        let resolved = if axis < 0 {
            usize::try_from(axis.checked_add(rank_i64)?).ok()?
        } else {
            usize::try_from(axis).ok()?
        };
        if resolved >= rank || result.contains(&resolved) {
            return None;
        }
        result.push(resolved);
    }
    Some(result)
}

fn reduce_sum_tensor(
    input: &ExactTensor,
    axes: &[i64],
    keepdims: bool,
    output_shape: &[usize],
    budget: &mut ExactBudget<'_>,
) -> Option<ExactTensor> {
    let reduced = resolved_reduction_axes(axes, input.shape.len())?;
    let expected = if keepdims {
        input
            .shape
            .iter()
            .enumerate()
            .map(|(axis, &dimension)| {
                if reduced.contains(&axis) {
                    1
                } else {
                    dimension
                }
            })
            .collect::<Vec<_>>()
    } else {
        input
            .shape
            .iter()
            .enumerate()
            .filter_map(|(axis, &dimension)| (!reduced.contains(&axis)).then_some(dimension))
            .collect::<Vec<_>>()
    };
    if expected != output_shape {
        return None;
    }
    let output_elements = tensor_elements(output_shape)?;
    let mut values = vec![ExactAffine::constant(BigRational::zero()); output_elements];
    let input_strides = strides(&input.shape)?;
    let output_strides = strides(output_shape)?;
    for input_flat in 0..input.values.len() {
        let coordinate = flat_coordinate(input_flat, &input.shape, &input_strides);
        let output_coordinate = if keepdims {
            coordinate
                .iter()
                .enumerate()
                .map(|(axis, &index)| if reduced.contains(&axis) { 0 } else { index })
                .collect::<Vec<_>>()
        } else {
            coordinate
                .iter()
                .enumerate()
                .filter_map(|(axis, &index)| (!reduced.contains(&axis)).then_some(index))
                .collect::<Vec<_>>()
        };
        let output_flat = coordinate_flat(&output_coordinate, &output_strides)?;
        values[output_flat] = budget.affine_add(&values[output_flat], &input.values[input_flat])?;
    }
    Some(ExactTensor {
        shape: output_shape.to_vec(),
        values,
    })
}

fn concat_tensor(
    inputs: &[&ExactTensor],
    layer: &crate::layers::ConcatLayer,
    output_shape: &[usize],
    budget: &mut ExactBudget<'_>,
) -> Option<ExactTensor> {
    let first = inputs.first()?;
    let axis = layer.normalize_axis(first.shape.len()).ok()?;
    let mut expected = first.shape.clone();
    for input in &inputs[1..] {
        if input.shape.len() != expected.len() {
            return None;
        }
        for dimension in 0..expected.len() {
            if dimension != axis && input.shape[dimension] != expected[dimension] {
                return None;
            }
        }
        expected[axis] = expected[axis].checked_add(input.shape[axis])?;
    }
    if expected != output_shape {
        return None;
    }
    let output_elements = tensor_elements(output_shape)?;
    let output_strides = strides(output_shape)?;
    let input_strides = inputs
        .iter()
        .map(|input| strides(&input.shape))
        .collect::<Option<Vec<_>>>()?;
    let mut values = Vec::with_capacity(output_elements);
    for flat in 0..output_elements {
        if !budget.poll_work() {
            return None;
        }
        let mut coordinate = flat_coordinate(flat, output_shape, &output_strides);
        let mut axis_index = coordinate[axis];
        let mut chosen = None;
        for (input, strides) in inputs.iter().zip(&input_strides) {
            if axis_index < input.shape[axis] {
                coordinate[axis] = axis_index;
                chosen = input
                    .values
                    .get(coordinate_flat(&coordinate, strides)?)
                    .cloned();
                break;
            }
            axis_index = axis_index.checked_sub(input.shape[axis])?;
        }
        values.push(chosen?);
    }
    Some(ExactTensor {
        shape: output_shape.to_vec(),
        values,
    })
}

struct PhaseContext<'point, 'budget, 'limits> {
    cursor: &'point BigRational,
    domain_upper: &'point BigRational,
    next_root: Option<BigRational>,
    relu_hasher: Sha256,
    relu_scalars: usize,
    budget: &'budget mut ExactBudget<'limits>,
}

impl PhaseContext<'_, '_, '_> {
    fn relu(
        &mut self,
        node: &str,
        input: &ExactTensor,
        output_shape: &[usize],
    ) -> Option<ExactTensor> {
        if input.shape != output_shape {
            return None;
        }
        let mut values = Vec::with_capacity(input.values.len());
        for (index, affine) in input.values.iter().enumerate() {
            self.relu_scalars = self.relu_scalars.checked_add(1)?;
            if self.relu_scalars > self.budget.limits.max_relu_scalars_per_phase {
                self.budget.failure = Some(OneAxisPhaseDeclineReason::ReluScalarLimit);
                return None;
            }
            let value = affine.value_at(self.cursor, self.budget)?;
            let active = value.is_positive() || (value.is_zero() && affine.slope.is_positive());
            hash_bytes(&mut self.relu_hasher, node.as_bytes());
            hash_usize(&mut self.relu_hasher, index);
            self.relu_hasher.update([u8::from(active)]);

            if !affine.slope.is_zero() {
                let negative_bias = self.budget.neg(&affine.bias)?;
                let root = self.budget.div(&negative_bias, &affine.slope)?;
                if root > *self.cursor && root <= *self.domain_upper {
                    match &self.next_root {
                        Some(current) if *current <= root => {}
                        _ => self.next_root = Some(root),
                    }
                }
            }
            values.push(if active {
                affine.clone()
            } else {
                ExactAffine {
                    slope: BigRational::zero(),
                    bias: BigRational::zero(),
                    depends: affine.depends,
                }
            });
        }
        Some(ExactTensor {
            shape: output_shape.to_vec(),
            values,
        })
    }
}

struct PhaseEvaluation {
    endpoint: BigRational,
    wrapper: WrapperValue,
    relu_phase_digest: [u8; 32],
    relu_scalars: usize,
}

fn phase_error(
    budget: &ExactBudget<'_>,
    fallback: OneAxisPhaseDeclineReason,
    node: &str,
) -> OneAxisPhaseDecline {
    decline(budget.failure.unwrap_or(fallback), Some(node))
}

fn expect_affine(value: &PhaseValue) -> Option<&ExactTensor> {
    match value {
        PhaseValue::Affine(tensor) => Some(tensor),
        _ => None,
    }
}

fn phase_value_is_static(value: &PhaseValue) -> bool {
    match value {
        PhaseValue::Affine(tensor) => tensor.values.iter().all(|affine| !affine.depends),
        PhaseValue::StaticSigmoid(_) => true,
        PhaseValue::DynamicSigmoid(_) | PhaseValue::Wrapper(_) => false,
    }
}

fn clone_exact_tensor(tensor: &ExactTensor, budget: &mut ExactBudget<'_>) -> Option<ExactTensor> {
    let mut values = Vec::with_capacity(tensor.values.len());
    for value in &tensor.values {
        if !budget.poll_work() {
            return None;
        }
        values.push(value.clone());
    }
    Some(ExactTensor {
        shape: tensor.shape.clone(),
        values,
    })
}

fn clone_phase_value(value: &PhaseValue, budget: &mut ExactBudget<'_>) -> Option<PhaseValue> {
    Some(match value {
        PhaseValue::Affine(tensor) => PhaseValue::Affine(clone_exact_tensor(tensor, budget)?),
        PhaseValue::StaticSigmoid(interval) => PhaseValue::StaticSigmoid(*interval),
        PhaseValue::DynamicSigmoid(affine) => PhaseValue::DynamicSigmoid(affine.clone()),
        PhaseValue::Wrapper(wrapper) => PhaseValue::Wrapper(wrapper.clone()),
    })
}

fn static_interval(value: &PhaseValue, deadline: Instant) -> Option<DirectedInterval> {
    match value {
        PhaseValue::Affine(tensor) if tensor.values.len() == 1 && !tensor.values[0].depends => {
            rational_enclosure(&tensor.values[0].bias)
        }
        PhaseValue::StaticSigmoid(interval) => Some(*interval),
        _ => None,
    }
    .filter(|_| Instant::now() < deadline)
}

fn negate_interval(interval: DirectedInterval) -> Option<DirectedInterval> {
    DirectedInterval::new(-interval.upper, -interval.lower)
}

fn evaluate_phase(
    graph: &GraphNetwork,
    problem: &OneAxisExactProblem,
    cursor: &BigRational,
    static_cache: &mut HashMap<String, PhaseValue>,
    linear_static_cache: &mut HashMap<String, Vec<ExactAffine>>,
    linear_phase_cache: &mut HashMap<String, LinearPhaseCache>,
    budget: &mut ExactBudget<'_>,
) -> Result<PhaseEvaluation, OneAxisPhaseDecline> {
    if !budget.check_deadline() {
        return Err(decline(OneAxisPhaseDeclineReason::Deadline, None));
    }
    let network_input = input_tensor(problem, budget).ok_or_else(|| {
        phase_error(
            budget,
            OneAxisPhaseDeclineReason::InvalidProblem,
            NETWORK_INPUT,
        )
    })?;
    let order = graph
        .exec_order()
        .map_err(|_| decline(OneAxisPhaseDeclineReason::GraphShape, None))?;
    if order.len() != graph.num_nodes() {
        return Err(decline(OneAxisPhaseDeclineReason::GraphShape, None));
    }
    let mut states = HashMap::<String, PhaseValue>::with_capacity(order.len());
    let mut context = PhaseContext {
        cursor,
        domain_upper: &problem.upper.0,
        next_root: None,
        relu_hasher: Sha256::new(),
        relu_scalars: 0,
        budget,
    };

    for name in order {
        if !context.budget.check_deadline() {
            return Err(decline(OneAxisPhaseDeclineReason::Deadline, Some(name)));
        }
        if let Some(cached) = static_cache.get(name) {
            let cloned = clone_phase_value(cached, context.budget).ok_or_else(|| {
                phase_error(context.budget, OneAxisPhaseDeclineReason::GraphShape, name)
            })?;
            states.insert(name.clone(), cloned);
            continue;
        }
        let node = graph
            .node(name)
            .ok_or_else(|| decline(OneAxisPhaseDeclineReason::GraphShape, Some(name)))?;
        let output_shape = graph
            .declared_shape(name)
            .ok_or_else(|| decline(OneAxisPhaseDeclineReason::GraphShape, Some(name)))?;
        if tensor_elements(output_shape).is_none() {
            return Err(decline(OneAxisPhaseDeclineReason::GraphShape, Some(name)));
        }
        let mut inputs = Vec::with_capacity(node.inputs().len());
        for input_name in node.inputs() {
            let cloned = if input_name == NETWORK_INPUT {
                clone_exact_tensor(&network_input, context.budget).map(PhaseValue::Affine)
            } else {
                states
                    .get(input_name)
                    .and_then(|value| clone_phase_value(value, context.budget))
            }
            .ok_or_else(|| {
                phase_error(context.budget, OneAxisPhaseDeclineReason::GraphShape, name)
            })?;
            inputs.push(cloned);
        }

        let value = match node.layer() {
            Layer::Slice(layer) => {
                let [input] = inputs.as_slice() else {
                    return Err(decline(OneAxisPhaseDeclineReason::GraphShape, Some(name)));
                };
                let input = expect_affine(input).ok_or_else(|| {
                    decline(OneAxisPhaseDeclineReason::UnsupportedAlgebra, Some(name))
                })?;
                PhaseValue::Affine(
                    slice_tensor(input, layer, output_shape, context.budget).ok_or_else(|| {
                        phase_error(context.budget, OneAxisPhaseDeclineReason::GraphShape, name)
                    })?,
                )
            }
            Layer::Linear(layer) => {
                let [input] = inputs.as_slice() else {
                    return Err(decline(OneAxisPhaseDeclineReason::GraphShape, Some(name)));
                };
                let input = expect_affine(input).ok_or_else(|| {
                    decline(OneAxisPhaseDeclineReason::UnsupportedAlgebra, Some(name))
                })?;
                let mut refresh_phase_cache = false;
                let cached_tensor = match linear_phase_cache.get(name) {
                    Some(cached) => {
                        let delta = linear_tensor_from_phase_delta(
                            input,
                            layer,
                            output_shape,
                            cached,
                            context.budget,
                        )
                        .ok_or_else(|| {
                            phase_error(context.budget, OneAxisPhaseDeclineReason::GraphShape, name)
                        })?;
                        match delta {
                            LinearPhaseDelta::Unchanged => Some(
                                clone_exact_tensor(&cached.output, context.budget).ok_or_else(
                                    || {
                                        phase_error(
                                            context.budget,
                                            OneAxisPhaseDeclineReason::GraphShape,
                                            name,
                                        )
                                    },
                                )?,
                            ),
                            LinearPhaseDelta::Recompute => None,
                            LinearPhaseDelta::Updated(tensor) => {
                                if cached.context_epoch != context.budget.context_epoch {
                                    context.budget.cross_context_sparse_linear_updates = context
                                        .budget
                                        .cross_context_sparse_linear_updates
                                        .saturating_add(1);
                                }
                                refresh_phase_cache = true;
                                Some(tensor)
                            }
                        }
                    }
                    None => None,
                };
                let tensor = if let Some(tensor) = cached_tensor {
                    tensor
                } else {
                    let cached_static = linear_static_cache.get(name).map(Vec::as_slice);
                    let (tensor, static_contributions) =
                        linear_tensor(input, layer, output_shape, cached_static, context.budget)
                            .ok_or_else(|| {
                                phase_error(
                                    context.budget,
                                    OneAxisPhaseDeclineReason::GraphShape,
                                    name,
                                )
                            })?;
                    linear_static_cache
                        .entry(name.clone())
                        .or_insert(static_contributions);
                    refresh_phase_cache = true;
                    tensor
                };
                if refresh_phase_cache {
                    let new_elements = input.values.len().checked_add(tensor.values.len());
                    let old_elements = linear_phase_cache
                        .get(name)
                        .and_then(LinearPhaseCache::retained_elements)
                        .unwrap_or(0);
                    if new_elements.is_some_and(|new_elements| {
                        context
                            .budget
                            .try_replace_phase_cache(old_elements, new_elements)
                    }) {
                        let cached_input =
                            clone_exact_tensor(input, context.budget).ok_or_else(|| {
                                phase_error(
                                    context.budget,
                                    OneAxisPhaseDeclineReason::GraphShape,
                                    name,
                                )
                            })?;
                        let cached_output = clone_exact_tensor(&tensor, context.budget)
                            .ok_or_else(|| {
                                phase_error(
                                    context.budget,
                                    OneAxisPhaseDeclineReason::GraphShape,
                                    name,
                                )
                            })?;
                        linear_phase_cache.insert(
                            name.clone(),
                            LinearPhaseCache {
                                input: cached_input,
                                output: cached_output,
                                context_epoch: context.budget.context_epoch,
                            },
                        );
                    }
                }
                PhaseValue::Affine(tensor)
            }
            Layer::AddConstant(layer) => {
                let [input] = inputs.as_slice() else {
                    return Err(decline(OneAxisPhaseDeclineReason::GraphShape, Some(name)));
                };
                let input = expect_affine(input).ok_or_else(|| {
                    decline(OneAxisPhaseDeclineReason::UnsupportedAlgebra, Some(name))
                })?;
                PhaseValue::Affine(
                    add_constant_tensor(input, layer, output_shape, context.budget).ok_or_else(
                        || phase_error(context.budget, OneAxisPhaseDeclineReason::GraphShape, name),
                    )?,
                )
            }
            Layer::ReduceSum(layer) => {
                let [input] = inputs.as_slice() else {
                    return Err(decline(OneAxisPhaseDeclineReason::GraphShape, Some(name)));
                };
                let input = expect_affine(input).ok_or_else(|| {
                    decline(OneAxisPhaseDeclineReason::UnsupportedAlgebra, Some(name))
                })?;
                PhaseValue::Affine(
                    reduce_sum_tensor(
                        input,
                        &layer.axes,
                        layer.keepdims,
                        output_shape,
                        context.budget,
                    )
                    .ok_or_else(|| {
                        phase_error(context.budget, OneAxisPhaseDeclineReason::GraphShape, name)
                    })?,
                )
            }
            Layer::Concat(layer) => {
                let affine_inputs = inputs
                    .iter()
                    .map(expect_affine)
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        decline(OneAxisPhaseDeclineReason::UnsupportedAlgebra, Some(name))
                    })?;
                PhaseValue::Affine(
                    concat_tensor(&affine_inputs, layer, output_shape, context.budget).ok_or_else(
                        || phase_error(context.budget, OneAxisPhaseDeclineReason::GraphShape, name),
                    )?,
                )
            }
            Layer::ReLU(_) => {
                let [input] = inputs.as_slice() else {
                    return Err(decline(OneAxisPhaseDeclineReason::GraphShape, Some(name)));
                };
                let input = expect_affine(input).ok_or_else(|| {
                    decline(OneAxisPhaseDeclineReason::UnsupportedAlgebra, Some(name))
                })?;
                PhaseValue::Affine(context.relu(name, input, output_shape).ok_or_else(|| {
                    phase_error(context.budget, OneAxisPhaseDeclineReason::GraphShape, name)
                })?)
            }
            Layer::MulBinary(_) => {
                let [left, right] = inputs.as_slice() else {
                    return Err(decline(OneAxisPhaseDeclineReason::GraphShape, Some(name)));
                };
                let (left, right) = (expect_affine(left), expect_affine(right));
                let (Some(left), Some(right)) = (left, right) else {
                    return Err(decline(
                        OneAxisPhaseDeclineReason::UnsupportedAlgebra,
                        Some(name),
                    ));
                };
                let tensor = binary_affine_tensor(
                    left,
                    right,
                    output_shape,
                    context.budget,
                    |left, right, budget| {
                        if left.depends && right.depends {
                            budget.failure = Some(OneAxisPhaseDeclineReason::DynamicMulOperands);
                            return None;
                        }
                        if !left.depends {
                            budget.affine_scale(right, &left.bias)
                        } else {
                            budget.affine_scale(left, &right.bias)
                        }
                    },
                )
                .ok_or_else(|| {
                    phase_error(context.budget, OneAxisPhaseDeclineReason::GraphShape, name)
                })?;
                PhaseValue::Affine(tensor)
            }
            Layer::Div(_) => {
                let [numerator, denominator] = inputs.as_slice() else {
                    return Err(decline(OneAxisPhaseDeclineReason::GraphShape, Some(name)));
                };
                let (Some(numerator), Some(denominator)) =
                    (expect_affine(numerator), expect_affine(denominator))
                else {
                    return Err(decline(
                        OneAxisPhaseDeclineReason::UnsupportedAlgebra,
                        Some(name),
                    ));
                };
                let tensor = binary_affine_tensor(
                    numerator,
                    denominator,
                    output_shape,
                    context.budget,
                    |numerator, denominator, budget| {
                        if denominator.depends || !denominator.bias.is_positive() {
                            budget.failure = Some(OneAxisPhaseDeclineReason::InvalidDivisor);
                            return None;
                        }
                        let reciprocal = budget.div(&BigRational::one(), &denominator.bias)?;
                        budget.affine_scale(numerator, &reciprocal)
                    },
                )
                .ok_or_else(|| {
                    phase_error(context.budget, OneAxisPhaseDeclineReason::GraphShape, name)
                })?;
                PhaseValue::Affine(tensor)
            }
            Layer::Sub(_) => {
                let [left, right] = inputs.as_slice() else {
                    return Err(decline(OneAxisPhaseDeclineReason::GraphShape, Some(name)));
                };
                match (left, right) {
                    (PhaseValue::Affine(left), PhaseValue::Affine(right)) => PhaseValue::Affine(
                        binary_affine_tensor(
                            left,
                            right,
                            output_shape,
                            context.budget,
                            |left, right, budget| budget.affine_sub(left, right),
                        )
                        .ok_or_else(|| {
                            phase_error(context.budget, OneAxisPhaseDeclineReason::GraphShape, name)
                        })?,
                    ),
                    (PhaseValue::DynamicSigmoid(core), static_value) => {
                        if tensor_elements(output_shape) != Some(1) {
                            return Err(decline(
                                OneAxisPhaseDeclineReason::NonScalarOutput,
                                Some(name),
                            ));
                        }
                        let offset = negate_interval(
                            static_interval(static_value, context.budget.deadline).ok_or_else(
                                || {
                                    decline(
                                        if Instant::now() >= context.budget.deadline {
                                            OneAxisPhaseDeclineReason::Deadline
                                        } else {
                                            OneAxisPhaseDeclineReason::UnsupportedAlgebra
                                        },
                                        Some(name),
                                    )
                                },
                            )?,
                        )
                        .ok_or_else(|| {
                            decline(
                                if Instant::now() >= context.budget.deadline {
                                    OneAxisPhaseDeclineReason::Deadline
                                } else {
                                    OneAxisPhaseDeclineReason::DirectedArithmetic
                                },
                                Some(name),
                            )
                        })?;
                        PhaseValue::Wrapper(WrapperValue {
                            offset,
                            sign: 1,
                            core: core.clone(),
                        })
                    }
                    (static_value, PhaseValue::DynamicSigmoid(core)) => {
                        if tensor_elements(output_shape) != Some(1) {
                            return Err(decline(
                                OneAxisPhaseDeclineReason::NonScalarOutput,
                                Some(name),
                            ));
                        }
                        let offset = static_interval(static_value, context.budget.deadline)
                            .ok_or_else(|| {
                                decline(
                                    if Instant::now() >= context.budget.deadline {
                                        OneAxisPhaseDeclineReason::Deadline
                                    } else {
                                        OneAxisPhaseDeclineReason::UnsupportedAlgebra
                                    },
                                    Some(name),
                                )
                            })?;
                        PhaseValue::Wrapper(WrapperValue {
                            offset,
                            sign: -1,
                            core: core.clone(),
                        })
                    }
                    _ => {
                        return Err(decline(
                            OneAxisPhaseDeclineReason::UnsupportedAlgebra,
                            Some(name),
                        ));
                    }
                }
            }
            Layer::Sigmoid(_) => {
                let [input] = inputs.as_slice() else {
                    return Err(decline(OneAxisPhaseDeclineReason::GraphShape, Some(name)));
                };
                let input = expect_affine(input).ok_or_else(|| {
                    decline(OneAxisPhaseDeclineReason::UnsupportedAlgebra, Some(name))
                })?;
                if input.values.len() != 1 || tensor_elements(output_shape) != Some(1) {
                    return Err(decline(
                        OneAxisPhaseDeclineReason::NonScalarOutput,
                        Some(name),
                    ));
                }
                let affine = &input.values[0];
                if !affine.depends {
                    let exact = rational_enclosure(&affine.bias).ok_or_else(|| {
                        decline(OneAxisPhaseDeclineReason::DirectedArithmetic, Some(name))
                    })?;
                    PhaseValue::StaticSigmoid(
                        sigmoid_enclosure(exact, context.budget.deadline).ok_or_else(|| {
                            decline(
                                if Instant::now() >= context.budget.deadline {
                                    OneAxisPhaseDeclineReason::Deadline
                                } else {
                                    OneAxisPhaseDeclineReason::DirectedArithmetic
                                },
                                Some(name),
                            )
                        })?,
                    )
                } else {
                    PhaseValue::DynamicSigmoid(affine.clone())
                }
            }
            _ => {
                return Err(decline(
                    OneAxisPhaseDeclineReason::UnsupportedAlgebra,
                    Some(name),
                ));
            }
        };
        // ReLU decisions are part of the phase digest, including structurally
        // static ReLUs, so those nodes are deliberately re-evaluated.  Any
        // static non-ReLU descendant can still be cached after that decision.
        if !matches!(node.layer(), Layer::ReLU(_)) && phase_value_is_static(&value) {
            let cached = clone_phase_value(&value, context.budget).ok_or_else(|| {
                phase_error(context.budget, OneAxisPhaseDeclineReason::GraphShape, name)
            })?;
            static_cache.insert(name.clone(), cached);
        }
        states.insert(name.clone(), value);
    }

    let output = states
        .remove(graph.output_name())
        .ok_or_else(|| decline(OneAxisPhaseDeclineReason::GraphShape, None))?;
    let wrapper = match output {
        PhaseValue::DynamicSigmoid(core) => WrapperValue {
            offset: DirectedInterval {
                lower: 0.0,
                upper: 0.0,
            },
            sign: 1,
            core,
        },
        PhaseValue::Wrapper(wrapper) => wrapper,
        _ => {
            return Err(decline(
                OneAxisPhaseDeclineReason::UnsupportedAlgebra,
                Some(graph.output_name()),
            ));
        }
    };
    let endpoint = context.next_root.unwrap_or_else(|| problem.upper.0.clone());
    if endpoint < *cursor
        || endpoint > problem.upper.0
        || (endpoint == *cursor && cursor != &problem.upper.0)
    {
        return Err(decline(
            OneAxisPhaseDeclineReason::GraphShape,
            Some(graph.output_name()),
        ));
    }
    Ok(PhaseEvaluation {
        endpoint,
        wrapper,
        relu_phase_digest: context.relu_hasher.finalize().into(),
        relu_scalars: context.relu_scalars,
    })
}

fn exact_from_f64(value: f64) -> Option<OneAxisRational> {
    value
        .is_finite()
        .then(|| BigRational::from_float(value))
        .flatten()
        .map(OneAxisRational)
}

fn wrapper_enclosure(wrapper: &WrapperValue) -> OneAxisWrapperEnclosure {
    OneAxisWrapperEnclosure {
        offset_lower: wrapper.offset.lower,
        offset_upper: wrapper.offset.upper,
        sigmoid_sign: wrapper.sign,
    }
}

fn same_wrapper(left: OneAxisWrapperEnclosure, right: OneAxisWrapperEnclosure) -> bool {
    left.sigmoid_sign == right.sigmoid_sign
        && left.offset_lower.to_bits() == right.offset_lower.to_bits()
        && left.offset_upper.to_bits() == right.offset_upper.to_bits()
}

fn peel_relation(
    relation: OneAxisConstraintRelation,
    probability: DirectedInterval,
    deadline: Instant,
) -> Option<OneAxisPeeledConstraint> {
    let point_logit = |value: f64| logit_enclosure(DirectedInterval::point(value)?, deadline);
    let guard_le = |value: f64, upper: bool| {
        let enclosure = point_logit(value)?;
        exact_from_f64(if upper {
            enclosure.upper
        } else {
            enclosure.lower
        })
        .map(OneAxisCoreGuard::LessEqual)
    };
    let guard_ge = |value: f64, lower: bool| {
        let enclosure = point_logit(value)?;
        exact_from_f64(if lower {
            enclosure.lower
        } else {
            enclosure.upper
        })
        .map(OneAxisCoreGuard::GreaterEqual)
    };

    match relation {
        OneAxisConstraintRelation::LessEqual => {
            let necessary = if probability.upper <= 0.0 {
                OneAxisCoreGuard::Impossible
            } else if probability.upper >= 1.0 {
                OneAxisCoreGuard::Always
            } else {
                guard_le(probability.upper, true)?
            };
            let sufficient = if probability.lower <= 0.0 {
                OneAxisCoreGuard::Impossible
            } else if probability.lower >= 1.0 {
                OneAxisCoreGuard::Always
            } else {
                guard_le(probability.lower, false)?
            };
            Some(OneAxisPeeledConstraint {
                necessary,
                sufficient,
            })
        }
        OneAxisConstraintRelation::GreaterEqual => {
            let necessary = if probability.lower <= 0.0 {
                OneAxisCoreGuard::Always
            } else if probability.lower >= 1.0 {
                OneAxisCoreGuard::Impossible
            } else {
                guard_ge(probability.lower, true)?
            };
            let sufficient = if probability.upper <= 0.0 {
                OneAxisCoreGuard::Always
            } else if probability.upper >= 1.0 {
                OneAxisCoreGuard::Impossible
            } else {
                guard_ge(probability.upper, false)?
            };
            Some(OneAxisPeeledConstraint {
                necessary,
                sufficient,
            })
        }
    }
}

fn peel_constraints(
    problem: &OneAxisExactProblem,
    wrapper: &WrapperValue,
    deadline: Instant,
) -> Option<Vec<OneAxisPeeledConstraint>> {
    let mut result = Vec::with_capacity(problem.constraints.len());
    for constraint in &problem.constraints {
        if Instant::now() >= deadline {
            return None;
        }
        let bound = rational_enclosure(&constraint.bound.0)?;
        let (relation, probability) = match wrapper.sign {
            1 => (constraint.relation, bound.sub(wrapper.offset)?),
            -1 => {
                let flipped = match constraint.relation {
                    OneAxisConstraintRelation::LessEqual => OneAxisConstraintRelation::GreaterEqual,
                    OneAxisConstraintRelation::GreaterEqual => OneAxisConstraintRelation::LessEqual,
                };
                (flipped, wrapper.offset.sub(bound)?)
            }
            _ => return None,
        };
        result.push(peel_relation(relation, probability, deadline)?);
    }
    Some(result)
}

#[derive(Clone)]
struct ExactRegion {
    lower: BigRational,
    upper: BigRational,
}

impl ExactRegion {
    fn apply(
        &mut self,
        affine: &ExactAffine,
        guard: &OneAxisCoreGuard,
        budget: &mut ExactBudget<'_>,
    ) -> Option<bool> {
        let (relation, bound) = match guard {
            OneAxisCoreGuard::Always => return Some(true),
            OneAxisCoreGuard::Impossible => return Some(false),
            OneAxisCoreGuard::LessEqual(bound) => (OneAxisConstraintRelation::LessEqual, &bound.0),
            OneAxisCoreGuard::GreaterEqual(bound) => {
                (OneAxisConstraintRelation::GreaterEqual, &bound.0)
            }
        };
        if affine.slope.is_zero() {
            return Some(match relation {
                OneAxisConstraintRelation::LessEqual => affine.bias <= *bound,
                OneAxisConstraintRelation::GreaterEqual => affine.bias >= *bound,
            });
        }
        let numerator = budget.sub(bound, &affine.bias)?;
        let root = budget.div(&numerator, &affine.slope)?;
        match (relation, affine.slope.is_positive()) {
            (OneAxisConstraintRelation::LessEqual, true)
            | (OneAxisConstraintRelation::GreaterEqual, false) => {
                if root < self.upper {
                    self.upper = root;
                }
            }
            (OneAxisConstraintRelation::LessEqual, false)
            | (OneAxisConstraintRelation::GreaterEqual, true) => {
                if root > self.lower {
                    self.lower = root;
                }
            }
        }
        Some(self.lower <= self.upper)
    }
}

fn observation_from_cells(
    cells: &[OneAxisPhaseCellCertificate],
    peeled: &[OneAxisPeeledConstraint],
    budget: &mut ExactBudget<'_>,
) -> Option<OneAxisPhaseObservation> {
    let mut necessary_nonempty = false;
    let mut witness = None;
    for cell in cells {
        let affine = ExactAffine {
            slope: cell.core.slope.0.clone(),
            bias: cell.core.bias.0.clone(),
            depends: true,
        };
        let initial = ExactRegion {
            lower: cell.lower.0.clone(),
            upper: cell.upper.0.clone(),
        };
        let mut necessary = initial.clone();
        let mut necessary_ok = true;
        for constraint in peeled {
            if !necessary.apply(&affine, &constraint.necessary, budget)? {
                necessary_ok = false;
                break;
            }
        }
        necessary_nonempty |= necessary_ok;

        if witness.is_none() {
            let mut sufficient = initial;
            let mut sufficient_ok = true;
            for constraint in peeled {
                if !sufficient.apply(&affine, &constraint.sufficient, budget)? {
                    sufficient_ok = false;
                    break;
                }
            }
            if sufficient_ok {
                witness = Some(OneAxisRational(sufficient.lower));
            }
        }
    }
    Some(match witness {
        Some(free_value) => OneAxisPhaseObservation::ExactWitness { free_value },
        None if !necessary_nonempty => OneAxisPhaseObservation::CertifiedEmpty,
        None => OneAxisPhaseObservation::Inconclusive,
    })
}

fn preflight_graph(
    graph: &GraphNetwork,
    problem: &OneAxisExactProblem,
    limits: &OneAxisPhaseLimits,
    deadline: Instant,
) -> Result<(), OneAxisPhaseDecline> {
    if graph.num_nodes() == 0 || graph.num_nodes() > ONE_AXIS_MAX_NODES {
        return Err(decline(OneAxisPhaseDeclineReason::StructuralRefusal, None));
    }
    let mut edges = 0usize;
    let mut retained = tensor_elements(&problem.input_shape)
        .ok_or_else(|| decline(OneAxisPhaseDeclineReason::InvalidProblem, None))?;
    for name in graph.node_names() {
        if Instant::now() >= deadline {
            return Err(decline(OneAxisPhaseDeclineReason::Deadline, None));
        }
        let node = graph
            .node(name)
            .ok_or_else(|| decline(OneAxisPhaseDeclineReason::GraphShape, Some(name)))?;
        edges = edges
            .checked_add(node.inputs().len())
            .ok_or_else(|| decline(OneAxisPhaseDeclineReason::StructuralRefusal, Some(name)))?;
        if edges > ONE_AXIS_MAX_EDGES {
            return Err(decline(
                OneAxisPhaseDeclineReason::StructuralRefusal,
                Some(name),
            ));
        }
        let elements = graph
            .declared_shape(name)
            .and_then(tensor_elements)
            .ok_or_else(|| decline(OneAxisPhaseDeclineReason::GraphShape, Some(name)))?;
        retained = retained
            .checked_add(elements)
            .ok_or_else(|| decline(OneAxisPhaseDeclineReason::StructuralRefusal, Some(name)))?;
        if elements > ONE_AXIS_MAX_TENSOR_ELEMENTS
            || elements > limits.max_tensor_elements
            || retained > ONE_AXIS_MAX_TOTAL_ELEMENTS
            || retained > limits.max_total_tensor_elements
        {
            return Err(decline(
                OneAxisPhaseDeclineReason::StructuralRefusal,
                Some(name),
            ));
        }
    }
    let recognition = graph.recognize_one_free_axis_algebra_until(
        &problem.input_shape,
        problem.free_axis,
        deadline,
    );
    if recognition.class != Some(OneAxisAlgebraClass::PeelableMonotoneSigmoid)
        || recognition.decline.is_some()
    {
        return Err(decline(
            if Instant::now() >= deadline {
                OneAxisPhaseDeclineReason::Deadline
            } else {
                OneAxisPhaseDeclineReason::StructuralRefusal
            },
            recognition
                .decline
                .as_ref()
                .and_then(|item| item.node.as_deref()),
        ));
    }
    Ok(())
}

fn attempt_decline(reason: OneAxisPhaseDecline) -> OneAxisPhaseAttempt {
    OneAxisPhaseAttempt {
        certificate: None,
        decline: Some(reason),
        phase_cells_examined: 0,
        exact_operations: 0,
    }
}

fn attempt_decline_after(
    reason: OneAxisPhaseDecline,
    phase_cells_examined: usize,
    exact_operations: usize,
) -> OneAxisPhaseAttempt {
    OneAxisPhaseAttempt {
        certificate: None,
        decline: Some(reason),
        phase_cells_examined,
        exact_operations,
    }
}

fn certificate_cell(
    lower: &BigRational,
    evaluation: &PhaseEvaluation,
) -> OneAxisPhaseCellCertificate {
    OneAxisPhaseCellCertificate {
        lower: OneAxisRational(lower.clone()),
        upper: OneAxisRational(evaluation.endpoint.clone()),
        core: OneAxisAffineCertificate {
            slope: OneAxisRational(evaluation.wrapper.core.slope.clone()),
            bias: OneAxisRational(evaluation.wrapper.core.bias.clone()),
        },
        relu_phase_digest: evaluation.relu_phase_digest,
        relu_scalars: evaluation.relu_scalars,
    }
}

fn public_rational_within_limit(value: &OneAxisRational, limits: &OneAxisPhaseLimits) -> bool {
    value.0.numer().bits().max(value.0.denom().bits()) <= limits.max_rational_bits
}

fn guard_within_limit(guard: &OneAxisCoreGuard, limits: &OneAxisPhaseLimits) -> bool {
    match guard {
        OneAxisCoreGuard::Always | OneAxisCoreGuard::Impossible => true,
        OneAxisCoreGuard::LessEqual(bound) | OneAxisCoreGuard::GreaterEqual(bound) => {
            public_rational_within_limit(bound, limits)
        }
    }
}

fn certificate_within_limits(
    certificate: &OneAxisPhaseCertificate,
    limits: &OneAxisPhaseLimits,
    deadline: Instant,
) -> bool {
    if Instant::now() >= deadline
        || certificate.cells.len() > limits.max_phase_cells
        || certificate.peeled_constraints.len() > limits.max_constraints
        || !certificate.wrapper.offset_lower.is_finite()
        || !certificate.wrapper.offset_upper.is_finite()
        || certificate.wrapper.offset_lower > certificate.wrapper.offset_upper
        || !matches!(certificate.wrapper.sigmoid_sign, -1 | 1)
    {
        return false;
    }
    for (index, cell) in certificate.cells.iter().enumerate() {
        if index.is_multiple_of(64) && Instant::now() >= deadline {
            return false;
        }
        if cell.relu_scalars > limits.max_relu_scalars_per_phase
            || !public_rational_within_limit(&cell.lower, limits)
            || !public_rational_within_limit(&cell.upper, limits)
            || !public_rational_within_limit(&cell.core.slope, limits)
            || !public_rational_within_limit(&cell.core.bias, limits)
            || cell.lower.0 > cell.upper.0
        {
            return false;
        }
    }
    for constraint in &certificate.peeled_constraints {
        if !guard_within_limit(&constraint.necessary, limits)
            || !guard_within_limit(&constraint.sufficient, limits)
        {
            return false;
        }
    }
    match &certificate.observation {
        OneAxisPhaseObservation::ExactWitness { free_value } => {
            public_rational_within_limit(free_value, limits)
        }
        OneAxisPhaseObservation::CertifiedEmpty | OneAxisPhaseObservation::Inconclusive => true,
    }
}

impl GraphNetwork {
    /// Generate a bounded, verdict-neutral exact one-axis phase certificate.
    pub fn exact_one_axis_phase_certificate_until(
        &self,
        problem: &OneAxisExactProblem,
        limits: OneAxisPhaseLimits,
        deadline: Instant,
    ) -> OneAxisPhaseAttempt {
        if let Err(reason) = validate_problem(problem, &limits, deadline) {
            return attempt_decline(decline(reason, None));
        }
        if let Err(reason) = preflight_graph(self, problem, &limits, deadline) {
            return attempt_decline(reason);
        }
        let Some(admitted_graph_digest) = graph_digest(self, problem, deadline) else {
            return attempt_decline(decline(
                if Instant::now() >= deadline {
                    OneAxisPhaseDeclineReason::Deadline
                } else {
                    OneAxisPhaseDeclineReason::StructuralRefusal
                },
                None,
            ));
        };
        let Some(admitted_problem_digest) = problem_digest(problem, deadline) else {
            return attempt_decline(decline(OneAxisPhaseDeclineReason::Deadline, None));
        };
        let mut budget = ExactBudget::new(&limits, deadline);
        let mut cursor = problem.lower.0.clone();
        let mut cells = Vec::new();
        let mut global_wrapper = None;
        let mut static_cache = HashMap::new();
        let mut linear_static_cache = HashMap::new();
        let mut linear_phase_cache = HashMap::new();

        loop {
            if cells.len() >= limits.max_phase_cells {
                return attempt_decline_after(
                    decline(OneAxisPhaseDeclineReason::PhaseCellLimit, None),
                    cells.len(),
                    budget.operations,
                );
            }
            let evaluation = match evaluate_phase(
                self,
                problem,
                &cursor,
                &mut static_cache,
                &mut linear_static_cache,
                &mut linear_phase_cache,
                &mut budget,
            ) {
                Ok(value) => value,
                Err(reason) => {
                    return attempt_decline_after(reason, cells.len(), budget.operations)
                }
            };
            let current_wrapper = wrapper_enclosure(&evaluation.wrapper);
            match global_wrapper {
                Some(expected) if !same_wrapper(expected, current_wrapper) => {
                    return attempt_decline_after(
                        decline(
                            OneAxisPhaseDeclineReason::ReplayMismatch,
                            Some(self.output_name()),
                        ),
                        cells.len(),
                        budget.operations,
                    );
                }
                None => global_wrapper = Some(current_wrapper),
                _ => {}
            }
            let endpoint = evaluation.endpoint.clone();
            cells.push(certificate_cell(&cursor, &evaluation));
            if cursor == problem.upper.0 {
                break;
            }
            cursor = endpoint;
            if cursor == problem.upper.0 {
                break;
            }
        }

        let Some(wrapper) = global_wrapper else {
            return attempt_decline_after(
                decline(OneAxisPhaseDeclineReason::CertificateMalformed, None),
                cells.len(),
                budget.operations,
            );
        };
        let wrapper_value = WrapperValue {
            offset: DirectedInterval {
                lower: wrapper.offset_lower,
                upper: wrapper.offset_upper,
            },
            sign: wrapper.sigmoid_sign,
            core: ExactAffine::constant(BigRational::zero()),
        };
        let Some(peeled_constraints) = peel_constraints(problem, &wrapper_value, budget.deadline)
        else {
            return attempt_decline_after(
                decline(
                    if Instant::now() >= deadline {
                        OneAxisPhaseDeclineReason::Deadline
                    } else {
                        OneAxisPhaseDeclineReason::DirectedArithmetic
                    },
                    None,
                ),
                cells.len(),
                budget.operations,
            );
        };
        if peeled_constraints.iter().any(|constraint| {
            !guard_within_limit(&constraint.necessary, &limits)
                || !guard_within_limit(&constraint.sufficient, &limits)
        }) {
            return attempt_decline_after(
                decline(OneAxisPhaseDeclineReason::RationalBitLimit, None),
                cells.len(),
                budget.operations,
            );
        }
        let Some(observation) = observation_from_cells(&cells, &peeled_constraints, &mut budget)
        else {
            return attempt_decline_after(
                decline(
                    budget
                        .failure
                        .unwrap_or(OneAxisPhaseDeclineReason::ExactOperationLimit),
                    None,
                ),
                cells.len(),
                budget.operations,
            );
        };
        let phase_cells_examined = cells.len();
        let exact_operations = budget.operations;
        OneAxisPhaseAttempt {
            certificate: Some(OneAxisPhaseCertificate {
                version: ONE_AXIS_PHASE_CERTIFICATE_VERSION,
                verdict_authority: false,
                problem_digest: admitted_problem_digest,
                graph_digest: admitted_graph_digest,
                cells,
                wrapper,
                peeled_constraints,
                observation,
            }),
            decline: None,
            phase_cells_examined,
            exact_operations,
        }
    }

    /// Replay an untrusted certificate by rebuilding every right-local phase.
    ///
    /// Replay does not invoke the generator or trust supplied phase endpoints,
    /// affine maps, wrapper bounds, peeled guards, or observation.
    pub fn replay_exact_one_axis_phase_certificate_until(
        &self,
        problem: &OneAxisExactProblem,
        certificate: &OneAxisPhaseCertificate,
        limits: OneAxisPhaseLimits,
        deadline: Instant,
    ) -> OneAxisReplayResult {
        let reject = |reason, node| OneAxisReplayResult {
            accepted: false,
            observation: None,
            decline: Some(decline(reason, node)),
        };
        if Instant::now() >= deadline {
            return reject(OneAxisPhaseDeclineReason::Deadline, None);
        }
        if certificate.version != ONE_AXIS_PHASE_CERTIFICATE_VERSION
            || certificate.verdict_authority
            || certificate.cells.is_empty()
            || !certificate_within_limits(certificate, &limits, deadline)
        {
            return reject(OneAxisPhaseDeclineReason::CertificateMalformed, None);
        }
        if let Err(reason) = validate_problem(problem, &limits, deadline) {
            return reject(reason, None);
        }
        let Some(admitted_problem_digest) = problem_digest(problem, deadline) else {
            return reject(OneAxisPhaseDeclineReason::Deadline, None);
        };
        if certificate.problem_digest != admitted_problem_digest {
            return reject(OneAxisPhaseDeclineReason::ProblemDigestMismatch, None);
        }
        if let Err(reason) = preflight_graph(self, problem, &limits, deadline) {
            return OneAxisReplayResult {
                accepted: false,
                observation: None,
                decline: Some(reason),
            };
        }
        let Some(admitted_graph_digest) = graph_digest(self, problem, deadline) else {
            return reject(
                if Instant::now() >= deadline {
                    OneAxisPhaseDeclineReason::Deadline
                } else {
                    OneAxisPhaseDeclineReason::StructuralRefusal
                },
                None,
            );
        };
        if certificate.graph_digest != admitted_graph_digest {
            return reject(OneAxisPhaseDeclineReason::ReplayMismatch, None);
        }

        let mut budget = ExactBudget::new(&limits, deadline);
        let mut cursor = problem.lower.0.clone();
        let mut rebuilt_cells = Vec::with_capacity(certificate.cells.len());
        let mut rebuilt_wrapper = None;
        let mut static_cache = HashMap::new();
        let mut linear_static_cache = HashMap::new();
        let mut linear_phase_cache = HashMap::new();
        for supplied in &certificate.cells {
            if supplied.lower.0 != cursor {
                return reject(OneAxisPhaseDeclineReason::ReplayMismatch, None);
            }
            let evaluation = match evaluate_phase(
                self,
                problem,
                &cursor,
                &mut static_cache,
                &mut linear_static_cache,
                &mut linear_phase_cache,
                &mut budget,
            ) {
                Ok(value) => value,
                Err(reason) => {
                    return OneAxisReplayResult {
                        accepted: false,
                        observation: None,
                        decline: Some(reason),
                    }
                }
            };
            let expected = certificate_cell(&cursor, &evaluation);
            if supplied != &expected {
                return reject(OneAxisPhaseDeclineReason::ReplayMismatch, None);
            }
            let current_wrapper = wrapper_enclosure(&evaluation.wrapper);
            match rebuilt_wrapper {
                Some(wrapper) if !same_wrapper(wrapper, current_wrapper) => {
                    return reject(
                        OneAxisPhaseDeclineReason::ReplayMismatch,
                        Some(self.output_name()),
                    );
                }
                None => rebuilt_wrapper = Some(current_wrapper),
                _ => {}
            }
            cursor = evaluation.endpoint.clone();
            rebuilt_cells.push(expected);
            if cursor == problem.upper.0 {
                break;
            }
        }
        if cursor != problem.upper.0 || rebuilt_cells.len() != certificate.cells.len() {
            return reject(OneAxisPhaseDeclineReason::ReplayMismatch, None);
        }
        let Some(wrapper) = rebuilt_wrapper else {
            return reject(OneAxisPhaseDeclineReason::CertificateMalformed, None);
        };
        if !same_wrapper(wrapper, certificate.wrapper) {
            return reject(OneAxisPhaseDeclineReason::ReplayMismatch, None);
        }
        let wrapper_value = WrapperValue {
            offset: DirectedInterval {
                lower: wrapper.offset_lower,
                upper: wrapper.offset_upper,
            },
            sign: wrapper.sigmoid_sign,
            core: ExactAffine::constant(BigRational::zero()),
        };
        let Some(peeled) = peel_constraints(problem, &wrapper_value, budget.deadline) else {
            return reject(
                if Instant::now() >= deadline {
                    OneAxisPhaseDeclineReason::Deadline
                } else {
                    OneAxisPhaseDeclineReason::DirectedArithmetic
                },
                None,
            );
        };
        if peeled != certificate.peeled_constraints {
            return reject(OneAxisPhaseDeclineReason::ReplayMismatch, None);
        }
        let Some(observation) = observation_from_cells(&rebuilt_cells, &peeled, &mut budget) else {
            return reject(
                budget
                    .failure
                    .unwrap_or(OneAxisPhaseDeclineReason::ExactOperationLimit),
                None,
            );
        };
        if observation != certificate.observation {
            return reject(OneAxisPhaseDeclineReason::ReplayMismatch, None);
        }
        OneAxisReplayResult {
            accepted: true,
            observation: Some(observation),
            decline: None,
        }
    }
}

/// Verdict-neutral composition of independently replayed DNF clauses.
pub fn compose_one_axis_dnf_observations(
    clauses: &[OneAxisPhaseObservation],
) -> OneAxisPhaseObservation {
    for clause in clauses {
        if let OneAxisPhaseObservation::ExactWitness { free_value } = clause {
            return OneAxisPhaseObservation::ExactWitness {
                free_value: free_value.clone(),
            };
        }
    }
    if !clauses.is_empty()
        && clauses
            .iter()
            .all(|clause| *clause == OneAxisPhaseObservation::CertifiedEmpty)
    {
        OneAxisPhaseObservation::CertifiedEmpty
    } else {
        OneAxisPhaseObservation::Inconclusive
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use ndarray::{arr0, arr1, arr2};
    use proptest::prelude::*;

    use super::*;
    use crate::layers::{
        AddConstantLayer, ConcatLayer, LinearLayer, ReLULayer, SigmoidLayer, SliceLayer, SubLayer,
    };
    use crate::network::GraphNode;

    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(10)
    }

    fn exact(text: &str) -> OneAxisRational {
        OneAxisRational::parse_decimal(text).expect("exact decimal")
    }

    fn shape(graph: &mut GraphNetwork, name: &str, dimensions: &[usize]) {
        graph.set_declared_shape(name, dimensions.to_vec());
    }

    fn nested_relu_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        shape(&mut graph, NETWORK_INPUT, &[1]);
        graph.add_node(GraphNode::from_input("r1", Layer::ReLU(ReLULayer)));
        shape(&mut graph, "r1", &[1]);
        graph.add_node(GraphNode::from_input(
            "shift",
            Layer::AddConstant(AddConstantLayer::new(arr0(-1.0_f32).into_dyn())),
        ));
        shape(&mut graph, "shift", &[1]);
        graph.add_node(GraphNode::new(
            "r2",
            Layer::ReLU(ReLULayer),
            vec!["shift".to_string()],
        ));
        shape(&mut graph, "r2", &[1]);
        graph.add_node(GraphNode::binary(
            "concat",
            Layer::Concat(ConcatLayer::new(0)),
            "r1",
            "r2",
        ));
        shape(&mut graph, "concat", &[2]);
        graph.add_node(GraphNode::new(
            "linear",
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.0_f32, -2.0]]), Some(arr1(&[-0.5])))
                    .expect("valid linear"),
            ),
            vec!["concat".to_string()],
        ));
        shape(&mut graph, "linear", &[1]);
        graph.add_node(GraphNode::new(
            "r3",
            Layer::ReLU(ReLULayer),
            vec!["linear".to_string()],
        ));
        shape(&mut graph, "r3", &[1]);
        graph.add_node(GraphNode::new(
            "sigmoid",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["r3".to_string()],
        ));
        shape(&mut graph, "sigmoid", &[1]);
        graph.set_output("sigmoid");
        graph
    }

    fn scalar_problem(lower: &str, upper: &str) -> OneAxisExactProblem {
        OneAxisExactProblem {
            input_shape: vec![1],
            fixed_inputs: vec![exact("0")],
            free_axis: 0,
            lower: exact(lower),
            upper: exact(upper),
            constraints: vec![OneAxisOutputConstraint {
                relation: OneAxisConstraintRelation::GreaterEqual,
                bound: exact("0.5"),
            }],
        }
    }

    #[test]
    fn decimal_parser_is_exact_and_bounded() {
        assert_eq!(
            exact("-1.25e2"),
            OneAxisRational::new((-125).into(), 1.into()).unwrap()
        );
        assert_eq!(
            exact("0.001"),
            OneAxisRational::new(1.into(), 1000.into()).unwrap()
        );
        assert!(OneAxisRational::parse_decimal("NaN").is_none());
        assert!(OneAxisRational::parse_decimal("1e5000").is_none());
    }

    #[test]
    fn complete_nested_relu_sweep_finds_all_downstream_roots() {
        let graph = nested_relu_graph();
        let problem = scalar_problem("-1", "2");
        let attempt = graph.exact_one_axis_phase_certificate_until(
            &problem,
            OneAxisPhaseLimits::default(),
            deadline(),
        );
        let certificate = attempt.certificate.expect("phase certificate");
        assert_eq!(attempt.decline, None);
        assert!(!certificate.verdict_authority);
        let endpoints = certificate
            .cells
            .iter()
            .map(|cell| cell.upper.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            endpoints,
            vec![
                exact("0"),
                exact("0.5"),
                exact("1"),
                exact("1.5"),
                exact("2")
            ]
        );
        assert_eq!(certificate.cells.len(), 5);
        assert_eq!(
            certificate
                .cells
                .iter()
                .map(|cell| cell.relu_scalars)
                .collect::<Vec<_>>(),
            vec![3; 5]
        );

        let replay = graph.replay_exact_one_axis_phase_certificate_until(
            &problem,
            &certificate,
            OneAxisPhaseLimits::default(),
            deadline(),
        );
        assert!(replay.accepted, "{replay:?}");
        assert_eq!(replay.observation, Some(certificate.observation));
    }

    #[test]
    fn replay_rejects_endpoint_core_and_observation_tampering() {
        let graph = nested_relu_graph();
        let problem = scalar_problem("-1", "2");
        let certificate = graph
            .exact_one_axis_phase_certificate_until(
                &problem,
                OneAxisPhaseLimits::default(),
                deadline(),
            )
            .certificate
            .expect("phase certificate");

        let mut endpoint = certificate.clone();
        endpoint.cells[0].upper = exact("0.25");
        assert!(
            !graph
                .replay_exact_one_axis_phase_certificate_until(
                    &problem,
                    &endpoint,
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .accepted
        );

        let mut core = certificate.clone();
        core.cells[2].core.bias = exact("123");
        assert!(
            !graph
                .replay_exact_one_axis_phase_certificate_until(
                    &problem,
                    &core,
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .accepted
        );

        let mut observation = certificate.clone();
        observation.observation = OneAxisPhaseObservation::CertifiedEmpty;
        assert!(
            !graph
                .replay_exact_one_axis_phase_certificate_until(
                    &problem,
                    &observation,
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .accepted
        );

        let mut graph_identity = certificate.clone();
        graph_identity.graph_digest[0] ^= 1;
        assert!(
            !graph
                .replay_exact_one_axis_phase_certificate_until(
                    &problem,
                    &graph_identity,
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .accepted
        );

        let mut phase_digest = certificate.clone();
        phase_digest.cells[0].relu_phase_digest[0] ^= 1;
        assert!(
            !graph
                .replay_exact_one_axis_phase_certificate_until(
                    &problem,
                    &phase_digest,
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .accepted
        );

        let mut wrapper = certificate.clone();
        wrapper.wrapper.offset_upper = f64::from_bits(1);
        assert!(
            !graph
                .replay_exact_one_axis_phase_certificate_until(
                    &problem,
                    &wrapper,
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .accepted
        );

        let mut peeled = certificate.clone();
        peeled.peeled_constraints[0].necessary = OneAxisCoreGuard::Always;
        assert!(
            !graph
                .replay_exact_one_axis_phase_certificate_until(
                    &problem,
                    &peeled,
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .accepted
        );

        let mut authority = certificate.clone();
        authority.verdict_authority = true;
        assert!(
            !graph
                .replay_exact_one_axis_phase_certificate_until(
                    &problem,
                    &authority,
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .accepted
        );

        let mut truncated = certificate;
        truncated.cells.pop();
        assert!(
            !graph
                .replay_exact_one_axis_phase_certificate_until(
                    &problem,
                    &truncated,
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .accepted
        );
    }

    #[test]
    fn simultaneous_roots_are_one_boundary_and_identically_zero_is_stable() {
        let mut simultaneous = GraphNetwork::new();
        shape(&mut simultaneous, NETWORK_INPUT, &[1]);
        for name in ["left", "right"] {
            simultaneous.add_node(GraphNode::from_input(name, Layer::ReLU(ReLULayer)));
            shape(&mut simultaneous, name, &[1]);
        }
        simultaneous.add_node(GraphNode::binary(
            "concat",
            Layer::Concat(ConcatLayer::new(0)),
            "left",
            "right",
        ));
        shape(&mut simultaneous, "concat", &[2]);
        simultaneous.add_node(GraphNode::new(
            "sum",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32, 1.0]]), None).expect("valid linear")),
            vec!["concat".to_string()],
        ));
        shape(&mut simultaneous, "sum", &[1]);
        simultaneous.add_node(GraphNode::new(
            "sigmoid",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["sum".to_string()],
        ));
        shape(&mut simultaneous, "sigmoid", &[1]);
        simultaneous.set_output("sigmoid");
        let certificate = simultaneous
            .exact_one_axis_phase_certificate_until(
                &scalar_problem("-1", "1"),
                OneAxisPhaseLimits::default(),
                deadline(),
            )
            .certificate
            .expect("simultaneous-root certificate");
        assert_eq!(certificate.cells.len(), 2);
        assert_eq!(certificate.cells[0].upper, exact("0"));
        assert_eq!(certificate.cells[0].relu_scalars, 2);

        let mut zero = GraphNetwork::new();
        shape(&mut zero, NETWORK_INPUT, &[1]);
        zero.add_node(GraphNode::from_input(
            "zero_linear",
            Layer::Linear(
                LinearLayer::new(arr2(&[[0.0_f32]]), Some(arr1(&[0.0]))).expect("valid linear"),
            ),
        ));
        shape(&mut zero, "zero_linear", &[1]);
        zero.add_node(GraphNode::new(
            "zero_relu",
            Layer::ReLU(ReLULayer),
            vec!["zero_linear".to_string()],
        ));
        shape(&mut zero, "zero_relu", &[1]);
        zero.add_node(GraphNode::new(
            "sigmoid",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["zero_relu".to_string()],
        ));
        shape(&mut zero, "sigmoid", &[1]);
        zero.set_output("sigmoid");
        let zero_problem = scalar_problem("-1", "1");
        let zero_certificate = zero
            .exact_one_axis_phase_certificate_until(
                &zero_problem,
                OneAxisPhaseLimits::default(),
                deadline(),
            )
            .certificate
            .expect("identically-zero certificate");
        assert_eq!(zero_certificate.cells.len(), 1);
        assert_eq!(zero_certificate.cells[0].core.slope, exact("0"));
        assert!(
            zero.replay_exact_one_axis_phase_certificate_until(
                &zero_problem,
                &zero_certificate,
                OneAxisPhaseLimits::default(),
                deadline(),
            )
            .accepted
        );
    }

    #[test]
    fn dual_static_sigmoid_sibling_peels_to_exact_empty_observation() {
        let mut graph = GraphNetwork::new();
        shape(&mut graph, NETWORK_INPUT, &[2]);
        graph.add_node(GraphNode::from_input(
            "dynamic",
            Layer::Slice(SliceLayer::new(0, 0, 1)),
        ));
        shape(&mut graph, "dynamic", &[1]);
        graph.add_node(GraphNode::from_input(
            "static",
            Layer::Slice(SliceLayer::new(0, 1, 2)),
        ));
        shape(&mut graph, "static", &[1]);
        graph.add_node(GraphNode::new(
            "dynamic_sigmoid",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["dynamic".to_string()],
        ));
        shape(&mut graph, "dynamic_sigmoid", &[1]);
        graph.add_node(GraphNode::new(
            "static_sigmoid",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["static".to_string()],
        ));
        shape(&mut graph, "static_sigmoid", &[1]);
        graph.add_node(GraphNode::binary(
            "output",
            Layer::Sub(SubLayer),
            "static_sigmoid",
            "dynamic_sigmoid",
        ));
        shape(&mut graph, "output", &[1]);
        graph.set_output("output");

        let problem = OneAxisExactProblem {
            input_shape: vec![2],
            fixed_inputs: vec![exact("0"), exact("0")],
            free_axis: 0,
            lower: exact("-2"),
            upper: exact("2"),
            constraints: vec![OneAxisOutputConstraint {
                relation: OneAxisConstraintRelation::GreaterEqual,
                bound: exact("0.6"),
            }],
        };
        let certificate = graph
            .exact_one_axis_phase_certificate_until(
                &problem,
                OneAxisPhaseLimits::default(),
                deadline(),
            )
            .certificate
            .expect("dual certificate");
        assert_eq!(
            certificate.observation,
            OneAxisPhaseObservation::CertifiedEmpty
        );
        assert_eq!(certificate.wrapper.sigmoid_sign, -1);
        assert!(
            graph
                .replay_exact_one_axis_phase_certificate_until(
                    &problem,
                    &certificate,
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .accepted
        );
    }

    #[test]
    fn add_constant_rank_one_channel_bias_matches_layer_semantics() {
        let mut graph = GraphNetwork::new();
        shape(&mut graph, NETWORK_INPUT, &[2, 1, 2]);
        graph.add_node(GraphNode::from_input(
            "bias",
            Layer::AddConstant(AddConstantLayer::new(arr1(&[10.0_f32, 20.0]).into_dyn())),
        ));
        shape(&mut graph, "bias", &[2, 1, 2]);
        graph.add_node(GraphNode::new(
            "channel_zero",
            Layer::Slice(SliceLayer::new(0, 0, 1)),
            vec!["bias".to_string()],
        ));
        shape(&mut graph, "channel_zero", &[1, 1, 2]);
        graph.add_node(GraphNode::new(
            "width_one",
            Layer::Slice(SliceLayer::new(2, 1, 2)),
            vec!["channel_zero".to_string()],
        ));
        shape(&mut graph, "width_one", &[1, 1, 1]);
        graph.add_node(GraphNode::new(
            "sigmoid",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["width_one".to_string()],
        ));
        shape(&mut graph, "sigmoid", &[1, 1, 1]);
        graph.set_output("sigmoid");

        // Flat axis 1 is channel 0, width 1.  AddConstant's rank-1 special
        // case adds channel bias 10 here; ordinary trailing-axis broadcast
        // would incorrectly add width bias 20.
        let problem = OneAxisExactProblem {
            input_shape: vec![2, 1, 2],
            fixed_inputs: vec![exact("0"); 4],
            free_axis: 1,
            lower: exact("-1"),
            upper: exact("1"),
            constraints: vec![OneAxisOutputConstraint {
                relation: OneAxisConstraintRelation::GreaterEqual,
                bound: exact("0.5"),
            }],
        };
        let certificate = graph
            .exact_one_axis_phase_certificate_until(
                &problem,
                OneAxisPhaseLimits::default(),
                deadline(),
            )
            .certificate
            .expect("channel-bias certificate");
        assert_eq!(certificate.cells.len(), 1);
        assert_eq!(certificate.cells[0].core.slope, exact("1"));
        assert_eq!(certificate.cells[0].core.bias, exact("10"));
        assert!(
            graph
                .replay_exact_one_axis_phase_certificate_until(
                    &problem,
                    &certificate,
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .accepted
        );
    }

    #[test]
    fn deadline_and_phase_cap_decline_closed() {
        let graph = nested_relu_graph();
        let problem = scalar_problem("-1", "2");
        let expired = graph.exact_one_axis_phase_certificate_until(
            &problem,
            OneAxisPhaseLimits::default(),
            Instant::now(),
        );
        assert_eq!(
            expired.decline.map(|item| item.reason),
            Some(OneAxisPhaseDeclineReason::Deadline)
        );
        let capped = graph.exact_one_axis_phase_certificate_until(
            &problem,
            OneAxisPhaseLimits {
                max_phase_cells: 1,
                ..OneAxisPhaseLimits::default()
            },
            deadline(),
        );
        assert_eq!(
            capped.decline.map(|item| item.reason),
            Some(OneAxisPhaseDeclineReason::PhaseCellLimit)
        );
        let rational_capped = graph.exact_one_axis_phase_certificate_until(
            &problem,
            OneAxisPhaseLimits {
                max_rational_bits: 64,
                ..OneAxisPhaseLimits::default()
            },
            deadline(),
        );
        assert_eq!(
            rational_capped.decline.map(|item| item.reason),
            Some(OneAxisPhaseDeclineReason::RationalBitLimit)
        );

        let mut coefficient_capped_graph = GraphNetwork::new();
        shape(&mut coefficient_capped_graph, NETWORK_INPUT, &[1]);
        coefficient_capped_graph.add_node(GraphNode::from_input(
            "linear",
            Layer::Linear(
                LinearLayer::new(arr2(&[[0.0_f32]]), Some(arr1(&[f32::from_bits(1)])))
                    .expect("valid linear"),
            ),
        ));
        shape(&mut coefficient_capped_graph, "linear", &[1]);
        coefficient_capped_graph.add_node(GraphNode::new(
            "sigmoid",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["linear".to_string()],
        ));
        shape(&mut coefficient_capped_graph, "sigmoid", &[1]);
        coefficient_capped_graph.set_output("sigmoid");
        let mut permissive_problem = scalar_problem("-1", "1");
        permissive_problem.constraints[0].bound = exact("-1");
        let coefficient_capped = coefficient_capped_graph.exact_one_axis_phase_certificate_until(
            &permissive_problem,
            OneAxisPhaseLimits {
                max_rational_bits: 64,
                ..OneAxisPhaseLimits::default()
            },
            deadline(),
        );
        assert_eq!(
            coefficient_capped.decline.map(|item| item.reason),
            Some(OneAxisPhaseDeclineReason::RationalBitLimit)
        );
    }

    fn inactive_relu_weight_graph(weight: f32) -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        shape(&mut graph, NETWORK_INPUT, &[1]);
        graph.add_node(GraphNode::from_input(
            "shift",
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[-2.0_f32]))).expect("valid shift"),
            ),
        ));
        shape(&mut graph, "shift", &[1]);
        graph.add_node(GraphNode::new(
            "inactive",
            Layer::ReLU(ReLULayer),
            vec!["shift".to_string()],
        ));
        shape(&mut graph, "inactive", &[1]);
        graph.add_node(GraphNode::new(
            "coefficient",
            Layer::Linear(
                LinearLayer::new(arr2(&[[weight]]), Some(arr1(&[0.0_f32])))
                    .expect("linear constructor accepts the audit coefficient"),
            ),
            vec!["inactive".to_string()],
        ));
        shape(&mut graph, "coefficient", &[1]);
        graph.add_node(GraphNode::new(
            "sigmoid",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["coefficient".to_string()],
        ));
        shape(&mut graph, "sigmoid", &[1]);
        graph.set_output("sigmoid");
        graph
    }

    #[test]
    fn audit_inactive_dynamic_zero_does_not_bypass_coefficient_bit_cap() {
        let graph = inactive_relu_weight_graph(f32::from_bits(1));
        let mut problem = scalar_problem("-1", "1");
        problem.constraints[0].bound = exact("-1");
        let attempt = graph.exact_one_axis_phase_certificate_until(
            &problem,
            OneAxisPhaseLimits {
                max_rational_bits: 64,
                ..OneAxisPhaseLimits::default()
            },
            deadline(),
        );
        assert!(
            attempt.certificate.is_none()
                && attempt.decline.as_ref().map(|item| item.reason)
                    == Some(OneAxisPhaseDeclineReason::RationalBitLimit),
            "an inactive structurally dynamic zero hid an over-cap coefficient: {attempt:?}"
        );
    }

    #[test]
    fn audit_inactive_dynamic_zero_does_not_hide_nonfinite_weight_from_replay() {
        let graph = inactive_relu_weight_graph(f32::NAN);
        let problem = scalar_problem("-1", "1");
        let attempt = graph.exact_one_axis_phase_certificate_until(
            &problem,
            OneAxisPhaseLimits::default(),
            deadline(),
        );
        let replay_accepted = attempt.certificate.as_ref().is_some_and(|certificate| {
            graph
                .replay_exact_one_axis_phase_certificate_until(
                    &problem,
                    certificate,
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .accepted
        });
        assert!(
            attempt.certificate.is_none() && !replay_accepted,
            "generation and replay admitted a graph whose hidden coefficient is non-finite: \
             attempt={attempt:?}, replay_accepted={replay_accepted}"
        );
    }

    fn two_relu_graph(w1: i32, b1: i32, w2: i32, b2: i32) -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        shape(&mut graph, NETWORK_INPUT, &[1]);
        graph.add_node(GraphNode::from_input(
            "linear_1",
            Layer::Linear(
                LinearLayer::new(arr2(&[[w1 as f32]]), Some(arr1(&[b1 as f32])))
                    .expect("valid first linear"),
            ),
        ));
        shape(&mut graph, "linear_1", &[1]);
        graph.add_node(GraphNode::new(
            "relu_1",
            Layer::ReLU(ReLULayer),
            vec!["linear_1".to_string()],
        ));
        shape(&mut graph, "relu_1", &[1]);
        graph.add_node(GraphNode::new(
            "linear_2",
            Layer::Linear(
                LinearLayer::new(arr2(&[[w2 as f32]]), Some(arr1(&[b2 as f32])))
                    .expect("valid second linear"),
            ),
            vec!["relu_1".to_string()],
        ));
        shape(&mut graph, "linear_2", &[1]);
        graph.add_node(GraphNode::new(
            "relu_2",
            Layer::ReLU(ReLULayer),
            vec!["linear_2".to_string()],
        ));
        shape(&mut graph, "relu_2", &[1]);
        graph.add_node(GraphNode::new(
            "sigmoid",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec!["relu_2".to_string()],
        ));
        shape(&mut graph, "sigmoid", &[1]);
        graph.set_output("sigmoid");
        graph
    }

    fn direct_two_relu(point: &BigRational, w1: i32, b1: i32, w2: i32, b2: i32) -> BigRational {
        let integer = |value: i32| BigRational::from_integer(value.into());
        let relu = |value: BigRational| {
            if value.is_negative() {
                BigRational::zero()
            } else {
                value
            }
        };
        let first = relu(integer(w1) * point + integer(b1));
        relu(integer(w2) * first + integer(b2))
    }

    #[test]
    fn sparse_linear_phase_delta_is_exactly_the_full_recompute() {
        let weights = ndarray::Array2::from_shape_vec(
            (3, 8),
            vec![
                0.25, -0.5, 0.75, 1.25, -1.5, 1.75, 2.25, -2.5, -0.75, 1.5, -2.25, 3.0, 0.5, -1.25,
                2.0, 2.75, 1.125, -1.375, 1.625, -1.875, 2.125, -2.375, 2.625, -2.875,
            ],
        )
        .expect("linear weights");
        let layer =
            LinearLayer::new(weights, Some(arr1(&[0.5, -0.25, 0.75]))).expect("linear layer");
        let affine = |slope: i64, bias: i64| ExactAffine {
            slope: BigRational::from_integer(slope.into()),
            bias: BigRational::from_integer(bias.into()),
            depends: true,
        };
        let previous = ExactTensor {
            shape: vec![1, 8],
            values: (0_i64..8)
                .map(|index| affine(index + 1, index + 2))
                .collect(),
        };
        let mut current = previous.clone();
        current.values[3] = ExactAffine {
            slope: BigRational::zero(),
            bias: BigRational::zero(),
            depends: true,
        };
        let limits = OneAxisPhaseLimits::default();
        let mut previous_budget = ExactBudget::new(&limits, deadline());
        let previous_output = linear_tensor(&previous, &layer, &[1, 3], None, &mut previous_budget)
            .expect("previous full result")
            .0;
        let cached = LinearPhaseCache {
            input: previous,
            output: previous_output,
            context_epoch: 0,
        };
        let mut delta_budget = ExactBudget::new(&limits, deadline());
        let delta = match linear_tensor_from_phase_delta(
            &current,
            &layer,
            &[1, 3],
            &cached,
            &mut delta_budget,
        )
        .expect("delta arithmetic")
        {
            LinearPhaseDelta::Updated(tensor) => tensor,
            LinearPhaseDelta::Unchanged => panic!("the test input changed"),
            LinearPhaseDelta::Recompute => panic!("sparse delta must be admitted"),
        };
        let mut full_budget = ExactBudget::new(&limits, deadline());
        let full = linear_tensor(&current, &layer, &[1, 3], None, &mut full_budget)
            .expect("current full result")
            .0;
        assert_eq!(delta, full);
        assert!(
            delta_budget.operations < full_budget.operations,
            "sparse update must perform less exact rational work: delta={} full={}",
            delta_budget.operations,
            full_budget.operations
        );
    }

    #[test]
    fn phase_cache_retention_is_bounded_without_refusing_proof_work() {
        let limits = OneAxisPhaseLimits {
            max_total_tensor_elements: 3,
            ..OneAxisPhaseLimits::default()
        };
        let mut budget = ExactBudget::new(&limits, deadline());
        assert!(budget.try_replace_phase_cache(0, 3));
        assert!(!budget.try_replace_phase_cache(3, 4));
        assert_eq!(budget.failure, None);
        assert_eq!(budget.phase_cache_elements, 3);
    }

    #[test]
    fn phase_cache_cap_does_not_shrink_the_admitted_graph_surface() {
        let mut graph = GraphNetwork::new();
        shape(&mut graph, NETWORK_INPUT, &[1]);
        let mut previous = NETWORK_INPUT.to_string();
        for index in 0..3 {
            let name = format!("linear_{index}");
            graph.add_node(GraphNode::new(
                &name,
                Layer::Linear(
                    LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32])))
                        .expect("identity linear"),
                ),
                vec![previous],
            ));
            shape(&mut graph, &name, &[1]);
            previous = name;
        }
        graph.add_node(GraphNode::new(
            "sigmoid",
            Layer::Sigmoid(SigmoidLayer::new()),
            vec![previous],
        ));
        shape(&mut graph, "sigmoid", &[1]);
        graph.set_output("sigmoid");

        let problem = scalar_problem("-1", "1");
        let limits = OneAxisPhaseLimits {
            // Input plus three linear nodes plus the sigmoid exactly fills
            // the pre-existing graph-retention admission cap.  The phase
            // cache must remain an optional optimization above that surface.
            max_total_tensor_elements: 5,
            ..OneAxisPhaseLimits::default()
        };
        let attempt = graph.exact_one_axis_phase_certificate_until(&problem, limits, deadline());
        let certificate = attempt
            .certificate
            .expect("optional phase caching cannot reject the admitted graph");
        assert_eq!(attempt.decline, None);
        let replay = graph.replay_exact_one_axis_phase_certificate_until(
            &problem,
            &certificate,
            limits,
            deadline(),
        );
        assert!(replay.accepted, "{replay:?}");
    }

    #[test]
    fn sparse_phase_scan_polls_the_absolute_deadline() {
        let width = 1024;
        let layer =
            LinearLayer::new(ndarray::Array2::zeros((1, width)), None).expect("linear layer");
        let input = ExactTensor {
            shape: vec![1, width],
            values: vec![ExactAffine::constant(BigRational::zero()); width],
        };
        let cached = LinearPhaseCache {
            input: input.clone(),
            output: ExactTensor {
                shape: vec![1, 1],
                values: vec![ExactAffine::constant(BigRational::zero())],
            },
            context_epoch: 0,
        };
        let limits = OneAxisPhaseLimits::default();
        let mut budget = ExactBudget::new(&limits, Instant::now());
        assert!(
            linear_tensor_from_phase_delta(&input, &layer, &[1, 1], &cached, &mut budget).is_none()
        );
        assert_eq!(budget.failure, Some(OneAxisPhaseDeclineReason::Deadline));
    }

    #[test]
    fn sparse_zero_weight_update_polls_the_absolute_deadline() {
        let layer = LinearLayer::new(arr2(&[[0.0_f32, 0.0, 0.0, 0.0]]), None)
            .expect("zero-weight linear layer");
        let previous = ExactTensor {
            shape: vec![1, 4],
            values: vec![
                ExactAffine {
                    slope: BigRational::zero(),
                    bias: BigRational::zero(),
                    depends: true,
                };
                4
            ],
        };
        let mut current = previous.clone();
        current.values[0].slope = BigRational::one();
        let cached = LinearPhaseCache {
            input: previous,
            output: ExactTensor {
                shape: vec![1, 1],
                values: vec![ExactAffine::constant(BigRational::zero())],
            },
            context_epoch: 0,
        };
        let limits = OneAxisPhaseLimits::default();
        let mut budget = ExactBudget::new(&limits, Instant::now());
        // Arrange for the scan, cached-output clone, and two exact deltas to
        // end immediately before the periodic poll in the zero-weight update
        // loop. Without that loop-local poll this expired request completes.
        budget.work_items = 1016;
        assert!(
            linear_tensor_from_phase_delta(&current, &layer, &[1, 1], &cached, &mut budget)
                .is_none()
        );
        assert_eq!(budget.failure, Some(OneAxisPhaseDeclineReason::Deadline));
    }

    proptest! {
        #[test]
        fn affine_relu_root_is_never_missed(root in -4_i32..=4) {
            let mut graph = GraphNetwork::new();
            shape(&mut graph, NETWORK_INPUT, &[1]);
            graph.add_node(GraphNode::from_input(
                "shift",
                Layer::AddConstant(AddConstantLayer::new(
                    arr0(-(root as f32)).into_dyn(),
                )),
            ));
            shape(&mut graph, "shift", &[1]);
            graph.add_node(GraphNode::new(
                "relu",
                Layer::ReLU(ReLULayer),
                vec!["shift".to_string()],
            ));
            shape(&mut graph, "relu", &[1]);
            graph.add_node(GraphNode::new(
                "sigmoid",
                Layer::Sigmoid(SigmoidLayer::new()),
                vec!["relu".to_string()],
            ));
            shape(&mut graph, "sigmoid", &[1]);
            graph.set_output("sigmoid");
            let problem = scalar_problem("-5", "5");
            let certificate = graph
                .exact_one_axis_phase_certificate_until(
                    &problem,
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .certificate
                .expect("certificate");
            prop_assert_eq!(certificate.cells.len(), 2);
            prop_assert_eq!(&certificate.cells[0].upper, &OneAxisRational::from_integer(i64::from(root)));
            let replay = graph.replay_exact_one_axis_phase_certificate_until(
                &problem,
                &certificate,
                OneAxisPhaseLimits::default(),
                deadline(),
            );
            prop_assert!(replay.accepted);
        }

        #[test]
        fn random_two_relu_cells_match_an_exact_piecewise_oracle(
            w1 in -3_i32..=3,
            b1 in -4_i32..=4,
            w2 in -3_i32..=3,
            b2 in -4_i32..=4,
        ) {
            let graph = two_relu_graph(w1, b1, w2, b2);
            let problem = scalar_problem("-5", "5");
            let certificate = graph
                .exact_one_axis_phase_certificate_until(
                    &problem,
                    OneAxisPhaseLimits::default(),
                    deadline(),
                )
                .certificate
                .expect("two-ReLU certificate");
            let mut cursor = problem.lower.0.clone();
            for cell in &certificate.cells {
                prop_assert_eq!(&cell.lower.0, &cursor);
                prop_assert!(cell.lower.0 <= cell.upper.0);
                let midpoint = (&cell.lower.0 + &cell.upper.0)
                    / BigRational::from_integer(2.into());
                for point in [&cell.lower.0, &midpoint, &cell.upper.0] {
                    let certified = &cell.core.slope.0 * point + &cell.core.bias.0;
                    let oracle = direct_two_relu(point, w1, b1, w2, b2);
                    prop_assert_eq!(certified, oracle);
                }
                cursor = cell.upper.0.clone();
            }
            prop_assert_eq!(&cursor, &problem.upper.0);
            let replay = graph.replay_exact_one_axis_phase_certificate_until(
                &problem,
                &certificate,
                OneAxisPhaseLimits::default(),
                deadline(),
            );
            prop_assert!(replay.accepted);
        }
    }
}
