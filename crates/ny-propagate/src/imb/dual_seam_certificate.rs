// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verdict-neutral cGAN row-5 dual-seam certificate envelope.
//!
//! This is the source-only M0 boundary for a proof-carrying replacement of the
//! measured loose whole-network replay. It serializes one exact 16-region input
//! cover, the two outward binary32 lower-affine tail rows for `-Y0` and `+Y0`,
//! and a fresh prefix-bound commitment for every `(clause, region)` pair.
//!
//! The checker deliberately has **no verdict channel**. A successful check says
//! only that the complete two-clause envelope is finite, internally committed,
//! bound to the expected run context, and structurally exact. It does not prove
//! the affine inequalities. M1 must add independently replayable tail and prefix
//! proof payloads before a separate authority boundary may consume this format.
//! Consequently every emitted artifact and every successful report carries
//! `authority = false`.

use std::collections::HashSet;
use std::fmt;
use std::time::Instant;

use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::root_inject::validate_binary_partition_cover;

/// Schema number for the first verdict-neutral dual-seam envelope.
pub const DUAL_SEAM_CERTIFICATE_SCHEMA: u32 = 1;

/// Exact number of cGAN row-5 regions admitted by M0.
pub const DUAL_SEAM_REGION_COUNT: usize = 16;

/// Exact number of independent single-row clauses admitted by M0.
pub const DUAL_SEAM_CLAUSE_COUNT: usize = 2;

/// Hard cap checked before parsing any untrusted M0 JSON.
pub const DUAL_SEAM_CERTIFICATE_MAX_BYTES: usize = 16 * 1024 * 1024;

// These schema caps comfortably contain the measured row-5 input (5 values),
// Relu_17 seam (2,048 values), and the existing authority path's immutable
// 64-leaf per-region frontier while bounding pre-proof diagnostic artifacts.
const DUAL_SEAM_MAX_BOX_RANK: usize = 16;
const DUAL_SEAM_MAX_INPUT_ELEMENTS: usize = 64;
const DUAL_SEAM_MAX_SEAM_ELEMENTS: usize = 8_192;
const DUAL_SEAM_MAX_PREFIX_LEAVES_PER_REGION: usize = 64;
const NEGATIVE_Y0_BITS: u32 = (-1.0_f32).to_bits();
const POSITIVE_Y0_BITS: u32 = 1.0_f32.to_bits();
const PREFIX_ID_DOMAIN: &[u8] = b"ny.imb.cgan-dual-seam.prefix-id.v1\0";
const BODY_ID_DOMAIN: &[u8] = b"ny.imb.cgan-dual-seam.body-id.v1\0";
const INPUT_ID_DOMAIN: &[u8] = b"ny.imb.cgan-dual-seam.input-id.v1\0";
const OBJECTIVE_ORDER_ID_DOMAIN: &[u8] = b"ny.imb.cgan-dual-seam.objective-order-id.v1\0";
const POLICY_ID_DOMAIN: &[u8] = b"ny.imb.cgan-dual-seam.policy-id.v1\0";
const REGION_ORDER: &str = "mixed_radix_low_split_dimension_first";
const TAIL_ROUNDING: &str = "binary32_directed_down";
const PREFIX_ROUNDING: &str = "binary32_directed_down";
const PREFIX_PROOF_KIND: &str = "imb_prefix_bab_lower_v1";

/// Canonical sign/order of one cGAN row-5 objective.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DualSeamOrientation {
    /// First clause, lower functional for `-Y0`.
    NegativeY0,
    /// Second clause, lower functional for `+Y0`.
    PositiveY0,
}

/// Immutable policy surface committed by the M0 checker.
///
/// The only admitted constructor is [`Self::cgan_row5_k2`]. Keeping the policy
/// payload explicit lets a later standalone checker reject an artifact produced
/// under a different partition order or rounding convention.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DualSeamPolicy {
    region_k: usize,
    split_dimensions: Vec<usize>,
    region_order: String,
    tail_rounding: String,
    prefix_rounding: String,
    prefix_proof_kind: String,
}

impl DualSeamPolicy {
    /// Construct the exact 2^4 cGAN row-5 policy.
    #[must_use]
    pub fn cgan_row5_k2() -> Self {
        Self {
            region_k: 2,
            split_dimensions: vec![0, 1, 2, 4],
            region_order: REGION_ORDER.to_owned(),
            tail_rounding: TAIL_ROUNDING.to_owned(),
            prefix_rounding: PREFIX_ROUNDING.to_owned(),
            prefix_proof_kind: PREFIX_PROOF_KIND.to_owned(),
        }
    }

    fn is_cgan_row5_k2(&self) -> bool {
        self == &Self::cgan_row5_k2()
    }
}

/// One region's source evidence supplied to the M0 emitter.
///
/// The binary32 tail row is a lower affine functional
/// `coefficients · seam + bias`. `prefix_lower` is the independently computed
/// lower bound on the matching prefix functional. The prefix frontier must be
/// an exact binary partition cover of `region`.
#[derive(Clone, Debug)]
pub struct DualSeamRegionSource {
    /// Canonical region box.
    pub region: BoundedTensor,
    /// Outward binary32 tail coefficients over the flattened seam tensor.
    pub tail_coefficients: Vec<f32>,
    /// Outward binary32 tail bias.
    pub tail_bias: f32,
    /// Outward binary32 regional prefix lower.
    pub prefix_lower: f32,
    /// Exact prefix-BaB terminal frontier for this region.
    pub prefix_frontier: Vec<BoundedTensor>,
}

/// One clause's source evidence supplied to the M0 emitter.
#[derive(Clone, Debug)]
pub struct DualSeamClauseSource {
    /// Original property-row index.
    pub objective_index: usize,
    /// Canonical `-Y0`, then `+Y0`, order.
    pub orientation: DualSeamOrientation,
    /// Original output objective row.
    pub objective: Vec<f32>,
    /// Original strict verification threshold.
    pub threshold: f32,
    /// Exactly 16 region rows in shared-cover order.
    pub regions: Vec<DualSeamRegionSource>,
}

/// Complete input to [`emit_dual_seam_certificate_json`].
pub struct DualSeamEmissionRequest<'a> {
    /// SHA-256 of the exact model bytes, supplied by the model-owning caller.
    pub graph_sha256: [u8; 32],
    /// SHA-256 of the exact property bytes, supplied by the property-owning caller.
    pub property_sha256: [u8; 32],
    /// Original root input domain.
    pub root_input: &'a BoundedTensor,
    /// One shared exact 16-region cover.
    pub shared_cover: &'a [BoundedTensor],
    /// Exact graph seam node.
    pub seam: &'a str,
    /// Flattened seam width independently obtained from the graph owner.
    pub seam_elements: usize,
    /// Exact M0 policy.
    pub policy: &'a DualSeamPolicy,
    /// Canonically ordered `-Y0`, `+Y0` clause sources.
    pub clauses: &'a [DualSeamClauseSource],
    /// Wall-clock issuance time recorded by the caller.
    pub issued_at_unix_ms: u64,
    /// Wall-clock validity boundary corresponding to the verifier deadline.
    pub valid_until_unix_ms: u64,
    /// Live monotonic work deadline for emission and exact-cover validation.
    pub work_deadline: Instant,
}

/// Context that an independent checker must supply rather than trusting the
/// serialized artifact to identify itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DualSeamExpectedBindings {
    graph_sha256: [u8; 32],
    input_sha256: [u8; 32],
    property_sha256: [u8; 32],
    seam: String,
    seam_elements: usize,
    objective_order_sha256: [u8; 32],
    policy_sha256: [u8; 32],
    issued_at_unix_ms: u64,
    valid_until_unix_ms: u64,
}

impl DualSeamExpectedBindings {
    /// Build expected bindings from independently obtained identities.
    #[must_use]
    pub fn new(
        graph_sha256: [u8; 32],
        input_sha256: [u8; 32],
        property_sha256: [u8; 32],
        seam: impl Into<String>,
        seam_elements: usize,
        objective_order_sha256: [u8; 32],
        policy_sha256: [u8; 32],
        issued_at_unix_ms: u64,
        valid_until_unix_ms: u64,
    ) -> Self {
        Self {
            graph_sha256,
            input_sha256,
            property_sha256,
            seam: seam.into(),
            seam_elements,
            objective_order_sha256,
            policy_sha256,
            issued_at_unix_ms,
            valid_until_unix_ms,
        }
    }
}

/// All-or-nothing structural checker result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DualSeamCheckReport {
    /// Number of clauses checked atomically.
    clauses_checked: usize,
    /// Number of shared-cover regions checked.
    regions_checked: usize,
    /// Number of independently recomputed prefix identities.
    prefix_identities_checked: usize,
    /// Always false in M0.
    authority: bool,
}

impl DualSeamCheckReport {
    /// Number of clauses checked atomically.
    #[must_use]
    pub fn clauses_checked(self) -> usize {
        self.clauses_checked
    }

    /// Number of shared-cover regions checked.
    #[must_use]
    pub fn regions_checked(self) -> usize {
        self.regions_checked
    }

    /// Number of independently recomputed prefix identities.
    #[must_use]
    pub fn prefix_identities_checked(self) -> usize {
        self.prefix_identities_checked
    }

    /// Whether the report carries verdict authority. Always false in M0.
    #[must_use]
    pub fn authority(self) -> bool {
        self.authority
    }
}

/// Fail-closed checker/emitter error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DualSeamCertificateError {
    reason: String,
}

impl DualSeamCertificateError {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    /// Stable diagnostic reason.
    #[must_use]
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl fmt::Display for DualSeamCertificateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.reason)
    }
}

impl std::error::Error for DualSeamCertificateError {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BoxBits {
    shape: Vec<usize>,
    lower_bits: Vec<u32>,
    upper_bits: Vec<u32>,
}

impl BoxBits {
    fn from_tensor(tensor: &BoundedTensor) -> Result<Self, DualSeamCertificateError> {
        if tensor.has_l2_constraint() {
            return Err(DualSeamCertificateError::new(
                "L2-annotated boxes are outside the M0 schema",
            ));
        }
        let shape = tensor.shape().to_vec();
        let elements = checked_shape_elements(&shape)?;
        if elements == 0
            || elements > DUAL_SEAM_MAX_INPUT_ELEMENTS
            || tensor.lower().len() != elements
            || tensor.upper().len() != elements
        {
            return Err(DualSeamCertificateError::new(
                "box shape/endpoint cardinality mismatch",
            ));
        }
        let lower_bits: Vec<u32> = tensor.lower().iter().map(|value| value.to_bits()).collect();
        let upper_bits: Vec<u32> = tensor.upper().iter().map(|value| value.to_bits()).collect();
        validate_endpoint_bits(&lower_bits, &upper_bits)?;
        Ok(Self {
            shape,
            lower_bits,
            upper_bits,
        })
    }

    fn to_tensor(&self) -> Result<BoundedTensor, DualSeamCertificateError> {
        let elements = checked_shape_elements(&self.shape)?;
        if elements == 0
            || elements > DUAL_SEAM_MAX_INPUT_ELEMENTS
            || self.lower_bits.len() != elements
            || self.upper_bits.len() != elements
        {
            return Err(DualSeamCertificateError::new(
                "serialized box shape/endpoint cardinality mismatch",
            ));
        }
        validate_endpoint_bits(&self.lower_bits, &self.upper_bits)?;
        let lower = self
            .lower_bits
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect();
        let upper = self
            .upper_bits
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect();
        let lower = ArrayD::from_shape_vec(IxDyn(&self.shape), lower)
            .map_err(|_| DualSeamCertificateError::new("serialized lower shape rejected"))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&self.shape), upper)
            .map_err(|_| DualSeamCertificateError::new("serialized upper shape rejected"))?;
        BoundedTensor::new(lower, upper)
            .map_err(|_| DualSeamCertificateError::new("serialized box rejected"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrefixBoundEvidence {
    lower_bits: u32,
    frontier: Vec<BoxBits>,
    identity_sha256: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TailRow {
    region_index: usize,
    coefficient_bits: Vec<u32>,
    bias_bits: u32,
    prefix: PrefixBoundEvidence,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Clause {
    objective_index: usize,
    orientation: DualSeamOrientation,
    objective_bits: Vec<u32>,
    threshold_bits: u32,
    rows: Vec<TailRow>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Bindings {
    graph_sha256: [u8; 32],
    input_sha256: [u8; 32],
    property_sha256: [u8; 32],
    seam: String,
    seam_elements: usize,
    objective_order_sha256: [u8; 32],
    policy_sha256: [u8; 32],
    issued_at_unix_ms: u64,
    valid_until_unix_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CertificateBody {
    bindings: Bindings,
    policy: DualSeamPolicy,
    root_input: BoxBits,
    shared_cover: Vec<BoxBits>,
    clauses: Vec<Clause>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CertificateEnvelope {
    schema: u32,
    authority: bool,
    body: CertificateBody,
    body_sha256: [u8; 32],
}

#[derive(Serialize)]
struct ObjectiveOrderMaterial<'a> {
    objective_index: usize,
    orientation: DualSeamOrientation,
    objective_bits: &'a [u32],
    threshold_bits: u32,
}

#[derive(Serialize)]
struct PrefixIdentityMaterial<'a> {
    bindings: &'a Bindings,
    clause_index: usize,
    objective_index: usize,
    orientation: DualSeamOrientation,
    objective_bits: &'a [u32],
    threshold_bits: u32,
    region_index: usize,
    region: &'a BoxBits,
    coefficient_bits: &'a [u32],
    bias_bits: u32,
    prefix_lower_bits: u32,
    prefix_frontier: &'a [BoxBits],
}

fn checked_shape_elements(shape: &[usize]) -> Result<usize, DualSeamCertificateError> {
    if shape.is_empty() || shape.len() > DUAL_SEAM_MAX_BOX_RANK {
        return Err(DualSeamCertificateError::new(
            "box rank is empty or exceeds the M0 cap",
        ));
    }
    shape.iter().try_fold(1_usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or_else(|| DualSeamCertificateError::new("box shape element count overflow"))
    })
}

fn validate_endpoint_bits(
    lower_bits: &[u32],
    upper_bits: &[u32],
) -> Result<(), DualSeamCertificateError> {
    if lower_bits.len() != upper_bits.len() {
        return Err(DualSeamCertificateError::new(
            "box endpoint cardinality mismatch",
        ));
    }
    for (&lower_bits, &upper_bits) in lower_bits.iter().zip(upper_bits) {
        let lower = f32::from_bits(lower_bits);
        let upper = f32::from_bits(upper_bits);
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(DualSeamCertificateError::new(
                "box contains non-finite or inverted endpoint",
            ));
        }
    }
    Ok(())
}

fn ensure_work_deadline(deadline: Instant, phase: &str) -> Result<(), DualSeamCertificateError> {
    if Instant::now() >= deadline {
        return Err(DualSeamCertificateError::new(format!(
            "{phase} work deadline expired"
        )));
    }
    Ok(())
}

fn validate_finite_bits(bits: &[u32], name: &str) -> Result<(), DualSeamCertificateError> {
    if bits.is_empty() || bits.iter().any(|bits| !f32::from_bits(*bits).is_finite()) {
        return Err(DualSeamCertificateError::new(format!(
            "{name} is empty or non-finite"
        )));
    }
    Ok(())
}

fn digest_serialized<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<[u8; 32], DualSeamCertificateError> {
    let encoded = serde_json::to_vec(value).map_err(|error| {
        DualSeamCertificateError::new(format!("canonical serialization failed: {error}"))
    })?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((encoded.len() as u64).to_le_bytes());
    hasher.update(encoded);
    Ok(hasher.finalize().into())
}

fn input_identity(root_input: &BoxBits) -> Result<[u8; 32], DualSeamCertificateError> {
    digest_serialized(INPUT_ID_DOMAIN, root_input)
}

fn policy_identity(policy: &DualSeamPolicy) -> Result<[u8; 32], DualSeamCertificateError> {
    digest_serialized(POLICY_ID_DOMAIN, policy)
}

fn objective_order_identity(clauses: &[Clause]) -> Result<[u8; 32], DualSeamCertificateError> {
    let material: Vec<ObjectiveOrderMaterial<'_>> = clauses
        .iter()
        .map(|clause| ObjectiveOrderMaterial {
            objective_index: clause.objective_index,
            orientation: clause.orientation,
            objective_bits: &clause.objective_bits,
            threshold_bits: clause.threshold_bits,
        })
        .collect();
    digest_serialized(OBJECTIVE_ORDER_ID_DOMAIN, &material)
}

fn prefix_identity(
    bindings: &Bindings,
    clause_index: usize,
    clause: &Clause,
    region_index: usize,
    region: &BoxBits,
    row: &TailRow,
) -> Result<[u8; 32], DualSeamCertificateError> {
    digest_serialized(
        PREFIX_ID_DOMAIN,
        &PrefixIdentityMaterial {
            bindings,
            clause_index,
            objective_index: clause.objective_index,
            orientation: clause.orientation,
            objective_bits: &clause.objective_bits,
            threshold_bits: clause.threshold_bits,
            region_index,
            region,
            coefficient_bits: &row.coefficient_bits,
            bias_bits: row.bias_bits,
            prefix_lower_bits: row.prefix.lower_bits,
            prefix_frontier: &row.prefix.frontier,
        },
    )
}

fn body_identity(body: &CertificateBody) -> Result<[u8; 32], DualSeamCertificateError> {
    digest_serialized(BODY_ID_DOMAIN, body)
}

fn same_box_bits(left: &BoundedTensor, right: &BoundedTensor) -> bool {
    left.shape() == right.shape()
        && !left.has_l2_constraint()
        && !right.has_l2_constraint()
        && left.lower().len() == right.lower().len()
        && left.upper().len() == right.upper().len()
        && left
            .lower()
            .iter()
            .zip(right.lower())
            .all(|(&left, &right)| left.to_bits() == right.to_bits())
        && left
            .upper()
            .iter()
            .zip(right.upper())
            .all(|(&left, &right)| left.to_bits() == right.to_bits())
}

fn validate_cgan_row5_policy_cover(
    root: &BoundedTensor,
    cover: &[BoundedTensor],
    deadline: Instant,
) -> Result<(), DualSeamCertificateError> {
    ensure_work_deadline(deadline, "policy-cover validation")?;
    if cover.len() != DUAL_SEAM_REGION_COUNT {
        return Err(DualSeamCertificateError::new(
            "policy cover has the wrong region count",
        ));
    }
    let root = root.flatten();
    let root_lower: Vec<f32> = root.lower().iter().copied().collect();
    let root_upper: Vec<f32> = root.upper().iter().copied().collect();
    let split_dimensions = [0_usize, 1, 2, 4];
    if root_lower.len() != root_upper.len()
        || split_dimensions
            .iter()
            .any(|&dimension| dimension >= root_lower.len())
    {
        return Err(DualSeamCertificateError::new(
            "policy split dimension is outside the root input",
        ));
    }
    let mut edges = Vec::with_capacity(split_dimensions.len());
    for &dimension in &split_dimensions {
        let lower = root_lower[dimension];
        let upper = root_upper[dimension];
        let width = upper - lower;
        let dimension_edges = [
            lower + width * 0.0,
            lower + width * 0.5,
            lower + width * 1.0,
        ];
        if dimension_edges.iter().any(|edge| !edge.is_finite()) {
            return Err(DualSeamCertificateError::new(
                "policy split edge is non-finite",
            ));
        }
        edges.push(dimension_edges);
    }

    for (region_index, region) in cover.iter().enumerate() {
        ensure_work_deadline(deadline, "policy-cover validation")?;
        let region = region.flatten();
        if region.lower().len() != root_lower.len() || region.upper().len() != root_upper.len() {
            return Err(DualSeamCertificateError::new(
                "policy region dimensionality mismatch",
            ));
        }
        let mut expected_lower = root_lower.clone();
        let mut expected_upper = root_upper.clone();
        let mut mixed_radix_index = region_index;
        for (split_index, &dimension) in split_dimensions.iter().enumerate() {
            let cell = mixed_radix_index % 2;
            mixed_radix_index /= 2;
            expected_lower[dimension] = edges[split_index][cell];
            expected_upper[dimension] = edges[split_index][cell + 1];
        }
        if region
            .lower()
            .iter()
            .zip(&expected_lower)
            .any(|(&actual, &expected)| actual.to_bits() != expected.to_bits())
            || region
                .upper()
                .iter()
                .zip(&expected_upper)
                .any(|(&actual, &expected)| actual.to_bits() != expected.to_bits())
        {
            return Err(DualSeamCertificateError::new(
                "shared cover does not match the committed policy/order",
            ));
        }
    }
    ensure_work_deadline(deadline, "policy-cover validation")
}

fn expected_orientation(index: usize) -> Option<DualSeamOrientation> {
    match index {
        0 => Some(DualSeamOrientation::NegativeY0),
        1 => Some(DualSeamOrientation::PositiveY0),
        _ => None,
    }
}

fn expected_objective_bits(index: usize) -> Option<u32> {
    match index {
        0 => Some(NEGATIVE_Y0_BITS),
        1 => Some(POSITIVE_Y0_BITS),
        _ => None,
    }
}

fn expected_from_body(body: &CertificateBody) -> DualSeamExpectedBindings {
    DualSeamExpectedBindings::new(
        body.bindings.graph_sha256,
        body.bindings.input_sha256,
        body.bindings.property_sha256,
        body.bindings.seam.clone(),
        body.bindings.seam_elements,
        body.bindings.objective_order_sha256,
        body.bindings.policy_sha256,
        body.bindings.issued_at_unix_ms,
        body.bindings.valid_until_unix_ms,
    )
}

/// Serialize and self-check one complete verdict-neutral dual-seam envelope.
///
/// This function performs no file I/O and has no access to verifier bounds or
/// verdict state. A caller may write the returned JSON as diagnostic evidence.
pub fn emit_dual_seam_certificate_json(
    request: &DualSeamEmissionRequest<'_>,
) -> Result<String, DualSeamCertificateError> {
    ensure_work_deadline(request.work_deadline, "emission")?;
    if request.seam.is_empty() || request.seam.len() > 256 {
        return Err(DualSeamCertificateError::new(
            "seam identity is empty or oversized",
        ));
    }
    if request.seam_elements == 0 || request.seam_elements > DUAL_SEAM_MAX_SEAM_ELEMENTS {
        return Err(DualSeamCertificateError::new(
            "seam width is empty or exceeds the M0 cap",
        ));
    }
    if request.issued_at_unix_ms >= request.valid_until_unix_ms {
        return Err(DualSeamCertificateError::new(
            "certificate wall-clock deadline is not in the future",
        ));
    }
    if !request.policy.is_cgan_row5_k2() {
        return Err(DualSeamCertificateError::new(
            "unsupported dual-seam policy",
        ));
    }
    if request.shared_cover.len() != DUAL_SEAM_REGION_COUNT
        || request.clauses.len() != DUAL_SEAM_CLAUSE_COUNT
    {
        return Err(DualSeamCertificateError::new(
            "M0 requires exactly 16 regions and two clauses",
        ));
    }
    validate_binary_partition_cover(
        request.root_input,
        request.shared_cover,
        request.work_deadline,
    )
    .map_err(|reason| DualSeamCertificateError::new(format!("shared cover rejected: {reason}")))?;
    validate_cgan_row5_policy_cover(
        request.root_input,
        request.shared_cover,
        request.work_deadline,
    )?;

    let root_input = BoxBits::from_tensor(request.root_input)?;
    let shared_cover: Vec<BoxBits> = request
        .shared_cover
        .iter()
        .map(BoxBits::from_tensor)
        .collect::<Result<_, _>>()?;
    ensure_work_deadline(request.work_deadline, "emission")?;
    let mut clauses = Vec::with_capacity(DUAL_SEAM_CLAUSE_COUNT);
    for (clause_index, source) in request.clauses.iter().enumerate() {
        ensure_work_deadline(request.work_deadline, "emission")?;
        if source.objective_index != clause_index
            || Some(source.orientation) != expected_orientation(clause_index)
            || source.objective.len() != 1
            || Some(source.objective[0].to_bits()) != expected_objective_bits(clause_index)
            || !source.threshold.is_finite()
            || source.regions.len() != DUAL_SEAM_REGION_COUNT
        {
            return Err(DualSeamCertificateError::new(
                "clause identity/order/objective/shape rejected",
            ));
        }
        let mut rows = Vec::with_capacity(DUAL_SEAM_REGION_COUNT);
        for (region_index, region) in source.regions.iter().enumerate() {
            if !same_box_bits(&region.region, &request.shared_cover[region_index]) {
                return Err(DualSeamCertificateError::new(
                    "clause region does not match the shared cover",
                ));
            }
            if region.tail_coefficients.is_empty()
                || region.tail_coefficients.len() != request.seam_elements
                || region
                    .tail_coefficients
                    .iter()
                    .any(|value| !value.is_finite())
                || !region.tail_bias.is_finite()
                || !region.prefix_lower.is_finite()
            {
                return Err(DualSeamCertificateError::new(
                    "tail width, bias, or prefix lower is invalid",
                ));
            }
            if region.prefix_frontier.len() > DUAL_SEAM_MAX_PREFIX_LEAVES_PER_REGION {
                return Err(DualSeamCertificateError::new(
                    "prefix frontier exceeds the M0 leaf cap",
                ));
            }
            validate_binary_partition_cover(
                &region.region,
                &region.prefix_frontier,
                request.work_deadline,
            )
            .map_err(|reason| {
                DualSeamCertificateError::new(format!(
                    "prefix frontier rejected for clause {clause_index} region {region_index}: {reason}"
                ))
            })?;
            ensure_work_deadline(request.work_deadline, "emission")?;
            let frontier = region
                .prefix_frontier
                .iter()
                .map(BoxBits::from_tensor)
                .collect::<Result<_, _>>()?;
            rows.push(TailRow {
                region_index,
                coefficient_bits: region
                    .tail_coefficients
                    .iter()
                    .map(|value| value.to_bits())
                    .collect(),
                bias_bits: region.tail_bias.to_bits(),
                prefix: PrefixBoundEvidence {
                    lower_bits: region.prefix_lower.to_bits(),
                    frontier,
                    identity_sha256: [0; 32],
                },
            });
        }
        clauses.push(Clause {
            objective_index: source.objective_index,
            orientation: source.orientation,
            objective_bits: source
                .objective
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            threshold_bits: source.threshold.to_bits(),
            rows,
        });
    }

    ensure_work_deadline(request.work_deadline, "emission")?;
    let input_sha256 = input_identity(&root_input)?;
    let objective_order_sha256 = objective_order_identity(&clauses)?;
    let policy_sha256 = policy_identity(request.policy)?;
    ensure_work_deadline(request.work_deadline, "emission")?;
    let bindings = Bindings {
        graph_sha256: request.graph_sha256,
        input_sha256,
        property_sha256: request.property_sha256,
        seam: request.seam.to_owned(),
        seam_elements: request.seam_elements,
        objective_order_sha256,
        policy_sha256,
        issued_at_unix_ms: request.issued_at_unix_ms,
        valid_until_unix_ms: request.valid_until_unix_ms,
    };
    for clause_index in 0..clauses.len() {
        let identities = {
            let clause = &clauses[clause_index];
            let mut identities = Vec::with_capacity(clause.rows.len());
            for (region_index, row) in clause.rows.iter().enumerate() {
                ensure_work_deadline(request.work_deadline, "emission")?;
                identities.push(prefix_identity(
                    &bindings,
                    clause_index,
                    clause,
                    region_index,
                    &shared_cover[region_index],
                    row,
                )?);
                ensure_work_deadline(request.work_deadline, "emission")?;
            }
            identities
        };
        for (row, identity) in clauses[clause_index].rows.iter_mut().zip(identities) {
            row.prefix.identity_sha256 = identity;
        }
    }
    let body = CertificateBody {
        bindings,
        policy: request.policy.clone(),
        root_input,
        shared_cover,
        clauses,
    };
    let body_sha256 = body_identity(&body)?;
    ensure_work_deadline(request.work_deadline, "emission")?;
    let envelope = CertificateEnvelope {
        schema: DUAL_SEAM_CERTIFICATE_SCHEMA,
        authority: false,
        body_sha256,
        body,
    };
    let encoded = serde_json::to_string_pretty(&envelope).map_err(|error| {
        DualSeamCertificateError::new(format!("certificate serialization failed: {error}"))
    })?;
    ensure_work_deadline(request.work_deadline, "emission")?;
    if encoded.len() > DUAL_SEAM_CERTIFICATE_MAX_BYTES {
        return Err(DualSeamCertificateError::new(
            "certificate exceeds the M0 encoded-size cap",
        ));
    }

    // Self-check through the same public parser/checker before returning bytes.
    let expected = expected_from_body(&envelope.body);
    check_dual_seam_certificate_json(
        &encoded,
        &expected,
        request.issued_at_unix_ms,
        request.work_deadline,
    )?;
    ensure_work_deadline(request.work_deadline, "emission")?;
    Ok(encoded)
}

/// Atomically validate both clauses of a serialized M0 envelope.
///
/// `expected` must come from an independent model/property owner. `now_unix_ms`
/// and `work_deadline` are caller-owned clocks. Success remains
/// verdict-neutral and always reports `authority = false`.
pub fn check_dual_seam_certificate_json(
    encoded: &str,
    expected: &DualSeamExpectedBindings,
    now_unix_ms: u64,
    work_deadline: Instant,
) -> Result<DualSeamCheckReport, DualSeamCertificateError> {
    ensure_work_deadline(work_deadline, "checker")?;
    if encoded.len() > DUAL_SEAM_CERTIFICATE_MAX_BYTES {
        return Err(DualSeamCertificateError::new(
            "certificate exceeds the M0 encoded-size cap",
        ));
    }
    let envelope: CertificateEnvelope = serde_json::from_str(encoded).map_err(|error| {
        DualSeamCertificateError::new(format!("certificate parse failed: {error}"))
    })?;
    ensure_work_deadline(work_deadline, "checker")?;
    if envelope.schema != DUAL_SEAM_CERTIFICATE_SCHEMA || envelope.authority {
        return Err(DualSeamCertificateError::new(
            "schema or M0 authority marker rejected",
        ));
    }
    let body_sha256 = body_identity(&envelope.body)?;
    ensure_work_deadline(work_deadline, "checker")?;
    if body_sha256 != envelope.body_sha256 {
        return Err(DualSeamCertificateError::new(
            "certificate body commitment mismatch",
        ));
    }
    let bindings = &envelope.body.bindings;
    if bindings.graph_sha256 != expected.graph_sha256
        || bindings.input_sha256 != expected.input_sha256
        || bindings.property_sha256 != expected.property_sha256
        || bindings.seam != expected.seam
        || bindings.seam_elements != expected.seam_elements
        || bindings.objective_order_sha256 != expected.objective_order_sha256
        || bindings.policy_sha256 != expected.policy_sha256
        || bindings.issued_at_unix_ms != expected.issued_at_unix_ms
        || bindings.valid_until_unix_ms != expected.valid_until_unix_ms
    {
        return Err(DualSeamCertificateError::new(
            "certificate request binding mismatch",
        ));
    }
    if bindings.seam.is_empty()
        || bindings.seam.len() > 256
        || bindings.seam_elements == 0
        || bindings.seam_elements > DUAL_SEAM_MAX_SEAM_ELEMENTS
        || bindings.issued_at_unix_ms >= bindings.valid_until_unix_ms
        || now_unix_ms < bindings.issued_at_unix_ms
        || now_unix_ms >= bindings.valid_until_unix_ms
    {
        return Err(DualSeamCertificateError::new(
            "seam or certificate deadline rejected",
        ));
    }
    if !envelope.body.policy.is_cgan_row5_k2()
        || policy_identity(&envelope.body.policy)? != bindings.policy_sha256
    {
        return Err(DualSeamCertificateError::new("policy identity mismatch"));
    }
    ensure_work_deadline(work_deadline, "checker")?;
    if input_identity(&envelope.body.root_input)? != bindings.input_sha256 {
        return Err(DualSeamCertificateError::new("input identity mismatch"));
    }
    ensure_work_deadline(work_deadline, "checker")?;
    if envelope.body.shared_cover.len() != DUAL_SEAM_REGION_COUNT
        || envelope.body.clauses.len() != DUAL_SEAM_CLAUSE_COUNT
    {
        return Err(DualSeamCertificateError::new(
            "M0 requires exactly 16 regions and two clauses",
        ));
    }

    let root_input = envelope.body.root_input.to_tensor()?;
    let shared_cover: Vec<BoundedTensor> = envelope
        .body
        .shared_cover
        .iter()
        .map(BoxBits::to_tensor)
        .collect::<Result<_, _>>()?;
    validate_binary_partition_cover(&root_input, &shared_cover, work_deadline).map_err(
        |reason| DualSeamCertificateError::new(format!("shared cover rejected: {reason}")),
    )?;
    validate_cgan_row5_policy_cover(&root_input, &shared_cover, work_deadline)?;
    if objective_order_identity(&envelope.body.clauses)? != bindings.objective_order_sha256 {
        return Err(DualSeamCertificateError::new(
            "objective ordering identity mismatch",
        ));
    }
    ensure_work_deadline(work_deadline, "checker")?;

    let mut prefix_identities =
        HashSet::with_capacity(DUAL_SEAM_CLAUSE_COUNT * DUAL_SEAM_REGION_COUNT);
    for (clause_index, clause) in envelope.body.clauses.iter().enumerate() {
        ensure_work_deadline(work_deadline, "checker")?;
        if clause.objective_index != clause_index
            || Some(clause.orientation) != expected_orientation(clause_index)
            || clause.objective_bits.len() != 1
            || clause.objective_bits.first().copied() != expected_objective_bits(clause_index)
            || !f32::from_bits(clause.threshold_bits).is_finite()
            || clause.rows.len() != DUAL_SEAM_REGION_COUNT
        {
            return Err(DualSeamCertificateError::new(
                "clause identity/order/objective/shape rejected",
            ));
        }
        for (region_index, row) in clause.rows.iter().enumerate() {
            ensure_work_deadline(work_deadline, "checker")?;
            if row.region_index != region_index {
                return Err(DualSeamCertificateError::new(
                    "tail row region ordering mismatch",
                ));
            }
            validate_finite_bits(&row.coefficient_bits, "tail coefficients")?;
            if row.coefficient_bits.len() != bindings.seam_elements
                || !f32::from_bits(row.bias_bits).is_finite()
                || !f32::from_bits(row.prefix.lower_bits).is_finite()
            {
                return Err(DualSeamCertificateError::new(
                    "tail width, bias, or prefix lower is invalid",
                ));
            }
            if row.prefix.frontier.len() > DUAL_SEAM_MAX_PREFIX_LEAVES_PER_REGION {
                return Err(DualSeamCertificateError::new(
                    "prefix frontier exceeds the M0 leaf cap",
                ));
            }
            let prefix_frontier: Vec<BoundedTensor> = row
                .prefix
                .frontier
                .iter()
                .map(BoxBits::to_tensor)
                .collect::<Result<_, _>>()?;
            validate_binary_partition_cover(
                &shared_cover[region_index],
                &prefix_frontier,
                work_deadline,
            )
            .map_err(|reason| {
                DualSeamCertificateError::new(format!(
                    "prefix frontier rejected for clause {clause_index} region {region_index}: {reason}"
                ))
            })?;
            let recomputed = prefix_identity(
                bindings,
                clause_index,
                clause,
                region_index,
                &envelope.body.shared_cover[region_index],
                row,
            )?;
            if recomputed != row.prefix.identity_sha256 {
                return Err(DualSeamCertificateError::new(
                    "prefix-bound certificate identity mismatch",
                ));
            }
            if !prefix_identities.insert(row.prefix.identity_sha256) {
                return Err(DualSeamCertificateError::new(
                    "prefix-bound certificate identity was reused",
                ));
            }
            ensure_work_deadline(work_deadline, "checker")?;
        }
    }
    if prefix_identities.len() != DUAL_SEAM_CLAUSE_COUNT * DUAL_SEAM_REGION_COUNT {
        return Err(DualSeamCertificateError::new(
            "incomplete prefix-bound identity set",
        ));
    }

    // No partial result is returned above: both complete clauses and all 32
    // region-prefix obligations reach this point in one atomic transaction.
    ensure_work_deadline(work_deadline, "checker")?;
    Ok(DualSeamCheckReport {
        clauses_checked: DUAL_SEAM_CLAUSE_COUNT,
        regions_checked: DUAL_SEAM_REGION_COUNT,
        prefix_identities_checked: prefix_identities.len(),
        authority: false,
    })
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use ndarray::{ArrayD, IxDyn};

    use super::*;

    fn vector_box(lower: [f32; 5], upper: [f32; 5]) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[5]), lower.to_vec()).expect("lower"),
            ArrayD::from_shape_vec(IxDyn(&[5]), upper.to_vec()).expect("upper"),
        )
        .expect("box")
    }

    fn fixture_sources() -> (
        BoundedTensor,
        Vec<BoundedTensor>,
        DualSeamPolicy,
        Vec<DualSeamClauseSource>,
    ) {
        let root = vector_box([0.0, 0.0, 0.0, 7.0, 0.0], [2.0, 2.0, 2.0, 7.0, 2.0]);
        let cover: Vec<BoundedTensor> = (0..DUAL_SEAM_REGION_COUNT)
            .map(|region| {
                let mut lower = [0.0, 0.0, 0.0, 7.0, 0.0];
                let mut upper = [2.0, 2.0, 2.0, 7.0, 2.0];
                let mut mixed_radix_index = region;
                for dimension in [0_usize, 1, 2, 4] {
                    let cell = mixed_radix_index % 2;
                    mixed_radix_index /= 2;
                    lower[dimension] = cell as f32;
                    upper[dimension] = cell as f32 + 1.0;
                }
                vector_box(lower, upper)
            })
            .collect();
        let clauses = [
            DualSeamOrientation::NegativeY0,
            DualSeamOrientation::PositiveY0,
        ]
        .into_iter()
        .enumerate()
        .map(|(clause_index, orientation)| {
            let sign = if clause_index == 0 { -1.0 } else { 1.0 };
            DualSeamClauseSource {
                objective_index: clause_index,
                orientation,
                objective: vec![sign],
                threshold: if clause_index == 0 {
                    -0.679_319_44
                } else {
                    0.639_319_4
                },
                regions: cover
                    .iter()
                    .enumerate()
                    .map(|(region_index, region)| DualSeamRegionSource {
                        region: region.clone(),
                        tail_coefficients: vec![sign * (region_index as f32 + 1.0), sign * 0.25],
                        tail_bias: -0.5 - region_index as f32 / 128.0,
                        prefix_lower: 0.75 + region_index as f32 / 64.0,
                        prefix_frontier: vec![region.clone()],
                    })
                    .collect(),
            }
        })
        .collect();
        (root, cover, DualSeamPolicy::cgan_row5_k2(), clauses)
    }

    fn emit_fixture() -> (String, DualSeamExpectedBindings) {
        let (root, cover, policy, clauses) = fixture_sources();
        let work_deadline = Instant::now() + Duration::from_secs(10);
        let encoded = emit_dual_seam_certificate_json(&DualSeamEmissionRequest {
            graph_sha256: [0x11; 32],
            property_sha256: [0x22; 32],
            root_input: &root,
            shared_cover: &cover,
            seam: "Relu_17",
            seam_elements: 2,
            policy: &policy,
            clauses: &clauses,
            issued_at_unix_ms: 1_000,
            valid_until_unix_ms: 2_000,
            work_deadline,
        })
        .expect("fixture emits");
        let envelope: CertificateEnvelope =
            serde_json::from_str(&encoded).expect("fixture envelope");
        (encoded, expected_from_body(&envelope.body))
    }

    fn parse(encoded: &str) -> CertificateEnvelope {
        serde_json::from_str(encoded).expect("parse")
    }

    fn reseal(envelope: &mut CertificateEnvelope) -> String {
        envelope.body_sha256 = body_identity(&envelope.body).expect("body identity");
        serde_json::to_string_pretty(envelope).expect("serialize")
    }

    fn rebind_prefix_identities(envelope: &mut CertificateEnvelope) {
        for clause_index in 0..envelope.body.clauses.len() {
            for region_index in 0..envelope.body.clauses[clause_index].rows.len() {
                let identity = {
                    let clause = &envelope.body.clauses[clause_index];
                    let row = &clause.rows[region_index];
                    prefix_identity(
                        &envelope.body.bindings,
                        clause_index,
                        clause,
                        region_index,
                        &envelope.body.shared_cover[region_index],
                        row,
                    )
                    .expect("prefix identity")
                };
                envelope.body.clauses[clause_index].rows[region_index]
                    .prefix
                    .identity_sha256 = identity;
            }
        }
    }

    fn check(
        encoded: &str,
        expected: &DualSeamExpectedBindings,
    ) -> Result<DualSeamCheckReport, DualSeamCertificateError> {
        check_dual_seam_certificate_json(
            encoded,
            expected,
            1_500,
            Instant::now() + Duration::from_secs(10),
        )
    }

    #[test]
    fn exact_dual_clause_envelope_checks_atomically_without_authority() {
        let (encoded, expected) = emit_fixture();
        let report = check(&encoded, &expected).expect("complete fixture checks");
        assert_eq!(report.clauses_checked(), 2);
        assert_eq!(report.regions_checked(), 16);
        assert_eq!(report.prefix_identities_checked(), 32);
        assert!(!report.authority());
        let envelope = parse(&encoded);
        assert!(!envelope.authority);
    }

    #[test]
    fn m0_emitter_and_checker_have_no_score_path_callsite() {
        let root_inject = include_str!("root_inject.rs");
        assert!(!root_inject.contains("dual_seam_certificate"));
        assert!(!root_inject.contains("emit_dual_seam_certificate_json"));
        assert!(!root_inject.contains("check_dual_seam_certificate_json"));
    }

    #[test]
    fn flipped_tail_coefficient_is_rejected_even_if_outer_body_is_resealed() {
        let (encoded, expected) = emit_fixture();
        let mut envelope = parse(&encoded);
        envelope.body.clauses[0].rows[7].coefficient_bits[0] ^= 1;
        let tampered = reseal(&mut envelope);
        assert!(check(&tampered, &expected).is_err());
    }

    #[test]
    fn omitted_or_overlapping_shared_region_is_rejected() {
        let (encoded, expected) = emit_fixture();
        let mut omitted = parse(&encoded);
        omitted.body.shared_cover.pop();
        let omitted = reseal(&mut omitted);
        assert!(check(&omitted, &expected).is_err());

        let mut overlap = parse(&encoded);
        overlap.body.shared_cover[8].lower_bits[0] = 7.5_f32.to_bits();
        rebind_prefix_identities(&mut overlap);
        let overlap = reseal(&mut overlap);
        assert!(check(&overlap, &expected).is_err());
    }

    #[test]
    fn wrong_seam_or_graph_hash_is_rejected_against_external_context() {
        let (encoded, expected) = emit_fixture();
        let mut wrong_seam = parse(&encoded);
        wrong_seam.body.bindings.seam = "Relu_16".to_owned();
        rebind_prefix_identities(&mut wrong_seam);
        let wrong_seam = reseal(&mut wrong_seam);
        assert!(check(&wrong_seam, &expected).is_err());

        let mut wrong_graph = parse(&encoded);
        wrong_graph.body.bindings.graph_sha256[0] ^= 1;
        rebind_prefix_identities(&mut wrong_graph);
        let wrong_graph = reseal(&mut wrong_graph);
        assert!(check(&wrong_graph, &expected).is_err());
    }

    #[test]
    fn wrong_clause_order_is_rejected_even_if_all_hashes_are_recomputed() {
        let (encoded, expected) = emit_fixture();
        let mut envelope = parse(&encoded);
        envelope.body.clauses.swap(0, 1);
        envelope.body.bindings.objective_order_sha256 =
            objective_order_identity(&envelope.body.clauses).expect("order identity");
        rebind_prefix_identities(&mut envelope);
        let tampered = reseal(&mut envelope);
        assert!(check(&tampered, &expected).is_err());
    }

    #[test]
    fn reordered_cover_is_rejected_even_if_rows_and_all_hashes_follow_it() {
        let (encoded, expected) = emit_fixture();
        let mut envelope = parse(&encoded);
        envelope.body.shared_cover.swap(0, 1);
        for clause in &mut envelope.body.clauses {
            clause.rows.swap(0, 1);
            for (region_index, row) in clause.rows.iter_mut().enumerate() {
                row.region_index = region_index;
            }
        }
        rebind_prefix_identities(&mut envelope);
        let tampered = reseal(&mut envelope);
        let error = check(&tampered, &expected).expect_err("non-canonical cover order");
        assert!(error.reason().contains("committed policy/order"));
    }

    #[test]
    fn exact_but_off_policy_cover_is_rejected_after_complete_reseal() {
        let (encoded, expected) = emit_fixture();
        let mut envelope = parse(&encoded);
        let off_policy_cover: Vec<BoxBits> = (0..DUAL_SEAM_REGION_COUNT)
            .map(|region_index| {
                let lower = region_index as f32 / 8.0;
                let upper = (region_index + 1) as f32 / 8.0;
                BoxBits::from_tensor(&vector_box(
                    [lower, 0.0, 0.0, 7.0, 0.0],
                    [upper, 2.0, 2.0, 7.0, 2.0],
                ))
                .expect("off-policy region")
            })
            .collect();
        envelope.body.shared_cover = off_policy_cover;
        for clause in &mut envelope.body.clauses {
            for (region_index, row) in clause.rows.iter_mut().enumerate() {
                row.prefix.frontier = vec![envelope.body.shared_cover[region_index].clone()];
            }
        }
        rebind_prefix_identities(&mut envelope);
        let tampered = reseal(&mut envelope);
        let error = check(&tampered, &expected).expect_err("off-policy exact cover");
        assert!(error.reason().contains("committed policy/order"));
    }

    #[test]
    fn seam_width_and_issuance_are_bound_to_external_context() {
        let (encoded, expected) = emit_fixture();

        let mut wrong_width = parse(&encoded);
        wrong_width.body.bindings.seam_elements = 1;
        for clause in &mut wrong_width.body.clauses {
            for row in &mut clause.rows {
                row.coefficient_bits.truncate(1);
            }
        }
        rebind_prefix_identities(&mut wrong_width);
        let wrong_width = reseal(&mut wrong_width);
        assert!(check(&wrong_width, &expected).is_err());

        let mut wrong_issuance = parse(&encoded);
        wrong_issuance.body.bindings.issued_at_unix_ms -= 1;
        rebind_prefix_identities(&mut wrong_issuance);
        let wrong_issuance = reseal(&mut wrong_issuance);
        assert!(check(&wrong_issuance, &expected).is_err());
    }

    #[test]
    fn parser_and_prefix_resources_are_capped_before_structural_work() {
        let (encoded, expected) = emit_fixture();
        let mut oversized = encoded.clone();
        oversized.extend(std::iter::repeat_n(
            ' ',
            DUAL_SEAM_CERTIFICATE_MAX_BYTES - encoded.len() + 1,
        ));
        let error = check(&oversized, &expected).expect_err("oversized JSON");
        assert!(error.reason().contains("encoded-size cap"));

        let mut envelope = parse(&encoded);
        let leaf = envelope.body.clauses[0].rows[0].prefix.frontier[0].clone();
        envelope.body.clauses[0].rows[0].prefix.frontier =
            vec![leaf; DUAL_SEAM_MAX_PREFIX_LEAVES_PER_REGION + 1];
        rebind_prefix_identities(&mut envelope);
        let oversized_frontier = reseal(&mut envelope);
        let error = check(&oversized_frontier, &expected).expect_err("oversized frontier");
        assert!(error.reason().contains("leaf cap"));
    }

    #[test]
    fn non_finite_tail_prefix_and_box_payloads_are_rejected() {
        let (encoded, expected) = emit_fixture();
        let mut tail_nan = parse(&encoded);
        tail_nan.body.clauses[1].rows[3].coefficient_bits[1] = f32::NAN.to_bits();
        rebind_prefix_identities(&mut tail_nan);
        let tail_nan = reseal(&mut tail_nan);
        assert!(check(&tail_nan, &expected).is_err());

        let mut prefix_inf = parse(&encoded);
        prefix_inf.body.clauses[0].rows[4].prefix.lower_bits = f32::INFINITY.to_bits();
        rebind_prefix_identities(&mut prefix_inf);
        let prefix_inf = reseal(&mut prefix_inf);
        assert!(check(&prefix_inf, &expected).is_err());

        let mut box_nan = parse(&encoded);
        box_nan.body.shared_cover[2].upper_bits[0] = f32::NAN.to_bits();
        rebind_prefix_identities(&mut box_nan);
        let box_nan = reseal(&mut box_nan);
        assert!(check(&box_nan, &expected).is_err());
    }

    #[test]
    fn expired_certificate_is_rejected_without_partial_clause_result() {
        let (encoded, expected) = emit_fixture();
        let result = check_dual_seam_certificate_json(
            &encoded,
            &expected,
            2_000,
            Instant::now() + Duration::from_secs(10),
        );
        assert!(result.is_err());
    }

    #[test]
    fn certificate_issued_in_the_future_is_rejected() {
        let (encoded, expected) = emit_fixture();
        let result = check_dual_seam_certificate_json(
            &encoded,
            &expected,
            999,
            Instant::now() + Duration::from_secs(10),
        );
        assert!(result.is_err());
    }
}
