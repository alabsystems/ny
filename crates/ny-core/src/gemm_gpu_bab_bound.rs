// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// Phase dispositions and transcripts deliberately remain inline and owned so
// preaccept/terminal paths stay allocation-free and preserve exact-copy
// authority semantics rather than adding API-level boxing or reference churn.
#![allow(clippy::large_enum_variant, clippy::large_types_passed_by_value)]

//! Core-owned authority for an atomic, phase-resident BaB bound transaction.
//!
//! Backend-facing values and receipts in this module are raw and structurally
//! untrusted: core revalidates every association, transcript, resource count,
//! and endpoint shape. Numerical enclosure still relies on an explicitly
//! source-reviewed [`GpuBabBoundNumericalTcb`] implementation. Only a live
//! [`GpuBabBoundPhaseLease`] can issue a consuming per-wave capability,
//! validate the backend's terminal disposition, and construct a
//! [`ValidatedGpuBabBoundWaveResult`].
//!
//! Session authority is never backend-selected: a stable registration owns an
//! O(1) core ledger, and core burns a monotonic generation and derives its
//! nonce before any accepted open allocation. Public transcript fields are
//! audit echoes, not capabilities.
//!
//! ```compile_fail
//! use ny_core::{GpuBabBoundPhaseDescriptor, GpuBabBoundTcbInvocation};
//! fn forge<'a>(descriptor: &'a GpuBabBoundPhaseDescriptor) {
//!     let _ = GpuBabBoundTcbInvocation { descriptor };
//! }
//! ```

use std::{
    collections::HashSet,
    mem::{align_of, size_of},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, MutexGuard,
    },
    time::Instant,
};

use sha2::{Digest, Sha256};

use super::GpuCrownBackward;
use crate::{NyError, Result};

#[path = "gemm_gpu_bab_bound_retained_v2.rs"]
mod retained_v2;
pub use retained_v2::*;

/// Host-side validation ceiling for domains in one resident bound wave.
pub const GPU_BAB_BOUND_MAX_DOMAINS: usize = 1 << 16;
/// Host-side validation ceiling for the canonical objective union of a phase.
pub const GPU_BAB_BOUND_MAX_OBJECTIVES: usize = 1 << 16;
/// Finite policy/plan cap for dispatches in one wave.
pub const GPU_BAB_BOUND_MAX_DISPATCHES_PER_WAVE: usize = 1 << 20;
/// Finite policy cap for queue submissions in one wave.
pub const GPU_BAB_BOUND_MAX_SUBMITS_PER_WAVE: usize = 1 << 12;
/// Host validation cap for each owned f32 arena.
pub const GPU_BAB_BOUND_MAX_ARENA_VALUES: usize = 1 << 28;

const GPU_BAB_BOUND_OWNED_SLICE_MIN_FIXED_BYTES: usize =
    size_of::<Arc<Vec<()>>>() + 2 * size_of::<AtomicUsize>() + size_of::<Vec<()>>();
const GPU_BAB_BOUND_OWNED_SLICE_MAX_HEADER_ALIGN: usize =
    if align_of::<AtomicUsize>() > align_of::<Vec<()>>() {
        align_of::<AtomicUsize>()
    } else {
        align_of::<Vec<()>>()
    };

/// Conservative fixed host charge for one [`GpuBabBoundOwnedSlice`] header.
///
/// The charge covers the inline `Arc` handle, both `Arc` reference counters,
/// the moved `Vec` metadata, and conservative internal layout padding. It does
/// not claim to measure allocator bookkeeping, size-class slack, or process
/// RSS. Add this once per wrapper, including empty wrappers, when accounting
/// host ownership.
pub const GPU_BAB_BOUND_OWNED_SLICE_FIXED_CHARGED_BYTES: usize =
    GPU_BAB_BOUND_OWNED_SLICE_MIN_FIXED_BYTES
        + 2 * (GPU_BAB_BOUND_OWNED_SLICE_MAX_HEADER_ALIGN - 1);

const _: () = assert!(
    GPU_BAB_BOUND_OWNED_SLICE_FIXED_CHARGED_BYTES
        >= size_of::<Arc<Vec<()>>>() + 2 * size_of::<AtomicUsize>() + size_of::<Vec<()>>()
);

const VALIDATION_POLL_STRIDE: usize = 1_024;

/// Effective hard deadline for request-scaled retained-v2 core work.
///
/// The phase descriptor was fully validated when it was constructed and is
/// held immutably by the lease. V2 therefore reuses that sealed certificate
/// and polls this effective deadline while validating/hashing the mutable-size
/// wave owned by its consuming capability.
#[derive(Clone, Copy, Debug)]
struct ResidentValidationDeadline {
    effective: Instant,
    test_injection_enabled: bool,
}

impl ResidentValidationDeadline {
    fn new(request: Instant, phase: Instant) -> Self {
        Self {
            effective: request.min(phase),
            test_injection_enabled: true,
        }
    }

    /// V1 shares the bounded ledger scanner but deliberately stays outside the
    /// retained-v2 deterministic test hook and offline validation contract.
    fn new_without_test_injection(request: Instant, phase: Instant) -> Self {
        Self {
            effective: request.min(phase),
            test_injection_enabled: false,
        }
    }

    fn check(self, label: &str) -> Result<()> {
        if self.test_injection_enabled && resident_injected_validation_deadline(label) {
            return Err(NyError::DeadlineExceeded(format!(
                "GPU BaB resident bound {label} injected validation deadline"
            )));
        }
        check_live(self.effective, label)
    }

    fn poll(self, offset: usize, label: &str) -> Result<()> {
        if offset.is_multiple_of(VALIDATION_POLL_STRIDE) {
            self.check(label)?;
        }
        Ok(())
    }

    fn expired(self, label: &str) -> bool {
        (self.test_injection_enabled && resident_injected_validation_deadline(label))
            || Instant::now() >= self.effective
    }
}

fn poll_resident_validation(
    deadline: Option<ResidentValidationDeadline>,
    offset: usize,
    label: &str,
) -> Result<()> {
    if let Some(deadline) = deadline {
        deadline.poll(offset, label)?;
    }
    Ok(())
}

fn finish_resident_validation(
    deadline: Option<ResidentValidationDeadline>,
    label: &str,
) -> Result<()> {
    if let Some(deadline) = deadline {
        deadline.check(label)?;
    }
    Ok(())
}
const ENDPOINT_BYTES_PER_ROW: usize = 2 * size_of::<f32>();
// The device row sidecar contains objective index, canonical q, status, and
// taint. Parent/child/domain association echoes are reconstructed by the raw
// adapter from q plus the sealed schedule and are independently revalidated.
const RESULT_SIDECAR_BYTES_PER_ROW: usize = 4 * size_of::<u32>();
// This slice admits bounded-domain outcomes only, encoded by one class tag.
// Certified pruning stays closed until core owns a parent/split proof checker.
const DOMAIN_OUTCOME_SIDECAR_BYTES: usize = size_of::<u32>();
const OBJECTIVE_INDEX_WIRE_BYTES: usize = size_of::<u32>();
const SUBCHUNK_WIRE_BYTES: usize = 5 * size_of::<u64>();

// Process-lifetime registration epochs are core-selected, monotonic, and
// never wrapped. Receipt authority is in-memory only and is never persisted or
// accepted across a process restart.
static NEXT_GPU_BAB_BOUND_REGISTRATION_EPOCH: AtomicU64 = AtomicU64::new(1);

/// Catch one TCB unwind without subsequently running attacker-defined panic
/// payload destruction. A payload destructor can itself panic; forgetting the
/// opaque payload makes every core catch boundary single-unwind and fail-closed.
fn catch_tcb_unwind<T>(operation: impl FnOnce() -> T) -> std::result::Result<T, ()> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)) {
        Ok(value) => Ok(value),
        Err(payload) => {
            std::mem::forget(payload);
            Err(())
        }
    }
}

/// Backend-recommended scheduling policy for a retained BaB bound phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBabBoundPhasePolicy {
    pub max_device_bytes: usize,
    pub preferred_domains_per_wave: usize,
    pub minimum_domains_per_wave: usize,
    pub maximum_domains_per_wave: usize,
    pub maximum_objectives: usize,
    pub maximum_dispatches_per_wave: usize,
    pub maximum_submits_per_wave: usize,
}

impl GpuBabBoundPhasePolicy {
    /// Whether the independent policy fields are bounded and usable.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.max_device_bytes > 0
            && self.minimum_domains_per_wave > 0
            && self.minimum_domains_per_wave <= self.preferred_domains_per_wave
            && self.preferred_domains_per_wave <= self.maximum_domains_per_wave
            && self.maximum_domains_per_wave <= GPU_BAB_BOUND_MAX_DOMAINS
            && self.maximum_objectives > 0
            && self.maximum_objectives <= GPU_BAB_BOUND_MAX_OBJECTIVES
            && self.maximum_dispatches_per_wave > 0
            && self.maximum_dispatches_per_wave <= GPU_BAB_BOUND_MAX_DISPATCHES_PER_WAVE
            && self.maximum_submits_per_wave > 0
            && self.maximum_submits_per_wave <= GPU_BAB_BOUND_MAX_SUBMITS_PER_WAVE
    }

    /// Whether one joint `D * R` shape fits this policy and the u32 q sidecar.
    #[must_use]
    pub fn is_valid_for_shape(self, domains: usize, objectives: usize) -> bool {
        self.is_valid()
            && domains >= self.minimum_domains_per_wave
            && domains <= self.maximum_domains_per_wave
            && objectives > 0
            && objectives <= self.maximum_objectives
            && domains
                .checked_mul(objectives)
                .is_some_and(|rows| u32::try_from(rows).is_ok())
    }
}

/// Checked half-open range into one owned typed arena.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuBabBoundArenaRange {
    pub start: usize,
    pub len: usize,
}

impl GpuBabBoundArenaRange {
    fn checked_end(self, arena_len: usize, label: &str) -> Result<usize> {
        let end = self
            .start
            .checked_add(self.len)
            .ok_or_else(|| invalid(format!("{label} range overflows usize")))?;
        if end > arena_len {
            return Err(invalid(format!(
                "{label} range {}/{} exceeds arena length {arena_len}",
                self.start, self.len
            )));
        }
        Ok(end)
    }

    fn slice<'a>(self, arena: &'a [f32], label: &str) -> Result<&'a [f32]> {
        let end = self.checked_end(arena.len(), label)?;
        Ok(&arena[self.start..end])
    }
}

/// Closed semantic roles for immutable f32 phase tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuBabBoundF32TensorRole {
    Parameters,
    CertifiedErrors,
    Relaxations,
    InputLower,
    InputUpper,
    RootLower,
    RootUpper,
    ObjectiveCoefficients,
}

impl GpuBabBoundF32TensorRole {
    fn wire_tag(self) -> u8 {
        match self {
            Self::Parameters => 1,
            Self::CertifiedErrors => 2,
            Self::Relaxations => 3,
            Self::InputLower => 4,
            Self::InputUpper => 5,
            Self::RootLower => 6,
            Self::RootUpper => 7,
            Self::ObjectiveCoefficients => 8,
        }
    }
}

/// Closed semantic roles for immutable u32 phase tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuBabBoundU32TensorRole {
    ObjectiveIndices,
    TopologyMetadata,
}

impl GpuBabBoundU32TensorRole {
    fn wire_tag(self) -> u8 {
        match self {
            Self::ObjectiveIndices => 1,
            Self::TopologyMetadata => 2,
        }
    }
}

/// Immutable, clone-cheap ownership for one bounded GPU-BaB payload slice.
///
/// Construction moves an existing `Vec<T>` into an `Arc` without copying,
/// moving, or reallocating its element buffer. The only infallible allocation
/// performed here is the fixed-size `Arc` control block plus `Vec` metadata;
/// it is never request-scaled. Producers must size-check the request, call
/// `Vec::try_reserve` or `Vec::try_reserve_exact`, and populate the vector
/// before transferring it to [`Self::new`]. This type intentionally exposes no
/// mutable or `Arc<Vec<T>>` access after construction.
///
/// ```compile_fail
/// use ny_core::GpuBabBoundOwnedSlice;
///
/// let mut payload = GpuBabBoundOwnedSlice::new(vec![1_u32, 2]);
/// payload[0] = 3;
/// ```
pub struct GpuBabBoundOwnedSlice<T> {
    // `Arc<[T]>` is normally preferable, but converting a populated `Vec<T>`
    // to it may perform a second request-sized allocation/copy. Retaining the
    // Vec allocation is this authority wrapper's defining requirement.
    #[allow(clippy::rc_buffer)]
    values: Arc<Vec<T>>,
}

impl<T> GpuBabBoundOwnedSlice<T> {
    /// Move a producer-populated vector into immutable shared ownership.
    ///
    /// The vector's element pointer and observed capacity are preserved. The
    /// caller is responsible for having obtained that capacity through a
    /// fallible reserve before populating request-scaled data.
    #[must_use]
    pub fn new(values: Vec<T>) -> Self {
        Self {
            values: Arc::new(values),
        }
    }

    /// Borrow the exact logical payload.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.values.as_slice()
    }

    /// Logical element count used by validation and identity hashing.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Observed backing-vector capacity, including unused reserved elements.
    ///
    /// Accountable host memory must use this value rather than logical length.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.values.capacity()
    }

    /// Conservative fixed charge for this wrapper's handle/control metadata.
    #[must_use]
    pub const fn fixed_charged_bytes() -> usize {
        GPU_BAB_BOUND_OWNED_SLICE_FIXED_CHARGED_BYTES
    }

    /// Conservative accountable host bytes for this wrapper and its element
    /// capacity.
    ///
    /// This is `fixed_charged_bytes() + capacity() * size_of::<T>()`, checked
    /// for overflow. Allocator bookkeeping, size-class slack, and RSS remain
    /// outside this narrow deterministic charge.
    #[must_use]
    pub fn accountable_bytes(&self) -> Option<usize> {
        self.capacity()
            .checked_mul(size_of::<T>())?
            .checked_add(Self::fixed_charged_bytes())
    }
}

impl<T> Clone for GpuBabBoundOwnedSlice<T> {
    fn clone(&self) -> Self {
        Self {
            values: Arc::clone(&self.values),
        }
    }
}

impl<T> Default for GpuBabBoundOwnedSlice<T> {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl<T> From<Vec<T>> for GpuBabBoundOwnedSlice<T> {
    fn from(values: Vec<T>) -> Self {
        Self::new(values)
    }
}

impl<T> std::ops::Deref for GpuBabBoundOwnedSlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> AsRef<[T]> for GpuBabBoundOwnedSlice<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for GpuBabBoundOwnedSlice<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(formatter)
    }
}

impl<T: PartialEq> PartialEq for GpuBabBoundOwnedSlice<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for GpuBabBoundOwnedSlice<T> {}

impl<T: std::hash::Hash> std::hash::Hash for GpuBabBoundOwnedSlice<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(self.as_slice(), state);
    }
}

/// One immutable layer-neutral f32 tensor with an exact semantic role/shape.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuBabBoundF32Tensor {
    pub role: GpuBabBoundF32TensorRole,
    pub shape: Vec<usize>,
    pub values: GpuBabBoundOwnedSlice<f32>,
}

/// One immutable layer-neutral u32 tensor with an exact semantic role/shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuBabBoundU32Tensor {
    pub role: GpuBabBoundU32TensorRole,
    pub shape: Vec<usize>,
    pub values: GpuBabBoundOwnedSlice<u32>,
}

/// Borrowed, schedule-free static payload offered to a numerical TCB.
///
/// Construction validates the complete canonical payload and recomputes the
/// producer's v1 identity under the supplied absolute deadline. The request
/// deliberately contains no dispatch count: only a reviewed backend may
/// propose one through [`GpuBabBoundBackendScheduleDisposition`]. The device
/// cap is explicit phase-owner policy; it is never inferred from host vector
/// capacity or retained-root accounting.
pub struct GpuBabBoundStaticScheduleRequest<'a> {
    topology_schema_version: u32,
    topology_bytes: &'a [u8],
    f32_tensors: &'a [GpuBabBoundF32Tensor],
    u32_tensors: &'a [GpuBabBoundU32Tensor],
    static_payload_identity_sha256: [u8; 32],
    logical_static_device_bytes: usize,
    deadline: Instant,
    requested_max_device_bytes: usize,
}

impl<'a> GpuBabBoundStaticScheduleRequest<'a> {
    /// Validate and bind one exact schedule-independent static payload.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        topology_schema_version: u32,
        topology_bytes: &'a [u8],
        f32_tensors: &'a [GpuBabBoundF32Tensor],
        u32_tensors: &'a [GpuBabBoundU32Tensor],
        claimed_static_payload_identity_sha256: [u8; 32],
        deadline: Instant,
        requested_max_device_bytes: usize,
    ) -> Result<Self> {
        let validation_deadline =
            ResidentValidationDeadline::new_without_test_injection(deadline, deadline);
        validation_deadline.check("static schedule request admission")?;
        if topology_schema_version == 0
            || topology_bytes.is_empty()
            || topology_bytes.len() > GPU_BAB_BOUND_MAX_ARENA_VALUES
        {
            return Err(invalid(
                "static schedule topology schema/bytes must be nonzero, nonempty, and bounded",
            ));
        }
        if requested_max_device_bytes == 0
            || requested_max_device_bytes > GPU_BAB_BOUND_MAX_RESIDENT_DEVICE_BYTES
        {
            return Err(invalid(
                "static schedule requested device cap must be finite and nonzero",
            ));
        }
        if is_zero_identity(claimed_static_payload_identity_sha256) {
            return Err(invalid(
                "static schedule claimed payload identity must be nonzero",
            ));
        }
        validate_static_schedule_payload(f32_tensors, u32_tensors, validation_deadline)?;
        let logical_payload_bytes =
            static_schedule_logical_payload_bytes(topology_bytes.len(), f32_tensors, u32_tensors)?;
        let mut check = |label| validation_deadline.check(label);
        let recomputed = gpu_bab_bound_static_payload_identity_v1(
            topology_schema_version,
            topology_bytes,
            f32_tensors,
            u32_tensors,
            &mut check,
        )?;
        if recomputed != claimed_static_payload_identity_sha256 {
            return Err(invalid(
                "static schedule claimed payload identity does not match exact payload bytes",
            ));
        }
        validation_deadline.check("static schedule request completion")?;
        Ok(Self {
            topology_schema_version,
            topology_bytes,
            f32_tensors,
            u32_tensors,
            static_payload_identity_sha256: recomputed,
            logical_static_device_bytes: logical_payload_bytes,
            deadline,
            requested_max_device_bytes,
        })
    }

    #[must_use]
    pub fn topology_schema_version(&self) -> u32 {
        self.topology_schema_version
    }

    #[must_use]
    pub fn topology_bytes(&self) -> &[u8] {
        self.topology_bytes
    }

    #[must_use]
    pub fn f32_tensors(&self) -> &[GpuBabBoundF32Tensor] {
        self.f32_tensors
    }

    #[must_use]
    pub fn u32_tensors(&self) -> &[GpuBabBoundU32Tensor] {
        self.u32_tensors
    }

    #[must_use]
    pub fn static_payload_identity_sha256(&self) -> &[u8; 32] {
        &self.static_payload_identity_sha256
    }

    /// Exact logical static bytes that a later descriptor must account on the
    /// device. This excludes all adapter-host capacity and receipt charges.
    #[must_use]
    pub fn logical_static_device_bytes(&self) -> usize {
        self.logical_static_device_bytes
    }

    #[must_use]
    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    #[must_use]
    pub fn requested_max_device_bytes(&self) -> usize {
        self.requested_max_device_bytes
    }
}

fn validate_static_schedule_payload(
    f32_tensors: &[GpuBabBoundF32Tensor],
    u32_tensors: &[GpuBabBoundU32Tensor],
    deadline: ResidentValidationDeadline,
) -> Result<()> {
    if f32_tensors.len() != 8 || u32_tensors.len() != 2 {
        return Err(invalid(
            "static schedule payload must contain exactly eight f32 and two u32 tensors",
        ));
    }
    validate_tensor_order(f32_tensors, |tensor| tensor.role.wire_tag(), "f32")?;
    validate_tensor_order(u32_tensors, |tensor| tensor.role.wire_tag(), "u32")?;
    for (index, tensor) in f32_tensors.iter().enumerate() {
        deadline.poll(index, "static schedule f32 tensor metadata")?;
        validate_static_schedule_tensor_shape(
            &tensor.shape,
            tensor.values.len(),
            "f32",
            tensor.role == GpuBabBoundF32TensorRole::Relaxations,
        )?;
        validate_f32_arena_with_deadline(
            tensor.values.as_ref(),
            "static schedule tensor",
            false,
            Some(deadline),
        )?;
    }
    for (index, tensor) in u32_tensors.iter().enumerate() {
        deadline.poll(index, "static schedule u32 tensor metadata")?;
        validate_static_schedule_tensor_shape(
            &tensor.shape,
            tensor.values.len(),
            "u32",
            tensor.role == GpuBabBoundU32TensorRole::TopologyMetadata,
        )?;
    }

    let f32 = |role| {
        f32_tensors
            .iter()
            .find(|tensor| tensor.role == role)
            .ok_or_else(|| invalid(format!("missing required {role:?} f32 tensor")))
    };
    let u32 = |role| {
        u32_tensors
            .iter()
            .find(|tensor| tensor.role == role)
            .ok_or_else(|| invalid(format!("missing required {role:?} u32 tensor")))
    };
    let parameters = f32(GpuBabBoundF32TensorRole::Parameters)?;
    let errors = f32(GpuBabBoundF32TensorRole::CertifiedErrors)?;
    let input_lower = f32(GpuBabBoundF32TensorRole::InputLower)?;
    let input_upper = f32(GpuBabBoundF32TensorRole::InputUpper)?;
    let root_lower = f32(GpuBabBoundF32TensorRole::RootLower)?;
    let root_upper = f32(GpuBabBoundF32TensorRole::RootUpper)?;
    let objective_coefficients = f32(GpuBabBoundF32TensorRole::ObjectiveCoefficients)?;
    let objective_indices = u32(GpuBabBoundU32TensorRole::ObjectiveIndices)?;
    let _topology_metadata = u32(GpuBabBoundU32TensorRole::TopologyMetadata)?;

    if parameters.values.is_empty() {
        return Err(invalid("static schedule parameter tensor must be nonempty"));
    }
    for (index, &value) in errors.values.iter().enumerate() {
        deadline.poll(index, "static schedule certified-error validation")?;
        if value < 0.0 {
            return Err(invalid(
                "static schedule certified-error tensor must be nonnegative",
            ));
        }
    }
    validate_static_schedule_bound_pair(input_lower, input_upper, "input", deadline)?;
    validate_static_schedule_bound_pair(root_lower, root_upper, "root", deadline)?;
    if objective_indices.shape.len() != 1
        || objective_indices.values.is_empty()
        || objective_indices.values.len() > GPU_BAB_BOUND_MAX_OBJECTIVES
        || objective_coefficients.shape.len() != 2
        || objective_coefficients.shape[0] != objective_indices.values.len()
        || objective_coefficients.shape[1] != root_lower.values.len()
    {
        return Err(invalid(
            "static schedule objective index/coefficient tensors have inconsistent shapes",
        ));
    }
    for (index, &objective) in objective_indices.values.iter().enumerate() {
        deadline.poll(index, "static schedule objective index validation")?;
        if usize::try_from(objective).ok() != Some(index) {
            return Err(invalid(
                "static schedule objective indices must be dense and canonical",
            ));
        }
    }
    deadline.check("static schedule payload validation completion")
}

fn validate_static_schedule_tensor_shape(
    shape: &[usize],
    values: usize,
    label: &str,
    allows_canonical_empty: bool,
) -> Result<()> {
    let canonical_empty = shape == [0] && values == 0;
    if shape.is_empty()
        || shape.len() > 8
        || (allows_canonical_empty && !canonical_empty)
        || (!allows_canonical_empty && shape.contains(&0))
    {
        return Err(invalid(format!(
            "static schedule {label} tensor rank/dimensions are not canonical"
        )));
    }
    validate_tensor_shape(shape, values, label)
}

fn static_schedule_logical_payload_bytes(
    topology_bytes: usize,
    f32_tensors: &[GpuBabBoundF32Tensor],
    u32_tensors: &[GpuBabBoundU32Tensor],
) -> Result<usize> {
    let mut total = topology_bytes;
    for tensor in f32_tensors {
        total = total
            .checked_add(checked_element_bytes(
                tensor.values.len(),
                size_of::<f32>(),
                "static schedule f32 tensor",
            )?)
            .ok_or_else(|| invalid("static schedule logical payload bytes overflow usize"))?;
    }
    for tensor in u32_tensors {
        total = total
            .checked_add(checked_element_bytes(
                tensor.values.len(),
                size_of::<u32>(),
                "static schedule u32 tensor",
            )?)
            .ok_or_else(|| invalid("static schedule logical payload bytes overflow usize"))?;
    }
    Ok(total)
}

fn validate_static_schedule_bound_pair(
    lower: &GpuBabBoundF32Tensor,
    upper: &GpuBabBoundF32Tensor,
    label: &str,
    deadline: ResidentValidationDeadline,
) -> Result<()> {
    if lower.shape != upper.shape || lower.values.is_empty() {
        return Err(invalid(format!(
            "static schedule {label} bound tensor shapes must match and be nonempty"
        )));
    }
    for (index, (&lower, &upper)) in lower.values.iter().zip(upper.values.iter()).enumerate() {
        deadline.poll(index, "static schedule bound validation")?;
        validate_interval(lower, upper, label, index)?;
    }
    deadline.check("static schedule bound validation completion")
}

/// Compute the canonical schedule-independent retained-BaB v1 payload identity.
///
/// This is the single owner of the identity wire format used by both payload
/// composition and schedule admission. `check` is invoked at bounded strides
/// and at every variable-length section boundary; callers supply their own
/// deadline/budget poll without changing the hashed bytes.
pub fn gpu_bab_bound_static_payload_identity_v1(
    topology_schema_version: u32,
    topology_bytes: &[u8],
    f32_tensors: &[GpuBabBoundF32Tensor],
    u32_tensors: &[GpuBabBoundU32Tensor],
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(b"ny.resident-bab.static-payload.v1\0");
    hash.update(topology_schema_version.to_le_bytes());
    hash_u64(
        &mut hash,
        usize_to_u64(topology_bytes.len(), "topology bytes")?,
    );
    for chunk in topology_bytes.chunks(VALIDATION_POLL_STRIDE) {
        check("resident static identity topology")?;
        hash.update(chunk);
    }
    check("resident static identity topology final")?;
    hash_u64(
        &mut hash,
        usize_to_u64(f32_tensors.len(), "f32 tensor count")?,
    );
    for tensor in f32_tensors {
        hash.update([tensor.role.wire_tag()]);
        hash_static_schedule_shape(&mut hash, &tensor.shape, check)?;
        hash_u64(
            &mut hash,
            usize_to_u64(tensor.values.len(), "f32 tensor values")?,
        );
        for (index, value) in tensor.values.iter().enumerate() {
            if index.is_multiple_of(VALIDATION_POLL_STRIDE) {
                check("resident static identity f32")?;
            }
            hash.update(value.to_bits().to_le_bytes());
        }
        check("resident static identity f32 final")?;
    }
    hash_u64(
        &mut hash,
        usize_to_u64(u32_tensors.len(), "u32 tensor count")?,
    );
    for tensor in u32_tensors {
        hash.update([tensor.role.wire_tag()]);
        hash_static_schedule_shape(&mut hash, &tensor.shape, check)?;
        hash_u64(
            &mut hash,
            usize_to_u64(tensor.values.len(), "u32 tensor values")?,
        );
        for (index, value) in tensor.values.iter().enumerate() {
            if index.is_multiple_of(VALIDATION_POLL_STRIDE) {
                check("resident static identity u32")?;
            }
            hash.update(value.to_le_bytes());
        }
        check("resident static identity u32 final")?;
    }
    Ok(hash.finalize().into())
}

fn hash_static_schedule_shape(
    hash: &mut Sha256,
    shape: &[usize],
    check: &mut dyn FnMut(&'static str) -> Result<()>,
) -> Result<()> {
    hash_u64(hash, usize_to_u64(shape.len(), "tensor rank")?);
    for (index, &dim) in shape.iter().enumerate() {
        if index.is_multiple_of(VALIDATION_POLL_STRIDE) {
            check("resident static identity shape")?;
        }
        hash_u64(hash, usize_to_u64(dim, "tensor dimension")?);
    }
    check("resident static identity shape final")?;
    Ok(())
}

fn usize_to_u64(value: usize, label: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| {
        invalid(format!(
            "static schedule {label} does not fit the v1 wire width"
        ))
    })
}

/// Layer-neutral immutable phase plan.
///
/// Graph-specific canonical encoding belongs to the graph owner; core accepts
/// only nonempty versioned topology bytes plus closed-role typed tensors. Every
/// transcript identity and byte count is computed from these exact immutable
/// bytes/bits. `dispatches_per_subchunk` is a checked, hashed work-plan field,
/// not an unbounded receipt claim.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuBabBoundGraphPlan {
    pub topology_schema_version: u32,
    pub topology_bytes: GpuBabBoundOwnedSlice<u8>,
    pub f32_tensors: Vec<GpuBabBoundF32Tensor>,
    pub u32_tensors: Vec<GpuBabBoundU32Tensor>,
    pub dispatches_per_subchunk: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpuBabBoundPhaseAuthority {
    graph_identity_sha256: [u8; 32],
    static_phase_identity_sha256: [u8; 32],
    input_identity_sha256: [u8; 32],
    root_bounds_identity_sha256: [u8; 32],
    relaxation_identity_sha256: [u8; 32],
    objective_set_identity_sha256: [u8; 32],
    total_objectives: usize,
    dispatches_per_subchunk: usize,
    static_graph_payload_bytes: usize,
    static_phase_payload_bytes: usize,
}

impl GpuBabBoundGraphPlan {
    fn validate_and_authority(&self) -> Result<GpuBabBoundPhaseAuthority> {
        if self.topology_schema_version == 0
            || self.topology_bytes.is_empty()
            || self.topology_bytes.len() > GPU_BAB_BOUND_MAX_ARENA_VALUES
        {
            return Err(invalid(
                "topology schema/bytes must be nonzero, nonempty, and bounded",
            ));
        }
        if self.dispatches_per_subchunk == 0
            || self.dispatches_per_subchunk > GPU_BAB_BOUND_MAX_DISPATCHES_PER_WAVE
        {
            return Err(invalid("plan dispatch count must be finite and nonzero"));
        }
        validate_tensor_order(&self.f32_tensors, |tensor| tensor.role.wire_tag(), "f32")?;
        validate_tensor_order(&self.u32_tensors, |tensor| tensor.role.wire_tag(), "u32")?;
        for tensor in &self.f32_tensors {
            validate_tensor_shape(&tensor.shape, tensor.values.len(), "f32")?;
            validate_f32_arena(tensor.values.as_ref(), "phase tensor", false)?;
        }
        for tensor in &self.u32_tensors {
            validate_tensor_shape(&tensor.shape, tensor.values.len(), "u32")?;
        }

        let parameters = self.f32(GpuBabBoundF32TensorRole::Parameters)?;
        let errors = self.f32(GpuBabBoundF32TensorRole::CertifiedErrors)?;
        let relaxations = self.f32(GpuBabBoundF32TensorRole::Relaxations)?;
        let input_lower = self.f32(GpuBabBoundF32TensorRole::InputLower)?;
        let input_upper = self.f32(GpuBabBoundF32TensorRole::InputUpper)?;
        let root_lower = self.f32(GpuBabBoundF32TensorRole::RootLower)?;
        let root_upper = self.f32(GpuBabBoundF32TensorRole::RootUpper)?;
        let objective_coefficients = self.f32(GpuBabBoundF32TensorRole::ObjectiveCoefficients)?;
        let objective_indices = self.u32(GpuBabBoundU32TensorRole::ObjectiveIndices)?;
        if parameters.values.is_empty() {
            return Err(invalid("parameter tensor must be nonempty"));
        }
        if errors.values.iter().any(|&value| value < 0.0) {
            return Err(invalid("certified-error tensor must be nonnegative"));
        }
        validate_matching_bound_tensors(input_lower, input_upper, "input")?;
        validate_matching_bound_tensors(root_lower, root_upper, "root")?;
        if objective_indices.shape.len() != 1
            || objective_indices.values.is_empty()
            || objective_indices.values.len() > GPU_BAB_BOUND_MAX_OBJECTIVES
            || objective_coefficients.shape.len() != 2
            || objective_coefficients.shape[0] != objective_indices.values.len()
            || objective_coefficients.shape[1] != root_lower.values.len()
        {
            return Err(invalid(
                "objective index/coefficient tensors have inconsistent shapes",
            ));
        }
        for (index, &objective) in objective_indices.values.iter().enumerate() {
            if usize::try_from(objective).ok() != Some(index) {
                return Err(invalid("objective indices must be dense and canonical"));
            }
        }

        let graph_identity_sha256 = hash_graph_plan(self);
        let input_identity_sha256 =
            hash_tensor_pair(b"ny.gpu-bab-bound.input.v2\0", input_lower, input_upper);
        let root_bounds_identity_sha256 =
            hash_tensor_pair(b"ny.gpu-bab-bound.root.v2\0", root_lower, root_upper);
        let relaxation_identity_sha256 =
            hash_f32_tensor(b"ny.gpu-bab-bound.relaxation.v2\0", relaxations);
        let objective_set_identity_sha256 = hash_objective_plan(self);
        let topology_metadata = self.u32(GpuBabBoundU32TensorRole::TopologyMetadata)?;
        let static_graph_payload_bytes = checked_payload_total(
            "static graph",
            [
                self.topology_bytes.len(),
                checked_element_bytes(parameters.values.len(), size_of::<f32>(), "parameters")?,
                checked_element_bytes(errors.values.len(), size_of::<f32>(), "errors")?,
                checked_element_bytes(
                    topology_metadata.values.len(),
                    size_of::<u32>(),
                    "topology metadata",
                )?,
            ],
        )?;
        let static_phase_payload_bytes = checked_payload_total(
            "static phase",
            [
                checked_element_bytes(relaxations.values.len(), size_of::<f32>(), "relaxations")?,
                checked_element_bytes(input_lower.values.len(), size_of::<f32>(), "input lower")?,
                checked_element_bytes(input_upper.values.len(), size_of::<f32>(), "input upper")?,
                checked_element_bytes(root_lower.values.len(), size_of::<f32>(), "root lower")?,
                checked_element_bytes(root_upper.values.len(), size_of::<f32>(), "root upper")?,
                checked_element_bytes(
                    objective_coefficients.values.len(),
                    size_of::<f32>(),
                    "objective coefficients",
                )?,
                checked_element_bytes(
                    objective_indices.values.len(),
                    size_of::<u32>(),
                    "objective indices",
                )?,
            ],
        )?;
        let static_phase_identity_sha256 = hash_static_phase_identity(
            input_identity_sha256,
            root_bounds_identity_sha256,
            relaxation_identity_sha256,
            objective_set_identity_sha256,
            static_phase_payload_bytes,
        );

        Ok(GpuBabBoundPhaseAuthority {
            graph_identity_sha256,
            static_phase_identity_sha256,
            input_identity_sha256,
            root_bounds_identity_sha256,
            relaxation_identity_sha256,
            objective_set_identity_sha256,
            total_objectives: objective_indices.values.len(),
            dispatches_per_subchunk: self.dispatches_per_subchunk,
            static_graph_payload_bytes,
            static_phase_payload_bytes,
        })
    }

    fn f32(&self, role: GpuBabBoundF32TensorRole) -> Result<&GpuBabBoundF32Tensor> {
        self.f32_tensors
            .iter()
            .find(|tensor| tensor.role == role)
            .ok_or_else(|| invalid(format!("missing required {role:?} f32 tensor")))
    }

    fn u32(&self, role: GpuBabBoundU32TensorRole) -> Result<&GpuBabBoundU32Tensor> {
        self.u32_tensors
            .iter()
            .find(|tensor| tensor.role == role)
            .ok_or_else(|| invalid(format!("missing required {role:?} u32 tensor")))
    }
}

fn validate_tensor_order<T>(tensors: &[T], tag: impl Fn(&T) -> u8, label: &str) -> Result<()> {
    let mut prior = None;
    for tensor in tensors {
        let current = tag(tensor);
        if prior.is_some_and(|value| value >= current) {
            return Err(invalid(format!(
                "{label} tensor roles must be strictly canonical and unique"
            )));
        }
        prior = Some(current);
    }
    Ok(())
}

fn validate_tensor_shape(shape: &[usize], values: usize, label: &str) -> Result<()> {
    if shape.is_empty() || shape.len() > 8 {
        return Err(invalid(format!("{label} tensor rank must be in 1..=8")));
    }
    let product = shape.iter().try_fold(1usize, |product, &dim| {
        product
            .checked_mul(dim)
            .ok_or_else(|| invalid(format!("{label} tensor shape overflows usize")))
    })?;
    if product != values || values > GPU_BAB_BOUND_MAX_ARENA_VALUES {
        return Err(invalid(format!(
            "{label} tensor shape product {product} != bounded value count {values}"
        )));
    }
    Ok(())
}

fn validate_matching_bound_tensors(
    lower: &GpuBabBoundF32Tensor,
    upper: &GpuBabBoundF32Tensor,
    label: &str,
) -> Result<()> {
    if lower.shape != upper.shape || lower.values.is_empty() {
        return Err(invalid(format!(
            "{label} bound tensor shapes must match and be nonempty"
        )));
    }
    for (index, (&lower, &upper)) in lower.values.iter().zip(upper.values.iter()).enumerate() {
        validate_interval(lower, upper, label, index)?;
    }
    Ok(())
}

fn validate_f32_arena(values: &[f32], label: &str, require_nonempty: bool) -> Result<()> {
    validate_f32_arena_with_deadline(values, label, require_nonempty, None)
}

fn validate_f32_arena_with_deadline(
    values: &[f32],
    label: &str,
    require_nonempty: bool,
    deadline: Option<ResidentValidationDeadline>,
) -> Result<()> {
    if values.len() > GPU_BAB_BOUND_MAX_ARENA_VALUES || (require_nonempty && values.is_empty()) {
        return Err(invalid(format!(
            "{label} arena must be bounded{} and finite",
            if require_nonempty { ", nonempty," } else { "" }
        )));
    }
    for (index, value) in values.iter().enumerate() {
        poll_resident_validation(deadline, index, "resident f32 arena validation")?;
        if !value.is_finite() {
            return Err(invalid(format!(
                "{label} arena must be bounded{} and finite",
                if require_nonempty { ", nonempty," } else { "" }
            )));
        }
    }
    finish_resident_validation(deadline, "resident f32 arena validation")?;
    Ok(())
}

fn checked_element_bytes(values: usize, element_bytes: usize, label: &str) -> Result<usize> {
    values
        .checked_mul(element_bytes)
        .ok_or_else(|| invalid(format!("{label} static payload bytes overflow usize")))
}

fn checked_payload_total(label: &str, parts: impl IntoIterator<Item = usize>) -> Result<usize> {
    let total = parts.into_iter().try_fold(0usize, |total, part| {
        total
            .checked_add(part)
            .ok_or_else(|| invalid(format!("{label} payload total overflows usize")))
    })?;
    if total == 0 {
        return Err(invalid(format!("{label} payload must be nonzero")));
    }
    Ok(total)
}

/// Immutable phase descriptor whose transcript identities are core-computed.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuBabBoundPhaseDescriptor {
    plan: GpuBabBoundGraphPlan,
    authority: GpuBabBoundPhaseAuthority,
    deadline: Instant,
    max_device_bytes: usize,
}

impl GpuBabBoundPhaseDescriptor {
    pub fn new(
        plan: GpuBabBoundGraphPlan,
        deadline: Instant,
        max_device_bytes: usize,
    ) -> Result<Self> {
        let authority = plan.validate_and_authority()?;
        let descriptor = Self {
            plan,
            authority,
            deadline,
            max_device_bytes,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    #[must_use]
    pub fn plan(&self) -> &GpuBabBoundGraphPlan {
        &self.plan
    }

    #[must_use]
    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    #[must_use]
    pub fn max_device_bytes(&self) -> usize {
        self.max_device_bytes
    }

    #[must_use]
    pub fn total_objectives(&self) -> usize {
        self.authority.total_objectives
    }

    #[must_use]
    pub fn static_graph_payload_bytes(&self) -> usize {
        self.authority.static_graph_payload_bytes
    }

    #[must_use]
    pub fn static_phase_payload_bytes(&self) -> usize {
        self.authority.static_phase_payload_bytes
    }

    /// Validate core-computed authority/cap fields and current liveness.
    pub fn validate(&self) -> Result<()> {
        self.validate_static()?;
        check_live(self.deadline, "phase descriptor")
    }

    fn validate_static(&self) -> Result<()> {
        if self.max_device_bytes == 0 {
            return Err(invalid("phase max_device_bytes must be nonzero"));
        }
        let recomputed = self.plan.validate_and_authority()?;
        if recomputed != self.authority {
            return Err(invalid(
                "phase authority does not match its typed graph plan",
            ));
        }
        let static_payload_bytes = self
            .authority
            .static_graph_payload_bytes
            .checked_add(self.authority.static_phase_payload_bytes)
            .ok_or_else(|| invalid("aggregate static payload bytes overflow usize"))?;
        if static_payload_bytes > self.max_device_bytes {
            return Err(invalid(format!(
                "aggregate static payload bytes {static_payload_bytes} exceed phase cap {}",
                self.max_device_bytes
            )));
        }
        Ok(())
    }

    fn validate_sealed_for_resident(&self, deadline: ResidentValidationDeadline) -> Result<()> {
        // Construction and phase opening already recomputed the complete
        // immutable plan authority. Repeating those potentially huge static
        // tensor scans inside every v2 wave would defeat that wave's hard
        // deadline without adding a safe-code mutation check: all descriptor
        // fields are private and exposed only by shared borrow.
        deadline.check("resident sealed phase descriptor")?;
        if self.max_device_bytes == 0 {
            return Err(invalid("phase max_device_bytes must be nonzero"));
        }
        Ok(())
    }
}

fn hash_graph_plan(plan: &GpuBabBoundGraphPlan) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"ny.gpu-bab-bound.graph-plan.v2\0");
    hash.update(plan.topology_schema_version.to_le_bytes());
    hash_u64(&mut hash, plan.topology_bytes.len() as u64);
    hash.update(plan.topology_bytes.as_ref());
    hash_u64(&mut hash, plan.dispatches_per_subchunk as u64);
    for tensor in &plan.f32_tensors {
        if matches!(
            tensor.role,
            GpuBabBoundF32TensorRole::Parameters | GpuBabBoundF32TensorRole::CertifiedErrors
        ) {
            hash_f32_tensor_into(&mut hash, tensor);
        }
    }
    for tensor in &plan.u32_tensors {
        if tensor.role == GpuBabBoundU32TensorRole::TopologyMetadata {
            hash_u32_tensor_into(&mut hash, tensor);
        }
    }
    hash.finalize().into()
}

fn hash_tensor_pair(
    domain: &[u8],
    lower: &GpuBabBoundF32Tensor,
    upper: &GpuBabBoundF32Tensor,
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash_f32_tensor_into(&mut hash, lower);
    hash_f32_tensor_into(&mut hash, upper);
    hash.finalize().into()
}

fn hash_f32_tensor(domain: &[u8], tensor: &GpuBabBoundF32Tensor) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash_f32_tensor_into(&mut hash, tensor);
    hash.finalize().into()
}

fn hash_objective_plan(plan: &GpuBabBoundGraphPlan) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"ny.gpu-bab-bound.objective-plan.v2\0");
    hash_u32_tensor_into(
        &mut hash,
        plan.u32(GpuBabBoundU32TensorRole::ObjectiveIndices)
            .expect("validated plan has objective indices"),
    );
    hash_f32_tensor_into(
        &mut hash,
        plan.f32(GpuBabBoundF32TensorRole::ObjectiveCoefficients)
            .expect("validated plan has objective coefficients"),
    );
    hash.finalize().into()
}

fn hash_static_phase_identity(
    input_identity_sha256: [u8; 32],
    root_bounds_identity_sha256: [u8; 32],
    relaxation_identity_sha256: [u8; 32],
    objective_set_identity_sha256: [u8; 32],
    payload_bytes: usize,
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"ny.gpu-bab-bound.static-phase.v1\0");
    hash.update(input_identity_sha256);
    hash.update(root_bounds_identity_sha256);
    hash.update(relaxation_identity_sha256);
    hash.update(objective_set_identity_sha256);
    hash_u64(&mut hash, payload_bytes as u64);
    hash.finalize().into()
}

fn hash_f32_tensor_into(hash: &mut Sha256, tensor: &GpuBabBoundF32Tensor) {
    hash.update([tensor.role.wire_tag()]);
    hash_shape_into(hash, &tensor.shape);
    hash_f32s_into(hash, tensor.values.as_ref());
}

fn hash_u32_tensor_into(hash: &mut Sha256, tensor: &GpuBabBoundU32Tensor) {
    hash.update([tensor.role.wire_tag()]);
    hash_shape_into(hash, &tensor.shape);
    hash_u64(hash, tensor.values.len() as u64);
    for value in tensor.values.iter() {
        hash.update(value.to_le_bytes());
    }
}

fn hash_shape_into(hash: &mut Sha256, shape: &[usize]) {
    hash_u64(hash, shape.len() as u64);
    for &dim in shape {
        hash_u64(hash, dim as u64);
    }
}

fn hash_f32s_into(hash: &mut Sha256, values: &[f32]) {
    hash_f32s_into_with_deadline(hash, values, None, "f32 identity")
        .expect("deadline-free f32 hashing is infallible");
}

fn hash_f32s_into_with_deadline(
    hash: &mut Sha256,
    values: &[f32],
    deadline: Option<ResidentValidationDeadline>,
    label: &str,
) -> Result<()> {
    hash_u64(hash, values.len() as u64);
    for (index, value) in values.iter().enumerate() {
        poll_resident_validation(deadline, index, label)?;
        hash.update(value.to_bits().to_le_bytes());
    }
    finish_resident_validation(deadline, label)
}

/// Core-issued backend/session identity; this is an audit echo, never a
/// capability. The process registration epoch, generation, and nonce are
/// core-selected; two registrations with the same configured backend identity
/// cannot collide during one process lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuBabBoundBackendIssuerIdentity {
    pub backend_issuer_sha256: [u8; 32],
    pub registration_epoch: u64,
    pub generation: u64,
    pub session_nonce_sha256: [u8; 32],
}

impl GpuBabBoundBackendIssuerIdentity {
    fn validate(self) -> Result<()> {
        if is_zero_identity(self.backend_issuer_sha256) {
            return Err(invalid("backend issuer identity must be nonzero"));
        }
        if self.registration_epoch == 0 {
            return Err(invalid("backend registration epoch must be nonzero"));
        }
        if self.generation == 0 {
            return Err(invalid("backend generation must be nonzero"));
        }
        if is_zero_identity(self.session_nonce_sha256) {
            return Err(invalid("backend session nonce must be nonzero"));
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct GpuBabBoundBurnLedger {
    highest_generation: u64,
    live: Option<GpuBabBoundBackendIssuerIdentity>,
    poisoned: bool,
}

struct GpuBabBoundTerminalClaim {
    identity: GpuBabBoundBackendIssuerIdentity,
}

/// Stable backend registration with an O(1), no-eviction burn ledger.
///
/// The object is intentionally non-`Clone`; a backend may share one explicit
/// `Arc` registration across its instances. Core assigns every new object a
/// checked, nonrepeating process-lifetime epoch. A reviewed provider must expose
/// one stable borrowed registration for its exact qualified device epoch;
/// replacing it is explicit device requalification/TCB expansion, never a way
/// to clear poison. Receipt authority is not persisted across process restart.
pub struct GpuBabBoundBackendRegistration {
    backend_issuer_sha256: [u8; 32],
    registration_epoch: u64,
    schedule_identity: Option<GpuBabBoundBackendScheduleIdentity>,
    ledger: Mutex<GpuBabBoundBurnLedger>,
}

impl GpuBabBoundBackendRegistration {
    pub fn new(backend_issuer_sha256: [u8; 32]) -> Result<Self> {
        Self::new_inner(backend_issuer_sha256, None)
    }

    /// Create a registration qualified for exact pre-descriptor scheduling.
    ///
    /// Existing registrations remain schedule-dark through [`Self::new`]. The
    /// immutable schema/kernel bundle supplied here is the trusted expectation
    /// against which core checks every raw schedule evidence echo.
    pub fn new_with_schedule_identity(
        backend_issuer_sha256: [u8; 32],
        schedule_identity: GpuBabBoundBackendScheduleIdentity,
    ) -> Result<Self> {
        schedule_identity.validate()?;
        Self::new_inner(backend_issuer_sha256, Some(schedule_identity))
    }

    fn new_inner(
        backend_issuer_sha256: [u8; 32],
        schedule_identity: Option<GpuBabBoundBackendScheduleIdentity>,
    ) -> Result<Self> {
        if is_zero_identity(backend_issuer_sha256) {
            return Err(invalid("backend registration identity must be nonzero"));
        }
        let registration_epoch = next_registration_epoch(&NEXT_GPU_BAB_BOUND_REGISTRATION_EPOCH)?;
        Ok(Self {
            backend_issuer_sha256,
            registration_epoch,
            schedule_identity,
            ledger: Mutex::new(GpuBabBoundBurnLedger::default()),
        })
    }

    #[must_use]
    pub fn backend_issuer_sha256(&self) -> &[u8; 32] {
        &self.backend_issuer_sha256
    }

    /// Core-selected process-lifetime authority epoch. This is an audit echo,
    /// not a capability or persistence promise.
    #[must_use]
    pub fn registration_epoch(&self) -> u64 {
        self.registration_epoch
    }

    /// Immutable reviewed schedule schema/kernel bundle, when explicitly
    /// qualified. `None` keeps certification incapable of succeeding.
    #[must_use]
    pub fn schedule_identity(&self) -> Option<GpuBabBoundBackendScheduleIdentity> {
        self.schedule_identity
    }

    fn claim(
        &self,
        phase: &GpuBabBoundPhaseDescriptor,
    ) -> (GpuBabBoundBackendIssuerIdentity, Result<()>) {
        let mut ledger = match self.ledger.lock() {
            Ok(ledger) => ledger,
            Err(poisoned) => {
                let mut ledger = poisoned.into_inner();
                ledger.poisoned = true;
                let generation = ledger.highest_generation.saturating_add(1).max(1);
                ledger.highest_generation = generation;
                return (
                    derive_session_identity(
                        self.backend_issuer_sha256,
                        self.registration_epoch,
                        generation,
                        phase,
                    ),
                    Err(invalid("backend registration lock was poisoned")),
                );
            }
        };
        let generation = match ledger.highest_generation.checked_add(1) {
            Some(generation) if generation != 0 => generation,
            _ => {
                ledger.poisoned = true;
                let identity = derive_session_identity(
                    self.backend_issuer_sha256,
                    self.registration_epoch,
                    u64::MAX,
                    phase,
                );
                return (
                    identity,
                    Err(invalid(
                        "backend generation exhausted; registration is poisoned",
                    )),
                );
            }
        };
        // Burn before every admission decision, including concurrent and
        // poisoned observations. No history table or eviction exists.
        ledger.highest_generation = generation;
        let identity = derive_session_identity(
            self.backend_issuer_sha256,
            self.registration_epoch,
            generation,
            phase,
        );
        if generation == u64::MAX {
            ledger.poisoned = true;
            return (
                identity,
                Err(invalid(
                    "backend generation reached u64::MAX; registration is poisoned",
                )),
            );
        }
        if ledger.poisoned {
            return (
                identity,
                Err(invalid("backend registration is permanently poisoned")),
            );
        }
        if ledger.live.is_some() {
            ledger.poisoned = true;
            return (
                identity,
                Err(invalid(
                    "concurrent backend registration claim permanently poisons authority",
                )),
            );
        }
        ledger.live = Some(identity);
        (identity, Ok(()))
    }

    /// No-allocation registration release used after a raw terminal has
    /// transferred physical authority. The ledger is poisoned on every
    /// mismatch before this returns `false`.
    fn release_noalloc(&self, identity: GpuBabBoundBackendIssuerIdentity) -> bool {
        let mut ledger = match self.ledger.lock() {
            Ok(ledger) => ledger,
            Err(poisoned) => {
                poisoned.into_inner().poisoned = true;
                return false;
            }
        };
        if ledger.poisoned {
            return false;
        }
        if identity.backend_issuer_sha256 != self.backend_issuer_sha256
            || identity.registration_epoch != self.registration_epoch
            || ledger.highest_generation != identity.generation
            || ledger.live != Some(identity)
        {
            ledger.poisoned = true;
            return false;
        }
        ledger.live = None;
        true
    }

    /// No-allocation exact-live lookup for authority-settlement critical
    /// regions. Any poisoned lock or identity mismatch is absorbed into the
    /// registration ledger before returning `None`.
    fn live_guard_noalloc(
        &self,
        identity: GpuBabBoundBackendIssuerIdentity,
    ) -> Option<MutexGuard<'_, GpuBabBoundBurnLedger>> {
        let mut ledger = match self.ledger.lock() {
            Ok(ledger) => ledger,
            Err(poisoned) => {
                poisoned.into_inner().poisoned = true;
                return None;
            }
        };
        if identity.backend_issuer_sha256 != self.backend_issuer_sha256
            || identity.registration_epoch != self.registration_epoch
            || identity.generation == 0
            || is_zero_identity(identity.session_nonce_sha256)
            || ledger.poisoned
            || ledger.highest_generation != identity.generation
            || ledger.live != Some(identity)
        {
            ledger.poisoned = true;
            return None;
        }
        Some(ledger)
    }

    fn live_guard(
        &self,
        identity: GpuBabBoundBackendIssuerIdentity,
    ) -> Result<MutexGuard<'_, GpuBabBoundBurnLedger>> {
        self.live_guard_noalloc(identity).ok_or_else(|| {
            invalid("backend registration no longer owns the exact live session identity")
        })
    }

    fn check_live(&self, identity: GpuBabBoundBackendIssuerIdentity) -> Result<()> {
        drop(self.live_guard(identity)?);
        Ok(())
    }

    fn check_live_noalloc(&self, identity: GpuBabBoundBackendIssuerIdentity) -> bool {
        match self.live_guard_noalloc(identity) {
            Some(guard) => {
                drop(guard);
                true
            }
            None => false,
        }
    }

    fn terminal_claim_noalloc(
        &self,
        identity: GpuBabBoundBackendIssuerIdentity,
    ) -> Option<GpuBabBoundTerminalClaim> {
        let mut guard = self.live_guard_noalloc(identity)?;
        guard.poisoned = true;
        Some(GpuBabBoundTerminalClaim { identity })
    }

    fn terminal_claim(
        &self,
        identity: GpuBabBoundBackendIssuerIdentity,
    ) -> Result<GpuBabBoundTerminalClaim> {
        self.terminal_claim_noalloc(identity)
            .ok_or_else(|| invalid("backend registration terminal claim lost live authority"))
    }

    fn available_guard(&self) -> Result<MutexGuard<'_, GpuBabBoundBurnLedger>> {
        let ledger = match self.ledger.lock() {
            Ok(ledger) => ledger,
            Err(poisoned) => {
                poisoned.into_inner().poisoned = true;
                return Err(invalid("backend registration lock was poisoned"));
            }
        };
        if ledger.poisoned || ledger.live.is_some() {
            return Err(invalid(
                "backend registration is poisoned or already owns a live phase",
            ));
        }
        Ok(ledger)
    }

    fn poison_registration(&self) {
        let mut ledger = match self.ledger.lock() {
            Ok(ledger) => ledger,
            Err(poisoned) => poisoned.into_inner(),
        };
        ledger.poisoned = true;
    }

    fn poison(&self, identity: GpuBabBoundBackendIssuerIdentity) {
        let mut ledger = match self.ledger.lock() {
            Ok(ledger) => ledger,
            Err(poisoned) => poisoned.into_inner(),
        };
        if identity.backend_issuer_sha256 == self.backend_issuer_sha256
            && identity.registration_epoch == self.registration_epoch
            && identity.generation > ledger.highest_generation
        {
            ledger.highest_generation = identity.generation;
        }
        ledger.poisoned = true;
    }
}

fn next_registration_epoch(counter: &AtomicU64) -> Result<u64> {
    counter
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |epoch| {
            if epoch == 0 || epoch == u64::MAX {
                None
            } else {
                epoch.checked_add(1)
            }
        })
        .map_err(|_| invalid("process registration epoch exhausted; authority is closed"))
}

fn derive_session_identity(
    backend_issuer_sha256: [u8; 32],
    registration_epoch: u64,
    generation: u64,
    phase: &GpuBabBoundPhaseDescriptor,
) -> GpuBabBoundBackendIssuerIdentity {
    let mut hash = Sha256::new();
    hash.update(b"ny.gpu-bab-bound.core-session-nonce.v2\0");
    hash.update(backend_issuer_sha256);
    hash.update(registration_epoch.to_le_bytes());
    hash.update(generation.to_le_bytes());
    hash.update(phase.authority.graph_identity_sha256);
    hash.update(phase.authority.static_phase_identity_sha256);
    hash.update(phase.authority.input_identity_sha256);
    hash.update(phase.authority.root_bounds_identity_sha256);
    hash.update(phase.authority.relaxation_identity_sha256);
    hash.update(phase.authority.objective_set_identity_sha256);
    hash.update(phase.authority.static_graph_payload_bytes.to_le_bytes());
    hash.update(phase.authority.static_phase_payload_bytes.to_le_bytes());
    let mut session_nonce_sha256: [u8; 32] = hash.finalize().into();
    session_nonce_sha256[0] |= 1;
    GpuBabBoundBackendIssuerIdentity {
        backend_issuer_sha256,
        registration_epoch,
        generation,
        session_nonce_sha256,
    }
}

/// Canonical descriptor of one parent and all of its children in this wave.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuBabBoundParentGroup {
    pub parent_group_id: u64,
    pub parent_identity_sha256: [u8; 32],
    pub first_domain: usize,
    pub child_cardinality: usize,
}

/// Actual owned dynamic operands for all domains in one wave.
///
/// Domain views below must exactly partition every arena. Identities and wire
/// byte counts are computed by core from these values; callers cannot supply a
/// hash standing in for data the backend never received.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuBabBoundDomainArena {
    pub activation: GpuBabBoundOwnedSlice<f32>,
    pub beta: GpuBabBoundOwnedSlice<f32>,
    pub abs: GpuBabBoundOwnedSlice<f32>,
    pub box_lower: GpuBabBoundOwnedSlice<f32>,
    pub box_upper: GpuBabBoundOwnedSlice<f32>,
    pub cached_la: GpuBabBoundOwnedSlice<f32>,
}

/// Exact per-domain views into [`GpuBabBoundDomainArena`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuBabBoundOperandView {
    pub activation: GpuBabBoundArenaRange,
    pub beta: GpuBabBoundArenaRange,
    pub abs: GpuBabBoundArenaRange,
    pub box_lower: GpuBabBoundArenaRange,
    pub box_upper: GpuBabBoundArenaRange,
    pub cached_la: GpuBabBoundArenaRange,
}

/// Exact caller slot, parent membership, child ordinal, state, and operands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuBabBoundDomainTranscript {
    pub parent_group_id: u64,
    pub child_ordinal: usize,
    pub child_cardinality: usize,
    pub domain_slot: u64,
    pub operands: GpuBabBoundOperandView,
}

/// Contiguous same-parent domain subchunk carrying the complete objective union.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBabBoundSubchunk {
    pub parent_group_id: u64,
    pub first_domain: usize,
    pub domain_count: usize,
    pub first_q: usize,
    pub row_count: usize,
}

/// Owned, replay-safe input data for one candidate wave.
///
/// This request carries no acceptance or verdict authority. Core validates and
/// hashes it before asking the raw backend for a clean preaccept decision.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuBabBoundWaveRequest {
    pub parent_groups: Vec<GpuBabBoundParentGroup>,
    pub domains: Vec<GpuBabBoundDomainTranscript>,
    pub domain_arena: GpuBabBoundDomainArena,
    pub objective_indices: Vec<u32>,
    pub subchunks: Vec<GpuBabBoundSubchunk>,
    pub inherited_lower: Vec<f32>,
    pub inherited_upper: Vec<f32>,
    pub deadline: Instant,
    pub max_device_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct ValidatedWaveShape {
    domains: usize,
    rows: usize,
    // Exact returned rows for completed validation; the static value is the
    // all-bounded upper bound and is replaced after outcome validation.
    returned_rows: usize,
    domain_operand_bytes: usize,
    activation_operand_bytes: usize,
    beta_operand_bytes: usize,
    abs_operand_bytes: usize,
    box_operand_bytes: usize,
    cached_la_operand_bytes: usize,
    inherited_endpoint_bytes: usize,
    objective_index_bytes: usize,
    subchunk_descriptor_bytes: usize,
    required_dispatches: usize,
    schedule_identity_sha256: [u8; 32],
    inherited_endpoints_sha256: [u8; 32],
}

impl GpuBabBoundWaveRequest {
    /// Checked domain-major `D * R` row count.
    pub fn row_count(&self) -> Result<usize> {
        self.domains
            .len()
            .checked_mul(self.objective_indices.len())
            .ok_or_else(|| invalid("wave D * R row count overflows usize"))
    }

    /// Recompute the canonical schedule identity after full static validation.
    pub fn canonical_schedule_sha256(
        &self,
        phase: &GpuBabBoundPhaseDescriptor,
    ) -> Result<[u8; 32]> {
        Ok(self.validate_static(phase)?.schedule_identity_sha256)
    }

    /// Core-computed identity for the exact domain association and operand
    /// values at `domain_index`.
    pub fn domain_identity_sha256(&self, domain_index: usize) -> Result<[u8; 32]> {
        let domain = self
            .domains
            .get(domain_index)
            .ok_or_else(|| invalid("domain identity index is out of range"))?;
        hash_domain_identity(domain, &self.domain_arena, domain_index)
    }

    fn validate_for_prepare(
        &self,
        phase: &GpuBabBoundPhaseDescriptor,
    ) -> Result<ValidatedWaveShape> {
        phase.validate()?;
        let shape = self.validate_static(phase)?;
        if self.deadline > phase.deadline {
            return Err(invalid("wave deadline exceeds phase deadline"));
        }
        check_live(self.deadline, "wave request")?;
        Ok(shape)
    }

    fn validate_for_resident_prepare(
        &self,
        phase: &GpuBabBoundPhaseDescriptor,
        host_budget: &mut ResidentHostAdmissionBudget,
    ) -> Result<ValidatedWaveShape> {
        if self.deadline > phase.deadline {
            return Err(invalid("wave deadline exceeds phase deadline"));
        }
        let deadline = ResidentValidationDeadline::new(self.deadline, phase.deadline);
        phase.validate_sealed_for_resident(deadline)?;
        let shape =
            self.validate_static_with_resident_budget(phase, Some(host_budget), Some(deadline))?;
        deadline.check("resident wave request")?;
        Ok(shape)
    }

    fn validate_static(&self, phase: &GpuBabBoundPhaseDescriptor) -> Result<ValidatedWaveShape> {
        self.validate_static_with_resident_budget(phase, None, None)
    }

    fn validate_static_with_resident_budget(
        &self,
        phase: &GpuBabBoundPhaseDescriptor,
        mut host_budget: Option<&mut ResidentHostAdmissionBudget>,
        deadline: Option<ResidentValidationDeadline>,
    ) -> Result<ValidatedWaveShape> {
        if deadline.is_none() {
            phase.validate_static()?;
        }
        if self.max_device_bytes == 0 || self.max_device_bytes > phase.max_device_bytes {
            return Err(invalid(format!(
                "wave max_device_bytes {} must be in 1..={}",
                self.max_device_bytes, phase.max_device_bytes
            )));
        }
        if self.domains.is_empty() || self.domains.len() > GPU_BAB_BOUND_MAX_DOMAINS {
            return Err(invalid(format!(
                "domain count {} must be in 1..={GPU_BAB_BOUND_MAX_DOMAINS}",
                self.domains.len()
            )));
        }
        validate_objectives(&self.objective_indices, phase.total_objectives(), deadline)?;
        validate_parent_partition(
            &self.parent_groups,
            &self.domains,
            host_budget.as_deref_mut(),
            deadline,
        )?;
        let operand_bytes =
            validate_domains(&self.domains, &self.domain_arena, host_budget, deadline)?;

        let rows = self.row_count()?;
        if rows > u32::MAX as usize {
            return Err(invalid("D * R exceeds the u32 q-sidecar range"));
        }
        if self.inherited_lower.len() != rows || self.inherited_upper.len() != rows {
            return Err(invalid(format!(
                "inherited endpoint lengths ({}, {}) != D * R ({rows})",
                self.inherited_lower.len(),
                self.inherited_upper.len()
            )));
        }
        for (q, (&lower, &upper)) in self
            .inherited_lower
            .iter()
            .zip(self.inherited_upper.iter())
            .enumerate()
        {
            poll_resident_validation(deadline, q, "resident inherited intervals")?;
            validate_interval(lower, upper, "inherited", q)?;
        }
        finish_resident_validation(deadline, "resident inherited intervals")?;
        validate_subchunks(
            &self.subchunks,
            &self.parent_groups,
            self.objective_indices.len(),
            rows,
            deadline,
        )?;

        let inherited_endpoint_bytes = rows
            .checked_mul(ENDPOINT_BYTES_PER_ROW)
            .ok_or_else(|| invalid("inherited endpoint bytes overflow usize"))?;
        let objective_index_bytes = self
            .objective_indices
            .len()
            .checked_mul(OBJECTIVE_INDEX_WIRE_BYTES)
            .ok_or_else(|| invalid("objective-index bytes overflow usize"))?;
        let subchunk_descriptor_bytes = self
            .subchunks
            .len()
            .checked_mul(SUBCHUNK_WIRE_BYTES)
            .ok_or_else(|| invalid("subchunk descriptor bytes overflow usize"))?;
        let required_dispatches = phase
            .authority
            .dispatches_per_subchunk
            .checked_mul(self.subchunks.len())
            .ok_or_else(|| invalid("wave dispatch count overflows usize"))?;
        if required_dispatches == 0 || required_dispatches > GPU_BAB_BOUND_MAX_DISPATCHES_PER_WAVE {
            return Err(invalid(
                "wave dispatch count exceeds the finite core ceiling",
            ));
        }
        Ok(ValidatedWaveShape {
            domains: self.domains.len(),
            rows,
            returned_rows: rows,
            domain_operand_bytes: operand_bytes.total()?,
            activation_operand_bytes: operand_bytes.activation,
            beta_operand_bytes: operand_bytes.beta,
            abs_operand_bytes: operand_bytes.abs,
            box_operand_bytes: operand_bytes.box_bounds,
            cached_la_operand_bytes: operand_bytes.cached_la,
            inherited_endpoint_bytes,
            objective_index_bytes,
            subchunk_descriptor_bytes,
            required_dispatches,
            schedule_identity_sha256: hash_schedule_with_deadline(self, phase, deadline)?,
            inherited_endpoints_sha256: hash_inherited_endpoints_with_deadline(self, deadline)?,
        })
    }
}

fn validate_objectives(
    objectives: &[u32],
    total: usize,
    deadline: Option<ResidentValidationDeadline>,
) -> Result<()> {
    if objectives.is_empty() || objectives.len() > GPU_BAB_BOUND_MAX_OBJECTIVES {
        return Err(invalid(format!(
            "objective union length {} must be in 1..={GPU_BAB_BOUND_MAX_OBJECTIVES}",
            objectives.len()
        )));
    }
    let mut previous = None;
    for (position, &objective) in objectives.iter().enumerate() {
        poll_resident_validation(deadline, position, "resident objective validation")?;
        if usize::try_from(objective).map_or(true, |value| value >= total) {
            return Err(invalid(format!(
                "objective index {objective} at position {position} is out of range"
            )));
        }
        if previous.is_some_and(|prior| prior >= objective) {
            return Err(invalid(
                "objective union must be strictly ascending and duplicate-free",
            ));
        }
        previous = Some(objective);
    }
    finish_resident_validation(deadline, "resident objective validation")?;
    Ok(())
}

fn validator_capacity_error(requested: usize) -> NyError {
    NyError::GpuBatchCapacityExceeded {
        requested,
        capacity: 0,
        unit: "validator entries",
        site: "gpu_bab_bound_core_validation",
    }
}

fn validate_parent_partition(
    groups: &[GpuBabBoundParentGroup],
    domains: &[GpuBabBoundDomainTranscript],
    host_budget: Option<&mut ResidentHostAdmissionBudget>,
    deadline: Option<ResidentValidationDeadline>,
) -> Result<()> {
    if groups.is_empty() {
        return Err(invalid("parent group partition must be nonempty"));
    }
    let mut next_domain = 0usize;
    let mut previous_group = None;
    let mut parent_identities = HashSet::new();
    if resident_injected_allocation_failure() {
        return Err(validator_capacity_error(groups.len()));
    }
    parent_identities
        .try_reserve(groups.len())
        .map_err(|_| validator_capacity_error(groups.len()))?;
    if let Some(host_budget) = host_budget {
        host_budget
            .charge_metadata_capacity(
                groups.len(),
                parent_identities.capacity(),
                GPU_BAB_BOUND_HOST_HISTORY_RECORD_VALIDATION_BYTES,
            )
            .map_err(|()| validator_capacity_error(groups.len()))?;
    }
    finish_resident_validation(deadline, "resident parent validator reserve")?;
    for (group_index, group) in groups.iter().enumerate() {
        poll_resident_validation(deadline, group_index, "resident parent partition")?;
        if group.parent_group_id == 0
            || previous_group.is_some_and(|previous| previous >= group.parent_group_id)
        {
            return Err(invalid(
                "parent group IDs must be nonzero and strictly ascending",
            ));
        }
        if is_zero_identity(group.parent_identity_sha256) {
            return Err(invalid(format!(
                "parent group {group_index} identity must be nonzero"
            )));
        }
        if !parent_identities.insert(group.parent_identity_sha256) {
            return Err(invalid(format!(
                "parent identity is duplicated at group {group_index}"
            )));
        }
        if group.child_cardinality == 0 || group.first_domain != next_domain {
            return Err(invalid(format!(
                "parent group {group_index} leaves a gap, overlap, or empty cardinality"
            )));
        }
        let end = group
            .first_domain
            .checked_add(group.child_cardinality)
            .ok_or_else(|| invalid("parent group coverage overflows usize"))?;
        if end > domains.len() {
            return Err(invalid(format!(
                "parent group {group_index} exceeds the domain array"
            )));
        }
        for (ordinal, domain) in domains[group.first_domain..end].iter().enumerate() {
            poll_resident_validation(
                deadline,
                group.first_domain + ordinal,
                "resident parent-domain partition",
            )?;
            if domain.parent_group_id != group.parent_group_id
                || domain.child_ordinal != ordinal
                || domain.child_cardinality != group.child_cardinality
            {
                return Err(invalid(format!(
                    "domain {} does not exactly echo parent {}, ordinal {ordinal}, cardinality {}",
                    group.first_domain + ordinal,
                    group.parent_group_id,
                    group.child_cardinality
                )));
            }
        }
        next_domain = end;
        previous_group = Some(group.parent_group_id);
    }
    finish_resident_validation(deadline, "resident parent partition")?;
    if next_domain != domains.len() {
        return Err(invalid(
            "parent groups must cover every domain exactly once",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
struct OperandByteTotals {
    activation: usize,
    beta: usize,
    abs: usize,
    box_bounds: usize,
    cached_la: usize,
}

impl OperandByteTotals {
    fn total(self) -> Result<usize> {
        self.activation
            .checked_add(self.beta)
            .and_then(|value| value.checked_add(self.abs))
            .and_then(|value| value.checked_add(self.box_bounds))
            .and_then(|value| value.checked_add(self.cached_la))
            .ok_or_else(|| invalid("aggregate typed operand bytes overflow usize"))
    }
}

fn validate_domains(
    domains: &[GpuBabBoundDomainTranscript],
    arena: &GpuBabBoundDomainArena,
    mut host_budget: Option<&mut ResidentHostAdmissionBudget>,
    deadline: Option<ResidentValidationDeadline>,
) -> Result<OperandByteTotals> {
    for (label, values) in [
        ("activation", arena.activation.as_ref()),
        ("beta", arena.beta.as_ref()),
        ("abs", arena.abs.as_ref()),
        ("box lower", arena.box_lower.as_ref()),
        ("box upper", arena.box_upper.as_ref()),
        ("cached-lA", arena.cached_la.as_ref()),
    ] {
        validate_f32_arena_with_deadline(values, label, false, deadline)?;
    }
    for (index, &value) in arena.abs.iter().enumerate() {
        poll_resident_validation(deadline, index, "resident absolute-value operands")?;
        if value < 0.0 {
            return Err(invalid("absolute-value operand arena must be nonnegative"));
        }
    }
    finish_resident_validation(deadline, "resident absolute-value operands")?;
    if arena.box_lower.len() != arena.box_upper.len() {
        return Err(invalid("box lower/upper arenas must have equal lengths"));
    }

    let mut seen_slots = HashSet::new();
    if resident_injected_allocation_failure() {
        return Err(validator_capacity_error(domains.len()));
    }
    seen_slots
        .try_reserve(domains.len())
        .map_err(|_| validator_capacity_error(domains.len()))?;
    if let Some(host_budget) = host_budget.as_deref_mut() {
        host_budget
            .charge_metadata_capacity(
                domains.len(),
                seen_slots.capacity(),
                GPU_BAB_BOUND_HOST_HISTORY_RECORD_VALIDATION_BYTES,
            )
            .map_err(|()| validator_capacity_error(domains.len()))?;
    }
    finish_resident_validation(deadline, "resident domain-slot validator reserve")?;
    let mut seen_identities = HashSet::new();
    if resident_injected_allocation_failure() {
        return Err(validator_capacity_error(domains.len()));
    }
    seen_identities
        .try_reserve(domains.len())
        .map_err(|_| validator_capacity_error(domains.len()))?;
    if let Some(host_budget) = host_budget {
        host_budget
            .charge_metadata_capacity(
                domains.len(),
                seen_identities.capacity(),
                GPU_BAB_BOUND_HOST_HISTORY_RECORD_VALIDATION_BYTES,
            )
            .map_err(|()| validator_capacity_error(domains.len()))?;
    }
    finish_resident_validation(deadline, "resident domain-identity validator reserve")?;
    let mut totals = OperandByteTotals::default();
    for (index, domain) in domains.iter().enumerate() {
        poll_resident_validation(deadline, index, "resident domain association")?;
        if domain.domain_slot == 0 {
            return Err(invalid(format!(
                "domain slot at position {index} must be nonzero"
            )));
        }
        if !seen_slots.insert(domain.domain_slot) {
            return Err(invalid(format!(
                "domain slot {} is duplicated at position {index}",
                domain.domain_slot
            )));
        }
        let identity = hash_domain_identity_with_deadline(domain, arena, index, deadline)?;
        if !seen_identities.insert(identity) {
            return Err(invalid(format!(
                "domain identity is duplicated at position {index}"
            )));
        }
    }
    finish_resident_validation(deadline, "resident domain association")?;
    validate_exact_partition(
        domains,
        arena.activation.len(),
        "activation",
        |view| view.activation,
        true,
        deadline,
    )?;
    validate_exact_partition(
        domains,
        arena.beta.len(),
        "beta",
        |view| view.beta,
        false,
        deadline,
    )?;
    validate_exact_partition(
        domains,
        arena.abs.len(),
        "abs",
        |view| view.abs,
        false,
        deadline,
    )?;
    validate_exact_partition(
        domains,
        arena.box_lower.len(),
        "box lower",
        |view| view.box_lower,
        true,
        deadline,
    )?;
    validate_exact_partition(
        domains,
        arena.box_upper.len(),
        "box upper",
        |view| view.box_upper,
        true,
        deadline,
    )?;
    validate_exact_partition(
        domains,
        arena.cached_la.len(),
        "cached-lA",
        |view| view.cached_la,
        false,
        deadline,
    )?;
    for (index, domain) in domains.iter().enumerate() {
        poll_resident_validation(deadline, index, "resident domain box validation")?;
        if domain.operands.box_lower.len != domain.operands.box_upper.len {
            return Err(invalid(format!(
                "domain {index} box lower/upper views must have equal lengths"
            )));
        }
        let lower = domain
            .operands
            .box_lower
            .slice(arena.box_lower.as_ref(), "box lower")?;
        let upper = domain
            .operands
            .box_upper
            .slice(arena.box_upper.as_ref(), "box upper")?;
        for (coordinate, (&lower, &upper)) in lower.iter().zip(upper).enumerate() {
            poll_resident_validation(deadline, coordinate, "resident domain box coordinates")?;
            validate_interval(lower, upper, "domain box", coordinate)?;
        }
        finish_resident_validation(deadline, "resident domain box coordinates")?;
    }
    finish_resident_validation(deadline, "resident domain box validation")?;
    totals.activation = checked_f32_bytes(arena.activation.len(), "activation")?;
    totals.beta = checked_f32_bytes(arena.beta.len(), "beta")?;
    totals.abs = checked_f32_bytes(arena.abs.len(), "abs")?;
    totals.box_bounds = checked_f32_bytes(
        arena
            .box_lower
            .len()
            .checked_add(arena.box_upper.len())
            .ok_or_else(|| invalid("box arena length overflows usize"))?,
        "box",
    )?;
    totals.cached_la = checked_f32_bytes(arena.cached_la.len(), "cached-lA")?;
    let _ = totals.total()?;
    Ok(totals)
}

fn validate_exact_partition(
    domains: &[GpuBabBoundDomainTranscript],
    arena_len: usize,
    label: &str,
    select: impl Fn(&GpuBabBoundOperandView) -> GpuBabBoundArenaRange,
    require_each_nonempty: bool,
    deadline: Option<ResidentValidationDeadline>,
) -> Result<()> {
    let mut next = 0usize;
    for (index, domain) in domains.iter().enumerate() {
        poll_resident_validation(deadline, index, "resident exact operand partition")?;
        let range = select(&domain.operands);
        if range.start != next || (require_each_nonempty && range.len == 0) {
            return Err(invalid(format!(
                "domain {index} {label} view leaves a gap/overlap or is empty"
            )));
        }
        next = range.checked_end(arena_len, label)?;
    }
    finish_resident_validation(deadline, "resident exact operand partition")?;
    if next != arena_len {
        return Err(invalid(format!(
            "{label} views do not cover the exact arena"
        )));
    }
    Ok(())
}

fn checked_f32_bytes(values: usize, label: &str) -> Result<usize> {
    values
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| invalid(format!("{label} operand bytes overflow usize")))
}

fn hash_domain_identity(
    domain: &GpuBabBoundDomainTranscript,
    arena: &GpuBabBoundDomainArena,
    index: usize,
) -> Result<[u8; 32]> {
    hash_domain_identity_with_deadline(domain, arena, index, None)
}

fn hash_domain_identity_with_deadline(
    domain: &GpuBabBoundDomainTranscript,
    arena: &GpuBabBoundDomainArena,
    index: usize,
    deadline: Option<ResidentValidationDeadline>,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(b"ny.gpu-bab-bound.domain-arena.v2\0");
    hash_u64(&mut hash, domain.parent_group_id);
    hash_u64(&mut hash, domain.child_ordinal as u64);
    hash_u64(&mut hash, domain.child_cardinality as u64);
    hash_u64(&mut hash, domain.domain_slot);
    for (label, range, values) in [
        (
            "activation",
            domain.operands.activation,
            arena.activation.as_ref(),
        ),
        ("beta", domain.operands.beta, arena.beta.as_ref()),
        ("abs", domain.operands.abs, arena.abs.as_ref()),
        (
            "box lower",
            domain.operands.box_lower,
            arena.box_lower.as_ref(),
        ),
        (
            "box upper",
            domain.operands.box_upper,
            arena.box_upper.as_ref(),
        ),
        (
            "cached-lA",
            domain.operands.cached_la,
            arena.cached_la.as_ref(),
        ),
    ] {
        hash_u64(&mut hash, range.start as u64);
        hash_u64(&mut hash, range.len as u64);
        hash_f32s_into_with_deadline(
            &mut hash,
            range.slice(values, label).map_err(|error| {
                invalid(format!("domain {index} operand view is invalid: {error}"))
            })?,
            deadline,
            "resident domain identity operands",
        )?;
    }
    finish_resident_validation(deadline, "resident domain identity")?;
    Ok(hash.finalize().into())
}

fn validate_subchunks(
    subchunks: &[GpuBabBoundSubchunk],
    groups: &[GpuBabBoundParentGroup],
    objective_rows: usize,
    total_rows: usize,
    deadline: Option<ResidentValidationDeadline>,
) -> Result<()> {
    if subchunks.is_empty() {
        return Err(invalid("subchunk partition must be nonempty"));
    }
    let mut next_domain = 0usize;
    let mut next_q = 0usize;
    let mut group_index = 0usize;
    for (index, subchunk) in subchunks.iter().enumerate() {
        poll_resident_validation(deadline, index, "resident subchunk partition")?;
        if subchunk.domain_count == 0
            || subchunk.first_domain != next_domain
            || subchunk.first_q != next_q
        {
            return Err(invalid(format!(
                "subchunk {index} leaves a domain/q gap or overlap"
            )));
        }
        let expected_rows = subchunk
            .domain_count
            .checked_mul(objective_rows)
            .ok_or_else(|| invalid("subchunk full-R rows overflow usize"))?;
        if subchunk.row_count != expected_rows {
            return Err(invalid(format!(
                "subchunk {index} row_count {} != domain_count * R ({expected_rows})",
                subchunk.row_count
            )));
        }
        let end_domain = subchunk
            .first_domain
            .checked_add(subchunk.domain_count)
            .ok_or_else(|| invalid("subchunk domain range overflows usize"))?;
        while let Some(group) = groups.get(group_index) {
            let group_end = group
                .first_domain
                .checked_add(group.child_cardinality)
                .ok_or_else(|| invalid("parent group coverage overflows usize"))?;
            if subchunk.first_domain < group_end {
                break;
            }
            group_index += 1;
            poll_resident_validation(deadline, group_index, "resident subchunk parent cursor")?;
        }
        let group = groups
            .get(group_index)
            .ok_or_else(|| invalid(format!("subchunk {index} crosses a parent boundary")))?;
        let group_end = group
            .first_domain
            .checked_add(group.child_cardinality)
            .ok_or_else(|| invalid("parent group coverage overflows usize"))?;
        if subchunk.first_domain < group.first_domain || end_domain > group_end {
            return Err(invalid(format!(
                "subchunk {index} crosses a parent boundary"
            )));
        }
        if subchunk.parent_group_id != group.parent_group_id {
            return Err(invalid(format!(
                "subchunk {index} parent ID does not echo its covered group"
            )));
        }
        next_domain = end_domain;
        next_q = next_q
            .checked_add(subchunk.row_count)
            .ok_or_else(|| invalid("subchunk q range overflows usize"))?;
    }
    finish_resident_validation(deadline, "resident subchunk partition")?;
    let domain_count = groups
        .last()
        .map_or(0, |group| group.first_domain + group.child_cardinality);
    if next_domain != domain_count || next_q != total_rows {
        return Err(invalid(
            "subchunks must cover every domain and full-R row exactly once",
        ));
    }
    Ok(())
}

fn hash_schedule_with_deadline(
    request: &GpuBabBoundWaveRequest,
    phase: &GpuBabBoundPhaseDescriptor,
    deadline: Option<ResidentValidationDeadline>,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(b"ny.gpu-bab-bound.schedule.v2\0");
    hash.update(phase.authority.graph_identity_sha256);
    hash.update(phase.authority.static_phase_identity_sha256);
    hash.update(phase.authority.input_identity_sha256);
    hash.update(phase.authority.root_bounds_identity_sha256);
    hash.update(phase.authority.relaxation_identity_sha256);
    hash.update(phase.authority.objective_set_identity_sha256);
    hash_u64(&mut hash, phase.authority.total_objectives as u64);
    hash_u64(&mut hash, phase.authority.static_graph_payload_bytes as u64);
    hash_u64(&mut hash, phase.authority.static_phase_payload_bytes as u64);
    hash_u64(&mut hash, request.parent_groups.len() as u64);
    for (index, group) in request.parent_groups.iter().enumerate() {
        poll_resident_validation(deadline, index, "resident schedule parent groups")?;
        hash_u64(&mut hash, group.parent_group_id);
        hash.update(group.parent_identity_sha256);
        hash_u64(&mut hash, group.first_domain as u64);
        hash_u64(&mut hash, group.child_cardinality as u64);
    }
    finish_resident_validation(deadline, "resident schedule parent groups")?;
    hash_u64(&mut hash, request.domains.len() as u64);
    for (index, domain) in request.domains.iter().enumerate() {
        poll_resident_validation(deadline, index, "resident schedule domains")?;
        hash_u64(&mut hash, domain.parent_group_id);
        hash_u64(&mut hash, domain.child_ordinal as u64);
        hash_u64(&mut hash, domain.child_cardinality as u64);
        hash_u64(&mut hash, domain.domain_slot);
        // Static validation proved every view valid, so this recomputation is
        // infallible for the unchanged owned request.
        hash.update(hash_domain_identity_with_deadline(
            domain,
            &request.domain_arena,
            index,
            deadline,
        )?);
    }
    finish_resident_validation(deadline, "resident schedule domains")?;
    hash_u64(&mut hash, request.objective_indices.len() as u64);
    for (index, objective) in request.objective_indices.iter().enumerate() {
        poll_resident_validation(deadline, index, "resident schedule objectives")?;
        hash.update(objective.to_le_bytes());
    }
    finish_resident_validation(deadline, "resident schedule objectives")?;
    hash_u64(&mut hash, request.subchunks.len() as u64);
    for (index, subchunk) in request.subchunks.iter().enumerate() {
        poll_resident_validation(deadline, index, "resident schedule subchunks")?;
        hash_u64(&mut hash, subchunk.parent_group_id);
        hash_u64(&mut hash, subchunk.first_domain as u64);
        hash_u64(&mut hash, subchunk.domain_count as u64);
        hash_u64(&mut hash, subchunk.first_q as u64);
        hash_u64(&mut hash, subchunk.row_count as u64);
    }
    finish_resident_validation(deadline, "resident schedule subchunks")?;
    hash_u64(&mut hash, request.inherited_lower.len() as u64);
    for (index, (&lower, &upper)) in request
        .inherited_lower
        .iter()
        .zip(request.inherited_upper.iter())
        .enumerate()
    {
        poll_resident_validation(deadline, index, "resident schedule inherited endpoints")?;
        hash.update(lower.to_bits().to_le_bytes());
        hash.update(upper.to_bits().to_le_bytes());
    }
    finish_resident_validation(deadline, "resident schedule inherited endpoints")?;
    hash_u64(&mut hash, request.max_device_bytes as u64);
    Ok(hash.finalize().into())
}

#[cfg(test)]
fn hash_inherited_endpoints(request: &GpuBabBoundWaveRequest) -> [u8; 32] {
    hash_inherited_endpoints_with_deadline(request, None)
        .expect("deadline-free inherited-endpoint hashing is infallible")
}

fn hash_inherited_endpoints_with_deadline(
    request: &GpuBabBoundWaveRequest,
    deadline: Option<ResidentValidationDeadline>,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(b"ny.gpu-bab-bound.inherited-endpoints.v1\0");
    hash_u64(&mut hash, request.inherited_lower.len() as u64);
    for (index, (&lower, &upper)) in request
        .inherited_lower
        .iter()
        .zip(request.inherited_upper.iter())
        .enumerate()
    {
        poll_resident_validation(deadline, index, "resident inherited-endpoint identity")?;
        hash.update(lower.to_bits().to_le_bytes());
        hash.update(upper.to_bits().to_le_bytes());
    }
    finish_resident_validation(deadline, "resident inherited-endpoint identity")?;
    Ok(hash.finalize().into())
}

fn hash_u64(hash: &mut Sha256, value: u64) {
    hash.update(value.to_le_bytes());
}

/// Exact phase/session echo used by raw open and close receipts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBabBoundPhaseTranscript {
    pub backend: GpuBabBoundBackendIssuerIdentity,
    pub graph_identity_sha256: [u8; 32],
    pub static_phase_identity_sha256: [u8; 32],
    pub input_identity_sha256: [u8; 32],
    pub root_bounds_identity_sha256: [u8; 32],
    pub relaxation_identity_sha256: [u8; 32],
    pub objective_set_identity_sha256: [u8; 32],
    pub total_objectives: usize,
    pub static_graph_payload_bytes: usize,
    pub static_phase_payload_bytes: usize,
    pub deadline: Instant,
    pub max_device_bytes: usize,
}

impl GpuBabBoundPhaseTranscript {
    fn expected(
        backend: GpuBabBoundBackendIssuerIdentity,
        phase: &GpuBabBoundPhaseDescriptor,
    ) -> Self {
        Self {
            backend,
            graph_identity_sha256: phase.authority.graph_identity_sha256,
            static_phase_identity_sha256: phase.authority.static_phase_identity_sha256,
            input_identity_sha256: phase.authority.input_identity_sha256,
            root_bounds_identity_sha256: phase.authority.root_bounds_identity_sha256,
            relaxation_identity_sha256: phase.authority.relaxation_identity_sha256,
            objective_set_identity_sha256: phase.authority.objective_set_identity_sha256,
            total_objectives: phase.authority.total_objectives,
            static_graph_payload_bytes: phase.authority.static_graph_payload_bytes,
            static_phase_payload_bytes: phase.authority.static_phase_payload_bytes,
            deadline: phase.deadline,
            max_device_bytes: phase.max_device_bytes,
        }
    }
}

/// Exact all-terminal echo for one accepted wave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBabBoundTerminalTranscript {
    pub phase: GpuBabBoundPhaseTranscript,
    pub wave_index: u64,
    pub schedule_identity_sha256: [u8; 32],
    pub inherited_endpoints_sha256: [u8; 32],
    pub deadline: Instant,
    pub max_device_bytes: usize,
}

/// Disjoint accountable numerical-buffer classes at the allocation peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBabBoundMemoryReceipt {
    pub retained_graph_bytes: usize,
    pub retained_phase_bytes: usize,
    pub wave_working_bytes: usize,
    pub queued_upload_bytes: usize,
    pub result_readback_bytes: usize,
    pub peak_device_bytes: usize,
}

impl GpuBabBoundMemoryReceipt {
    fn checked_sum(self) -> Result<usize> {
        self.retained_graph_bytes
            .checked_add(self.retained_phase_bytes)
            .and_then(|value| value.checked_add(self.wave_working_bytes))
            .and_then(|value| value.checked_add(self.queued_upload_bytes))
            .and_then(|value| value.checked_add(self.result_readback_bytes))
            .ok_or_else(|| invalid("memory receipt byte sum overflows usize"))
    }

    fn validate_peak(self, cap: usize) -> Result<()> {
        let accounted = self.checked_sum()?;
        if accounted == 0 || self.peak_device_bytes != accounted {
            return Err(invalid(format!(
                "peak device bytes {} != nonzero checked accountable sum {accounted}",
                self.peak_device_bytes
            )));
        }
        if self.peak_device_bytes > cap {
            return Err(invalid(format!(
                "peak device bytes {} exceed cap {cap}",
                self.peak_device_bytes
            )));
        }
        Ok(())
    }

    fn validate_open(
        self,
        phase: &GpuBabBoundPhaseDescriptor,
        static_transfers: &GpuBabBoundStaticTransferReceipt,
        terminal: bool,
    ) -> Result<()> {
        let accounted = self.checked_sum()?;
        if (!terminal && accounted == 0)
            || self.peak_device_bytes != accounted
            || accounted > phase.max_device_bytes
        {
            return Err(invalid(
                "open peak must exactly equal its bounded accountable allocation sum",
            ));
        }
        if self.wave_working_bytes != 0 || self.result_readback_bytes != 0 {
            return Err(invalid(
                "open memory cannot contain wave-working or result-readback bytes",
            ));
        }
        static_transfers.validate(phase, self, terminal)?;
        Ok(())
    }

    fn retained_residency(self) -> Result<Self> {
        let retained = self
            .retained_graph_bytes
            .checked_add(self.retained_phase_bytes)
            .ok_or_else(|| invalid("retained open residency overflows usize"))?;
        Ok(Self {
            retained_graph_bytes: self.retained_graph_bytes,
            retained_phase_bytes: self.retained_phase_bytes,
            wave_working_bytes: 0,
            queued_upload_bytes: 0,
            result_readback_bytes: 0,
            peak_device_bytes: retained,
        })
    }
}

/// Exact source of one typed static resident payload class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundStaticPayloadSource {
    /// Terminal open stopped before this complete typed payload transferred.
    NotTransferred,
    /// The exact core-hashed payload was uploaded during this accepted open.
    FreshUpload,
    /// A qualified exact-identity resident cache supplied the payload.
    QualifiedCacheHit {
        cache_epoch: u64,
        resident_identity_sha256: [u8; 32],
    },
}

impl GpuBabBoundStaticPayloadSource {
    fn validate(
        self,
        expected_identity: [u8; 32],
        payload_bytes: usize,
        terminal: bool,
        label: &str,
    ) -> Result<(usize, usize)> {
        match self {
            Self::NotTransferred if terminal => Ok((0, 0)),
            Self::NotTransferred => Err(invalid(format!(
                "opened {label} payload cannot be NotTransferred"
            ))),
            Self::FreshUpload => Ok((payload_bytes, payload_bytes)),
            Self::QualifiedCacheHit {
                cache_epoch,
                resident_identity_sha256,
            } if cache_epoch != 0 && resident_identity_sha256 == expected_identity => {
                Ok((payload_bytes, 0))
            }
            Self::QualifiedCacheHit { .. } => Err(invalid(format!(
                "{label} cache hit must echo a nonzero epoch and exact resident identity"
            ))),
        }
    }
}

/// Exact typed static-payload residency and open-transfer equation.
///
/// Payload bytes are core-derived raw typed bytes. Explicit padding accounts
/// for the backend's reviewed alignment/layout without pretending core knows
/// the physical buffer representation. A qualified cache hit must name the
/// exact class identity and charges the full resident payload plus padding,
/// while contributing zero H2D bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBabBoundStaticTransferReceipt {
    pub graph_identity_sha256: [u8; 32],
    pub phase_identity_sha256: [u8; 32],
    pub graph_payload_bytes: usize,
    pub phase_payload_bytes: usize,
    pub graph_padding_bytes: usize,
    pub phase_padding_bytes: usize,
    pub graph_source: GpuBabBoundStaticPayloadSource,
    pub phase_source: GpuBabBoundStaticPayloadSource,
    pub graph_host_to_device_bytes: usize,
    pub phase_host_to_device_bytes: usize,
    pub host_to_device_bytes: usize,
}

impl GpuBabBoundStaticTransferReceipt {
    fn validate(
        self,
        phase: &GpuBabBoundPhaseDescriptor,
        memory: GpuBabBoundMemoryReceipt,
        terminal: bool,
    ) -> Result<()> {
        let authority = phase.authority;
        if self.graph_identity_sha256 != authority.graph_identity_sha256
            || self.phase_identity_sha256 != authority.static_phase_identity_sha256
            || self.graph_payload_bytes != authority.static_graph_payload_bytes
            || self.phase_payload_bytes != authority.static_phase_payload_bytes
        {
            return Err(invalid(
                "static transfer does not exactly echo graph/phase identities and payload totals",
            ));
        }
        let (graph_resident_payload, graph_h2d) = self.graph_source.validate(
            authority.graph_identity_sha256,
            authority.static_graph_payload_bytes,
            terminal,
            "graph",
        )?;
        let (phase_resident_payload, phase_h2d) = self.phase_source.validate(
            authority.static_phase_identity_sha256,
            authority.static_phase_payload_bytes,
            terminal,
            "phase",
        )?;
        let expected_graph_resident = graph_resident_payload
            .checked_add(self.graph_padding_bytes)
            .ok_or_else(|| invalid("graph resident payload plus padding overflows usize"))?;
        let expected_phase_resident = phase_resident_payload
            .checked_add(self.phase_padding_bytes)
            .ok_or_else(|| invalid("phase resident payload plus padding overflows usize"))?;
        let expected_h2d = graph_h2d
            .checked_add(phase_h2d)
            .ok_or_else(|| invalid("static H2D total overflows usize"))?;
        if self.graph_host_to_device_bytes != graph_h2d
            || self.phase_host_to_device_bytes != phase_h2d
            || self.host_to_device_bytes != expected_h2d
            || memory.retained_graph_bytes != expected_graph_resident
            || memory.retained_phase_bytes != expected_phase_resident
            || memory.queued_upload_bytes != expected_h2d
        {
            return Err(invalid(
                "static source/H2D/resident payload+padding equation is not exact",
            ));
        }
        Ok(())
    }
}

/// Raw open allocation receipt supplied by the backend adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBabBoundBackendOpenReceipt {
    pub transcript: GpuBabBoundPhaseTranscript,
    pub authorized_device_bytes: usize,
    pub memory: GpuBabBoundMemoryReceipt,
    pub static_transfers: GpuBabBoundStaticTransferReceipt,
    /// Zero for `Opened`; exact retained graph release for terminal open failure.
    pub released_graph_bytes: usize,
    /// Zero for `Opened`; exact retained phase release for terminal open failure.
    pub released_phase_bytes: usize,
}

/// Typed byte-level transfer accounting for one accepted wave.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuBabBoundTransferReceipt {
    pub activation_operand_bytes: usize,
    pub beta_operand_bytes: usize,
    pub abs_operand_bytes: usize,
    pub box_operand_bytes: usize,
    pub cached_la_operand_bytes: usize,
    pub domain_operand_bytes: usize,
    pub inherited_endpoint_bytes: usize,
    pub objective_index_bytes: usize,
    pub subchunk_descriptor_bytes: usize,
    pub host_to_device_bytes: usize,
    pub result_endpoint_bytes: usize,
    pub result_sidecar_bytes: usize,
    pub domain_outcome_sidecar_bytes: usize,
    pub coefficient_device_to_host_bytes: usize,
    pub device_to_host_bytes: usize,
    pub readbacks: usize,
    pub synchronizations: usize,
}

impl GpuBabBoundTransferReceipt {
    fn validate_equations(self, expected: ValidatedWaveShape, completed: bool) -> Result<()> {
        let typed_operands = self
            .activation_operand_bytes
            .checked_add(self.beta_operand_bytes)
            .and_then(|value| value.checked_add(self.abs_operand_bytes))
            .and_then(|value| value.checked_add(self.box_operand_bytes))
            .and_then(|value| value.checked_add(self.cached_la_operand_bytes))
            .ok_or_else(|| invalid("typed operand transfer sum overflows usize"))?;
        if self.domain_operand_bytes != typed_operands {
            return Err(invalid(
                "domain operand H2D does not equal activation/beta/abs/box/cached-lA components",
            ));
        }
        let h2d = self
            .domain_operand_bytes
            .checked_add(self.inherited_endpoint_bytes)
            .and_then(|value| value.checked_add(self.objective_index_bytes))
            .and_then(|value| value.checked_add(self.subchunk_descriptor_bytes))
            .ok_or_else(|| invalid("typed H2D receipt sum overflows usize"))?;
        if self.host_to_device_bytes != h2d {
            return Err(invalid("H2D total does not equal its typed components"));
        }
        let d2h = self
            .result_endpoint_bytes
            .checked_add(self.result_sidecar_bytes)
            .and_then(|value| value.checked_add(self.domain_outcome_sidecar_bytes))
            .and_then(|value| value.checked_add(self.coefficient_device_to_host_bytes))
            .ok_or_else(|| invalid("typed D2H receipt sum overflows usize"))?;
        if self.device_to_host_bytes != d2h {
            return Err(invalid("D2H total does not equal its typed components"));
        }
        if self.coefficient_device_to_host_bytes != 0 {
            return Err(invalid("coefficient D2H must be zero"));
        }
        let returned_rows = expected.returned_rows;
        let expected_result_endpoints = returned_rows
            .checked_mul(ENDPOINT_BYTES_PER_ROW)
            .ok_or_else(|| invalid("result endpoint bytes overflow usize"))?;
        let expected_result_sidecars = returned_rows
            .checked_mul(RESULT_SIDECAR_BYTES_PER_ROW)
            .ok_or_else(|| invalid("result sidecar bytes overflow usize"))?;
        let expected_domain_outcomes =
            expected
                .domains
                .checked_mul(DOMAIN_OUTCOME_SIDECAR_BYTES)
                .ok_or_else(|| invalid("domain outcome sidecar bytes overflow usize"))?;
        if completed {
            if self.activation_operand_bytes != expected.activation_operand_bytes
                || self.beta_operand_bytes != expected.beta_operand_bytes
                || self.abs_operand_bytes != expected.abs_operand_bytes
                || self.box_operand_bytes != expected.box_operand_bytes
                || self.cached_la_operand_bytes != expected.cached_la_operand_bytes
                || self.domain_operand_bytes != expected.domain_operand_bytes
                || self.inherited_endpoint_bytes != expected.inherited_endpoint_bytes
                || self.objective_index_bytes != expected.objective_index_bytes
                || self.subchunk_descriptor_bytes != expected.subchunk_descriptor_bytes
                || self.result_endpoint_bytes != expected_result_endpoints
                || self.result_sidecar_bytes != expected_result_sidecars
                || self.domain_outcome_sidecar_bytes != expected_domain_outcomes
                || self.readbacks != 1
                || self.synchronizations != 1
            {
                return Err(invalid(
                    "completed transfer receipt is not the exact typed H2D/D2H equation with one readback/synchronization",
                ));
            }
        } else {
            if self.activation_operand_bytes > expected.activation_operand_bytes
                || self.beta_operand_bytes > expected.beta_operand_bytes
                || self.abs_operand_bytes > expected.abs_operand_bytes
                || self.box_operand_bytes > expected.box_operand_bytes
                || self.cached_la_operand_bytes > expected.cached_la_operand_bytes
                || self.domain_operand_bytes > expected.domain_operand_bytes
                || self.inherited_endpoint_bytes > expected.inherited_endpoint_bytes
                || self.objective_index_bytes > expected.objective_index_bytes
                || self.subchunk_descriptor_bytes > expected.subchunk_descriptor_bytes
                || self.result_endpoint_bytes > expected_result_endpoints
                || self.result_sidecar_bytes > expected_result_sidecars
                || self.domain_outcome_sidecar_bytes > expected_domain_outcomes
                || self.readbacks > 1
                || self.synchronizations > 1
                || self.readbacks != usize::from(self.device_to_host_bytes > 0)
            {
                return Err(invalid(
                    "failed transfer receipt exceeds typed work or has inconsistent readback/synchronization counts",
                ));
            }
            for (actual, full, label) in [
                (
                    self.activation_operand_bytes,
                    expected.activation_operand_bytes,
                    "activation",
                ),
                (self.beta_operand_bytes, expected.beta_operand_bytes, "beta"),
                (self.abs_operand_bytes, expected.abs_operand_bytes, "abs"),
                (self.box_operand_bytes, expected.box_operand_bytes, "box"),
                (
                    self.cached_la_operand_bytes,
                    expected.cached_la_operand_bytes,
                    "cached-lA",
                ),
                (
                    self.inherited_endpoint_bytes,
                    expected.inherited_endpoint_bytes,
                    "inherited endpoint",
                ),
                (
                    self.objective_index_bytes,
                    expected.objective_index_bytes,
                    "objective index",
                ),
                (
                    self.subchunk_descriptor_bytes,
                    expected.subchunk_descriptor_bytes,
                    "subchunk descriptor",
                ),
                (
                    self.result_endpoint_bytes,
                    expected_result_endpoints,
                    "result endpoint",
                ),
                (
                    self.result_sidecar_bytes,
                    expected_result_sidecars,
                    "result sidecar",
                ),
                (
                    self.domain_outcome_sidecar_bytes,
                    expected_domain_outcomes,
                    "domain outcome sidecar",
                ),
            ] {
                if actual != 0 && actual != full {
                    return Err(invalid(format!(
                        "failed {label} transfer must contain zero or one complete typed buffer"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Raw backend receipt present on every postaccept wave disposition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBabBoundBackendWaveReceipt {
    pub transcript: GpuBabBoundTerminalTranscript,
    pub requested_parent_groups: usize,
    pub completed_parent_groups: usize,
    pub requested_domains: usize,
    pub completed_domains: usize,
    pub bounded_domains: usize,
    /// Must be zero in this slice; no caller-supplied box can authorize prune.
    pub pruned_domains: usize,
    pub objective_rows: usize,
    pub requested_rows: usize,
    pub completed_rows: usize,
    pub returned_rows: usize,
    pub requested_subchunks: usize,
    pub completed_subchunks: usize,
    pub authorized_device_bytes: usize,
    pub memory: GpuBabBoundMemoryReceipt,
    pub transfers: GpuBabBoundTransferReceipt,
    pub dispatches: usize,
    pub submits: usize,
    pub waves: usize,
    pub tightened_rows: usize,
}

/// Untrusted row returned by a backend adapter.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuBabBoundBackendRow {
    pub parent_group_id: u64,
    pub child_ordinal: usize,
    pub child_cardinality: usize,
    pub domain_slot: u64,
    pub domain_identity_sha256: [u8; 32],
    pub objective_index: u32,
    pub q: u32,
    pub lower: f32,
    pub upper: f32,
    pub status: u32,
    pub taint: u32,
}

/// Exact per-domain terminal class returned for a completed wave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundBackendDomainOutcomeKind {
    Bounded,
}

/// Untrusted association echo for one completed domain outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuBabBoundBackendDomainOutcome {
    pub parent_group_id: u64,
    pub child_ordinal: usize,
    pub child_cardinality: usize,
    pub domain_slot: u64,
    pub domain_identity_sha256: [u8; 32],
    pub kind: GpuBabBoundBackendDomainOutcomeKind,
}

/// Raw failure kind from the trusted backend adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundBackendFailureKind {
    Device,
    Allocation,
    Numerical,
    Association,
    AuthorityLost,
}

/// Raw postaccept disposition. Core treats even `IllegalCleanDecline` as a
/// terminal accepted protocol failure; it never restores fallback authority.
pub enum GpuBabBoundBackendWaveDisposition {
    Completed {
        domain_outcomes: Vec<GpuBabBoundBackendDomainOutcome>,
        rows: Vec<GpuBabBoundBackendRow>,
        receipt: GpuBabBoundBackendWaveReceipt,
    },
    AcceptedFailure {
        kind: GpuBabBoundBackendFailureKind,
        detail: String,
        receipt: GpuBabBoundBackendWaveReceipt,
    },
    DeadlineExpired {
        detail: String,
        receipt: GpuBabBoundBackendWaveReceipt,
    },
    IllegalCleanDecline {
        reason: GpuBabBoundWaveDecline,
        receipt: GpuBabBoundBackendWaveReceipt,
    },
}

/// Raw, allocation-free preaccept decision from a backend session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundBackendPrepareDisposition {
    CleanDecline(GpuBabBoundWaveDecline),
    Accepted,
}

/// Read-only, core-created context for raw preaccept capacity checking.
pub struct GpuBabBoundPreparedWave<'a> {
    request: &'a GpuBabBoundWaveRequest,
    schedule_identity_sha256: [u8; 32],
    inherited_endpoints_sha256: [u8; 32],
}

impl GpuBabBoundPreparedWave<'_> {
    #[must_use]
    pub fn request(&self) -> &GpuBabBoundWaveRequest {
        self.request
    }

    #[must_use]
    pub fn schedule_identity_sha256(&self) -> &[u8; 32] {
        &self.schedule_identity_sha256
    }

    #[must_use]
    pub fn inherited_endpoints_sha256(&self) -> &[u8; 32] {
        &self.inherited_endpoints_sha256
    }
}

/// Read-only, core-created context for exactly one accepted raw execution.
pub struct GpuBabBoundAcceptedWave<'a> {
    request: &'a GpuBabBoundWaveRequest,
    transcript: GpuBabBoundTerminalTranscript,
}

/// Read-only, core-created authority for the accepted phase-open transition.
///
/// The backend must perform no device allocation before core has validated and
/// permanently burned the prepared issuer/generation/nonce and created this
/// context.
pub struct GpuBabBoundAcceptedOpen<'a> {
    descriptor: &'a GpuBabBoundPhaseDescriptor,
    transcript: GpuBabBoundPhaseTranscript,
}

impl GpuBabBoundAcceptedOpen<'_> {
    #[must_use]
    pub fn descriptor(&self) -> &GpuBabBoundPhaseDescriptor {
        self.descriptor
    }

    #[must_use]
    pub fn transcript(&self) -> GpuBabBoundPhaseTranscript {
        self.transcript
    }
}

impl GpuBabBoundAcceptedWave<'_> {
    #[must_use]
    pub fn request(&self) -> &GpuBabBoundWaveRequest {
        self.request
    }

    /// Exact terminal transcript the raw backend must echo in every outcome.
    #[must_use]
    pub fn transcript(&self) -> GpuBabBoundTerminalTranscript {
        self.transcript
    }
}

/// Raw close receipt, including exact release of retained residency charges.
/// A qualified cache may keep physical buffers alive; `released_*` means this
/// phase no longer owns or is charged for that residency lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBabBoundBackendCloseReceipt {
    pub transcript: GpuBabBoundPhaseTranscript,
    pub released_graph_bytes: usize,
    pub released_phase_bytes: usize,
    pub released_resident_device_bytes: usize,
    /// Exact number of physically Resident logical slots whose lease ended.
    pub released_resident_slots: usize,
    /// Exact number of host-snapshotted RefreshOnly slots whose lease ended.
    pub released_refresh_only_slots: usize,
    /// Redundant total, required to equal the checked sum of both classes.
    pub released_resident_logical_slots: usize,
}

/// Raw backend close outcome. Closing never restores wave fallback authority.
pub enum GpuBabBoundBackendCloseDisposition {
    Closed(GpuBabBoundBackendCloseReceipt),
    AcceptedFailure {
        detail: String,
        receipt: GpuBabBoundBackendCloseReceipt,
    },
}

/// Private-field invocation minted only by [`GpuBabBoundPhaseLease::open`].
///
/// It carries no public constructor and is not `Clone`; a numerical-TCB
/// implementation receives it only after core descriptor validation and both
/// soundness gates. Production WGPU code must additionally revalidate its
/// device-local qualification before every raw transition.
///
/// ```compile_fail
/// use ny_core::{GpuBabBoundPhaseDescriptor, GpuBabBoundTcbInvocation};
/// fn forge(descriptor: &GpuBabBoundPhaseDescriptor) -> GpuBabBoundTcbInvocation<'_> {
///     GpuBabBoundTcbInvocation { descriptor }
/// }
/// ```
pub struct GpuBabBoundTcbInvocation<'a> {
    descriptor: &'a GpuBabBoundPhaseDescriptor,
}

impl GpuBabBoundTcbInvocation<'_> {
    #[must_use]
    pub fn descriptor(&self) -> &GpuBabBoundPhaseDescriptor {
        self.descriptor
    }
}

/// Private-field pre-descriptor invocation minted only by core certification.
///
/// The payload borrow cannot escape into the resulting certificate: a future
/// phase owner may end this borrow and consume the owned static payload without
/// cloning its request-sized arenas.
pub struct GpuBabBoundScheduleTcbInvocation<'request, 'payload> {
    request: &'request GpuBabBoundStaticScheduleRequest<'payload>,
}

impl<'payload> GpuBabBoundScheduleTcbInvocation<'_, 'payload> {
    #[must_use]
    pub fn request(&self) -> &GpuBabBoundStaticScheduleRequest<'payload> {
        self.request
    }
}

/// Immutable reviewed schema/kernel bundle for schedule qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBabBoundBackendScheduleIdentity {
    pub schema_bundle_version: u32,
    pub provider_abi_sha256: [u8; 32],
    pub receipt_abi_sha256: [u8; 32],
    pub kernel_sha256: [u8; 32],
    pub topology_schema_sha256: [u8; 32],
    pub selfcheck_schema_sha256: [u8; 32],
    pub transcript_schema_sha256: [u8; 32],
}

impl GpuBabBoundBackendScheduleIdentity {
    fn validate(self) -> Result<()> {
        if self.schema_bundle_version == 0
            || [
                self.provider_abi_sha256,
                self.receipt_abi_sha256,
                self.kernel_sha256,
                self.topology_schema_sha256,
                self.selfcheck_schema_sha256,
                self.transcript_schema_sha256,
            ]
            .into_iter()
            .any(is_zero_identity)
        {
            return Err(invalid(
                "schedule schema bundle version/identities must all be nonzero",
            ));
        }
        Ok(())
    }
}

/// Untrusted backend proposal for one exact pre-descriptor static schedule.
///
/// Every field is an audit echo. Core checks it against the stable
/// registration, borrowed request, finite policy limits, and the complete
/// reviewed WGPU schema bundle before minting an opaque certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBabBoundBackendScheduleEvidence {
    pub backend_issuer_sha256: [u8; 32],
    pub registration_epoch: u64,
    pub static_payload_identity_sha256: [u8; 32],
    pub topology_schema_version: u32,
    pub schedule_identity: GpuBabBoundBackendScheduleIdentity,
    pub requested_max_device_bytes: usize,
    pub phase_policy: GpuBabBoundPhasePolicy,
    pub dispatches_per_subchunk: usize,
}

/// Allocation-free backend answer to one pre-descriptor schedule request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundBackendScheduleDisposition {
    /// No schedule authority was created; legacy fallback remains permitted.
    CleanDecline(GpuBabBoundPhaseDecline),
    /// Raw, structurally untrusted evidence for core validation.
    Certified(GpuBabBoundBackendScheduleEvidence),
}

/// Reviewed numerical trusted-computing-base phase factory.
///
/// Implementations are not ordinary extension points: adding one is a source-
/// reviewed expansion of verdict authority and must be allowlisted by repository
/// policy. The marker is safe because `ny-core` globally forbids unsafe code;
/// it can act only when core supplies a private-field invocation. Association
/// validation cannot prove that a TCB implementation did not fabricate finite
/// endpoints, so every implementation is security-sensitive.
pub trait GpuBabBoundNumericalTcb: Send + Sync {
    /// Stable registration for this qualified backend/device epoch.
    ///
    /// This reference must identify the same registration on every call for
    /// the provider's lifetime. Returning a fresh/swapped registration is a
    /// numerical-TCB contract violation and an explicit authority reset, not a
    /// supported recovery path.
    fn registration(&self) -> &GpuBabBoundBackendRegistration;

    /// Propose a dispatch schedule for exact validated static payload bytes.
    ///
    /// The default keeps the new surface dark for every existing numerical
    /// TCB. Implementations must remain finite, allocation-free, nonblocking,
    /// and resource-inaccessible; core independently validates every echo.
    fn certify_static_schedule(
        &self,
        _invocation: &GpuBabBoundScheduleTcbInvocation<'_, '_>,
    ) -> GpuBabBoundBackendScheduleDisposition {
        GpuBabBoundBackendScheduleDisposition::CleanDecline(GpuBabBoundPhaseDecline::Unsupported)
    }

    /// Read-only, allocation-free policy query for this exact phase.
    fn phase_policy(
        &self,
        invocation: &GpuBabBoundTcbInvocation<'_>,
    ) -> Option<GpuBabBoundPhasePolicy>;

    /// Allocation-free preparation for this exact core-authorized phase.
    fn prepare_phase<'a>(
        &'a self,
        invocation: &GpuBabBoundTcbInvocation<'_>,
    ) -> GpuBabBoundBackendOpenPreparation<'a>;
}

/// Backend-facing raw session adapter.
///
/// Implementing this safe trait is an explicit expansion of the numerical TCB:
/// association/receipt validation cannot prove that fabricated finite
/// endpoints are sound. Repository source policy must allowlist every
/// implementation. The crate retains `forbid(unsafe_code)`; reviewed WGPU
/// wiring must additionally recheck its own default-closed kernel/selfcheck
/// qualification on every raw transition. Individual public raw values remain
/// non-authoritative for ordinary consumers: core-created accepted-open/wave
/// capabilities own lifecycle and are the only route to constructing validated
/// results. Implementing and exposing either TCB trait is itself an explicit
/// source-reviewed expansion of numerical verdict authority.
pub trait GpuBabBoundBackendSession: Send {
    /// Cross the accepted-open boundary and allocate retained device state.
    ///
    /// The session object returned by raw preparation is dormant: creating it
    /// may not allocate/upload/dispatch accelerator resources. Only this call,
    /// made after core issuer reservation, may begin retained device work.
    fn open_accepted(&mut self, accepted: &GpuBabBoundAcceptedOpen<'_>) -> GpuBabBoundBackendOpen;

    /// Pure preflight: no allocation, wait, upload, dispatch, or generation use.
    fn prepare_wave(
        &mut self,
        prepared: &GpuBabBoundPreparedWave<'_>,
    ) -> GpuBabBoundBackendPrepareDisposition;

    /// Optional, stable v2 retained-domain limits for this exact open session.
    /// This query is a pure, O(1), resource-inaccessible TCB operation: it may
    /// not allocate, wait, inspect/mutate accelerator resources, or change any
    /// generation/slot state. Core may call it before and after raw boundaries
    /// solely to confirm the immutable policy echo. The default keeps the
    /// promotion-grade retained-domain surface dark.
    fn resident_domain_policy(&self) -> Option<GpuBabBoundResidentDomainPolicy> {
        None
    }

    /// Pure v2 preflight: no allocation, release, upload, copy, dispatch,
    /// generation use, or logical-slot mutation. The default declines before
    /// acceptance so existing v1 implementations retain their behavior. An
    /// opting-in numerical TCB must map every history topology ID to the exact
    /// versioned phase plan and unique execution-order node, prove it names an
    /// in-range ReLU/Sign preactivation with the exact flattened width, bind
    /// phase bits to beta sign/order and all six operand slices for the same
    /// logical domain/phase, and reject any mismatch. Core validates only the
    /// structural grammar/identity/resource contract. The provider must
    /// preserve all seven retained inputs immutably and separate from
    /// working/output buffers until receipted release; core deliberately
    /// provides no topology-based pruning authority here.
    fn prepare_resident_wave(
        &mut self,
        _prepared: &GpuBabBoundPreparedResidentWave<'_>,
    ) -> GpuBabBoundBackendResidentPrepareDisposition {
        GpuBabBoundBackendResidentPrepareDisposition::CleanDecline(
            GpuBabBoundResidentWaveDecline::Unsupported,
        )
    }

    /// Mandatory allocation-free release/eviction preflight after this session
    /// has advertised a retained-domain policy. Only `TemporarilyUnavailable`
    /// is a clean decline; every other decline strands core-owned residency and
    /// is treated as terminal authority loss.
    fn prepare_resident_maintenance(
        &mut self,
        _prepared: &GpuBabBoundPreparedResidentMaintenance<'_>,
    ) -> GpuBabBoundBackendResidentMaintenancePrepareDisposition {
        GpuBabBoundBackendResidentMaintenancePrepareDisposition::AuthorityLost
    }

    /// Execute an already accepted v1 wave and return exactly one raw terminal.
    /// V1 has no resident-transition transcript: it must not create, release,
    /// evict, copy from, or mutate any v2 resident slot. A healthy v1
    /// `Completed` leaves the v2 ledger unchanged; every v1 accepted terminal
    /// failure causes core to poison all v2 authority.
    fn execute_accepted(
        &mut self,
        accepted: &GpuBabBoundAcceptedWave<'_>,
    ) -> GpuBabBoundBackendWaveDisposition;

    /// Execute one already accepted v2 retained-domain transaction.
    ///
    /// This default is unreachable while the default policy/preflight remain
    /// dark. A provider opting into v2 must override it; core catches the panic
    /// and poisons the phase if the contract is violated. Execution must use
    /// exactly the topology/history/beta/six-family association proven during
    /// preflight and keep the seven immutable retained inputs distinct from
    /// all mutable work/output storage.
    fn execute_accepted_resident(
        &mut self,
        _accepted: &GpuBabBoundAcceptedResidentWave<'_>,
    ) -> GpuBabBoundBackendResidentWaveDisposition {
        panic!("retained-domain v2 execution was not implemented")
    }

    /// Execute accepted zero-destination release/eviction maintenance.
    fn execute_accepted_resident_maintenance(
        &mut self,
        _accepted: &GpuBabBoundAcceptedResidentMaintenance<'_>,
    ) -> GpuBabBoundBackendResidentMaintenanceDisposition {
        panic!("retained-domain maintenance execution was not implemented")
    }

    /// Release retained state exactly once while core retains ownership of the
    /// session box. Core catches this call and session destruction separately,
    /// preventing a core-induced close-method unwind from also running a
    /// separately panicking session destructor during that unwind. This cannot
    /// contain a TCB-internal double panic, panicking panic hook, abort, hang,
    /// `panic=abort`, or foreign unwind; those remain fail-stop availability
    /// obligations of the reviewed numerical TCB. A reviewed implementation
    /// must make repeated calls invalid; core itself calls this method at most
    /// once.
    fn close(&mut self) -> GpuBabBoundBackendCloseDisposition;
}

/// Raw backend open failure category.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundBackendOpenFailureKind {
    Allocation,
    Device,
    AuthorityLost,
}

/// Raw backend disposition after core accepted a prepared phase open.
pub enum GpuBabBoundBackendOpen {
    Opened {
        receipt: GpuBabBoundBackendOpenReceipt,
    },
    AcceptedFailure {
        kind: GpuBabBoundBackendOpenFailureKind,
        detail: String,
        receipt: GpuBabBoundBackendOpenReceipt,
    },
    DeadlineExpired {
        detail: String,
        receipt: GpuBabBoundBackendOpenReceipt,
    },
}

/// Allocation-free raw phase preparation. `Prepared` is not acceptance: core
/// first reserves and burns a core-derived ticket, then calls `open_accepted`.
/// Preparation cannot select or swap registration authority; core separately
/// borrows the provider's stable registration before this call.
pub enum GpuBabBoundBackendOpenPreparation<'a> {
    CleanDecline(GpuBabBoundPhaseDecline),
    Prepared {
        session: Box<dyn GpuBabBoundBackendSession + 'a>,
    },
}

/// Clean, zero-authority reason a backend did not open a phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundPhaseDecline {
    Unsupported,
    InsufficientCapacity,
    BelowMinimumUsefulWidth,
}

/// Clean, zero-work reason a live phase did not accept one wave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundWaveDecline {
    InsufficientCapacity,
    BelowMinimumUsefulWidth,
    TemporarilyUnavailable,
}

/// Core classification of a postaccept/open protocol failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundTerminalFailureKind {
    Backend(GpuBabBoundBackendFailureKind),
    OpenBackend(GpuBabBoundBackendOpenFailureKind),
    ContractViolation,
    CapabilityAbandoned,
    WaveSequenceExhausted,
}

/// Preclaim provider failure. It carries no receipt because the TCB contract
/// requires all provider discovery/policy/preparation calls to be allocation-
/// free. It is deliberately nonfallback even when caused by an unwind or a
/// deadline crossed inside one of those calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundProviderFailureKind {
    SoundnessGatePanicked,
    AccessorPanicked,
    RegistrationPanicked,
    SchedulePanicked,
    InvalidScheduleEvidence,
    RegistrationChanged,
    PolicyPanicked,
    InvalidPolicy,
    PreparationPanicked,
    DormantSessionDropPanicked,
    RegistrationUnavailable,
    DeadlineExpired,
}

/// Typed, nonfallback terminal from provider discovery before issuer claim.
pub struct GpuBabBoundProviderFailure {
    kind: GpuBabBoundProviderFailureKind,
    detail: String,
}

/// Core-sealed backend schedule for one exact static payload.
///
/// This value is intentionally non-`Clone` and owns no payload borrow. It is
/// also deliberately not phase authority: a future conversion must separately
/// validate finalized-root host custody before consuming the payload into a
/// graph plan or descriptor. In particular, `requested_max_device_bytes` is a
/// device-local cap and says nothing about adapter-host admission. Future
/// consumption must also re-observe the exact provider, pointer-identical
/// registration issuer/epoch with unpoisoned availability, and live device
/// qualification. This detached identity never revives registration authority.
#[derive(Debug)]
pub struct GpuBabBoundScheduleCertificate {
    evidence: GpuBabBoundBackendScheduleEvidence,
    certificate_identity_sha256: [u8; 32],
    deadline: Instant,
}

impl GpuBabBoundScheduleCertificate {
    /// Exact core-validated backend evidence sealed by this certificate.
    #[must_use]
    pub fn evidence(&self) -> GpuBabBoundBackendScheduleEvidence {
        self.evidence
    }

    #[must_use]
    /// Stable audit digest of fixed evidence only.
    ///
    /// The process-local [`Instant`] deadline is deliberately not encoded in
    /// this digest. Future consumers must separately validate [`Self::deadline`]
    /// and may never treat this identity alone as live authority.
    pub fn certificate_identity_sha256(&self) -> &[u8; 32] {
        &self.certificate_identity_sha256
    }

    /// Absolute local deadline inherited from the validated request.
    #[must_use]
    pub fn deadline(&self) -> Instant {
        self.deadline
    }
}

/// Typed outcome of pre-descriptor backend schedule certification.
#[must_use = "only a clean decline permits untouched legacy fallback"]
pub enum GpuBabBoundScheduleCertification {
    CleanDecline(GpuBabBoundPhaseDecline),
    ProviderFailure(GpuBabBoundProviderFailure),
    Certified(GpuBabBoundScheduleCertificate),
}

impl GpuBabBoundScheduleCertification {
    /// Whether no provider authority/fault was created and legacy may proceed.
    #[must_use]
    pub fn permits_legacy_fallback(&self) -> bool {
        matches!(self, Self::CleanDecline(_))
    }
}

/// Ask a reviewed backend to certify a schedule for exact static payload bytes.
///
/// The request has already been fully validated and identity-bound. Core still
/// rechecks both soundness gates, stable registration identity before and after
/// the TCB call, all fixed evidence, and the absolute deadline. Panics,
/// malformed echoes, and registration drift permanently poison registration;
/// a deadline or merely occupied registration is terminal for this request but
/// does not fault a pure, resource-inaccessible provider.
pub fn certify_gpu_bab_bound_static_schedule(
    backend: &dyn GpuCrownBackward,
    request: &GpuBabBoundStaticScheduleRequest<'_>,
) -> GpuBabBoundScheduleCertification {
    if Instant::now() >= request.deadline {
        return schedule_provider_failure(
            GpuBabBoundProviderFailureKind::DeadlineExpired,
            "static schedule request expired before provider observation",
        );
    }
    if request.logical_static_device_bytes > request.requested_max_device_bytes {
        return GpuBabBoundScheduleCertification::CleanDecline(
            GpuBabBoundPhaseDecline::InsufficientCapacity,
        );
    }
    let gates = catch_tcb_unwind(|| {
        (
            backend.provides_sound_gpu_crown(),
            backend.provides_sound_gpu_bab_bound_phase(),
        )
    });
    let (sound_crown, sound_phase) = match gates {
        Ok(gates) => gates,
        Err(()) => {
            return schedule_provider_failure(
                GpuBabBoundProviderFailureKind::SoundnessGatePanicked,
                "backend soundness gate panicked during static schedule certification",
            );
        }
    };
    if Instant::now() >= request.deadline {
        return schedule_provider_failure(
            GpuBabBoundProviderFailureKind::DeadlineExpired,
            "static schedule request expired during soundness-gate query",
        );
    }
    if !sound_crown || !sound_phase {
        return GpuBabBoundScheduleCertification::CleanDecline(
            GpuBabBoundPhaseDecline::Unsupported,
        );
    }
    let numerical_tcb = match catch_tcb_unwind(|| backend.gpu_bab_bound_numerical_tcb()) {
        Err(()) => {
            return schedule_provider_failure(
                GpuBabBoundProviderFailureKind::AccessorPanicked,
                "backend numerical-TCB accessor panicked during static schedule certification",
            );
        }
        Ok(result) if Instant::now() >= request.deadline => {
            let _ = result;
            return schedule_provider_failure(
                GpuBabBoundProviderFailureKind::DeadlineExpired,
                "static schedule request expired during numerical-TCB accessor query",
            );
        }
        Ok(Some(numerical_tcb)) => numerical_tcb,
        Ok(None) => {
            return GpuBabBoundScheduleCertification::CleanDecline(
                GpuBabBoundPhaseDecline::Unsupported,
            );
        }
    };
    let registration = match catch_tcb_unwind(|| numerical_tcb.registration()) {
        Ok(registration) => registration,
        Err(()) => {
            return schedule_provider_failure(
                GpuBabBoundProviderFailureKind::RegistrationPanicked,
                "backend registration accessor panicked during static schedule certification",
            );
        }
    };
    if Instant::now() >= request.deadline {
        return schedule_provider_failure(
            GpuBabBoundProviderFailureKind::DeadlineExpired,
            "static schedule request expired during stable-registration query",
        );
    }
    let invocation = GpuBabBoundScheduleTcbInvocation { request };
    let disposition = match catch_tcb_unwind(|| numerical_tcb.certify_static_schedule(&invocation))
    {
        Ok(disposition) => disposition,
        Err(()) => {
            registration.poison_registration();
            return schedule_provider_failure(
                GpuBabBoundProviderFailureKind::SchedulePanicked,
                "backend static schedule certification panicked",
            );
        }
    };
    if Instant::now() >= request.deadline {
        return schedule_provider_failure(
            GpuBabBoundProviderFailureKind::DeadlineExpired,
            "static schedule request expired during backend certification",
        );
    }
    let registration_after = match catch_tcb_unwind(|| numerical_tcb.registration()) {
        Ok(registration_after) => registration_after,
        Err(()) => {
            registration.poison_registration();
            return schedule_provider_failure(
                GpuBabBoundProviderFailureKind::RegistrationPanicked,
                "backend registration accessor panicked after static schedule certification",
            );
        }
    };
    if !std::ptr::eq(registration, registration_after) {
        registration_after.poison_registration();
        registration.poison_registration();
        return schedule_provider_failure(
            GpuBabBoundProviderFailureKind::RegistrationChanged,
            "backend registration changed during static schedule certification",
        );
    }
    if Instant::now() >= request.deadline {
        return schedule_provider_failure(
            GpuBabBoundProviderFailureKind::DeadlineExpired,
            "static schedule request expired during registration stability check",
        );
    }
    // Never hold the burn-ledger mutex across an untrusted TCB callback: a
    // reentrant provider must fail closed at this final guard, not deadlock.
    let registration_guard = match registration.available_guard() {
        Ok(guard) => guard,
        Err(error) => {
            return schedule_provider_failure(
                GpuBabBoundProviderFailureKind::RegistrationUnavailable,
                format!("static schedule registration is unavailable: {error}"),
            );
        }
    };
    if Instant::now() >= request.deadline {
        drop(registration_guard);
        return schedule_provider_failure(
            GpuBabBoundProviderFailureKind::DeadlineExpired,
            "static schedule request expired while guarding disposition issuance",
        );
    }

    match disposition {
        GpuBabBoundBackendScheduleDisposition::CleanDecline(reason) => {
            drop(registration_guard);
            GpuBabBoundScheduleCertification::CleanDecline(reason)
        }
        GpuBabBoundBackendScheduleDisposition::Certified(evidence) => {
            if let Err(error) =
                validate_gpu_bab_bound_schedule_evidence(evidence, registration, request)
            {
                drop(registration_guard);
                registration.poison_registration();
                return schedule_provider_failure(
                    GpuBabBoundProviderFailureKind::InvalidScheduleEvidence,
                    format!("backend static schedule evidence is invalid: {error}"),
                );
            }
            let certificate_identity_sha256 =
                match hash_gpu_bab_bound_schedule_certificate(evidence) {
                    Ok(identity) => identity,
                    Err(error) => {
                        drop(registration_guard);
                        registration.poison_registration();
                        return schedule_provider_failure(
                            GpuBabBoundProviderFailureKind::InvalidScheduleEvidence,
                            format!("backend static schedule certificate hash failed: {error}"),
                        );
                    }
                };
            if Instant::now() >= request.deadline {
                drop(registration_guard);
                return schedule_provider_failure(
                    GpuBabBoundProviderFailureKind::DeadlineExpired,
                    "static schedule request expired during core evidence sealing",
                );
            }
            let certificate = GpuBabBoundScheduleCertificate {
                evidence,
                certificate_identity_sha256,
                deadline: request.deadline,
            };
            drop(registration_guard);
            GpuBabBoundScheduleCertification::Certified(certificate)
        }
    }
}

fn schedule_provider_failure(
    kind: GpuBabBoundProviderFailureKind,
    detail: impl Into<String>,
) -> GpuBabBoundScheduleCertification {
    GpuBabBoundScheduleCertification::ProviderFailure(make_provider_failure(kind, detail))
}

fn validate_gpu_bab_bound_schedule_evidence(
    evidence: GpuBabBoundBackendScheduleEvidence,
    registration: &GpuBabBoundBackendRegistration,
    request: &GpuBabBoundStaticScheduleRequest<'_>,
) -> Result<()> {
    if evidence.backend_issuer_sha256 != *registration.backend_issuer_sha256()
        || evidence.registration_epoch != registration.registration_epoch()
    {
        return Err(invalid(
            "schedule evidence does not exactly bind the stable backend registration",
        ));
    }
    if evidence.static_payload_identity_sha256 != request.static_payload_identity_sha256
        || evidence.topology_schema_version != request.topology_schema_version
        || evidence.requested_max_device_bytes != request.requested_max_device_bytes
    {
        return Err(invalid(
            "schedule evidence does not exactly echo the static request identity/schema/device cap",
        ));
    }
    evidence.schedule_identity.validate()?;
    if registration.schedule_identity() != Some(evidence.schedule_identity) {
        return Err(invalid(
            "schedule evidence does not exactly echo the registered schema/kernel bundle",
        ));
    }
    if !evidence.phase_policy.is_valid()
        || evidence.phase_policy.max_device_bytes > GPU_BAB_BOUND_MAX_RESIDENT_DEVICE_BYTES
        || evidence.requested_max_device_bytes > evidence.phase_policy.max_device_bytes
    {
        return Err(invalid(
            "schedule evidence phase policy is malformed or below the requested device cap",
        ));
    }
    if evidence.dispatches_per_subchunk == 0
        || evidence.dispatches_per_subchunk > GPU_BAB_BOUND_MAX_DISPATCHES_PER_WAVE
        || evidence.dispatches_per_subchunk > evidence.phase_policy.maximum_dispatches_per_wave
    {
        return Err(invalid(
            "schedule evidence dispatch count is zero or exceeds finite policy/core limits",
        ));
    }
    Ok(())
}

fn hash_gpu_bab_bound_schedule_certificate(
    evidence: GpuBabBoundBackendScheduleEvidence,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(b"ny.gpu-bab-bound.schedule-certificate.v1\0");
    hash.update(evidence.backend_issuer_sha256);
    hash.update(evidence.registration_epoch.to_le_bytes());
    hash.update(evidence.static_payload_identity_sha256);
    hash.update(evidence.topology_schema_version.to_le_bytes());
    hash.update(
        evidence
            .schedule_identity
            .schema_bundle_version
            .to_le_bytes(),
    );
    hash.update(evidence.schedule_identity.provider_abi_sha256);
    hash.update(evidence.schedule_identity.receipt_abi_sha256);
    hash.update(evidence.schedule_identity.kernel_sha256);
    hash.update(evidence.schedule_identity.topology_schema_sha256);
    hash.update(evidence.schedule_identity.selfcheck_schema_sha256);
    hash.update(evidence.schedule_identity.transcript_schema_sha256);
    for value in [
        evidence.requested_max_device_bytes,
        evidence.phase_policy.max_device_bytes,
        evidence.phase_policy.preferred_domains_per_wave,
        evidence.phase_policy.minimum_domains_per_wave,
        evidence.phase_policy.maximum_domains_per_wave,
        evidence.phase_policy.maximum_objectives,
        evidence.phase_policy.maximum_dispatches_per_wave,
        evidence.phase_policy.maximum_submits_per_wave,
        evidence.dispatches_per_subchunk,
    ] {
        hash_u64(&mut hash, usize_to_u64(value, "certificate field")?);
    }
    Ok(hash.finalize().into())
}

impl GpuBabBoundProviderFailure {
    #[must_use]
    pub fn kind(&self) -> GpuBabBoundProviderFailureKind {
        self.kind
    }

    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

/// Typed terminal from an open attempt that crossed backend acceptance.
pub struct GpuBabBoundPhaseOpenFailure {
    kind: GpuBabBoundTerminalFailureKind,
    detail: String,
    receipt: GpuBabBoundBackendOpenReceipt,
    receipt_validated: bool,
}

impl GpuBabBoundPhaseOpenFailure {
    #[must_use]
    pub fn kind(&self) -> GpuBabBoundTerminalFailureKind {
        self.kind
    }

    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }

    #[must_use]
    pub fn receipt(&self) -> &GpuBabBoundBackendOpenReceipt {
        &self.receipt
    }

    #[must_use]
    pub fn receipt_validated(&self) -> bool {
        self.receipt_validated
    }
}

/// Public phase-open state. Only `CleanDecline` permits legacy fallback.
#[must_use = "phase-open disposition determines fallback and ownership"]
pub enum GpuBabBoundPhaseOpen<'a> {
    InvalidDescriptor(NyError),
    CleanDecline(GpuBabBoundPhaseDecline),
    ProviderFailure(GpuBabBoundProviderFailure),
    Opened(GpuBabBoundPhaseLease<'a>),
    AcceptedFailure(GpuBabBoundPhaseOpenFailure),
    DeadlineExpired(GpuBabBoundPhaseOpenFailure),
}

impl GpuBabBoundPhaseOpen<'_> {
    #[must_use]
    pub fn permits_legacy_fallback(&self) -> bool {
        matches!(self, Self::CleanDecline(_))
    }
}

fn provider_failure<'a>(
    kind: GpuBabBoundProviderFailureKind,
    detail: impl Into<String>,
) -> GpuBabBoundPhaseOpen<'a> {
    GpuBabBoundPhaseOpen::ProviderFailure(make_provider_failure(kind, detail))
}

fn make_provider_failure(
    kind: GpuBabBoundProviderFailureKind,
    detail: impl Into<String>,
) -> GpuBabBoundProviderFailure {
    GpuBabBoundProviderFailure {
        kind,
        detail: detail.into(),
    }
}

fn provider_deadline<'a>(detail: impl Into<String>) -> GpuBabBoundPhaseOpen<'a> {
    provider_failure(GpuBabBoundProviderFailureKind::DeadlineExpired, detail)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LeaseState {
    Open,
    WaveAccepted(u64),
    Poisoned,
    Closed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResidentResourceCertainty {
    /// Healthy synchronized ledger; normal exact close may release authority.
    HealthyKnown,
    /// A named rollback/validated-failure proof restored exact resources after
    /// absorbing poison. Close may validate cleanup but never release issuer
    /// authority or restore fallback.
    PoisonedKnown,
    /// Raw code may have allocated/released/mutated resources without a fully
    /// validated receipt and atomic core transition.
    PoisonedUnknown,
}

fn validate_open_receipt(
    issuer: GpuBabBoundBackendIssuerIdentity,
    receipt: &GpuBabBoundBackendOpenReceipt,
    phase: &GpuBabBoundPhaseDescriptor,
    terminal: bool,
) -> Result<()> {
    issuer.validate()?;
    let expected = GpuBabBoundPhaseTranscript::expected(issuer, phase);
    if receipt.transcript != expected || receipt.authorized_device_bytes != phase.max_device_bytes {
        return Err(invalid(
            "open receipt does not exactly echo backend/session/phase/cap",
        ));
    }
    receipt
        .memory
        .validate_open(phase, &receipt.static_transfers, terminal)?;
    let expected_graph_release = if terminal {
        receipt.memory.retained_graph_bytes
    } else {
        0
    };
    let expected_phase_release = if terminal {
        receipt.memory.retained_phase_bytes
    } else {
        0
    };
    if receipt.released_graph_bytes != expected_graph_release
        || receipt.released_phase_bytes != expected_phase_release
    {
        return Err(invalid(
            "open receipt retained releases do not match opened/terminal ownership",
        ));
    }
    Ok(())
}

fn validate_raw_terminal_open_before_close(
    issuer: GpuBabBoundBackendIssuerIdentity,
    receipt: &GpuBabBoundBackendOpenReceipt,
    phase: &GpuBabBoundPhaseDescriptor,
) -> Result<()> {
    issuer.validate()?;
    if receipt.transcript != GpuBabBoundPhaseTranscript::expected(issuer, phase)
        || receipt.authorized_device_bytes != phase.max_device_bytes
    {
        return Err(invalid(
            "terminal open does not exactly echo phase authority",
        ));
    }
    receipt
        .memory
        .validate_open(phase, &receipt.static_transfers, true)?;
    if receipt.released_graph_bytes != 0 || receipt.released_phase_bytes != 0 {
        return Err(invalid(
            "raw terminal open cannot claim release before consuming session close",
        ));
    }
    Ok(())
}

fn validate_claimed_terminal_open(
    claim: GpuBabBoundTerminalClaim,
    issuer: GpuBabBoundBackendIssuerIdentity,
    receipt: &GpuBabBoundBackendOpenReceipt,
    phase: &GpuBabBoundPhaseDescriptor,
) -> Result<()> {
    if claim.identity != issuer {
        return Err(invalid(
            "terminal claim does not own the exact issuer identity",
        ));
    }
    validate_open_receipt(issuer, receipt, phase, true)
}

struct GpuBabBoundPreparedSessionSlot<'a> {
    session: Option<Box<dyn GpuBabBoundBackendSession + 'a>>,
    registration: &'a GpuBabBoundBackendRegistration,
    issuer: Option<GpuBabBoundBackendIssuerIdentity>,
    open_invoked: bool,
}

impl<'a> GpuBabBoundPreparedSessionSlot<'a> {
    fn new(
        session: Box<dyn GpuBabBoundBackendSession + 'a>,
        registration: &'a GpuBabBoundBackendRegistration,
    ) -> Self {
        Self {
            session: Some(session),
            registration,
            issuer: None,
            open_invoked: false,
        }
    }

    fn arm(&mut self, issuer: GpuBabBoundBackendIssuerIdentity) {
        self.issuer = Some(issuer);
    }

    fn mark_open_invoked(&mut self) {
        self.open_invoked = true;
    }

    fn session_mut(&mut self) -> &mut (dyn GpuBabBoundBackendSession + 'a) {
        self.session
            .as_deref_mut()
            .expect("prepared session slot owns its backend session")
    }

    fn take(&mut self) -> Box<dyn GpuBabBoundBackendSession + 'a> {
        self.session
            .take()
            .expect("prepared session slot is consumed exactly once")
    }
}

impl Drop for GpuBabBoundPreparedSessionSlot<'_> {
    fn drop(&mut self) {
        let Some(session) = self.session.take() else {
            return;
        };
        if let Some(issuer) = self.issuer {
            self.registration.poison(issuer);
        } else {
            self.registration.poison_registration();
        }
        if std::thread::panicking() {
            std::mem::forget(session);
        } else if self.open_invoked {
            let _ = close_and_destroy_poisoned_session(session);
        } else {
            let _ = discard_dormant_session(session);
        }
    }
}

/// Core-owned, non-cloneable authority for one live retained phase.
///
/// The backend registration's core-owned O(1) ledger enforces one live session.
/// Explicit close consumes this value and releases registration authority
/// only after an exact `Closed` receipt. `Drop` performs best-effort cleanup but
/// permanently poisons the issuer because no close terminal reaches the caller.
///
/// ```compile_fail
/// use ny_core::{GpuBabBoundPhaseLease, GpuBabBoundWaveRequest};
/// fn use_after_close(mut lease: GpuBabBoundPhaseLease<'_>, request: GpuBabBoundWaveRequest) {
///     let _ = lease.close();
///     let _ = lease.prepare_wave(request);
/// }
/// ```
pub struct GpuBabBoundPhaseLease<'a> {
    phase: GpuBabBoundPhaseDescriptor,
    policy: GpuBabBoundPhasePolicy,
    transcript: GpuBabBoundPhaseTranscript,
    open_memory: GpuBabBoundMemoryReceipt,
    registration: &'a GpuBabBoundBackendRegistration,
    session: Option<Box<dyn GpuBabBoundBackendSession + 'a>>,
    last_wave_index: u64,
    state: LeaseState,
    resource_certainty: ResidentResourceCertainty,
    issuer_claimed: bool,
    abandoned_terminal: Option<GpuBabBoundWaveFailure>,
    abandoned_resident_terminal: Option<GpuBabBoundResidentWaveFailure>,
    abandoned_resident_maintenance_terminal: Option<GpuBabBoundResidentMaintenanceFailure>,
    resident_domains: GpuBabBoundResidentDomainState,
}

impl<'a> GpuBabBoundPhaseLease<'a> {
    /// Validate, reserve, and open a phase through the backend's raw adapter.
    pub fn open(
        backend: &'a dyn GpuCrownBackward,
        phase: GpuBabBoundPhaseDescriptor,
    ) -> GpuBabBoundPhaseOpen<'a> {
        if let Err(error) = phase.validate() {
            return GpuBabBoundPhaseOpen::InvalidDescriptor(error);
        }
        let soundness_gates = catch_tcb_unwind(|| {
            (
                backend.provides_sound_gpu_crown(),
                backend.provides_sound_gpu_bab_bound_phase(),
            )
        });
        let (sound_crown, sound_phase) = match soundness_gates {
            Ok(gates) => gates,
            Err(()) => {
                return provider_failure(
                    GpuBabBoundProviderFailureKind::SoundnessGatePanicked,
                    "backend soundness gate panicked",
                );
            }
        };
        if Instant::now() >= phase.deadline {
            return provider_deadline("phase expired during backend soundness-gate query");
        }
        if !sound_crown || !sound_phase {
            return GpuBabBoundPhaseOpen::CleanDecline(GpuBabBoundPhaseDecline::Unsupported);
        }
        let invocation = GpuBabBoundTcbInvocation { descriptor: &phase };
        let numerical_tcb = match catch_tcb_unwind(|| backend.gpu_bab_bound_numerical_tcb()) {
            Err(()) => {
                return provider_failure(
                    GpuBabBoundProviderFailureKind::AccessorPanicked,
                    "backend numerical-TCB accessor panicked",
                );
            }
            Ok(result) if Instant::now() >= phase.deadline => {
                let _ = result;
                return provider_deadline("phase expired during numerical-TCB accessor query");
            }
            Ok(result) => match result {
                Some(numerical_tcb) => numerical_tcb,
                None => {
                    return GpuBabBoundPhaseOpen::CleanDecline(
                        GpuBabBoundPhaseDecline::Unsupported,
                    );
                }
            },
        };
        let registration = match catch_tcb_unwind(|| numerical_tcb.registration()) {
            Ok(registration) => registration,
            Err(()) => {
                return provider_failure(
                    GpuBabBoundProviderFailureKind::RegistrationPanicked,
                    "backend stable-registration accessor panicked",
                );
            }
        };
        if Instant::now() >= phase.deadline {
            return provider_deadline("phase expired during stable-registration query");
        }
        let policy_result = catch_tcb_unwind(|| numerical_tcb.phase_policy(&invocation));
        let policy_result = match policy_result {
            Ok(policy) => policy,
            Err(()) => {
                return provider_failure(
                    GpuBabBoundProviderFailureKind::PolicyPanicked,
                    "backend phase-policy query panicked",
                );
            }
        };
        if Instant::now() >= phase.deadline {
            return provider_deadline("phase expired during backend phase-policy query");
        }
        let policy = match policy_result {
            Some(policy) if !policy.is_valid() => {
                registration.poison_registration();
                return provider_failure(
                    GpuBabBoundProviderFailureKind::InvalidPolicy,
                    "backend returned a malformed phase policy",
                );
            }
            Some(policy) if phase.max_device_bytes <= policy.max_device_bytes => policy,
            None | Some(_) => {
                let available_guard = match registration.available_guard() {
                    Ok(guard) => guard,
                    Err(error) => {
                        return provider_failure(
                            GpuBabBoundProviderFailureKind::RegistrationUnavailable,
                            format!("phase policy declined after registration authority was lost: {error}"),
                        );
                    }
                };
                if Instant::now() >= phase.deadline {
                    let disposition = provider_deadline(
                        "phase expired while guarding phase-policy decline issuance",
                    );
                    drop(available_guard);
                    return disposition;
                }
                let decline = GpuBabBoundPhaseOpen::CleanDecline(
                    GpuBabBoundPhaseDecline::InsufficientCapacity,
                );
                drop(available_guard);
                return decline;
            }
        };
        let preparation = catch_tcb_unwind(|| numerical_tcb.prepare_phase(&invocation));
        let preparation = match preparation {
            Ok(preparation) => preparation,
            Err(()) => {
                return provider_failure(
                    GpuBabBoundProviderFailureKind::PreparationPanicked,
                    "backend phase preparation panicked",
                );
            }
        };
        if Instant::now() >= phase.deadline {
            return match preparation {
                GpuBabBoundBackendOpenPreparation::CleanDecline(_) => {
                    provider_deadline("phase expired during backend phase preparation")
                }
                GpuBabBoundBackendOpenPreparation::Prepared { session } => {
                    let mut session = GpuBabBoundPreparedSessionSlot::new(session, registration);
                    // Poison before running an untrusted dormant destructor. A
                    // destructor may reenter `open` through the same stable
                    // registration; it must observe absorbing poison even when
                    // destruction itself returns cleanly.
                    registration.poison_registration();
                    if discard_dormant_session(session.take()) {
                        provider_deadline("phase expired during backend phase preparation")
                    } else {
                        provider_failure(
                            GpuBabBoundProviderFailureKind::DormantSessionDropPanicked,
                            "dormant backend session destructor panicked after late phase preparation",
                        )
                    }
                }
            };
        }
        match preparation {
            GpuBabBoundBackendOpenPreparation::CleanDecline(reason) => {
                let available_guard = match registration.available_guard() {
                    Ok(guard) => guard,
                    Err(error) => {
                        return provider_failure(
                            GpuBabBoundProviderFailureKind::RegistrationUnavailable,
                            format!("phase preparation declined after registration authority was lost: {error}"),
                        );
                    }
                };
                if Instant::now() >= phase.deadline {
                    let disposition = provider_deadline(
                        "phase expired while guarding phase-preparation decline issuance",
                    );
                    drop(available_guard);
                    return disposition;
                }
                let decline = GpuBabBoundPhaseOpen::CleanDecline(reason);
                drop(available_guard);
                decline
            }
            GpuBabBoundBackendOpenPreparation::Prepared { session } => {
                let mut session = GpuBabBoundPreparedSessionSlot::new(session, registration);
                let (issuer, claim_result) = registration.claim(&phase);
                let transcript = GpuBabBoundPhaseTranscript::expected(issuer, &phase);
                if let Err(error) = claim_result {
                    let discard_clean = discard_dormant_session(session.take());
                    if !discard_clean {
                        registration.poison_registration();
                    }
                    let receipt = zero_terminal_open_receipt(transcript);
                    return GpuBabBoundPhaseOpen::AcceptedFailure(GpuBabBoundPhaseOpenFailure {
                        kind: GpuBabBoundTerminalFailureKind::ContractViolation,
                        detail: if discard_clean {
                            error.to_string()
                        } else {
                            format!("{error}; dormant backend session destructor panicked")
                        },
                        receipt,
                        receipt_validated: false,
                    });
                }
                session.arm(issuer);
                if Instant::now() >= phase.deadline {
                    let terminal_claim = registration.terminal_claim(issuer);
                    let discard_clean = discard_dormant_session(session.take());
                    let receipt = zero_terminal_open_receipt(transcript);
                    let receipt_validated = if discard_clean {
                        terminal_claim.is_ok_and(|claim| {
                            validate_claimed_terminal_open(claim, issuer, &receipt, &phase).is_ok()
                        })
                    } else {
                        registration.poison(issuer);
                        false
                    };
                    let failure = GpuBabBoundPhaseOpenFailure {
                        kind: if receipt_validated {
                            GpuBabBoundTerminalFailureKind::OpenBackend(
                                GpuBabBoundBackendOpenFailureKind::AuthorityLost,
                            )
                        } else {
                            GpuBabBoundTerminalFailureKind::ContractViolation
                        },
                        detail: if discard_clean {
                            "phase expired after core issuer claim and before raw open".into()
                        } else {
                            "phase expired after core issuer claim; dormant session destructor panicked"
                                .into()
                        },
                        receipt,
                        receipt_validated,
                    };
                    return if receipt_validated {
                        GpuBabBoundPhaseOpen::DeadlineExpired(failure)
                    } else {
                        GpuBabBoundPhaseOpen::AcceptedFailure(failure)
                    };
                }
                let accepted = GpuBabBoundAcceptedOpen {
                    descriptor: &phase,
                    transcript,
                };
                if let Err(error) = registration.check_live(issuer) {
                    registration.poison(issuer);
                    let discard_clean = discard_dormant_session(session.take());
                    return GpuBabBoundPhaseOpen::AcceptedFailure(GpuBabBoundPhaseOpenFailure {
                        kind: GpuBabBoundTerminalFailureKind::ContractViolation,
                        detail: if discard_clean {
                            format!("live authority was lost before raw open: {error}")
                        } else {
                            format!(
                                    "live authority was lost before raw open: {error}; dormant session destructor panicked"
                                )
                        },
                        receipt: zero_terminal_open_receipt(transcript),
                        receipt_validated: false,
                    });
                }
                if Instant::now() >= phase.deadline {
                    let terminal_claim = registration.terminal_claim(issuer);
                    let discard_clean = discard_dormant_session(session.take());
                    let receipt = zero_terminal_open_receipt(transcript);
                    let receipt_validated = discard_clean
                        && terminal_claim.is_ok_and(|claim| {
                            validate_claimed_terminal_open(claim, issuer, &receipt, &phase).is_ok()
                        });
                    if !discard_clean {
                        registration.poison(issuer);
                    }
                    let failure = GpuBabBoundPhaseOpenFailure {
                        kind: if receipt_validated {
                            GpuBabBoundTerminalFailureKind::OpenBackend(
                                GpuBabBoundBackendOpenFailureKind::AuthorityLost,
                            )
                        } else {
                            GpuBabBoundTerminalFailureKind::ContractViolation
                        },
                        detail: if discard_clean {
                            "phase expired after final live check and before raw open".into()
                        } else {
                            "phase expired before raw open; dormant session destructor panicked"
                                .into()
                        },
                        receipt,
                        receipt_validated,
                    };
                    return if receipt_validated {
                        GpuBabBoundPhaseOpen::DeadlineExpired(failure)
                    } else {
                        GpuBabBoundPhaseOpen::AcceptedFailure(failure)
                    };
                }
                session.mark_open_invoked();
                let raw = catch_tcb_unwind(|| session.session_mut().open_accepted(&accepted));
                let raw = match raw {
                    Ok(raw) => raw,
                    Err(()) => {
                        registration.poison(issuer);
                        let _ = best_effort_close_poisoned(session.take(), transcript, None);
                        return GpuBabBoundPhaseOpen::AcceptedFailure(
                            GpuBabBoundPhaseOpenFailure {
                                kind: GpuBabBoundTerminalFailureKind::ContractViolation,
                                detail: "backend panicked after accepted phase open".into(),
                                receipt: zero_terminal_open_receipt(transcript),
                                receipt_validated: false,
                            },
                        );
                    }
                };
                match raw {
                    GpuBabBoundBackendOpen::Opened { receipt } => {
                        let mut live_guard = match registration.live_guard(issuer) {
                            Ok(guard) => guard,
                            Err(error) => {
                                registration.poison(issuer);
                                let _ = best_effort_close_poisoned(
                                    session.take(),
                                    transcript,
                                    Some(receipt.memory),
                                );
                                return GpuBabBoundPhaseOpen::AcceptedFailure(
                                    GpuBabBoundPhaseOpenFailure {
                                        kind: GpuBabBoundTerminalFailureKind::ContractViolation,
                                        detail: format!(
                                            "live authority was lost during raw open: {error}"
                                        ),
                                        receipt,
                                        receipt_validated: false,
                                    },
                                );
                            }
                        };
                        if let Err(error) = validate_open_receipt(issuer, &receipt, &phase, false) {
                            drop(live_guard);
                            registration.poison(issuer);
                            let close_validated = best_effort_close_poisoned(
                                session.take(),
                                transcript,
                                Some(receipt.memory),
                            );
                            let mut terminal_receipt = receipt;
                            if close_validated {
                                terminal_receipt.released_graph_bytes =
                                    receipt.memory.retained_graph_bytes;
                                terminal_receipt.released_phase_bytes =
                                    receipt.memory.retained_phase_bytes;
                            }
                            return GpuBabBoundPhaseOpen::AcceptedFailure(
                                GpuBabBoundPhaseOpenFailure {
                                    kind: GpuBabBoundTerminalFailureKind::ContractViolation,
                                    detail: error.to_string(),
                                    receipt: terminal_receipt,
                                    receipt_validated: false,
                                },
                            );
                        }
                        if Instant::now() >= phase.deadline {
                            // Preliminary open validation completed while this
                            // exact identity was live. Mint the sole terminal
                            // validation right and atomically absorb authority
                            // before invoking any untrusted cleanup.
                            live_guard.poisoned = true;
                            let terminal_claim = GpuBabBoundTerminalClaim { identity: issuer };
                            drop(live_guard);
                            let close_validated = best_effort_close_poisoned(
                                session.take(),
                                transcript,
                                Some(receipt.memory),
                            );
                            let mut terminal_receipt = receipt;
                            if close_validated {
                                terminal_receipt.released_graph_bytes =
                                    receipt.memory.retained_graph_bytes;
                                terminal_receipt.released_phase_bytes =
                                    receipt.memory.retained_phase_bytes;
                            }
                            let terminal_validated = close_validated
                                && validate_claimed_terminal_open(
                                    terminal_claim,
                                    issuer,
                                    &terminal_receipt,
                                    &phase,
                                )
                                .is_ok();
                            if terminal_validated {
                                return GpuBabBoundPhaseOpen::DeadlineExpired(
                                    GpuBabBoundPhaseOpenFailure {
                                        kind: GpuBabBoundTerminalFailureKind::OpenBackend(
                                            GpuBabBoundBackendOpenFailureKind::AuthorityLost,
                                        ),
                                        detail: "phase expired after accepted open".into(),
                                        receipt: terminal_receipt,
                                        receipt_validated: true,
                                    },
                                );
                            }
                            return GpuBabBoundPhaseOpen::AcceptedFailure(
                                GpuBabBoundPhaseOpenFailure {
                                    kind: GpuBabBoundTerminalFailureKind::ContractViolation,
                                    detail: "phase expired but retained close was not validated"
                                        .into(),
                                    receipt: terminal_receipt,
                                    receipt_validated: false,
                                },
                            );
                        }
                        let open_memory = receipt
                            .memory
                            .retained_residency()
                            .expect("validated open memory has checked retained residency");
                        let lease = Self {
                            transcript,
                            open_memory,
                            phase,
                            policy,
                            registration,
                            last_wave_index: 0,
                            state: LeaseState::Open,
                            resource_certainty: ResidentResourceCertainty::HealthyKnown,
                            issuer_claimed: true,
                            abandoned_terminal: None,
                            abandoned_resident_terminal: None,
                            abandoned_resident_maintenance_terminal: None,
                            resident_domains: GpuBabBoundResidentDomainState::default(),
                            // Keep the protective prepared-session slot armed
                            // until every other lease field is constructed.
                            // This final operand transfers the box directly
                            // into the persistent fail-closed owner.
                            session: Some(session.take()),
                        };
                        let disposition = GpuBabBoundPhaseOpen::Opened(lease);
                        drop(live_guard);
                        disposition
                    }
                    GpuBabBoundBackendOpen::AcceptedFailure {
                        kind,
                        detail,
                        receipt,
                    } => make_open_terminal_claimed(
                        registration,
                        issuer,
                        kind,
                        detail,
                        receipt,
                        &phase,
                        false,
                        session,
                    ),
                    GpuBabBoundBackendOpen::DeadlineExpired { detail, receipt } => {
                        make_open_terminal_claimed(
                            registration,
                            issuer,
                            GpuBabBoundBackendOpenFailureKind::AuthorityLost,
                            detail,
                            receipt,
                            &phase,
                            true,
                            session,
                        )
                    }
                }
            }
        }
    }

    /// Validate and preflight one request, issuing a consuming capability only
    /// after the raw backend has crossed its typed acceptance boundary.
    pub fn prepare_wave<'lease>(
        &'lease mut self,
        request: GpuBabBoundWaveRequest,
    ) -> GpuBabBoundWavePreparation<'lease, 'a> {
        if self.state != LeaseState::Open
            || self.resource_certainty != ResidentResourceCertainty::HealthyKnown
        {
            return GpuBabBoundWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::PoisonedOrBusy,
            );
        }
        let registration = self.registration;
        let identity = self.transcript.backend;
        let mut entry_guard = match registration.live_guard(identity) {
            Ok(guard) => guard,
            Err(_) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.issuer_claimed = false;
                self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                return GpuBabBoundWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                );
            }
        };
        if Instant::now() >= request.deadline || Instant::now() >= self.phase.deadline {
            let disposition = GpuBabBoundWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
            );
            entry_guard.poisoned = true;
            self.poison_guarded_registry_with_known_resources();
            drop(entry_guard);
            return disposition;
        }
        drop(entry_guard);
        let resident_policy_observation_sha256 = match self.observe_resident_policy_for_v1() {
            Ok(identity) => identity,
            Err(terminal) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                if terminal == GpuBabBoundSessionTerminal::BackendResidentPolicyPanicked {
                    self.poison_registry();
                } else {
                    self.poison_registry_with_known_resources();
                }
                return GpuBabBoundWavePreparation::SessionTerminal(terminal);
            }
        };
        let mut entry_guard = match registration.live_guard(identity) {
            Ok(guard) => guard,
            Err(_) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.issuer_claimed = false;
                self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                return GpuBabBoundWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                );
            }
        };
        if Instant::now() >= request.deadline || Instant::now() >= self.phase.deadline {
            entry_guard.poisoned = true;
            self.poison_guarded_registry_with_known_resources();
            drop(entry_guard);
            return GpuBabBoundWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
            );
        }
        let resident_validation_deadline = ResidentValidationDeadline::new_without_test_injection(
            request.deadline,
            self.phase.deadline,
        );
        match self
            .resident_domains
            .ledger_audit_with_deadline(Some(resident_validation_deadline))
        {
            Ok(audit) if audit.resident_device_bytes == 0 => {}
            Ok(_) => {
                if Instant::now() >= request.deadline || Instant::now() >= self.phase.deadline {
                    entry_guard.poisoned = true;
                    self.poison_guarded_registry_with_known_resources();
                    drop(entry_guard);
                    return GpuBabBoundWavePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                    );
                }
                // V1 receipts have no retained-v2 residency term. Until that
                // schema is versioned, never call raw v1 preflight/work while
                // any physical v2 slot is charged; doing so could validate a
                // peak that omits provider-owned VRAM. RefreshOnly (zero-byte)
                // logical slots may coexist because they contribute no device
                // residency.
                let disposition = GpuBabBoundWavePreparation::CleanDecline(
                    GpuBabBoundWaveDecline::TemporarilyUnavailable,
                );
                drop(entry_guard);
                return disposition;
            }
            Err(NyError::DeadlineExceeded(_)) => {
                entry_guard.poisoned = true;
                self.poison_guarded_registry_with_known_resources();
                drop(entry_guard);
                return GpuBabBoundWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                );
            }
            Err(_) => {
                entry_guard.poisoned = true;
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.issuer_claimed = false;
                self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                drop(entry_guard);
                return GpuBabBoundWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                );
            }
        }
        drop(entry_guard);
        let shape = match request.validate_for_prepare(&self.phase) {
            Ok(shape) => shape,
            Err(NyError::DeadlineExceeded(_)) => {
                let mut live_guard = match registration.live_guard(identity) {
                    Ok(guard) => guard,
                    Err(_) => {
                        self.resident_domains.poison_all();
                        self.state = LeaseState::Poisoned;
                        self.issuer_claimed = false;
                        self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                        return GpuBabBoundWavePreparation::SessionTerminal(
                            GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                        );
                    }
                };
                let disposition = GpuBabBoundWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                );
                live_guard.poisoned = true;
                self.poison_guarded_registry_with_known_resources();
                drop(live_guard);
                return disposition;
            }
            Err(error) => return GpuBabBoundWavePreparation::InvalidRequest(error),
        };
        if !self
            .policy
            .is_valid_for_shape(request.domains.len(), request.objective_indices.len())
            || shape.required_dispatches > self.policy.maximum_dispatches_per_wave
            || self.open_memory.peak_device_bytes > request.max_device_bytes
        {
            let registration = self.registration;
            let identity = self.transcript.backend;
            let live_guard = match registration.live_guard(identity) {
                Ok(guard) => guard,
                Err(_) => {
                    self.resident_domains.poison_all();
                    self.state = LeaseState::Poisoned;
                    self.issuer_claimed = false;
                    self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                    return GpuBabBoundWavePreparation::SessionTerminal(
                        GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                    );
                }
            };
            if Instant::now() >= request.deadline || Instant::now() >= self.phase.deadline {
                let mut live_guard = live_guard;
                let disposition = GpuBabBoundWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
                );
                live_guard.poisoned = true;
                self.poison_guarded_registry_with_known_resources();
                drop(live_guard);
                return disposition;
            }
            let disposition = GpuBabBoundWavePreparation::CleanDecline(
                GpuBabBoundWaveDecline::InsufficientCapacity,
            );
            drop(live_guard);
            return disposition;
        }
        let next_wave_index = match self.last_wave_index.checked_add(1) {
            Some(index) if index != 0 => index,
            _ => {
                self.state = LeaseState::Poisoned;
                self.poison_registry_with_known_resources();
                return GpuBabBoundWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::WaveSequenceExhausted,
                );
            }
        };
        let schedule_identity_sha256 = bind_v1_schedule_to_resident_policy(
            shape.schedule_identity_sha256,
            resident_policy_observation_sha256,
        );
        let prepared = GpuBabBoundPreparedWave {
            request: &request,
            schedule_identity_sha256,
            inherited_endpoints_sha256: shape.inherited_endpoints_sha256,
        };
        match self.recheck_resident_policy_for_close() {
            GpuBabBoundResidentClosePolicyRecheck::Stable => {}
            GpuBabBoundResidentClosePolicyRecheck::Changed => {
                self.state = LeaseState::Poisoned;
                self.poison_registry_with_known_resources();
                return GpuBabBoundWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                );
            }
            GpuBabBoundResidentClosePolicyRecheck::Panicked => {
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentPolicyPanicked,
                );
            }
        }
        let mut pre_raw_guard = match registration.live_guard(identity) {
            Ok(guard) => guard,
            Err(_) => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.issuer_claimed = false;
                self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                return GpuBabBoundWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                );
            }
        };
        if Instant::now() >= request.deadline || Instant::now() >= self.phase.deadline {
            let disposition = GpuBabBoundWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
            );
            pre_raw_guard.poisoned = true;
            self.poison_guarded_registry_with_known_resources();
            drop(pre_raw_guard);
            return disposition;
        }
        drop(pre_raw_guard);
        self.mark_resources_unknown();
        let decision = catch_tcb_unwind(|| {
            self.session
                .as_mut()
                .expect("open lease owns a raw session")
                .prepare_wave(&prepared)
        });
        let decision = match decision {
            Ok(decision) => {
                // Raw prepare is contractually pure. A normal return restores
                // the exact pre-call retained-resource certainty.
                self.mark_resources_healthy_known();
                decision
            }
            Err(()) => {
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPreparePanicked,
                );
            }
        };
        match self.recheck_resident_policy_for_close() {
            GpuBabBoundResidentClosePolicyRecheck::Stable => {}
            GpuBabBoundResidentClosePolicyRecheck::Changed => {
                self.state = LeaseState::Poisoned;
                self.poison_registry_with_known_resources();
                return GpuBabBoundWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentAuthorityLost,
                );
            }
            GpuBabBoundResidentClosePolicyRecheck::Panicked => {
                self.state = LeaseState::Poisoned;
                self.poison_registry();
                return GpuBabBoundWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendResidentPolicyPanicked,
                );
            }
        }
        let registration = self.registration;
        let identity = self.transcript.backend;
        let mut live_guard = match registration.live_guard_noalloc(identity) {
            Some(guard) => guard,
            None => {
                self.resident_domains.poison_all();
                self.state = LeaseState::Poisoned;
                self.issuer_claimed = false;
                self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                return GpuBabBoundWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::RegistrationAuthorityLost,
                );
            }
        };
        let after_prepare = Instant::now();
        if after_prepare >= request.deadline || after_prepare >= self.phase.deadline {
            live_guard.poisoned = true;
            self.poison_guarded_registry_with_known_resources();
            drop(live_guard);
            return GpuBabBoundWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired,
            );
        }
        match decision {
            GpuBabBoundBackendPrepareDisposition::CleanDecline(reason) => {
                let disposition = GpuBabBoundWavePreparation::CleanDecline(reason);
                drop(live_guard);
                disposition
            }
            GpuBabBoundBackendPrepareDisposition::Accepted => {
                self.last_wave_index = next_wave_index;
                self.state = LeaseState::WaveAccepted(next_wave_index);
                let transcript = GpuBabBoundTerminalTranscript {
                    phase: self.transcript,
                    wave_index: next_wave_index,
                    schedule_identity_sha256,
                    inherited_endpoints_sha256: shape.inherited_endpoints_sha256,
                    deadline: request.deadline,
                    max_device_bytes: request.max_device_bytes,
                };
                let disposition = GpuBabBoundWavePreparation::Accepted(GpuBabBoundWaveCapability {
                    lease: self,
                    request: Some(request),
                    shape,
                    transcript,
                    execution_started: false,
                    executed: false,
                });
                drop(live_guard);
                disposition
            }
        }
    }

    /// Core-owned terminal retained when an accepted capability was dropped
    /// without execution. The phase remains poisoned and cannot retry/fallback.
    #[must_use]
    pub fn abandoned_terminal(&self) -> Option<&GpuBabBoundWaveFailure> {
        self.abandoned_terminal.as_ref()
    }

    /// Consume the phase and close its raw backend session exactly once.
    /// Cleanup is deliberately deadline-exempt: it performs no bound dispatch
    /// and must remain available to release or poison retained authority.
    #[must_use = "close disposition determines whether retained authority released cleanly"]
    pub fn close(mut self) -> GpuBabBoundPhaseCloseDisposition {
        let prior_state = self.state;
        let prior_certainty = self.resource_certainty;
        let ledger = self.resident_domains.close_ledger_audit();
        let quiescent = self.resident_domains.resources_are_quiescent()
            && ledger
                .as_ref()
                .is_ok_and(|audit| audit.in_flight_slots == 0 && audit.reserved_slots == 0);
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum CloseAuthority {
            Healthy,
            KnownPoisoned,
            Unknown,
        }
        #[derive(Clone, Copy)]
        enum CloseReason {
            Normal,
            PriorLifecycle,
            LedgerInvalid,
            PolicyChanged,
            PolicyPanicked,
            RegistrationLost,
        }
        let mut authority = match (prior_state, prior_certainty, quiescent) {
            (LeaseState::Open, ResidentResourceCertainty::HealthyKnown, true) => {
                CloseAuthority::Healthy
            }
            (LeaseState::Poisoned, ResidentResourceCertainty::PoisonedKnown, true) => {
                CloseAuthority::KnownPoisoned
            }
            _ => CloseAuthority::Unknown,
        };
        let mut reason = if ledger.is_err() {
            CloseReason::LedgerInvalid
        } else if authority == CloseAuthority::Unknown {
            CloseReason::PriorLifecycle
        } else {
            CloseReason::Normal
        };
        self.state = LeaseState::Closed;
        let core_host_audit = ledger
            .as_ref()
            .ok()
            .map(|audit| GpuBabBoundResidentHostAudit {
                retained_v2_core_host_before_charged_bytes: audit.core_host_charged_bytes,
                retained_v2_core_host_peak_charged_bytes: audit.core_host_charged_bytes,
                retained_v2_core_host_after_charged_bytes: 0,
                history_before_words: audit.history_words,
                history_peak_words: audit.history_words,
                history_after_words: 0,
            });
        if ledger.is_err() {
            authority = CloseAuthority::Unknown;
            self.poison_registry();
        }
        if authority == CloseAuthority::Healthy && self.resident_domains.policy_was_observed() {
            self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
            match self.recheck_resident_policy_for_close() {
                GpuBabBoundResidentClosePolicyRecheck::Stable => {
                    self.resource_certainty = ResidentResourceCertainty::HealthyKnown;
                }
                GpuBabBoundResidentClosePolicyRecheck::Changed => {
                    if self.poison_registry_with_known_resources() {
                        authority = CloseAuthority::KnownPoisoned;
                        reason = CloseReason::PolicyChanged;
                    } else {
                        authority = CloseAuthority::Unknown;
                        reason = CloseReason::RegistrationLost;
                    }
                }
                GpuBabBoundResidentClosePolicyRecheck::Panicked => {
                    authority = CloseAuthority::Unknown;
                    reason = CloseReason::PolicyPanicked;
                    self.poison_registry();
                }
            }
        }
        if authority == CloseAuthority::Healthy
            && !self
                .registration
                .check_live_noalloc(self.transcript.backend)
        {
            authority = CloseAuthority::Unknown;
            reason = CloseReason::RegistrationLost;
            self.poison_registry();
        }
        if authority == CloseAuthority::KnownPoisoned {
            // A prior PoisonedKnown lifecycle has already absorbed the exact
            // issuer claim. Preserve its local quiescence proof; there is no
            // live registration left to reacquire.
            self.resident_domains.poison_all();
            self.resource_certainty = ResidentResourceCertainty::PoisonedKnown;
        } else if authority == CloseAuthority::Unknown {
            self.poison_registry();
        }
        let Some(mut session) = self.session.take() else {
            self.poison_registry();
            return GpuBabBoundPhaseCloseDisposition::AcceptedFailure {
                detail: "phase close had no raw session".into(),
                receipt: None,
                receipt_validated: false,
                core_host_audit,
            };
        };
        // Every resource-capable raw boundary is Unknown while it executes.
        self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
        let raw = call_session_close(&mut session);
        let raw = match raw {
            Ok(raw) => raw,
            Err(()) => {
                self.poison_registry();
                let drop_clean = destroy_session(session);
                return GpuBabBoundPhaseCloseDisposition::AcceptedFailure {
                    detail: if drop_clean {
                        "raw backend panicked during phase close".into()
                    } else {
                        "raw backend close and session destructor both panicked".into()
                    },
                    receipt: None,
                    receipt_validated: false,
                    core_host_audit,
                };
            }
        };
        let receipt = raw_close_receipt(&raw);
        let receipt_exact = ledger.as_ref().is_ok_and(|audit| {
            close_receipt_matches_ledger(&receipt, self.transcript, self.open_memory, *audit)
        });
        let raw_is_closed = matches!(raw, GpuBabBoundBackendCloseDisposition::Closed(_));
        let terminal_claimed =
            if authority == CloseAuthority::Healthy && receipt_exact && !raw_is_closed {
                let claimed = self
                    .registration
                    .terminal_claim_noalloc(self.transcript.backend)
                    .is_some();
                self.issuer_claimed = false;
                claimed
            } else {
                false
            };
        if !receipt_exact || authority == CloseAuthority::Unknown {
            self.poison_registry();
        }
        let drop_clean = destroy_session(session);
        if !drop_clean {
            self.poison_registry();
            return GpuBabBoundPhaseCloseDisposition::AcceptedFailure {
                detail: "raw backend session destructor panicked after phase close".into(),
                receipt: Some(receipt),
                receipt_validated: false,
                core_host_audit,
            };
        }
        if authority == CloseAuthority::Healthy && receipt_exact && raw_is_closed {
            if let Some(exact_host_audit) = core_host_audit {
                if self.registration.release_noalloc(self.transcript.backend) {
                    self.issuer_claimed = false;
                    return GpuBabBoundPhaseCloseDisposition::Closed(
                        GpuBabBoundValidatedPhaseClose {
                            receipt,
                            core_host_audit: exact_host_audit,
                        },
                    );
                }
            }
            self.poison_registry();
            return GpuBabBoundPhaseCloseDisposition::AcceptedFailure {
                detail: "validated close could not release issuer".into(),
                receipt: Some(receipt),
                receipt_validated: false,
                core_host_audit,
            };
        }
        if authority == CloseAuthority::KnownPoisoned && receipt_exact {
            self.resource_certainty = ResidentResourceCertainty::PoisonedKnown;
            return GpuBabBoundPhaseCloseDisposition::AcceptedFailure {
                detail: match reason {
                    CloseReason::PolicyChanged => {
                        "resident policy changed before exact poisoned cleanup"
                    }
                    _ => "exact cleanup followed a known-poisoned phase lifecycle",
                }
                .into(),
                receipt: Some(receipt),
                receipt_validated: true,
                core_host_audit,
            };
        }
        if authority == CloseAuthority::Healthy && receipt_exact && terminal_claimed {
            return GpuBabBoundPhaseCloseDisposition::AcceptedFailure {
                detail: match raw {
                    GpuBabBoundBackendCloseDisposition::AcceptedFailure { detail, .. } => detail,
                    GpuBabBoundBackendCloseDisposition::Closed(_) => {
                        "raw backend reported an exact accepted close failure".into()
                    }
                },
                receipt: Some(receipt),
                receipt_validated: true,
                core_host_audit,
            };
        }
        self.poison_registry();
        GpuBabBoundPhaseCloseDisposition::AcceptedFailure {
            detail: match reason {
                CloseReason::LedgerInvalid => "core resident close ledger was inconsistent",
                CloseReason::PolicyPanicked => "resident policy recheck panicked before close",
                CloseReason::RegistrationLost => {
                    "phase close lost live registration authority before cleanup"
                }
                CloseReason::PriorLifecycle | CloseReason::PolicyChanged => {
                    "phase close followed an unvalidated poisoned lifecycle"
                }
                CloseReason::Normal => "raw backend close receipt violated the phase contract",
            }
            .into(),
            receipt: Some(receipt),
            receipt_validated: false,
            core_host_audit,
        }
    }

    fn poison_registry(&mut self) {
        self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
        self.resident_domains.poison_all();
        if self.issuer_claimed {
            self.registration.poison(self.transcript.backend);
            self.issuer_claimed = false;
        }
    }

    fn mark_resources_unknown(&mut self) {
        self.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
    }

    fn mark_resources_healthy_known(&mut self) {
        self.resource_certainty = ResidentResourceCertainty::HealthyKnown;
    }

    /// Finish poisoning after the caller has atomically marked its held live
    /// registration guard poisoned. No raw resource-capable call occurred, or
    /// a validated terminal proved quiescence, so the resident ledger remains
    /// structurally auditable even though issuer authority is absorbed.
    fn poison_guarded_registry_with_known_resources(&mut self) {
        self.resident_domains.poison_all();
        self.state = LeaseState::Poisoned;
        self.issuer_claimed = false;
        self.resource_certainty = if self.resident_domains.resources_are_quiescent() {
            ResidentResourceCertainty::PoisonedKnown
        } else {
            ResidentResourceCertainty::PoisonedUnknown
        };
    }

    fn poison_registry_with_known_resources(&mut self) -> bool {
        let registration = self.registration;
        let identity = self.transcript.backend;
        match registration.live_guard_noalloc(identity) {
            Some(mut guard) => {
                guard.poisoned = true;
                self.poison_guarded_registry_with_known_resources();
                drop(guard);
                self.resource_certainty == ResidentResourceCertainty::PoisonedKnown
            }
            None => {
                // Losing the exact live claim invalidates a local quiescence
                // proof even when the slot counters themselves are zero.
                self.poison_registry();
                false
            }
        }
    }
}

impl Drop for GpuBabBoundPhaseLease<'_> {
    fn drop(&mut self) {
        self.state = LeaseState::Closed;
        // An implicit drop has no caller-observed validated close terminal.
        // Poison before untrusted cleanup. During an outer unwind, forgetting
        // the session is a deliberate fail-closed resource leak that prevents
        // an attacker-defined destructor from creating a double panic.
        self.poison_registry();
        if let Some(session) = self.session.take() {
            if std::thread::panicking() {
                std::mem::forget(session);
            } else {
                let _ = close_and_destroy_poisoned_session(session);
            }
        }
    }
}

struct SessionCloseAttempt {
    raw: Option<GpuBabBoundBackendCloseDisposition>,
    close_clean: bool,
    drop_clean: bool,
}

fn discard_dormant_session(session: Box<dyn GpuBabBoundBackendSession + '_>) -> bool {
    destroy_session(session)
}

fn call_session_close(
    session: &mut Box<dyn GpuBabBoundBackendSession + '_>,
) -> std::result::Result<GpuBabBoundBackendCloseDisposition, ()> {
    catch_tcb_unwind(|| session.close())
}

fn destroy_session(session: Box<dyn GpuBabBoundBackendSession + '_>) -> bool {
    catch_tcb_unwind(|| drop(session)).is_ok()
}

/// Best-effort cleanup only after the registration has already entered its
/// absorbing poisoned state. Explicit close must preserve the session in its
/// protective lease owner until raw validation selects the terminal ordering.
fn close_and_destroy_poisoned_session(
    mut session: Box<dyn GpuBabBoundBackendSession + '_>,
) -> SessionCloseAttempt {
    let raw = call_session_close(&mut session);
    let close_clean = raw.is_ok();
    let raw = raw.ok();
    let drop_clean = destroy_session(session);
    SessionCloseAttempt {
        raw,
        close_clean,
        drop_clean,
    }
}

fn raw_close_receipt(raw: &GpuBabBoundBackendCloseDisposition) -> GpuBabBoundBackendCloseReceipt {
    match raw {
        GpuBabBoundBackendCloseDisposition::Closed(receipt)
        | GpuBabBoundBackendCloseDisposition::AcceptedFailure { receipt, .. } => *receipt,
    }
}

fn close_receipt_matches_ledger(
    receipt: &GpuBabBoundBackendCloseReceipt,
    transcript: GpuBabBoundPhaseTranscript,
    open_memory: GpuBabBoundMemoryReceipt,
    ledger: GpuBabBoundResidentLedgerAudit,
) -> bool {
    ledger
        .resident_slots
        .checked_add(ledger.refresh_only_slots)
        .is_some_and(|resident_logical_slots| {
            receipt.transcript == transcript
                && receipt.released_graph_bytes == open_memory.retained_graph_bytes
                && receipt.released_phase_bytes == open_memory.retained_phase_bytes
                && receipt.released_resident_device_bytes == ledger.resident_device_bytes
                && receipt.released_resident_slots == ledger.resident_slots
                && receipt.released_refresh_only_slots == ledger.refresh_only_slots
                && receipt.released_resident_logical_slots == resident_logical_slots
        })
}

/// Validate best-effort cleanup only after the registration has atomically
/// entered absorbing poison. The name encodes this mandatory call-site
/// precondition; live explicit close uses its split validation protocol.
fn best_effort_close_poisoned(
    session: Box<dyn GpuBabBoundBackendSession + '_>,
    transcript: GpuBabBoundPhaseTranscript,
    open_memory: Option<GpuBabBoundMemoryReceipt>,
) -> bool {
    let attempt = close_and_destroy_poisoned_session(session);
    match (attempt.raw, open_memory) {
        (Some(raw), Some(memory)) if attempt.close_clean && attempt.drop_clean => {
            raw_close_matches_open(&raw, transcript, memory)
        }
        _ => false,
    }
}

fn raw_close_matches_open(
    raw: &GpuBabBoundBackendCloseDisposition,
    transcript: GpuBabBoundPhaseTranscript,
    open_memory: GpuBabBoundMemoryReceipt,
) -> bool {
    match raw {
        GpuBabBoundBackendCloseDisposition::Closed(receipt) => {
            receipt.transcript == transcript
                && receipt.released_graph_bytes == open_memory.retained_graph_bytes
                && receipt.released_phase_bytes == open_memory.retained_phase_bytes
                && receipt.released_resident_device_bytes == 0
                && receipt.released_resident_slots == 0
                && receipt.released_refresh_only_slots == 0
                && receipt.released_resident_logical_slots == 0
        }
        GpuBabBoundBackendCloseDisposition::AcceptedFailure { .. } => false,
    }
}

fn zero_terminal_open_receipt(
    transcript: GpuBabBoundPhaseTranscript,
) -> GpuBabBoundBackendOpenReceipt {
    GpuBabBoundBackendOpenReceipt {
        transcript,
        authorized_device_bytes: transcript.max_device_bytes,
        memory: GpuBabBoundMemoryReceipt {
            retained_graph_bytes: 0,
            retained_phase_bytes: 0,
            wave_working_bytes: 0,
            queued_upload_bytes: 0,
            result_readback_bytes: 0,
            peak_device_bytes: 0,
        },
        static_transfers: GpuBabBoundStaticTransferReceipt {
            graph_identity_sha256: transcript.graph_identity_sha256,
            phase_identity_sha256: transcript.static_phase_identity_sha256,
            graph_payload_bytes: transcript.static_graph_payload_bytes,
            phase_payload_bytes: transcript.static_phase_payload_bytes,
            graph_padding_bytes: 0,
            phase_padding_bytes: 0,
            graph_source: GpuBabBoundStaticPayloadSource::NotTransferred,
            phase_source: GpuBabBoundStaticPayloadSource::NotTransferred,
            graph_host_to_device_bytes: 0,
            phase_host_to_device_bytes: 0,
            host_to_device_bytes: 0,
        },
        released_graph_bytes: 0,
        released_phase_bytes: 0,
    }
}

fn make_open_terminal_claimed<'a>(
    registration: &GpuBabBoundBackendRegistration,
    issuer: GpuBabBoundBackendIssuerIdentity,
    backend_kind: GpuBabBoundBackendOpenFailureKind,
    detail: String,
    receipt: GpuBabBoundBackendOpenReceipt,
    phase: &GpuBabBoundPhaseDescriptor,
    claimed_deadline: bool,
    mut session: GpuBabBoundPreparedSessionSlot<'_>,
) -> GpuBabBoundPhaseOpen<'a> {
    let preliminary = validate_raw_terminal_open_before_close(issuer, &receipt, phase);
    let preliminary_error = preliminary.err().map(|error| error.to_string());
    let actually_expired = Instant::now() >= phase.deadline;
    let deadline_valid = !claimed_deadline || actually_expired;
    let mut final_error = None;
    let terminal_claim = if preliminary_error.is_none() && deadline_valid {
        match registration.terminal_claim(issuer) {
            Ok(claim) => Some(claim),
            Err(error) => {
                final_error = Some(error.to_string());
                None
            }
        }
    } else {
        registration.poison(issuer);
        None
    };
    let expected_transcript = GpuBabBoundPhaseTranscript::expected(issuer, phase);
    let close_validated =
        best_effort_close_poisoned(session.take(), expected_transcript, Some(receipt.memory));
    let mut terminal_receipt = receipt;
    if close_validated {
        terminal_receipt.released_graph_bytes = receipt.memory.retained_graph_bytes;
        terminal_receipt.released_phase_bytes = receipt.memory.retained_phase_bytes;
    }
    let receipt_validated = match terminal_claim {
        Some(claim) if close_validated => {
            let validation =
                validate_claimed_terminal_open(claim, issuer, &terminal_receipt, phase);
            let valid = validation.is_ok();
            final_error = validation.err().map(|error| error.to_string());
            valid
        }
        Some(_) | None => false,
    };
    if claimed_deadline && actually_expired && receipt_validated {
        GpuBabBoundPhaseOpen::DeadlineExpired(GpuBabBoundPhaseOpenFailure {
            kind: GpuBabBoundTerminalFailureKind::OpenBackend(backend_kind),
            detail,
            receipt: terminal_receipt,
            receipt_validated: true,
        })
    } else {
        let validation_detail = preliminary_error
            .or_else(|| (!close_validated).then(|| "terminal open close was not exact".into()))
            .or(final_error)
            .map_or_else(String::new, |error| format!("; contract: {error}"));
        let kind = if claimed_deadline && !actually_expired || !receipt_validated {
            GpuBabBoundTerminalFailureKind::ContractViolation
        } else {
            GpuBabBoundTerminalFailureKind::OpenBackend(backend_kind)
        };
        GpuBabBoundPhaseOpen::AcceptedFailure(GpuBabBoundPhaseOpenFailure {
            kind,
            detail: format!("{detail}{validation_detail}"),
            receipt: terminal_receipt,
            receipt_validated,
        })
    }
}

/// Core-validated close result.
pub struct GpuBabBoundValidatedPhaseClose {
    receipt: GpuBabBoundBackendCloseReceipt,
    core_host_audit: GpuBabBoundResidentHostAudit,
}

impl GpuBabBoundValidatedPhaseClose {
    #[must_use]
    pub fn receipt(&self) -> &GpuBabBoundBackendCloseReceipt {
        &self.receipt
    }

    #[must_use]
    pub fn core_host_audit(&self) -> GpuBabBoundResidentHostAudit {
        self.core_host_audit
    }
}

#[must_use = "close disposition determines whether retained authority released cleanly"]
pub enum GpuBabBoundPhaseCloseDisposition {
    Closed(GpuBabBoundValidatedPhaseClose),
    AcceptedFailure {
        detail: String,
        receipt: Option<GpuBabBoundBackendCloseReceipt>,
        /// On an accepted failure this certifies only exact cleanup-receipt
        /// equality plus clean session destruction. It never means `Closed`,
        /// registration release, restored authority, or fallback permission.
        receipt_validated: bool,
        core_host_audit: Option<GpuBabBoundResidentHostAudit>,
    },
}

/// Reason a new wave cannot be prepared on this lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundSessionTerminal {
    PoisonedOrBusy,
    RegistrationAuthorityLost,
    WaveSequenceExhausted,
    BackendPreparePanicked,
    BackendPrepareDeadlineExpired,
    BackendResidentPolicyPanicked,
    InvalidResidentPolicy,
    BackendResidentAuthorityLost,
}

/// Split preaccept result. Only `CleanDecline` permits legacy fallback.
#[must_use = "wave preparation may own an accepted capability"]
pub enum GpuBabBoundWavePreparation<'lease, 'backend> {
    InvalidRequest(NyError),
    CleanDecline(GpuBabBoundWaveDecline),
    Accepted(GpuBabBoundWaveCapability<'lease, 'backend>),
    SessionTerminal(GpuBabBoundSessionTerminal),
}

impl GpuBabBoundWavePreparation<'_, '_> {
    #[must_use]
    pub fn permits_legacy_fallback(&self) -> bool {
        matches!(self, Self::CleanDecline(_))
    }
}

/// Non-cloneable exact-once authority for one accepted wave.
///
/// This type has no public constructor. Dropping it without executing poisons
/// the phase; executing consumes it. Ordinary consumers cannot directly mint
/// or replay it, and silently abandoning issued work cannot continue the phase.
///
/// ```compile_fail
/// use ny_core::GpuBabBoundWaveCapability;
/// fn replay(capability: GpuBabBoundWaveCapability<'_, '_>) {
///     let _ = capability.execute_accepted();
///     let _ = capability.execute_accepted();
/// }
/// ```
///
/// ```compile_fail
/// use ny_core::GpuBabBoundWaveCapability;
/// let _: GpuBabBoundWaveCapability<'static, 'static> =
///     GpuBabBoundWaveCapability {};
/// ```
#[must_use = "dropping an accepted capability poisons the phase"]
pub struct GpuBabBoundWaveCapability<'lease, 'backend> {
    lease: &'lease mut GpuBabBoundPhaseLease<'backend>,
    request: Option<GpuBabBoundWaveRequest>,
    shape: ValidatedWaveShape,
    transcript: GpuBabBoundTerminalTranscript,
    execution_started: bool,
    executed: bool,
}

impl GpuBabBoundWaveCapability<'_, '_> {
    /// Execute and consume this accepted wave. No generic error can cross the
    /// acceptance boundary; every raw outcome becomes one typed terminal.
    pub fn execute_accepted(mut self) -> GpuBabBoundWaveDisposition {
        let request = self
            .request
            .as_ref()
            .expect("unexecuted capability owns its request");
        let accepted = GpuBabBoundAcceptedWave {
            request,
            transcript: self.transcript,
        };
        match self.lease.recheck_resident_policy_for_close() {
            GpuBabBoundResidentClosePolicyRecheck::Stable => {}
            GpuBabBoundResidentClosePolicyRecheck::Changed => {
                self.lease.state = LeaseState::Poisoned;
                self.lease.poison_registry_with_known_resources();
                let receipt = core_predispatch_failure_receipt(
                    request,
                    self.shape,
                    self.transcript,
                    self.lease.open_memory,
                );
                self.executed = true;
                return contract_failure(
                    "resident policy changed before accepted v1 execution".into(),
                    receipt,
                );
            }
            GpuBabBoundResidentClosePolicyRecheck::Panicked => {
                self.lease.state = LeaseState::Poisoned;
                self.lease.poison_registry();
                let receipt = core_predispatch_failure_receipt(
                    request,
                    self.shape,
                    self.transcript,
                    self.lease.open_memory,
                );
                self.executed = true;
                return contract_failure(
                    "resident policy recheck panicked before accepted v1 execution".into(),
                    receipt,
                );
            }
        }
        let registration = self.lease.registration;
        let identity = self.transcript.phase.backend;
        let mut live_guard = match registration.live_guard_noalloc(identity) {
            Some(guard) => guard,
            None => {
                self.lease.resident_domains.poison_all();
                self.lease.state = LeaseState::Poisoned;
                self.lease.issuer_claimed = false;
                self.lease.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                let receipt = core_predispatch_failure_receipt(
                    request,
                    self.shape,
                    self.transcript,
                    self.lease.open_memory,
                );
                self.executed = true;
                return contract_failure(
                    "live registration authority was lost before accepted execution".into(),
                    receipt,
                );
            }
        };
        if Instant::now() >= request.deadline || Instant::now() >= self.lease.phase.deadline {
            let receipt = core_predispatch_failure_receipt(
                request,
                self.shape,
                self.transcript,
                self.lease.open_memory,
            );
            let validation = validate_failure_receipt(
                &receipt,
                request,
                self.shape,
                self.transcript,
                self.lease.open_memory,
                self.lease.policy,
            );
            let disposition = match validation {
                Ok(()) => GpuBabBoundWaveDisposition::DeadlineExpired(GpuBabBoundWaveFailure {
                    kind: GpuBabBoundTerminalFailureKind::Backend(
                        GpuBabBoundBackendFailureKind::AuthorityLost,
                    ),
                    detail: "accepted capability expired before raw execution began".into(),
                    receipt,
                    receipt_validated: true,
                }),
                Err(error) => contract_failure(
                    format!("core predispatch deadline receipt was invalid: {error}"),
                    receipt,
                ),
            };
            live_guard.poisoned = true;
            self.lease.poison_guarded_registry_with_known_resources();
            self.executed = true;
            drop(live_guard);
            return disposition;
        }
        drop(live_guard);
        self.lease.mark_resources_unknown();
        self.execution_started = true;
        let raw = catch_tcb_unwind(|| {
            self.lease
                .session
                .as_mut()
                .expect("accepted lease owns a raw session")
                .execute_accepted(&accepted)
        });
        let raw = match raw {
            Ok(raw) => raw,
            Err(()) => {
                self.lease.state = LeaseState::Poisoned;
                self.lease.poison_registry();
                let receipt = core_predispatch_failure_receipt(
                    request,
                    self.shape,
                    self.transcript,
                    self.lease.open_memory,
                );
                self.executed = true;
                return contract_failure(
                    "raw backend panicked after accepted wave; resource receipt is unknown".into(),
                    receipt,
                );
            }
        };
        let disposition =
            finish_accepted_wave(&mut *self.lease, request, self.shape, self.transcript, raw);
        self.executed = true;
        disposition
    }
}

impl Drop for GpuBabBoundWaveCapability<'_, '_> {
    fn drop(&mut self) {
        if !self.executed {
            let registration = self.lease.registration;
            let identity = self.transcript.phase.backend;
            let mut live_guard = registration.live_guard_noalloc(identity);
            let resources_known = !self.execution_started && live_guard.is_some();
            self.lease.resident_domains.poison_all();
            if let Some(guard) = live_guard.as_mut() {
                guard.poisoned = true;
            }
            self.lease.state = LeaseState::Poisoned;
            self.lease.issuer_claimed = false;
            self.lease.resource_certainty =
                if resources_known && self.lease.resident_domains.resources_are_quiescent() {
                    ResidentResourceCertainty::PoisonedKnown
                } else {
                    ResidentResourceCertainty::PoisonedUnknown
                };
            if let Some(request) = self.request.as_ref() {
                let receipt = core_predispatch_failure_receipt(
                    request,
                    self.shape,
                    self.transcript,
                    self.lease.open_memory,
                );
                let receipt_validated = resources_known;
                self.lease.abandoned_terminal = Some(GpuBabBoundWaveFailure {
                    kind: if receipt_validated {
                        GpuBabBoundTerminalFailureKind::CapabilityAbandoned
                    } else {
                        GpuBabBoundTerminalFailureKind::ContractViolation
                    },
                    detail: if self.execution_started {
                        "accepted execution unwound without a trustworthy resource receipt".into()
                    } else {
                        "accepted capability dropped before execution".into()
                    },
                    receipt,
                    receipt_validated,
                });
            }
        }
    }
}

/// Validated row. Its fields cannot be constructed or mutated outside ny-core.
#[derive(Debug, PartialEq)]
pub struct GpuBabBoundValidatedRow {
    parent_group_id: u64,
    child_ordinal: usize,
    child_cardinality: usize,
    domain_slot: u64,
    domain_identity_sha256: [u8; 32],
    objective_index: u32,
    q: u32,
    lower: f32,
    upper: f32,
}

impl GpuBabBoundValidatedRow {
    #[must_use]
    pub fn parent_group_id(&self) -> u64 {
        self.parent_group_id
    }

    #[must_use]
    pub fn child_ordinal(&self) -> usize {
        self.child_ordinal
    }

    #[must_use]
    pub fn child_cardinality(&self) -> usize {
        self.child_cardinality
    }

    #[must_use]
    pub fn domain_slot(&self) -> u64 {
        self.domain_slot
    }

    #[must_use]
    pub fn domain_identity_sha256(&self) -> &[u8; 32] {
        &self.domain_identity_sha256
    }

    #[must_use]
    pub fn objective_index(&self) -> u32 {
        self.objective_index
    }

    #[must_use]
    pub fn q(&self) -> u32 {
        self.q
    }

    #[must_use]
    pub fn lower(&self) -> f32 {
        self.lower
    }

    #[must_use]
    pub fn upper(&self) -> f32 {
        self.upper
    }
}

/// Receipt that passed the mandatory core validator through a live capability.
#[derive(Debug, PartialEq, Eq)]
pub struct GpuBabBoundValidatedWaveReceipt {
    raw: GpuBabBoundBackendWaveReceipt,
}

impl GpuBabBoundValidatedWaveReceipt {
    #[must_use]
    pub fn transcript(&self) -> &GpuBabBoundTerminalTranscript {
        &self.raw.transcript
    }

    #[must_use]
    pub fn tightened_rows(&self) -> usize {
        self.raw.tightened_rows
    }

    #[must_use]
    pub fn raw_audit_receipt(&self) -> &GpuBabBoundBackendWaveReceipt {
        &self.raw
    }
}

/// Core-checked per-domain terminal class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBabBoundValidatedDomainOutcomeKind {
    Bounded,
}

/// Validated association and typed outcome for one admitted domain.
#[derive(Debug, PartialEq, Eq)]
pub struct GpuBabBoundValidatedDomainOutcome {
    parent_group_id: u64,
    child_ordinal: usize,
    child_cardinality: usize,
    domain_slot: u64,
    domain_identity_sha256: [u8; 32],
    kind: GpuBabBoundValidatedDomainOutcomeKind,
}

impl GpuBabBoundValidatedDomainOutcome {
    #[must_use]
    pub fn parent_group_id(&self) -> u64 {
        self.parent_group_id
    }

    #[must_use]
    pub fn child_ordinal(&self) -> usize {
        self.child_ordinal
    }

    #[must_use]
    pub fn child_cardinality(&self) -> usize {
        self.child_cardinality
    }

    #[must_use]
    pub fn domain_slot(&self) -> u64 {
        self.domain_slot
    }

    #[must_use]
    pub fn domain_identity_sha256(&self) -> &[u8; 32] {
        &self.domain_identity_sha256
    }

    #[must_use]
    pub fn kind(&self) -> GpuBabBoundValidatedDomainOutcomeKind {
        self.kind
    }
}

/// Authoritative completed result issued only by a consuming live capability.
///
/// Neither fields nor constructors are public. A backend can construct only
/// raw rows/receipts; ordinary consumers cannot directly construct this
/// wrapper. A safe downstream implementation exposed through
/// [`GpuBabBoundNumericalTcb`] is an explicit numerical-TCB expansion that can
/// cause core to construct it from raw endpoints and must be source-reviewed.
///
/// ```compile_fail
/// use ny_core::{GpuBabBoundValidatedDomainOutcome, GpuBabBoundValidatedRow,
///     GpuBabBoundValidatedWaveReceipt, ValidatedGpuBabBoundWaveResult};
/// fn forge(
///     domain_outcomes: Vec<GpuBabBoundValidatedDomainOutcome>,
///     rows: Vec<GpuBabBoundValidatedRow>,
///     receipt: GpuBabBoundValidatedWaveReceipt,
/// ) -> ValidatedGpuBabBoundWaveResult {
///     ValidatedGpuBabBoundWaveResult { domain_outcomes, rows, receipt }
/// }
/// ```
#[derive(Debug, PartialEq)]
pub struct ValidatedGpuBabBoundWaveResult {
    domain_outcomes: Vec<GpuBabBoundValidatedDomainOutcome>,
    rows: Vec<GpuBabBoundValidatedRow>,
    receipt: GpuBabBoundValidatedWaveReceipt,
}

impl ValidatedGpuBabBoundWaveResult {
    #[must_use]
    pub fn domain_outcomes(&self) -> &[GpuBabBoundValidatedDomainOutcome] {
        &self.domain_outcomes
    }

    #[must_use]
    pub fn rows(&self) -> &[GpuBabBoundValidatedRow] {
        &self.rows
    }

    #[must_use]
    pub fn receipt(&self) -> &GpuBabBoundValidatedWaveReceipt {
        &self.receipt
    }

    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        Vec<GpuBabBoundValidatedDomainOutcome>,
        Vec<GpuBabBoundValidatedRow>,
        GpuBabBoundValidatedWaveReceipt,
    ) {
        (self.domain_outcomes, self.rows, self.receipt)
    }
}

/// Core-owned terminal failure, always carrying the raw postaccept receipt.
pub struct GpuBabBoundWaveFailure {
    kind: GpuBabBoundTerminalFailureKind,
    detail: String,
    receipt: GpuBabBoundBackendWaveReceipt,
    receipt_validated: bool,
}

impl GpuBabBoundWaveFailure {
    #[must_use]
    pub fn kind(&self) -> GpuBabBoundTerminalFailureKind {
        self.kind
    }

    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }

    #[must_use]
    pub fn receipt(&self) -> &GpuBabBoundBackendWaveReceipt {
        &self.receipt
    }

    #[must_use]
    pub fn receipt_validated(&self) -> bool {
        self.receipt_validated
    }
}

/// Mandatory core-validated terminal for one consumed accepted capability.
#[must_use = "postaccept disposition owns the terminal result or failure"]
pub enum GpuBabBoundWaveDisposition {
    Completed(ValidatedGpuBabBoundWaveResult),
    AcceptedFailure(GpuBabBoundWaveFailure),
    DeadlineExpired(GpuBabBoundWaveFailure),
}

impl GpuBabBoundWaveDisposition {
    /// Postaccept outcomes are terminal and can never restore fallback.
    #[must_use]
    pub fn permits_legacy_fallback(&self) -> bool {
        false
    }
}

fn finish_accepted_wave(
    lease: &mut GpuBabBoundPhaseLease<'_>,
    request: &GpuBabBoundWaveRequest,
    shape: ValidatedWaveShape,
    transcript: GpuBabBoundTerminalTranscript,
    raw: GpuBabBoundBackendWaveDisposition,
) -> GpuBabBoundWaveDisposition {
    if lease.state != LeaseState::WaveAccepted(transcript.wave_index) {
        lease.state = LeaseState::Poisoned;
        lease.poison_registry();
        let receipt = raw_receipt(&raw);
        return contract_failure(
            "accepted wave did not own the live lease state".into(),
            receipt,
        );
    }
    if lease.recheck_resident_policy_for_close() != GpuBabBoundResidentClosePolicyRecheck::Stable {
        lease.resident_domains.poison_all();
        lease.state = LeaseState::Poisoned;
        lease.poison_registry();
        let receipt = raw_receipt(&raw);
        return contract_failure(
            "resident policy changed before v1 terminal publication".into(),
            receipt,
        );
    }
    let registration = lease.registration;
    let mut live_guard = match registration.live_guard_noalloc(transcript.phase.backend) {
        Some(guard) => guard,
        None => {
            lease.resident_domains.poison_all();
            lease.state = LeaseState::Poisoned;
            lease.issuer_claimed = false;
            lease.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
            let receipt = raw_receipt(&raw);
            return contract_failure(
                "live registration authority was lost during raw execution".into(),
                receipt,
            );
        }
    };
    if let Err(error) = request.validate_static(&lease.phase) {
        let receipt = raw_receipt(&raw);
        let disposition = contract_failure(error.to_string(), receipt);
        live_guard.poisoned = true;
        lease.resident_domains.poison_all();
        lease.state = LeaseState::Poisoned;
        lease.issuer_claimed = false;
        drop(live_guard);
        return disposition;
    }
    match raw {
        GpuBabBoundBackendWaveDisposition::Completed {
            domain_outcomes,
            rows,
            receipt,
        } => {
            match validate_completed(
                domain_outcomes,
                rows,
                receipt,
                request,
                shape,
                transcript,
                lease.open_memory,
                lease.policy,
            ) {
                Ok(result) => {
                    lease.state = LeaseState::Open;
                    lease.mark_resources_healthy_known();
                    let disposition = GpuBabBoundWaveDisposition::Completed(result);
                    drop(live_guard);
                    disposition
                }
                Err(error) => {
                    let disposition = contract_failure(error.to_string(), receipt);
                    live_guard.poisoned = true;
                    lease.resident_domains.poison_all();
                    lease.state = LeaseState::Poisoned;
                    lease.issuer_claimed = false;
                    lease.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
                    drop(live_guard);
                    disposition
                }
            }
        }
        GpuBabBoundBackendWaveDisposition::AcceptedFailure {
            kind,
            detail,
            receipt,
        } => {
            let validation = validate_failure_receipt(
                &receipt,
                request,
                shape,
                transcript,
                lease.open_memory,
                lease.policy,
            );
            live_guard.poisoned = true;
            lease.resident_domains.poison_all();
            lease.state = LeaseState::Poisoned;
            lease.issuer_claimed = false;
            // V1 receipts do not attest retained-v2 residency. Even a valid
            // v1 failure receipt cannot recover resource certainty after raw
            // execution may have touched coexisting RefreshOnly state.
            lease.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
            let disposition = match validation {
                Ok(()) => GpuBabBoundWaveDisposition::AcceptedFailure(GpuBabBoundWaveFailure {
                    kind: GpuBabBoundTerminalFailureKind::Backend(kind),
                    detail,
                    receipt,
                    receipt_validated: true,
                }),
                Err(error) => contract_failure(format!("{detail}; contract: {error}"), receipt),
            };
            drop(live_guard);
            disposition
        }
        GpuBabBoundBackendWaveDisposition::DeadlineExpired { detail, receipt } => {
            let validation = validate_failure_receipt(
                &receipt,
                request,
                shape,
                transcript,
                lease.open_memory,
                lease.policy,
            );
            let truly_expired =
                Instant::now() >= request.deadline || Instant::now() >= lease.phase.deadline;
            live_guard.poisoned = true;
            lease.resident_domains.poison_all();
            lease.state = LeaseState::Poisoned;
            lease.issuer_claimed = false;
            lease.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
            let disposition = match (truly_expired, validation) {
                (false, _) => contract_failure(
                    format!("{detail}; contract: deadline disposition was early"),
                    receipt,
                ),
                (true, Ok(())) => {
                    GpuBabBoundWaveDisposition::DeadlineExpired(GpuBabBoundWaveFailure {
                        kind: GpuBabBoundTerminalFailureKind::Backend(
                            GpuBabBoundBackendFailureKind::AuthorityLost,
                        ),
                        detail,
                        receipt,
                        receipt_validated: true,
                    })
                }
                (true, Err(error)) => {
                    contract_failure(format!("{detail}; contract: {error}"), receipt)
                }
            };
            drop(live_guard);
            disposition
        }
        GpuBabBoundBackendWaveDisposition::IllegalCleanDecline { reason, receipt } => {
            live_guard.poisoned = true;
            lease.resident_domains.poison_all();
            lease.state = LeaseState::Poisoned;
            lease.issuer_claimed = false;
            lease.resource_certainty = ResidentResourceCertainty::PoisonedUnknown;
            let disposition = contract_failure(
                format!("backend returned postaccept decline {reason:?}"),
                receipt,
            );
            drop(live_guard);
            disposition
        }
    }
}

fn raw_receipt(raw: &GpuBabBoundBackendWaveDisposition) -> GpuBabBoundBackendWaveReceipt {
    match raw {
        GpuBabBoundBackendWaveDisposition::Completed { receipt, .. }
        | GpuBabBoundBackendWaveDisposition::AcceptedFailure { receipt, .. }
        | GpuBabBoundBackendWaveDisposition::DeadlineExpired { receipt, .. }
        | GpuBabBoundBackendWaveDisposition::IllegalCleanDecline { receipt, .. } => *receipt,
    }
}

fn contract_failure(
    detail: String,
    receipt: GpuBabBoundBackendWaveReceipt,
) -> GpuBabBoundWaveDisposition {
    GpuBabBoundWaveDisposition::AcceptedFailure(GpuBabBoundWaveFailure {
        kind: GpuBabBoundTerminalFailureKind::ContractViolation,
        detail,
        receipt,
        receipt_validated: false,
    })
}

fn validate_domain_outcomes(
    outcomes: Vec<GpuBabBoundBackendDomainOutcome>,
    request: &GpuBabBoundWaveRequest,
) -> Result<(Vec<GpuBabBoundValidatedDomainOutcome>, usize)> {
    if outcomes.len() != request.domains.len() {
        return Err(invalid(format!(
            "completed domain outcomes {} != admitted domains {}",
            outcomes.len(),
            request.domains.len()
        )));
    }
    let mut bounded = 0usize;
    let mut validated = Vec::with_capacity(outcomes.len());
    for (index, outcome) in outcomes.into_iter().enumerate() {
        let domain = &request.domains[index];
        let identity = request.domain_identity_sha256(index)?;
        if outcome.parent_group_id != domain.parent_group_id
            || outcome.child_ordinal != domain.child_ordinal
            || outcome.child_cardinality != domain.child_cardinality
            || outcome.domain_slot != domain.domain_slot
            || outcome.domain_identity_sha256 != identity
        {
            return Err(invalid(format!(
                "domain outcome {index} does not exactly echo parent/child/domain association"
            )));
        }
        let kind = match outcome.kind {
            GpuBabBoundBackendDomainOutcomeKind::Bounded => {
                let lower = domain
                    .operands
                    .box_lower
                    .slice(request.domain_arena.box_lower.as_ref(), "box lower")?;
                let upper = domain
                    .operands
                    .box_upper
                    .slice(request.domain_arena.box_upper.as_ref(), "box upper")?;
                if lower
                    .iter()
                    .zip(upper)
                    .any(|(&lower, &upper)| lower > upper)
                {
                    return Err(invalid(format!(
                        "bounded domain outcome {index} contains an empty box coordinate"
                    )));
                }
                bounded = bounded
                    .checked_add(1)
                    .ok_or_else(|| invalid("bounded domain count overflows usize"))?;
                GpuBabBoundValidatedDomainOutcomeKind::Bounded
            }
        };
        validated.push(GpuBabBoundValidatedDomainOutcome {
            parent_group_id: outcome.parent_group_id,
            child_ordinal: outcome.child_ordinal,
            child_cardinality: outcome.child_cardinality,
            domain_slot: outcome.domain_slot,
            domain_identity_sha256: outcome.domain_identity_sha256,
            kind,
        });
    }
    Ok((validated, bounded))
}

fn validate_completed(
    domain_outcomes: Vec<GpuBabBoundBackendDomainOutcome>,
    rows: Vec<GpuBabBoundBackendRow>,
    receipt: GpuBabBoundBackendWaveReceipt,
    request: &GpuBabBoundWaveRequest,
    shape: ValidatedWaveShape,
    transcript: GpuBabBoundTerminalTranscript,
    open_memory: GpuBabBoundMemoryReceipt,
    policy: GpuBabBoundPhasePolicy,
) -> Result<ValidatedGpuBabBoundWaveResult> {
    let (validated_outcomes, bounded_domains) = validate_domain_outcomes(domain_outcomes, request)?;
    let mut result_shape = shape;
    result_shape.returned_rows = bounded_domains
        .checked_mul(request.objective_indices.len())
        .ok_or_else(|| invalid("bounded-domain returned rows overflow usize"))?;
    validate_wave_receipt(
        &receipt,
        request,
        result_shape,
        transcript,
        open_memory,
        true,
        policy,
    )?;
    if rows.len() != result_shape.returned_rows {
        return Err(invalid(format!(
            "completed result rows {} != bounded D * R ({})",
            rows.len(),
            result_shape.returned_rows
        )));
    }
    let r = request.objective_indices.len();
    let mut tightened_rows = 0usize;
    let mut validated = Vec::with_capacity(rows.len());
    let mut row_cursor = 0usize;
    for (domain_index, outcome) in validated_outcomes.iter().enumerate() {
        if outcome.kind != GpuBabBoundValidatedDomainOutcomeKind::Bounded {
            continue;
        }
        for objective_offset in 0..r {
            let q = domain_index
                .checked_mul(r)
                .and_then(|value| value.checked_add(objective_offset))
                .ok_or_else(|| invalid("canonical q overflows usize"))?;
            if row_cursor.is_multiple_of(VALIDATION_POLL_STRIDE) {
                check_live(request.deadline, "completed row validation")?;
            }
            let row = rows
                .get(row_cursor)
                .ok_or_else(|| invalid("completed bounded row cursor is missing"))?;
            let domain = &request.domains[domain_index];
            let objective = request.objective_indices[objective_offset];
            let domain_identity = outcome.domain_identity_sha256;
            if row.q != q as u32
                || row.parent_group_id != domain.parent_group_id
                || row.child_ordinal != domain.child_ordinal
                || row.child_cardinality != domain.child_cardinality
                || row.domain_slot != domain.domain_slot
                || row.domain_identity_sha256 != domain_identity
                || row.objective_index != objective
            {
                return Err(invalid(format!(
                "result row {q} does not exactly echo q/parent/child/domain/objective association"
            )));
            }
            if row.status != 0 || row.taint != 0 {
                return Err(invalid(format!(
                    "result row {q} has nonzero status {} or taint {}",
                    row.status, row.taint
                )));
            }
            validate_interval(row.lower, row.upper, "result", q)?;
            if row.lower > request.inherited_upper[q] || row.upper < request.inherited_lower[q] {
                return Err(invalid(format!(
                    "result row {q} is disjoint from its inherited interval"
                )));
            }
            if row.lower > request.inherited_lower[q] || row.upper < request.inherited_upper[q] {
                tightened_rows = tightened_rows
                    .checked_add(1)
                    .ok_or_else(|| invalid("tightened row count overflows usize"))?;
            }
            validated.push(GpuBabBoundValidatedRow {
                parent_group_id: row.parent_group_id,
                child_ordinal: row.child_ordinal,
                child_cardinality: row.child_cardinality,
                domain_slot: row.domain_slot,
                domain_identity_sha256: row.domain_identity_sha256,
                objective_index: row.objective_index,
                q: row.q,
                lower: row.lower,
                upper: row.upper,
            });
            row_cursor += 1;
        }
    }
    if row_cursor != rows.len() {
        return Err(invalid(
            "completed result contains rows for a pruned domain",
        ));
    }
    if receipt.tightened_rows != tightened_rows {
        return Err(invalid(format!(
            "receipt tightened rows {} != validated {tightened_rows}",
            receipt.tightened_rows
        )));
    }
    check_live(request.deadline, "completed result")?;
    Ok(ValidatedGpuBabBoundWaveResult {
        domain_outcomes: validated_outcomes,
        rows: validated,
        receipt: GpuBabBoundValidatedWaveReceipt { raw: receipt },
    })
}

/// Allocation-free completed-result validator used by retained v2.
///
/// The raw vectors remain owned by the accepted disposition and are moved
/// unchanged into a private validated wrapper only after every association,
/// interval, row, receipt, and deadline check below succeeds.
fn validate_failure_receipt(
    receipt: &GpuBabBoundBackendWaveReceipt,
    request: &GpuBabBoundWaveRequest,
    shape: ValidatedWaveShape,
    transcript: GpuBabBoundTerminalTranscript,
    open_memory: GpuBabBoundMemoryReceipt,
    policy: GpuBabBoundPhasePolicy,
) -> Result<()> {
    validate_wave_receipt(
        receipt,
        request,
        shape,
        transcript,
        open_memory,
        false,
        policy,
    )
}

fn validate_wave_receipt(
    receipt: &GpuBabBoundBackendWaveReceipt,
    request: &GpuBabBoundWaveRequest,
    shape: ValidatedWaveShape,
    transcript: GpuBabBoundTerminalTranscript,
    open_memory: GpuBabBoundMemoryReceipt,
    completed: bool,
    policy: GpuBabBoundPhasePolicy,
) -> Result<()> {
    if receipt.transcript != transcript {
        return Err(invalid(
            "terminal receipt does not exactly echo backend/gen/nonce/phase/schedule/endpoints/deadline/cap",
        ));
    }
    let groups = request.parent_groups.len();
    let domains = request.domains.len();
    let objectives = request.objective_indices.len();
    let subchunks = request.subchunks.len();
    if receipt.requested_parent_groups != groups
        || receipt.requested_domains != domains
        || receipt.objective_rows != objectives
        || receipt.requested_rows != shape.rows
        || receipt.requested_subchunks != subchunks
    {
        return Err(invalid(
            "terminal requested counts do not exactly echo groups/D/R/D*R/subchunks",
        ));
    }
    if completed {
        let bounded_domains = shape
            .returned_rows
            .checked_div(objectives)
            .ok_or_else(|| invalid("completed objective count is zero"))?;
        let pruned_domains = domains
            .checked_sub(bounded_domains)
            .ok_or_else(|| invalid("bounded domains exceed requested domains"))?;
        if receipt.completed_parent_groups != groups
            || receipt.completed_domains != domains
            || receipt.completed_rows != shape.rows
            || receipt.completed_subchunks != subchunks
            || receipt.bounded_domains != bounded_domains
            || receipt.pruned_domains != pruned_domains
            || receipt.returned_rows != shape.returned_rows
        {
            return Err(invalid("completed receipt contains a partial count"));
        }
    } else {
        if receipt.completed_parent_groups != 0
            || receipt.completed_domains != 0
            || receipt.completed_rows != 0
            || receipt.completed_subchunks != 0
            || receipt.bounded_domains != 0
            || receipt.pruned_domains != 0
            || receipt.returned_rows != 0
        {
            return Err(invalid(
                "failure receipt cannot claim completed result prefixes; dispatch/transfer counters record attempted work",
            ));
        }
    }
    if completed {
        if receipt.tightened_rows > receipt.completed_rows {
            return Err(invalid("tightened rows exceed completed rows"));
        }
    } else if receipt.tightened_rows != 0 {
        return Err(invalid(
            "failure receipt cannot claim unreturned tightened rows",
        ));
    }
    if receipt.authorized_device_bytes != request.max_device_bytes {
        return Err(invalid("terminal receipt cap does not echo the request"));
    }
    if receipt.memory.retained_graph_bytes != open_memory.retained_graph_bytes
        || receipt.memory.retained_phase_bytes != open_memory.retained_phase_bytes
    {
        return Err(invalid(
            "wave receipt does not carry forward exact retained open allocations",
        ));
    }
    receipt.memory.validate_peak(request.max_device_bytes)?;
    receipt.transfers.validate_equations(shape, completed)?;
    if receipt.memory.queued_upload_bytes != receipt.transfers.host_to_device_bytes
        || receipt.memory.result_readback_bytes != receipt.transfers.device_to_host_bytes
    {
        return Err(invalid(
            "memory upload/readback allocations do not equal transfer totals",
        ));
    }
    if receipt.waves != 1
        || receipt.dispatches > shape.required_dispatches
        || receipt.dispatches > policy.maximum_dispatches_per_wave
        || receipt.submits > receipt.dispatches
        || receipt.submits > policy.maximum_submits_per_wave
    {
        return Err(invalid(
            "accepted receipt exceeds the finite request/policy dispatch or submit equation",
        ));
    }
    if completed {
        if receipt.memory.wave_working_bytes == 0
            || receipt.transfers.host_to_device_bytes == 0
            || receipt.transfers.device_to_host_bytes == 0
            || receipt.dispatches == 0
            || receipt.submits == 0
            || receipt.dispatches != shape.required_dispatches
        {
            return Err(invalid(
                "completed receipt requires nonzero working/upload/readback/dispatch/submit work",
            ));
        }
    } else if receipt.dispatches == 0 {
        if receipt.submits != 0
            || receipt.transfers.synchronizations != 0
            || receipt.transfers.device_to_host_bytes != 0
            || receipt.transfers.result_endpoint_bytes != 0
            || receipt.transfers.result_sidecar_bytes != 0
            || receipt.transfers.domain_outcome_sidecar_bytes != 0
            || receipt.transfers.readbacks != 0
            || receipt.memory.result_readback_bytes != 0
        {
            return Err(invalid(
                "predispatch failure cannot report submits, synchronization, result D2H, or readback allocation",
            ));
        }
    } else if receipt.submits == 0
        || receipt.transfers.synchronizations != 1
        || receipt.memory.wave_working_bytes == 0
        || receipt.transfers.activation_operand_bytes != shape.activation_operand_bytes
        || receipt.transfers.beta_operand_bytes != shape.beta_operand_bytes
        || receipt.transfers.abs_operand_bytes != shape.abs_operand_bytes
        || receipt.transfers.box_operand_bytes != shape.box_operand_bytes
        || receipt.transfers.cached_la_operand_bytes != shape.cached_la_operand_bytes
        || receipt.transfers.inherited_endpoint_bytes != shape.inherited_endpoint_bytes
        || receipt.transfers.objective_index_bytes != shape.objective_index_bytes
        || receipt.transfers.subchunk_descriptor_bytes != shape.subchunk_descriptor_bytes
    {
        return Err(invalid(
            "postdispatch failure requires full typed inputs, working memory, a submit, and one completion synchronization",
        ));
    }
    Ok(())
}

fn core_predispatch_failure_receipt(
    request: &GpuBabBoundWaveRequest,
    shape: ValidatedWaveShape,
    transcript: GpuBabBoundTerminalTranscript,
    open_memory: GpuBabBoundMemoryReceipt,
) -> GpuBabBoundBackendWaveReceipt {
    GpuBabBoundBackendWaveReceipt {
        transcript,
        requested_parent_groups: request.parent_groups.len(),
        completed_parent_groups: 0,
        requested_domains: request.domains.len(),
        completed_domains: 0,
        bounded_domains: 0,
        pruned_domains: 0,
        objective_rows: request.objective_indices.len(),
        requested_rows: shape.rows,
        completed_rows: 0,
        returned_rows: 0,
        requested_subchunks: request.subchunks.len(),
        completed_subchunks: 0,
        authorized_device_bytes: request.max_device_bytes,
        memory: open_memory,
        transfers: GpuBabBoundTransferReceipt::default(),
        dispatches: 0,
        submits: 0,
        waves: 1,
        tightened_rows: 0,
    }
}

fn validate_interval(lower: f32, upper: f32, label: &str, q: usize) -> Result<()> {
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(invalid(format!(
            "{label} row {q} is not a finite ordered interval"
        )));
    }
    Ok(())
}

fn check_live(deadline: Instant, label: &str) -> Result<()> {
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(format!(
            "GPU BaB resident bound {label} deadline expired"
        )));
    }
    Ok(())
}

fn is_zero_identity(identity: [u8; 32]) -> bool {
    identity == [0; 32]
}

fn invalid(message: impl Into<String>) -> NyError {
    NyError::InvalidSpec(format!("GPU BaB resident bound phase: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
            Arc, Barrier,
        },
        time::Duration,
    };

    use super::*;

    static NEXT_ISSUER: AtomicU64 = AtomicU64::new(1);

    struct PanickingDropPayload;

    impl Drop for PanickingDropPayload {
        fn drop(&mut self) {
            panic!("hostile panic payload destructor must be quarantined");
        }
    }

    fn panic_with_hostile_payload() -> ! {
        std::panic::panic_any(PanickingDropPayload)
    }

    fn mutate_owned_slice<T: Clone>(
        source: &GpuBabBoundOwnedSlice<T>,
        mutate: impl FnOnce(&mut [T]),
    ) -> GpuBabBoundOwnedSlice<T> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(source.len())
            .expect("test owned-slice copy reserve");
        values.extend_from_slice(source.as_ref());
        mutate(values.as_mut_slice());
        GpuBabBoundOwnedSlice::new(values)
    }

    fn copy_owned_slice<T: Clone>(source: &[T]) -> GpuBabBoundOwnedSlice<T> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(source.len())
            .expect("test owned-slice copy reserve");
        values.extend_from_slice(source);
        GpuBabBoundOwnedSlice::new(values)
    }

    #[test]
    fn owned_slice_moves_reserved_storage_and_preserves_slice_identity_traits() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut values = Vec::new();
        values
            .try_reserve_exact(8)
            .expect("test owned-slice reserve");
        values.extend([3_u32, 5, 8]);
        let original_pointer = values.as_ptr();
        let original_capacity = values.capacity();

        let owned = GpuBabBoundOwnedSlice::new(values);
        assert_eq!(owned.as_ptr(), original_pointer);
        assert_eq!(owned.capacity(), original_capacity);
        assert_eq!(owned.len(), 3);
        assert!(!owned.is_empty());
        assert_eq!(owned.as_ref(), &[3, 5, 8]);
        assert_eq!(owned[1], 5);
        assert_eq!(format!("{owned:?}"), "[3, 5, 8]");
        assert_eq!(
            owned.accountable_bytes(),
            original_capacity
                .checked_mul(size_of::<u32>())
                .and_then(|bytes| bytes.checked_add(GPU_BAB_BOUND_OWNED_SLICE_FIXED_CHARGED_BYTES))
        );

        let cloned = owned.clone();
        assert_eq!(cloned.as_ptr(), original_pointer);
        assert_eq!(cloned, owned);

        let same_values = copy_owned_slice(&[3_u32, 5, 8]);
        assert_eq!(
            owned, same_values,
            "capacity is not part of payload identity"
        );
        let mut owned_hash = DefaultHasher::new();
        owned.hash(&mut owned_hash);
        let mut slice_hash = DefaultHasher::new();
        owned.as_slice().hash(&mut slice_hash);
        assert_eq!(owned_hash.finish(), slice_hash.finish());

        let empty = GpuBabBoundOwnedSlice::<u8>::default();
        assert!(empty.is_empty());
        assert_eq!(
            empty.accountable_bytes(),
            Some(GPU_BAB_BOUND_OWNED_SLICE_FIXED_CHARGED_BYTES)
        );

        struct NotClone(u8);
        let nonclone = GpuBabBoundOwnedSlice::new(vec![NotClone(7)]);
        let nonclone_copy = nonclone.clone();
        assert_eq!(nonclone[0].0, 7);
        assert_eq!(nonclone_copy[0].0, 7);
    }

    #[test]
    fn canonical_schema_accounts_all_eighteen_owned_slice_headers() {
        let phase = phase();
        let wave = request();
        let history = GpuBabBoundSplitHistoryArena::new(Vec::new());
        let plan = phase.plan();
        let dynamic_arenas = [
            &wave.domain_arena.activation,
            &wave.domain_arena.beta,
            &wave.domain_arena.abs,
            &wave.domain_arena.box_lower,
            &wave.domain_arena.box_upper,
            &wave.domain_arena.cached_la,
        ];

        let header_count =
            1 + plan.f32_tensors.len() + plan.u32_tensors.len() + dynamic_arenas.len() + 1;
        assert_eq!(header_count, 18);

        let mut accountable_bytes = plan.topology_bytes.accountable_bytes().unwrap();
        let mut capacity_bytes = plan.topology_bytes.capacity();
        for tensor in &plan.f32_tensors {
            accountable_bytes += tensor.values.accountable_bytes().unwrap();
            capacity_bytes += tensor.values.capacity() * size_of::<f32>();
        }
        for tensor in &plan.u32_tensors {
            accountable_bytes += tensor.values.accountable_bytes().unwrap();
            capacity_bytes += tensor.values.capacity() * size_of::<u32>();
        }
        for arena in dynamic_arenas {
            accountable_bytes += arena.accountable_bytes().unwrap();
            capacity_bytes += arena.capacity() * size_of::<f32>();
        }
        accountable_bytes += history.accountable_bytes().unwrap();
        capacity_bytes += history.capacity() * size_of::<u32>();

        assert_eq!(
            accountable_bytes,
            capacity_bytes + header_count * GPU_BAB_BOUND_OWNED_SLICE_FIXED_CHARGED_BYTES
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum Corruption {
        RowPermutation,
        ParentSidecar,
        ChildOrdinal,
        DomainEcho,
        ObjectiveEcho,
        QSidecar,
        PartialRows,
        Nonfinite,
        Inverted,
        Disjoint,
        Status,
        Taint,
        BackendEcho,
        ScheduleEcho,
        EndpointEcho,
        DeadlineEcho,
        PartialCount,
        OverCap,
        MemoryOverflow,
        Readbacks,
        Synchronizations,
        CoefficientD2h,
        H2dEquation,
        D2hEquation,
        OperandEquation,
        ResultSidecarEquation,
        DomainOutcomeSidecarEquation,
        TighteningCount,
        PrunedCount,
        DispatchesMax,
        SubmitsMax,
        OutcomePermutation,
        OutcomeAssociation,
    }

    #[derive(Clone, Copy, Debug)]
    enum Mode {
        CompleteZero,
        CompleteOneTightening,
        AcceptedFailure,
        AcceptedFailurePostdispatch,
        AcceptedFailureBadEcho,
        AcceptedFailurePartialParent,
        AcceptedFailureD2hWithoutDispatch,
        AcceptedFailureTightening,
        AcceptedFailurePartialTypedUpload,
        AcceptedFailureFullPrefixZeroWork,
        AcceptedFailureDispatchesMax,
        AcceptedFailureSubmitsMax,
        IllegalPostacceptDecline,
        EarlyDeadline,
        WaitForDeadline,
        WaitThenComplete,
        ReplayWaveOne,
        PanicAfterAccept,
        Corrupt(Corruption),
    }

    #[derive(Clone, Copy)]
    enum OpenMode {
        Opened,
        BadTranscript,
        BadMemory,
        AcceptedFailure,
        AcceptedFailureZero,
        AcceptedFailureGraphOnly,
        EarlyDeadline,
        WaitForDeadline,
        Panic,
    }

    #[derive(Clone, Copy)]
    enum CloseMode {
        Closed,
        BadReceipt,
        AcceptedFailure,
        Panic,
    }

    #[derive(Clone)]
    struct TransitionGate {
        entered: Arc<Barrier>,
        resume: Arc<Barrier>,
    }

    impl TransitionGate {
        fn new() -> Self {
            Self {
                entered: Arc::new(Barrier::new(2)),
                resume: Arc::new(Barrier::new(2)),
            }
        }

        fn block(&self) {
            self.entered.wait();
            self.resume.wait();
        }
    }

    #[derive(Clone)]
    struct DropReentry {
        registration: Arc<GpuBabBoundBackendRegistration>,
        nested_terminal: Arc<AtomicBool>,
        nested_accepted_open_calls: Arc<AtomicUsize>,
    }

    struct FakeBackend {
        registration: Arc<GpuBabBoundBackendRegistration>,
        modes: Mutex<Vec<Mode>>,
        decline_prepare: bool,
        panic_prepare: bool,
        wait_prepare_deadline: bool,
        decline_phase_prepare: bool,
        panic_phase_policy: bool,
        invalid_phase_policy: bool,
        panic_phase_prepare: bool,
        wait_phase_policy_deadline: bool,
        wait_phase_prepare_deadline: bool,
        sound_gpu_crown: bool,
        sound_bab_phase: bool,
        open_mode: OpenMode,
        close_mode: CloseMode,
        accepted_open_calls: Arc<AtomicUsize>,
        phase_policy_calls: Arc<AtomicUsize>,
        phase_prepare_calls: Arc<AtomicUsize>,
        prepare_wave_calls: Arc<AtomicUsize>,
        execute_calls: Arc<AtomicUsize>,
        close_calls: Arc<AtomicUsize>,
        session_drop_calls: Arc<AtomicUsize>,
        panic_session_drop: bool,
        session_drop_reentry: Option<DropReentry>,
        execute_gate: Option<TransitionGate>,
    }

    impl FakeBackend {
        fn new(modes: Vec<Mode>) -> Self {
            Self {
                registration: fresh_registration(),
                modes: Mutex::new(modes),
                decline_prepare: false,
                panic_prepare: false,
                wait_prepare_deadline: false,
                decline_phase_prepare: false,
                panic_phase_policy: false,
                invalid_phase_policy: false,
                panic_phase_prepare: false,
                wait_phase_policy_deadline: false,
                wait_phase_prepare_deadline: false,
                sound_gpu_crown: true,
                sound_bab_phase: true,
                open_mode: OpenMode::Opened,
                close_mode: CloseMode::Closed,
                accepted_open_calls: Arc::new(AtomicUsize::new(0)),
                phase_policy_calls: Arc::new(AtomicUsize::new(0)),
                phase_prepare_calls: Arc::new(AtomicUsize::new(0)),
                prepare_wave_calls: Arc::new(AtomicUsize::new(0)),
                execute_calls: Arc::new(AtomicUsize::new(0)),
                close_calls: Arc::new(AtomicUsize::new(0)),
                session_drop_calls: Arc::new(AtomicUsize::new(0)),
                panic_session_drop: false,
                session_drop_reentry: None,
                execute_gate: None,
            }
        }

        fn with_registration(
            registration: Arc<GpuBabBoundBackendRegistration>,
            modes: Vec<Mode>,
        ) -> Self {
            Self {
                registration,
                modes: Mutex::new(modes),
                decline_prepare: false,
                panic_prepare: false,
                wait_prepare_deadline: false,
                decline_phase_prepare: false,
                panic_phase_policy: false,
                invalid_phase_policy: false,
                panic_phase_prepare: false,
                wait_phase_policy_deadline: false,
                wait_phase_prepare_deadline: false,
                sound_gpu_crown: true,
                sound_bab_phase: true,
                open_mode: OpenMode::Opened,
                close_mode: CloseMode::Closed,
                accepted_open_calls: Arc::new(AtomicUsize::new(0)),
                phase_policy_calls: Arc::new(AtomicUsize::new(0)),
                phase_prepare_calls: Arc::new(AtomicUsize::new(0)),
                prepare_wave_calls: Arc::new(AtomicUsize::new(0)),
                execute_calls: Arc::new(AtomicUsize::new(0)),
                close_calls: Arc::new(AtomicUsize::new(0)),
                session_drop_calls: Arc::new(AtomicUsize::new(0)),
                panic_session_drop: false,
                session_drop_reentry: None,
                execute_gate: None,
            }
        }

        fn with_open_mode(open_mode: OpenMode) -> Self {
            Self {
                registration: fresh_registration(),
                modes: Mutex::new(Vec::new()),
                decline_prepare: false,
                panic_prepare: false,
                wait_prepare_deadline: false,
                decline_phase_prepare: false,
                panic_phase_policy: false,
                invalid_phase_policy: false,
                panic_phase_prepare: false,
                wait_phase_policy_deadline: false,
                wait_phase_prepare_deadline: false,
                sound_gpu_crown: true,
                sound_bab_phase: true,
                open_mode,
                close_mode: CloseMode::Closed,
                accepted_open_calls: Arc::new(AtomicUsize::new(0)),
                phase_policy_calls: Arc::new(AtomicUsize::new(0)),
                phase_prepare_calls: Arc::new(AtomicUsize::new(0)),
                prepare_wave_calls: Arc::new(AtomicUsize::new(0)),
                execute_calls: Arc::new(AtomicUsize::new(0)),
                close_calls: Arc::new(AtomicUsize::new(0)),
                session_drop_calls: Arc::new(AtomicUsize::new(0)),
                panic_session_drop: false,
                session_drop_reentry: None,
                execute_gate: None,
            }
        }

        fn accepted_open_calls(&self) -> usize {
            self.accepted_open_calls.load(Ordering::Relaxed)
        }

        fn prepare_wave_calls(&self) -> usize {
            self.prepare_wave_calls.load(Ordering::Relaxed)
        }

        fn phase_policy_calls(&self) -> usize {
            self.phase_policy_calls.load(Ordering::Relaxed)
        }

        fn phase_prepare_calls(&self) -> usize {
            self.phase_prepare_calls.load(Ordering::Relaxed)
        }

        fn execute_calls(&self) -> usize {
            self.execute_calls.load(Ordering::Relaxed)
        }
    }

    impl GpuCrownBackward for FakeBackend {
        fn provides_sound_gpu_crown(&self) -> bool {
            self.sound_gpu_crown
        }

        fn provides_sound_gpu_bab_bound_phase(&self) -> bool {
            self.sound_bab_phase
        }

        fn gpu_bab_bound_numerical_tcb(&self) -> Option<&dyn GpuBabBoundNumericalTcb> {
            Some(self)
        }

        fn crown_backward_gpu(
            &self,
            _layers: &[super::super::GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<super::super::GpuCrownResult> {
            unreachable!("contract tests never execute legacy CROWN")
        }
    }

    impl GpuBabBoundNumericalTcb for FakeBackend {
        fn registration(&self) -> &GpuBabBoundBackendRegistration {
            self.registration.as_ref()
        }

        fn phase_policy(
            &self,
            invocation: &GpuBabBoundTcbInvocation<'_>,
        ) -> Option<GpuBabBoundPhasePolicy> {
            self.phase_policy_calls.fetch_add(1, Ordering::Relaxed);
            if self.panic_phase_policy {
                panic_with_hostile_payload();
            }
            if self.wait_phase_policy_deadline {
                while Instant::now() < invocation.descriptor().deadline() {
                    std::hint::spin_loop();
                }
            }
            let mut policy = GpuBabBoundPhasePolicy {
                max_device_bytes: 8_192,
                preferred_domains_per_wave: 4,
                minimum_domains_per_wave: 1,
                maximum_domains_per_wave: GPU_BAB_BOUND_MAX_DOMAINS,
                maximum_objectives: GPU_BAB_BOUND_MAX_OBJECTIVES,
                maximum_dispatches_per_wave: GPU_BAB_BOUND_MAX_DISPATCHES_PER_WAVE,
                maximum_submits_per_wave: GPU_BAB_BOUND_MAX_SUBMITS_PER_WAVE,
            };
            if self.invalid_phase_policy {
                policy.maximum_dispatches_per_wave = usize::MAX;
            }
            Some(policy)
        }

        fn prepare_phase<'a>(
            &'a self,
            invocation: &GpuBabBoundTcbInvocation<'_>,
        ) -> GpuBabBoundBackendOpenPreparation<'a> {
            self.phase_prepare_calls.fetch_add(1, Ordering::Relaxed);
            if self.panic_phase_prepare {
                panic_with_hostile_payload();
            }
            if self.wait_phase_prepare_deadline {
                while Instant::now() < invocation.descriptor().deadline() {
                    std::hint::spin_loop();
                }
            }
            if self.decline_phase_prepare {
                return GpuBabBoundBackendOpenPreparation::CleanDecline(
                    GpuBabBoundPhaseDecline::InsufficientCapacity,
                );
            }
            GpuBabBoundBackendOpenPreparation::Prepared {
                session: Box::new(FakeSession {
                    phase: invocation.descriptor().clone(),
                    transcript: None,
                    open_memory: open_memory(),
                    modes: std::mem::take(
                        &mut *self
                            .modes
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner),
                    ),
                    decline_prepare: self.decline_prepare,
                    panic_prepare: self.panic_prepare,
                    wait_prepare_deadline: self.wait_prepare_deadline,
                    open_mode: self.open_mode,
                    close_mode: self.close_mode,
                    accepted_open_calls: Arc::clone(&self.accepted_open_calls),
                    prepare_wave_calls: Arc::clone(&self.prepare_wave_calls),
                    execute_calls: Arc::clone(&self.execute_calls),
                    close_calls: Arc::clone(&self.close_calls),
                    session_drop_calls: Arc::clone(&self.session_drop_calls),
                    panic_session_drop: self.panic_session_drop,
                    session_drop_reentry: self.session_drop_reentry.clone(),
                    execute_gate: self.execute_gate.clone(),
                }),
            }
        }
    }

    struct FakeSession {
        phase: GpuBabBoundPhaseDescriptor,
        transcript: Option<GpuBabBoundPhaseTranscript>,
        open_memory: GpuBabBoundMemoryReceipt,
        modes: Vec<Mode>,
        decline_prepare: bool,
        panic_prepare: bool,
        wait_prepare_deadline: bool,
        open_mode: OpenMode,
        close_mode: CloseMode,
        accepted_open_calls: Arc<AtomicUsize>,
        prepare_wave_calls: Arc<AtomicUsize>,
        execute_calls: Arc<AtomicUsize>,
        close_calls: Arc<AtomicUsize>,
        session_drop_calls: Arc<AtomicUsize>,
        panic_session_drop: bool,
        session_drop_reentry: Option<DropReentry>,
        execute_gate: Option<TransitionGate>,
    }

    impl GpuBabBoundBackendSession for FakeSession {
        fn open_accepted(
            &mut self,
            accepted: &GpuBabBoundAcceptedOpen<'_>,
        ) -> GpuBabBoundBackendOpen {
            self.accepted_open_calls.fetch_add(1, Ordering::Relaxed);
            self.transcript = Some(accepted.transcript());
            let mut memory = self.open_memory;
            let mut static_transfers = fresh_static_transfers(
                accepted.transcript(),
                memory.retained_graph_bytes,
                memory.retained_phase_bytes,
            );
            memory.queued_upload_bytes = static_transfers.host_to_device_bytes;
            memory.peak_device_bytes = memory
                .retained_graph_bytes
                .checked_add(memory.retained_phase_bytes)
                .and_then(|bytes| bytes.checked_add(memory.queued_upload_bytes))
                .unwrap();
            let mut receipt = GpuBabBoundBackendOpenReceipt {
                transcript: accepted.transcript(),
                authorized_device_bytes: accepted.descriptor().max_device_bytes,
                memory,
                static_transfers,
                released_graph_bytes: 0,
                released_phase_bytes: 0,
            };
            match self.open_mode {
                OpenMode::AcceptedFailure => {
                    self.open_memory = receipt.memory;
                    GpuBabBoundBackendOpen::AcceptedFailure {
                        kind: GpuBabBoundBackendOpenFailureKind::Device,
                        detail: "accepted open failure".into(),
                        receipt,
                    }
                }
                OpenMode::AcceptedFailureZero => {
                    receipt.memory = GpuBabBoundMemoryReceipt {
                        retained_graph_bytes: 0,
                        retained_phase_bytes: 0,
                        wave_working_bytes: 0,
                        queued_upload_bytes: 0,
                        result_readback_bytes: 0,
                        peak_device_bytes: 0,
                    };
                    receipt.static_transfers = not_transferred_static_receipt(receipt.transcript);
                    self.open_memory = receipt.memory;
                    GpuBabBoundBackendOpen::AcceptedFailure {
                        kind: GpuBabBoundBackendOpenFailureKind::Allocation,
                        detail: "accepted open failed before allocation".into(),
                        receipt,
                    }
                }
                OpenMode::AcceptedFailureGraphOnly => {
                    receipt.memory.retained_phase_bytes = 0;
                    static_transfers.phase_source = GpuBabBoundStaticPayloadSource::NotTransferred;
                    static_transfers.phase_padding_bytes = 0;
                    static_transfers.phase_host_to_device_bytes = 0;
                    static_transfers.host_to_device_bytes =
                        static_transfers.graph_host_to_device_bytes;
                    receipt.static_transfers = static_transfers;
                    receipt.memory.queued_upload_bytes =
                        receipt.static_transfers.host_to_device_bytes;
                    receipt.memory.peak_device_bytes = receipt
                        .memory
                        .retained_graph_bytes
                        .checked_add(receipt.memory.queued_upload_bytes)
                        .unwrap();
                    self.open_memory = receipt.memory;
                    GpuBabBoundBackendOpen::AcceptedFailure {
                        kind: GpuBabBoundBackendOpenFailureKind::Allocation,
                        detail: "accepted open failed after graph allocation".into(),
                        receipt,
                    }
                }
                OpenMode::EarlyDeadline => {
                    self.open_memory = receipt.memory;
                    GpuBabBoundBackendOpen::DeadlineExpired {
                        detail: "early open deadline".into(),
                        receipt,
                    }
                }
                OpenMode::WaitForDeadline => {
                    while Instant::now() < accepted.descriptor().deadline {
                        std::hint::spin_loop();
                    }
                    GpuBabBoundBackendOpen::Opened { receipt }
                }
                OpenMode::BadTranscript => {
                    receipt.transcript.graph_identity_sha256 = hash(248);
                    GpuBabBoundBackendOpen::Opened { receipt }
                }
                OpenMode::BadMemory => {
                    memory.wave_working_bytes = 1;
                    memory.peak_device_bytes += 1;
                    receipt.memory = memory;
                    GpuBabBoundBackendOpen::Opened { receipt }
                }
                OpenMode::Opened => GpuBabBoundBackendOpen::Opened { receipt },
                OpenMode::Panic => panic_with_hostile_payload(),
            }
        }

        fn prepare_wave(
            &mut self,
            prepared: &GpuBabBoundPreparedWave<'_>,
        ) -> GpuBabBoundBackendPrepareDisposition {
            self.prepare_wave_calls.fetch_add(1, Ordering::Relaxed);
            if self.panic_prepare {
                panic_with_hostile_payload();
            }
            if self.wait_prepare_deadline {
                while Instant::now() < prepared.request().deadline {
                    std::hint::spin_loop();
                }
            }
            if self.decline_prepare {
                GpuBabBoundBackendPrepareDisposition::CleanDecline(
                    GpuBabBoundWaveDecline::InsufficientCapacity,
                )
            } else {
                GpuBabBoundBackendPrepareDisposition::Accepted
            }
        }

        fn execute_accepted(
            &mut self,
            accepted: &GpuBabBoundAcceptedWave<'_>,
        ) -> GpuBabBoundBackendWaveDisposition {
            self.execute_calls.fetch_add(1, Ordering::Relaxed);
            if let Some(gate) = &self.execute_gate {
                gate.block();
            }
            let mode = if self.modes.is_empty() {
                Mode::CompleteZero
            } else {
                self.modes.remove(0)
            };
            match mode {
                Mode::CompleteZero => completed_raw(accepted, &self.phase, self.open_memory, None),
                Mode::CompleteOneTightening => {
                    completed_raw(accepted, &self.phase, self.open_memory, Some(0))
                }
                Mode::AcceptedFailure => GpuBabBoundBackendWaveDisposition::AcceptedFailure {
                    kind: GpuBabBoundBackendFailureKind::Device,
                    detail: "accepted device loss".into(),
                    receipt: failure_receipt(accepted, &self.phase, self.open_memory),
                },
                Mode::AcceptedFailurePostdispatch => {
                    let mut receipt = failure_receipt(accepted, &self.phase, self.open_memory);
                    let shape = accepted.request().validate_static(&self.phase).unwrap();
                    receipt.transfers.activation_operand_bytes = shape.activation_operand_bytes;
                    receipt.transfers.beta_operand_bytes = shape.beta_operand_bytes;
                    receipt.transfers.abs_operand_bytes = shape.abs_operand_bytes;
                    receipt.transfers.box_operand_bytes = shape.box_operand_bytes;
                    receipt.transfers.cached_la_operand_bytes = shape.cached_la_operand_bytes;
                    receipt.transfers.domain_operand_bytes = shape.domain_operand_bytes;
                    receipt.transfers.inherited_endpoint_bytes = shape.inherited_endpoint_bytes;
                    receipt.transfers.objective_index_bytes = shape.objective_index_bytes;
                    receipt.transfers.subchunk_descriptor_bytes = shape.subchunk_descriptor_bytes;
                    receipt.transfers.host_to_device_bytes = shape.domain_operand_bytes
                        + shape.inherited_endpoint_bytes
                        + shape.objective_index_bytes
                        + shape.subchunk_descriptor_bytes;
                    receipt.transfers.synchronizations = 1;
                    receipt.dispatches = 1;
                    receipt.submits = 1;
                    receipt.memory.wave_working_bytes = 512;
                    receipt.memory.queued_upload_bytes = receipt.transfers.host_to_device_bytes;
                    receipt.memory.peak_device_bytes = receipt.memory.retained_graph_bytes
                        + receipt.memory.retained_phase_bytes
                        + receipt.memory.wave_working_bytes
                        + receipt.memory.queued_upload_bytes;
                    GpuBabBoundBackendWaveDisposition::AcceptedFailure {
                        kind: GpuBabBoundBackendFailureKind::Device,
                        detail: "accepted postdispatch device loss".into(),
                        receipt,
                    }
                }
                Mode::AcceptedFailureBadEcho => {
                    let mut receipt = failure_receipt(accepted, &self.phase, self.open_memory);
                    receipt.transcript.schedule_identity_sha256 = hash(249);
                    GpuBabBoundBackendWaveDisposition::AcceptedFailure {
                        kind: GpuBabBoundBackendFailureKind::Device,
                        detail: "bad failure echo".into(),
                        receipt,
                    }
                }
                Mode::AcceptedFailurePartialParent => {
                    let mut receipt = failure_receipt(accepted, &self.phase, self.open_memory);
                    receipt.completed_domains = 1;
                    receipt.completed_rows = accepted.request().objective_indices.len();
                    GpuBabBoundBackendWaveDisposition::AcceptedFailure {
                        kind: GpuBabBoundBackendFailureKind::Device,
                        detail: "partial parent".into(),
                        receipt,
                    }
                }
                Mode::AcceptedFailureD2hWithoutDispatch => {
                    let mut receipt = failure_receipt(accepted, &self.phase, self.open_memory);
                    let endpoint_bytes =
                        accepted.request().row_count().unwrap() * ENDPOINT_BYTES_PER_ROW;
                    receipt.transfers.result_endpoint_bytes = endpoint_bytes;
                    receipt.transfers.device_to_host_bytes = endpoint_bytes;
                    receipt.transfers.readbacks = 1;
                    receipt.memory.result_readback_bytes = endpoint_bytes;
                    receipt.memory.peak_device_bytes += endpoint_bytes;
                    GpuBabBoundBackendWaveDisposition::AcceptedFailure {
                        kind: GpuBabBoundBackendFailureKind::Device,
                        detail: "impossible predispatch D2H".into(),
                        receipt,
                    }
                }
                Mode::AcceptedFailureTightening => {
                    let mut receipt = failure_receipt(accepted, &self.phase, self.open_memory);
                    receipt.tightened_rows = 1;
                    GpuBabBoundBackendWaveDisposition::AcceptedFailure {
                        kind: GpuBabBoundBackendFailureKind::Device,
                        detail: "unreturned tightening".into(),
                        receipt,
                    }
                }
                Mode::AcceptedFailurePartialTypedUpload => {
                    let mut receipt = failure_receipt(accepted, &self.phase, self.open_memory);
                    receipt.transfers.activation_operand_bytes = 1;
                    receipt.transfers.domain_operand_bytes = 1;
                    receipt.transfers.host_to_device_bytes = 1;
                    receipt.memory.queued_upload_bytes = 1;
                    receipt.memory.peak_device_bytes += 1;
                    GpuBabBoundBackendWaveDisposition::AcceptedFailure {
                        kind: GpuBabBoundBackendFailureKind::Device,
                        detail: "partial typed upload".into(),
                        receipt,
                    }
                }
                Mode::AcceptedFailureFullPrefixZeroWork => {
                    let mut receipt = failure_receipt(accepted, &self.phase, self.open_memory);
                    receipt.completed_parent_groups = 1;
                    receipt.completed_domains = 2;
                    receipt.completed_rows = 2 * accepted.request().objective_indices.len();
                    receipt.completed_subchunks = 1;
                    GpuBabBoundBackendWaveDisposition::AcceptedFailure {
                        kind: GpuBabBoundBackendFailureKind::Device,
                        detail: "completed prefix without attempted work".into(),
                        receipt,
                    }
                }
                Mode::AcceptedFailureDispatchesMax => {
                    let mut receipt = failure_receipt(accepted, &self.phase, self.open_memory);
                    receipt.dispatches = usize::MAX;
                    receipt.submits = 1;
                    receipt.transfers.synchronizations = 1;
                    GpuBabBoundBackendWaveDisposition::AcceptedFailure {
                        kind: GpuBabBoundBackendFailureKind::Device,
                        detail: "unbounded failure dispatch count".into(),
                        receipt,
                    }
                }
                Mode::AcceptedFailureSubmitsMax => {
                    let mut receipt = failure_receipt(accepted, &self.phase, self.open_memory);
                    receipt.submits = usize::MAX;
                    GpuBabBoundBackendWaveDisposition::AcceptedFailure {
                        kind: GpuBabBoundBackendFailureKind::Device,
                        detail: "unbounded failure submit count".into(),
                        receipt,
                    }
                }
                Mode::IllegalPostacceptDecline => {
                    GpuBabBoundBackendWaveDisposition::IllegalCleanDecline {
                        reason: GpuBabBoundWaveDecline::InsufficientCapacity,
                        receipt: failure_receipt(accepted, &self.phase, self.open_memory),
                    }
                }
                Mode::EarlyDeadline => GpuBabBoundBackendWaveDisposition::DeadlineExpired {
                    detail: "claimed early deadline".into(),
                    receipt: failure_receipt(accepted, &self.phase, self.open_memory),
                },
                Mode::WaitForDeadline => {
                    while Instant::now() < accepted.request().deadline {
                        std::hint::spin_loop();
                    }
                    GpuBabBoundBackendWaveDisposition::DeadlineExpired {
                        detail: "actual deadline".into(),
                        receipt: failure_receipt(accepted, &self.phase, self.open_memory),
                    }
                }
                Mode::WaitThenComplete => {
                    while Instant::now() < accepted.request().deadline {
                        std::hint::spin_loop();
                    }
                    completed_raw(accepted, &self.phase, self.open_memory, None)
                }
                Mode::ReplayWaveOne => {
                    let mut disposition =
                        completed_raw(accepted, &self.phase, self.open_memory, None);
                    if let GpuBabBoundBackendWaveDisposition::Completed { receipt, .. } =
                        &mut disposition
                    {
                        receipt.transcript.wave_index = 1;
                    }
                    disposition
                }
                Mode::PanicAfterAccept => panic_with_hostile_payload(),
                Mode::Corrupt(corruption) => {
                    corrupted_raw(accepted, &self.phase, self.open_memory, corruption)
                }
            }
        }

        fn close(&mut self) -> GpuBabBoundBackendCloseDisposition {
            self.close_calls.fetch_add(1, Ordering::Relaxed);
            let mut receipt = GpuBabBoundBackendCloseReceipt {
                transcript: self
                    .transcript
                    .expect("accepted fake open records a phase transcript"),
                released_graph_bytes: self.open_memory.retained_graph_bytes,
                released_phase_bytes: self.open_memory.retained_phase_bytes,
                released_resident_device_bytes: 0,
                released_resident_slots: 0,
                released_refresh_only_slots: 0,
                released_resident_logical_slots: 0,
            };
            match self.close_mode {
                CloseMode::Closed => GpuBabBoundBackendCloseDisposition::Closed(receipt),
                CloseMode::BadReceipt => {
                    receipt.released_phase_bytes = receipt.released_phase_bytes.saturating_add(1);
                    GpuBabBoundBackendCloseDisposition::Closed(receipt)
                }
                CloseMode::AcceptedFailure => GpuBabBoundBackendCloseDisposition::AcceptedFailure {
                    detail: "raw close failure".into(),
                    receipt,
                },
                CloseMode::Panic => panic_with_hostile_payload(),
            }
        }
    }

    impl Drop for FakeSession {
        fn drop(&mut self) {
            self.session_drop_calls.fetch_add(1, Ordering::Relaxed);
            if let Some(reentry) = &self.session_drop_reentry {
                let nested =
                    FakeBackend::with_registration(Arc::clone(&reentry.registration), vec![]);
                let open = GpuBabBoundPhaseLease::open(&nested, phase());
                reentry.nested_terminal.store(
                    matches!(open, GpuBabBoundPhaseOpen::AcceptedFailure(_)),
                    Ordering::Relaxed,
                );
                reentry
                    .nested_accepted_open_calls
                    .store(nested.accepted_open_calls(), Ordering::Relaxed);
            }
            if self.panic_session_drop {
                panic_with_hostile_payload();
            }
        }
    }

    struct ReentrantPreparedBackend {
        registration: Arc<GpuBabBoundBackendRegistration>,
        nested_terminal: Arc<AtomicBool>,
        nested_fallback: Arc<AtomicBool>,
        nested_accepted_open_calls: Arc<AtomicUsize>,
    }

    impl GpuCrownBackward for ReentrantPreparedBackend {
        fn provides_sound_gpu_crown(&self) -> bool {
            true
        }

        fn provides_sound_gpu_bab_bound_phase(&self) -> bool {
            true
        }

        fn gpu_bab_bound_numerical_tcb(&self) -> Option<&dyn GpuBabBoundNumericalTcb> {
            Some(self)
        }

        fn crown_backward_gpu(
            &self,
            _layers: &[super::super::GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<super::super::GpuCrownResult> {
            unreachable!("reentrant-drop test never executes legacy CROWN")
        }
    }

    impl GpuBabBoundNumericalTcb for ReentrantPreparedBackend {
        fn registration(&self) -> &GpuBabBoundBackendRegistration {
            self.registration.as_ref()
        }

        fn phase_policy(
            &self,
            _invocation: &GpuBabBoundTcbInvocation<'_>,
        ) -> Option<GpuBabBoundPhasePolicy> {
            Some(GpuBabBoundPhasePolicy {
                max_device_bytes: 8_192,
                preferred_domains_per_wave: 4,
                minimum_domains_per_wave: 1,
                maximum_domains_per_wave: GPU_BAB_BOUND_MAX_DOMAINS,
                maximum_objectives: GPU_BAB_BOUND_MAX_OBJECTIVES,
                maximum_dispatches_per_wave: GPU_BAB_BOUND_MAX_DISPATCHES_PER_WAVE,
                maximum_submits_per_wave: GPU_BAB_BOUND_MAX_SUBMITS_PER_WAVE,
            })
        }

        fn prepare_phase<'a>(
            &'a self,
            invocation: &GpuBabBoundTcbInvocation<'_>,
        ) -> GpuBabBoundBackendOpenPreparation<'a> {
            while Instant::now() < invocation.descriptor().deadline() {
                std::hint::spin_loop();
            }
            GpuBabBoundBackendOpenPreparation::Prepared {
                session: Box::new(ReentrantDropSession {
                    registration: Arc::clone(&self.registration),
                    nested_terminal: Arc::clone(&self.nested_terminal),
                    nested_fallback: Arc::clone(&self.nested_fallback),
                    nested_accepted_open_calls: Arc::clone(&self.nested_accepted_open_calls),
                }),
            }
        }
    }

    struct ReentrantDropSession {
        registration: Arc<GpuBabBoundBackendRegistration>,
        nested_terminal: Arc<AtomicBool>,
        nested_fallback: Arc<AtomicBool>,
        nested_accepted_open_calls: Arc<AtomicUsize>,
    }

    impl GpuBabBoundBackendSession for ReentrantDropSession {
        fn open_accepted(
            &mut self,
            _accepted: &GpuBabBoundAcceptedOpen<'_>,
        ) -> GpuBabBoundBackendOpen {
            unreachable!("late prepared session must remain dormant")
        }

        fn prepare_wave(
            &mut self,
            _prepared: &GpuBabBoundPreparedWave<'_>,
        ) -> GpuBabBoundBackendPrepareDisposition {
            unreachable!("late prepared session never owns a wave")
        }

        fn execute_accepted(
            &mut self,
            _accepted: &GpuBabBoundAcceptedWave<'_>,
        ) -> GpuBabBoundBackendWaveDisposition {
            unreachable!("late prepared session never executes")
        }

        fn close(&mut self) -> GpuBabBoundBackendCloseDisposition {
            unreachable!("dormant prepared session is destroyed without close")
        }
    }

    impl Drop for ReentrantDropSession {
        fn drop(&mut self) {
            let nested = FakeBackend::with_registration(Arc::clone(&self.registration), vec![]);
            let open = GpuBabBoundPhaseLease::open(&nested, phase());
            self.nested_fallback
                .store(open.permits_legacy_fallback(), Ordering::Relaxed);
            self.nested_terminal.store(
                matches!(open, GpuBabBoundPhaseOpen::AcceptedFailure(_)),
                Ordering::Relaxed,
            );
            self.nested_accepted_open_calls
                .store(nested.accepted_open_calls(), Ordering::Relaxed);
        }
    }

    struct DefaultDecliningBackend;

    impl GpuCrownBackward for DefaultDecliningBackend {
        fn crown_backward_gpu(
            &self,
            _layers: &[super::super::GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<super::super::GpuCrownResult> {
            unreachable!("default-decline test never executes CROWN")
        }
    }

    struct UnattestedRawBackend;

    impl GpuCrownBackward for UnattestedRawBackend {
        fn crown_backward_gpu(
            &self,
            _layers: &[super::super::GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<super::super::GpuCrownResult> {
            unreachable!("soundness gate test never executes legacy CROWN")
        }
    }

    fn fresh_registration() -> Arc<GpuBabBoundBackendRegistration> {
        let sequence = NEXT_ISSUER.fetch_add(1, Ordering::Relaxed);
        registration_with_id(sequence)
    }

    fn registration_with_id(id: u64) -> Arc<GpuBabBoundBackendRegistration> {
        let mut backend = [0xA5; 32];
        backend[..8].copy_from_slice(&id.to_le_bytes());
        Arc::new(GpuBabBoundBackendRegistration::new(backend).unwrap())
    }

    fn hash(value: u8) -> [u8; 32] {
        [value; 32]
    }

    struct StaticSchedulePayload {
        topology_schema_version: u32,
        topology_bytes: Vec<u8>,
        f32_tensors: Vec<GpuBabBoundF32Tensor>,
        u32_tensors: Vec<GpuBabBoundU32Tensor>,
        identity: [u8; 32],
    }

    impl StaticSchedulePayload {
        fn request(
            &self,
            deadline: Instant,
            requested_max_device_bytes: usize,
        ) -> GpuBabBoundStaticScheduleRequest<'_> {
            GpuBabBoundStaticScheduleRequest::new(
                self.topology_schema_version,
                &self.topology_bytes,
                &self.f32_tensors,
                &self.u32_tensors,
                self.identity,
                deadline,
                requested_max_device_bytes,
            )
            .unwrap()
        }
    }

    fn static_schedule_payload() -> StaticSchedulePayload {
        let topology_schema_version = 1;
        let topology_bytes = vec![1_u8, 2, 3, 4];
        let f32_tensors = vec![
            f32_tensor(
                GpuBabBoundF32TensorRole::Parameters,
                vec![4],
                vec![0.25, -0.5, 0.75, 1.0],
            ),
            f32_tensor(
                GpuBabBoundF32TensorRole::CertifiedErrors,
                vec![2],
                vec![0.0, 0.125],
            ),
            f32_tensor(GpuBabBoundF32TensorRole::Relaxations, vec![0], vec![]),
            f32_tensor(
                GpuBabBoundF32TensorRole::InputLower,
                vec![2],
                vec![-1.0, -1.0],
            ),
            f32_tensor(
                GpuBabBoundF32TensorRole::InputUpper,
                vec![2],
                vec![1.0, 1.0],
            ),
            f32_tensor(
                GpuBabBoundF32TensorRole::RootLower,
                vec![2],
                vec![-2.0, -2.0],
            ),
            f32_tensor(GpuBabBoundF32TensorRole::RootUpper, vec![2], vec![2.0, 2.0]),
            f32_tensor(
                GpuBabBoundF32TensorRole::ObjectiveCoefficients,
                vec![2, 2],
                vec![1.0, -1.0, -1.0, 1.0],
            ),
        ];
        let u32_tensors = vec![
            GpuBabBoundU32Tensor {
                role: GpuBabBoundU32TensorRole::ObjectiveIndices,
                shape: vec![2],
                values: GpuBabBoundOwnedSlice::new(vec![0, 1]),
            },
            GpuBabBoundU32Tensor {
                role: GpuBabBoundU32TensorRole::TopologyMetadata,
                shape: vec![0],
                values: GpuBabBoundOwnedSlice::new(Vec::new()),
            },
        ];
        let mut check = |_| Ok(());
        let identity = gpu_bab_bound_static_payload_identity_v1(
            topology_schema_version,
            &topology_bytes,
            &f32_tensors,
            &u32_tensors,
            &mut check,
        )
        .unwrap();
        StaticSchedulePayload {
            topology_schema_version,
            topology_bytes,
            f32_tensors,
            u32_tensors,
            identity,
        }
    }

    fn schedule_identity() -> GpuBabBoundBackendScheduleIdentity {
        GpuBabBoundBackendScheduleIdentity {
            schema_bundle_version: 1,
            provider_abi_sha256: hash(71),
            receipt_abi_sha256: hash(72),
            kernel_sha256: hash(73),
            topology_schema_sha256: hash(74),
            selfcheck_schema_sha256: hash(75),
            transcript_schema_sha256: hash(76),
        }
    }

    fn fresh_schedule_registration() -> Arc<GpuBabBoundBackendRegistration> {
        let sequence = NEXT_ISSUER.fetch_add(1, Ordering::Relaxed);
        let mut backend = [0xB6; 32];
        backend[..8].copy_from_slice(&sequence.to_le_bytes());
        Arc::new(
            GpuBabBoundBackendRegistration::new_with_schedule_identity(
                backend,
                schedule_identity(),
            )
            .unwrap(),
        )
    }

    #[derive(Clone, Copy, Debug)]
    enum ScheduleEvidenceMutation {
        BackendIssuer,
        RegistrationEpoch,
        StaticIdentityZero,
        StaticIdentityMismatch,
        TopologySchemaVersion,
        SchemaBundleVersion,
        ProviderAbi,
        ReceiptAbi,
        Kernel,
        TopologySchema,
        SelfcheckSchema,
        TranscriptSchema,
        RequestedDeviceCap,
        InvalidPolicy,
        PolicyBelowCap,
        DispatchZero,
        DispatchAboveCore,
        DispatchAbovePolicy,
    }

    #[derive(Clone, Copy, Debug)]
    enum ScheduleMode {
        Exact,
        Decline,
        Panic,
        WaitForDeadline,
        ReenterRegistration,
        Mutate(ScheduleEvidenceMutation),
    }

    struct ScheduleBackend {
        registration: Arc<GpuBabBoundBackendRegistration>,
        alternate_registration: Option<Arc<GpuBabBoundBackendRegistration>>,
        registration_calls: AtomicUsize,
        schedule_calls: AtomicUsize,
        mode: Mutex<ScheduleMode>,
    }

    impl ScheduleBackend {
        fn new(mode: ScheduleMode) -> Self {
            Self {
                registration: fresh_schedule_registration(),
                alternate_registration: None,
                registration_calls: AtomicUsize::new(0),
                schedule_calls: AtomicUsize::new(0),
                mode: Mutex::new(mode),
            }
        }

        fn with_registration_swap() -> Self {
            Self {
                registration: fresh_schedule_registration(),
                alternate_registration: Some(fresh_schedule_registration()),
                registration_calls: AtomicUsize::new(0),
                schedule_calls: AtomicUsize::new(0),
                mode: Mutex::new(ScheduleMode::Exact),
            }
        }

        fn set_mode(&self, mode: ScheduleMode) {
            *self
                .mode
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = mode;
        }

        fn schedule_calls(&self) -> usize {
            self.schedule_calls.load(Ordering::Relaxed)
        }

        fn exact_evidence(
            &self,
            request: &GpuBabBoundStaticScheduleRequest<'_>,
        ) -> GpuBabBoundBackendScheduleEvidence {
            GpuBabBoundBackendScheduleEvidence {
                backend_issuer_sha256: *self.registration.backend_issuer_sha256(),
                registration_epoch: self.registration.registration_epoch(),
                static_payload_identity_sha256: *request.static_payload_identity_sha256(),
                topology_schema_version: request.topology_schema_version(),
                schedule_identity: schedule_identity(),
                requested_max_device_bytes: request.requested_max_device_bytes(),
                phase_policy: GpuBabBoundPhasePolicy {
                    max_device_bytes: request.requested_max_device_bytes(),
                    preferred_domains_per_wave: 4,
                    minimum_domains_per_wave: 1,
                    maximum_domains_per_wave: 16,
                    maximum_objectives: 16,
                    maximum_dispatches_per_wave: 64,
                    maximum_submits_per_wave: 8,
                },
                dispatches_per_subchunk: 3,
            }
        }

        fn mutated_evidence(
            &self,
            request: &GpuBabBoundStaticScheduleRequest<'_>,
            mutation: ScheduleEvidenceMutation,
        ) -> GpuBabBoundBackendScheduleEvidence {
            let mut evidence = self.exact_evidence(request);
            match mutation {
                ScheduleEvidenceMutation::BackendIssuer => {
                    evidence.backend_issuer_sha256 = hash(81);
                }
                ScheduleEvidenceMutation::RegistrationEpoch => {
                    evidence.registration_epoch += 1;
                }
                ScheduleEvidenceMutation::StaticIdentityZero => {
                    evidence.static_payload_identity_sha256 = [0; 32];
                }
                ScheduleEvidenceMutation::StaticIdentityMismatch => {
                    evidence.static_payload_identity_sha256 = hash(82);
                }
                ScheduleEvidenceMutation::TopologySchemaVersion => {
                    evidence.topology_schema_version += 1;
                }
                ScheduleEvidenceMutation::SchemaBundleVersion => {
                    evidence.schedule_identity.schema_bundle_version += 1;
                }
                ScheduleEvidenceMutation::ProviderAbi => {
                    evidence.schedule_identity.provider_abi_sha256 = hash(83);
                }
                ScheduleEvidenceMutation::ReceiptAbi => {
                    evidence.schedule_identity.receipt_abi_sha256 = hash(84);
                }
                ScheduleEvidenceMutation::Kernel => {
                    evidence.schedule_identity.kernel_sha256 = hash(85);
                }
                ScheduleEvidenceMutation::TopologySchema => {
                    evidence.schedule_identity.topology_schema_sha256 = hash(86);
                }
                ScheduleEvidenceMutation::SelfcheckSchema => {
                    evidence.schedule_identity.selfcheck_schema_sha256 = hash(87);
                }
                ScheduleEvidenceMutation::TranscriptSchema => {
                    evidence.schedule_identity.transcript_schema_sha256 = hash(88);
                }
                ScheduleEvidenceMutation::RequestedDeviceCap => {
                    evidence.requested_max_device_bytes += 1;
                }
                ScheduleEvidenceMutation::InvalidPolicy => {
                    evidence.phase_policy.maximum_dispatches_per_wave = usize::MAX;
                }
                ScheduleEvidenceMutation::PolicyBelowCap => {
                    evidence.phase_policy.max_device_bytes -= 1;
                }
                ScheduleEvidenceMutation::DispatchZero => {
                    evidence.dispatches_per_subchunk = 0;
                }
                ScheduleEvidenceMutation::DispatchAboveCore => {
                    evidence.dispatches_per_subchunk = GPU_BAB_BOUND_MAX_DISPATCHES_PER_WAVE + 1;
                }
                ScheduleEvidenceMutation::DispatchAbovePolicy => {
                    evidence.phase_policy.maximum_dispatches_per_wave = 1;
                    evidence.dispatches_per_subchunk = 2;
                }
            }
            evidence
        }
    }

    impl GpuCrownBackward for ScheduleBackend {
        fn provides_sound_gpu_crown(&self) -> bool {
            true
        }

        fn provides_sound_gpu_bab_bound_phase(&self) -> bool {
            true
        }

        fn gpu_bab_bound_numerical_tcb(&self) -> Option<&dyn GpuBabBoundNumericalTcb> {
            Some(self)
        }

        fn crown_backward_gpu(
            &self,
            _layers: &[super::super::GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<super::super::GpuCrownResult> {
            unreachable!("schedule tests never execute legacy CROWN")
        }
    }

    impl GpuBabBoundNumericalTcb for ScheduleBackend {
        fn registration(&self) -> &GpuBabBoundBackendRegistration {
            let call = self.registration_calls.fetch_add(1, Ordering::Relaxed);
            if call > 0 {
                if let Some(alternate) = &self.alternate_registration {
                    return alternate.as_ref();
                }
            }
            self.registration.as_ref()
        }

        fn certify_static_schedule(
            &self,
            invocation: &GpuBabBoundScheduleTcbInvocation<'_, '_>,
        ) -> GpuBabBoundBackendScheduleDisposition {
            self.schedule_calls.fetch_add(1, Ordering::Relaxed);
            let mode = *self
                .mode
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match mode {
                ScheduleMode::Exact => GpuBabBoundBackendScheduleDisposition::Certified(
                    self.exact_evidence(invocation.request()),
                ),
                ScheduleMode::Decline => GpuBabBoundBackendScheduleDisposition::CleanDecline(
                    GpuBabBoundPhaseDecline::Unsupported,
                ),
                ScheduleMode::Panic => panic_with_hostile_payload(),
                ScheduleMode::WaitForDeadline => {
                    while Instant::now() < invocation.request().deadline() {
                        std::hint::spin_loop();
                    }
                    GpuBabBoundBackendScheduleDisposition::Certified(
                        self.exact_evidence(invocation.request()),
                    )
                }
                ScheduleMode::ReenterRegistration => {
                    drop(self.registration.available_guard().unwrap());
                    GpuBabBoundBackendScheduleDisposition::Certified(
                        self.exact_evidence(invocation.request()),
                    )
                }
                ScheduleMode::Mutate(mutation) => GpuBabBoundBackendScheduleDisposition::Certified(
                    self.mutated_evidence(invocation.request(), mutation),
                ),
            }
        }

        fn phase_policy(
            &self,
            _invocation: &GpuBabBoundTcbInvocation<'_>,
        ) -> Option<GpuBabBoundPhasePolicy> {
            None
        }

        fn prepare_phase<'a>(
            &'a self,
            _invocation: &GpuBabBoundTcbInvocation<'_>,
        ) -> GpuBabBoundBackendOpenPreparation<'a> {
            GpuBabBoundBackendOpenPreparation::CleanDecline(GpuBabBoundPhaseDecline::Unsupported)
        }
    }

    fn phase() -> GpuBabBoundPhaseDescriptor {
        GpuBabBoundPhaseDescriptor::new(
            GpuBabBoundGraphPlan {
                topology_schema_version: 1,
                topology_bytes: GpuBabBoundOwnedSlice::new(vec![1_u8, 2, 3, 4]),
                f32_tensors: vec![
                    f32_tensor(
                        GpuBabBoundF32TensorRole::Parameters,
                        vec![4],
                        vec![0.25, -0.5, 0.75, 1.0],
                    ),
                    f32_tensor(
                        GpuBabBoundF32TensorRole::CertifiedErrors,
                        vec![2],
                        vec![0.0, 0.0],
                    ),
                    f32_tensor(
                        GpuBabBoundF32TensorRole::Relaxations,
                        vec![2],
                        vec![0.25, 0.75],
                    ),
                    f32_tensor(
                        GpuBabBoundF32TensorRole::InputLower,
                        vec![2],
                        vec![-1.0, -1.0],
                    ),
                    f32_tensor(
                        GpuBabBoundF32TensorRole::InputUpper,
                        vec![2],
                        vec![1.0, 1.0],
                    ),
                    f32_tensor(
                        GpuBabBoundF32TensorRole::RootLower,
                        vec![2],
                        vec![-2.0, -2.0],
                    ),
                    f32_tensor(GpuBabBoundF32TensorRole::RootUpper, vec![2], vec![2.0, 2.0]),
                    f32_tensor(
                        GpuBabBoundF32TensorRole::ObjectiveCoefficients,
                        vec![8, 2],
                        (0..8).flat_map(|_| [1.0, -1.0]).collect(),
                    ),
                ],
                u32_tensors: vec![
                    GpuBabBoundU32Tensor {
                        role: GpuBabBoundU32TensorRole::ObjectiveIndices,
                        shape: vec![8],
                        values: GpuBabBoundOwnedSlice::new((0..8_u32).collect::<Vec<_>>()),
                    },
                    GpuBabBoundU32Tensor {
                        role: GpuBabBoundU32TensorRole::TopologyMetadata,
                        shape: vec![2],
                        values: GpuBabBoundOwnedSlice::new(vec![2_u32, 3]),
                    },
                ],
                dispatches_per_subchunk: 3,
            },
            Instant::now() + Duration::from_mins(1),
            8_192,
        )
        .unwrap()
    }

    fn f32_tensor(
        role: GpuBabBoundF32TensorRole,
        shape: Vec<usize>,
        values: Vec<f32>,
    ) -> GpuBabBoundF32Tensor {
        GpuBabBoundF32Tensor {
            role,
            shape,
            values: GpuBabBoundOwnedSlice::new(values),
        }
    }

    fn request() -> GpuBabBoundWaveRequest {
        GpuBabBoundWaveRequest {
            parent_groups: vec![
                GpuBabBoundParentGroup {
                    parent_group_id: 10,
                    parent_identity_sha256: hash(31),
                    first_domain: 0,
                    child_cardinality: 2,
                },
                GpuBabBoundParentGroup {
                    parent_group_id: 20,
                    parent_identity_sha256: hash(32),
                    first_domain: 2,
                    child_cardinality: 2,
                },
            ],
            domains: vec![
                domain(10, 0, 41, 0),
                domain(10, 1, 42, 1),
                domain(20, 0, 43, 2),
                domain(20, 1, 44, 3),
            ],
            domain_arena: GpuBabBoundDomainArena {
                activation: GpuBabBoundOwnedSlice::new(
                    (0..16).map(|value| value as f32 / 16.0).collect(),
                ),
                beta: GpuBabBoundOwnedSlice::new(vec![0.1; 8]),
                abs: GpuBabBoundOwnedSlice::new(vec![0.25; 4]),
                box_lower: GpuBabBoundOwnedSlice::new(vec![-1.0; 4]),
                box_upper: GpuBabBoundOwnedSlice::new(vec![1.0; 4]),
                cached_la: GpuBabBoundOwnedSlice::new(vec![0.5; 16]),
            },
            objective_indices: vec![1, 3],
            subchunks: vec![
                GpuBabBoundSubchunk {
                    parent_group_id: 10,
                    first_domain: 0,
                    domain_count: 2,
                    first_q: 0,
                    row_count: 4,
                },
                GpuBabBoundSubchunk {
                    parent_group_id: 20,
                    first_domain: 2,
                    domain_count: 2,
                    first_q: 4,
                    row_count: 4,
                },
            ],
            inherited_lower: vec![-1.0; 8],
            inherited_upper: vec![1.0; 8],
            deadline: Instant::now() + Duration::from_secs(30),
            max_device_bytes: 4_096,
        }
    }

    fn domain(
        parent_group_id: u64,
        child_ordinal: usize,
        domain_slot: u64,
        arena_index: usize,
    ) -> GpuBabBoundDomainTranscript {
        GpuBabBoundDomainTranscript {
            parent_group_id,
            child_ordinal,
            child_cardinality: 2,
            domain_slot,
            operands: GpuBabBoundOperandView {
                activation: GpuBabBoundArenaRange {
                    start: arena_index * 4,
                    len: 4,
                },
                beta: GpuBabBoundArenaRange {
                    start: arena_index * 2,
                    len: 2,
                },
                abs: GpuBabBoundArenaRange {
                    start: arena_index,
                    len: 1,
                },
                box_lower: GpuBabBoundArenaRange {
                    start: arena_index,
                    len: 1,
                },
                box_upper: GpuBabBoundArenaRange {
                    start: arena_index,
                    len: 1,
                },
                cached_la: GpuBabBoundArenaRange {
                    start: arena_index * 4,
                    len: 4,
                },
            },
        }
    }

    fn open_memory() -> GpuBabBoundMemoryReceipt {
        GpuBabBoundMemoryReceipt {
            retained_graph_bytes: 512,
            retained_phase_bytes: 256,
            wave_working_bytes: 0,
            queued_upload_bytes: 0,
            result_readback_bytes: 0,
            peak_device_bytes: 768,
        }
    }

    fn fresh_static_transfers(
        transcript: GpuBabBoundPhaseTranscript,
        retained_graph_bytes: usize,
        retained_phase_bytes: usize,
    ) -> GpuBabBoundStaticTransferReceipt {
        GpuBabBoundStaticTransferReceipt {
            graph_identity_sha256: transcript.graph_identity_sha256,
            phase_identity_sha256: transcript.static_phase_identity_sha256,
            graph_payload_bytes: transcript.static_graph_payload_bytes,
            phase_payload_bytes: transcript.static_phase_payload_bytes,
            graph_padding_bytes: retained_graph_bytes
                .checked_sub(transcript.static_graph_payload_bytes)
                .expect("fake graph residency covers its typed payload"),
            phase_padding_bytes: retained_phase_bytes
                .checked_sub(transcript.static_phase_payload_bytes)
                .expect("fake phase residency covers its typed payload"),
            graph_source: GpuBabBoundStaticPayloadSource::FreshUpload,
            phase_source: GpuBabBoundStaticPayloadSource::FreshUpload,
            graph_host_to_device_bytes: transcript.static_graph_payload_bytes,
            phase_host_to_device_bytes: transcript.static_phase_payload_bytes,
            host_to_device_bytes: transcript
                .static_graph_payload_bytes
                .checked_add(transcript.static_phase_payload_bytes)
                .unwrap(),
        }
    }

    fn not_transferred_static_receipt(
        transcript: GpuBabBoundPhaseTranscript,
    ) -> GpuBabBoundStaticTransferReceipt {
        GpuBabBoundStaticTransferReceipt {
            graph_identity_sha256: transcript.graph_identity_sha256,
            phase_identity_sha256: transcript.static_phase_identity_sha256,
            graph_payload_bytes: transcript.static_graph_payload_bytes,
            phase_payload_bytes: transcript.static_phase_payload_bytes,
            graph_padding_bytes: 0,
            phase_padding_bytes: 0,
            graph_source: GpuBabBoundStaticPayloadSource::NotTransferred,
            phase_source: GpuBabBoundStaticPayloadSource::NotTransferred,
            graph_host_to_device_bytes: 0,
            phase_host_to_device_bytes: 0,
            host_to_device_bytes: 0,
        }
    }

    fn static_open_receipt_for_sources(
        transcript: GpuBabBoundPhaseTranscript,
        graph_source: GpuBabBoundStaticPayloadSource,
        phase_source: GpuBabBoundStaticPayloadSource,
    ) -> GpuBabBoundBackendOpenReceipt {
        let graph_h2d = usize::from(matches!(
            graph_source,
            GpuBabBoundStaticPayloadSource::FreshUpload
        )) * transcript.static_graph_payload_bytes;
        let phase_h2d = usize::from(matches!(
            phase_source,
            GpuBabBoundStaticPayloadSource::FreshUpload
        )) * transcript.static_phase_payload_bytes;
        let graph_padding_bytes = 16;
        let phase_padding_bytes = 8;
        let retained_graph_bytes = transcript.static_graph_payload_bytes + graph_padding_bytes;
        let retained_phase_bytes = transcript.static_phase_payload_bytes + phase_padding_bytes;
        let host_to_device_bytes = graph_h2d + phase_h2d;
        let peak_device_bytes = retained_graph_bytes + retained_phase_bytes + host_to_device_bytes;
        GpuBabBoundBackendOpenReceipt {
            transcript,
            authorized_device_bytes: transcript.max_device_bytes,
            memory: GpuBabBoundMemoryReceipt {
                retained_graph_bytes,
                retained_phase_bytes,
                wave_working_bytes: 0,
                queued_upload_bytes: host_to_device_bytes,
                result_readback_bytes: 0,
                peak_device_bytes,
            },
            static_transfers: GpuBabBoundStaticTransferReceipt {
                graph_identity_sha256: transcript.graph_identity_sha256,
                phase_identity_sha256: transcript.static_phase_identity_sha256,
                graph_payload_bytes: transcript.static_graph_payload_bytes,
                phase_payload_bytes: transcript.static_phase_payload_bytes,
                graph_padding_bytes,
                phase_padding_bytes,
                graph_source,
                phase_source,
                graph_host_to_device_bytes: graph_h2d,
                phase_host_to_device_bytes: phase_h2d,
                host_to_device_bytes,
            },
            released_graph_bytes: 0,
            released_phase_bytes: 0,
        }
    }

    fn completed_rows(
        accepted: &GpuBabBoundAcceptedWave<'_>,
        tighten_q: Option<usize>,
    ) -> Vec<GpuBabBoundBackendRow> {
        let request = accepted.request();
        let r = request.objective_indices.len();
        (0..request.row_count().unwrap())
            .map(|q| {
                let domain = &request.domains[q / r];
                GpuBabBoundBackendRow {
                    parent_group_id: domain.parent_group_id,
                    child_ordinal: domain.child_ordinal,
                    child_cardinality: domain.child_cardinality,
                    domain_slot: domain.domain_slot,
                    domain_identity_sha256: request.domain_identity_sha256(q / r).unwrap(),
                    objective_index: request.objective_indices[q % r],
                    q: q as u32,
                    lower: if tighten_q == Some(q) {
                        -0.5
                    } else {
                        request.inherited_lower[q]
                    },
                    upper: request.inherited_upper[q],
                    status: 0,
                    taint: 0,
                }
            })
            .collect()
    }

    fn completed_outcomes(
        accepted: &GpuBabBoundAcceptedWave<'_>,
    ) -> Vec<GpuBabBoundBackendDomainOutcome> {
        accepted
            .request()
            .domains
            .iter()
            .enumerate()
            .map(|(index, domain)| GpuBabBoundBackendDomainOutcome {
                parent_group_id: domain.parent_group_id,
                child_ordinal: domain.child_ordinal,
                child_cardinality: domain.child_cardinality,
                domain_slot: domain.domain_slot,
                domain_identity_sha256: accepted.request().domain_identity_sha256(index).unwrap(),
                kind: GpuBabBoundBackendDomainOutcomeKind::Bounded,
            })
            .collect()
    }

    fn completed_receipt(
        accepted: &GpuBabBoundAcceptedWave<'_>,
        phase: &GpuBabBoundPhaseDescriptor,
        open_memory: GpuBabBoundMemoryReceipt,
        tightened_rows: usize,
    ) -> GpuBabBoundBackendWaveReceipt {
        let request = accepted.request();
        let shape = request.validate_static(phase).unwrap();
        let h2d = shape.domain_operand_bytes
            + shape.inherited_endpoint_bytes
            + shape.objective_index_bytes
            + shape.subchunk_descriptor_bytes;
        let endpoint_bytes = shape.rows * ENDPOINT_BYTES_PER_ROW;
        let sidecar_bytes = shape.rows * RESULT_SIDECAR_BYTES_PER_ROW;
        let outcome_bytes = request.domains.len() * DOMAIN_OUTCOME_SIDECAR_BYTES;
        let d2h = endpoint_bytes + sidecar_bytes + outcome_bytes;
        let memory = GpuBabBoundMemoryReceipt {
            retained_graph_bytes: open_memory.retained_graph_bytes,
            retained_phase_bytes: open_memory.retained_phase_bytes,
            wave_working_bytes: 1_024,
            queued_upload_bytes: h2d,
            result_readback_bytes: d2h,
            peak_device_bytes: open_memory.retained_graph_bytes
                + open_memory.retained_phase_bytes
                + 1_024
                + h2d
                + d2h,
        };
        GpuBabBoundBackendWaveReceipt {
            transcript: accepted.transcript(),
            requested_parent_groups: request.parent_groups.len(),
            completed_parent_groups: request.parent_groups.len(),
            requested_domains: request.domains.len(),
            completed_domains: request.domains.len(),
            bounded_domains: request.domains.len(),
            pruned_domains: 0,
            objective_rows: request.objective_indices.len(),
            requested_rows: shape.rows,
            completed_rows: shape.rows,
            returned_rows: shape.rows,
            requested_subchunks: request.subchunks.len(),
            completed_subchunks: request.subchunks.len(),
            authorized_device_bytes: request.max_device_bytes,
            memory,
            transfers: GpuBabBoundTransferReceipt {
                activation_operand_bytes: shape.activation_operand_bytes,
                beta_operand_bytes: shape.beta_operand_bytes,
                abs_operand_bytes: shape.abs_operand_bytes,
                box_operand_bytes: shape.box_operand_bytes,
                cached_la_operand_bytes: shape.cached_la_operand_bytes,
                domain_operand_bytes: shape.domain_operand_bytes,
                inherited_endpoint_bytes: shape.inherited_endpoint_bytes,
                objective_index_bytes: shape.objective_index_bytes,
                subchunk_descriptor_bytes: shape.subchunk_descriptor_bytes,
                host_to_device_bytes: h2d,
                result_endpoint_bytes: endpoint_bytes,
                result_sidecar_bytes: sidecar_bytes,
                domain_outcome_sidecar_bytes: outcome_bytes,
                coefficient_device_to_host_bytes: 0,
                device_to_host_bytes: d2h,
                readbacks: 1,
                synchronizations: 1,
            },
            dispatches: shape.required_dispatches,
            submits: 1,
            waves: 1,
            tightened_rows,
        }
    }

    fn failure_receipt(
        accepted: &GpuBabBoundAcceptedWave<'_>,
        phase: &GpuBabBoundPhaseDescriptor,
        open_memory: GpuBabBoundMemoryReceipt,
    ) -> GpuBabBoundBackendWaveReceipt {
        let request = accepted.request();
        let shape = request.validate_static(phase).unwrap();
        GpuBabBoundBackendWaveReceipt {
            transcript: accepted.transcript(),
            requested_parent_groups: request.parent_groups.len(),
            completed_parent_groups: 0,
            requested_domains: request.domains.len(),
            completed_domains: 0,
            bounded_domains: 0,
            pruned_domains: 0,
            objective_rows: request.objective_indices.len(),
            requested_rows: shape.rows,
            completed_rows: 0,
            returned_rows: 0,
            requested_subchunks: request.subchunks.len(),
            completed_subchunks: 0,
            authorized_device_bytes: request.max_device_bytes,
            memory: open_memory,
            transfers: GpuBabBoundTransferReceipt::default(),
            dispatches: 0,
            submits: 0,
            waves: 1,
            tightened_rows: 0,
        }
    }

    fn completed_raw(
        accepted: &GpuBabBoundAcceptedWave<'_>,
        phase: &GpuBabBoundPhaseDescriptor,
        open_memory: GpuBabBoundMemoryReceipt,
        tighten_q: Option<usize>,
    ) -> GpuBabBoundBackendWaveDisposition {
        GpuBabBoundBackendWaveDisposition::Completed {
            domain_outcomes: completed_outcomes(accepted),
            rows: completed_rows(accepted, tighten_q),
            receipt: completed_receipt(
                accepted,
                phase,
                open_memory,
                usize::from(tighten_q.is_some()),
            ),
        }
    }

    fn corrupted_raw(
        accepted: &GpuBabBoundAcceptedWave<'_>,
        phase: &GpuBabBoundPhaseDescriptor,
        open_memory: GpuBabBoundMemoryReceipt,
        corruption: Corruption,
    ) -> GpuBabBoundBackendWaveDisposition {
        let mut domain_outcomes = completed_outcomes(accepted);
        let mut rows = completed_rows(accepted, None);
        let mut receipt = completed_receipt(accepted, phase, open_memory, 0);
        match corruption {
            Corruption::RowPermutation => rows.swap(0, 4),
            Corruption::ParentSidecar => rows[0].parent_group_id = 20,
            Corruption::ChildOrdinal => rows[0].child_ordinal = 1,
            Corruption::DomainEcho => rows[0].domain_identity_sha256 = hash(250),
            Corruption::ObjectiveEcho => rows[0].objective_index = 7,
            Corruption::QSidecar => rows[0].q = 1,
            Corruption::PartialRows => {
                rows.pop();
            }
            Corruption::Nonfinite => rows[0].lower = f32::NAN,
            Corruption::Inverted => {
                rows[0].lower = 2.0;
                rows[0].upper = 1.0;
            }
            Corruption::Disjoint => {
                rows[0].lower = 2.0;
                rows[0].upper = 3.0;
            }
            Corruption::Status => rows[0].status = 1,
            Corruption::Taint => rows[0].taint = 1,
            Corruption::BackendEcho => {
                receipt.transcript.phase.backend.backend_issuer_sha256 = hash(251);
            }
            Corruption::ScheduleEcho => {
                receipt.transcript.schedule_identity_sha256 = hash(252);
            }
            Corruption::EndpointEcho => {
                receipt.transcript.inherited_endpoints_sha256 = hash(253);
            }
            Corruption::DeadlineEcho => {
                receipt.transcript.deadline += Duration::from_nanos(1);
            }
            Corruption::PartialCount => receipt.completed_parent_groups -= 1,
            Corruption::OverCap => {
                receipt.memory.wave_working_bytes += accepted.request().max_device_bytes;
                receipt.memory.peak_device_bytes += accepted.request().max_device_bytes;
            }
            Corruption::MemoryOverflow => {
                receipt.memory.wave_working_bytes = usize::MAX;
                receipt.memory.peak_device_bytes = usize::MAX;
            }
            Corruption::Readbacks => receipt.transfers.readbacks = 2,
            Corruption::Synchronizations => receipt.transfers.synchronizations = 0,
            Corruption::CoefficientD2h => {
                receipt.transfers.coefficient_device_to_host_bytes = 4;
                receipt.transfers.device_to_host_bytes += 4;
                receipt.memory.result_readback_bytes += 4;
                receipt.memory.peak_device_bytes += 4;
            }
            Corruption::H2dEquation => receipt.transfers.host_to_device_bytes += 1,
            Corruption::D2hEquation => receipt.transfers.device_to_host_bytes += 1,
            Corruption::OperandEquation => {
                receipt.transfers.activation_operand_bytes += 1;
                receipt.transfers.domain_operand_bytes += 1;
                receipt.transfers.host_to_device_bytes += 1;
                receipt.memory.queued_upload_bytes += 1;
                receipt.memory.peak_device_bytes += 1;
            }
            Corruption::ResultSidecarEquation => {
                receipt.transfers.result_sidecar_bytes += size_of::<u32>();
                receipt.transfers.device_to_host_bytes += size_of::<u32>();
                receipt.memory.result_readback_bytes += size_of::<u32>();
                receipt.memory.peak_device_bytes += size_of::<u32>();
            }
            Corruption::DomainOutcomeSidecarEquation => {
                receipt.transfers.domain_outcome_sidecar_bytes += size_of::<u32>();
                receipt.transfers.device_to_host_bytes += size_of::<u32>();
                receipt.memory.result_readback_bytes += size_of::<u32>();
                receipt.memory.peak_device_bytes += size_of::<u32>();
            }
            Corruption::TighteningCount => receipt.tightened_rows = 1,
            Corruption::PrunedCount => receipt.pruned_domains = 1,
            Corruption::DispatchesMax => receipt.dispatches = usize::MAX,
            Corruption::SubmitsMax => receipt.submits = usize::MAX,
            Corruption::OutcomePermutation => domain_outcomes.swap(0, 1),
            Corruption::OutcomeAssociation => domain_outcomes[0].domain_slot += 1,
        }
        GpuBabBoundBackendWaveDisposition::Completed {
            domain_outcomes,
            rows,
            receipt,
        }
    }

    fn open_lease(
        backend: &dyn GpuCrownBackward,
        phase: GpuBabBoundPhaseDescriptor,
    ) -> GpuBabBoundPhaseLease<'_> {
        match GpuBabBoundPhaseLease::open(backend, phase) {
            GpuBabBoundPhaseOpen::Opened(lease) => lease,
            _ => panic!("fake backend must open"),
        }
    }

    fn execute(
        lease: &mut GpuBabBoundPhaseLease<'_>,
        request: GpuBabBoundWaveRequest,
    ) -> GpuBabBoundWaveDisposition {
        let preparation = lease.prepare_wave(request);
        assert!(!preparation.permits_legacy_fallback());
        match preparation {
            GpuBabBoundWavePreparation::Accepted(capability) => capability.execute_accepted(),
            _ => panic!("fake backend must accept valid request"),
        }
    }

    fn assert_contract_failure(mode: Mode) {
        let backend = FakeBackend::new(vec![mode]);
        let mut lease = open_lease(&backend, phase());
        match execute(&mut lease, request()) {
            GpuBabBoundWaveDisposition::AcceptedFailure(failure) => {
                assert_eq!(
                    failure.kind(),
                    GpuBabBoundTerminalFailureKind::ContractViolation
                );
                assert!(!failure.receipt_validated());
            }
            _ => panic!("corruption must become a terminal contract failure"),
        }
    }

    #[test]
    fn trait_is_object_safe_and_default_path_declines_before_acceptance() {
        let mut backend = DefaultDecliningBackend;
        let backend: &mut dyn GpuCrownBackward = &mut backend;
        assert!(!backend.provides_sound_gpu_bab_bound_phase());
        assert!(backend.gpu_bab_bound_numerical_tcb().is_none());
        let open = GpuBabBoundPhaseLease::open(backend, phase());
        assert!(open.permits_legacy_fallback());
        assert!(matches!(
            open,
            GpuBabBoundPhaseOpen::CleanDecline(GpuBabBoundPhaseDecline::Unsupported)
        ));

        let unattested = UnattestedRawBackend;
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&unattested, phase()),
            GpuBabBoundPhaseOpen::CleanDecline(GpuBabBoundPhaseDecline::Unsupported)
        ));

        for (sound_gpu_crown, sound_bab_phase) in [(false, true), (true, false)] {
            let mut gated = FakeBackend::new(vec![]);
            gated.sound_gpu_crown = sound_gpu_crown;
            gated.sound_bab_phase = sound_bab_phase;
            assert!(matches!(
                GpuBabBoundPhaseLease::open(&gated, phase()),
                GpuBabBoundPhaseOpen::CleanDecline(GpuBabBoundPhaseDecline::Unsupported)
            ));
            assert_eq!(
                gated.accepted_open_calls(),
                0,
                "a false soundness gate must prevent raw accepted open"
            );
            assert_eq!(
                gated.registration.ledger.lock().unwrap().highest_generation,
                0
            );
        }
    }

    #[test]
    fn core_module_has_no_builtin_tcb_and_core_forbids_unsafe_code() {
        let module_source = include_str!("gemm_gpu_bab_bound.rs");
        let production_source = module_source
            .split("#[cfg(test)]")
            .next()
            .expect("module has a production prefix");
        assert_eq!(
            production_source
                .matches("impl GpuBabBoundNumericalTcb for ")
                .count(),
            0,
            "a core-module TCB implementation requires an explicit review-policy update"
        );
        assert_eq!(
            production_source
                .matches("std::panic::catch_unwind")
                .count(),
            1,
            "all production TCB catches must pass through payload-quarantining catch_tcb_unwind"
        );
        assert_eq!(production_source.matches("AssertUnwindSafe").count(), 1);
        assert!(!production_source.contains("fn close(self: Box<Self>)"));
        assert!(include_str!("lib.rs").contains("#![forbid(unsafe_code)]"));
    }

    #[test]
    fn provider_policy_and_phase_preparation_panics_are_nonfallback_terminals() {
        for (panic_policy, panic_prepare, expected) in [
            (true, false, GpuBabBoundProviderFailureKind::PolicyPanicked),
            (
                false,
                true,
                GpuBabBoundProviderFailureKind::PreparationPanicked,
            ),
        ] {
            let backend = {
                let mut backend = FakeBackend::new(vec![]);
                backend.panic_phase_policy = panic_policy;
                backend.panic_phase_prepare = panic_prepare;
                backend
            };
            let open = GpuBabBoundPhaseLease::open(&backend, phase());
            assert!(!open.permits_legacy_fallback());
            match open {
                GpuBabBoundPhaseOpen::ProviderFailure(failure) => {
                    assert_eq!(failure.kind(), expected);
                }
                _ => panic!("provider panic must be caught as a typed terminal"),
            }
            assert_eq!(backend.accepted_open_calls(), 0);
            assert_eq!(
                backend
                    .registration
                    .ledger
                    .lock()
                    .unwrap()
                    .highest_generation,
                0
            );
        }
    }

    #[test]
    fn malformed_phase_policy_is_nonfallback_and_poisons_registration() {
        let registration = fresh_registration();
        let mut malformed = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        malformed.invalid_phase_policy = true;
        let open = GpuBabBoundPhaseLease::open(&malformed, phase());
        assert!(!open.permits_legacy_fallback());
        assert!(matches!(
            open,
            GpuBabBoundPhaseOpen::ProviderFailure(GpuBabBoundProviderFailure {
                kind: GpuBabBoundProviderFailureKind::InvalidPolicy,
                ..
            })
        ));
        assert_eq!(malformed.accepted_open_calls(), 0);

        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn late_preclaim_prepared_session_drop_panic_is_typed_and_absorbing() {
        let registration = fresh_registration();
        let mut backend = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        backend.wait_phase_prepare_deadline = true;
        backend.panic_session_drop = true;
        let mut short_phase = phase();
        short_phase.deadline = Instant::now() + Duration::from_millis(50);
        let open = GpuBabBoundPhaseLease::open(&backend, short_phase);
        assert!(!open.permits_legacy_fallback());
        assert!(matches!(
            open,
            GpuBabBoundPhaseOpen::ProviderFailure(GpuBabBoundProviderFailure {
                kind: GpuBabBoundProviderFailureKind::DormantSessionDropPanicked,
                ..
            })
        ));
        assert_eq!(backend.accepted_open_calls(), 0);
        assert_eq!(backend.session_drop_calls.load(Ordering::Relaxed), 1);

        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn late_preclaim_discard_poison_precedes_reentrant_session_drop() {
        let registration = fresh_registration();
        let nested_terminal = Arc::new(AtomicBool::new(false));
        let nested_fallback = Arc::new(AtomicBool::new(true));
        let nested_accepted_open_calls = Arc::new(AtomicUsize::new(usize::MAX));
        let backend = ReentrantPreparedBackend {
            registration: Arc::clone(&registration),
            nested_terminal: Arc::clone(&nested_terminal),
            nested_fallback: Arc::clone(&nested_fallback),
            nested_accepted_open_calls: Arc::clone(&nested_accepted_open_calls),
        };
        let mut short_phase = phase();
        short_phase.deadline = Instant::now() + Duration::from_millis(50);
        let outer = GpuBabBoundPhaseLease::open(&backend, short_phase);
        assert!(!outer.permits_legacy_fallback());
        assert!(matches!(
            outer,
            GpuBabBoundPhaseOpen::ProviderFailure(GpuBabBoundProviderFailure {
                kind: GpuBabBoundProviderFailureKind::DeadlineExpired,
                ..
            })
        ));
        assert!(nested_terminal.load(Ordering::Relaxed));
        assert!(!nested_fallback.load(Ordering::Relaxed));
        assert_eq!(nested_accepted_open_calls.load(Ordering::Relaxed), 0);
        assert!(registration.ledger.lock().unwrap().poisoned);

        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn late_policy_and_phase_preparation_decline_or_prepare_never_grant_fallback() {
        let mut late_policy = FakeBackend::new(vec![]);
        late_policy.wait_phase_policy_deadline = true;
        let mut short_phase = phase();
        short_phase.deadline = Instant::now() + Duration::from_millis(50);
        let open = GpuBabBoundPhaseLease::open(&late_policy, short_phase);
        assert!(!open.permits_legacy_fallback());
        assert!(matches!(
            open,
            GpuBabBoundPhaseOpen::ProviderFailure(GpuBabBoundProviderFailure {
                kind: GpuBabBoundProviderFailureKind::DeadlineExpired,
                ..
            })
        ));
        assert_eq!(late_policy.accepted_open_calls(), 0);
        assert_eq!(
            late_policy
                .registration
                .ledger
                .lock()
                .unwrap()
                .highest_generation,
            0
        );

        for decline in [false, true] {
            let mut backend = FakeBackend::new(vec![]);
            backend.wait_phase_prepare_deadline = true;
            backend.decline_phase_prepare = decline;
            let mut short_phase = phase();
            short_phase.deadline = Instant::now() + Duration::from_millis(50);
            let open = GpuBabBoundPhaseLease::open(&backend, short_phase);
            assert!(!open.permits_legacy_fallback());
            assert!(matches!(
                open,
                GpuBabBoundPhaseOpen::ProviderFailure(GpuBabBoundProviderFailure {
                    kind: GpuBabBoundProviderFailureKind::DeadlineExpired,
                    ..
                })
            ));
            assert_eq!(backend.accepted_open_calls(), 0);
            assert_eq!(
                backend
                    .registration
                    .ledger
                    .lock()
                    .unwrap()
                    .highest_generation,
                0
            );
        }
    }

    #[test]
    fn guard_wait_cannot_publish_phase_decline_after_deadline() {
        let registration = fresh_registration();
        let backend = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        let mut descriptor = phase();
        descriptor.max_device_bytes = 8_193;
        descriptor.deadline = Instant::now() + Duration::from_millis(100);
        let deadline = descriptor.deadline;
        let ledger_guard = registration.ledger.lock().unwrap();
        std::thread::scope(|scope| {
            let backend_ref = &backend;
            let open = scope.spawn(move || {
                let disposition = GpuBabBoundPhaseLease::open(backend_ref, descriptor);
                let permits_fallback = disposition.permits_legacy_fallback();
                let deadline_terminal = matches!(
                    disposition,
                    GpuBabBoundPhaseOpen::ProviderFailure(GpuBabBoundProviderFailure {
                        kind: GpuBabBoundProviderFailureKind::DeadlineExpired,
                        ..
                    })
                );
                (permits_fallback, deadline_terminal)
            });
            let counter_deadline = Instant::now() + Duration::from_secs(2);
            while backend.phase_policy_calls() == 0 && Instant::now() < counter_deadline {
                std::thread::yield_now();
            }
            if backend.phase_policy_calls() == 0 {
                drop(ledger_guard);
                let _ = open.join();
                panic!("phase policy was not reached before guard-test timeout");
            }
            while Instant::now() < deadline {
                std::hint::spin_loop();
            }
            drop(ledger_guard);
            let observation = open.join().expect("phase-policy guard test must return");
            assert_eq!(observation, (false, true));
        });
        assert_eq!(backend.accepted_open_calls(), 0);

        let registration = fresh_registration();
        let mut backend = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        backend.decline_phase_prepare = true;
        let mut descriptor = phase();
        descriptor.deadline = Instant::now() + Duration::from_millis(100);
        let deadline = descriptor.deadline;
        let ledger_guard = registration.ledger.lock().unwrap();
        std::thread::scope(|scope| {
            let backend_ref = &backend;
            let open = scope.spawn(move || {
                let disposition = GpuBabBoundPhaseLease::open(backend_ref, descriptor);
                let permits_fallback = disposition.permits_legacy_fallback();
                let deadline_terminal = matches!(
                    disposition,
                    GpuBabBoundPhaseOpen::ProviderFailure(GpuBabBoundProviderFailure {
                        kind: GpuBabBoundProviderFailureKind::DeadlineExpired,
                        ..
                    })
                );
                (permits_fallback, deadline_terminal)
            });
            let counter_deadline = Instant::now() + Duration::from_secs(2);
            while backend.phase_prepare_calls() == 0 && Instant::now() < counter_deadline {
                std::thread::yield_now();
            }
            if backend.phase_prepare_calls() == 0 {
                drop(ledger_guard);
                let _ = open.join();
                panic!("phase preparation was not reached before guard-test timeout");
            }
            while Instant::now() < deadline {
                std::hint::spin_loop();
            }
            drop(ledger_guard);
            let observation = open
                .join()
                .expect("phase-preparation guard test must return");
            assert_eq!(observation, (false, true));
        });
        assert_eq!(backend.accepted_open_calls(), 0);
    }

    #[test]
    fn open_terminals_never_turn_accepted_failures_into_clean_declines() {
        let accepted = FakeBackend::with_open_mode(OpenMode::AcceptedFailure);
        let open = GpuBabBoundPhaseLease::open(&accepted, phase());
        assert!(!open.permits_legacy_fallback());
        match open {
            GpuBabBoundPhaseOpen::AcceptedFailure(failure) => {
                assert_eq!(
                    failure.kind(),
                    GpuBabBoundTerminalFailureKind::OpenBackend(
                        GpuBabBoundBackendOpenFailureKind::Device
                    )
                );
                assert!(failure.receipt_validated());
                assert_eq!(
                    failure.receipt().released_graph_bytes,
                    failure.receipt().memory.retained_graph_bytes
                );
                assert_eq!(
                    failure.receipt().released_phase_bytes,
                    failure.receipt().memory.retained_phase_bytes
                );
            }
            _ => panic!("accepted open failure must remain terminal"),
        }

        let early = FakeBackend::with_open_mode(OpenMode::EarlyDeadline);
        match GpuBabBoundPhaseLease::open(&early, phase()) {
            GpuBabBoundPhaseOpen::AcceptedFailure(failure) => assert_eq!(
                failure.kind(),
                GpuBabBoundTerminalFailureKind::ContractViolation
            ),
            _ => panic!("early open deadline must be a contract failure"),
        };
    }

    #[test]
    fn accepted_open_failure_cleanup_never_releases_registration_authority() {
        let registration = fresh_registration();
        let mut accepted = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        accepted.open_mode = OpenMode::AcceptedFailure;
        match GpuBabBoundPhaseLease::open(&accepted, phase()) {
            GpuBabBoundPhaseOpen::AcceptedFailure(failure) => {
                assert!(failure.receipt_validated());
                assert_eq!(
                    failure.receipt().released_graph_bytes,
                    failure.receipt().memory.retained_graph_bytes
                );
            }
            _ => panic!("accepted open failure must be terminal"),
        }
        let accepted_calls = accepted.accepted_open_calls();
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&accepted, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(accepted.accepted_open_calls(), accepted_calls);
        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn accepted_open_failure_with_bad_error_or_panicking_cleanup_is_contract_failure() {
        for close_mode in [
            CloseMode::BadReceipt,
            CloseMode::AcceptedFailure,
            CloseMode::Panic,
        ] {
            let registration = fresh_registration();
            let mut backend = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
            backend.open_mode = OpenMode::AcceptedFailure;
            backend.close_mode = close_mode;
            match GpuBabBoundPhaseLease::open(&backend, phase()) {
                GpuBabBoundPhaseOpen::AcceptedFailure(failure) => {
                    assert_eq!(
                        failure.kind(),
                        GpuBabBoundTerminalFailureKind::ContractViolation
                    );
                    assert!(!failure.receipt_validated());
                }
                _ => panic!("unvalidated accepted-open cleanup must fail terminally"),
            }
            let reused = FakeBackend::with_registration(registration, vec![]);
            assert!(matches!(
                GpuBabBoundPhaseLease::open(&reused, phase()),
                GpuBabBoundPhaseOpen::AcceptedFailure(_)
            ));
            assert_eq!(reused.accepted_open_calls(), 0);
        }
    }

    #[test]
    fn terminal_open_receipts_allow_exact_zero_and_partial_allocation() {
        for mode in [
            OpenMode::AcceptedFailureZero,
            OpenMode::AcceptedFailureGraphOnly,
        ] {
            let backend = FakeBackend::with_open_mode(mode);
            match GpuBabBoundPhaseLease::open(&backend, phase()) {
                GpuBabBoundPhaseOpen::AcceptedFailure(failure) => {
                    assert_eq!(
                        failure.kind(),
                        GpuBabBoundTerminalFailureKind::OpenBackend(
                            GpuBabBoundBackendOpenFailureKind::Allocation
                        )
                    );
                    assert!(failure.receipt_validated());
                }
                _ => panic!("exact partial allocation failure must remain typed"),
            };
        }
    }

    #[test]
    fn static_payload_residency_upload_cache_and_padding_equations_are_exact() {
        let baseline = phase();
        let mut large_plan = baseline.plan;
        let parameters = large_plan
            .f32_tensors
            .iter_mut()
            .find(|tensor| tensor.role == GpuBabBoundF32TensorRole::Parameters)
            .unwrap();
        parameters.shape = vec![1_024];
        parameters.values = GpuBabBoundOwnedSlice::new(vec![0.25; 1_024]);
        let large = GpuBabBoundPhaseDescriptor::new(
            large_plan,
            Instant::now() + Duration::from_mins(1),
            32_768,
        )
        .unwrap();
        assert!(large.static_graph_payload_bytes() > 4_000);
        assert!(large.static_phase_payload_bytes() > 100);

        let static_total = large
            .static_graph_payload_bytes()
            .checked_add(large.static_phase_payload_bytes())
            .unwrap();
        let mut under_cap = large.clone();
        under_cap.max_device_bytes = static_total - 1;
        let backend = FakeBackend::new(vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&backend, under_cap),
            GpuBabBoundPhaseOpen::InvalidDescriptor(_)
        ));
        assert_eq!(backend.accepted_open_calls(), 0);

        let registration = registration_with_id(70_001);
        let (issuer, claim) = registration.claim(&large);
        claim.unwrap();
        let transcript = GpuBabBoundPhaseTranscript::expected(issuer, &large);
        let fresh = static_open_receipt_for_sources(
            transcript,
            GpuBabBoundStaticPayloadSource::FreshUpload,
            GpuBabBoundStaticPayloadSource::FreshUpload,
        );
        assert!(validate_open_receipt(issuer, &fresh, &large, false).is_ok());

        let graph_hit = GpuBabBoundStaticPayloadSource::QualifiedCacheHit {
            cache_epoch: 9,
            resident_identity_sha256: transcript.graph_identity_sha256,
        };
        let mixed = static_open_receipt_for_sources(
            transcript,
            graph_hit,
            GpuBabBoundStaticPayloadSource::FreshUpload,
        );
        assert!(validate_open_receipt(issuer, &mixed, &large, false).is_ok());
        assert_eq!(mixed.static_transfers.graph_host_to_device_bytes, 0);
        assert_eq!(
            mixed.static_transfers.phase_host_to_device_bytes,
            transcript.static_phase_payload_bytes
        );
        assert!(mixed.memory.retained_graph_bytes >= transcript.static_graph_payload_bytes);
        assert!(raw_close_matches_open(
            &GpuBabBoundBackendCloseDisposition::Closed(GpuBabBoundBackendCloseReceipt {
                transcript,
                released_graph_bytes: mixed.memory.retained_graph_bytes,
                released_phase_bytes: mixed.memory.retained_phase_bytes,
                released_resident_device_bytes: 0,
                released_resident_slots: 0,
                released_refresh_only_slots: 0,
                released_resident_logical_slots: 0,
            }),
            transcript,
            mixed.memory,
        ));

        let mut bad = fresh;
        bad.transcript.static_phase_identity_sha256 = hash(210);
        assert!(validate_open_receipt(issuer, &bad, &large, false).is_err());
        bad = fresh;
        bad.transcript.static_graph_payload_bytes += 1;
        assert!(validate_open_receipt(issuer, &bad, &large, false).is_err());
        bad = fresh;
        bad.transcript.static_phase_payload_bytes += 1;
        assert!(validate_open_receipt(issuer, &bad, &large, false).is_err());
        bad = fresh;
        bad.static_transfers.graph_identity_sha256 = hash(211);
        assert!(validate_open_receipt(issuer, &bad, &large, false).is_err());
        bad = fresh;
        bad.static_transfers.phase_identity_sha256 = hash(212);
        assert!(validate_open_receipt(issuer, &bad, &large, false).is_err());
        bad = fresh;
        bad.static_transfers.graph_payload_bytes += 1;
        assert!(validate_open_receipt(issuer, &bad, &large, false).is_err());
        bad = fresh;
        bad.static_transfers.phase_payload_bytes += 1;
        assert!(validate_open_receipt(issuer, &bad, &large, false).is_err());
        bad = fresh;
        bad.static_transfers.graph_host_to_device_bytes -= 1;
        bad.static_transfers.host_to_device_bytes -= 1;
        bad.memory.queued_upload_bytes -= 1;
        bad.memory.peak_device_bytes -= 1;
        assert!(validate_open_receipt(issuer, &bad, &large, false).is_err());
        bad = fresh;
        bad.static_transfers.graph_source = GpuBabBoundStaticPayloadSource::NotTransferred;
        bad.static_transfers.graph_host_to_device_bytes = 0;
        bad.static_transfers.host_to_device_bytes -= transcript.static_graph_payload_bytes;
        bad.memory.queued_upload_bytes -= transcript.static_graph_payload_bytes;
        bad.memory.peak_device_bytes -= transcript.static_graph_payload_bytes;
        assert!(validate_open_receipt(issuer, &bad, &large, false).is_err());

        bad = mixed;
        bad.static_transfers.graph_host_to_device_bytes = 1;
        bad.static_transfers.host_to_device_bytes += 1;
        bad.memory.queued_upload_bytes += 1;
        bad.memory.peak_device_bytes += 1;
        assert!(validate_open_receipt(issuer, &bad, &large, false).is_err());
        bad = mixed;
        bad.static_transfers.graph_source = GpuBabBoundStaticPayloadSource::QualifiedCacheHit {
            cache_epoch: 0,
            resident_identity_sha256: transcript.graph_identity_sha256,
        };
        assert!(validate_open_receipt(issuer, &bad, &large, false).is_err());
        bad = mixed;
        bad.static_transfers.graph_source = GpuBabBoundStaticPayloadSource::QualifiedCacheHit {
            cache_epoch: 9,
            resident_identity_sha256: hash(213),
        };
        assert!(validate_open_receipt(issuer, &bad, &large, false).is_err());
        bad = mixed;
        bad.memory.retained_graph_bytes -= 1;
        bad.memory.peak_device_bytes -= 1;
        assert!(validate_open_receipt(issuer, &bad, &large, false).is_err());
        bad = fresh;
        bad.static_transfers.graph_padding_bytes += 1;
        assert!(validate_open_receipt(issuer, &bad, &large, false).is_err());
        bad = fresh;
        bad.static_transfers.graph_padding_bytes = usize::MAX;
        assert!(validate_open_receipt(issuer, &bad, &large, false).is_err());

        bad = fresh;
        bad.memory.retained_graph_bytes = 1;
        bad.memory.retained_phase_bytes = 1;
        bad.memory.peak_device_bytes = 2 + bad.memory.queued_upload_bytes;
        bad.static_transfers.graph_padding_bytes = 0;
        bad.static_transfers.phase_padding_bytes = 0;
        assert!(validate_open_receipt(issuer, &bad, &large, false).is_err());
        assert!(registration.release_noalloc(issuer));
    }

    #[test]
    fn true_open_expiry_requires_valid_close_and_bad_close_poison() {
        let expiry_registration = fresh_registration();
        let mut expiring = FakeBackend::with_registration(Arc::clone(&expiry_registration), vec![]);
        expiring.open_mode = OpenMode::WaitForDeadline;
        let mut expiring_phase = phase();
        expiring_phase.deadline = Instant::now() + Duration::from_millis(100);
        match GpuBabBoundPhaseLease::open(&expiring, expiring_phase) {
            GpuBabBoundPhaseOpen::DeadlineExpired(failure) => {
                assert!(failure.receipt_validated());
            }
            _ => panic!("true open expiry with exact close must remain typed"),
        }
        let expired_reuse = FakeBackend::with_registration(expiry_registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&expired_reuse, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(expired_reuse.accepted_open_calls(), 0);

        let registration = fresh_registration();
        let mut bad_close = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        bad_close.open_mode = OpenMode::WaitForDeadline;
        bad_close.close_mode = CloseMode::BadReceipt;
        let mut expiring_phase = phase();
        expiring_phase.deadline = Instant::now() + Duration::from_millis(100);
        match GpuBabBoundPhaseLease::open(&bad_close, expiring_phase) {
            GpuBabBoundPhaseOpen::AcceptedFailure(failure) => {
                assert_eq!(
                    failure.kind(),
                    GpuBabBoundTerminalFailureKind::ContractViolation
                );
                assert!(!failure.receipt_validated());
            }
            _ => panic!("invalid expiry cleanup must be a contract failure"),
        }
        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn accepted_open_panic_is_caught_and_permanently_poisons_issuer() {
        let registration = fresh_registration();
        let mut panicking = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        panicking.open_mode = OpenMode::Panic;
        match GpuBabBoundPhaseLease::open(&panicking, phase()) {
            GpuBabBoundPhaseOpen::AcceptedFailure(failure) => {
                assert_eq!(
                    failure.kind(),
                    GpuBabBoundTerminalFailureKind::ContractViolation
                );
                assert!(!failure.receipt_validated());
            }
            _ => panic!("accepted open panic must be caught as terminal failure"),
        }
        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn corrupt_open_transcript_memory_and_generation_are_terminal() {
        for mode in [OpenMode::BadTranscript, OpenMode::BadMemory] {
            let backend = FakeBackend::with_open_mode(mode);
            match GpuBabBoundPhaseLease::open(&backend, phase()) {
                GpuBabBoundPhaseOpen::AcceptedFailure(failure) => {
                    assert_eq!(
                        failure.kind(),
                        GpuBabBoundTerminalFailureKind::ContractViolation
                    );
                    assert!(!failure.receipt_validated());
                }
                _ => panic!("corrupt accepted open must fail terminally"),
            };
        }

        assert!(GpuBabBoundBackendRegistration::new([0; 32]).is_err());

        let registration = fresh_registration();
        let mut corrupt = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        corrupt.open_mode = OpenMode::BadTranscript;
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&corrupt, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn policy_validates_joint_d_times_r_q_range() {
        let policy = GpuBabBoundPhasePolicy {
            max_device_bytes: 1,
            preferred_domains_per_wave: 4,
            minimum_domains_per_wave: 1,
            maximum_domains_per_wave: GPU_BAB_BOUND_MAX_DOMAINS,
            maximum_objectives: GPU_BAB_BOUND_MAX_OBJECTIVES,
            maximum_dispatches_per_wave: GPU_BAB_BOUND_MAX_DISPATCHES_PER_WAVE,
            maximum_submits_per_wave: GPU_BAB_BOUND_MAX_SUBMITS_PER_WAVE,
        };
        assert!(policy.is_valid_for_shape(4, 2));
        assert!(!policy.is_valid_for_shape(0, 2));
        assert!(!policy.is_valid_for_shape(GPU_BAB_BOUND_MAX_DOMAINS, GPU_BAB_BOUND_MAX_OBJECTIVES));
        let mut unbounded = policy;
        unbounded.maximum_dispatches_per_wave = usize::MAX;
        assert!(!unbounded.is_valid());
        unbounded = policy;
        unbounded.maximum_submits_per_wave = usize::MAX;
        assert!(!unbounded.is_valid());
    }

    #[test]
    fn completed_zero_tightening_is_authoritative_and_terminal() {
        let backend = FakeBackend::new(vec![Mode::CompleteZero]);
        let mut lease = open_lease(&backend, phase());
        let disposition = execute(&mut lease, request());
        assert!(!disposition.permits_legacy_fallback());
        match disposition {
            GpuBabBoundWaveDisposition::Completed(result) => {
                assert_eq!(result.rows().len(), 8);
                assert_eq!(result.domain_outcomes().len(), 4);
                assert_eq!(result.receipt().tightened_rows(), 0);
                assert_eq!(result.receipt().raw_audit_receipt().bounded_domains, 4);
                assert_eq!(result.receipt().raw_audit_receipt().pruned_domains, 0);
                assert_eq!(
                    result
                        .receipt()
                        .raw_audit_receipt()
                        .transfers
                        .result_sidecar_bytes,
                    8 * 4 * size_of::<u32>()
                );
                assert_eq!(
                    result
                        .receipt()
                        .raw_audit_receipt()
                        .transfers
                        .domain_outcome_sidecar_bytes,
                    4 * size_of::<u32>()
                );
                assert_eq!(result.receipt().transcript().wave_index, 1);
            }
            _ => panic!("zero tightening remains a completed terminal"),
        }
        assert!(matches!(
            lease.close(),
            GpuBabBoundPhaseCloseDisposition::Closed(_)
        ));
    }

    #[test]
    fn exact_tightening_count_is_validated() {
        let backend = FakeBackend::new(vec![Mode::CompleteOneTightening]);
        let mut lease = open_lease(&backend, phase());
        match execute(&mut lease, request()) {
            GpuBabBoundWaveDisposition::Completed(result) => {
                assert_eq!(result.receipt().tightened_rows(), 1);
                assert_eq!(result.rows()[0].lower(), -0.5);
            }
            _ => panic!("one exact tightening must validate"),
        }
    }

    #[test]
    fn clean_wave_decline_exists_only_before_capability_issuance() {
        let mut backend = FakeBackend::new(vec![]);
        backend.decline_prepare = true;
        let mut lease = open_lease(&backend, phase());
        let preparation = lease.prepare_wave(request());
        assert!(preparation.permits_legacy_fallback());
        assert!(matches!(
            preparation,
            GpuBabBoundWavePreparation::CleanDecline(GpuBabBoundWaveDecline::InsufficientCapacity)
        ));
    }

    #[test]
    fn prepare_panic_poison_prevents_retry_close_success_and_reopen() {
        let registration = fresh_registration();
        let mut backend = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        backend.panic_prepare = true;
        let mut lease = open_lease(&backend, phase());
        assert!(matches!(
            lease.prepare_wave(request()),
            GpuBabBoundWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendPreparePanicked
            )
        ));
        assert!(matches!(
            lease.prepare_wave(request()),
            GpuBabBoundWavePreparation::SessionTerminal(GpuBabBoundSessionTerminal::PoisonedOrBusy)
        ));
        assert!(matches!(
            lease.close(),
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure { .. }
        ));
        let accepted_calls = backend.accepted_open_calls();
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&backend, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(backend.accepted_open_calls(), accepted_calls);
        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn late_wave_prepare_decline_or_accept_is_nonfallback_and_poisons_authority() {
        for decline in [false, true] {
            let registration = fresh_registration();
            let mut backend = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
            backend.wait_prepare_deadline = true;
            backend.decline_prepare = decline;
            let mut lease = open_lease(&backend, phase());
            let mut short = request();
            short.deadline = Instant::now() + Duration::from_millis(50);
            let preparation = lease.prepare_wave(short);
            assert!(!preparation.permits_legacy_fallback());
            assert!(matches!(
                &preparation,
                GpuBabBoundWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired
                )
            ));
            drop(preparation);
            assert!(matches!(
                lease.prepare_wave(request()),
                GpuBabBoundWavePreparation::SessionTerminal(
                    GpuBabBoundSessionTerminal::PoisonedOrBusy
                )
            ));
            assert!(matches!(
                lease.close(),
                GpuBabBoundPhaseCloseDisposition::AcceptedFailure { .. }
            ));
            let reused = FakeBackend::with_registration(registration, vec![]);
            assert!(matches!(
                GpuBabBoundPhaseLease::open(&reused, phase()),
                GpuBabBoundPhaseOpen::AcceptedFailure(_)
            ));
            assert_eq!(reused.accepted_open_calls(), 0);
        }
    }

    #[test]
    fn delayed_accepted_capability_expires_before_raw_execution() {
        let registration = fresh_registration();
        let backend =
            FakeBackend::with_registration(Arc::clone(&registration), vec![Mode::CompleteZero]);
        let mut lease = open_lease(&backend, phase());
        let mut short = request();
        short.deadline = Instant::now() + Duration::from_millis(250);
        let deadline = short.deadline;
        let capability = match lease.prepare_wave(short) {
            GpuBabBoundWavePreparation::Accepted(capability) => capability,
            _ => panic!("timely request must issue one accepted capability"),
        };
        while Instant::now() < deadline {
            std::hint::spin_loop();
        }
        let disposition = capability.execute_accepted();
        assert!(!disposition.permits_legacy_fallback());
        match disposition {
            GpuBabBoundWaveDisposition::DeadlineExpired(failure) => {
                assert_eq!(
                    failure.kind(),
                    GpuBabBoundTerminalFailureKind::Backend(
                        GpuBabBoundBackendFailureKind::AuthorityLost
                    )
                );
                assert!(failure.receipt_validated());
                assert_eq!(failure.receipt().dispatches, 0);
                assert_eq!(failure.receipt().submits, 0);
                assert_eq!(failure.receipt().transfers.host_to_device_bytes, 0);
                assert_eq!(failure.receipt().transfers.device_to_host_bytes, 0);
                assert_eq!(failure.receipt().transfers.readbacks, 0);
                assert_eq!(failure.receipt().transfers.synchronizations, 0);
                assert_eq!(failure.receipt().memory.result_readback_bytes, 0);
            }
            _ => panic!("expired accepted capability must return typed deadline terminal"),
        }
        assert_eq!(backend.execute_calls(), 0);
        assert!(matches!(
            lease.prepare_wave(request()),
            GpuBabBoundWavePreparation::SessionTerminal(GpuBabBoundSessionTerminal::PoisonedOrBusy)
        ));
        assert!(matches!(
            lease.close(),
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure { .. }
        ));

        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn expired_wave_request_is_terminal_before_raw_prepare() {
        let registration = fresh_registration();
        let backend = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        let mut lease = open_lease(&backend, phase());
        let mut expired = request();
        expired.deadline = Instant::now();
        assert!(matches!(
            lease.prepare_wave(expired),
            GpuBabBoundWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired
            )
        ));
        assert_eq!(backend.prepare_wave_calls(), 0);
        assert!(matches!(
            lease.close(),
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure { .. }
        ));
        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn accepted_failure_is_typed_validated_and_poisoning() {
        let registration = fresh_registration();
        let backend =
            FakeBackend::with_registration(Arc::clone(&registration), vec![Mode::AcceptedFailure]);
        let mut lease = open_lease(&backend, phase());
        match execute(&mut lease, request()) {
            GpuBabBoundWaveDisposition::AcceptedFailure(failure) => {
                assert_eq!(
                    failure.kind(),
                    GpuBabBoundTerminalFailureKind::Backend(GpuBabBoundBackendFailureKind::Device)
                );
                assert!(failure.receipt_validated());
                assert_eq!(failure.receipt().transcript.wave_index, 1);
            }
            _ => panic!("accepted failure must remain typed"),
        }
        assert!(matches!(
            lease.prepare_wave(request()),
            GpuBabBoundWavePreparation::SessionTerminal(GpuBabBoundSessionTerminal::PoisonedOrBusy)
        ));
        assert!(matches!(
            lease.close(),
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure { .. }
        ));
        let accepted_calls = backend.accepted_open_calls();
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&backend, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(backend.accepted_open_calls(), accepted_calls);
        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn accepted_failure_revalidates_echo_and_parent_atomicity() {
        assert_contract_failure(Mode::AcceptedFailureBadEcho);
        assert_contract_failure(Mode::AcceptedFailurePartialParent);
        assert_contract_failure(Mode::AcceptedFailureD2hWithoutDispatch);
        assert_contract_failure(Mode::AcceptedFailureTightening);
        assert_contract_failure(Mode::AcceptedFailurePartialTypedUpload);
        assert_contract_failure(Mode::AcceptedFailureFullPrefixZeroWork);
        assert_contract_failure(Mode::AcceptedFailureDispatchesMax);
        assert_contract_failure(Mode::AcceptedFailureSubmitsMax);
    }

    #[test]
    fn postdispatch_failure_requires_and_accepts_exact_attempted_work() {
        let backend = FakeBackend::new(vec![Mode::AcceptedFailurePostdispatch]);
        let mut lease = open_lease(&backend, phase());
        match execute(&mut lease, request()) {
            GpuBabBoundWaveDisposition::AcceptedFailure(failure) => {
                assert!(failure.receipt_validated());
                assert_eq!(failure.receipt().completed_parent_groups, 0);
                assert_eq!(failure.receipt().completed_domains, 0);
                assert_eq!(failure.receipt().completed_rows, 0);
                assert_eq!(failure.receipt().dispatches, 1);
                assert_eq!(failure.receipt().submits, 1);
                assert!(failure.receipt().transfers.host_to_device_bytes > 0);
            }
            _ => panic!("exact attempted postdispatch work must remain typed failure"),
        }
    }

    #[test]
    fn postaccept_decline_is_rejected_without_restoring_fallback() {
        assert_contract_failure(Mode::IllegalPostacceptDecline);
    }

    #[test]
    fn early_deadline_is_rejected_but_true_expiry_is_typed() {
        assert_contract_failure(Mode::EarlyDeadline);

        let backend = FakeBackend::new(vec![Mode::WaitForDeadline]);
        let mut lease = open_lease(&backend, phase());
        let mut short = request();
        short.deadline = Instant::now() + Duration::from_millis(100);
        match execute(&mut lease, short) {
            GpuBabBoundWaveDisposition::DeadlineExpired(failure) => {
                assert!(failure.receipt_validated());
            }
            _ => panic!("actually expired deadline must remain typed"),
        }
    }

    #[test]
    fn late_completed_payload_is_terminal_contract_failure() {
        let backend = FakeBackend::new(vec![Mode::WaitThenComplete]);
        let mut lease = open_lease(&backend, phase());
        let mut short = request();
        short.deadline = Instant::now() + Duration::from_millis(100);
        match execute(&mut lease, short) {
            GpuBabBoundWaveDisposition::AcceptedFailure(failure) => assert_eq!(
                failure.kind(),
                GpuBabBoundTerminalFailureKind::ContractViolation
            ),
            _ => panic!("late completed bounds can never publish"),
        }
    }

    #[test]
    fn abandoned_capability_poisoning_prevents_reuse() {
        let backend = FakeBackend::new(vec![Mode::CompleteZero]);
        let mut lease = open_lease(&backend, phase());
        let capability = match lease.prepare_wave(request()) {
            GpuBabBoundWavePreparation::Accepted(capability) => capability,
            _ => panic!("valid fake wave must accept"),
        };
        drop(capability);
        let abandoned = lease
            .abandoned_terminal()
            .expect("drop must leave a core-owned terminal receipt");
        assert_eq!(
            abandoned.kind(),
            GpuBabBoundTerminalFailureKind::CapabilityAbandoned
        );
        assert!(abandoned.receipt_validated());
        assert_eq!(abandoned.receipt().transfers.host_to_device_bytes, 0);
        assert!(matches!(
            lease.prepare_wave(request()),
            GpuBabBoundWavePreparation::SessionTerminal(GpuBabBoundSessionTerminal::PoisonedOrBusy)
        ));
    }

    #[test]
    fn dropped_accepted_capability_close_and_reopen_fail_closed() {
        let registration = fresh_registration();
        let backend =
            FakeBackend::with_registration(Arc::clone(&registration), vec![Mode::CompleteZero]);
        let mut lease = open_lease(&backend, phase());
        let capability = match lease.prepare_wave(request()) {
            GpuBabBoundWavePreparation::Accepted(capability) => capability,
            _ => panic!("valid fake wave must accept"),
        };
        drop(capability);
        assert!(matches!(
            lease.close(),
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure { .. }
        ));
        let accepted_calls = backend.accepted_open_calls();
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&backend, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(backend.accepted_open_calls(), accepted_calls);
        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn forgotten_capability_cannot_close_as_success() {
        let registration = fresh_registration();
        let backend =
            FakeBackend::with_registration(Arc::clone(&registration), vec![Mode::CompleteZero]);
        let mut lease = open_lease(&backend, phase());
        let capability = match lease.prepare_wave(request()) {
            GpuBabBoundWavePreparation::Accepted(capability) => capability,
            _ => panic!("valid fake wave must accept"),
        };
        std::mem::forget(capability);
        assert!(matches!(
            lease.close(),
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure { .. }
        ));
        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn forgotten_capability_then_implicit_drop_cannot_restore_authority() {
        let registration = fresh_registration();
        let backend =
            FakeBackend::with_registration(Arc::clone(&registration), vec![Mode::CompleteZero]);
        let mut lease = open_lease(&backend, phase());
        let capability = match lease.prepare_wave(request()) {
            GpuBabBoundWavePreparation::Accepted(capability) => capability,
            _ => panic!("valid fake wave must accept"),
        };
        std::mem::forget(capability);
        drop(lease);

        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn backend_unwind_after_acceptance_is_caught_and_never_claims_a_valid_receipt() {
        let backend = FakeBackend::new(vec![Mode::PanicAfterAccept]);
        let mut lease = open_lease(&backend, phase());
        let capability = match lease.prepare_wave(request()) {
            GpuBabBoundWavePreparation::Accepted(capability) => capability,
            _ => panic!("valid fake wave must accept"),
        };
        match capability.execute_accepted() {
            GpuBabBoundWaveDisposition::AcceptedFailure(failure) => {
                assert_eq!(
                    failure.kind(),
                    GpuBabBoundTerminalFailureKind::ContractViolation
                );
                assert!(!failure.receipt_validated());
            }
            _ => panic!("backend unwind must become a typed contract failure"),
        }
        assert!(lease.abandoned_terminal().is_none());
        assert!(matches!(
            lease.prepare_wave(request()),
            GpuBabBoundWavePreparation::SessionTerminal(GpuBabBoundSessionTerminal::PoisonedOrBusy)
        ));
    }

    #[test]
    fn wave_numbers_are_nonzero_monotonic_and_replay_echo_is_rejected() {
        let backend = FakeBackend::new(vec![Mode::CompleteZero, Mode::ReplayWaveOne]);
        let mut lease = open_lease(&backend, phase());
        match execute(&mut lease, request()) {
            GpuBabBoundWaveDisposition::Completed(result) => {
                assert_eq!(result.receipt().transcript().wave_index, 1);
            }
            _ => panic!("first wave must complete"),
        }
        match execute(&mut lease, request()) {
            GpuBabBoundWaveDisposition::AcceptedFailure(failure) => {
                assert_eq!(
                    failure.kind(),
                    GpuBabBoundTerminalFailureKind::ContractViolation
                );
            }
            _ => panic!("replayed wave echo must fail terminally"),
        }
    }

    #[test]
    fn wave_index_never_wraps_or_reuses_zero() {
        let backend = FakeBackend::new(vec![]);
        let mut lease = open_lease(&backend, phase());
        lease.last_wave_index = u64::MAX;
        assert!(matches!(
            lease.prepare_wave(request()),
            GpuBabBoundWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::WaveSequenceExhausted
            )
        ));
    }

    #[test]
    fn core_registration_concurrent_claim_burns_and_permanently_poisons() {
        let registration = fresh_registration();
        let first = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        let first_lease = open_lease(&first, phase());
        let first_identity = first_lease.transcript.backend;
        assert_eq!(first_identity.generation, 1);

        let concurrent = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        match GpuBabBoundPhaseLease::open(&concurrent, phase()) {
            GpuBabBoundPhaseOpen::AcceptedFailure(failure) => assert_eq!(
                failure.kind(),
                GpuBabBoundTerminalFailureKind::ContractViolation
            ),
            _ => panic!("same issuer cannot own two live sessions"),
        }
        assert_eq!(
            concurrent.accepted_open_calls(),
            0,
            "one-live rejection must occur before raw allocation"
        );
        assert!(matches!(
            first_lease.close(),
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure { .. }
        ));

        let next = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&next, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(next.accepted_open_calls(), 0);
        let ledger = registration.ledger.lock().unwrap();
        assert!(ledger.poisoned);
        assert_eq!(ledger.highest_generation, 3);
        assert_ne!(first_identity.generation, ledger.highest_generation);
    }

    #[test]
    fn concurrent_claim_revokes_open_lease_before_raw_prepare() {
        let registration = fresh_registration();
        let first =
            FakeBackend::with_registration(Arc::clone(&registration), vec![Mode::CompleteZero]);
        let mut lease = open_lease(&first, phase());

        let concurrent = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&concurrent, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(concurrent.accepted_open_calls(), 0);
        assert!(matches!(
            lease.prepare_wave(request()),
            GpuBabBoundWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::RegistrationAuthorityLost
            )
        ));
        assert_eq!(first.prepare_wave_calls(), 0);
        assert!(matches!(
            lease.close(),
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure { .. }
        ));
    }

    #[test]
    fn concurrent_claim_after_capability_acceptance_prevents_raw_execution() {
        let registration = fresh_registration();
        let first =
            FakeBackend::with_registration(Arc::clone(&registration), vec![Mode::CompleteZero]);
        let mut lease = open_lease(&first, phase());
        let capability = match lease.prepare_wave(request()) {
            GpuBabBoundWavePreparation::Accepted(capability) => capability,
            _ => panic!("valid wave must receive a capability"),
        };

        let concurrent = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&concurrent, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        match capability.execute_accepted() {
            GpuBabBoundWaveDisposition::AcceptedFailure(failure) => {
                assert_eq!(
                    failure.kind(),
                    GpuBabBoundTerminalFailureKind::ContractViolation
                );
                assert!(!failure.receipt_validated());
            }
            _ => panic!("revoked capability cannot execute or publish bounds"),
        }
        assert_eq!(first.execute_calls(), 0);
        assert!(matches!(
            lease.close(),
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure { .. }
        ));
    }

    #[test]
    fn concurrent_claim_during_raw_execution_blocks_validated_publication() {
        let registration = fresh_registration();
        let gate = TransitionGate::new();
        let mut first =
            FakeBackend::with_registration(Arc::clone(&registration), vec![Mode::CompleteZero]);
        first.execute_gate = Some(gate.clone());
        let mut lease = open_lease(&first, phase());
        let capability = match lease.prepare_wave(request()) {
            GpuBabBoundWavePreparation::Accepted(capability) => capability,
            _ => panic!("valid wave must receive a capability"),
        };

        std::thread::scope(|scope| {
            let execution = scope.spawn(move || capability.execute_accepted());
            gate.entered.wait();
            let (completed_tx, completed_rx) = std::sync::mpsc::channel();
            let concurrent_registration = Arc::clone(&registration);
            let concurrent = scope.spawn(move || {
                let backend = FakeBackend::with_registration(concurrent_registration, vec![]);
                let rejected = matches!(
                    GpuBabBoundPhaseLease::open(&backend, phase()),
                    GpuBabBoundPhaseOpen::AcceptedFailure(_)
                );
                let observation = (rejected, backend.accepted_open_calls());
                let _ = completed_tx.send(observation);
                observation
            });
            let concurrent_observation = completed_rx.recv_timeout(Duration::from_secs(2));
            if concurrent_observation.is_err() {
                // Always release the raw gate before joining, converting a
                // forbidden ledger-lock-across-TCB-call regression into a
                // deterministic assertion instead of a hung test process.
                gate.resume.wait();
                let _ = execution.join();
                let _ = concurrent.join();
                panic!("concurrent claim blocked behind raw backend execution");
            }
            assert_eq!(concurrent_observation.unwrap(), (true, 0));
            gate.resume.wait();
            match execution.join().expect("raw execution thread must return") {
                GpuBabBoundWaveDisposition::AcceptedFailure(failure) => {
                    assert_eq!(
                        failure.kind(),
                        GpuBabBoundTerminalFailureKind::ContractViolation
                    );
                    assert!(!failure.receipt_validated());
                }
                _ => panic!("authority loss during raw execution cannot publish bounds"),
            }
            assert_eq!(
                concurrent
                    .join()
                    .expect("concurrent claim thread must return"),
                (true, 0)
            );
        });
        assert_eq!(first.execute_calls(), 1);
        assert!(matches!(
            lease.close(),
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure { .. }
        ));
    }

    #[test]
    fn same_configured_issuer_has_distinct_core_epochs_nonces_and_no_cross_replay() {
        let first_registration = registration_with_id(88_001);
        let second_registration = registration_with_id(88_001);
        assert_ne!(
            first_registration.registration_epoch(),
            second_registration.registration_epoch()
        );
        let first_backend = FakeBackend::with_registration(Arc::clone(&first_registration), vec![]);
        let second_backend =
            FakeBackend::with_registration(Arc::clone(&second_registration), vec![]);
        let first_lease = open_lease(&first_backend, phase());
        let second_lease = open_lease(&second_backend, phase());
        let first_identity = first_lease.transcript.backend;
        let second_identity = second_lease.transcript.backend;
        assert_eq!(
            first_identity.backend_issuer_sha256,
            second_identity.backend_issuer_sha256
        );
        assert_eq!(first_identity.generation, 1);
        assert_eq!(second_identity.generation, 1);
        assert_ne!(
            first_identity.registration_epoch,
            second_identity.registration_epoch
        );
        assert_ne!(
            first_identity.session_nonce_sha256,
            second_identity.session_nonce_sha256
        );

        let request = request();
        let shape = request.validate_static(&first_lease.phase).unwrap();
        let transcript = GpuBabBoundTerminalTranscript {
            phase: first_lease.transcript,
            wave_index: 1,
            schedule_identity_sha256: shape.schedule_identity_sha256,
            inherited_endpoints_sha256: shape.inherited_endpoints_sha256,
            deadline: request.deadline,
            max_device_bytes: request.max_device_bytes,
        };
        let mut cross_receipt =
            core_predispatch_failure_receipt(&request, shape, transcript, first_lease.open_memory);
        cross_receipt.transcript.phase.backend = second_identity;
        assert!(validate_failure_receipt(
            &cross_receipt,
            &request,
            shape,
            transcript,
            first_lease.open_memory,
            first_lease.policy,
        )
        .is_err());

        let mut cross_open = static_open_receipt_for_sources(
            first_lease.transcript,
            GpuBabBoundStaticPayloadSource::FreshUpload,
            GpuBabBoundStaticPayloadSource::FreshUpload,
        );
        cross_open.transcript.backend = second_identity;
        assert!(
            validate_open_receipt(first_identity, &cross_open, &first_lease.phase, false,).is_err()
        );
        assert!(matches!(
            first_lease.close(),
            GpuBabBoundPhaseCloseDisposition::Closed(_)
        ));
        assert!(matches!(
            second_lease.close(),
            GpuBabBoundPhaseCloseDisposition::Closed(_)
        ));
    }

    #[test]
    fn process_registration_epoch_counter_never_wraps_or_uses_zero() {
        let zero = AtomicU64::new(0);
        assert!(next_registration_epoch(&zero).is_err());
        assert_eq!(zero.load(Ordering::SeqCst), 0);

        let last = AtomicU64::new(u64::MAX - 1);
        assert_eq!(next_registration_epoch(&last).unwrap(), u64::MAX - 1);
        assert_eq!(last.load(Ordering::SeqCst), u64::MAX);
        assert!(next_registration_epoch(&last).is_err());
        assert_eq!(last.load(Ordering::SeqCst), u64::MAX);
    }

    #[test]
    fn implicit_drop_permanently_poisons_the_issuer() {
        let registration = fresh_registration();
        let first = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        drop(open_lease(&first, phase()));
        let second = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&second, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(second.accepted_open_calls(), 0);
    }

    #[test]
    fn core_derived_session_tickets_never_reuse_backend_supplied_a_b_a_values() {
        let registration = fresh_registration();
        let mut identities = Vec::new();
        for expected_generation in 1..=3 {
            let backend = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
            let lease = open_lease(&backend, phase());
            let identity = lease.transcript.backend;
            assert_eq!(identity.generation, expected_generation);
            assert!(!identities
                .iter()
                .any(|prior: &GpuBabBoundBackendIssuerIdentity| {
                    prior.session_nonce_sha256 == identity.session_nonce_sha256
                }));
            identities.push(identity);
            assert!(matches!(
                lease.close(),
                GpuBabBoundPhaseCloseDisposition::Closed(_)
            ));
        }
    }

    #[test]
    fn maximum_generation_poison_is_permanent_and_never_allocates() {
        let registration = fresh_registration();
        registration.ledger.lock().unwrap().highest_generation = u64::MAX - 1;
        let maximum = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&maximum, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(maximum.accepted_open_calls(), 0);

        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn poisoned_registration_mutex_is_absorbing_and_never_allocates() {
        let registration = fresh_registration();
        let poison_target = Arc::clone(&registration);
        assert!(std::thread::spawn(move || {
            let _guard = poison_target.ledger.lock().unwrap();
            panic!("deterministically poison the registration mutex");
        })
        .join()
        .is_err());

        let backend = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&backend, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(backend.accepted_open_calls(), 0);
    }

    #[test]
    fn bad_error_panic_and_missing_close_each_poison_reuse() {
        for close_mode in [
            CloseMode::BadReceipt,
            CloseMode::AcceptedFailure,
            CloseMode::Panic,
        ] {
            let registration = fresh_registration();
            let mut first = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
            first.close_mode = close_mode;
            let lease = open_lease(&first, phase());
            assert!(matches!(
                lease.close(),
                GpuBabBoundPhaseCloseDisposition::AcceptedFailure { .. }
            ));
            let reused = FakeBackend::with_registration(registration, vec![]);
            assert!(matches!(
                GpuBabBoundPhaseLease::open(&reused, phase()),
                GpuBabBoundPhaseOpen::AcceptedFailure(_)
            ));
            assert_eq!(reused.accepted_open_calls(), 0);
        }

        let registration = fresh_registration();
        let missing = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        let mut lease = open_lease(&missing, phase());
        let orphaned_session = lease.session.take();
        assert!(matches!(
            lease.close(),
            GpuBabBoundPhaseCloseDisposition::AcceptedFailure { receipt: None, .. }
        ));
        drop(orphaned_session);
        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn close_method_and_session_drop_panics_are_separately_contained() {
        for close_mode in [CloseMode::Closed, CloseMode::Panic] {
            let registration = fresh_registration();
            let mut backend = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
            backend.close_mode = close_mode;
            backend.panic_session_drop = true;
            let lease = open_lease(&backend, phase());
            match lease.close() {
                GpuBabBoundPhaseCloseDisposition::AcceptedFailure {
                    receipt_validated, ..
                } => assert!(!receipt_validated),
                _ => panic!("a panicking session destructor cannot release authority"),
            }
            assert_eq!(backend.close_calls.load(Ordering::Relaxed), 1);
            assert_eq!(backend.session_drop_calls.load(Ordering::Relaxed), 1);

            let reused = FakeBackend::with_registration(registration, vec![]);
            assert!(matches!(
                GpuBabBoundPhaseLease::open(&reused, phase()),
                GpuBabBoundPhaseOpen::AcceptedFailure(_)
            ));
            assert_eq!(reused.accepted_open_calls(), 0);
        }
    }

    #[test]
    fn explicit_close_orders_poison_or_live_ownership_before_reentrant_drop() {
        for close_mode in [
            CloseMode::AcceptedFailure,
            CloseMode::Panic,
            CloseMode::Closed,
        ] {
            let registration = fresh_registration();
            let nested_terminal = Arc::new(AtomicBool::new(false));
            let nested_accepted_open_calls = Arc::new(AtomicUsize::new(usize::MAX));
            let mut backend = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
            backend.close_mode = close_mode;
            backend.session_drop_reentry = Some(DropReentry {
                registration: Arc::clone(&registration),
                nested_terminal: Arc::clone(&nested_terminal),
                nested_accepted_open_calls: Arc::clone(&nested_accepted_open_calls),
            });
            let lease = open_lease(&backend, phase());
            assert!(matches!(
                lease.close(),
                GpuBabBoundPhaseCloseDisposition::AcceptedFailure { .. }
            ));
            assert!(nested_terminal.load(Ordering::Relaxed));
            assert_eq!(nested_accepted_open_calls.load(Ordering::Relaxed), 0);
            assert!(registration.ledger.lock().unwrap().poisoned);

            let reused = FakeBackend::with_registration(registration, vec![]);
            assert!(matches!(
                GpuBabBoundPhaseLease::open(&reused, phase()),
                GpuBabBoundPhaseOpen::AcceptedFailure(_)
            ));
            assert_eq!(reused.accepted_open_calls(), 0);
        }
    }

    #[test]
    fn outer_unwind_forgets_session_after_poison_without_tcb_cleanup() {
        let registration = fresh_registration();
        let backend = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _lease = open_lease(&backend, phase());
            panic!("outer caller unwind");
        }));
        assert!(result.is_err());
        assert_eq!(backend.close_calls.load(Ordering::Relaxed), 0);
        assert_eq!(backend.session_drop_calls.load(Ordering::Relaxed), 0);

        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    #[test]
    fn prepared_session_slot_outer_unwind_forgets_before_lease_construction() {
        let registration = fresh_registration();
        let backend = FakeBackend::with_registration(Arc::clone(&registration), vec![]);
        let descriptor = phase();
        let invocation = GpuBabBoundTcbInvocation {
            descriptor: &descriptor,
        };
        let session = match backend.prepare_phase(&invocation) {
            GpuBabBoundBackendOpenPreparation::Prepared { session } => session,
            GpuBabBoundBackendOpenPreparation::CleanDecline(_) => {
                panic!("fake backend must create a dormant session")
            }
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _slot = GpuBabBoundPreparedSessionSlot::new(session, registration.as_ref());
            panic!("core-side unwind before lease construction");
        }));
        assert!(result.is_err());
        assert_eq!(backend.close_calls.load(Ordering::Relaxed), 0);
        assert_eq!(backend.session_drop_calls.load(Ordering::Relaxed), 0);
        assert!(registration.ledger.lock().unwrap().poisoned);

        let reused = FakeBackend::with_registration(registration, vec![]);
        assert!(matches!(
            GpuBabBoundPhaseLease::open(&reused, phase()),
            GpuBabBoundPhaseOpen::AcceptedFailure(_)
        ));
        assert_eq!(reused.accepted_open_calls(), 0);
    }

    fn static_error(
        request: &GpuBabBoundWaveRequest,
        phase: &GpuBabBoundPhaseDescriptor,
    ) -> String {
        request.validate_static(phase).unwrap_err().to_string()
    }

    #[test]
    fn parent_groups_reject_missing_duplicate_permuted_overlap_gap_and_cross_parent() {
        let phase = phase();

        let mut candidate = request();
        candidate.parent_groups.pop();
        assert!(static_error(&candidate, &phase).contains("cover"));

        candidate = request();
        candidate.parent_groups[1].parent_group_id = 10;
        assert!(static_error(&candidate, &phase).contains("strictly ascending"));

        candidate = request();
        let duplicate_parent = candidate.parent_groups[0].parent_identity_sha256;
        candidate.parent_groups[1].parent_identity_sha256 = duplicate_parent;
        assert!(static_error(&candidate, &phase).contains("parent identity is duplicated"));

        candidate = request();
        candidate.parent_groups.swap(0, 1);
        assert!(static_error(&candidate, &phase).contains("gap, overlap"));

        candidate = request();
        candidate.parent_groups[1].first_domain = 1;
        assert!(static_error(&candidate, &phase).contains("gap, overlap"));

        candidate = request();
        candidate.parent_groups[1].first_domain = 3;
        assert!(static_error(&candidate, &phase).contains("gap, overlap"));

        candidate = request();
        candidate.domains[2].parent_group_id = 10;
        assert!(static_error(&candidate, &phase).contains("does not exactly echo parent"));

        candidate = request();
        candidate.domains.swap(0, 1);
        assert!(static_error(&candidate, &phase).contains("ordinal"));

        candidate = request();
        candidate.domains[1].child_ordinal = 0;
        assert!(static_error(&candidate, &phase).contains("ordinal"));

        candidate = request();
        candidate.domains[1].child_cardinality = 3;
        assert!(static_error(&candidate, &phase).contains("cardinality"));

        candidate = request();
        candidate.subchunks[0].domain_count = 3;
        candidate.subchunks[0].row_count = 6;
        candidate.subchunks[1].first_domain = 3;
        candidate.subchunks[1].first_q = 6;
        candidate.subchunks[1].domain_count = 1;
        candidate.subchunks[1].row_count = 2;
        assert!(static_error(&candidate, &phase).contains("crosses a parent boundary"));
    }

    #[test]
    fn group_and_subchunk_partitions_reject_missing_duplicate_q_gap_and_overlap() {
        let phase = phase();
        let mut candidate = request();
        candidate.domains.pop();
        assert!(static_error(&candidate, &phase).contains("exceeds the domain array"));

        candidate = request();
        candidate.domains[1].domain_slot = candidate.domains[0].domain_slot;
        assert!(static_error(&candidate, &phase).contains("duplicated"));

        candidate = request();
        candidate.domains[0].domain_slot = 0;
        assert!(static_error(&candidate, &phase).contains("must be nonzero"));

        candidate = request();
        candidate.domains[1].operands.activation.start += 1;
        assert!(static_error(&candidate, &phase).contains("gap/overlap"));

        candidate = request();
        candidate.subchunks.pop();
        assert!(static_error(&candidate, &phase).contains("cover every domain"));

        candidate = request();
        candidate.subchunks[1].first_domain = 1;
        assert!(static_error(&candidate, &phase).contains("gap or overlap"));

        candidate = request();
        candidate.subchunks[1].first_q = 3;
        assert!(static_error(&candidate, &phase).contains("gap or overlap"));

        candidate = request();
        candidate.subchunks[0].row_count = 3;
        assert!(static_error(&candidate, &phase).contains("domain_count * R"));
    }

    #[test]
    fn objective_union_and_inherited_intervals_are_canonical() {
        let phase = phase();
        let mut candidate = request();
        candidate.objective_indices.clear();
        assert!(static_error(&candidate, &phase).contains("objective union"));

        candidate = request();
        candidate.objective_indices = vec![1, 1];
        assert!(static_error(&candidate, &phase).contains("strictly ascending"));

        candidate = request();
        candidate.objective_indices = vec![3, 1];
        assert!(static_error(&candidate, &phase).contains("strictly ascending"));

        candidate = request();
        candidate.inherited_lower[0] = f32::NAN;
        assert!(static_error(&candidate, &phase).contains("finite ordered"));

        candidate = request();
        candidate.inherited_lower[0] = 2.0;
        candidate.inherited_upper[0] = 1.0;
        assert!(static_error(&candidate, &phase).contains("finite ordered"));
    }

    #[test]
    fn typed_domain_arena_rejects_gap_overlap_range_nonfinite_abs_and_box_corruption() {
        let phase = phase();

        let mut candidate = request();
        candidate.domains[1].operands.activation.start += 1;
        assert!(static_error(&candidate, &phase).contains("gap/overlap"));

        let mut candidate = request();
        candidate.domains[3].operands.cached_la.len += 1;
        assert!(static_error(&candidate, &phase).contains("exceeds arena length"));

        let mut candidate = request();
        candidate.domain_arena.activation =
            mutate_owned_slice(&candidate.domain_arena.activation, |values| {
                values[0] = f32::NAN;
            });
        assert!(static_error(&candidate, &phase).contains("finite"));

        let mut candidate = request();
        candidate.domain_arena.abs = mutate_owned_slice(&candidate.domain_arena.abs, |values| {
            values[0] = -0.25;
        });
        assert!(static_error(&candidate, &phase).contains("nonnegative"));

        let mut candidate = request();
        candidate.domain_arena.box_lower =
            mutate_owned_slice(&candidate.domain_arena.box_lower, |values| {
                values[0] = 2.0;
            });
        assert!(static_error(&candidate, &phase).contains("finite ordered"));
    }

    #[test]
    fn canonical_schedule_rebinds_every_declared_input_class() {
        let phase = phase();
        let baseline = request();
        let baseline_hash = baseline.canonical_schedule_sha256(&phase).unwrap();
        let baseline_endpoints = hash_inherited_endpoints(&baseline);

        let mut variants = Vec::new();
        let mut changed = baseline.clone();
        changed.parent_groups[0].parent_identity_sha256 = hash(201);
        variants.push(changed);
        let mut changed = baseline.clone();
        changed.domains[0].domain_slot += 100;
        variants.push(changed);
        let mut changed = baseline.clone();
        changed.domain_arena.activation =
            mutate_owned_slice(&changed.domain_arena.activation, |values| {
                values[0] += 0.25;
            });
        variants.push(changed);
        let mut changed = baseline.clone();
        changed.domain_arena.beta = mutate_owned_slice(&changed.domain_arena.beta, |values| {
            values[0] += 0.25;
        });
        variants.push(changed);
        let mut changed = baseline.clone();
        changed.domain_arena.abs = mutate_owned_slice(&changed.domain_arena.abs, |values| {
            values[0] += 0.25;
        });
        variants.push(changed);
        let mut changed = baseline.clone();
        changed.domain_arena.box_lower =
            mutate_owned_slice(&changed.domain_arena.box_lower, |values| {
                values[0] += 0.25;
            });
        variants.push(changed);
        let mut changed = baseline.clone();
        changed.domain_arena.box_upper =
            mutate_owned_slice(&changed.domain_arena.box_upper, |values| {
                values[0] -= 0.25;
            });
        variants.push(changed);
        let mut changed = baseline.clone();
        changed.domain_arena.cached_la =
            mutate_owned_slice(&changed.domain_arena.cached_la, |values| {
                values[0] += 0.25;
            });
        variants.push(changed);
        let mut changed = baseline.clone();
        changed.objective_indices = vec![1, 4];
        variants.push(changed);
        let mut changed = baseline.clone();
        changed.subchunks = vec![
            GpuBabBoundSubchunk {
                parent_group_id: 10,
                first_domain: 0,
                domain_count: 1,
                first_q: 0,
                row_count: 2,
            },
            GpuBabBoundSubchunk {
                parent_group_id: 10,
                first_domain: 1,
                domain_count: 1,
                first_q: 2,
                row_count: 2,
            },
            baseline.subchunks[1],
        ];
        variants.push(changed);
        let mut changed = baseline.clone();
        changed.inherited_lower[0] = -0.75;
        variants.push(changed);
        let mut changed = baseline.clone();
        changed.max_device_bytes -= 1;
        variants.push(changed);

        for variant in variants {
            assert_ne!(
                variant.canonical_schedule_sha256(&phase).unwrap(),
                baseline_hash
            );
        }
        let mut endpoint_change = baseline;
        endpoint_change.inherited_upper[0] = 0.75;
        assert_ne!(
            hash_inherited_endpoints(&endpoint_change),
            baseline_endpoints
        );
    }

    #[test]
    fn phase_authority_is_computed_from_exact_topology_and_typed_tensor_bits() {
        let baseline = phase();
        let baseline_authority = baseline.authority;

        let mutate = |mut plan: GpuBabBoundGraphPlan,
                      role: GpuBabBoundF32TensorRole,
                      value: f32| {
            let tensor = plan
                .f32_tensors
                .iter_mut()
                .find(|tensor| tensor.role == role)
                .unwrap();
            tensor.values = mutate_owned_slice(&tensor.values, |values| values[0] = value);
            GpuBabBoundPhaseDescriptor::new(plan, Instant::now() + Duration::from_mins(1), 8_192)
                .unwrap()
        };

        let mut topology_plan = baseline.plan().clone();
        topology_plan.topology_bytes =
            mutate_owned_slice(&topology_plan.topology_bytes, |values| values[0] ^= 0x80);
        let topology = GpuBabBoundPhaseDescriptor::new(
            topology_plan,
            Instant::now() + Duration::from_mins(1),
            8_192,
        )
        .unwrap();
        assert_ne!(
            topology.authority.graph_identity_sha256,
            baseline_authority.graph_identity_sha256
        );

        let mut schema_plan = baseline.plan().clone();
        schema_plan.topology_schema_version += 1;
        let schema = GpuBabBoundPhaseDescriptor::new(
            schema_plan,
            Instant::now() + Duration::from_mins(1),
            8_192,
        )
        .unwrap();
        assert_ne!(
            schema.authority.graph_identity_sha256,
            baseline_authority.graph_identity_sha256
        );

        let parameters = mutate(
            baseline.plan().clone(),
            GpuBabBoundF32TensorRole::Parameters,
            0.5,
        );
        assert_ne!(
            parameters.authority.graph_identity_sha256,
            baseline_authority.graph_identity_sha256
        );
        let mut parameter_shape_plan = baseline.plan().clone();
        parameter_shape_plan
            .f32_tensors
            .iter_mut()
            .find(|tensor| tensor.role == GpuBabBoundF32TensorRole::Parameters)
            .unwrap()
            .shape = vec![2, 2];
        let parameter_shape = GpuBabBoundPhaseDescriptor::new(
            parameter_shape_plan,
            Instant::now() + Duration::from_mins(1),
            8_192,
        )
        .unwrap();
        assert_ne!(
            parameter_shape.authority.graph_identity_sha256,
            baseline_authority.graph_identity_sha256
        );
        let errors = mutate(
            baseline.plan().clone(),
            GpuBabBoundF32TensorRole::CertifiedErrors,
            0.125,
        );
        assert_ne!(
            errors.authority.graph_identity_sha256,
            baseline_authority.graph_identity_sha256
        );
        let input = mutate(
            baseline.plan().clone(),
            GpuBabBoundF32TensorRole::InputLower,
            -0.75,
        );
        assert_ne!(
            input.authority.input_identity_sha256,
            baseline_authority.input_identity_sha256
        );
        let root = mutate(
            baseline.plan().clone(),
            GpuBabBoundF32TensorRole::RootLower,
            -1.75,
        );
        assert_ne!(
            root.authority.root_bounds_identity_sha256,
            baseline_authority.root_bounds_identity_sha256
        );
        let relaxation = mutate(
            baseline.plan().clone(),
            GpuBabBoundF32TensorRole::Relaxations,
            0.5,
        );
        assert_ne!(
            relaxation.authority.relaxation_identity_sha256,
            baseline_authority.relaxation_identity_sha256
        );
        let objective = mutate(
            baseline.plan().clone(),
            GpuBabBoundF32TensorRole::ObjectiveCoefficients,
            0.5,
        );
        assert_ne!(
            objective.authority.objective_set_identity_sha256,
            baseline_authority.objective_set_identity_sha256
        );
        let mut objective_count_plan = baseline.plan().clone();
        let indices = objective_count_plan
            .u32_tensors
            .iter_mut()
            .find(|tensor| tensor.role == GpuBabBoundU32TensorRole::ObjectiveIndices)
            .unwrap();
        indices.shape = vec![7];
        indices.values = GpuBabBoundOwnedSlice::new((0..7_u32).collect());
        let coefficients = objective_count_plan
            .f32_tensors
            .iter_mut()
            .find(|tensor| tensor.role == GpuBabBoundF32TensorRole::ObjectiveCoefficients)
            .unwrap();
        coefficients.shape = vec![7, 2];
        coefficients.values = copy_owned_slice(&coefficients.values[..14]);
        let objective_count = GpuBabBoundPhaseDescriptor::new(
            objective_count_plan,
            Instant::now() + Duration::from_mins(1),
            8_192,
        )
        .unwrap();
        assert_ne!(
            objective_count.authority.objective_set_identity_sha256,
            baseline_authority.objective_set_identity_sha256
        );
        let mut metadata_plan = baseline.plan().clone();
        let metadata = metadata_plan
            .u32_tensors
            .iter_mut()
            .find(|tensor| tensor.role == GpuBabBoundU32TensorRole::TopologyMetadata)
            .unwrap();
        metadata.values = mutate_owned_slice(&metadata.values, |values| values[0] += 1);
        let metadata = GpuBabBoundPhaseDescriptor::new(
            metadata_plan,
            Instant::now() + Duration::from_mins(1),
            8_192,
        )
        .unwrap();
        assert_ne!(
            metadata.authority.graph_identity_sha256,
            baseline_authority.graph_identity_sha256
        );

        let mut dispatch_plan = baseline.plan().clone();
        dispatch_plan.dispatches_per_subchunk += 1;
        let dispatch = GpuBabBoundPhaseDescriptor::new(
            dispatch_plan,
            Instant::now() + Duration::from_mins(1),
            8_192,
        )
        .unwrap();
        assert_ne!(
            dispatch.authority.graph_identity_sha256,
            baseline_authority.graph_identity_sha256
        );

        let mut reordered = baseline.plan().clone();
        reordered.f32_tensors.swap(0, 1);
        assert!(GpuBabBoundPhaseDescriptor::new(
            reordered,
            Instant::now() + Duration::from_mins(1),
            8_192,
        )
        .is_err());
        let mut unbounded = baseline.plan().clone();
        unbounded.dispatches_per_subchunk = usize::MAX;
        assert!(GpuBabBoundPhaseDescriptor::new(
            unbounded,
            Instant::now() + Duration::from_mins(1),
            8_192,
        )
        .is_err());
    }

    #[test]
    fn completed_validator_rejects_full_association_and_numerical_corruption_matrix() {
        for corruption in [
            Corruption::RowPermutation,
            Corruption::ParentSidecar,
            Corruption::ChildOrdinal,
            Corruption::DomainEcho,
            Corruption::ObjectiveEcho,
            Corruption::QSidecar,
            Corruption::PartialRows,
            Corruption::Nonfinite,
            Corruption::Inverted,
            Corruption::Disjoint,
            Corruption::Status,
            Corruption::Taint,
            Corruption::BackendEcho,
            Corruption::ScheduleEcho,
            Corruption::EndpointEcho,
            Corruption::DeadlineEcho,
            Corruption::PartialCount,
            Corruption::TighteningCount,
            Corruption::PrunedCount,
            Corruption::OutcomePermutation,
            Corruption::OutcomeAssociation,
        ] {
            assert_contract_failure(Mode::Corrupt(corruption));
        }
    }

    #[test]
    fn completed_validator_rejects_full_resource_and_transfer_corruption_matrix() {
        for corruption in [
            Corruption::OverCap,
            Corruption::MemoryOverflow,
            Corruption::Readbacks,
            Corruption::Synchronizations,
            Corruption::CoefficientD2h,
            Corruption::H2dEquation,
            Corruption::D2hEquation,
            Corruption::OperandEquation,
            Corruption::ResultSidecarEquation,
            Corruption::DomainOutcomeSidecarEquation,
            Corruption::DispatchesMax,
            Corruption::SubmitsMax,
        ] {
            assert_contract_failure(Mode::Corrupt(corruption));
        }
    }

    #[test]
    fn invalid_request_never_reaches_backend_acceptance() {
        let backend = FakeBackend::new(vec![Mode::CompleteZero]);
        let mut lease = open_lease(&backend, phase());
        let mut invalid = request();
        invalid.objective_indices = vec![3, 1];
        let preparation = lease.prepare_wave(invalid);
        assert!(!preparation.permits_legacy_fallback());
        assert!(matches!(
            &preparation,
            GpuBabBoundWavePreparation::InvalidRequest(_)
        ));
        drop(preparation);
        assert_eq!(lease.last_wave_index, 0);
        assert_eq!(lease.state, LeaseState::Open);

        let mut expired = request();
        expired.deadline = Instant::now();
        assert!(matches!(
            lease.prepare_wave(expired),
            GpuBabBoundWavePreparation::SessionTerminal(
                GpuBabBoundSessionTerminal::BackendPrepareDeadlineExpired
            )
        ));
        assert_eq!(lease.last_wave_index, 0);
        assert_eq!(lease.state, LeaseState::Poisoned);
    }

    #[test]
    fn failure_receipt_requires_full_static_request_and_exact_terminal_echo() {
        let backend = FakeBackend::new(vec![Mode::AcceptedFailure]);
        let mut lease = open_lease(&backend, phase());
        let disposition = execute(&mut lease, request());
        let failure = match disposition {
            GpuBabBoundWaveDisposition::AcceptedFailure(failure) => failure,
            _ => panic!("fake failure must remain failure"),
        };
        assert!(failure.receipt_validated());
        assert_eq!(failure.receipt().completed_parent_groups, 0);
        assert_eq!(failure.receipt().completed_domains, 0);
        assert_eq!(failure.receipt().completed_rows, 0);
        assert_eq!(failure.receipt().transfers.host_to_device_bytes, 0);
        assert_eq!(failure.receipt().transfers.device_to_host_bytes, 0);
        assert_eq!(
            failure.receipt().transfers.coefficient_device_to_host_bytes,
            0
        );
    }

    #[test]
    fn open_receipt_binds_backend_generation_nonce_phase_and_retained_memory() {
        let backend = FakeBackend::new(vec![]);
        let expected_backend = *backend.registration.backend_issuer_sha256();
        let phase = phase();
        let expected_graph = phase.authority.graph_identity_sha256;
        let lease = open_lease(&backend, phase);
        assert_eq!(
            lease.transcript.backend.backend_issuer_sha256,
            expected_backend
        );
        assert_eq!(
            lease.transcript.backend.registration_epoch,
            backend.registration.registration_epoch()
        );
        assert_eq!(lease.transcript.backend.generation, 1);
        assert!(!is_zero_identity(
            lease.transcript.backend.session_nonce_sha256
        ));
        assert_eq!(lease.transcript.graph_identity_sha256, expected_graph);
        assert_eq!(
            lease.transcript.static_graph_payload_bytes,
            lease.phase.static_graph_payload_bytes()
        );
        assert_eq!(
            lease.transcript.static_phase_payload_bytes,
            lease.phase.static_phase_payload_bytes()
        );
        assert_eq!(lease.open_memory, open_memory());
        assert!(matches!(
            lease.close(),
            GpuBabBoundPhaseCloseDisposition::Closed(_)
        ));
    }

    #[test]
    fn static_schedule_request_recomputes_identity_and_enforces_canonical_payload() {
        let payload = static_schedule_payload();
        let deadline = Instant::now() + Duration::from_secs(10);
        let request = payload.request(deadline, 4_096);
        assert_eq!(request.topology_schema_version(), 1);
        assert_eq!(request.static_payload_identity_sha256(), &payload.identity);
        assert_eq!(request.logical_static_device_bytes(), 84);
        assert_eq!(request.f32_tensors()[2].shape, [0]);
        assert_eq!(request.u32_tensors()[1].shape, [0]);

        assert!(GpuBabBoundStaticScheduleRequest::new(
            payload.topology_schema_version,
            &payload.topology_bytes,
            &payload.f32_tensors,
            &payload.u32_tensors,
            [0; 32],
            deadline,
            4_096,
        )
        .is_err());

        let mut mismatched_identity = payload.identity;
        mismatched_identity[0] ^= 1;
        assert!(GpuBabBoundStaticScheduleRequest::new(
            payload.topology_schema_version,
            &payload.topology_bytes,
            &payload.f32_tensors,
            &payload.u32_tensors,
            mismatched_identity,
            deadline,
            4_096,
        )
        .is_err());

        let mut missing_tensor = payload.f32_tensors.clone();
        missing_tensor.pop();
        assert!(GpuBabBoundStaticScheduleRequest::new(
            payload.topology_schema_version,
            &payload.topology_bytes,
            &missing_tensor,
            &payload.u32_tensors,
            payload.identity,
            deadline,
            4_096,
        )
        .is_err());

        let mut noncanonical_empty = payload.f32_tensors.clone();
        noncanonical_empty[2].shape = vec![1];
        assert!(GpuBabBoundStaticScheduleRequest::new(
            payload.topology_schema_version,
            &payload.topology_bytes,
            &noncanonical_empty,
            &payload.u32_tensors,
            payload.identity,
            deadline,
            4_096,
        )
        .is_err());

        let mut illegal_zero = payload.f32_tensors.clone();
        illegal_zero[0].shape = vec![0];
        illegal_zero[0].values = GpuBabBoundOwnedSlice::new(Vec::new());
        assert!(GpuBabBoundStaticScheduleRequest::new(
            payload.topology_schema_version,
            &payload.topology_bytes,
            &illegal_zero,
            &payload.u32_tensors,
            payload.identity,
            deadline,
            4_096,
        )
        .is_err());

        let mut changed_value = payload.f32_tensors.clone();
        let mut values = changed_value[0].values.as_slice().to_vec();
        values[0] = f32::from_bits(values[0].to_bits() ^ 1);
        changed_value[0].values = GpuBabBoundOwnedSlice::new(values);
        assert!(GpuBabBoundStaticScheduleRequest::new(
            payload.topology_schema_version,
            &payload.topology_bytes,
            &changed_value,
            &payload.u32_tensors,
            payload.identity,
            deadline,
            4_096,
        )
        .is_err());

        let mut changed_shape = payload.f32_tensors.clone();
        changed_shape[0].shape = vec![1, 4];
        assert!(GpuBabBoundStaticScheduleRequest::new(
            payload.topology_schema_version,
            &payload.topology_bytes,
            &changed_shape,
            &payload.u32_tensors,
            payload.identity,
            deadline,
            4_096,
        )
        .is_err());
        assert!(GpuBabBoundStaticScheduleRequest::new(
            payload.topology_schema_version + 1,
            &payload.topology_bytes,
            &payload.f32_tensors,
            &payload.u32_tensors,
            payload.identity,
            deadline,
            4_096,
        )
        .is_err());
    }

    #[test]
    fn static_schedule_only_clean_decline_permits_fallback_and_low_cap_is_clean() {
        let payload = static_schedule_payload();
        let request = payload.request(Instant::now() + Duration::from_secs(10), 4_096);

        let default_backend = FakeBackend::new(Vec::new());
        let default = certify_gpu_bab_bound_static_schedule(&default_backend, &request);
        assert!(default.permits_legacy_fallback());
        assert!(matches!(
            default,
            GpuBabBoundScheduleCertification::CleanDecline(GpuBabBoundPhaseDecline::Unsupported)
        ));

        let explicit_decline = ScheduleBackend::new(ScheduleMode::Decline);
        let declined = certify_gpu_bab_bound_static_schedule(&explicit_decline, &request);
        assert!(declined.permits_legacy_fallback());

        let low_cap_request = payload.request(Instant::now() + Duration::from_secs(10), 1);
        let untouched = ScheduleBackend::new(ScheduleMode::Exact);
        let low_cap = certify_gpu_bab_bound_static_schedule(&untouched, &low_cap_request);
        assert!(low_cap.permits_legacy_fallback());
        assert!(matches!(
            low_cap,
            GpuBabBoundScheduleCertification::CleanDecline(
                GpuBabBoundPhaseDecline::InsufficientCapacity
            )
        ));
        assert_eq!(untouched.schedule_calls(), 0);
    }

    #[test]
    fn exact_static_schedule_certificate_owns_all_fixed_bindings() {
        let payload = static_schedule_payload();
        let deadline = Instant::now() + Duration::from_secs(10);
        let request = payload.request(deadline, 4_096);
        let backend = ScheduleBackend::new(ScheduleMode::Exact);
        let outcome = certify_gpu_bab_bound_static_schedule(&backend, &request);
        assert!(!outcome.permits_legacy_fallback());
        let certificate = match outcome {
            GpuBabBoundScheduleCertification::Certified(certificate) => certificate,
            _ => panic!("exact schedule must certify"),
        };
        let evidence = certificate.evidence();
        assert_eq!(
            evidence.backend_issuer_sha256,
            *backend.registration.backend_issuer_sha256()
        );
        assert_eq!(
            evidence.registration_epoch,
            backend.registration.registration_epoch()
        );
        assert_eq!(evidence.static_payload_identity_sha256, payload.identity);
        assert_eq!(evidence.schedule_identity, schedule_identity());
        assert_eq!(evidence.requested_max_device_bytes, 4_096);
        assert_eq!(evidence.dispatches_per_subchunk, 3);
        assert_eq!(certificate.deadline(), deadline);
        assert!(!is_zero_identity(
            *certificate.certificate_identity_sha256()
        ));
        drop(backend.registration.available_guard().unwrap());
    }

    #[test]
    fn every_static_schedule_registration_schema_cap_policy_and_count_mismatch_poison() {
        let payload = static_schedule_payload();
        let mutations = [
            ScheduleEvidenceMutation::BackendIssuer,
            ScheduleEvidenceMutation::RegistrationEpoch,
            ScheduleEvidenceMutation::StaticIdentityZero,
            ScheduleEvidenceMutation::StaticIdentityMismatch,
            ScheduleEvidenceMutation::TopologySchemaVersion,
            ScheduleEvidenceMutation::SchemaBundleVersion,
            ScheduleEvidenceMutation::ProviderAbi,
            ScheduleEvidenceMutation::ReceiptAbi,
            ScheduleEvidenceMutation::Kernel,
            ScheduleEvidenceMutation::TopologySchema,
            ScheduleEvidenceMutation::SelfcheckSchema,
            ScheduleEvidenceMutation::TranscriptSchema,
            ScheduleEvidenceMutation::RequestedDeviceCap,
            ScheduleEvidenceMutation::InvalidPolicy,
            ScheduleEvidenceMutation::PolicyBelowCap,
            ScheduleEvidenceMutation::DispatchZero,
            ScheduleEvidenceMutation::DispatchAboveCore,
            ScheduleEvidenceMutation::DispatchAbovePolicy,
        ];
        for mutation in mutations {
            let request = payload.request(Instant::now() + Duration::from_secs(10), 4_096);
            let backend = ScheduleBackend::new(ScheduleMode::Mutate(mutation));
            let outcome = certify_gpu_bab_bound_static_schedule(&backend, &request);
            assert!(!outcome.permits_legacy_fallback(), "{mutation:?}");
            let failure = match outcome {
                GpuBabBoundScheduleCertification::ProviderFailure(failure) => failure,
                _ => panic!("{mutation:?} must be a provider failure"),
            };
            assert_eq!(
                failure.kind(),
                GpuBabBoundProviderFailureKind::InvalidScheduleEvidence,
                "{mutation:?}"
            );
            assert!(
                backend.registration.available_guard().is_err(),
                "{mutation:?} must poison registration"
            );
        }
    }

    #[test]
    fn schedule_panic_payload_and_registration_drift_are_quarantined_and_poisoned() {
        let payload = static_schedule_payload();
        let request = payload.request(Instant::now() + Duration::from_secs(10), 4_096);

        let panicking = ScheduleBackend::new(ScheduleMode::Panic);
        let panic_outcome = certify_gpu_bab_bound_static_schedule(&panicking, &request);
        let panic_failure = match panic_outcome {
            GpuBabBoundScheduleCertification::ProviderFailure(failure) => failure,
            _ => panic!("schedule panic must be terminal"),
        };
        assert_eq!(
            panic_failure.kind(),
            GpuBabBoundProviderFailureKind::SchedulePanicked
        );
        assert!(panicking.registration.available_guard().is_err());

        let drifting = ScheduleBackend::with_registration_swap();
        let drift_outcome = certify_gpu_bab_bound_static_schedule(&drifting, &request);
        let drift_failure = match drift_outcome {
            GpuBabBoundScheduleCertification::ProviderFailure(failure) => failure,
            _ => panic!("registration drift must be terminal"),
        };
        assert_eq!(
            drift_failure.kind(),
            GpuBabBoundProviderFailureKind::RegistrationChanged
        );
        assert!(drifting.registration.available_guard().is_err());
        assert!(drifting
            .alternate_registration
            .as_ref()
            .unwrap()
            .available_guard()
            .is_err());
    }

    #[test]
    fn schedule_deadlines_and_occupied_registration_are_terminal_but_not_poisoning() {
        let payload = static_schedule_payload();
        let backend = ScheduleBackend::new(ScheduleMode::Exact);

        let deadline = Instant::now() + Duration::from_millis(20);
        let expired_request = payload.request(deadline, 4_096);
        while Instant::now() < deadline {
            std::hint::spin_loop();
        }
        let expired = certify_gpu_bab_bound_static_schedule(&backend, &expired_request);
        let failure = match expired {
            GpuBabBoundScheduleCertification::ProviderFailure(failure) => failure,
            _ => panic!("expired request must be terminal"),
        };
        assert_eq!(
            failure.kind(),
            GpuBabBoundProviderFailureKind::DeadlineExpired
        );
        assert_eq!(backend.schedule_calls(), 0);
        drop(backend.registration.available_guard().unwrap());

        backend.set_mode(ScheduleMode::WaitForDeadline);
        let callback_deadline = Instant::now() + Duration::from_millis(20);
        let callback_request = payload.request(callback_deadline, 4_096);
        let callback_expired = certify_gpu_bab_bound_static_schedule(&backend, &callback_request);
        let failure = match callback_expired {
            GpuBabBoundScheduleCertification::ProviderFailure(failure) => failure,
            _ => panic!("callback deadline must be terminal"),
        };
        assert_eq!(
            failure.kind(),
            GpuBabBoundProviderFailureKind::DeadlineExpired
        );
        drop(backend.registration.available_guard().unwrap());

        backend.set_mode(ScheduleMode::Exact);
        let phase = phase();
        let (issuer, claimed) = backend.registration.claim(&phase);
        claimed.unwrap();
        let occupied_request = payload.request(Instant::now() + Duration::from_secs(10), 4_096);
        let occupied = certify_gpu_bab_bound_static_schedule(&backend, &occupied_request);
        let failure = match occupied {
            GpuBabBoundScheduleCertification::ProviderFailure(failure) => failure,
            _ => panic!("occupied registration must be terminal"),
        };
        assert_eq!(
            failure.kind(),
            GpuBabBoundProviderFailureKind::RegistrationUnavailable
        );
        assert!(backend.registration.release_noalloc(issuer));

        let recovered_request = payload.request(Instant::now() + Duration::from_secs(10), 4_096);
        assert!(matches!(
            certify_gpu_bab_bound_static_schedule(&backend, &recovered_request),
            GpuBabBoundScheduleCertification::Certified(_)
        ));
    }

    #[test]
    fn schedule_callback_can_reenter_registration_and_concurrent_certification_stays_finite() {
        let payload = static_schedule_payload();
        let request = payload.request(Instant::now() + Duration::from_secs(10), 4_096);
        let reentrant = ScheduleBackend::new(ScheduleMode::ReenterRegistration);
        assert!(matches!(
            certify_gpu_bab_bound_static_schedule(&reentrant, &request),
            GpuBabBoundScheduleCertification::Certified(_)
        ));

        let concurrent = ScheduleBackend::new(ScheduleMode::Exact);
        let barrier = Barrier::new(5);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..4 {
                handles.push(scope.spawn(|| {
                    barrier.wait();
                    certify_gpu_bab_bound_static_schedule(&concurrent, &request)
                }));
            }
            barrier.wait();
            for handle in handles {
                assert!(matches!(
                    handle.join().unwrap(),
                    GpuBabBoundScheduleCertification::Certified(_)
                ));
            }
        });
        assert_eq!(concurrent.schedule_calls(), 4);
        drop(concurrent.registration.available_guard().unwrap());
    }
}
