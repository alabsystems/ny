// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Explicit, default-closed qualification seam for the WGPU retained-BaB provider.
//!
//! This module still contains no bound kernel and no raw session. The only
//! production route that may ever install a provider is the separately typed
//! [`WgpuBabBoundVerdictRequest`] constructor. Its static source,
//! implementation, and reviewed-kernel-schema gates are all independently
//! closed in this slice, so it refuses before creating a WGPU device, reading
//! an environment-backed probe cache, or allocating a core registration epoch.
//!
//! A future reviewed opening must first qualify the exact uncharged verdict
//! device, pass every BaB admission conjunct, and then atomically install one
//! [`WgpuBabBoundInstalled`] value. That value owns the one stable core
//! registration, exact-epoch qualification, cloned exact device/queue handles,
//! and immutable adapter/configuration/loading/schema evidence. Public verdict
//! reports are diagnostic outputs only and are never accepted as evidence.
//!
//! Cloning WGPU handles establishes identity and a safe future session
//! lifetime; it does not grant custody of unrelated allocations on the shared
//! device. Future receipts may account only for dedicated provider-owned BaB
//! resources. This foundation has no such resources and cannot return a phase
//! policy, `Prepared`, a session, a receipt, or a completion claim.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use ny_core::{
    GpuBabBoundBackendOpenPreparation, GpuBabBoundBackendRegistration, GpuBabBoundNumericalTcb,
    GpuBabBoundPhaseDecline, GpuBabBoundPhasePolicy, GpuBabBoundTcbInvocation, NyError,
};
use sha2::{Digest, Sha256};

use super::super::shader_loading::DenormPreservePolicy;
use super::super::WgpuDevice;
use super::sound_authority::{
    WgpuVerdictQualificationError, WgpuVerdictReport, WgpuVerdictRequest, WgpuVerdictRung,
    WgpuVerdictRungOutcome,
};

/// Reviewed source gate for admitting WGPU into the retained-BaB numerical TCB.
/// Environment state cannot open it.
pub(super) const PRODUCTION_WGPU_BAB_BOUND_AUTHORITY_ENABLED: bool = false;

/// Independent implementation gate. A source-gate flip alone cannot admit an
/// unfinished self-check or kernel.
const WGPU_BAB_BOUND_KERNEL_SELFCHECK_IMPLEMENTED: bool = false;

const WGPU_BAB_BOUND_CONFIG_HASH_DOMAIN: &[u8] = b"ny/wgpu/bab-bound/provider-config/v1\0";
const WGPU_BAB_BOUND_CONFIG_SCHEMA_VERSION: u32 = 1;

/// Complete reviewed schema bundle needed before provider installation. No
/// digest is invented for the unfinished implementation: production returns
/// `None` until each artifact is reviewed together.
#[derive(Clone, Debug, PartialEq, Eq)]
struct WgpuBabBoundReviewedSchemaBundle {
    bundle_version: u32,
    provider_abi_sha256: [u8; 32],
    receipt_abi_sha256: [u8; 32],
    kernel_sha256: [u8; 32],
    topology_schema_sha256: [u8; 32],
    selfcheck_schema_sha256: [u8; 32],
    transcript_schema_sha256: [u8; 32],
}

impl WgpuBabBoundReviewedSchemaBundle {
    fn complete(&self) -> bool {
        self.bundle_version != 0
            && [
                self.provider_abi_sha256,
                self.receipt_abi_sha256,
                self.kernel_sha256,
                self.topology_schema_sha256,
                self.selfcheck_schema_sha256,
                self.transcript_schema_sha256,
            ]
            .into_iter()
            .all(|identity| identity != [0; 32])
    }
}

const fn reviewed_schema_bundle() -> Option<WgpuBabBoundReviewedSchemaBundle> {
    None
}

static TEST_FORCE_SELFCHECK_FAIL: AtomicBool = AtomicBool::new(false);

fn env_forces_selfcheck_failure() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| {
        ny_levers::read_presence(&ny_levers::decls::diagnostics::FORCE_GPU_BAB_BOUND_SELFCHECK_FAIL)
    })
}

fn selfcheck_forced_to_fail() -> bool {
    TEST_FORCE_SELFCHECK_FAIL.load(Ordering::Relaxed) || env_forces_selfcheck_failure()
}

#[cfg(test)]
fn set_force_selfcheck_fail(force: bool) {
    TEST_FORCE_SELFCHECK_FAIL.store(force, Ordering::Relaxed);
}

/// Explicit request for a WGPU device qualified for retained-BaB bounds.
///
/// The private field prevents struct-literal construction and the absence of
/// [`Default`] keeps this stronger authority request visible at every call
/// site. The value is not evidence and carries no authority by itself.
///
/// ```compile_fail
/// let _ = ny_gpu::WgpuBabBoundVerdictRequest::default();
/// ```
///
/// ```compile_fail
/// let _ = ny_gpu::WgpuBabBoundVerdictRequest { _explicit: () };
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct WgpuBabBoundVerdictRequest {
    _explicit: (),
}

#[allow(clippy::new_without_default)]
impl WgpuBabBoundVerdictRequest {
    /// Make one explicit retained-BaB qualification request.
    #[must_use]
    pub const fn new() -> Self {
        Self { _explicit: () }
    }
}

#[derive(Debug)]
enum WgpuBabBoundQualificationErrorSource {
    Verdict(WgpuVerdictQualificationError),
    Provider(NyError),
}

/// Typed refusal from [`WgpuDevice::new_for_verdict_bab_bound`].
#[derive(Debug)]
pub struct WgpuBabBoundQualificationError {
    verdict_report: Option<WgpuVerdictReport>,
    source: WgpuBabBoundQualificationErrorSource,
}

impl WgpuBabBoundQualificationError {
    fn refused(refusal: WgpuBabBoundAdmissionRefusal) -> Self {
        Self {
            verdict_report: None,
            source: WgpuBabBoundQualificationErrorSource::Provider(
                NyError::UnsupportedConfiguration(refusal.message().to_string()),
            ),
        }
    }

    fn verdict(error: WgpuVerdictQualificationError) -> Self {
        Self {
            verdict_report: Some(error.report().clone()),
            source: WgpuBabBoundQualificationErrorSource::Verdict(error),
        }
    }

    fn provider(report: Option<WgpuVerdictReport>, source: NyError) -> Self {
        Self {
            verdict_report: report,
            source: WgpuBabBoundQualificationErrorSource::Provider(source),
        }
    }

    /// Base verdict report measured on the attempted exact device, if device
    /// construction reached that stage. Static default-dark refusal is `None`.
    #[must_use]
    pub const fn verdict_report(&self) -> Option<&WgpuVerdictReport> {
        self.verdict_report.as_ref()
    }
}

impl std::fmt::Display for WgpuBabBoundQualificationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.source {
            WgpuBabBoundQualificationErrorSource::Verdict(error) => {
                write!(
                    formatter,
                    "WGPU retained-BaB qualification refused: {error}"
                )
            }
            WgpuBabBoundQualificationErrorSource::Provider(error) => {
                write!(
                    formatter,
                    "WGPU retained-BaB qualification refused: {error}"
                )
            }
        }
    }
}

impl std::error::Error for WgpuBabBoundQualificationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.source {
            WgpuBabBoundQualificationErrorSource::Verdict(error) => Some(error),
            WgpuBabBoundQualificationErrorSource::Provider(error) => Some(error),
        }
    }
}

/// Immutable identity/configuration evidence captured from the exact device
/// returned by the explicit base-verdict constructor.
#[derive(Clone, Debug, PartialEq, Eq)]
struct WgpuBabBoundConfigurationEvidence {
    adapter: wgpu::AdapterInfo,
    enabled_features: wgpu::Features,
    limits: wgpu::Limits,
    denorm_preserve_policy: DenormPreservePolicy,
    denorm_preserve_enabled: bool,
    loading_contract_intact_at_qualification: bool,
    base_source_gate_open: bool,
    base_rung_outcomes: [WgpuVerdictRungOutcome; 5],
    schemas: WgpuBabBoundReviewedSchemaBundle,
    issuer_sha256: [u8; 32],
}

impl WgpuBabBoundConfigurationEvidence {
    fn new(
        adapter: wgpu::AdapterInfo,
        enabled_features: wgpu::Features,
        limits: wgpu::Limits,
        denorm_preserve_policy: DenormPreservePolicy,
        denorm_preserve_enabled: bool,
        loading_contract_intact_at_qualification: bool,
        base_source_gate_open: bool,
        base_rung_outcomes: [WgpuVerdictRungOutcome; 5],
        schemas: WgpuBabBoundReviewedSchemaBundle,
    ) -> Self {
        let mut evidence = Self {
            adapter,
            enabled_features,
            limits,
            denorm_preserve_policy,
            denorm_preserve_enabled,
            loading_contract_intact_at_qualification,
            base_source_gate_open,
            base_rung_outcomes,
            schemas,
            issuer_sha256: [0; 32],
        };
        evidence.issuer_sha256 = configuration_sha256(&evidence);
        evidence
    }

    fn capture(
        device: &WgpuDevice,
        report: &WgpuVerdictReport,
        schemas: WgpuBabBoundReviewedSchemaBundle,
    ) -> Self {
        Self::new(
            device.adapter_info.clone(),
            device.device.features(),
            device.device.limits(),
            device.denorm_preserve_policy,
            device.denorm_preserve_enabled,
            device.denorm_preserve_contract_intact(),
            report.source_gate_open(),
            verdict_rung_outcomes(report),
            schemas,
        )
    }

    fn matches_device(&self, device: &WgpuDevice) -> bool {
        let report_matches = device.verdict_report.as_ref().is_some_and(|report| {
            self.base_source_gate_open == report.source_gate_open()
                && self.base_rung_outcomes == verdict_rung_outcomes(report)
        });
        self.adapter == device.adapter_info
            && self.enabled_features == device.device.features()
            && self.limits == device.device.limits()
            && self.denorm_preserve_policy == device.denorm_preserve_policy
            && self.denorm_preserve_enabled == device.denorm_preserve_enabled
            && self.loading_contract_intact_at_qualification
                == device.denorm_preserve_contract_intact()
            && report_matches
            && self.issuer_sha256 == configuration_sha256(self)
    }
}

fn verdict_rung_outcomes(report: &WgpuVerdictReport) -> [WgpuVerdictRungOutcome; 5] {
    [
        report.outcome(WgpuVerdictRung::IeeeF32Model),
        report.outcome(WgpuVerdictRung::EftPrimitives),
        report.outcome(WgpuVerdictRung::GradualUnderflow),
        report.outcome(WgpuVerdictRung::HostEftReference),
        report.outcome(WgpuVerdictRung::SentinelTaintSticky),
    ]
}

fn hash_bytes(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

const fn backend_tag(backend: wgpu::Backend) -> u8 {
    match backend {
        wgpu::Backend::Noop => 0,
        wgpu::Backend::Vulkan => 1,
        wgpu::Backend::Metal => 2,
        wgpu::Backend::Dx12 => 3,
        wgpu::Backend::Gl => 4,
        wgpu::Backend::BrowserWebGpu => 5,
    }
}

const fn device_type_tag(device_type: wgpu::DeviceType) -> u8 {
    match device_type {
        wgpu::DeviceType::Other => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::DiscreteGpu => 2,
        wgpu::DeviceType::VirtualGpu => 3,
        wgpu::DeviceType::Cpu => 4,
    }
}

const fn denorm_policy_tag(policy: DenormPreservePolicy) -> u8 {
    match policy {
        DenormPreservePolicy::Auto => 0,
        DenormPreservePolicy::Disabled => 1,
        DenormPreservePolicy::Required => 2,
        DenormPreservePolicy::ForcedDisabled => 3,
    }
}

const fn verdict_outcome_tag(outcome: WgpuVerdictRungOutcome) -> u8 {
    match outcome {
        WgpuVerdictRungOutcome::NotRun => 0,
        WgpuVerdictRungOutcome::Passed => 1,
        WgpuVerdictRungOutcome::Failed => 2,
    }
}

macro_rules! hash_all_wgpu_limits {
    ($hash:expr, $limits:expr, [$($field:ident),+ $(,)?]) => {{
        // Deliberately exhaustive: a wgpu Limits field addition must fail this
        // build until the config schema and canonical encoding are reviewed.
        let &wgpu::Limits { $($field: _,)+ } = $limits;
        $($hash.update($limits.$field.to_le_bytes());)+
    }};
}

fn configuration_sha256(evidence: &WgpuBabBoundConfigurationEvidence) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(WGPU_BAB_BOUND_CONFIG_HASH_DOMAIN);
    hash.update(WGPU_BAB_BOUND_CONFIG_SCHEMA_VERSION.to_le_bytes());

    // AdapterInfo is encoded field-by-field; enum tags are explicit and the
    // string fields are length-delimited. No Debug/display serialization is an
    // authority input.
    let wgpu::AdapterInfo {
        name,
        vendor,
        device,
        device_type,
        device_pci_bus_id,
        driver,
        driver_info,
        backend,
        subgroup_min_size,
        subgroup_max_size,
        transient_saves_memory,
    } = &evidence.adapter;
    hash_bytes(&mut hash, name.as_bytes());
    hash.update(vendor.to_le_bytes());
    hash.update(device.to_le_bytes());
    hash.update([device_type_tag(*device_type)]);
    hash_bytes(&mut hash, device_pci_bus_id.as_bytes());
    hash_bytes(&mut hash, driver.as_bytes());
    hash_bytes(&mut hash, driver_info.as_bytes());
    hash.update([backend_tag(*backend)]);
    hash.update(subgroup_min_size.to_le_bytes());
    hash.update(subgroup_max_size.to_le_bytes());
    hash.update([u8::from(*transient_saves_memory)]);

    for feature_word in evidence.enabled_features.bits().0 {
        hash.update(feature_word.to_le_bytes());
    }

    hash_all_wgpu_limits!(
        hash,
        &evidence.limits,
        [
            max_texture_dimension_1d,
            max_texture_dimension_2d,
            max_texture_dimension_3d,
            max_texture_array_layers,
            max_bind_groups,
            max_bindings_per_bind_group,
            max_dynamic_uniform_buffers_per_pipeline_layout,
            max_dynamic_storage_buffers_per_pipeline_layout,
            max_sampled_textures_per_shader_stage,
            max_samplers_per_shader_stage,
            max_storage_buffers_per_shader_stage,
            max_storage_textures_per_shader_stage,
            max_uniform_buffers_per_shader_stage,
            max_binding_array_elements_per_shader_stage,
            max_binding_array_acceleration_structure_elements_per_shader_stage,
            max_binding_array_sampler_elements_per_shader_stage,
            max_uniform_buffer_binding_size,
            max_storage_buffer_binding_size,
            max_vertex_buffers,
            max_buffer_size,
            max_vertex_attributes,
            max_vertex_buffer_array_stride,
            max_inter_stage_shader_variables,
            min_uniform_buffer_offset_alignment,
            min_storage_buffer_offset_alignment,
            max_color_attachments,
            max_color_attachment_bytes_per_sample,
            max_compute_workgroup_storage_size,
            max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x,
            max_compute_workgroup_size_y,
            max_compute_workgroup_size_z,
            max_compute_workgroups_per_dimension,
            max_immediate_size,
            max_non_sampler_bindings,
            max_task_mesh_workgroup_total_count,
            max_task_mesh_workgroups_per_dimension,
            max_task_invocations_per_workgroup,
            max_task_invocations_per_dimension,
            max_mesh_invocations_per_workgroup,
            max_mesh_invocations_per_dimension,
            max_task_payload_size,
            max_mesh_output_vertices,
            max_mesh_output_primitives,
            max_mesh_output_layers,
            max_mesh_multiview_view_count,
            max_blas_primitive_count,
            max_blas_geometry_count,
            max_tlas_instance_count,
            max_acceleration_structures_per_shader_stage,
            max_multiview_view_count,
        ]
    );

    hash.update([denorm_policy_tag(evidence.denorm_preserve_policy)]);
    hash.update([u8::from(evidence.denorm_preserve_enabled)]);
    hash.update([u8::from(evidence.loading_contract_intact_at_qualification)]);
    hash.update([u8::from(evidence.base_source_gate_open)]);
    for outcome in evidence.base_rung_outcomes {
        hash.update([verdict_outcome_tag(outcome)]);
    }
    hash.update(evidence.schemas.bundle_version.to_le_bytes());
    hash.update(evidence.schemas.provider_abi_sha256);
    hash.update(evidence.schemas.receipt_abi_sha256);
    hash.update(evidence.schemas.kernel_sha256);
    hash.update(evidence.schemas.topology_schema_sha256);
    hash.update(evidence.schemas.selfcheck_schema_sha256);
    hash.update(evidence.schemas.transcript_schema_sha256);
    hash.finalize().into()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WgpuBabBoundAdmissionRefusal {
    SourceGateClosed,
    KernelImplementationMissing,
    KernelSchemaMissing,
    SoundVerdictMissing,
    LoadingContractLost,
    BaseSelfcheckForcedFail,
    KernelSelfcheckFailed,
    BabSelfcheckForcedFail,
    ProviderContextFaulted,
    ConfigurationIdentityMismatch,
}

impl WgpuBabBoundAdmissionRefusal {
    const fn message(self) -> &'static str {
        match self {
            Self::SourceGateClosed => "the reviewed WGPU retained-BaB source gate is closed",
            Self::KernelImplementationMissing => {
                "the WGPU retained-BaB kernel/self-check implementation is absent"
            }
            Self::KernelSchemaMissing => {
                "the reviewed WGPU retained-BaB kernel schema identity is absent"
            }
            Self::SoundVerdictMissing => {
                "the exact device lacks live uncharged WGPU verdict authority"
            }
            Self::LoadingContractLost => {
                "the exact device's WGPU shader-loading contract is not intact"
            }
            Self::BaseSelfcheckForcedFail => "a base WGPU numerical self-check is forced to fail",
            Self::KernelSelfcheckFailed => "the WGPU retained-BaB kernel self-check did not pass",
            Self::BabSelfcheckForcedFail => "the WGPU retained-BaB self-check is forced to fail",
            Self::ProviderContextFaulted => {
                "the WGPU retained-BaB provider-owned resource context is faulted"
            }
            Self::ConfigurationIdentityMismatch => {
                "the WGPU retained-BaB exact-device configuration identity changed"
            }
        }
    }
}

fn static_admission_preflight(
    source_gate_open: bool,
    kernel_implemented: bool,
    schemas: Option<&WgpuBabBoundReviewedSchemaBundle>,
) -> Result<(), WgpuBabBoundAdmissionRefusal> {
    if !source_gate_open {
        return Err(WgpuBabBoundAdmissionRefusal::SourceGateClosed);
    }
    if !kernel_implemented {
        return Err(WgpuBabBoundAdmissionRefusal::KernelImplementationMissing);
    }
    if !schemas.is_some_and(WgpuBabBoundReviewedSchemaBundle::complete) {
        return Err(WgpuBabBoundAdmissionRefusal::KernelSchemaMissing);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct WgpuBabBoundLiveGates {
    source_gate_open: bool,
    kernel_implemented: bool,
    loading_contract_intact: bool,
    base_selfcheck_forced_fail: bool,
    kernel_selfcheck_passed: bool,
    bab_selfcheck_forced_fail: bool,
    provider_context_intact: bool,
}

struct WgpuBabBoundAdmissionInput<'a> {
    configuration: &'a WgpuBabBoundConfigurationEvidence,
    sound_report_qualified: bool,
    gates: WgpuBabBoundLiveGates,
}

#[derive(Debug)]
struct WgpuBabBoundAdmission {
    issuer_sha256: [u8; 32],
}

fn evaluate_admission(
    input: &WgpuBabBoundAdmissionInput<'_>,
) -> Result<WgpuBabBoundAdmission, WgpuBabBoundAdmissionRefusal> {
    static_admission_preflight(
        input.gates.source_gate_open,
        input.gates.kernel_implemented,
        Some(&input.configuration.schemas),
    )?;
    if !input.sound_report_qualified {
        return Err(WgpuBabBoundAdmissionRefusal::SoundVerdictMissing);
    }
    if !input.gates.loading_contract_intact {
        return Err(WgpuBabBoundAdmissionRefusal::LoadingContractLost);
    }
    if input.gates.base_selfcheck_forced_fail {
        return Err(WgpuBabBoundAdmissionRefusal::BaseSelfcheckForcedFail);
    }
    if !input.gates.kernel_selfcheck_passed {
        return Err(WgpuBabBoundAdmissionRefusal::KernelSelfcheckFailed);
    }
    if input.gates.bab_selfcheck_forced_fail {
        return Err(WgpuBabBoundAdmissionRefusal::BabSelfcheckForcedFail);
    }
    if !input.gates.provider_context_intact {
        return Err(WgpuBabBoundAdmissionRefusal::ProviderContextFaulted);
    }
    let issuer_sha256 = configuration_sha256(input.configuration);
    if issuer_sha256 == [0; 32] || issuer_sha256 != input.configuration.issuer_sha256 {
        return Err(WgpuBabBoundAdmissionRefusal::ConfigurationIdentityMismatch);
    }
    Ok(WgpuBabBoundAdmission { issuer_sha256 })
}

#[derive(Debug)]
enum WgpuBabBoundAdmissionAttemptError<E> {
    Refused(WgpuBabBoundAdmissionRefusal),
    Registration(E),
}

/// The injected issuer makes the epoch side effect mechanically testable
/// without ever constructing a real core registration in unit tests. `FnOnce`
/// also makes more than one issuance impossible in this attempt.
fn issue_registration_after_admission<R, E>(
    input: &WgpuBabBoundAdmissionInput<'_>,
    issue: impl FnOnce([u8; 32]) -> Result<R, E>,
) -> Result<(WgpuBabBoundAdmission, R), WgpuBabBoundAdmissionAttemptError<E>> {
    let admission =
        evaluate_admission(input).map_err(WgpuBabBoundAdmissionAttemptError::Refused)?;
    let registration =
        issue(admission.issuer_sha256).map_err(WgpuBabBoundAdmissionAttemptError::Registration)?;
    Ok((admission, registration))
}

/// Private evidence that one exact registration epoch passed every provider
/// admission conjunct. This is never reconstructed from a public diagnostic.
struct WgpuBabBoundQualification {
    registration_epoch: u64,
    issuer_sha256: [u8; 32],
    sound_report_qualified: bool,
    kernel_selfcheck_passed: bool,
}

/// Provider-owned exact-device context. The cloned handles bind future work to
/// this device but do not confer ownership of unrelated WGPU allocations.
/// Session execution remains unavailable until core has a reviewed typed
/// retained-domain/delta-transfer receipt; the current full-dynamic-H2D wave
/// contract is not claimed as retained BaB support.
struct WgpuBabBoundDeviceContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    configuration: WgpuBabBoundConfigurationEvidence,
    resources: WgpuBabBoundResourceState,
}

/// Empty provider-only synchronization/accounting namespace. Future sessions
/// may populate only this ledger with their own exact buffer sizes; pre-existing
/// WGPU allocations and shared CROWN caches are outside its custody.
struct WgpuBabBoundResourceState {
    submit_lock: Mutex<()>,
    allocations: Mutex<BTreeMap<u64, u64>>,
    next_buffer_namespace: AtomicU64,
    faulted: AtomicBool,
}

impl WgpuBabBoundResourceState {
    fn new() -> Self {
        Self {
            submit_lock: Mutex::new(()),
            allocations: Mutex::new(BTreeMap::new()),
            next_buffer_namespace: AtomicU64::new(1),
            faulted: AtomicBool::new(false),
        }
    }

    fn authority_intact(&self) -> bool {
        !self.submit_lock.is_poisoned()
            && !self.allocations.is_poisoned()
            && self.next_buffer_namespace.load(Ordering::Acquire) != 0
            && !self.faulted.load(Ordering::Acquire)
    }
}

impl WgpuBabBoundDeviceContext {
    fn capture(owner: &WgpuDevice, configuration: WgpuBabBoundConfigurationEvidence) -> Self {
        Self {
            device: owner.device.clone(),
            queue: owner.queue.clone(),
            configuration,
            resources: WgpuBabBoundResourceState::new(),
        }
    }

    fn matches_owner(&self, owner: &WgpuDevice) -> bool {
        self.device.eq(&owner.device)
            && self.queue.eq(&owner.queue)
            && self.configuration.matches_device(owner)
            && self.resources.authority_intact()
    }
}

/// One indivisible installed authority state. `OnceLock` gives the contained
/// registration a stable borrowed home and forbids later requalification.
struct WgpuBabBoundInstalled {
    registration: GpuBabBoundBackendRegistration,
    qualification: WgpuBabBoundQualification,
    context: WgpuBabBoundDeviceContext,
}

impl WgpuBabBoundInstalled {
    fn stable_qualification(&self) -> bool {
        stable_qualification_matches(
            &self.qualification,
            self.registration.registration_epoch(),
            self.registration.backend_issuer_sha256(),
            &self.context.configuration.issuer_sha256,
        )
    }
}

/// Default-empty retained-BaB provider. The only populated representation is
/// one immutable [`WgpuBabBoundInstalled`] value.
pub(in crate::wgpu_device) struct WgpuBabBoundProvider {
    installed: OnceLock<WgpuBabBoundInstalled>,
}

impl WgpuBabBoundProvider {
    pub(in crate::wgpu_device) const fn new() -> Self {
        Self {
            installed: OnceLock::new(),
        }
    }

    fn qualified(
        context: WgpuBabBoundDeviceContext,
        sound_report_qualified: bool,
        gates: WgpuBabBoundLiveGates,
    ) -> Result<Self, WgpuBabBoundAdmissionAttemptError<NyError>> {
        let input = WgpuBabBoundAdmissionInput {
            configuration: &context.configuration,
            sound_report_qualified,
            gates,
        };
        let (admission, registration) = issue_registration_after_admission(&input, |issuer| {
            GpuBabBoundBackendRegistration::new(issuer)
        })?;
        let qualification = WgpuBabBoundQualification {
            registration_epoch: registration.registration_epoch(),
            issuer_sha256: admission.issuer_sha256,
            sound_report_qualified,
            kernel_selfcheck_passed: gates.kernel_selfcheck_passed,
        };
        // Every fallible admission/hash/registration operation is complete.
        // The remaining plain moves and local `OnceLock::from` publication
        // cannot observe or expose a partially installed provider.
        Ok(Self {
            installed: OnceLock::from(WgpuBabBoundInstalled {
                registration,
                qualification,
                context,
            }),
        })
    }

    fn installed(&self) -> Option<&WgpuBabBoundInstalled> {
        self.installed.get()
    }

    fn current_live_gates(&self) -> WgpuBabBoundLiveGates {
        let installed = self.installed();
        let loading_contract_intact = installed.is_some_and(|installed| {
            crate::wgpu_device::shader_loading::denorm_preserve_contract_intact_for(
                installed.context.configuration.denorm_preserve_enabled,
            )
        });
        WgpuBabBoundLiveGates {
            source_gate_open: PRODUCTION_WGPU_BAB_BOUND_AUTHORITY_ENABLED,
            kernel_implemented: WGPU_BAB_BOUND_KERNEL_SELFCHECK_IMPLEMENTED,
            loading_contract_intact,
            base_selfcheck_forced_fail: base_selfcheck_forced_to_fail(),
            kernel_selfcheck_passed: installed
                .is_some_and(|installed| installed.qualification.kernel_selfcheck_passed),
            bab_selfcheck_forced_fail: selfcheck_forced_to_fail(),
            provider_context_intact: installed
                .is_some_and(|installed| installed.context.resources.authority_intact()),
        }
    }

    fn authority_live_with_gates(&self, gates: WgpuBabBoundLiveGates) -> bool {
        self.installed().is_some_and(|installed| {
            authority_predicate(
                &installed.qualification,
                installed.stable_qualification(),
                gates,
            )
        })
    }

    pub(super) fn matches_qualified_owner(&self, owner: &WgpuDevice) -> bool {
        self.installed().is_some_and(|installed| {
            installed.stable_qualification() && installed.context.matches_owner(owner)
        })
    }

    fn numerical_tcb_with_gates(
        &self,
        gates: WgpuBabBoundLiveGates,
    ) -> Option<&dyn GpuBabBoundNumericalTcb> {
        self.authority_live_with_gates(gates).then_some(self)
    }

    pub(super) fn numerical_tcb(&self) -> Option<&dyn GpuBabBoundNumericalTcb> {
        if !PRODUCTION_WGPU_BAB_BOUND_AUTHORITY_ENABLED
            || !WGPU_BAB_BOUND_KERNEL_SELFCHECK_IMPLEMENTED
            || self.installed().is_none()
        {
            return None;
        }
        self.numerical_tcb_with_gates(self.current_live_gates())
    }

    fn phase_policy_with_gates(
        &self,
        gates: WgpuBabBoundLiveGates,
    ) -> Option<GpuBabBoundPhasePolicy> {
        let _authority_live = self.authority_live_with_gates(gates);
        None
    }

    fn prepare_phase_with_gates(
        &self,
        gates: WgpuBabBoundLiveGates,
    ) -> GpuBabBoundBackendOpenPreparation<'_> {
        let _authority_live = self.authority_live_with_gates(gates);
        GpuBabBoundBackendOpenPreparation::CleanDecline(GpuBabBoundPhaseDecline::Unsupported)
    }

    fn phase_policy_now(&self) -> Option<GpuBabBoundPhasePolicy> {
        if !PRODUCTION_WGPU_BAB_BOUND_AUTHORITY_ENABLED
            || !WGPU_BAB_BOUND_KERNEL_SELFCHECK_IMPLEMENTED
            || self.installed().is_none()
        {
            return None;
        }
        self.phase_policy_with_gates(self.current_live_gates())
    }

    fn prepare_phase_now(&self) -> GpuBabBoundBackendOpenPreparation<'_> {
        if !PRODUCTION_WGPU_BAB_BOUND_AUTHORITY_ENABLED
            || !WGPU_BAB_BOUND_KERNEL_SELFCHECK_IMPLEMENTED
            || self.installed().is_none()
        {
            return GpuBabBoundBackendOpenPreparation::CleanDecline(
                GpuBabBoundPhaseDecline::Unsupported,
            );
        }
        self.prepare_phase_with_gates(self.current_live_gates())
    }
}

fn base_selfcheck_forced_to_fail() -> bool {
    super::f32_selfcheck::probe_forced_to_fail()
        || super::eft_selfcheck::probe_forced_to_fail()
        || super::subnormal_selfcheck::probe_forced_to_fail()
        || super::sentinel_taint_selfcheck::probe_forced_to_fail()
}

fn authority_predicate(
    qualification: &WgpuBabBoundQualification,
    stable_qualification: bool,
    gates: WgpuBabBoundLiveGates,
) -> bool {
    gates.source_gate_open
        && gates.kernel_implemented
        && gates.loading_contract_intact
        && !gates.base_selfcheck_forced_fail
        && gates.kernel_selfcheck_passed
        && !gates.bab_selfcheck_forced_fail
        && gates.provider_context_intact
        && qualification.sound_report_qualified
        && qualification.kernel_selfcheck_passed
        && stable_qualification
}

fn stable_qualification_matches(
    qualification: &WgpuBabBoundQualification,
    registration_epoch: u64,
    registration_issuer_sha256: &[u8; 32],
    configuration_issuer_sha256: &[u8; 32],
) -> bool {
    qualification.registration_epoch != 0
        && qualification.registration_epoch == registration_epoch
        && qualification.issuer_sha256 != [0; 32]
        && &qualification.issuer_sha256 == registration_issuer_sha256
        && &qualification.issuer_sha256 == configuration_issuer_sha256
}

/// There is intentionally no kernel dispatch in this slice. This independent
/// dynamic conjunct stays false even if either compile-time gate is changed in
/// isolation.
const fn kernel_selfcheck_for_qualification(_device: &WgpuDevice) -> bool {
    false
}

impl WgpuDevice {
    /// Construct one exact device through the explicit retained-BaB route.
    ///
    /// Static default-dark admission is the first operation. Therefore the
    /// current production build returns a typed refusal without initializing a
    /// GPU, touching environment-backed probe caches, or consuming a core
    /// registration epoch.
    pub fn new_for_verdict_bab_bound(
        _request: WgpuBabBoundVerdictRequest,
    ) -> Result<Self, WgpuBabBoundQualificationError> {
        let schemas = reviewed_schema_bundle();
        static_admission_preflight(
            PRODUCTION_WGPU_BAB_BOUND_AUTHORITY_ENABLED,
            WGPU_BAB_BOUND_KERNEL_SELFCHECK_IMPLEMENTED,
            schemas.as_ref(),
        )
        .map_err(WgpuBabBoundQualificationError::refused)?;
        let schemas = schemas.ok_or_else(|| {
            WgpuBabBoundQualificationError::refused(
                WgpuBabBoundAdmissionRefusal::KernelSchemaMissing,
            )
        })?;

        // This path is unreachable while any static gate above is closed. A
        // future opening reuses the reviewed uncharged constructor on the exact
        // device it will return; charged authority is never an admission input.
        let mut device = Self::new_for_verdict(WgpuVerdictRequest::new())
            .map_err(WgpuBabBoundQualificationError::verdict)?;
        let report = device.verdict_report().cloned().ok_or_else(|| {
            WgpuBabBoundQualificationError::provider(
                None,
                NyError::InternalError(
                    "explicit base-verdict construction returned without its report".to_string(),
                ),
            )
        })?;
        let configuration = WgpuBabBoundConfigurationEvidence::capture(&device, &report, schemas);
        let context = WgpuBabBoundDeviceContext::capture(&device, configuration);
        let kernel_selfcheck_passed = kernel_selfcheck_for_qualification(&device);
        let sound_report_qualified = report.qualified();
        // Final live recheck: no device/context construction or other fallible
        // work occurs between these observations and config hashing followed
        // by the single core registration attempt.
        let gates = WgpuBabBoundLiveGates {
            source_gate_open: PRODUCTION_WGPU_BAB_BOUND_AUTHORITY_ENABLED,
            kernel_implemented: WGPU_BAB_BOUND_KERNEL_SELFCHECK_IMPLEMENTED,
            loading_contract_intact: device.denorm_preserve_contract_intact(),
            base_selfcheck_forced_fail: base_selfcheck_forced_to_fail(),
            kernel_selfcheck_passed,
            bab_selfcheck_forced_fail: selfcheck_forced_to_fail(),
            provider_context_intact: context.resources.authority_intact(),
        };
        let provider = WgpuBabBoundProvider::qualified(context, sound_report_qualified, gates)
            .map_err(|error| match error {
                WgpuBabBoundAdmissionAttemptError::Refused(refusal) => {
                    WgpuBabBoundQualificationError::provider(
                        Some(report.clone()),
                        NyError::UnsupportedConfiguration(refusal.message().to_string()),
                    )
                }
                WgpuBabBoundAdmissionAttemptError::Registration(source) => {
                    WgpuBabBoundQualificationError::provider(Some(report.clone()), source)
                }
            })?;
        device.bab_bound_provider = provider;
        Ok(device)
    }

    pub(super) fn bab_bound_authority_cached(&self) -> bool {
        self.bab_bound_numerical_tcb_cached().is_some()
    }

    pub(super) fn bab_bound_numerical_tcb_cached(&self) -> Option<&dyn GpuBabBoundNumericalTcb> {
        if !self.bab_bound_provider.matches_qualified_owner(self)
            || !self.sound_gpu_authority_cached()
        {
            return None;
        }
        self.bab_bound_provider.numerical_tcb()
    }
}

/// Private TCB adapter. An ordinary caller cannot name this type or obtain a
/// trait object for it; the `WgpuDevice` accessor returns it only after the
/// complete live predicate. Its direct methods still remain authority-free
/// without core's private invocation and always decline before acceptance.
impl GpuBabBoundNumericalTcb for WgpuBabBoundProvider {
    fn registration(&self) -> &GpuBabBoundBackendRegistration {
        &self
            .installed()
            .expect("a WGPU BaB TCB object is exposed only after atomic installation")
            .registration
    }

    fn phase_policy(
        &self,
        _invocation: &GpuBabBoundTcbInvocation<'_>,
    ) -> Option<GpuBabBoundPhasePolicy> {
        self.phase_policy_now()
    }

    fn prepare_phase<'a>(
        &'a self,
        _invocation: &GpuBabBoundTcbInvocation<'_>,
    ) -> GpuBabBoundBackendOpenPreparation<'a> {
        self.prepare_phase_now()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    fn sample_adapter() -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: "test-adapter".to_string(),
            vendor: 0x1234,
            device: 0x5678,
            device_type: wgpu::DeviceType::DiscreteGpu,
            device_pci_bus_id: "0000:01:00.0".to_string(),
            driver: "test-driver".to_string(),
            driver_info: "1.2.3".to_string(),
            backend: wgpu::Backend::Vulkan,
            subgroup_min_size: 32,
            subgroup_max_size: 64,
            transient_saves_memory: false,
        }
    }

    fn sample_schemas() -> WgpuBabBoundReviewedSchemaBundle {
        WgpuBabBoundReviewedSchemaBundle {
            bundle_version: 1,
            provider_abi_sha256: [1; 32],
            receipt_abi_sha256: [2; 32],
            kernel_sha256: [3; 32],
            topology_schema_sha256: [4; 32],
            selfcheck_schema_sha256: [5; 32],
            transcript_schema_sha256: [6; 32],
        }
    }

    fn sample_configuration() -> WgpuBabBoundConfigurationEvidence {
        WgpuBabBoundConfigurationEvidence::new(
            sample_adapter(),
            wgpu::Features::TIMESTAMP_QUERY,
            wgpu::Limits::default(),
            DenormPreservePolicy::Auto,
            true,
            true,
            true,
            [WgpuVerdictRungOutcome::Passed; 5],
            sample_schemas(),
        )
    }

    fn open_gates() -> WgpuBabBoundLiveGates {
        WgpuBabBoundLiveGates {
            source_gate_open: true,
            kernel_implemented: true,
            loading_contract_intact: true,
            base_selfcheck_forced_fail: false,
            kernel_selfcheck_passed: true,
            bab_selfcheck_forced_fail: false,
            provider_context_intact: true,
        }
    }

    fn qualification(epoch: u64, issuer_sha256: [u8; 32]) -> WgpuBabBoundQualification {
        WgpuBabBoundQualification {
            registration_epoch: epoch,
            issuer_sha256,
            sound_report_qualified: true,
            kernel_selfcheck_passed: true,
        }
    }

    #[test]
    fn production_preflight_is_independently_default_closed_and_side_effect_free() {
        const {
            assert!(!PRODUCTION_WGPU_BAB_BOUND_AUTHORITY_ENABLED);
            assert!(!WGPU_BAB_BOUND_KERNEL_SELFCHECK_IMPLEMENTED);
            assert!(reviewed_schema_bundle().is_none());
        }

        let schemas = sample_schemas();

        assert_eq!(
            static_admission_preflight(false, true, Some(&schemas)),
            Err(WgpuBabBoundAdmissionRefusal::SourceGateClosed)
        );
        assert_eq!(
            static_admission_preflight(true, false, Some(&schemas)),
            Err(WgpuBabBoundAdmissionRefusal::KernelImplementationMissing)
        );
        assert_eq!(
            static_admission_preflight(true, true, None),
            Err(WgpuBabBoundAdmissionRefusal::KernelSchemaMissing)
        );
        let mut incomplete = schemas;
        incomplete.receipt_abi_sha256 = [0; 32];
        assert_eq!(
            static_admission_preflight(true, true, Some(&incomplete)),
            Err(WgpuBabBoundAdmissionRefusal::KernelSchemaMissing)
        );

        let result = WgpuDevice::new_for_verdict_bab_bound(WgpuBabBoundVerdictRequest::new());
        let error = match result {
            Ok(_) => panic!("default-dark BaB qualification unexpectedly created a device"),
            Err(error) => error,
        };
        assert!(error.verdict_report().is_none());
        assert!(error.to_string().contains("source gate is closed"));
    }

    #[test]
    fn registration_issuer_runs_zero_times_on_refusal_and_once_after_full_admission() {
        let configuration = sample_configuration();
        let calls = Cell::new(0_u32);
        let closed = WgpuBabBoundAdmissionInput {
            configuration: &configuration,
            sound_report_qualified: true,
            gates: WgpuBabBoundLiveGates {
                source_gate_open: false,
                ..open_gates()
            },
        };
        let refused = issue_registration_after_admission(&closed, |_| {
            calls.set(calls.get() + 1);
            Ok::<_, ()>(41_u64)
        });
        assert!(matches!(
            refused,
            Err(WgpuBabBoundAdmissionAttemptError::Refused(
                WgpuBabBoundAdmissionRefusal::SourceGateClosed
            ))
        ));
        assert_eq!(calls.get(), 0);

        let open = WgpuBabBoundAdmissionInput {
            configuration: &configuration,
            sound_report_qualified: true,
            gates: open_gates(),
        };
        let (_, fake_registration) = issue_registration_after_admission(&open, |issuer| {
            calls.set(calls.get() + 1);
            assert_eq!(issuer, configuration.issuer_sha256);
            Ok::<_, ()>(41_u64)
        })
        .expect("complete synthetic admission reaches the injected issuer");
        assert_eq!(fake_registration, 41);
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn every_dynamic_admission_conjunct_only_closes() {
        let configuration = sample_configuration();
        let assert_refusal = |sound_report_qualified, gates, expected| {
            let calls = Cell::new(0_u32);
            let input = WgpuBabBoundAdmissionInput {
                configuration: &configuration,
                sound_report_qualified,
                gates,
            };
            let result = issue_registration_after_admission(&input, |_| {
                calls.set(calls.get() + 1);
                Ok::<_, ()>(())
            });
            assert!(matches!(
                result,
                Err(WgpuBabBoundAdmissionAttemptError::Refused(actual)) if actual == expected
            ));
            assert_eq!(calls.get(), 0);
        };
        assert_refusal(
            false,
            open_gates(),
            WgpuBabBoundAdmissionRefusal::SoundVerdictMissing,
        );
        assert_refusal(
            true,
            WgpuBabBoundLiveGates {
                loading_contract_intact: false,
                ..open_gates()
            },
            WgpuBabBoundAdmissionRefusal::LoadingContractLost,
        );
        assert_refusal(
            true,
            WgpuBabBoundLiveGates {
                base_selfcheck_forced_fail: true,
                ..open_gates()
            },
            WgpuBabBoundAdmissionRefusal::BaseSelfcheckForcedFail,
        );
        assert_refusal(
            true,
            WgpuBabBoundLiveGates {
                kernel_selfcheck_passed: false,
                ..open_gates()
            },
            WgpuBabBoundAdmissionRefusal::KernelSelfcheckFailed,
        );
        assert_refusal(
            true,
            WgpuBabBoundLiveGates {
                bab_selfcheck_forced_fail: true,
                ..open_gates()
            },
            WgpuBabBoundAdmissionRefusal::BabSelfcheckForcedFail,
        );
        assert_refusal(
            true,
            WgpuBabBoundLiveGates {
                provider_context_intact: false,
                ..open_gates()
            },
            WgpuBabBoundAdmissionRefusal::ProviderContextFaulted,
        );
    }

    #[test]
    fn canonical_configuration_hash_is_deterministic_and_binds_each_evidence_class() {
        let baseline = sample_configuration();
        const GOLDEN_CONFIG_SHA256: [u8; 32] = [
            0x9f, 0x55, 0x23, 0x2c, 0x5c, 0xdc, 0xf2, 0xdf, 0x31, 0x83, 0xda, 0xf3, 0x54, 0x08,
            0xe1, 0xc5, 0xed, 0xab, 0xa2, 0x6a, 0x5c, 0xc6, 0x26, 0x12, 0xfd, 0xeb, 0x91, 0x6d,
            0x1d, 0xfe, 0x74, 0x4d,
        ];
        assert_eq!(baseline.issuer_sha256, GOLDEN_CONFIG_SHA256);
        assert_ne!(baseline.issuer_sha256, [0; 32]);
        assert_eq!(baseline.issuer_sha256, configuration_sha256(&baseline));
        assert_eq!(baseline.issuer_sha256, sample_configuration().issuer_sha256);

        let mut adapter = sample_adapter();
        adapter.driver_info.push_str("-changed");
        let adapter_changed = WgpuBabBoundConfigurationEvidence::new(
            adapter,
            baseline.enabled_features,
            baseline.limits.clone(),
            baseline.denorm_preserve_policy,
            baseline.denorm_preserve_enabled,
            baseline.loading_contract_intact_at_qualification,
            baseline.base_source_gate_open,
            baseline.base_rung_outcomes,
            baseline.schemas.clone(),
        );
        assert_ne!(baseline.issuer_sha256, adapter_changed.issuer_sha256);

        let feature_changed = WgpuBabBoundConfigurationEvidence::new(
            baseline.adapter.clone(),
            baseline.enabled_features | wgpu::Features::PASSTHROUGH_SHADERS,
            baseline.limits.clone(),
            baseline.denorm_preserve_policy,
            baseline.denorm_preserve_enabled,
            baseline.loading_contract_intact_at_qualification,
            baseline.base_source_gate_open,
            baseline.base_rung_outcomes,
            baseline.schemas.clone(),
        );
        assert_ne!(baseline.issuer_sha256, feature_changed.issuer_sha256);

        let mut limits = baseline.limits.clone();
        limits.max_storage_buffer_binding_size += 4;
        let limits_changed = WgpuBabBoundConfigurationEvidence::new(
            baseline.adapter.clone(),
            baseline.enabled_features,
            limits,
            baseline.denorm_preserve_policy,
            baseline.denorm_preserve_enabled,
            baseline.loading_contract_intact_at_qualification,
            baseline.base_source_gate_open,
            baseline.base_rung_outcomes,
            baseline.schemas.clone(),
        );
        assert_ne!(baseline.issuer_sha256, limits_changed.issuer_sha256);

        let loading_changed = WgpuBabBoundConfigurationEvidence::new(
            baseline.adapter.clone(),
            baseline.enabled_features,
            baseline.limits.clone(),
            DenormPreservePolicy::Required,
            baseline.denorm_preserve_enabled,
            baseline.loading_contract_intact_at_qualification,
            baseline.base_source_gate_open,
            baseline.base_rung_outcomes,
            baseline.schemas.clone(),
        );
        assert_ne!(baseline.issuer_sha256, loading_changed.issuer_sha256);

        let loading_observation_changed = WgpuBabBoundConfigurationEvidence::new(
            baseline.adapter.clone(),
            baseline.enabled_features,
            baseline.limits.clone(),
            baseline.denorm_preserve_policy,
            baseline.denorm_preserve_enabled,
            false,
            baseline.base_source_gate_open,
            baseline.base_rung_outcomes,
            baseline.schemas.clone(),
        );
        assert_ne!(
            baseline.issuer_sha256,
            loading_observation_changed.issuer_sha256
        );

        let mut schemas = baseline.schemas.clone();
        schemas.transcript_schema_sha256 = [8; 32];
        let schema_changed = WgpuBabBoundConfigurationEvidence::new(
            baseline.adapter.clone(),
            baseline.enabled_features,
            baseline.limits.clone(),
            baseline.denorm_preserve_policy,
            baseline.denorm_preserve_enabled,
            baseline.loading_contract_intact_at_qualification,
            baseline.base_source_gate_open,
            baseline.base_rung_outcomes,
            schemas,
        );
        assert_ne!(baseline.issuer_sha256, schema_changed.issuer_sha256);

        let base_rung_changed = WgpuBabBoundConfigurationEvidence::new(
            baseline.adapter.clone(),
            baseline.enabled_features,
            baseline.limits.clone(),
            baseline.denorm_preserve_policy,
            baseline.denorm_preserve_enabled,
            baseline.loading_contract_intact_at_qualification,
            baseline.base_source_gate_open,
            [
                WgpuVerdictRungOutcome::Failed,
                WgpuVerdictRungOutcome::Passed,
                WgpuVerdictRungOutcome::Passed,
                WgpuVerdictRungOutcome::Passed,
                WgpuVerdictRungOutcome::Passed,
            ],
            baseline.schemas.clone(),
        );
        assert_ne!(baseline.issuer_sha256, base_rung_changed.issuer_sha256);

        let base_source_changed = WgpuBabBoundConfigurationEvidence::new(
            baseline.adapter.clone(),
            baseline.enabled_features,
            baseline.limits.clone(),
            baseline.denorm_preserve_policy,
            baseline.denorm_preserve_enabled,
            baseline.loading_contract_intact_at_qualification,
            false,
            baseline.base_rung_outcomes,
            baseline.schemas.clone(),
        );
        assert_ne!(baseline.issuer_sha256, base_source_changed.issuer_sha256);
    }

    #[test]
    fn tampered_configuration_and_mismatched_epoch_or_issuer_refuse() {
        let mut configuration = sample_configuration();
        configuration.adapter.name.push_str("-tampered");
        let input = WgpuBabBoundAdmissionInput {
            configuration: &configuration,
            sound_report_qualified: true,
            gates: open_gates(),
        };
        assert_eq!(
            evaluate_admission(&input).unwrap_err(),
            WgpuBabBoundAdmissionRefusal::ConfigurationIdentityMismatch
        );

        let issuer = [9; 32];
        let qualification = qualification(73, issuer);
        assert!(stable_qualification_matches(
            &qualification,
            73,
            &issuer,
            &issuer
        ));
        assert!(!stable_qualification_matches(
            &qualification,
            74,
            &issuer,
            &issuer
        ));
        assert!(!stable_qualification_matches(
            &qualification,
            73,
            &[10; 32],
            &issuer
        ));
        assert!(!stable_qualification_matches(
            &qualification,
            73,
            &issuer,
            &[11; 32]
        ));
    }

    #[test]
    fn default_provider_never_installs_or_initializes_failure_env() {
        let provider = WgpuBabBoundProvider::new();
        assert!(provider.installed().is_none());
        for _ in 0..4 {
            assert!(provider.numerical_tcb().is_none());
        }
        assert!(provider.installed().is_none());
    }

    #[test]
    fn provider_resource_namespace_starts_empty_and_fault_only_closes() {
        let resources = WgpuBabBoundResourceState::new();
        assert!(resources.authority_intact());
        assert!(resources
            .allocations
            .lock()
            .expect("fresh provider allocation ledger")
            .is_empty());
        assert_eq!(resources.next_buffer_namespace.load(Ordering::Acquire), 1);

        resources.faulted.store(true, Ordering::Release);
        assert!(!resources.authority_intact());
    }

    #[test]
    fn generic_and_charged_constructors_have_no_bab_arming_hook() {
        // Normalize line endings before splitting: the separator spans a line
        // with `\n`, but `.rs` checks out CRLF under core.autocrlf, so on
        // Windows the split found nothing, `next()` returned the WHOLE file
        // including this test module, and the production-prefix counts below
        // came out too high.
        let source = include_str!("bab_bound_authority.rs").replace("\r\n", "\n");
        let provider_source = source
            .split("#[cfg(test)]\nmod tests")
            .next()
            .expect("provider has a production prefix");
        assert_eq!(
            provider_source
                .matches("GpuBabBoundBackendRegistration::new")
                .count(),
            1
        );
        assert_eq!(
            provider_source
                .matches("installed: OnceLock::from(")
                .count(),
            1
        );
        assert!(!provider_source.contains("installed.set("));
        assert_eq!(
            provider_source
                .matches("device.bab_bound_provider = provider")
                .count(),
            1
        );
        assert!(!provider_source.contains("impl Default for WgpuBabBoundVerdictRequest"));
        assert!(!provider_source.contains("GpuBabBoundBackendSession"));
        assert!(!provider_source.contains("GpuBabBoundPreparedWave"));

        let verdict_source = include_str!("sound_authority.rs");
        assert!(!verdict_source.contains("new_for_verdict_bab_bound"));
        assert!(!verdict_source.contains("WgpuBabBoundVerdictRequest"));

        let device_source = include_str!("../device.rs");
        assert_eq!(
            device_source.matches("WgpuBabBoundProvider::new").count(),
            1
        );
        assert!(!device_source.contains("new_for_verdict_bab_bound"));
    }

    #[test]
    fn loss_at_raw_transition_can_only_decline() {
        let provider = WgpuBabBoundProvider::new();
        let lost = WgpuBabBoundLiveGates {
            loading_contract_intact: false,
            ..open_gates()
        };
        assert!(provider.phase_policy_with_gates(lost).is_none());
        assert!(matches!(
            provider.prepare_phase_with_gates(lost),
            GpuBabBoundBackendOpenPreparation::CleanDecline(GpuBabBoundPhaseDecline::Unsupported)
        ));

        set_force_selfcheck_fail(true);
        assert!(selfcheck_forced_to_fail());
        set_force_selfcheck_fail(false);
    }

    #[test]
    fn repository_tcb_implementations_match_the_review_allowlist() {
        use std::collections::BTreeMap;
        use std::fs;
        use std::path::{Path, PathBuf};

        fn scan_rs(path: &Path, needle: &str, found: &mut BTreeMap<String, usize>, root: &Path) {
            for entry in fs::read_dir(path).expect("read source-policy directory") {
                let entry = entry.expect("read source-policy entry");
                let path = entry.path();
                if path.is_dir() {
                    scan_rs(&path, needle, found, root);
                } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
                    let source =
                        fs::read_to_string(&path).expect("read Rust source for TCB policy");
                    let normalized = source.split_whitespace().collect::<Vec<_>>().join(" ");
                    // Count only REAL impls. The needle also appears as a
                    // string literal inside the sibling guards that police this
                    // same trait (`gemm_gpu_bab_bound.rs` asserts its own
                    // production prefix contains zero of them), and a bare
                    // substring scan counted those quotes as implementations —
                    // it reported 4 in a file holding 3. An impl item is
                    // followed by the type name; the quoted form is followed by
                    // the closing quote.
                    let count = normalized
                        .match_indices(needle)
                        .filter(|(index, _)| {
                            normalized[index + needle.len()..]
                                .chars()
                                .next()
                                .is_some_and(|c| c.is_alphabetic() || c == '_')
                        })
                        .count();
                    if count != 0 {
                        let relative = path
                            .strip_prefix(root)
                            .expect("scanned path is below workspace root")
                            .to_string_lossy()
                            .replace('\\', "/");
                        found.insert(relative, count);
                    }
                }
            }
        }

        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("canonical workspace root");
        // Assembled from fragments so this file does not contain its own needle
        // contiguously — otherwise the scan would count itself.
        let needle = ["impl ", "GpuBabBoundNumericalTcb", " for "].concat();
        let mut found = BTreeMap::new();
        scan_rs(&workspace.join("crates"), &needle, &mut found, &workspace);

        let expected = BTreeMap::from([
            ("crates/ny-core/src/gemm_gpu_bab_bound.rs".to_string(), 3),
            (
                "crates/ny-gpu/src/wgpu_device/ops/bab_bound_authority.rs".to_string(),
                1,
            ),
            // REVIEW-POLICY UPDATE. `RetainedBabCapabilityMock` and
            // `LosingRetainedBabCapabilityMock`, added by 4160ce8fe ('add
            // retained BaB schedule certificates'). Both are `#[cfg(test)]`
            // doubles that model a backend LOSING its retained capability, so
            // they exercise the decline path rather than widening the trusted
            // base — the same standing the already-listed `FakeBackend` /
            // `ScheduleBackend` doubles have. A PRODUCTION implementation
            // arriving in this file would still trip the gate, which is the
            // property worth keeping.
            ("crates/ny-propagate/src/sound_gpu_gate.rs".to_string(), 2),
        ]);
        assert_eq!(
            found, expected,
            "adding or moving a numerical-TCB implementation requires an explicit review-policy update"
        );
    }
}
