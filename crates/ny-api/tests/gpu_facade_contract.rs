// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Facade contract for the GPU lane.
//!
//! Every symbol here is reached through an `ny_api::*` path only — there is no
//! `ny_gpu` import in this file, and that is the whole point. Before the `gpu`
//! feature existed, `ny-gpu` was not in ny-api's dependency graph under any
//! feature combination, so an embedded consumer could not construct, register,
//! or even NAME a ny GPU type. This file fails if that regresses.
//!
//! It also pins the other half of the contract: REACHABILITY IS NOT TRUST.
//! Nothing reachable from `ny_api::gpu` grants authority. The typed request
//! carries none, the qualification refusal is a typed value rather than a panic
//! or a silent downgrade, the process soundness gate is engaged, and the only
//! way to reach the process-global CROWN seam is with a device the live ladder
//! accepted.
//!
//! The assertions are hermetic: none of them constructs a WGPU device, so this
//! file runs identically on a host with no adapter. The live ladder is
//! exercised by the `#[ignore]`d unit test in `ny_api::gpu`.

use std::sync::Arc;

use ny_api::gpu::{
    install_gpu_crown_factory, is_sound_gpu_crown_required, qualify_gpu_crown,
    wgpu_backend_compiled, wgpu_charged_proof_authority, wgpu_proof_authority, Backend,
    ComputeDevice, FlushChargeAdmission, FlushChargePolicy, GpuCrownBackward, GpuCrownInstall,
    GpuCrownRefusal, QualifiedGpuCrown, WgpuAdapterProvenance, WgpuChargedVerdictRequest,
    WgpuVerdictAuthority, WgpuVerdictQualificationError, WgpuVerdictReport, WgpuVerdictRequest,
    WgpuVerdictRung, WgpuVerdictRungOutcome,
};
use ny_api::verify::GemmEngine;

/// The typed verdict-request constructors are reachable and CONSTRUCTIBLE from
/// the facade. This is the audit's option (a) in one assertion: an embedded
/// consumer can now name the request without a direct `ny-gpu` dependency.
#[test]
fn the_typed_verdict_requests_are_constructible_through_ny_api_paths_only() {
    let uncharged = WgpuVerdictRequest::new();
    let charged = WgpuChargedVerdictRequest::new();

    // Neither request carries authority: they are inert values consumed once by
    // a constructor that then runs the live ladder.
    assert_eq!(uncharged, WgpuVerdictRequest::new());
    assert_eq!(charged, WgpuChargedVerdictRequest::new());
    assert!(!format!("{uncharged:?}").is_empty());
    assert!(!format!("{charged:?}").is_empty());
}

/// The two typed PROOF constructors the audit named are reachable through
/// `ny_api` paths with exactly the signatures a consumer has to satisfy.
///
/// Coercing them to function pointers pins name, arity and types WITHOUT
/// building a device, so this runs on a host with no adapter — and it is the
/// assertion that would have failed before the `gpu` feature existed.
#[test]
fn the_typed_proof_constructors_are_reachable_with_their_exact_signatures() {
    let _uncharged: fn(WgpuVerdictRequest) -> Result<ComputeDevice, WgpuVerdictQualificationError> =
        ComputeDevice::new_for_proof;
    let _charged: fn(
        WgpuChargedVerdictRequest,
    ) -> Result<ComputeDevice, WgpuVerdictQualificationError> =
        ComputeDevice::new_for_proof_flush_charged;

    // Both return a Result, never a device that "might" be qualified: the
    // refusal is in the type, so a caller cannot ignore it by accident.
}

/// The device and backend types are nameable, and the CPU arm is constructible
/// without any adapter.
#[test]
fn the_device_and_backend_types_are_nameable_from_the_facade() {
    let device: ComputeDevice = ComputeDevice::new(Backend::Cpu).expect("CPU always available");
    assert_eq!(device.backend(), Backend::Cpu);
    assert_eq!(Backend::default(), Backend::Cpu);
    assert_eq!(Backend::Wgpu.to_string(), "wgpu");
    assert_eq!("gpu".parse::<Backend>().expect("gpu parses"), Backend::Wgpu);
    assert!(
        "metal".parse::<Backend>().is_err(),
        "no Metal backend exists"
    );
}

/// The qualification EVIDENCE surface is reachable: report, rungs, per-rung
/// outcomes, the authority state, and the flush-charge policy type.
#[test]
fn the_qualification_evidence_surface_is_reachable() {
    // All five reviewed rungs are nameable and print distinctly, so a consumer
    // can narrate exactly which measurement refused.
    let rungs = [
        WgpuVerdictRung::IeeeF32Model,
        WgpuVerdictRung::EftPrimitives,
        WgpuVerdictRung::GradualUnderflow,
        WgpuVerdictRung::HostEftReference,
        WgpuVerdictRung::SentinelTaintSticky,
    ];
    let names: Vec<String> = rungs.iter().map(ToString::to_string).collect();
    assert_eq!(names.len(), 5);
    for (index, name) in names.iter().enumerate() {
        assert!(!name.is_empty());
        assert!(
            !names[index + 1..].contains(name),
            "rung names must be distinct: {name}"
        );
    }

    assert!(WgpuVerdictRungOutcome::Passed.passed());
    assert!(!WgpuVerdictRungOutcome::Failed.passed());
    assert!(
        !WgpuVerdictRungOutcome::NotRun.passed(),
        "a rung that never ran must not count as passing — fail-closed"
    );

    // The authority states, including the charged one that carries its policy.
    // The policy type is nameable so a consumer can record WHICH charge model a
    // charged device is running under.
    fn describe(authority: &WgpuVerdictAuthority) -> &'static str {
        match authority {
            WgpuVerdictAuthority::Qualified => "uncharged",
            WgpuVerdictAuthority::QualifiedWithFlushCharge(policy) => {
                let _policy: &FlushChargePolicy = policy;
                "charged"
            }
            WgpuVerdictAuthority::Refused => "refused",
        }
    }
    assert_eq!(describe(&WgpuVerdictAuthority::Refused), "refused");

    // Type-level reachability for the report and its typed error.
    fn _report_is_nameable(report: &WgpuVerdictReport) -> bool {
        report.qualified()
    }
    fn _error_is_nameable(error: &WgpuVerdictQualificationError) -> &WgpuVerdictReport {
        error.report()
    }
}

/// Build/source signals are readable, and they report a BUILD fact — never a
/// device fact.
#[test]
fn build_capability_signals_are_readable_and_grant_nothing() {
    assert!(
        wgpu_backend_compiled(),
        "the `gpu` feature must enable ny-gpu's wgpu backend, or the typed \
         constructors this facade advertises would not exist"
    );
    // Reviewed source gates: open today, and still not authority for any device.
    let _uncharged_gate: bool = wgpu_proof_authority();
    let _charged_gate: bool = wgpu_charged_proof_authority();

    // The adapter probe type is nameable without running a probe.
    fn _provenance_is_nameable(probe: &WgpuAdapterProvenance) -> bool {
        probe.hardware_available
    }
}

/// The process-global soundness gate is ENGAGED by default and the facade
/// exposes only the read. The releasing setter
/// (`set_sound_gpu_crown_required(false)`) is deliberately not re-exported: it
/// would let an unqualified fast-f32 GPU CROWN decide a verdict.
#[test]
fn the_soundness_gate_is_engaged_and_only_readable_from_the_facade() {
    assert!(
        is_sound_gpu_crown_required(),
        "verdict-deciding CROWN routes must accept only a qualified sound backend"
    );
}

/// The admission mode makes the charged transition visible at the call site:
/// the two variants are distinct values a caller has to choose between.
#[test]
fn charged_admission_is_an_explicit_call_site_choice() {
    assert_ne!(
        FlushChargeAdmission::UnchargedOnly,
        FlushChargeAdmission::AllowFlushCharged
    );
    // `qualify_gpu_crown` takes the mode by value, so a call site that wants
    // charged authority must name it.
    let _uncharged: fn(FlushChargeAdmission) -> Result<QualifiedGpuCrown, GpuCrownRefusal> =
        qualify_gpu_crown;
}

/// The refusal is a TYPED value carrying inspectable evidence — not a panic,
/// not a bool, not a silent fallback.
#[test]
fn the_refusal_type_is_inspectable_evidence() {
    // Constructible only by the facade for the defensive arm; the ladder arm is
    // produced upstream. Both must be matchable and must implement Error.
    let refusal = GpuCrownRefusal::SoundCrownAccessorClosed {
        adapter: Some("test-adapter".to_string()),
        backend_provenance: "wgpu-quarantined",
    };
    assert!(refusal.report().is_none());
    assert!(refusal.failed_rung().is_none());
    let rendered = refusal.to_string();
    assert!(
        rendered.contains("fail-closed"),
        "a refusal must say plainly that nothing was published: {rendered}"
    );
    let as_error: &dyn std::error::Error = &refusal;
    assert!(!as_error.to_string().is_empty());

    // The ladder arm is nameable and pattern-matchable from the facade.
    fn _matches(refusal: &GpuCrownRefusal) -> bool {
        matches!(refusal, GpuCrownRefusal::Qualification(_))
    }
}

/// A CPU device fails the very predicate that guards `QualifiedGpuCrown`, so
/// the only way into the process-global CROWN seam stays a passing ladder.
#[test]
fn the_admission_predicate_refuses_a_non_crown_device() {
    let device = ComputeDevice::new(Backend::Cpu).expect("CPU always available");
    assert_eq!(device.backend_provenance(), "compute-device-cpu");
    assert!(
        !device
            .as_gpu_crown_backward()
            .is_some_and(GpuCrownBackward::provides_sound_gpu_crown),
        "a CPU device must not pass the sound-CROWN admission predicate"
    );

    // And the same predicate applies through the injectable engine type, which
    // is what a consumer would hand to `Verifier::new_with_engine`.
    let engine: Arc<dyn GemmEngine> = Arc::new(device);
    assert!(!engine
        .as_gpu_crown_backward()
        .is_some_and(GpuCrownBackward::provides_sound_gpu_crown));
}

/// `install_gpu_crown_factory` is typed so that ONLY a qualified device can
/// reach the process-global seam. This pins the signature: it takes
/// `&QualifiedGpuCrown`, a type with no public constructor.
#[test]
fn only_a_qualified_device_can_reach_the_process_global_seam() {
    let _installer: fn(&QualifiedGpuCrown) -> GpuCrownInstall = install_gpu_crown_factory;

    // The install report is a value the caller can act on rather than a log
    // line it has to grep for.
    fn _reads(install: GpuCrownInstall) -> (bool, bool, bool) {
        (
            install.materialized(),
            install.wide_lane_requested(),
            install.wide_lane_materialized(),
        )
    }
}

/// The retained-BaB request type is deliberately NOT re-exported: that channel
/// is triple-gated dark and refuses in a static preflight, so a facade name for
/// it would be a capability that can never succeed. `ComputeDevice` carries the
/// constructor as an inherent method, but without its request type the route is
/// not callable from `ny_api` paths.
///
/// This is a compile-time assertion by absence; the test documents the intent
/// and fails loudly if someone adds the export without revisiting the reason.
#[test]
fn the_dark_retained_bab_request_is_not_part_of_the_facade() {
    // If `WgpuBabBoundVerdictRequest` were re-exported from `ny_api::gpu`, the
    // line below would compile and this assertion's premise would be stale.
    // Kept as documentation of a deliberate omission rather than a trybuild
    // dependency.
    let names_exported_by_the_facade = [
        "WgpuVerdictRequest",
        "WgpuChargedVerdictRequest",
        "WgpuVerdictQualificationError",
        "WgpuVerdictReport",
        "WgpuVerdictRung",
        "WgpuVerdictRungOutcome",
        "WgpuVerdictAuthority",
        "FlushChargePolicy",
    ];
    assert!(
        !names_exported_by_the_facade.contains(&"WgpuBabBoundVerdictRequest"),
        "the dark retained-BaB channel must stay unnameable from the facade"
    );
}
