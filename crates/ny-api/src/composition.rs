// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Curated inter-model composition surface for external consumers.
//!
//! This module exposes the in-memory certificate, mixer, pipeline, and
//! property-checking surface.
//! After matching `BoundCertificationResult::Certified`, callers can feed the
//! certificate directly into `SpecBuilder::try_input_source(...)` through the
//! public `VerificationBoundsSource` bridge.
//! Persisted proof bundles are not part of this surface.

pub use ny_propagate::composition::certificate::{
    BoundCertificate, BoundCertificationResult, BoundProvenance,
};
pub use ny_propagate::composition::mixer::{compose_linear_mix, MixerSpec};
pub use ny_propagate::composition::pipeline::{
    PipelineCertificate, PipelineStage, PipelineVerifier,
};
pub use ny_propagate::composition::properties::{
    check_ducking_snr, check_priority_routing, check_spatial_ild, PropertyResult,
};
