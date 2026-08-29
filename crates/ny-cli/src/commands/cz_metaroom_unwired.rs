// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Gated bridge from an exact-decimal VNN-LIB box through the first
//! normalized ONNX `Conv2d` in a constrained-zonotope domain.
//!
//! HISTORY AND CURRENT STATUS. This module's stem/propagation surface and the
//! `qualification` submodule's probes were deliberately unwired from every
//! command and verdict path through 2026-08-24. Since #cz-hybrid, the
//! qualification submodule exposes ONE wired production entry —
//! `qualification::run_wired_metaroom_cz_hybrid` — reached from
//! `handle_vnncomp_command` via `commands::cz_metaroom_wired`, gated by the
//! typed `cz_hybrid` preset section's exact instance allowlist and wrapped in
//! a fail-closed `catch_unwind` boundary. Everything else here remains
//! probe-only.
//!
//! The stem closes the type/provenance seam between three independently
//! qualified pieces while retaining a fail-closed boundary:
//!
//! 1. [`ny_onnx::vnnlib::CertifiedInputBox`] supplies outward binary64 input
//!    endpoints plus exact-rational point hints.
//! 2. [`ny_mip::ConstrainedZonotope64::from_certified_bounds`] decomposes the
//!    complete enclosure without trusting those hints to remove width.
//! 3. [`ny_mip::constrained_zonotope_conv2d_unwired`] propagates the normalized
//!    binary32 convolution as exact dyadic binary64 parameters and charges all
//!    contraction error to the output remainder.
//!
//! The ONNX model and propagation graph are both required.  The bridge first
//! requires immutable raw-FLOAT32-initializer provenance, then checks that the
//! direct-input Conv2d name, shape, attributes, kernel bits, and bias bits agree
//! before doing any abstract propagation.  It supports only a single static
//! `[1,C,H,W]` float32 input and a single direct Conv2d consumer; all broader
//! graph surfaces reject with a typed error.

include!("cz_metaroom_unwired_impl.rs");

// Qualification probes plus the wired #cz-hybrid cascade core. Declared here
// (not inside the include!-shared impl file) so the `#[path]` resolves against
// `src/commands/` for the binary while `examples/cz_metaroom_qualification.rs`
// keeps its own manual include of the same source. Non-`#[test]` items compile
// into the release binary; the `#[test]` probes remain test-only as before.
#[path = "cz_metaroom_qualification.rs"]
pub(crate) mod qualification;
