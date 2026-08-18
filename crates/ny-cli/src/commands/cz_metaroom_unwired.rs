// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Gated, unwired bridge from an exact-decimal VNN-LIB box through the first
//! normalized ONNX `Conv2d` in a constrained-zonotope domain.
//!
//! This module is deliberately not called by a command or verdict path.  It
//! closes the type/provenance seam between three independently qualified
//! pieces while retaining a fail-closed boundary:
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
