// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-network bound composition for system-level verification.
//!
//! Composes per-model `BoundCertificate`s through pipeline chaining,
//! linear audio mixers, and system-level property checks.
//!
//! Part of #3517.

pub mod certificate;
pub mod mixer;
pub mod pipeline;
pub mod properties;

#[cfg(test)]
mod tests;
