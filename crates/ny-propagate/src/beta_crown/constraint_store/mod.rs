// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Arena-based constraint store for Clip-and-Verify integration.
//!
//! This module implements an arena-allocated constraint store following ay's
//! ClauseDB pattern for efficient storage of linear constraints during BaB.
//!
//! # Submodules
//!
//! - [`types`] — Shared enums, headers, and view types
//! - [`arena`] — Core `ArenaConstraintStore` with arena allocation
//! - [`splits`] — Split history conversion methods
//! - [`domain`] — `DomainConstraintStore` with base + delta pattern
//!
//! # Sources
//!
//! - ay ClauseDB: `crates/ay-sat/src/clause_db.rs`
//! - Design doc: `designs/2026-01-29-linear-constraint-store.md`
//! - Issue: #234

pub mod arena;
pub mod domain;
pub mod splits;
pub mod types;

// Re-export all public types for backwards compatibility.
pub use arena::ArenaConstraintStore;
pub use domain::DomainConstraintStore;
pub use types::{ConstraintHeader, ConstraintOrigin, ConstraintSense, LinearConstraintRef};

#[cfg(test)]
mod tests;
