// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Root avoice verifier-smoke entrypoint (#3950).
//!
//! Consolidates all avoice verifier-smoke tests behind a single root seam
//! while delegating verifier-only containment helpers and family lanes into
//! child modules.
//!
//! Family-local graph construction and assertion helpers stay in their
//! existing family modules. Shared verifier-spec assembly and `Unknown`
//! structural assertions stay in `avoice/common.rs`.

mod duration;
mod kokoro;
mod shared;
mod speaker;
mod talker;
