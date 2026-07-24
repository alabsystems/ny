// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dispatch module: delegation from core types to sibling extension traits.
//!
//! This module exists to break the bidirectional dependency between `core/` and
//! its sibling modules (`ibp`, `alpha_crown`, `graph_crown`, `graph_ibp`).
//!
//! Before: `core/` depended on siblings (for delegation) AND siblings depended
//! on `core/` (for types). This created a dependency cycle.
//!
//! After: `core/` is a pure type+utility module. `dispatch/` imports both `core/`
//! types and sibling extension traits, providing the unified public API surface
//! without any inversion.
//!
//! See #2380 and `designs/2026-02-27-core-module-dependency-inversion.md`.

mod graph_crown;
mod graph_ibp;
mod network_alpha;
mod network_ibp;
