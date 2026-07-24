// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// Options for batched domain storage.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchedDomainOptions {
    /// Enable static intermediate bound transfer (interm_transfer).
    pub enable_interm_transfer: bool,
}
