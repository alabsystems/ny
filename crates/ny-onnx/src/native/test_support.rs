// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::Result;
use std::path::Path;

/// Integration-test bridge for deterministic native loader fail-closed coverage.
#[doc(hidden)]
pub fn directory_contains_extension_in_entries<I, P>(
    dir: &Path,
    entries: I,
    extension: &str,
) -> Result<bool>
where
    I: IntoIterator<Item = std::io::Result<P>>,
    P: AsRef<Path>,
{
    super::weights::directory_contains_extension_in_entries(dir, entries, extension)
}
