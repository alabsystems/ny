// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Explicit OFF/FAER/Accelerate end-to-end attribution measurement.
//!
//! ```text
//! cargo run --release -p ny-accelerate --example audit_attribution
//! ```

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod implementation {
    include!("audits/attribution.rs");
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    implementation::run();
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("audit_attribution requires macOS on aarch64 with Apple Accelerate");
    std::process::exit(2);
}
