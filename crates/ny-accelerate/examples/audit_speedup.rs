// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Explicit wall-clock audits for the Apple Accelerate GEMM seam.
//!
//! Select one measurement, for example:
//!
//! ```text
//! cargo run --release -p ny-accelerate --example audit_speedup -- isolated
//! ```

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod implementation {
    include!("audits/speedup.rs");
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    implementation::run();
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("audit_speedup requires macOS on aarch64 with Apple Accelerate");
    std::process::exit(2);
}
