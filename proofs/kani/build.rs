// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

fn main() {
    // Tell Cargo that cfg(kani) is a valid configuration
    // (kani is injected by cargo-kani at verification time)
    println!("cargo::rustc-check-cfg=cfg(kani)");
}
