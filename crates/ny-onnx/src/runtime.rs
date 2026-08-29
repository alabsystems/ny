// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Which ONNX execution engine this build carries, and the one seam that installs it.
//!
//! `ort` is an API with a swappable engine behind it. Most targets link the C++
//! ONNX Runtime that upstream publishes as a prebuilt static library. Intel macOS
//! has no such prebuilt — upstream ships none for `x86_64-apple-darwin` in either
//! rc.12 or rc.13 — so that target links [tract], a pure-Rust ONNX engine, through
//! ort's `alternative-backend` seam.
//!
//! An alternative backend must be installed before the first session is built, and
//! installing it twice is not the caller's business to track. Every session in this
//! crate is therefore created through [`session_builder`], which installs the
//! engine exactly once and then defers to `ort` unchanged. Nothing else about the
//! call sites differs between the two engines: the same `Session`, the same
//! `inputs!`, the same `TensorRef`.
//!
//! ## Measured engine differences (269-model corpus, 2026-08-20)
//!
//! The two engines are independent implementations, so they are not
//! interchangeable without evidence. Running both over 269 models — the
//! VNN-COMP suites, this repo's fixtures, and third-party ONNX conformance
//! data — measured, on arm64 so the engine and not the architecture was under
//! test:
//!
//! * 167 models loaded and ran under both; 161 agreed to a 1e-3 relative
//!   tolerance on every output's sum/min/max. The six that did not are tract
//!   gaps on `celu`, `maxpool` with dilations, and `argmax`.
//! * 19 of 101 sampled VNN-COMP models failed under tract where the C++
//!   runtime succeeded — nearly all of them ACASXU, across the 2021 through
//!   2026 suites, plus `collins_rul_cnn` and `cctsdb_yolo`. ACASXU aborts
//!   inside `tract-core` at MODEL LOAD, before any input is supplied, and the
//!   panic crosses the C ABI (`panic in a function that cannot unwind`), so it
//!   takes the process down rather than surfacing as an error. tract-onnx
//!   0.23.5 is the latest release; no upgrade fixes this today.
//!
//! So this seam is the right shape and the pure-Rust engine is the right
//! direction, but until that load bug and those six operators are fixed the
//! two engines do not answer the same questions. Anything that reports a
//! result should say which engine produced it — see [`backend_name`].
//!
//! [tract]: https://github.com/sonos/tract

use std::sync::Once;

use ort::session::builder::SessionBuilder;

/// Human-readable name of the engine linked into this build.
///
/// Reported by diagnostics so a result can say which engine produced it: the two
/// engines are independent implementations of ONNX, and a fidelity or diff result
/// is only interpretable against the engine that computed it.
#[must_use]
pub const fn backend_name() -> &'static str {
    if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        "tract (pure Rust)"
    } else {
        "ONNX Runtime (C++)"
    }
}

/// Install this build's execution engine. Idempotent, and cheap after the first call.
///
/// Only the alternative backend needs installing; the C++ runtime is what `ort`
/// uses when no backend is set, so this is a no-op there.
fn install_backend() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        ort::set_api(ort_tract::api());
    });
}

/// Start building a session with this build's engine installed.
///
/// The one constructor every session in this crate goes through. Using
/// `ort::session::Session::builder()` directly would build a session before the
/// alternative backend is installed, which on Intel macOS is the difference
/// between running and aborting.
pub fn session_builder() -> ort::Result<SessionBuilder> {
    install_backend();
    ort::session::Session::builder()
}
