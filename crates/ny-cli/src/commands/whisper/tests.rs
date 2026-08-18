// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for the unavailable Whisper verification command guard.

use super::whisper_verification_unavailable;

#[test]
fn whisper_verification_guard_fails_closed_without_a_verdict() {
    let error = whisper_verification_unavailable()
        .expect_err("unimplemented Whisper verification must fail closed");

    assert_eq!(
        error.to_string(),
        "Whisper verification is unavailable: verifier execution is not implemented; \
         no verification verdict was produced"
    );
}
