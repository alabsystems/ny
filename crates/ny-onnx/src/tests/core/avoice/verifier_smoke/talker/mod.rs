// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::common;
use super::super::talker_attention::verifier_smoke as talker_verifier;
use super::shared::{
    assert_verified_result_contains_center, run_verifier_smoke_route, VerifierSmokeRoute,
};
use ny_core::{Bound, UnknownReason, VerificationResult};

mod crown;
mod real_rope;
mod round_trip;
mod short_seq;
