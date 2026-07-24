// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Complete verification via MIP (HiGHS).
//!
//! Re-exports the complete-MIP-verification surface: encode an FC+ReLU network
//! plus property as a MILP, then check feasibility (SAT = counterexample,
//! UNSAT = verified). Also exposes the LP bound tightener used to shrink the
//! branch-and-bound search space.

pub use ny_mip::{
    encode_feedforward, LpTightener, MipConfig, MipEncoder, MipError, MipParts, MipResult,
    MipSolver,
};
