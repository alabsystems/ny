// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Complete verification via branch-and-bound with CROWN bounds.
//!
//! Exposes the β-CROWN complete verifier: configuration, branching heuristics,
//! and the verification result/status surface. Branch-and-bound splits the
//! problem into subdomains, bounding each with CROWN, to reach a complete
//! (sound and exact) verdict rather than an incomplete over-approximation.
//!
//! ```rust
//! use ny_api::bab::{BetaCrownConfig, BetaCrownVerifier};
//!
//! let config = BetaCrownConfig::default();
//! let verifier = BetaCrownVerifier::new(config);
//! # let _ = verifier;
//! ```

pub use ny_propagate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownResult, BetaCrownVerifier, BranchingHeuristic,
    ConvMode, DomainSpecCrownResult, InputClipType, KfsbReduceOp, PhaseBudgetConfig,
};
