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

/// GenBaB nonlinear branching: the [`BranchingHeuristic::GenBaB`] payload and
/// its split-point / candidate-scoring selectors. Needed to configure β-CROWN
/// branching on non-ReLU nonlinearities (bilinear SwiGLU/attention nodes).
pub use ny_propagate::beta_crown::{
    BranchingPointMethod, NonlinearBranchingConfig, NonlinearHeuristicMethod,
};

/// Remaining [`BetaCrownConfig`] sub-config field types, so the full config
/// surface is constructible and inspectable through the facade alone:
/// α-CROWN inner-loop settings (`alpha_config`), adaptive optimizer settings
/// (`adaptive_config`), cut management (`cut_eviction_policy`,
/// `cut_score_weights`), depth-2 branch lookahead, and the artifact authority
/// switch (`verification_artifact_authority`).
pub use ny_propagate::beta_crown::{AdaptiveOptConfig, CutEvictionPolicy, CutScoreWeights};
pub use ny_propagate::{
    AlphaCrownConfig, DepthTwoBranchLookaheadConfig, DepthTwoBranchLookaheadMode, GradientMethod,
    VerificationArtifactAuthority,
};

/// PGD-attack sub-config field types of [`BetaCrownConfig`]
/// (`pgd_initialization`, `pgd_optimizer`, `pgd_alpha_mode`, `pgd_exact_vjp`).
/// ATTACK-side only: PGD search finds counterexamples and never feeds a
/// `Verified` verdict.
pub use ny_propagate::{ExactVjpPgdConfig, PgdAlphaMode, PgdInitialization, PgdOptimizer};

/// Concrete violation-witness payload carried by
/// [`BabVerificationStatus::PotentialViolation`] when the producing stage held
/// a concrete violating point.
pub use ny_propagate::ViolationWitness;
