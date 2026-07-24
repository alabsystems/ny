// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Curated probabilistic-bounds surface for external consumers.
//!
//! Two complementary families of probabilistic certificates over an input
//! region, both backed by CROWN sound over-approximations:
//!
//! - **Monte-Carlo + concentration** ([`MonteCarloVerifier`],
//!   [`ProbabilisticBound`], [`ConcentrationCertificate`]): sample inputs from a
//!   distribution, evaluate the network, and bound the true mean / deviation
//!   with non-asymptotic inequalities (Hoeffding, McDiarmid) whose ranges come
//!   from CROWN. Distribution-free beyond boundedness / Lipschitzness.
//! - **Distributional propagation** ([`propagate_distribution`],
//!   [`DistributionalBound`], plus CGF / higher-moment / branch-and-bound
//!   refinements): push an analytic input distribution
//!   ([`AnalyticDistribution`]) through the CROWN linear relaxation to obtain
//!   output mean / variance bounds and quantile or tail-probability estimates
//!   analytically (Boetius et al., ICML 2025).
//!
//! Promoted to the stable facade so downstream consumers no longer need to
//! reach past it into ny-propagate.

/// Monte-Carlo verification: sample inputs from a distribution, evaluate the
/// network, and bound output statistics against CROWN over-approximations.
pub use ny_propagate::probabilistic::monte_carlo::{
    InputDistribution, MonteCarloVerifier, ProbabilisticBound,
};

/// Concentration-inequality certificates (Hoeffding, McDiarmid) with Lipschitz
/// estimation for non-asymptotic probabilistic bounds.
pub use ny_propagate::probabilistic::concentration::{
    estimate_lipschitz_from_network, hoeffding_bound, mcdiarmid_bound, mcdiarmid_bound_optimistic,
    ConcentrationCertificate, HoeffdingBound, LipschitzEstimate, McDiarmidBound,
};

/// Distributional propagation through CROWN linear bounds: analytic output
/// mean / variance, quantile, and tail-probability bounds.
pub use ny_propagate::probabilistic::distributional::{
    propagate_distribution, AnalyticDistribution, DistributionalBound,
};

/// Distribution-aware (DA-CROWN) refinements: path-specific variance tightening
/// and distributional objective / gradient for alpha-CROWN guidance.
pub use ny_propagate::probabilistic::distributional_alpha::{
    distributional_gradient, distributional_objective, propagate_distribution_tight,
};

/// CGF (cumulant-generating-function) propagation: Chernoff tail bounds for
/// linear forms of independent inputs.
pub use ny_propagate::probabilistic::cgf::{propagate_cgf, CgfBound};

/// Higher-order moment propagation: skewness / kurtosis with Cornish-Fisher
/// quantile tightening.
pub use ny_propagate::probabilistic::moments::{cornish_fisher_w, propagate_moments, MomentBound};

/// Branch-and-bound refinement of distributional bounds over input subregions
/// (law-of-total-variance combination with configurable split strategy).
pub use ny_propagate::probabilistic::branch_and_bound::{
    refine_distributional_bounds, BranchAndBoundConfig, BranchAndBoundResult, SplitStrategy,
};
