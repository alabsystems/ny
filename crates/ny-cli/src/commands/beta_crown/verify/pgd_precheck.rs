// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! PGD precheck types and wrappers for the sequential MIP path.
//!
//! Part of #3865: PGD-to-HiGHS warm start.

use anyhow::Result;
use ndarray::ArrayD;
use ny_core::GemmEngine;
use ny_propagate::{PgdAlphaMode, PgdConfig, PgdInitialization, PgdOptimizer};
use ny_tensor::BoundedTensor;

/// Result of the PGD precheck before MIP solving.
///
/// Splits the PGD outcome into two paths:
/// 1. A confirmed full-spec counterexample that short-circuits to `Violated`
/// 2. A non-certifying best candidate that can warm-start the MIP solver
#[derive(Default)]
pub(in crate::commands::beta_crown) struct PgdMipPrecheck {
    /// A confirmed counterexample (input, output) that passed full-spec confirmation.
    /// When present, the caller should return `Violated` immediately.
    pub confirmed_counterexample: Option<(ArrayD<f32>, ArrayD<f32>)>,
    /// The best PGD candidate input, even when it did not prove a full counterexample.
    /// Used to warm-start the HiGHS MIP solver.
    pub warm_start_candidate: Option<ArrayD<f32>>,
}

/// PGD upfront check for the sequential MIP path.
///
/// Returns the richer `PgdMipPrecheck` so callers can preserve the best PGD
/// candidate for HiGHS warm-starting when no confirmed counterexample exists.
// Justification: the wrapper deliberately preserves the full attack context at
// the callsite because the MIP path chooses deadline and engine externally.
#[allow(clippy::too_many_arguments)]
pub(in crate::commands::beta_crown) fn try_pgd_before_mip(
    network: &ny_propagate::Network,
    input: &BoundedTensor,
    vnnlib: &ny_onnx::vnnlib::VnnLibSpec,
    pgd_restarts: usize,
    pgd_steps: usize,
    initialization: PgdInitialization,
    osi_steps: usize,
    deadline: Option<std::time::Instant>,
    restart_when_stuck: bool,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<PgdMipPrecheck> {
    if vnnlib.output_constraint_clauses.len() > 1 {
        return Ok(PgdMipPrecheck::default());
    }
    super::pgd::try_pgd_before_mip_with_candidate_with_config(
        network,
        input,
        vnnlib,
        PgdConfig {
            num_restarts: pgd_restarts,
            num_steps: pgd_steps,
            step_size: 0.01,
            spsa_delta: 0.001,
            seed: 42,
            parallel: true,
            deadline,
            restart_when_stuck,
            initialization,
            osi_steps,
            optimizer: PgdOptimizer::SignedGradient,
            alpha_mode: PgdAlphaMode::Scalar(0.01),
            ..Default::default()
        },
        gemm_engine,
        json,
    )
}
