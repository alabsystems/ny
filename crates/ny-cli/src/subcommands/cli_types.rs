// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CLI argument types shared across subcommands.
//!
//! Extracted from the monolith subcommands.rs to separate reusable types
//! from the `Commands` enum definition.

use clap::ValueEnum;
use ny_gpu::Backend;
use ny_propagate::layers::{LayerNormCrownMode, LayerNormMode};
use ny_propagate::MulBinaryRelaxationMode;
use ny_propagate::{GradientMethod, Optimizer};

/// Compute backend selection for accelerated operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub(crate) enum BackendArg {
    /// CPU with Rayon parallelization (default, always available)
    #[default]
    Cpu,
    /// wgpu compute request; verdict-bearing proof routes currently fall back to CPU
    Wgpu,
}

impl From<BackendArg> for Backend {
    fn from(arg: BackendArg) -> Self {
        match arg {
            BackendArg::Cpu => Backend::Cpu,
            BackendArg::Wgpu => Backend::Wgpu,
        }
    }
}

impl std::fmt::Display for BackendArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendArg::Cpu => write!(f, "cpu"),
            BackendArg::Wgpu => write!(f, "wgpu"),
        }
    }
}

/// Log output format for structured logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub(crate) enum LogFormat {
    /// Human-readable text output (default)
    #[default]
    Text,
    /// JSON lines format for machine parsing
    Json,
}

/// MulBinary relaxation mode for CROWN propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub(crate) enum MulBinaryRelaxationArg {
    /// McCormick envelope relaxation (tighter, default)
    #[default]
    Mccormick,
    /// Middle-point relaxation (looser but faster)
    Middle,
}

impl From<MulBinaryRelaxationArg> for MulBinaryRelaxationMode {
    fn from(arg: MulBinaryRelaxationArg) -> Self {
        match arg {
            MulBinaryRelaxationArg::Mccormick => MulBinaryRelaxationMode::McCormick,
            MulBinaryRelaxationArg::Middle => MulBinaryRelaxationMode::Middle,
        }
    }
}

impl MulBinaryRelaxationArg {
    /// Parse from a config string (for YAML config compatibility).
    pub(crate) fn from_config_str(s: &str) -> Option<Self> {
        match s {
            "mccormick" => Some(Self::Mccormick),
            "middle" => Some(Self::Middle),
            _ => None,
        }
    }
}

/// Normalization CROWN mode for LayerNorm/RMSNorm/GroupNorm/InstanceNorm1d/AdaIN1d.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub(crate) enum LayerNormModeArg {
    /// Sound Jacobian + IBP margin validation
    #[value(name = "ibp-validated")]
    IbpValidated,
    /// Returns error on unsound paths (default)
    #[default]
    Sound,
    /// Identity relaxation (sound but loses correlations)
    Cut,
    /// Heuristic sampling (not provably sound)
    Sampling,
}

impl From<LayerNormModeArg> for LayerNormCrownMode {
    fn from(arg: LayerNormModeArg) -> Self {
        match arg {
            LayerNormModeArg::IbpValidated => LayerNormCrownMode::IbpValidated,
            LayerNormModeArg::Sound => LayerNormCrownMode::Sound,
            LayerNormModeArg::Cut => LayerNormCrownMode::Cut,
            LayerNormModeArg::Sampling => LayerNormCrownMode::Sampling,
        }
    }
}

/// LayerNorm normalization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub(crate) enum LayerNormNormModeArg {
    /// Full LayerNorm (default)
    #[default]
    Standard,
    /// DeepT-style: subtract mean without variance normalization
    #[value(name = "mean-only")]
    MeanOnly,
}

impl From<LayerNormNormModeArg> for LayerNormMode {
    fn from(arg: LayerNormNormModeArg) -> Self {
        match arg {
            LayerNormNormModeArg::Standard => LayerNormMode::Standard,
            LayerNormNormModeArg::MeanOnly => LayerNormMode::MeanOnly,
        }
    }
}

/// Gradient method for alpha-CROWN optimization.
///
/// The `#[default]` mirrors the engine default (`GradientMethod::AnalyticChain`,
/// #2035); when the flag is omitted (`None`) the engine default applies directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub(crate) enum AlphaGradientMethodArg {
    /// SPSA zero-order optimization (O(1) passes per iteration; noise-dominated
    /// for networks with many unstable neurons)
    Spsa,
    /// Finite differences (O(n) passes, accurate but slow)
    Fd,
    /// Local gradients from CROWN backward (experimental)
    Analytic,
    /// True chain-rule gradients (default; closest to reference
    /// alpha-beta-CROWN's loss.backward())
    #[default]
    #[value(name = "analytic-chain")]
    AnalyticChain,
}

impl From<AlphaGradientMethodArg> for GradientMethod {
    fn from(arg: AlphaGradientMethodArg) -> Self {
        match arg {
            AlphaGradientMethodArg::Spsa => GradientMethod::Spsa,
            AlphaGradientMethodArg::Fd => GradientMethod::FiniteDifferences,
            AlphaGradientMethodArg::Analytic => GradientMethod::Analytic,
            AlphaGradientMethodArg::AnalyticChain => GradientMethod::AnalyticChain,
        }
    }
}

/// Optimizer for alpha-CROWN parameter updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub(crate) enum AlphaOptimizerArg {
    /// Adam optimizer (adaptive moment estimation, matches alpha-beta-CROWN)
    #[default]
    Adam,
    /// SGD with momentum
    Sgd,
}

impl From<AlphaOptimizerArg> for Optimizer {
    fn from(arg: AlphaOptimizerArg) -> Self {
        match arg {
            AlphaOptimizerArg::Adam => Optimizer::Adam,
            AlphaOptimizerArg::Sgd => Optimizer::Sgd,
        }
    }
}

/// Complete verifier method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub(crate) enum CompleteVerifierArg {
    /// Auto (default): run branch-and-bound, then escalate to the MIP
    /// (ay) complete verifier when BaB is inconclusive and the network is
    /// MIP-encodable (sequential Linear+ReLU within a size cap). This makes the
    /// exact MIP technique automatic for SAT-encoded / loose-CROWN nets without
    /// requiring an explicit `--complete-verifier mip`.
    #[default]
    Auto,
    /// Branch-and-bound with beta-CROWN bounds only (never escalates to MIP).
    Bab,
    /// Exact Big-M encoding solved by ay.
    Mip,
}

/// MIP solver backend.
///
/// SOLVER POLICY (ny-mip docs/SOLVER_POLICY.md): all solving in ny happens
/// on ay — foreign solvers are not selectable. The enum survives (single
/// variant) so the CLI surface and preset plumbing stay stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub(crate) enum MipSolverArg {
    /// ay solver: exact QF_LRA Big-M encoding (the only backend)
    #[default]
    AY,
}

impl MipSolverArg {
    /// The ny-mip backend for this solver arg.
    #[cfg(feature = "mip")]
    pub(crate) fn mip_backend(self) -> ny_mip::MipBackend {
        match self {
            MipSolverArg::AY => ny_mip::MipBackend::Ay,
        }
    }
}
