// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::ops::Deref;

use serde::{Deserialize, Serialize};

use crate::{Bound, InformativeCounterexample, SoundnessProvenance, VerificationProof};

/// Typed tag identifying which verification method produced a result.
///
/// Replaces the previous `Option<String>` storage in `VerificationResult`.
/// Known variants cover every current in-repo producer; `Other(String)` is
/// the forward-compatible escape hatch for new executors or downstream
/// integrations.
///
/// JSON serialization is a plain string (not a tagged enum object) so that
/// existing CLI/Python consumers see no format change.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MethodUsed {
    Ibp,
    Crown,
    AlphaCrown,
    SdpCrown,
    BetaCrown,
    SmtRefiner,
    LazySmtRefiner,
    Mip,
    MipHiGHS,
    MipVnnlib,
    IbpF64,
    CrownF64,
    Other(String),
}

impl MethodUsed {
    /// Canonical string representation of this method tag.
    pub fn as_str(&self) -> &str {
        match self {
            MethodUsed::Ibp => "Ibp",
            MethodUsed::Crown => "Crown",
            MethodUsed::AlphaCrown => "AlphaCrown",
            MethodUsed::SdpCrown => "SdpCrown",
            MethodUsed::BetaCrown => "BetaCrown",
            MethodUsed::SmtRefiner => "SmtRefiner",
            MethodUsed::LazySmtRefiner => "LazySmtRefiner",
            MethodUsed::Mip => "Mip",
            MethodUsed::MipHiGHS => "MipHiGHS",
            MethodUsed::MipVnnlib => "MipVnnlib",
            MethodUsed::IbpF64 => "Ibp_f64",
            MethodUsed::CrownF64 => "Crown_f64",
            MethodUsed::Other(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for MethodUsed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Deref for MethodUsed {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl From<&str> for MethodUsed {
    fn from(s: &str) -> Self {
        match s {
            "Ibp" => MethodUsed::Ibp,
            "Crown" => MethodUsed::Crown,
            "AlphaCrown" => MethodUsed::AlphaCrown,
            "SdpCrown" => MethodUsed::SdpCrown,
            "BetaCrown" => MethodUsed::BetaCrown,
            "SmtRefiner" => MethodUsed::SmtRefiner,
            "LazySmtRefiner" => MethodUsed::LazySmtRefiner,
            "Mip" => MethodUsed::Mip,
            "MipHiGHS" => MethodUsed::MipHiGHS,
            "MipVnnlib" => MethodUsed::MipVnnlib,
            "Ibp_f64" => MethodUsed::IbpF64,
            "Crown_f64" => MethodUsed::CrownF64,
            other => MethodUsed::Other(other.to_string()),
        }
    }
}

impl From<String> for MethodUsed {
    fn from(s: String) -> Self {
        // Try known variants first before allocating for Other
        match s.as_str() {
            "Ibp" | "Crown" | "AlphaCrown" | "SdpCrown" | "BetaCrown" | "SmtRefiner"
            | "LazySmtRefiner" | "Mip" | "MipHiGHS" | "MipVnnlib" | "Ibp_f64" | "Crown_f64" => {
                MethodUsed::from(s.as_str())
            }
            _ => MethodUsed::Other(s),
        }
    }
}

impl Serialize for MethodUsed {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for MethodUsed {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(MethodUsed::from(s))
    }
}

/// Reason for Unknown verification result.
///
/// Provides structured information about why verification was inconclusive.
/// Each variant captures a different category of incompleteness.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UnknownReason {
    /// Bounds too loose to verify (bound propagation incomplete).
    BoundsTooLoose {
        /// Gap between computed and required bounds (if known).
        #[serde(skip_serializing_if = "Option::is_none")]
        gap: Option<f32>,
    },

    /// SMT solver returned Unknown (theory limitations).
    SmtUnknown {
        /// Solver-provided reason string (if any).
        #[serde(skip_serializing_if = "Option::is_none")]
        solver_reason: Option<String>,
    },

    /// Resource limit hit (memory, iterations, etc.).
    ResourceLimit {
        /// Resource type that was exhausted.
        resource: String,
        /// Configured limit.
        limit: u64,
        /// Amount actually used.
        used: u64,
    },

    /// Unsupported operation encountered during verification.
    UnsupportedOp {
        /// Name of the unsupported operation.
        op_name: String,
    },

    /// SAT result downgraded by trust policy.
    SatTrustPolicy {
        /// Policy that triggered downgrade.
        policy: String,
    },

    /// Potential violation found but not confirmed.
    PotentialViolation,

    /// Other/unspecified reason.
    Other {
        /// Human-readable explanation.
        message: String,
    },
}

impl std::fmt::Display for UnknownReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnknownReason::BoundsTooLoose { gap } => {
                if let Some(g) = gap {
                    write!(f, "Bounds too loose (gap: {g:.4})")
                } else {
                    write!(f, "Bounds too loose")
                }
            }
            UnknownReason::SmtUnknown { solver_reason } => {
                if let Some(reason) = solver_reason {
                    write!(f, "SMT solver returned unknown: {reason}")
                } else {
                    write!(f, "SMT solver returned unknown")
                }
            }
            UnknownReason::ResourceLimit {
                resource,
                limit,
                used,
            } => {
                write!(
                    f,
                    "Resource limit exceeded: {resource} ({used} used, {limit} limit)"
                )
            }
            UnknownReason::UnsupportedOp { op_name } => {
                write!(f, "Unsupported operation: {op_name}")
            }
            UnknownReason::SatTrustPolicy { policy } => {
                write!(f, "SAT downgraded by trust policy: {policy}")
            }
            UnknownReason::PotentialViolation => {
                write!(f, "Potential violation region found")
            }
            UnknownReason::Other { message } => write!(f, "{message}"),
        }
    }
}

impl From<String> for UnknownReason {
    fn from(s: String) -> Self {
        // Parse known patterns for backward compatibility
        if s.contains("too loose") {
            UnknownReason::BoundsTooLoose { gap: None }
        } else if s.contains("SMT") || s.contains("solver") {
            UnknownReason::SmtUnknown {
                solver_reason: Some(s),
            }
        } else if s.contains("trust policy") || s.contains("downgraded") {
            UnknownReason::SatTrustPolicy { policy: s }
        } else if s.contains("potential violation") {
            UnknownReason::PotentialViolation
        } else {
            UnknownReason::Other { message: s }
        }
    }
}

impl From<&str> for UnknownReason {
    fn from(s: &str) -> Self {
        UnknownReason::from(s.to_string())
    }
}

/// Result of a verification query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationResult {
    /// Property verified: all outputs within bounds for all inputs in region.
    Verified {
        /// Soundness provenance tracking heuristics used.
        #[serde(default)]
        provenance: SoundnessProvenance,
        /// Certified output bounds.
        output_bounds: Vec<Bound>,
        /// Optional UNSAT proof certificate.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        proof: Option<Box<VerificationProof>>,
        /// Verifier-level propagation method label recorded on the result.
        ///
        /// This may differ from the requested method when the verifier itself
        /// falls back (for example, `Crown` -> `Ibp`). Internal layer/node
        /// fallback decisions inside a method are not guaranteed to change this
        /// string; use provenance or method-specific diagnostics for finer detail.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        actual_method: Option<MethodUsed>,
    },
    /// Property violated: counterexample found.
    Violated {
        /// Soundness provenance tracking heuristics used.
        #[serde(default)]
        provenance: SoundnessProvenance,
        /// Concrete counterexample input.
        counterexample: Vec<f32>,
        /// Output at counterexample.
        output: Vec<f32>,
        /// Detailed counterexample information (if available).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<Box<InformativeCounterexample>>,
        /// Verifier-level propagation method label recorded on the result.
        ///
        /// This may differ from the requested method when the verifier itself
        /// falls back (for example, `Crown` -> `Ibp`). Internal layer/node
        /// fallback decisions inside a method are not guaranteed to change this
        /// string; use provenance or method-specific diagnostics for finer detail.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        actual_method: Option<MethodUsed>,
    },
    /// Verification inconclusive: bounds too loose.
    Unknown {
        /// Soundness provenance tracking heuristics used.
        #[serde(default)]
        provenance: SoundnessProvenance,
        /// Best bounds achieved.
        bounds: Vec<Bound>,
        /// Reason verification couldn't complete.
        reason: UnknownReason,
        /// Verifier-level propagation method label recorded on the result.
        ///
        /// This may differ from the requested method when the verifier itself
        /// falls back (for example, `Crown` -> `Ibp`). Internal layer/node
        /// fallback decisions inside a method are not guaranteed to change this
        /// string; use provenance or method-specific diagnostics for finer detail.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        actual_method: Option<MethodUsed>,
    },
    /// Verification timed out.
    Timeout {
        /// Soundness provenance tracking heuristics used.
        #[serde(default)]
        provenance: SoundnessProvenance,
        /// Partial bounds at timeout.
        partial_bounds: Option<Vec<Bound>>,
        /// Verifier-level propagation method label recorded on the result.
        ///
        /// This may differ from the requested method when the verifier itself
        /// falls back (for example, `Crown` -> `Ibp`). Internal layer/node
        /// fallback decisions inside a method are not guaranteed to change this
        /// string; use provenance or method-specific diagnostics for finer detail.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        actual_method: Option<MethodUsed>,
    },
}

impl VerificationResult {
    /// Check if this result is verified.
    ///
    /// # REQUIRES
    /// - None
    ///
    /// # ENSURES
    /// - Returns `true` iff `self` is `VerificationResult::Verified`
    pub fn is_verified(&self) -> bool {
        matches!(self, VerificationResult::Verified { .. })
    }

    /// Set the verifier-level propagation method label recorded on the result.
    ///
    /// Use this when the verifier's externally reported method differs from
    /// what was requested (for example, a top-level `Crown` -> `Ibp`
    /// fallback). This is a coarse verifier-level label, not a full trace of
    /// internal fallback decisions taken inside the chosen method.
    ///
    /// # NOTES
    /// - `method` should describe the verifier-level method label to record
    /// - Prefer non-empty identifiers when available
    ///
    /// # REQUIRES
    /// - None
    ///
    /// # ENSURES
    /// - Returns the same variant as `self`
    /// - `actual_method()` returns `Some(method.into())` on the result
    #[must_use]
    pub fn with_actual_method(mut self, method: impl Into<MethodUsed>) -> Self {
        let tag = Some(method.into());
        match &mut self {
            VerificationResult::Verified { actual_method, .. } => *actual_method = tag,
            VerificationResult::Violated { actual_method, .. } => *actual_method = tag,
            VerificationResult::Unknown { actual_method, .. } => *actual_method = tag,
            VerificationResult::Timeout { actual_method, .. } => *actual_method = tag,
        }
        self
    }

    /// Get the verifier-level propagation method label as a string, if
    /// recorded.
    ///
    /// This is the backward-compatible string accessor. For typed matching
    /// in Rust, use [`actual_method_tag`](Self::actual_method_tag).
    pub fn actual_method(&self) -> Option<&str> {
        match self {
            VerificationResult::Verified { actual_method, .. } => actual_method.as_deref(),
            VerificationResult::Violated { actual_method, .. } => actual_method.as_deref(),
            VerificationResult::Unknown { actual_method, .. } => actual_method.as_deref(),
            VerificationResult::Timeout { actual_method, .. } => actual_method.as_deref(),
        }
    }

    /// Get the typed method tag, if recorded.
    pub fn actual_method_tag(&self) -> Option<&MethodUsed> {
        match self {
            VerificationResult::Verified { actual_method, .. } => actual_method.as_ref(),
            VerificationResult::Violated { actual_method, .. } => actual_method.as_ref(),
            VerificationResult::Unknown { actual_method, .. } => actual_method.as_ref(),
            VerificationResult::Timeout { actual_method, .. } => actual_method.as_ref(),
        }
    }

    /// Set the soundness provenance for this result.
    ///
    /// # REQUIRES
    /// - None
    ///
    /// # ENSURES
    /// - Returns the same variant as `self`
    /// - `provenance()` returns the provided `prov` on the result
    #[must_use]
    pub fn with_provenance(mut self, prov: SoundnessProvenance) -> Self {
        match &mut self {
            VerificationResult::Verified { provenance, .. } => *provenance = prov,
            VerificationResult::Violated { provenance, .. } => *provenance = prov,
            VerificationResult::Unknown { provenance, .. } => *provenance = prov,
            VerificationResult::Timeout { provenance, .. } => *provenance = prov,
        }
        self
    }

    /// Get the soundness provenance for this result.
    ///
    /// # REQUIRES
    /// - None
    ///
    /// # ENSURES
    /// - Returns a reference to the provenance stored in `self`
    pub fn provenance(&self) -> &SoundnessProvenance {
        match self {
            VerificationResult::Verified { provenance, .. } => provenance,
            VerificationResult::Violated { provenance, .. } => provenance,
            VerificationResult::Unknown { provenance, .. } => provenance,
            VerificationResult::Timeout { provenance, .. } => provenance,
        }
    }
}
