// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bound certificate types for multi-network composition.

use ny_core::{MethodUsed, NyError, Result, SoundnessProvenance};
use ny_tensor::BoundedTensor;

/// Provenance tag for how bounds were computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BoundProvenance {
    /// Interval Bound Propagation (fastest, loosest).
    Ibp,
    /// CROWN linear relaxation.
    Crown,
    /// α-CROWN with optimized parameters.
    AlphaCrown,
}

fn unsupported_method_error(method: &MethodUsed) -> NyError {
    NyError::UnsupportedOp(format!(
        "bound certificates do not support actual_method {method}; Packet A supports only \
         IBP, CROWN/SDP-CROWN, and AlphaCrown"
    ))
}

impl TryFrom<&MethodUsed> for BoundProvenance {
    type Error = NyError;

    fn try_from(method: &MethodUsed) -> Result<Self> {
        match method {
            MethodUsed::Ibp | MethodUsed::IbpF64 => Ok(Self::Ibp),
            // SdpCrown in this arm is dormant: every dispatch site refuses SDP-CROWN
            // over ℓ∞ box specs before a certificate is built, so no SdpCrown method
            // tag reaches this mapping today. If an SDP-CROWN execution path is
            // reintroduced (e.g. genuine ℓ2-ball specs), re-review this Crown
            // provenance mapping before it labels those bounds.
            MethodUsed::Crown | MethodUsed::CrownF64 | MethodUsed::SdpCrown => Ok(Self::Crown),
            MethodUsed::AlphaCrown => Ok(Self::AlphaCrown),
            MethodUsed::BetaCrown => Err(unsupported_method_error(method)),
            MethodUsed::SmtRefiner
            | MethodUsed::LazySmtRefiner
            | MethodUsed::Mip
            | MethodUsed::MipHiGHS
            | MethodUsed::MipVnnlib
            | MethodUsed::Other(_) => Err(unsupported_method_error(method)),
            _ => Err(unsupported_method_error(method)),
        }
    }
}

impl TryFrom<MethodUsed> for BoundProvenance {
    type Error = NyError;

    fn try_from(method: MethodUsed) -> Result<Self> {
        Self::try_from(&method)
    }
}

/// Result of a bound-only certification run.
#[derive(Debug, Clone)]
pub enum BoundCertificationResult {
    /// Propagation completed and produced a full certificate.
    Certified(BoundCertificate),
    /// Propagation hit the configured deadline without collapsing into an error.
    ///
    /// Timeout metadata (`actual_method`, `soundness`) is always present so
    /// callers can inspect the verifier's state even when no partial bounds
    /// are available.
    Timeout {
        /// Partial certificate, if the propagation path can surface one.
        partial: Option<BoundCertificate>,
        /// Exact verifier-level method that was running when the timeout hit.
        actual_method: MethodUsed,
        /// Soundness provenance at the time of timeout.
        soundness: SoundnessProvenance,
    },
}

/// A bound certificate from a single-model verification run.
///
/// Fields are private to enforce that all certificates are created through
/// [`BoundCertificate::try_new`], which validates the method/provenance
/// contract. Use the accessor methods to read certificate metadata.
#[derive(Debug, Clone)]
pub struct BoundCertificate {
    model_id: String,
    output_bounds: BoundedTensor,
    provenance: BoundProvenance,
    actual_method: MethodUsed,
    soundness: SoundnessProvenance,
}

impl BoundCertificate {
    /// Build a certificate from verifier output while preserving the exact
    /// method tag, coarse provenance summary, and soundness metadata.
    pub fn try_new(
        model_id: impl Into<String>,
        output_bounds: BoundedTensor,
        actual_method: MethodUsed,
        soundness: SoundnessProvenance,
    ) -> Result<Self> {
        let provenance = BoundProvenance::try_from(&actual_method)?;
        Ok(Self {
            model_id: model_id.into(),
            output_bounds,
            provenance,
            actual_method,
            soundness,
        })
    }

    /// Model identifier (e.g., "lead_voice", "backing_1").
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Output bounds: lower and upper per output dimension.
    pub fn output_bounds(&self) -> &BoundedTensor {
        &self.output_bounds
    }

    /// Coarse provenance summary derived from the verifier-level method tag.
    pub fn provenance(&self) -> BoundProvenance {
        self.provenance
    }

    /// Exact verifier-level method tag used to compute the bounds.
    pub fn actual_method(&self) -> &MethodUsed {
        &self.actual_method
    }

    /// Soundness provenance recording whether heuristics were used.
    pub fn soundness(&self) -> &SoundnessProvenance {
        &self.soundness
    }
}
