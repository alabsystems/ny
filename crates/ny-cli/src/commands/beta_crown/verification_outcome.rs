// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CLI publication ownership. A hull replay never becomes an unowned status.

use anyhow::Result;
use ny_propagate::BetaCrownResult;

pub(super) struct VerificationOutcome {
    report: BetaCrownResult,
    #[cfg(feature = "mip")]
    hull: Option<std::sync::Arc<super::hull_publication::HullPublicationProof>>,
}

impl VerificationOutcome {
    pub(super) fn ordinary(report: BetaCrownResult) -> Self {
        Self {
            report,
            #[cfg(feature = "mip")]
            hull: None,
        }
    }

    #[cfg(feature = "mip")]
    pub(super) fn hull(
        report: BetaCrownResult,
        proof: std::sync::Arc<super::hull_publication::HullPublicationProof>,
    ) -> Self {
        Self {
            report,
            hull: Some(proof),
        }
    }

    /// Compatibility callers cannot silently discard a verified hull proof.
    pub(super) fn into_ordinary(self) -> Result<BetaCrownResult> {
        #[cfg(feature = "mip")]
        anyhow::ensure!(
            self.hull.is_none(),
            "retained hull outcome requires its publisher"
        );
        Ok(self.report)
    }

    #[cfg(feature = "mip")]
    pub(super) fn into_parts(
        self,
    ) -> (
        BetaCrownResult,
        Option<std::sync::Arc<super::hull_publication::HullPublicationProof>>,
    ) {
        (self.report, self.hull)
    }
}
