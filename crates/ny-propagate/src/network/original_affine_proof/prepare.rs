// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::certify::scope::{NumericalSemantics, SCOPE_SCHEMA_VERSION};
use crate::certify::work::PersistentReservation;
use crate::layers::linear::original_affine::{
    self as kernel, OriginalAffineArrays, OriginalAffineView,
};
use crate::layers::PointLinearLayer;

use super::*;

pub(super) struct PreparedOwner {
    pub owner: PointLinearLayer,
    pub scope: ScopeIdentity,
    pub deadline: Option<Instant>,
    pub reservation: PersistentReservation,
}

/// One validated view of the immutable original owner, without materialized
/// signed/absolute/transposed/spectral caches. Validation is paid once; each
/// kernel query receives its own phase lease from the SAME original ledger.
pub struct PreparedOriginalAffine<'a> {
    pub(super) view: OriginalAffineArrays<'a>,
    pub(super) shared: Arc<PreparedOwner>,
}

impl<'a> PreparedOriginalAffine<'a> {
    pub fn preparation_request(owner: &PointLinearLayer) -> Result<WorkRequest> {
        extend_request(
            OriginalAffineView::validation_request(
                owner.out_features(),
                owner.in_features(),
                owner.bias().is_some(),
            )?,
            32,
        )
    }

    pub fn prepare(
        owner: &'a PointLinearLayer,
        ids: OriginalAffineIds,
        options: OriginalAffineOptions<'_>,
    ) -> Result<Self> {
        require_same_ledger(owner, options)?;
        if add(size_of::<PreparedOwner>(), size_of::<Self>())? > METADATA_BYTES {
            return Err(invalid("prepared owner metadata capacity"));
        }
        let request = Self::preparation_request(owner)?;
        let deadline = earlier(
            earlier(owner.original_deadline(), options.deadline),
            ny_core::instance_budget::deadline(),
        );
        let mut phase =
            OriginalAffinePhase::admit(options.budget, options.class, request, deadline)?;
        let view = OriginalAffineArrays::validate(owner.weight(), owner.bias(), &mut phase)?;
        phase.charge_work(32)?;
        let mut coordinates = Sha256::new();
        coordinates.update(b"ny-original-affine-local-coordinates/v1");
        for id in [ids.node, ids.input, ids.output] {
            coordinates.update(id.0);
        }
        coordinates.update(as_work(owner.in_features())?.to_le_bytes());
        coordinates.update(as_work(owner.out_features())?.to_le_bytes());
        let scope = ScopeIdentity {
            schema_version: SCOPE_SCHEMA_VERSION,
            // Derived from actual original coefficient bits/dimensions/bias,
            // never a caller-claimed model hash or diagnostic point receipt.
            model: owner.original_digest(),
            lowering: ids.lowering,
            coordinates: Digest(coordinates.finalize().into()),
            assumptions: Digest::of_bytes(
                b"original-affine-local-profile",
                ORIGINAL_AFFINE_EXACT_REAL_PROFILE.as_bytes(),
            ),
            numerical: NumericalSemantics::ExactReal,
        };
        phase.retain_metadata(METADATA_BYTES)?;
        let reservation = phase.commit()?;
        let shared = Arc::new(PreparedOwner {
            owner: owner.clone(),
            scope,
            deadline: earlier(deadline, ny_core::instance_budget::deadline()),
            reservation,
        });
        Ok(Self { view, shared })
    }

    pub fn scope(&self) -> &ScopeIdentity {
        &self.shared.scope
    }
    pub fn original_digest(&self) -> Digest {
        self.shared.owner.original_digest()
    }
    pub fn preparation_reserved_bytes(&self) -> usize {
        self.shared.reservation.bytes()
    }
    pub fn in_features(&self) -> usize {
        self.view.in_features()
    }
    pub fn out_features(&self) -> usize {
        self.view.out_features()
    }

    fn phase(
        &self,
        request: WorkRequest,
        options: OriginalAffineOptions<'_>,
    ) -> Result<OriginalAffinePhase> {
        require_same_ledger(&self.shared.owner, options)?;
        let deadline = earlier(self.shared.deadline, options.deadline);
        OriginalAffinePhase::admit(options.budget, options.class, request, deadline)
    }

    pub fn interval_request(&self, input: &BoundedTensor) -> Result<WorkRequest> {
        let request = kernel::ibp_request(self.out_features(), self.in_features(), input.shape())?;
        let output = mul(
            input.lower().len() / self.in_features(),
            self.out_features(),
        )?;
        let hash_output = add(mul(output, 2)?, mul(add(input.shape().len(), 1)?, 2)?)?;
        extend_request(request, add(interval_terms(input)?, hash_output)?)
    }

    /// Enclose this original W*x+b on the supplied finite binary32 endpoint
    /// box, interpreted over REAL inputs. No upstream box authority is minted.
    pub fn interval(
        &self,
        input: &BoundedTensor,
        options: OriginalAffineOptions<'_>,
    ) -> Result<OriginalAffineIntervalReport> {
        let mut phase = self.phase(self.interval_request(input)?, options)?;
        let source = hash_interval(input, &mut phase)?;
        let output = kernel::ibp(self.view.view(), input, &mut phase)?;
        let output_digest = hash_interval(&output, &mut phase)?;
        let binding = OriginalAffineBinding {
            kernel: OriginalAffineKernel::Interval,
            query: query_digest(self.scope(), OriginalAffineKernel::Interval, source),
            output: output_digest,
        };
        OriginalAffineIntervalReport::finish(output, self.shared.clone(), binding, phase)
    }

    pub fn signed_linear_request(&self, incoming: &LinearBounds) -> Result<WorkRequest> {
        let (rows, columns) = linear_shape(incoming)?;
        let request =
            kernel::crown_request(self.out_features(), self.in_features(), rows, columns)?;
        let target_columns = mul(columns / self.out_features(), self.in_features())?;
        extend_request(
            request,
            add(
                linear_terms(incoming)?,
                output_linear_terms(rows, target_columns)?,
            )?,
        )
    }

    /// Compose every requested signed row and its COMPLETE coefficient error
    /// planes with this original affine map. This is not a whole graph proof.
    pub fn signed_linear(
        &self,
        incoming: &LinearBounds,
        options: OriginalAffineOptions<'_>,
    ) -> Result<OriginalAffineLinearReport> {
        let mut phase = self.phase(self.signed_linear_request(incoming)?, options)?;
        let source = hash_linear(incoming, &mut phase)?;
        let output = kernel::crown(self.view.view(), incoming, &mut phase)?;
        let output_digest = hash_linear(&output, &mut phase)?;
        let binding = OriginalAffineBinding {
            kernel: OriginalAffineKernel::SignedLinear,
            query: query_digest(self.scope(), OriginalAffineKernel::SignedLinear, source),
            output: output_digest,
        };
        OriginalAffineLinearReport::finish(output, self.shared.clone(), binding, phase)
    }

    pub fn sparse_margin_request(
        &self,
        positions: usize,
        differences: &[OriginalLogitDifference],
    ) -> Result<WorkRequest> {
        let mut request = kernel::sparse_request(
            self.out_features(),
            self.in_features(),
            differences.len(),
            positions,
        )?;
        // The temporary exact ID translation is charged before allocation.
        request.scratch_bytes = add(
            request.scratch_bytes,
            mul(differences.len(), size_of::<kernel::LogitDifference>())?,
        )?;
        extend_request(
            request,
            add(
                add(mul(differences.len(), 6)?, 2)?,
                output_linear_terms(differences.len(), mul(positions, self.in_features())?)?,
            )?,
        )
    }

    /// Seed explicitly demanded margins directly over the original head INPUT.
    /// No rows-by-vocabulary selector matrix is allocated. Every ID/position
    /// and its order are bound to the report; absent competitors remain absent,
    /// so this API cannot certify a complete vocabulary claim on its own.
    pub fn sparse_margins(
        &self,
        positions: usize,
        differences: &[OriginalLogitDifference],
        options: OriginalAffineOptions<'_>,
    ) -> Result<OriginalAffineLinearReport> {
        let request = self.sparse_margin_request(positions, differences)?;
        let mut phase = self.phase(request, options)?;
        let mut source = Sha256::new();
        source.update(b"ny-original-affine-demanded-logit-differences/v1");
        hash_word(&mut source, positions, &mut phase)?;
        hash_word(&mut source, differences.len(), &mut phase)?;
        let mut translated = Vec::new();
        translated
            .try_reserve_exact(differences.len())
            .map_err(|_| invalid("sparse ID allocation"))?;
        if mul(translated.capacity(), size_of::<kernel::LogitDifference>())?
            > mul(differences.len(), size_of::<kernel::LogitDifference>())?
        {
            return Err(invalid("sparse ID capacity exceeded preflight"));
        }
        for (index, difference) in differences.iter().enumerate() {
            if index.is_multiple_of(HASH_POLL) {
                phase.checkpoint()?;
            }
            for id in [
                difference.target,
                difference.competitor,
                difference.position,
            ] {
                hash_word(&mut source, id, &mut phase)?;
            }
            phase.charge_work(3)?;
            translated.push(kernel::LogitDifference {
                target: difference.target,
                competitor: difference.competitor,
                position: difference.position,
            });
        }
        let source = Digest(source.finalize().into());
        let output =
            kernel::seed_logit_differences(self.view.view(), positions, &translated, &mut phase)?;
        drop(translated);
        let output_digest = hash_linear(&output, &mut phase)?;
        let binding = OriginalAffineBinding {
            kernel: OriginalAffineKernel::SparseLogitDifferences,
            query: query_digest(
                self.scope(),
                OriginalAffineKernel::SparseLogitDifferences,
                source,
            ),
            output: output_digest,
        };
        OriginalAffineLinearReport::finish(output, self.shared.clone(), binding, phase)
    }
}

fn require_same_ledger(owner: &PointLinearLayer, options: OriginalAffineOptions<'_>) -> Result<()> {
    if !owner.original_budget().shares_ledger(options.budget) {
        return Err(NyError::UnsupportedConfiguration(
            "original-affine context caller ledger mismatch".into(),
        ));
    }
    Ok(())
}
