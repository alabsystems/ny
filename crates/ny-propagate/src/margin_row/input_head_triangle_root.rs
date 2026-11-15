// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Transactional root-only geometry admission. No continuation row tags.
use super::*;

impl PhasePrefix<'_> {
    pub(in super::super) fn root_work_hint(&self) -> Option<usize> {
        self.rows.iter().try_fold(self.boxes.len(), |sum, row| {
            sum.checked_add(row.coefficients.len())
        })
    }
}

impl<'a> Model<'a> {
    /// Re-export an already admitted complete root prefix without deleting rows.
    /// Used only to recover the last model after a later append refuses.
    pub(in super::super) fn from_completed_root(
        source: &'a Source,
        prefix: &mut Option<PhasePrefix<'a>>,
        deadline: Instant,
    ) -> Option<Self> {
        let frozen = prefix.as_ref()?;
        let count = frozen.reserved_rows(source, FUNCTIONAL_MAX_ROWS, deadline)?;
        if count != frozen.rows.len() {
            return None;
        }
        let request = root_request(source, frozen, &frozen.rows, deadline)?;
        let frozen = prefix.take()?;
        Some(Self {
            source,
            boxes: frozen.boxes,
            rows: frozen.rows,
            scalar_rows: frozen.scalar_rows,
            row_tags: vec![],
            request,
            cuts: 0,
            #[cfg(test)]
            cone_cuts: 0,
            negative_groups: 0,
            negative_group_masks: 0,
            large_negative_groups: 0,
            large_negative_group_masks: 0,
        })
    }

    /// Plan capacity without changing the completed prefix. Older optional
    /// rows yield space only if a first new request passes every admission check.
    pub(in super::super) fn prepare_root_prefix(
        source: &'a Source,
        prefix: &mut Option<PhasePrefix<'a>>,
        slots: usize,
        deadline: Instant,
    ) -> Option<usize> {
        if prefix.is_none() {
            *prefix = Some(Self::build_phase_base(source, deadline)?.into_phase_prefix());
        }
        let frozen = prefix.as_ref()?;
        let available = FUNCTIONAL_MAX_ROWS.checked_sub(frozen.scalar_rows)?;
        let reserved = slots.min(available);
        frozen.reserved_rows(source, FUNCTIONAL_MAX_ROWS, deadline)?;
        (live(deadline) && fp_qualified()).then_some(reserved)
    }

    pub(in super::super) fn from_root_portfolio(
        source: &'a Source,
        prefix: &mut Option<PhasePrefix<'a>>,
        directions: &[directional::CertifiedDirection<'a>],
        deadline: Instant,
    ) -> Option<Self> {
        Self::append_root_rows(source, prefix, directions, deadline, 0)
    }

    pub(in super::super) fn from_root_portfolio_reserved(
        source: &'a Source,
        prefix: &mut Option<PhasePrefix<'a>>,
        directions: &[directional::CertifiedDirection<'a>],
        reserved_slots: usize,
        deadline: Instant,
    ) -> Option<Self> {
        Self::append_root_rows(source, prefix, directions, deadline, reserved_slots)
    }

    pub(super) fn append_root_rows(
        source: &'a Source,
        prefix: &mut Option<PhasePrefix<'a>>,
        directions: &[directional::CertifiedDirection<'a>],
        deadline: Instant,
        reserved_slots: usize,
    ) -> Option<Self> {
        if directions.is_empty() || directions.len() > directional::MAX_CUTS {
            return None;
        }
        if prefix.is_none() {
            *prefix = Some(Self::build_phase_base(source, deadline)?.into_phase_prefix());
        }
        let frozen = prefix.as_ref()?;
        let maximum = FUNCTIONAL_MAX_ROWS.checked_sub(directions.len().max(reserved_slots))?;
        if reserved_slots == 0 && frozen.rows.len() > maximum {
            return None;
        }
        let count = frozen.reserved_rows(source, maximum, deadline)?;
        let mut rows = frozen.rows[..count].to_vec();
        for direction in directions {
            rows.push(direction.owned_row(source, deadline)?);
        }
        let request = root_request(source, frozen, &rows, deadline)?;
        // No take/truncate of a completed prefix until every admission check
        // succeeds. A later refused round retains every old directional row.
        let frozen = prefix.take()?;
        Some(Self {
            source,
            boxes: frozen.boxes,
            rows,
            scalar_rows: frozen.scalar_rows,
            row_tags: vec![],
            request,
            cuts: 0,
            #[cfg(test)]
            cone_cuts: 0,
            negative_groups: 0,
            negative_group_masks: 0,
            large_negative_groups: 0,
            large_negative_group_masks: 0,
        })
    }

    pub(in super::super) fn scalar_count(&self) -> usize {
        self.scalar_rows
    }
}

fn root_request(
    source: &Source,
    frozen: &PhasePrefix<'_>,
    rows: &[OwnedRow],
    deadline: Instant,
) -> Option<HeadFunctionalLpRequest> {
    let mut role_rows = Vec::with_capacity(source.spec.heads);
    let mut at = 0usize;
    for y in &source.y {
        role_rows.push(Some(at));
        at = at.checked_add(1 + usize::from(y.hi > 0.0))?;
    }
    if at != frozen.scalar_rows {
        return None;
    }
    let mut objective = vec![0.0; source.spec.inputs];
    objective.extend(
        source
            .q
            .iter()
            .copied()
            .map(midpoint)
            .collect::<Option<Vec<_>>>()?,
    );
    let mut exported = Vec::with_capacity(rows.len());
    for row in rows {
        if !live(deadline) || !fp_qualified() || !row.valid(frozen.boxes.len()) {
            return None;
        }
        exported.push(HeadFunctionalLpRow {
            coefficients: row
                .coefficients
                .iter()
                .map(|&(j, a)| midpoint(a).map(|point| (j, point)))
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .filter(|&(_, a)| a != 0.0)
                .collect(),
            lower: row.lower.lo,
        });
    }
    let request = HeadFunctionalLpRequest {
        protocol_version: FUNCTIONAL_PROTOCOL_VERSION,
        feature_count: source.spec.inputs,
        bounds: frozen.boxes.iter().map(|v| [v.lo, v.hi]).collect(),
        objective,
        rows: exported,
        slope_rows: role_rows,
        phase_rows: vec![],
    };
    (request.validate() && live(deadline) && fp_qualified()).then_some(request)
}
