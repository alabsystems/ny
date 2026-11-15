// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Existing shared-ledger admission, including one protected signed attempt.
use super::*;

pub(super) fn first_signed<'a>(
    source: &'a Source,
    state: &mut Portfolio<'a>,
    round: &mut RoundReport,
) -> Option<CertifiedDirection<'a>> {
    if state.signed_attempted || state.report.attempted >= MAX_ATTEMPTS {
        return None;
    }
    state.signed_attempted = true;
    let (cells, masks) = (state.budget.cells, state.budget.masks);
    let Some((negative, positive)) = signed_search::anchor(source, &mut state.budget) else {
        state.signed_report = Some(signed::AttemptReport {
            cells: state.budget.cells - cells,
            ..signed::AttemptReport::default()
        });
        return None;
    };
    let (positive_heads, negative_heads) = (positive.0.len(), negative.heads.len());
    round.candidates += 1;
    round.attempted += 1;
    state.report.attempted += 1;
    // A completed anchor gets admission/native replay before optional local
    // search. This spends one existing round, never a new callback or deadline.
    let cut = signed::certify(source, negative, positive, vec![], &mut state.budget);
    state.signed_report = Some(signed::AttemptReport {
        attempted: true,
        positive_heads,
        negative_heads,
        cells: state.budget.cells - cells,
        masks: state.budget.masks - masks,
        lower: cut.as_ref().map(|cut| cut.lower),
        admitted: false,
    });
    cut
}

pub(super) fn admit<'a>(
    source: &'a Source,
    state: &mut Portfolio<'a>,
    cuts: Vec<CertifiedDirection<'a>>,
) -> Option<triangle::Model<'a>> {
    admit_inner(source, state, cuts, None)
}

pub(super) fn admit_allocated<'a>(
    source: &'a Source,
    state: &mut Portfolio<'a>,
    cuts: Vec<CertifiedDirection<'a>>,
) -> Option<triangle::Model<'a>> {
    let publication = state.publication;
    admit_inner(source, state, cuts, Some(publication))
}

fn admit_inner<'a>(
    source: &'a Source,
    state: &mut Portfolio<'a>,
    mut cuts: Vec<CertifiedDirection<'a>>,
    publication: Option<PublicationAuthority>,
) -> Option<triangle::Model<'a>> {
    // Charge weighted export/replay work BEFORE cloning rows, including all
    // positive terms. This is a charged-work ledger, not an instruction count.
    if cuts.is_empty() {
        return None;
    }
    let old = state.retained.as_ref()?.root_work_hint()?;
    let added = cuts.iter().try_fold(0usize, |n, cut| {
        n.checked_add(cut.work_terms(source.spec.inputs)?)
    })?;
    let cost = old.checked_add(added)?.checked_mul(8)?;
    let deadline = if let Some(authority) = publication {
        state.budget.publication_charge(cost, authority)?;
        authority.deadline
    } else {
        state.budget.charge(cost)?;
        state.budget.deadline
    };
    while !cuts.is_empty() {
        let reservation = if state.accepted.is_empty() {
            state.slots
        } else {
            0
        };
        // Select the existing constructor whose reservation contract exactly
        // matches this batch. A partial first batch must retain its larger
        // original reservation; later batches reserve no additional slots.
        let model = if reservation == 0 {
            triangle::Model::from_root_portfolio(source, &mut state.retained, &cuts, deadline)
        } else if reservation == cuts.len() {
            triangle::Model::from_root_directions(source, &mut state.retained, &cuts, deadline)
        } else {
            triangle::Model::from_root_portfolio_reserved(
                source,
                &mut state.retained,
                &cuts,
                reservation,
                deadline,
            )
        };
        if let Some(model) = model {
            if cuts
                .iter()
                .any(|cut| matches!(&cut.geometry, GeometryKind::Signed(_)))
            {
                if let Some(receipt) = &mut state.signed_report {
                    receipt.admitted = true;
                }
            }
            state.report.affine_cuts += cuts.iter().filter(|cut| !cut.eta.is_empty()).count();
            state.accepted.extend(cuts);
            return Some(model);
        }
        cuts.pop(); // Smaller valid batches preserve the older complete prefix.
    }
    None
}
