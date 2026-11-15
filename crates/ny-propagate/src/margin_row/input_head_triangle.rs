// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! A private outer model on original inputs and genuine authored ReLU heads.
//! Exported point rows guide AY; only the locally owned interval rows enter
//! weak-dual replay. No primal, status, objective or imported source has
//! certificate authority. The existing input source and one-class authorizer
//! remain responsible for original graph/property/FP/deadline qualification.

use super::*;
use ny_core::head_functional_advice::{
    HeadFunctionalLpHint, FUNCTIONAL_MAX_MESSAGE_BYTES, FUNCTIONAL_MAX_NONZEROS,
    FUNCTIONAL_MAX_ROWS, FUNCTIONAL_MAX_VARIABLES,
};

const MAX_CUTS: usize = 16;
const MAX_TOTAL_CUTS: usize = 32;
const MAX_POSITIVE_MASSES: usize = 32;
const MAX_NEIGHBORS: usize = 8;
const MAX_RANK_TRIALS: usize = 512;

#[derive(Clone)]
pub(super) struct OwnedRow {
    pub(super) coefficients: Vec<(usize, I)>,
    pub(super) lower: I,
}

impl OwnedRow {
    pub(super) fn valid(&self, variables: usize) -> bool {
        self.lower.valid()
            && self.coefficients.iter().enumerate().all(|(at, &(j, a))| {
                j < variables && a.valid() && (at == 0 || self.coefficients[at - 1].0 < j)
            })
    }
}

#[derive(Clone)]
pub(super) struct MassCut {
    positive: Vec<(usize, f64)>,
    negative: usize,
    tau: f64,
}

pub(super) fn mass_recipe_matches(cut: &MassCut, row: &[(usize, I)], inputs: usize) -> bool {
    let Some(mut expected) = cut
        .positive
        .iter()
        .filter(|(_, a)| *a != 0.0)
        .map(|&(j, a)| inputs.checked_add(j).map(|k| (k, a)))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    if cut.tau != 0.0 {
        let Some(index) = inputs.checked_add(cut.negative) else {
            return false;
        };
        expected.push((index, -cut.tau));
    }
    expected.sort_by_key(|&(j, _)| j);
    expected.len() == row.len()
        && expected
            .iter()
            .zip(row)
            .all(|(&(j, p), &(k, a))| j == k && a.lo == p && a.hi == p)
}

#[derive(Clone)]
enum RowTag {
    Plain,
    Mass(MassCut),
    Group(Vec<(usize, f64)>),
}

pub(super) struct Model<'a> {
    source: &'a Source,
    boxes: Vec<I>,
    rows: Vec<OwnedRow>,
    scalar_rows: usize,
    row_tags: Vec<RowTag>,
    request: HeadFunctionalLpRequest,
    cuts: usize,
    #[cfg(test)]
    cone_cuts: usize,
    negative_groups: usize,
    negative_group_masks: usize,
    large_negative_groups: usize,
    large_negative_group_masks: usize,
}

/// Only a completed same-Source model may supply this unconditional prefix.
/// No advice status or retained fresh-Source recipe participates in ownership.
pub(super) struct PhasePrefix<'a> {
    source: &'a Source,
    boxes: Vec<I>,
    rows: Vec<OwnedRow>,
    scalar_rows: usize,
}

impl PhasePrefix<'_> {
    fn reserved_rows(
        &self,
        source: &Source,
        maximum_rows: usize,
        deadline: Instant,
    ) -> Option<usize> {
        let variables = source.spec.inputs.checked_add(source.spec.heads)?;
        if !std::ptr::eq(self.source, source)
            || !Arc::ptr_eq(&self.source.spec, &source.spec)
            || !live(deadline)
            || !fp_qualified()
            || self.boxes.len() != variables
            || self.boxes.iter().any(|value| !value.valid())
            || self.scalar_rows > self.rows.len()
            || self.scalar_rows > maximum_rows
        {
            return None;
        }
        let count = self.rows.len().min(maximum_rows);
        for row in self.rows.iter().take(count) {
            if !live(deadline) || !row.valid(variables) {
                return None;
            }
        }
        (live(deadline) && fp_qualified()).then_some(count)
    }

    fn reserve(mut self, source: &Source, maximum_rows: usize, deadline: Instant) -> Option<Self> {
        let count = self.reserved_rows(source, maximum_rows, deadline)?;
        self.rows.truncate(count);
        Some(self)
    }

    pub(super) fn into_phase_parts(
        self,
        source: &Source,
        maximum_rows: usize,
        deadline: Instant,
    ) -> Option<(Vec<I>, Vec<OwnedRow>)> {
        // Scalar triangles are mandatory. Only the optional cut suffix may
        // give up space for the selected nominal sign/cap rows.
        let prefix = self.reserve(source, maximum_rows, deadline)?;
        Some((prefix.boxes, prefix.rows))
    }
}

fn midpoint(i: I) -> Option<f64> {
    let point = 0.5 * i.lo + 0.5 * i.hi;
    good(point).then_some(point)
}

impl<'a> Model<'a> {
    /// Root-only intersection of independently completed Source directions.
    /// No row tags exist in this model: retained/fresh-Source replay refuses it.
    pub(super) fn from_root_directions(
        source: &'a Source,
        prefix: &mut Option<PhasePrefix<'a>>,
        directions: &[directional::CertifiedDirection<'a>],
        deadline: Instant,
    ) -> Option<Self> {
        Self::append_root_rows(source, prefix, directions, deadline, directions.len())
    }

    pub(super) fn build(source: &'a Source, deadline: Instant) -> Option<Self> {
        Self::build_with_cuts(source, true, deadline)
    }

    /// The same Source-owned scalar triangle geometry, without optional cuts.
    /// Only the sibling complete-cover producer may consume these interval rows.
    pub(super) fn build_phase_base(source: &'a Source, deadline: Instant) -> Option<Self> {
        Self::build_with_large_groups(source, false, false, false, false, deadline)
    }

    pub(super) fn into_phase_parts(self) -> (Vec<I>, Vec<OwnedRow>) {
        (self.boxes, self.rows)
    }

    pub(super) fn into_phase_prefix(self) -> PhasePrefix<'a> {
        PhasePrefix {
            source: self.source,
            boxes: self.boxes,
            rows: self.rows,
            scalar_rows: self.scalar_rows,
        }
    }

    fn build_with_cuts(source: &'a Source, include_cuts: bool, deadline: Instant) -> Option<Self> {
        Self::build_with_options(source, include_cuts, include_cuts, deadline)
    }

    fn build_with_options(
        source: &'a Source,
        include_cuts: bool,
        include_cones: bool,
        deadline: Instant,
    ) -> Option<Self> {
        Self::build_with_stages(source, include_cuts, include_cones, include_cones, deadline)
    }

    fn build_with_stages(
        source: &'a Source,
        include_cuts: bool,
        include_cones: bool,
        include_groups: bool,
        deadline: Instant,
    ) -> Option<Self> {
        Self::build_with_large_groups(
            source,
            include_cuts,
            include_cones,
            include_groups,
            include_groups,
            deadline,
        )
    }

    fn build_with_large_groups(
        source: &'a Source,
        include_cuts: bool,
        include_cones: bool,
        include_groups: bool,
        include_large_groups: bool,
        deadline: Instant,
    ) -> Option<Self> {
        let d = source.spec.inputs;
        let h = source.spec.heads;
        let n = d.checked_add(h)?;
        if !live(deadline)
            || !fp_qualified()
            || n > FUNCTIONAL_MAX_VARIABLES
            || source.lo.len() != d
            || source.hi.len() != d
            || source.y.len() != h
            || source.first_error.len() != h
            || source.q.len() != h
            || source.spec.first.len() != d.checked_mul(h)?
            || source.spec.first_bias.len() != h
            || !source.bias.valid()
            || !good(source.output_error)
            || source.output_error < 0.0
            || source.q.iter().any(|q| !q.valid())
            || source.first_error.iter().any(|&e| !good(e) || e < 0.0)
        {
            return None;
        }
        let mut boxes: Vec<_> = source
            .lo
            .iter()
            .zip(&source.hi)
            .map(|(&lo, &hi)| I { lo, hi })
            .collect();
        boxes.extend(source.y.iter().map(|y| I {
            lo: y.lo.max(0.0),
            hi: y.hi.max(0.0),
        }));
        if boxes.len() != n || boxes.iter().any(|b| !b.valid()) {
            return None;
        }
        let mut rows = Vec::new();
        let mut role_rows = Vec::with_capacity(h);
        for j in 0..h {
            if !live(deadline) {
                return None;
            }
            let b = I::p(f64::from(source.spec.first_bias[j]));
            let e = I::p(source.first_error[j]);
            // Every real head, including negative-Q and inactive heads, has
            // a genuine lower epigraph row. No preactivation/dummy role exists.
            let mut lower = Vec::with_capacity(d + 1);
            for k in 0..d {
                if k & 63 == 0 && !live(deadline) {
                    return None;
                }
                let w = f64::from(source.spec.first[k * h + j]);
                if w != 0.0 {
                    lower.push((k, I::p(-w)));
                }
            }
            lower.push((d + j, I::p(1.0)));
            role_rows.push(Some(rows.len()));
            rows.push(OwnedRow {
                coefficients: lower,
                lower: b.sub(e)?,
            });
            let y = source.y[j];
            if !y.valid() {
                return None;
            }
            if y.hi <= 0.0 {
                // The owned ReLU box fixes this genuinely authored head to0.
                continue;
            }
            let mut upper = Vec::with_capacity(d + 1);
            if y.lo >= 0.0 {
                // y=Wx+b+epsilon and r=y, |epsilon|<=E1.
                for k in 0..d {
                    if k & 63 == 0 && !live(deadline) {
                        return None;
                    }
                    let w = f64::from(source.spec.first[k * h + j]);
                    if w != 0.0 {
                        upper.push((k, I::p(w)));
                    }
                }
                upper.push((d + j, I::p(-1.0)));
                rows.push(OwnedRow {
                    coefficients: upper,
                    lower: b.neg().sub(e)?,
                });
            } else {
                // Division-free projected triangle:
                // U*W*x-(U-L)*r >= U*(L-b-E1), L<0<U.
                // Products/difference are enclosed BEFORE later amplification.
                let u = I::p(y.hi);
                let l = I::p(y.lo);
                for k in 0..d {
                    if k & 63 == 0 && !live(deadline) {
                        return None;
                    }
                    let a = u.mul(I::p(f64::from(source.spec.first[k * h + j])))?;
                    if a.lo != 0.0 || a.hi != 0.0 {
                        upper.push((k, a));
                    }
                }
                upper.push((d + j, u.sub(l)?.neg()));
                rows.push(OwnedRow {
                    coefficients: upper,
                    lower: u.mul(l.sub(b)?.sub(e)?)?,
                });
            }
        }
        let scalar_rows = rows.len();
        let mut row_tags = vec![RowTag::Plain; scalar_rows];
        let mut cuts = 0;
        let mut cone_cuts = 0;
        let mut negative_groups = 0;
        let mut negative_group_masks = 0;
        let mut large_negative_groups = 0;
        let mut large_negative_group_masks = 0;
        if include_cuts {
            let legacy = proposals(source, deadline)?;
            for proposal in &legacy {
                if !live(deadline) {
                    return None;
                }
                if rows.len() >= FUNCTIONAL_MAX_ROWS || cuts >= MAX_CUTS {
                    break;
                }
                let lower = source_cut_lower(source, proposal, deadline)?;
                let mut coefficients: Vec<_> = proposal
                    .positive
                    .iter()
                    .filter(|(_, a)| *a != 0.0)
                    .map(|&(head, a)| (d + head, I::p(a)))
                    .collect();
                if proposal.tau != 0.0 {
                    coefficients.push((d + proposal.negative, I::p(-proposal.tau)));
                }
                coefficients.sort_by_key(|(j, _)| *j);
                rows.push(OwnedRow {
                    coefficients,
                    lower: I::p(lower),
                });
                row_tags.push(RowTag::Mass(proposal.clone()));
                cuts += 1;
            }
            // Preserve every admitted old row BEFORE spending optional cone
            // search work. A local fit expiry/invalid proposal cannot erase
            // these rows or restart the caller's shared advice allowance.
            if (include_cones || include_groups) && rows.len() < FUNCTIONAL_MAX_ROWS {
                let now = Instant::now();
                let fit_deadline = cone_fit::phase_deadline(now, deadline);
                if include_cones {
                    // With eligible group work, cone search/checking receives at
                    // most the first3/4 of the SAME optional window.
                    let cone_deadline = if include_groups && negative_groups::eligible(source) {
                        cone_fit::search_deadline(now, fit_deadline)
                    } else {
                        fit_deadline
                    };
                    let search_deadline = cone_fit::search_deadline(now, cone_deadline);
                    let (proposed, mut fit) = cone_fit::proposals(source, &legacy, search_deadline);
                    let mut nnz = rows
                        .iter()
                        .try_fold(0usize, |sum, row| sum.checked_add(row.coefficients.len()))?;
                    for proposal in proposed {
                        if cuts >= MAX_TOTAL_CUTS
                            || rows.len() >= FUNCTIONAL_MAX_ROWS
                            || !live(cone_deadline)
                        {
                            break;
                        }
                        let Some(row) = owned_cut_row(source, &proposal, cone_deadline) else {
                            continue;
                        };
                        fit.checked += 1;
                        let Some(next_nnz) = nnz.checked_add(row.coefficients.len()) else {
                            break;
                        };
                        let wire = n
                            .checked_mul(96)
                            .and_then(|n| 1024usize.checked_add(n))
                            .and_then(|n| {
                                (rows.len() + 1)
                                    .checked_mul(96)
                                    .and_then(|v| n.checked_add(v))
                            })
                            .and_then(|n| next_nnz.checked_mul(56).and_then(|v| n.checked_add(v)))
                            .and_then(|n| h.checked_mul(24).and_then(|v| n.checked_add(v)));
                        if next_nnz > FUNCTIONAL_MAX_NONZEROS
                            || wire.is_none_or(|n| n > FUNCTIONAL_MAX_MESSAGE_BYTES)
                            || !row.valid(n)
                            || row.lower.lo.abs() > ny_core::head_phase_advice::MAX_MAGNITUDE
                            || row.coefficients.iter().any(|&(_, a)| {
                                a.lo.abs() > ny_core::head_phase_advice::MAX_MAGNITUDE
                            })
                        {
                            continue;
                        }
                        nnz = next_nnz;
                        rows.push(row);
                        row_tags.push(RowTag::Mass(proposal));
                        cuts += 1;
                        cone_cuts += 1;
                    }
                    fit.added = cone_cuts;
                    fit.expired |= !live(cone_deadline);
                    fit.receipt();
                }
                // Joint negative-head rows spend ONLY the remainder of the
                // same optional window. Every admitted legacy/cone row stays
                // first; an unfinished group cannot become a partial row.
                if include_groups && live(fit_deadline) && rows.len() < FUNCTIONAL_MAX_ROWS {
                    let (proposed, mut report) = negative_groups::rows(source, fit_deadline);
                    for (row, mask_count) in proposed {
                        // These rows already finished ALL source masks within
                        // the local window. Later group expiry preserves them;
                        // mandatory admission/export uses the original caller.
                        if !live(deadline) || rows.len() >= FUNCTIONAL_MAX_ROWS {
                            break;
                        }
                        if !admit_optional_row(&rows, &row, n, h) {
                            continue;
                        }
                        // Private advice provenance only; re-pricing later
                        // independently checks the exact LHS on a fresh Source.
                        let Some(group) = row
                            .coefficients
                            .iter()
                            .map(|&(j, a)| {
                                (a.lo == a.hi && a.lo <= 0.0)
                                    .then(|| j.checked_sub(d).map(|head| (head, -a.lo)))
                                    .flatten()
                            })
                            .collect::<Option<Vec<_>>>()
                        else {
                            continue;
                        };
                        rows.push(row);
                        row_tags.push(RowTag::Group(group));
                        negative_groups += 1;
                        negative_group_masks += mask_count;
                    }
                    report.added = negative_groups;
                    report.expired |= !live(fit_deadline);
                    report.receipt();
                    // Preserve completed small groups and aligned private tags first.
                    // Growth shares their remaining mask/work/time allowance.
                    if include_large_groups
                        && live(fit_deadline)
                        && rows.len() < FUNCTIONAL_MAX_ROWS
                    {
                        let (grown, mut large) = large_negative_groups::rows(
                            source,
                            &rows,
                            report.completed_masks(),
                            report.work_cells(),
                            fit_deadline,
                        );
                        for (row, mask_count) in grown {
                            if !live(deadline) || rows.len() >= FUNCTIONAL_MAX_ROWS {
                                break;
                            }
                            if !admit_optional_row(&rows, &row, n, h) {
                                continue;
                            }
                            let Some(group) = row
                                .coefficients
                                .iter()
                                .map(|&(j, a)| {
                                    (a.lo == a.hi && a.lo < 0.0)
                                        .then(|| j.checked_sub(d).map(|head| (head, -a.lo)))
                                        .flatten()
                                })
                                .collect::<Option<Vec<_>>>()
                            else {
                                continue;
                            };
                            if group.len() != 5 || mask_count != 32 {
                                continue;
                            }
                            rows.push(row);
                            row_tags.push(RowTag::Group(group));
                            negative_groups += 1;
                            negative_group_masks += mask_count;
                            large_negative_groups += 1;
                            large_negative_group_masks += mask_count;
                        }
                        large.added = large_negative_groups;
                        large.expired |= !live(fit_deadline);
                        large.receipt();
                    }
                }
            }
        }
        if rows.is_empty()
            || row_tags.len() != rows.len()
            || rows.len() > FUNCTIONAL_MAX_ROWS
            || rows.iter().any(|row| !row.valid(n))
            || rows.iter().try_fold(0usize, |total, row| {
                total.checked_add(row.coefficients.len())
            })? > FUNCTIONAL_MAX_NONZEROS
            || !live(deadline)
            || !fp_qualified()
        {
            return None;
        }
        let mut objective = vec![0.0; d];
        objective.extend(
            source
                .q
                .iter()
                .map(|&q| midpoint(q))
                .collect::<Option<Vec<_>>>()?,
        );
        let mut exported = Vec::with_capacity(rows.len());
        for row in &rows {
            if !live(deadline) {
                return None;
            }
            // Plain finite point geometry only. The exact certified source
            // coefficient/RHS enclosures above are never imported from this.
            let coefficients = row
                .coefficients
                .iter()
                .map(|&(j, a)| midpoint(a).map(|point| (j, point)))
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .filter(|(_, a)| *a != 0.0)
                .collect();
            exported.push(HeadFunctionalLpRow {
                coefficients,
                lower: row.lower.lo,
            });
        }
        let request = HeadFunctionalLpRequest {
            protocol_version: FUNCTIONAL_PROTOCOL_VERSION,
            feature_count: d,
            bounds: boxes.iter().map(|b| [b.lo, b.hi]).collect(),
            objective,
            rows: exported,
            slope_rows: role_rows,
            phase_rows: vec![],
        };
        if !live(deadline) || !fp_qualified() || !request.validate() {
            return None;
        }
        Some(Self {
            source,
            boxes,
            rows,
            scalar_rows,
            row_tags,
            request,
            cuts,
            #[cfg(test)]
            cone_cuts,
            negative_groups,
            negative_group_masks,
            large_negative_groups,
            large_negative_group_masks,
        })
    }

    pub(super) fn request(&self) -> &HeadFunctionalLpRequest {
        &self.request
    }
    pub(super) fn cuts(&self) -> usize {
        self.cuts
    }
    #[cfg(test)]
    fn cone_cuts(&self) -> usize {
        self.cone_cuts
    }

    pub(super) fn negative_groups(&self) -> usize {
        self.negative_groups
    }

    pub(super) fn negative_group_masks(&self) -> usize {
        self.negative_group_masks
    }

    pub(super) fn large_negative_groups(&self) -> usize {
        self.large_negative_groups
    }
    pub(super) fn large_negative_group_masks(&self) -> usize {
        self.large_negative_group_masks
    }

    /// For locally owned GEQ rows A*z>=l and represented p>=0:
    /// M>=B-Eout+p*l+min_box((c-A^T*p)*z).
    /// Every actual row coefficient/RHS lies in the OWNED interval row;
    /// every residual coordinate, including zeros/tiny/all last heads, stays.
    pub(super) fn replay(
        &self,
        source: &Source,
        hint: &HeadFunctionalLpHint,
        sign: f64,
        deadline: Instant,
    ) -> Option<f64> {
        self.replay_recorded(source, hint, sign, deadline, false)
            .map(|(bound, _, _)| bound)
    }

    pub(super) fn replay_recorded(
        &self,
        source: &Source,
        hint: &HeadFunctionalLpHint,
        sign: f64,
        deadline: Instant,
        retain: bool,
    ) -> Option<(f64, Option<bridge::ResidualForm>, Option<bridge::Recipe>)> {
        if !std::ptr::eq(self.source, source)
            || !Arc::ptr_eq(&self.source.spec, &source.spec)
            || (sign != 1.0 && sign != -1.0)
            || !live(deadline)
            || !fp_qualified()
            || !hint.validate(&self.request)
            || hint.row_duals.len() != self.rows.len()
            // Provenance is required only when exporting a recipe for a
            // fresh evaluation Source. Bound-only replay still consumes the
            // complete independently owned numerical rows, as before.
            || (retain && self.row_tags.len() != self.rows.len())
            || self.boxes.len() != source.spec.inputs.checked_add(source.spec.heads)?
        {
            return None;
        }
        let mut residual = vec![I::p(0.0); source.spec.inputs];
        residual.extend(source.q.iter().copied());
        let mut constant = source.bias.sub(I::p(source.output_error))?;
        for (row, &raw) in self.rows.iter().zip(&hint.row_duals) {
            if !live(deadline) {
                return None;
            }
            if !row.valid(residual.len()) || !good(raw) {
                return None;
            }
            let point = I::p((sign * raw).max(0.0));
            if point.lo == 0.0 {
                continue;
            }
            constant = constant.add(point.mul(row.lower)?)?;
            for (at, &(j, a)) in row.coefficients.iter().enumerate() {
                if at & 63 == 0 && !live(deadline) {
                    return None;
                }
                residual[j] = residual[j].sub(point.mul(a)?)?;
            }
        }
        let form = retain.then(|| bridge::ResidualForm {
            generation: Arc::clone(&source.spec),
            constant,
            input: residual[..source.spec.inputs].to_vec(),
            relu: residual[source.spec.inputs..].to_vec(),
        });
        let recipe = retain.then(|| bridge::Recipe::Dual {
            rows: self
                .rows
                .iter()
                .zip(&self.row_tags)
                .map(|(row, tag)| bridge::RowRecipe {
                    coefficients: row.coefficients.clone(),
                    mass: match tag {
                        RowTag::Mass(cut) => Some(cut.clone()),
                        _ => None,
                    },
                    group: match tag {
                        RowTag::Group(group) => Some(group.clone()),
                        _ => None,
                    },
                })
                .collect(),
            points: hint
                .row_duals
                .iter()
                .map(|&p| (sign * p).max(0.0))
                .collect(),
        });
        for (j, (&coefficient, &box_)) in residual.iter().zip(&self.boxes).enumerate() {
            if j & 63 == 0 && !live(deadline) {
                return None;
            }
            constant = constant.add(coefficient.mul(box_)?)?;
        }
        (live(deadline) && fp_qualified() && constant.valid()).then_some((
            constant.lo,
            form,
            recipe,
        ))
    }
}

fn admit_optional_row(rows: &[OwnedRow], row: &OwnedRow, variables: usize, heads: usize) -> bool {
    let nnz = rows.iter().try_fold(row.coefficients.len(), |sum, old| {
        sum.checked_add(old.coefficients.len())
    });
    let Some(nnz) = nnz else {
        return false;
    };
    let wire = variables
        .checked_mul(96)
        .and_then(|n| 1024usize.checked_add(n))
        .and_then(|n| {
            (rows.len() + 1)
                .checked_mul(96)
                .and_then(|v| n.checked_add(v))
        })
        .and_then(|n| nnz.checked_mul(56).and_then(|v| n.checked_add(v)))
        .and_then(|n| heads.checked_mul(24).and_then(|v| n.checked_add(v)));
    rows.len() < FUNCTIONAL_MAX_ROWS
        && nnz <= FUNCTIONAL_MAX_NONZEROS
        && wire.is_some_and(|n| n <= FUNCTIONAL_MAX_MESSAGE_BYTES)
        && row.valid(variables)
        && row.lower.lo.abs() <= ny_core::head_phase_advice::MAX_MAGNITUDE
        && row
            .coefficients
            .iter()
            .all(|&(_, a)| a.lo.abs() <= ny_core::head_phase_advice::MAX_MAGNITUDE)
}

/// Direct represented masses: no tau*lambda identity or normalization.
/// sum(a_i*rho(y_i))-tau*rho(y_n)>=min(0,sum(a_i*y_i)-tau*y_n).
pub(super) fn source_cut_lower(source: &Source, cut: &MassCut, deadline: Instant) -> Option<f64> {
    let d = source.spec.inputs;
    let h = source.spec.heads;
    if !live(deadline)
        || !fp_qualified()
        || cut.negative >= h
        || cut.positive.is_empty()
        || cut.positive.len() > MAX_POSITIVE_MASSES
        || !good(cut.tau)
        || cut.tau < 0.0
        || source.lo.len() != d
        || source.hi.len() != d
        || source.spec.first.len() != d.checked_mul(h)?
        || source.spec.first_bias.len() != h
        || source.first_error.len() != h
        || source.first_error.iter().any(|&e| !good(e) || e < 0.0)
    {
        return None;
    }
    let mut terms = Vec::with_capacity(cut.positive.len() + 1);
    for (at, &(head, a)) in cut.positive.iter().enumerate() {
        if head >= h
            || head == cut.negative
            || !good(a)
            || a < 0.0
            || cut.positive[..at].iter().any(|&(old, _)| old == head)
        {
            return None;
        }
        terms.push((head, a));
    }
    terms.push((cut.negative, -cut.tau));
    let mut affine = Affine::zero(d);
    for (head, seed) in terms {
        if !live(deadline) {
            return None;
        }
        if seed == 0.0 {
            continue;
        }
        let point = I::p(seed);
        let error = I::p(seed.abs()).mul(I::p(source.first_error[head]))?.hi;
        affine.b = affine
            .b
            .add(point.mul(I::p(f64::from(source.spec.first_bias[head])))?)?
            .add(I {
                lo: -error,
                hi: error,
            })?;
        for k in 0..d {
            if k & 63 == 0 && !live(deadline) {
                return None;
            }
            affine.a[k] =
                affine.a[k].add(point.mul(I::p(f64::from(source.spec.first[k * h + head])))?)?;
        }
    }
    let box_: Vec<_> = source
        .lo
        .iter()
        .zip(&source.hi)
        .map(|(&lo, &hi)| I { lo, hi })
        .collect();
    let lower = affine.lower(&box_, deadline)?.min(0.0);
    (live(deadline) && fp_qualified() && good(lower)).then_some(lower)
}

fn owned_cut_row(source: &Source, cut: &MassCut, deadline: Instant) -> Option<OwnedRow> {
    let lower = source_cut_lower(source, cut, deadline)?;
    let d = source.spec.inputs;
    let mut coefficients: Vec<_> = cut
        .positive
        .iter()
        .filter(|(_, a)| *a != 0.0)
        .map(|&(j, a)| (d + j, I::p(a)))
        .collect();
    if cut.tau != 0.0 {
        coefficients.push((d + cut.negative, I::p(-cut.tau)));
    }
    coefficients.sort_by_key(|&(j, _)| j);
    Some(OwnedRow {
        coefficients,
        lower: I::p(lower),
    })
}

// Full-width nominal geometry chooses masses only. The complete certificate
// uses the fresh represented products a=RN(tau*lambda), consistently in BOTH
// its signed source seed and its final r-row. No RN product equality is used.
fn nominal_difference(source: &Source, terms: &[(usize, f64)], deadline: Instant) -> Option<f64> {
    let h = source.spec.heads;
    let mut value = 0.0;
    for &(head, s) in terms {
        value += s * f64::from(source.spec.first_bias[head]);
    }
    for k in 0..source.spec.inputs {
        if k & 63 == 0 && !live(deadline) {
            return None;
        }
        let a: f64 = terms
            .iter()
            .map(|&(head, s)| s * f64::from(source.spec.first[k * h + head]))
            .sum();
        value += (a * source.lo[k]).min(a * source.hi[k]);
    }
    (live(deadline) && good(value)).then_some(value.min(0.0))
}

fn proposals(source: &Source, deadline: Instant) -> Option<Vec<MassCut>> {
    let positives: Vec<_> = (0..source.spec.heads)
        .filter(|&j| source.q[j].lo > 0.0)
        .collect();
    let mut ranked = Vec::new();
    let mut trials = 0usize;
    for negative in (0..source.spec.heads).filter(|&j| source.q[j].hi < 0.0) {
        if !live(deadline) {
            return None;
        }
        if trials >= MAX_RANK_TRIALS {
            break;
        }
        let mut neighbors = Vec::new();
        for &positive in &positives {
            let score = nominal_difference(source, &[(positive, 1.0), (negative, -1.0)], deadline)?;
            neighbors.push((score, positive));
        }
        neighbors.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        neighbors.truncate(MAX_NEIGHBORS);
        let mut best: Option<(f64, MassCut)> = None;
        'search: for a in 0..neighbors.len() {
            for c in a + 1..neighbors.len() {
                for lambda in [0.25, 0.5, 0.75] {
                    if trials >= MAX_RANK_TRIALS {
                        break 'search;
                    }
                    trials += 1;
                    let p = neighbors[a].1;
                    let q = neighbors[c].1;
                    let tau = (-source.q[negative].hi)
                        .min(source.q[p].lo / lambda)
                        .min(source.q[q].lo / (1.0 - lambda));
                    let masses = [(p, tau * lambda), (q, tau * (1.0 - lambda))];
                    if !good(tau) || tau <= 0.0 || masses.iter().any(|&(_, a)| !good(a) || a < 0.0)
                    {
                        continue;
                    }
                    let score = nominal_difference(
                        source,
                        &[(p, masses[0].1), (q, masses[1].1), (negative, -tau)],
                        deadline,
                    )?;
                    let independent = masses
                        .iter()
                        .map(|&(head, a)| a * source.y[head].lo.max(0.0))
                        .sum::<f64>()
                        - tau * source.y[negative].hi.max(0.0);
                    let improvement = score - independent;
                    if good(improvement)
                        && improvement > 0.0
                        && best.as_ref().is_none_or(|(old, _)| improvement > *old)
                    {
                        best = Some((
                            improvement,
                            MassCut {
                                positive: masses.to_vec(),
                                negative,
                                tau,
                            },
                        ));
                    }
                }
            }
        }
        if let Some(best) = best {
            ranked.push(best);
        }
    }
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.negative.cmp(&b.1.negative)));
    ranked.truncate(MAX_CUTS);
    Some(ranked.into_iter().map(|(_, proposal)| proposal).collect())
}

#[cfg(test)]
#[path = "input_head_triangle_tests.rs"]
mod tests;

#[path = "input_head_cone_fit.rs"]
mod cone_fit;

#[path = "input_head_negative_groups.rs"]
mod negative_groups;

#[cfg(test)]
#[path = "input_head_retention_tests.rs"]
mod retention_tests;

#[path = "input_head_large_negative_groups.rs"]
mod large_negative_groups;

// Reuse the same complete 32-mask source checker, never an exported/narrow RHS.
pub(super) fn reprice_five_group(
    source: &Source,
    group: &[(usize, f64)],
    deadline: Instant,
    masks: &mut usize,
    cells: &mut usize,
) -> Option<f64> {
    large_negative_groups::source_lower(source, group, deadline, masks, cells)
}

#[cfg(test)]
pub(super) fn expire_five_reprice_after_mask(mask: usize) {
    large_negative_groups::EXPIRE_AFTER_MASK.with(|fault| fault.set(Some(mask)));
}

#[path = "input_head_triangle_root.rs"]
mod root_geometry;
