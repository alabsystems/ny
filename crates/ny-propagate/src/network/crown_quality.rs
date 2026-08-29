// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared quality classification for completed CROWN targets.

use ny_tensor::BoundedTensor;

/// Resolve the quality-publication guard through its governed declaration.
pub(crate) fn honest_provenance_enabled() -> bool {
    ny_levers::read(&ny_levers::decls::crown_ibp_budget::CROWN_HONEST_PROVENANCE)
        .value
        .as_bool()
}

/// Percentage of finite IBP elements a CROWN result must have turned non-finite
/// to count as vacuous. Not 100: one lucky finite row out of 21,125 must not be
/// able to mask a relation that exploded everywhere else.
const VACUOUS_LOST_MIN_PERCENT: usize = 99;

/// Percentage of improvable elements a vacuous CROWN result may still have
/// materially tightened. Not 0, for the same reason in the other direction.
const VACUOUS_TIGHTENED_MAX_PERCENT: usize = 1;

/// Relative width reduction below which a "tightening" is floating-point noise
/// rather than a bound improvement (#crown-honest-provenance).
const MATERIAL_TIGHTEN_REL_EPS: f32 = 1e-3;

/// Percentage of improvable elements on which the RAW CROWN box must be
/// materially wider than IBP before the node is reported as REGRESSED.
const REGRESSED_WIDENED_MIN_PERCENT: usize = 99;

/// Realized quality of one completed CROWN target, measured against the
/// forward IBP box it is about to be intersected with.
///
/// Every field is an integer count over elements, so the value is independent
/// of summation order and GEMM thread count.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CrownGainStat {
    /// Elements whose IBP interval has both endpoints finite.
    pub(crate) ibp_finite: usize,
    /// Finite, non-degenerate IBP elements that CROWN could tighten.
    pub(crate) improvable: usize,
    /// Finite IBP elements where a CROWN endpoint is not finite.
    pub(crate) crown_lost: usize,
    /// Elements whose intersected width materially improves on IBP.
    pub(crate) tightened_material: usize,
    /// Elements whose raw CROWN interval is materially wider than IBP.
    pub(crate) widened_material: usize,
}

impl CrownGainStat {
    /// Whether the completed CROWN result contributed nothing to this node.
    pub(crate) fn is_vacuous(&self) -> bool {
        self.ibp_finite > 0
            && self.crown_lost * 100 >= self.ibp_finite * VACUOUS_LOST_MIN_PERCENT
            && self.tightened_material * 100 <= self.improvable * VACUOUS_TIGHTENED_MAX_PERCENT
    }

    /// Whether the completed raw CROWN result was finite but materially wider
    /// than the IBP box it started from.
    pub(crate) fn is_regressed(&self) -> bool {
        self.improvable > 0
            && self.widened_material * 100 >= self.improvable * REGRESSED_WIDENED_MIN_PERCENT
            && self.tightened_material * 100 <= self.improvable * VACUOUS_TIGHTENED_MAX_PERCENT
    }
}

/// Measure a completed CROWN target against its forward IBP box.
///
/// This inspects the raw CROWN box before intersection. It fails open on a
/// shape disagreement by returning the all-zero statistic.
pub(crate) fn crown_gain_stat(ibp: &BoundedTensor, crown: &BoundedTensor) -> CrownGainStat {
    let mut stat = CrownGainStat::default();
    if ibp.shape() != crown.shape() {
        return stat;
    }
    for ((&il, &iu), (&cl, &cu)) in ibp
        .lower()
        .iter()
        .zip(ibp.upper().iter())
        .zip(crown.lower().iter().zip(crown.upper().iter()))
    {
        if !il.is_finite() || !iu.is_finite() {
            continue;
        }
        stat.ibp_finite += 1;
        if !cl.is_finite() || !cu.is_finite() {
            stat.crown_lost += 1;
        }
        let ibp_width = iu - il;
        if ibp_width <= 0.0 {
            continue;
        }
        stat.improvable += 1;
        // Mirror `intersection_per_element`: intersect when the intervals meet,
        // union when they do not. A union never counts as a tightening.
        let tl = il.max(cl);
        let tu = iu.min(cu);
        if tl <= tu && ibp_width - (tu - tl) > MATERIAL_TIGHTEN_REL_EPS * ibp_width {
            stat.tightened_material += 1;
        }
        // Regression is intentionally judged on the raw CROWN box; the stored
        // intersection cannot expose a result that was worse than IBP.
        if cl.is_finite()
            && cu.is_finite()
            && (cu - cl) - ibp_width > MATERIAL_TIGHTEN_REL_EPS * ibp_width
        {
            stat.widened_material += 1;
        }
    }
    stat
}
