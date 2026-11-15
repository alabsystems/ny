// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! V3 sparse complete phase-cover search data. This transports no source,
//! certificate, primal, solver status or certified endpoint. NY reconstructs
//! every leaf and replays any proposed multipliers against its owned source.

use super::head_functional_advice::HeadFunctionalLpRow;
use super::head_phase_advice::MAX_MAGNITUDE;

pub const PHASE_COVER_PROTOCOL_VERSION: u32 = 3;
pub const PHASE_COVER_MAX_SELECTED_HEADS: usize = 7;
pub const PHASE_COVER_MAX_LEAVES: usize = 128;
pub const PHASE_COVER_MAX_VARIABLES: usize = 640;
pub const PHASE_COVER_MAX_ROWS: usize = 1056;
/// Stored sparse coefficients, counting common rows only once on the wire.
pub const PHASE_COVER_MAX_NONZEROS: usize = 393_216;
/// Expanded work counts the common rows again for every cold leaf session.
pub const PHASE_COVER_MAX_EXPANDED_NONZEROS: usize = 16 * 1024 * 1024;
pub const PHASE_COVER_MAX_MESSAGE_BYTES: usize = 8 * 1024 * 1024;
pub const PHASE_COVER_MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseCoverLeaf {
    pub mask: usize,
    /// GEQ rows appended to the request's common rows, in exact source order.
    pub rows: Vec<HeadFunctionalLpRow>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseCoverLpRequest {
    pub protocol_version: u32,
    pub selected_heads: usize,
    pub bounds: Vec<[f64; 2]>,
    pub objective: Vec<f64>,
    pub rows: Vec<HeadFunctionalLpRow>,
    /// Exactly all masks in `0..2^selected_heads`, with no duplicates.
    pub leaves: Vec<PhaseCoverLeaf>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseCoverLeafHint {
    pub mask: usize,
    /// Dense multipliers in exact common-row then leaf-row order.
    pub row_duals: Vec<f64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseCoverLpHint {
    /// A unique subset only. Missing leaves grant no infeasibility authority.
    pub leaves: Vec<PhaseCoverLeafHint>,
}

fn finite(value: f64) -> bool {
    value.is_finite() && value.abs() <= MAX_MAGNITUDE
}

fn row_valid(row: &HeadFunctionalLpRow, variables: usize) -> bool {
    finite(row.lower)
        && row.coefficients.iter().enumerate().all(|(j, &(i, a))| {
            i < variables && finite(a) && a != 0.0 && (j == 0 || row.coefficients[j - 1].0 < i)
        })
}

impl PhaseCoverLpRequest {
    /// Conservative transport estimate; actual capped encoding/decoding is
    /// additionally mandatory. A large estimate alone need not reject sparse
    /// small numeric values whose actual JSON fits the independent byte cap.
    pub fn encoded_size_upper_bound(&self) -> Option<usize> {
        let (rows, terms) = self
            .rows
            .iter()
            .chain(self.leaves.iter().flat_map(|l| &l.rows))
            .try_fold((0usize, 0usize), |(r, n), row| {
                Some((r.checked_add(1)?, n.checked_add(row.coefficients.len())?))
            })?;
        1024usize
            .checked_add(self.bounds.len().checked_mul(96)?)?
            .checked_add(rows.checked_mul(96)?)?
            .checked_add(terms.checked_mul(56)?)?
            .checked_add(self.leaves.len().checked_mul(64)?)
    }

    /// Shape/resource validation only. Complete numeric masks are not proof
    /// that any model represents the original neural source or its cover.
    pub fn validate(&self) -> bool {
        let variables = self.bounds.len();
        if self.protocol_version != PHASE_COVER_PROTOCOL_VERSION
            || !(1..=PHASE_COVER_MAX_SELECTED_HEADS).contains(&self.selected_heads)
            || variables == 0
            || variables > PHASE_COVER_MAX_VARIABLES
            || self.objective.len() != variables
            || self.rows.len() > PHASE_COVER_MAX_ROWS
            || self
                .bounds
                .iter()
                .any(|b| !finite(b[0]) || !finite(b[1]) || b[0] > b[1])
            || self.objective.iter().any(|&v| !finite(v))
        {
            return false;
        }
        let count = 1usize << self.selected_heads;
        if self.leaves.len() != count || self.rows.iter().any(|r| !row_valid(r, variables)) {
            return false;
        }
        let Some(common) = self
            .rows
            .iter()
            .try_fold(0usize, |n, r| n.checked_add(r.coefficients.len()))
        else {
            return false;
        };
        let Some(mut expanded) = common.checked_mul(count) else {
            return false;
        };
        let mut stored = common;
        let mut masks = [false; PHASE_COVER_MAX_LEAVES];
        for leaf in &self.leaves {
            if leaf.mask >= count
                || masks[leaf.mask]
                || self
                    .rows
                    .len()
                    .checked_add(leaf.rows.len())
                    .is_none_or(|n| n > PHASE_COVER_MAX_ROWS)
                || leaf.rows.iter().any(|r| !row_valid(r, variables))
            {
                return false;
            }
            masks[leaf.mask] = true;
            let Some(terms) = leaf
                .rows
                .iter()
                .try_fold(0usize, |n, r| n.checked_add(r.coefficients.len()))
            else {
                return false;
            };
            let (Some(s), Some(e)) = (stored.checked_add(terms), expanded.checked_add(terms))
            else {
                return false;
            };
            stored = s;
            expanded = e;
        }
        stored <= PHASE_COVER_MAX_NONZEROS && expanded <= PHASE_COVER_MAX_EXPANDED_NONZEROS
    }
}

impl PhaseCoverLpHint {
    pub fn validate(&self, request: &PhaseCoverLpRequest) -> bool {
        if !request.validate() || self.leaves.len() > request.leaves.len() {
            return false;
        }
        let mut masks = [false; PHASE_COVER_MAX_LEAVES];
        for hint in &self.leaves {
            let Some(leaf) = request.leaves.iter().find(|leaf| leaf.mask == hint.mask) else {
                return false;
            };
            if masks[hint.mask]
                || request.rows.len().checked_add(leaf.rows.len()) != Some(hint.row_duals.len())
                || hint.row_duals.iter().any(|&v| !finite(v))
            {
                return false;
            }
            masks[hint.mask] = true;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> PhaseCoverLpRequest {
        PhaseCoverLpRequest {
            protocol_version: PHASE_COVER_PROTOCOL_VERSION,
            selected_heads: 1,
            bounds: vec![[-1.0, 1.0]],
            objective: vec![1.0],
            rows: vec![HeadFunctionalLpRow {
                coefficients: vec![(0, 1.0)],
                lower: -1.0,
            }],
            leaves: vec![
                PhaseCoverLeaf {
                    mask: 0,
                    rows: vec![],
                },
                PhaseCoverLeaf {
                    mask: 1,
                    rows: vec![HeadFunctionalLpRow {
                        coefficients: vec![(0, 1.0)],
                        lower: 0.0,
                    }],
                },
            ],
        }
    }

    #[test]
    fn complete_masks_dimensions_sparse_rows_and_finite_caps_are_required() {
        let good = request();
        assert!(good.validate());
        for fault in 0..9 {
            let mut bad = good.clone();
            match fault {
                0 => bad.protocol_version = 2,
                1 => bad.selected_heads = 0,
                2 => bad.selected_heads = 8,
                3 => {
                    bad.leaves.pop();
                }
                4 => bad.leaves[1].mask = 0,
                5 => bad.leaves[1].mask = 2,
                6 => bad.bounds[0][0] = f64::NAN,
                7 => bad.rows[0].coefficients.push((0, 1.0)),
                _ => bad.rows[0].coefficients[0].1 = 0.0,
            }
            assert!(!bad.validate(), "fault={fault}");
        }
        let mut reversed = good;
        reversed.leaves.reverse();
        assert!(
            reversed.validate(),
            "complete masks need not be numerically sorted"
        );
    }

    #[test]
    fn hints_are_unique_bounded_subsets_and_carry_no_endpoints() {
        let request = request();
        let good = PhaseCoverLpHint {
            leaves: vec![PhaseCoverLeafHint {
                mask: 1,
                row_duals: vec![0.0, -1.0],
            }],
        };
        assert!(good.validate(&request));
        assert!(PhaseCoverLpHint { leaves: vec![] }.validate(&request));
        let mut bad = good.clone();
        bad.leaves.push(bad.leaves[0].clone());
        assert!(!bad.validate(&request));
        let mut bad = good.clone();
        bad.leaves[0].mask = 2;
        assert!(!bad.validate(&request));
        let mut bad = good.clone();
        bad.leaves[0].row_duals.pop();
        assert!(!bad.validate(&request));
        let mut bad = good;
        bad.leaves[0].row_duals[0] = f64::INFINITY;
        assert!(!bad.validate(&request));
        assert!(
            serde_json::from_str::<PhaseCoverLpHint>(r#"{"leaves":[],"status":"infeasible"}"#)
                .is_err()
        );
    }

    #[test]
    fn stored_sparse_work_and_per_leaf_rows_have_independent_caps() {
        let mut request = request();
        request.bounds = vec![[-1.0, 1.0]; PHASE_COVER_MAX_VARIABLES];
        request.objective = vec![0.0; PHASE_COVER_MAX_VARIABLES];
        request.rows = vec![
            HeadFunctionalLpRow {
                coefficients: (0..PHASE_COVER_MAX_VARIABLES).map(|i| (i, 1.0)).collect(),
                lower: 0.0
            };
            640
        ];
        assert!(
            !request.validate(),
            "stored coefficients exceed aggregate admission"
        );
        request.rows.truncate(400);
        request.selected_heads = PHASE_COVER_MAX_SELECTED_HEADS;
        request.leaves = (0..PHASE_COVER_MAX_LEAVES)
            .map(|mask| PhaseCoverLeaf { mask, rows: vec![] })
            .collect();
        assert!(
            !request.validate(),
            "expanded common-row work has its own aggregate cap"
        );
        request.selected_heads = 1;
        request.leaves.truncate(2);
        request.rows = vec![
            HeadFunctionalLpRow {
                coefficients: vec![],
                lower: 0.0
            };
            PHASE_COVER_MAX_ROWS
        ];
        request.leaves[1].rows.push(HeadFunctionalLpRow {
            coefficients: vec![],
            lower: 0.0,
        });
        assert!(
            !request.validate(),
            "the second leaf adds one row beyond its cap"
        );
        request.leaves[1].rows.clear();
        assert!(request.validate());
        let encoded = serde_json::to_vec(&request).unwrap();
        assert!(encoded.len() <= request.encoded_size_upper_bound().unwrap());
    }
}
