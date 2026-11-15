// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Sparse projected head-functional search data, with no neural authority.
//! Returned row multipliers are replayed against NY's private full source.

use super::head_phase_advice::MAX_MAGNITUDE;

pub const FUNCTIONAL_PROTOCOL_VERSION: u32 = 2;
pub const FUNCTIONAL_MAX_FEATURES: usize = 128;
pub const FUNCTIONAL_MAX_HEAD: usize = 512;
pub const FUNCTIONAL_MAX_VARIABLES: usize = 640;
pub const FUNCTIONAL_MAX_ROWS: usize = 1028;
pub const FUNCTIONAL_MAX_NONZEROS: usize = 67072;
pub const FUNCTIONAL_MAX_MESSAGE_BYTES: usize = 4 * 1024 * 1024;
pub const FUNCTIONAL_MAX_RESPONSE_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadFunctionalLpRow {
    /// Strictly increasing, unique, nonzero `(variable index, coefficient)`.
    pub coefficients: Vec<(usize, f64)>,
    pub lower: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadFunctionalLpRequest {
    pub protocol_version: u32,
    /// Features precede genuine authored ReLU auxiliary variables, in source
    /// order. Objectives may be signed; these roles carry no proof authority.
    pub feature_count: usize,
    pub bounds: Vec<[f64; 2]>,
    pub objective: Vec<f64>,
    pub rows: Vec<HeadFunctionalLpRow>,
    /// Full-head ordering. Some identifies that head's lower epigraph affine
    /// row, including for a signed-objective outer model. None keeps a locally
    /// owned scalar slope or selected phase coefficient. Caller replay owns
    /// which raw duals are useful; this mapping imports no certified endpoint.
    pub slope_rows: Vec<Option<usize>>,
    /// Explicit selected sign rows, in the source's phase-bit ordering.
    pub phase_rows: Vec<usize>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadFunctionalLpHint {
    pub row_duals: Vec<f64>,
}

fn finite(x: f64) -> bool {
    x.is_finite() && x.abs() <= MAX_MAGNITUDE
}

impl HeadFunctionalLpRequest {
    pub fn encoded_size_upper_bound(&self) -> Option<usize> {
        let terms = self
            .rows
            .iter()
            .try_fold(0usize, |n, row| n.checked_add(row.coefficients.len()))?;
        1024usize
            .checked_add(self.bounds.len().checked_mul(96)?)?
            .checked_add(self.rows.len().checked_mul(96)?)?
            .checked_add(terms.checked_mul(56)?)?
            .checked_add(self.slope_rows.len().checked_mul(24)?)?
            .checked_add(self.phase_rows.len().checked_mul(24)?)
    }

    /// Structural and resource validation only. Role indices cannot import
    /// a model, a source identity, a status, or a certified endpoint.
    pub fn validate(&self) -> bool {
        let n = self.bounds.len();
        let positives = self.slope_rows.iter().flatten().count();
        if self.protocol_version != FUNCTIONAL_PROTOCOL_VERSION
            || !(1..=FUNCTIONAL_MAX_FEATURES).contains(&self.feature_count)
            || self.slope_rows.is_empty()
            || self.slope_rows.len() > FUNCTIONAL_MAX_HEAD
            || self.feature_count.checked_add(positives) != Some(n)
            || n > FUNCTIONAL_MAX_VARIABLES
            || self.objective.len() != n
            || self.rows.is_empty()
            || self.rows.len() > FUNCTIONAL_MAX_ROWS
            || self.phase_rows.len() > 4
            || self
                .bounds
                .iter()
                .any(|b| !finite(b[0]) || !finite(b[1]) || b[0] > b[1])
            || self.objective.iter().any(|&x| !finite(x))
            || self
                .encoded_size_upper_bound()
                .is_none_or(|n| n > FUNCTIONAL_MAX_MESSAGE_BYTES)
        {
            return false;
        }
        for (j, role) in self.slope_rows.iter().enumerate() {
            if let Some(r) = role {
                if *r >= self.rows.len()
                    || self.slope_rows[..j].contains(&Some(*r))
                    || self.phase_rows.contains(r)
                {
                    return false;
                }
            }
        }
        if self
            .phase_rows
            .iter()
            .enumerate()
            .any(|(j, &r)| r >= self.rows.len() || self.phase_rows[..j].contains(&r))
        {
            return false;
        }
        let mut nnz = 0usize;
        for row in &self.rows {
            if !finite(row.lower)
                || row.coefficients.iter().enumerate().any(|(j, &(i, a))| {
                    i >= n || !finite(a) || a == 0.0 || j > 0 && row.coefficients[j - 1].0 >= i
                })
            {
                return false;
            }
            let Some(total) = nnz.checked_add(row.coefficients.len()) else {
                return false;
            };
            nnz = total;
            if nnz > FUNCTIONAL_MAX_NONZEROS {
                return false;
            }
        }
        true
    }
}

impl HeadFunctionalLpHint {
    pub fn validate(&self, request: &HeadFunctionalLpRequest) -> bool {
        request.validate()
            && self.row_duals.len() == request.rows.len()
            && self.row_duals.iter().all(|&x| finite(x))
            && self
                .row_duals
                .len()
                .checked_mul(32)
                .and_then(|n| n.checked_add(128))
                .is_some_and(|n| n <= FUNCTIONAL_MAX_RESPONSE_BYTES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> HeadFunctionalLpRequest {
        HeadFunctionalLpRequest {
            protocol_version: FUNCTIONAL_PROTOCOL_VERSION,
            feature_count: 1,
            bounds: vec![[-1.0, 1.0], [0.0, 1.0]],
            objective: vec![-0.375, 1.0],
            rows: vec![
                HeadFunctionalLpRow {
                    coefficients: vec![(1, 1.0)],
                    lower: 0.0,
                },
                HeadFunctionalLpRow {
                    coefficients: vec![(0, -1.0), (1, 1.0)],
                    lower: 0.0,
                },
            ],
            slope_rows: vec![Some(1), None],
            phase_rows: vec![],
        }
    }

    #[test]
    fn functional_protocol_rejects_dimensions_roles_sparse_order_and_nonfinite_values() {
        let r = request();
        assert!(r.validate());
        let mut bad = r.clone();
        bad.feature_count = 2;
        assert!(!bad.validate());
        let mut bad = r.clone();
        bad.slope_rows[1] = Some(1);
        assert!(!bad.validate());
        let mut bad = r.clone();
        bad.phase_rows.push(1);
        assert!(!bad.validate());
        let mut bad = r.clone();
        bad.rows[1].coefficients.reverse();
        assert!(!bad.validate());
        let mut bad = r.clone();
        bad.rows[1].coefficients[0].1 = 0.0;
        assert!(!bad.validate());
        let mut bad = r.clone();
        bad.rows[1].coefficients[0].0 = 2;
        assert!(!bad.validate());
        let mut bad = r.clone();
        bad.bounds[0][0] = f64::NAN;
        assert!(!bad.validate());
        let mut bad = r.clone();
        bad.rows[0].lower = f64::INFINITY;
        assert!(!bad.validate());
        let mut bad = r;
        bad.protocol_version = 1;
        assert!(!bad.validate());
    }

    #[test]
    fn functional_protocol_caps_sparse_work_and_actual_wire_size() {
        let mut r = request();
        r.feature_count = FUNCTIONAL_MAX_FEATURES;
        r.slope_rows = vec![Some(1); FUNCTIONAL_MAX_HEAD];
        r.bounds = vec![[0.0, 1.0]; FUNCTIONAL_MAX_VARIABLES];
        r.objective = vec![0.0; FUNCTIONAL_MAX_VARIABLES];
        r.rows = (0..FUNCTIONAL_MAX_ROWS)
            .map(|_| HeadFunctionalLpRow {
                coefficients: (0..FUNCTIONAL_MAX_FEATURES).map(|i| (i, 1.0)).collect(),
                lower: 0.0,
            })
            .collect();
        for (j, role) in r.slope_rows.iter_mut().enumerate() {
            *role = Some(j);
        }
        assert!(
            !r.validate(),
            "nonzero cap is independent of variable/row caps"
        );
        for row in r.rows.iter_mut().skip(512) {
            row.coefficients.clear();
        }
        assert!(r.validate());
        assert!(serde_json::to_vec(&r).unwrap().len() <= r.encoded_size_upper_bound().unwrap());
        r.bounds.push([0.0, 1.0]);
        r.objective.push(0.0);
        assert!(!r.validate());
    }

    #[test]
    fn functional_hints_are_only_bounded_row_multipliers() {
        let r = request();
        assert!(HeadFunctionalLpHint {
            row_duals: vec![0.0, -0.375]
        }
        .validate(&r));
        assert!(!HeadFunctionalLpHint {
            row_duals: vec![0.0]
        }
        .validate(&r));
        assert!(!HeadFunctionalLpHint {
            row_duals: vec![0.0, f64::NAN]
        }
        .validate(&r));
        assert!(
            serde_json::from_str::<HeadFunctionalLpHint>(r#"{"row_duals":[0,0],"lower":1}"#)
                .is_err()
        );
    }
}
