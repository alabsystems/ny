// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bounded numeric LP advice for an owned complete head-phase certificate.
//!
//! This protocol carries no neural model, source identity, solver status or
//! certified endpoint. A provider returns only untrusted row multipliers.
//! NY retains its private source and must replay every proposed multiplier.
//! Serialization is transport only; decoding must cap actual message bytes
//! before allocating and then call these structural validators.

use std::time::Instant;

pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_VARIABLES: usize = 4100;
pub const MAX_ROWS: usize = 12;
pub const MAX_NONZEROS: usize = 32780;
pub const MAX_MESSAGE_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_MAGNITUDE: f64 = 1e100;

/// One explicit GEQ row: `coefficients dot variables >= lower`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadPhaseLpRow {
    pub coefficients: Vec<f64>,
    pub lower: f64,
}

/// Immutable advice snapshot; the objective is a nominal search direction.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadPhaseLpRequest {
    pub protocol_version: u32,
    pub bounds: Vec<[f64; 2]>,
    pub objective: Vec<f64>,
    pub rows: Vec<HeadPhaseLpRow>,
    /// Explicit row indices in the source exporter's phase-bit order.
    pub phase_rows: Vec<usize>,
}

/// Untrusted multipliers, in the request's exact row order.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadPhaseLpHint {
    pub row_duals: Vec<f64>,
}

fn finite_capped(x: f64) -> bool {
    x.is_finite() && x.abs() <= MAX_MAGNITUDE
}

impl HeadPhaseLpRequest {
    /// Conservative upper bound for compact serde_json transport. A finite
    /// binary64 JSON scalar occupies at most 32 bytes including its separator;
    /// fixed fields and row/phase syntax have separate conservative allowances.
    /// Providers must also enforce MAX_MESSAGE_BYTES on the actual wire data.
    pub fn encoded_size_upper_bound(&self) -> Option<usize> {
        let scalars = self
            .bounds
            .len()
            .checked_mul(2)?
            .checked_add(self.objective.len())?
            .checked_add(self.rows.iter().try_fold(0usize, |n, row| {
                n.checked_add(row.coefficients.len())?.checked_add(1)
            })?)?;
        512usize
            .checked_add(scalars.checked_mul(32)?)?
            .checked_add(self.rows.len().checked_mul(128)?)?
            .checked_add(self.phase_rows.len().checked_mul(32)?)
    }

    /// Validate allocation limits, finite caps and exact row/index dimensions.
    /// This grants no proof authority to the encoded problem.
    pub fn validate(&self) -> bool {
        let n = self.bounds.len();
        if self.protocol_version != PROTOCOL_VERSION
            || n == 0
            || n > MAX_VARIABLES
            || self.objective.len() != n
            || self.rows.is_empty()
            || self.rows.len() > MAX_ROWS
            || self.phase_rows.is_empty()
            || self.phase_rows.len() > 4
            || self
                .bounds
                .iter()
                .any(|b| !finite_capped(b[0]) || !finite_capped(b[1]) || b[0] > b[1])
            || self.objective.iter().any(|&v| !finite_capped(v))
            || self
                .phase_rows
                .iter()
                .enumerate()
                .any(|(i, &r)| r >= self.rows.len() || self.phase_rows[..i].contains(&r))
            || self
                .encoded_size_upper_bound()
                .is_none_or(|n| n > MAX_MESSAGE_BYTES)
        {
            return false;
        }
        let mut nnz = 0usize;
        for row in &self.rows {
            if row.coefficients.len() != n
                || !finite_capped(row.lower)
                || row.coefficients.iter().any(|&v| !finite_capped(v))
            {
                return false;
            }
            let Some(total) =
                nnz.checked_add(row.coefficients.iter().filter(|&&v| v != 0.0).count())
            else {
                return false;
            };
            nnz = total;
            if nnz > MAX_NONZEROS {
                return false;
            }
        }
        true
    }
}

impl HeadPhaseLpHint {
    /// Only well-shaped finite multipliers may be offered to owned replay.
    pub fn validate(&self, request: &HeadPhaseLpRequest) -> bool {
        request.validate()
            && self.row_duals.len() == request.rows.len()
            && self.row_duals.iter().all(|&v| finite_capped(v))
            && self
                .row_duals
                .len()
                .checked_mul(32)
                .and_then(|n| n.checked_add(128))
                .is_some_and(|n| n <= MAX_MESSAGE_BYTES)
    }
}

/// An optional search provider. Implementations must honor the original
/// passed deadline, include setup/transport/cleanup in their wait, and retain
/// no detached solver worker after return. A late or malformed hint is ignored
/// by NY. Neither this trait nor its hints can construct an owned certificate.
pub trait HeadPhaseAdvice: Send + Sync {
    fn propose(&self, request: &HeadPhaseLpRequest, deadline: Instant) -> Option<HeadPhaseLpHint>;

    /// Legacy phase-only providers retain both call slots; setup and capability
    /// checks consume the same shared wall allowance as proposal calls.
    fn supports_functional_advice(&self) -> bool {
        false
    }

    /// Optional sparse global-functional search. No returned multiplier has
    /// source or certificate authority; the default preserves V1 providers.
    fn propose_functional(
        &self,
        _request: &super::head_functional_advice::HeadFunctionalLpRequest,
        _deadline: Instant,
    ) -> Option<super::head_functional_advice::HeadFunctionalLpHint> {
        None
    }

    /// Optional V3 complete-cover batch. Missing leaves and all returned
    /// multipliers remain untrusted; source-owned replay retains the floor.
    fn supports_phase_cover_advice(&self) -> bool {
        false
    }

    fn propose_phase_cover(
        &self,
        _request: &super::head_phase_cover_advice::PhaseCoverLpRequest,
        _deadline: Instant,
    ) -> Option<super::head_phase_cover_advice::PhaseCoverLpHint> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> HeadPhaseLpRequest {
        HeadPhaseLpRequest {
            protocol_version: PROTOCOL_VERSION,
            bounds: vec![[-1.0, 1.0], [-1.0, 1.0]],
            objective: vec![1.0, 0.0],
            rows: vec![HeadPhaseLpRow {
                coefficients: vec![-1.0, 1.0],
                lower: 0.0,
            }],
            phase_rows: vec![0],
        }
    }

    #[test]
    fn protocol_rejects_wrong_dimensions_indices_versions_and_nonfinite_data() {
        let base = request();
        assert!(base.validate());
        let mut bad = base.clone();
        bad.protocol_version += 1;
        assert!(!bad.validate());
        let mut bad = base.clone();
        bad.bounds[0] = [1.0, -1.0];
        assert!(!bad.validate());
        let mut bad = base.clone();
        bad.objective.pop();
        assert!(!bad.validate());
        let mut bad = base.clone();
        bad.rows[0].coefficients.push(0.0);
        assert!(!bad.validate());
        let mut bad = base.clone();
        bad.phase_rows.push(0);
        assert!(!bad.validate());
        let mut bad = base.clone();
        bad.phase_rows[0] = 1;
        assert!(!bad.validate());
        let mut bad = base.clone();
        bad.objective[0] = f64::NAN;
        assert!(!bad.validate());
        let mut bad = base.clone();
        bad.rows[0].lower = f64::INFINITY;
        assert!(!bad.validate());
        let mut bad = base;
        bad.bounds[0][0] = -MAX_MAGNITUDE * 2.0;
        assert!(!bad.validate());
    }

    #[test]
    fn protocol_enforces_sparse_work_and_wire_admission() {
        let mut r = request();
        r.bounds = vec![[-1.0, 1.0]; MAX_VARIABLES];
        r.objective = vec![0.0; MAX_VARIABLES];
        r.rows = (0..MAX_ROWS)
            .map(|_| HeadPhaseLpRow {
                coefficients: vec![1.0; MAX_VARIABLES],
                lower: 0.0,
            })
            .collect();
        assert!(!r.validate(), "dense rows exceed the nonzero cap");
        for row in r.rows.iter_mut().skip(7) {
            row.coefficients.fill(0.0);
        }
        assert!(r.validate());
        let encoded = serde_json::to_vec(&r).unwrap();
        assert!(encoded.len() <= r.encoded_size_upper_bound().unwrap());
        assert!(encoded.len() < MAX_MESSAGE_BYTES);
        r.bounds.push([-1.0, 1.0]);
        r.objective.push(0.0);
        assert!(
            !r.validate(),
            "variable allocation cap is independent of sparsity"
        );
    }

    #[test]
    fn hints_have_no_status_endpoint_or_model_authority() {
        let r = request();
        assert!(HeadPhaseLpHint {
            row_duals: vec![-0.375]
        }
        .validate(&r));
        assert!(!HeadPhaseLpHint { row_duals: vec![] }.validate(&r));
        assert!(!HeadPhaseLpHint {
            row_duals: vec![f64::NAN]
        }
        .validate(&r));
        assert!(!HeadPhaseLpHint {
            row_duals: vec![MAX_MAGNITUDE * 2.0]
        }
        .validate(&r));
        assert!(serde_json::from_str::<HeadPhaseLpHint>(r#"{"row_duals":[0],"lower":1}"#).is_err());
    }
}
