// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Syntactically checked auxiliary bounds for concrete-witness propagation.
//!
//! This type is a proof boundary, not a proof generator. Construction checks
//! only shape, finiteness, and endpoint order. A caller passing these bounds
//! to an abstract transformer must separately establish that every concrete
//! value represented by the verification problem lies in the corresponding
//! interval. In particular, the bounds need not contain every spurious point
//! in another abstract domain.

use crate::certified_box64::{CertifiedBox64, BOX64_HARD_MAX_STORED_F64, BOX64_HARD_MAX_VALUES};
use crate::constrained_zonotope_call_budget::{
    ConstrainedZonotopeCallBudget, ConstrainedZonotopeCallBudgetError, ConstrainedZonotopeCallGate,
    ConstrainedZonotopeCallOutcome, ConstrainedZonotopeCallTracker,
    ConstrainedZonotopePeakLiveBytes,
};

/// Finite outward bounds accompanied by a caller-owned concrete-enclosure
/// proof obligation.
///
/// `CertifiedAuxiliaryBounds64` deliberately does not claim that construction
/// proves semantic enclosure. [`Self::try_new`] is the explicit trust boundary
/// for untyped endpoints: the caller supplies already-certified outward
/// values. [`Self::try_from_certified_box`] is the typed bridge that instead
/// preserves an existing Box enclosure. Both routes prevent later structural
/// misuse of the stored endpoints.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedAuxiliaryBounds64 {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

/// Invalid storage or exhausted conversion resources for
/// [`CertifiedAuxiliaryBounds64`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CertifiedAuxiliaryBounds64Error {
    /// The parallel endpoint vectors disagree in length.
    #[error("auxiliary bound length mismatch: lower has {lower}, upper has {upper}")]
    LengthMismatch {
        /// Number of lower endpoints.
        lower: usize,
        /// Number of upper endpoints.
        upper: usize,
    },

    /// An endpoint is NaN or infinite.
    #[error("auxiliary {side}[{coordinate}] must be finite")]
    NonFinite {
        /// Endpoint vector containing the invalid value.
        side: &'static str,
        /// Coordinate containing the invalid value.
        coordinate: usize,
    },

    /// A lower endpoint exceeds its corresponding upper endpoint.
    #[error("auxiliary lower[{coordinate}] exceeds upper[{coordinate}]")]
    Reversed {
        /// Coordinate with invalid endpoint order.
        coordinate: usize,
    },

    /// Checked storage accounting overflowed.
    #[error("resource size overflow while computing {operation}")]
    ResourceOverflow {
        /// Failed calculation.
        operation: &'static str,
    },

    /// A source Box invariant would exceed the absolute storage firewall.
    #[error("resource limit exceeded for {resource}: required {required}, limit {limit}")]
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Required count.
        required: usize,
        /// Absolute cap inherited from the Box domain.
        limit: usize,
    },

    /// Copying certified endpoints could not reserve bounded storage.
    #[error("unable to reserve storage for {resource}")]
    AllocationFailure {
        /// Requested endpoint buffer.
        resource: &'static str,
    },
}

/// Typed refusal from the budgeted certified-Box conversion.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CertifiedAuxiliaryBounds64BudgetError {
    /// Endpoint structure, storage limits, or bounded allocation failed.
    #[error(transparent)]
    Bounds(#[from] CertifiedAuxiliaryBounds64Error),

    /// The caller's deadline or aggregate peak-memory ceiling refused work.
    #[error(transparent)]
    Budget(#[from] ConstrainedZonotopeCallBudgetError),
}

impl CertifiedAuxiliaryBounds64 {
    /// Accept endpoints at the caller-owned semantic certification boundary.
    ///
    /// This validates only storage invariants. Before using the result for a
    /// proof, the caller must establish
    /// `lower[i] <= concrete_value[i] <= upper[i]` for every concrete witness
    /// and coordinate. Endpoints are interpreted as their exact binary64
    /// dyadic values.
    ///
    /// # Errors
    ///
    /// Returns an error when lengths differ, an endpoint is non-finite, or a
    /// lower endpoint exceeds its corresponding upper endpoint.
    pub fn try_new(
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> Result<Self, CertifiedAuxiliaryBounds64Error> {
        if lower.len() != upper.len() {
            return Err(CertifiedAuxiliaryBounds64Error::LengthMismatch {
                lower: lower.len(),
                upper: upper.len(),
            });
        }
        for (coordinate, (&lower_value, &upper_value)) in lower.iter().zip(&upper).enumerate() {
            if !lower_value.is_finite() {
                return Err(CertifiedAuxiliaryBounds64Error::NonFinite {
                    side: "lower",
                    coordinate,
                });
            }
            if !upper_value.is_finite() {
                return Err(CertifiedAuxiliaryBounds64Error::NonFinite {
                    side: "upper",
                    coordinate,
                });
            }
            if lower_value > upper_value {
                return Err(CertifiedAuxiliaryBounds64Error::Reversed { coordinate });
            }
        }
        Ok(Self { lower, upper })
    }

    /// Copy a certified Box into the auxiliary-bound proof boundary.
    ///
    /// The returned endpoints have exactly the same binary64 bits and
    /// dimension as `source`. Consequently, as a Cartesian product of closed
    /// intervals, the result represents exactly the same set enclosure as the
    /// source Box. This transfers the source's conditional enclosure proof; it
    /// does not create or strengthen that proof.
    ///
    /// Neither type carries tensor, layer, network, property, or witness-set
    /// identity. The caller must still establish that `source` belongs to the
    /// auxiliary location where the result is consumed. In particular, this
    /// conversion does not prove topology provenance.
    ///
    /// Storage remains within the absolute Box-domain firewall, and each
    /// endpoint vector is reserved fallibly before it is copied.
    ///
    /// # Errors
    ///
    /// Returns an error if the source's private storage invariants disagree,
    /// its dimension exceeds the Box-domain hard ceilings, or bounded endpoint
    /// allocation fails.
    pub fn try_from_certified_box(
        source: &CertifiedBox64,
    ) -> Result<Self, CertifiedAuxiliaryBounds64Error> {
        let value_dim = source.len();
        if value_dim != source.upper().len() {
            return Err(CertifiedAuxiliaryBounds64Error::LengthMismatch {
                lower: value_dim,
                upper: source.upper().len(),
            });
        }
        validate_box_conversion_storage(value_dim)?;

        let lower = clone_box_endpoints(source.lower(), "auxiliary lower endpoints")?;
        let upper = clone_box_endpoints(source.upper(), "auxiliary upper endpoints")?;
        Self::try_new(lower, upper)
    }

    /// Copy a certified Box under the shared synchronous call firewall.
    ///
    /// This is the budgeted equivalent of [`Self::try_from_certified_box`].
    /// The caller's baseline must include the borrowed source Box and all
    /// other retained state. The complete two-endpoint output is preflighted
    /// before allocation, each coordinate copy/validation is charged with the
    /// standard deadline cadence, and no value is published after a missed
    /// final checkpoint.
    pub fn try_from_certified_box_with_budget(
        source: &CertifiedBox64,
        budget: ConstrainedZonotopeCallBudget,
    ) -> Result<ConstrainedZonotopeCallOutcome<Self>, CertifiedAuxiliaryBounds64BudgetError> {
        let mut gate = ConstrainedZonotopeCallTracker::from_system_clock(budget)?;
        let value = try_from_certified_box_with_gate(source, &mut gate)?;
        Ok(ConstrainedZonotopeCallOutcome::new(value, gate.report()))
    }

    /// Number of bounded coordinates.
    #[must_use]
    pub fn value_dim(&self) -> usize {
        self.lower.len()
    }

    /// Certified lower endpoints.
    #[must_use]
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    /// Certified upper endpoints.
    #[must_use]
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }
}

#[cfg(test)]
fn try_from_certified_box_with_clock<N>(
    source: &CertifiedBox64,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> Result<
    ConstrainedZonotopeCallOutcome<CertifiedAuxiliaryBounds64>,
    CertifiedAuxiliaryBounds64BudgetError,
>
where
    N: FnMut(&'static str) -> std::time::Instant,
{
    let mut gate = ConstrainedZonotopeCallTracker::with_clock(budget, now)?;
    let value = try_from_certified_box_with_gate(source, &mut gate)?;
    Ok(ConstrainedZonotopeCallOutcome::new(value, gate.report()))
}

fn try_from_certified_box_with_gate<G>(
    source: &CertifiedBox64,
    gate: &mut G,
) -> Result<CertifiedAuxiliaryBounds64, CertifiedAuxiliaryBounds64BudgetError>
where
    G: ConstrainedZonotopeCallGate,
{
    gate.checkpoint("certified auxiliary Box-copy admission")?;
    let value_dim = source.len();
    if value_dim != source.upper().len() {
        return Err(CertifiedAuxiliaryBounds64Error::LengthMismatch {
            lower: value_dim,
            upper: source.upper().len(),
        }
        .into());
    }
    validate_box_conversion_storage(value_dim)?;
    gate.checkpoint("certified auxiliary Box-copy validation complete")?;

    let stored_endpoints =
        value_dim
            .checked_mul(2)
            .ok_or(CertifiedAuxiliaryBounds64Error::ResourceOverflow {
                operation: "budgeted Box lower plus upper endpoints",
            })?;
    let mut peak = ConstrainedZonotopePeakLiveBytes::new();
    peak.add_elements::<f64>(
        stored_endpoints,
        "budgeted certified auxiliary endpoint storage",
    )?;
    gate.preflight_peak_live_bytes(peak.finish())?;
    gate.checkpoint("certified auxiliary Box-copy peak-memory preflight complete")?;

    let mut lower = Vec::new();
    let mut upper = Vec::new();
    gate.checkpoint("certified auxiliary lower-endpoint allocation")?;
    try_reserve(&mut lower, value_dim, "budgeted auxiliary lower endpoints")?;
    gate.checkpoint("certified auxiliary upper-endpoint allocation")?;
    try_reserve(&mut upper, value_dim, "budgeted auxiliary upper endpoints")?;

    for coordinate in 0..value_dim {
        gate.charge_items(1, "certified auxiliary Box endpoint copy")?;
        let lower_value = source.lower()[coordinate];
        let upper_value = source.upper()[coordinate];
        if !lower_value.is_finite() {
            return Err(CertifiedAuxiliaryBounds64Error::NonFinite {
                side: "lower",
                coordinate,
            }
            .into());
        }
        if !upper_value.is_finite() {
            return Err(CertifiedAuxiliaryBounds64Error::NonFinite {
                side: "upper",
                coordinate,
            }
            .into());
        }
        if lower_value > upper_value {
            return Err(CertifiedAuxiliaryBounds64Error::Reversed { coordinate }.into());
        }
        lower.push(lower_value);
        upper.push(upper_value);
    }
    gate.checkpoint("certified auxiliary Box endpoint copy complete")?;

    let output = CertifiedAuxiliaryBounds64 { lower, upper };
    debug_assert_eq!(output.value_dim(), value_dim);
    gate.checkpoint("certified auxiliary Box-copy publication")?;
    Ok(output)
}

fn validate_box_conversion_storage(
    value_dim: usize,
) -> Result<(), CertifiedAuxiliaryBounds64Error> {
    check_limit("Box value count", value_dim, BOX64_HARD_MAX_VALUES)?;
    // The value-count ceiling above makes this multiplication non-overflowing
    // on every Rust target, but keep it checked so the dependency is explicit.
    let stored =
        value_dim
            .checked_mul(2)
            .ok_or(CertifiedAuxiliaryBounds64Error::ResourceOverflow {
                operation: "Box lower plus upper endpoints",
            })?;
    check_limit(
        "Box stored f64 endpoints",
        stored,
        BOX64_HARD_MAX_STORED_F64,
    )
}

fn check_limit(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), CertifiedAuxiliaryBounds64Error> {
    if required > limit {
        return Err(CertifiedAuxiliaryBounds64Error::ResourceLimit {
            resource,
            required,
            limit,
        });
    }
    Ok(())
}

fn clone_box_endpoints(
    source: &[f64],
    resource: &'static str,
) -> Result<Vec<f64>, CertifiedAuxiliaryBounds64Error> {
    let mut result = Vec::new();
    try_reserve(&mut result, source.len(), resource)?;
    result.extend_from_slice(source);
    Ok(result)
}

fn try_reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &'static str,
) -> Result<(), CertifiedAuxiliaryBounds64Error> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| CertifiedAuxiliaryBounds64Error::AllocationFailure { resource })
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;
    use std::time::{Duration, Instant};

    use super::*;
    use crate::{
        CertifiedBox64Error, CertifiedBox64Limits, CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL,
    };

    fn box_limits() -> CertifiedBox64Limits {
        box_limits_for(16)
    }

    fn box_limits_for(value_dim: usize) -> CertifiedBox64Limits {
        CertifiedBox64Limits {
            max_values: value_dim,
            max_stored_f64: value_dim * 2,
            max_weight_elements: 0,
            max_work_items: 0,
            max_scalar_products: 0,
        }
    }

    fn endpoint_bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn validates_storage_without_claiming_semantic_certification() {
        let bounds = CertifiedAuxiliaryBounds64::try_new(
            vec![-1.0, -0.0, f64::from_bits(1)],
            vec![2.0, 0.0, 1.0],
        )
        .unwrap();
        assert_eq!(bounds.value_dim(), 3);
        assert_eq!(bounds.lower(), &[-1.0, -0.0, f64::from_bits(1)]);
        assert_eq!(bounds.upper(), &[2.0, 0.0, 1.0]);

        assert!(matches!(
            CertifiedAuxiliaryBounds64::try_new(vec![0.0], vec![]),
            Err(CertifiedAuxiliaryBounds64Error::LengthMismatch { .. })
        ));
        assert!(matches!(
            CertifiedAuxiliaryBounds64::try_new(vec![f64::NAN], vec![0.0]),
            Err(CertifiedAuxiliaryBounds64Error::NonFinite {
                side: "lower",
                coordinate: 0
            })
        ));
        assert!(matches!(
            CertifiedAuxiliaryBounds64::try_new(vec![0.0], vec![f64::INFINITY]),
            Err(CertifiedAuxiliaryBounds64Error::NonFinite {
                side: "upper",
                coordinate: 0
            })
        ));
        assert!(matches!(
            CertifiedAuxiliaryBounds64::try_new(vec![1.0], vec![-1.0]),
            Err(CertifiedAuxiliaryBounds64Error::Reversed { coordinate: 0 })
        ));
    }

    #[test]
    fn certified_box_conversion_preserves_every_endpoint_bit() {
        let lower = [
            -f64::MAX,
            f64::from_bits(0x8000_0000_0000_0001),
            -0.0,
            f64::from_bits(1),
        ];
        let upper = [-1.0, f64::from_bits(1), 0.0, f64::MAX];
        let source = CertifiedBox64::from_certified_bounds(&lower, &upper, box_limits()).unwrap();

        let auxiliary = CertifiedAuxiliaryBounds64::try_from_certified_box(&source).unwrap();

        assert_eq!(auxiliary.value_dim(), source.len());
        assert_eq!(endpoint_bits(auxiliary.lower()), endpoint_bits(&lower));
        assert_eq!(endpoint_bits(auxiliary.upper()), endpoint_bits(&upper));
    }

    #[test]
    fn certified_box_conversion_accepts_empty_dimension() {
        let source = CertifiedBox64::from_certified_bounds(&[], &[], box_limits()).unwrap();
        let auxiliary = CertifiedAuxiliaryBounds64::try_from_certified_box(&source).unwrap();

        assert_eq!(auxiliary.value_dim(), 0);
        assert!(auxiliary.lower().is_empty());
        assert!(auxiliary.upper().is_empty());
    }

    #[test]
    fn certified_box_source_rejects_nonfinite_endpoints_before_conversion() {
        assert!(matches!(
            CertifiedBox64::from_certified_bounds(&[f64::NAN], &[0.0], box_limits()),
            Err(CertifiedBox64Error::NonFinite {
                field: "lower",
                index: 0
            })
        ));
        assert!(matches!(
            CertifiedBox64::from_certified_bounds(&[0.0], &[f64::INFINITY], box_limits()),
            Err(CertifiedBox64Error::NonFinite {
                field: "upper",
                index: 0
            })
        ));
    }

    #[test]
    fn certified_box_conversion_round_trips_endpoint_bits() {
        let source = CertifiedBox64::from_certified_bounds(
            &[f64::from_bits(0x8000_0000_0000_0001), -0.0, 4.5],
            &[f64::from_bits(1), 0.0, 4.5],
            box_limits(),
        )
        .unwrap();
        let auxiliary = CertifiedAuxiliaryBounds64::try_from_certified_box(&source).unwrap();
        let round_trip = CertifiedBox64::from_certified_bounds(
            auxiliary.lower(),
            auxiliary.upper(),
            box_limits(),
        )
        .unwrap();

        assert_eq!(
            endpoint_bits(round_trip.lower()),
            endpoint_bits(source.lower())
        );
        assert_eq!(
            endpoint_bits(round_trip.upper()),
            endpoint_bits(source.upper())
        );
    }

    #[test]
    fn budgeted_certified_box_conversion_is_bit_identical_with_exact_receipt() {
        let lower = [
            -f64::MAX,
            f64::from_bits(0x8000_0000_0000_0001),
            -0.0,
            f64::from_bits(1),
        ];
        let upper = [-1.0, f64::from_bits(1), 0.0, f64::MAX];
        let source = CertifiedBox64::from_certified_bounds(&lower, &upper, box_limits()).unwrap();
        let start = Instant::now();
        let baseline = 4_096;
        let output_bytes = lower.len() * 2 * size_of::<f64>();
        let budget = ConstrainedZonotopeCallBudget::new(
            start + Duration::from_mins(1),
            baseline,
            baseline + output_bytes,
        );

        let outcome =
            CertifiedAuxiliaryBounds64::try_from_certified_box_with_budget(&source, budget)
                .unwrap();
        assert_eq!(
            endpoint_bits(outcome.value().lower()),
            endpoint_bits(&lower)
        );
        assert_eq!(
            endpoint_bits(outcome.value().upper()),
            endpoint_bits(&upper)
        );
        assert_eq!(outcome.report().peak_live_bytes(), baseline + output_bytes);
        assert_eq!(outcome.report().charged_items(), lower.len());
        assert!(outcome.report().deadline_polls() > 0);

        let one_byte_low = ConstrainedZonotopeCallBudget::new(
            start + Duration::from_mins(1),
            baseline,
            baseline + output_bytes - 1,
        );
        assert!(matches!(
            CertifiedAuxiliaryBounds64::try_from_certified_box_with_budget(
                &source,
                one_byte_low,
            ),
            Err(CertifiedAuxiliaryBounds64BudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { required, limit }
            )) if required == baseline + output_bytes && limit + 1 == required
        ));
    }

    #[test]
    fn budgeted_certified_box_conversion_polls_copy_stride_and_publication() {
        let value_dim = CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL + 1;
        let source = CertifiedBox64::from_certified_bounds(
            &vec![-1.0; value_dim],
            &vec![1.0; value_dim],
            box_limits_for(value_dim),
        )
        .unwrap();
        let start = Instant::now();
        let deadline = start + Duration::from_mins(1);
        let output_bytes = value_dim * 2 * size_of::<f64>();
        let budget = ConstrainedZonotopeCallBudget::new(
            deadline,
            output_bytes,
            output_bytes.checked_mul(2).unwrap(),
        );

        assert!(matches!(
            try_from_certified_box_with_clock(&source, budget, |checkpoint| {
                if checkpoint == "certified auxiliary Box endpoint copy" {
                    deadline
                } else {
                    start
                }
            }),
            Err(CertifiedAuxiliaryBounds64BudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "certified auxiliary Box endpoint copy"
                }
            ))
        ));

        let small =
            CertifiedBox64::from_certified_bounds(&[-1.0, 2.0], &[1.0, 3.0], box_limits()).unwrap();
        let small_bytes = 4 * size_of::<f64>();
        assert!(matches!(
            try_from_certified_box_with_clock(
                &small,
                ConstrainedZonotopeCallBudget::new(deadline, small_bytes, small_bytes * 2),
                |checkpoint| {
                    if checkpoint == "certified auxiliary Box-copy publication" {
                        deadline
                    } else {
                        start
                    }
                },
            ),
            Err(CertifiedAuxiliaryBounds64BudgetError::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "certified auxiliary Box-copy publication"
                }
            ))
        ));
    }

    #[test]
    fn certified_box_conversion_resource_firewall_matches_box_hard_limits() {
        assert_eq!(
            validate_box_conversion_storage(BOX64_HARD_MAX_VALUES),
            Ok(())
        );
        assert_eq!(
            validate_box_conversion_storage(BOX64_HARD_MAX_VALUES + 1),
            Err(CertifiedAuxiliaryBounds64Error::ResourceLimit {
                resource: "Box value count",
                required: BOX64_HARD_MAX_VALUES + 1,
                limit: BOX64_HARD_MAX_VALUES,
            })
        );
    }

    #[test]
    fn certified_box_conversion_maps_reservation_failure() {
        let mut values = Vec::<f64>::new();
        assert_eq!(
            try_reserve(&mut values, usize::MAX, "test endpoints"),
            Err(CertifiedAuxiliaryBounds64Error::AllocationFailure {
                resource: "test endpoints"
            })
        );
        assert!(values.is_empty());
    }
}
