// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Disjunctive LessEq PGD attack: finds counterexamples for `Y_target >= Y_j`.
//!
//! Delegates to the shared [`super::attack_disjunctive`] implementation with
//! [`DisjunctiveDirection::LessEq`].

use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::Network;

use super::attack_disjunctive::DisjunctiveDirection;
use super::attacker::PgdAttacker;
use super::result::PgdResult;

impl PgdAttacker<'_> {
    /// Attack disjunctions of the form `Y_target >= Y_j` for any `j`.
    pub fn attack_disjunctive_less_eq(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        target_idx: usize,
        comparison_indices: &[usize],
    ) -> Result<PgdResult> {
        self.attack_disjunctive(
            network,
            input_bounds,
            target_idx,
            comparison_indices,
            DisjunctiveDirection::LessEq,
        )
    }

    #[cfg(test)]
    pub(super) fn attack_disjunctive_less_eq_parallel_for_test(
        &self,
        network: &Network,
        input_bounds: &BoundedTensor,
        target_idx: usize,
        comparison_indices: &[usize],
    ) -> Result<PgdResult> {
        self.attack_disjunctive_parallel(
            network,
            input_bounds,
            target_idx,
            comparison_indices,
            DisjunctiveDirection::LessEq,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pgd_attack::config::PgdConfig;

    #[test]
    fn test_disjunctive_less_eq_empty_comparison_indices_returns_err() {
        let attacker = PgdAttacker::new(PgdConfig::fast());
        let network = Network::new();
        let input_bounds = BoundedTensor::new(
            ndarray::arr1(&[0.0_f32]).into_dyn(),
            ndarray::arr1(&[1.0_f32]).into_dyn(),
        )
        .unwrap();

        let result = attacker.attack_disjunctive_less_eq(&network, &input_bounds, 0, &[]);
        assert!(
            result.is_err(),
            "empty comparison_indices should return Err, not panic"
        );
    }
}
