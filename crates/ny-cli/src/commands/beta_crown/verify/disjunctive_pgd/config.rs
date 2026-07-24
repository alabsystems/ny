// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use ny_propagate::{BetaCrownConfig, PgdConfig};

pub(in crate::commands::beta_crown::verify) fn beta_crown_pgd_config(
    config: &BetaCrownConfig,
    num_restarts: usize,
    num_steps: usize,
    deadline: Option<Instant>,
) -> PgdConfig {
    config.pgd_attack_config(num_restarts, num_steps, deadline)
}

#[cfg(test)]
mod tests {
    use super::beta_crown_pgd_config;
    use ny_propagate::{BetaCrownConfig, PgdAlphaMode, PgdOptimizer};

    #[test]
    fn beta_crown_helper_preserves_optimizer_policy() {
        let config = BetaCrownConfig {
            pgd_optimizer: PgdOptimizer::SignedGradient,
            pgd_alpha_mode: PgdAlphaMode::Scalar(0.25),
            ..BetaCrownConfig::default()
        };

        let pgd = beta_crown_pgd_config(&config, 7, 11, None);
        assert_eq!(pgd.optimizer, PgdOptimizer::SignedGradient);
        assert_eq!(pgd.alpha_mode, PgdAlphaMode::Scalar(0.25));
    }
}
