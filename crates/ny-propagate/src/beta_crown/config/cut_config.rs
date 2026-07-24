// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cut policy and scoring configuration for GCP-CROWN cut pools.

use serde::{Deserialize, Serialize};

/// Cut eviction policy for GCP-CROWN cut pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CutEvictionPolicy {
    /// Evict the oldest cut (FIFO).
    Fifo,
    /// Evict the lowest-utility cut using weighted scoring.
    #[default]
    UtilityWeighted,
}

/// Scoring weights for cut eviction (utility-weighted policy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutScoreWeights {
    /// Weight for lambda magnitude.
    pub w_lambda: f32,
    /// Weight for recency (exponential decay of age).
    pub w_recent: f32,
    /// Weight for use count (log-scaled).
    pub w_usage: f32,
    /// Weight for contribution magnitude (EMA).
    pub w_contrib: f32,
    /// Weight for source depth (log-scaled).
    pub w_depth: f32,
    /// Cap for lambda contribution.
    pub lambda_cap: f32,
    /// Cap for contribution magnitude.
    pub contrib_cap: f32,
    /// Time constant for recency decay (in iterations).
    pub tau_iters: f32,
    /// Bonus for verified-domain cuts.
    pub verified_bonus: f32,
    /// Bonus for near-miss cuts.
    pub near_miss_bonus: f32,
    /// Bonus for proactive cuts.
    pub proactive_bonus: f32,
}

impl Default for CutScoreWeights {
    fn default() -> Self {
        Self {
            w_lambda: 1.0,
            w_recent: 0.5,
            w_usage: 0.4,
            w_contrib: 0.8,
            w_depth: 0.2,
            lambda_cap: 10.0,
            contrib_cap: 1.0,
            tau_iters: 200.0,
            verified_bonus: 0.5,
            near_miss_bonus: 0.1,
            proactive_bonus: 0.0,
        }
    }
}
