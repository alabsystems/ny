// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bounded automatic routing. These controls limit work and never authorize
//! a closure: every selected class still needs the certified search ledger.

use std::time::{Duration, Instant};

use super::super::net::TwinOp;
use super::{breakpoint, RootGates, TwinNet, YBox};

pub(super) const MAX_SECS: f64 = 4.0;
pub(super) const SHARE: f64 = 0.05;
pub(super) const TREE_RESERVE: f64 = 1.0;
pub(super) const MIN_SECS: f64 = 0.05;
pub(super) const NODE_CAP: u64 = 8192;
pub(super) const MAX_CLASSES: usize = 2;
const SINGLE_CLASS_BYTES: u64 = 64 << 20;
pub(super) const STATE_BYTES: u64 = 256 << 20;
const MAX_DEPTH: usize = 32;
const MAX_ROW_TERMS: usize = 250_000;
const PILOT_CALLS: u64 = 32;
const SEARCH_SLICE: Duration = Duration::from_millis(5);
// Neither frame magnitude products nor a single row's breakpoint sorter can
// be interrupted internally. Refuse large estimated work before either begins.
// Streaming checks the deadline between chunks; it still scores every row.
const MAX_COMPUTE_WORK: u64 = 1_000_000_000;

pub(super) fn budget_secs(left: Option<f64>) -> f64 {
    left.map_or(MAX_SECS, |left| {
        let left = left.max(0.0);
        MAX_SECS
            .min(SHARE * left)
            .min((left - TREE_RESERVE).max(0.0))
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Skip {
    NoObjective,
    NoBackend,
    Time,
    NoUnstable,
    DenseMemory,
    ComputeWork,
}

impl Skip {
    pub(super) fn token(self) -> &'static str {
        match self {
            Self::NoObjective => "no_open_objective",
            Self::NoBackend => "no_backend",
            Self::Time => "time",
            Self::NoUnstable => "no_unstable",
            Self::DenseMemory => "dense_memory",
            Self::ComputeWork => "compute_work",
        }
    }
}

/// Facts computed before constructing the magnitude frame or state matrices.
#[derive(Debug, Clone, Copy)]
pub(super) struct Facts {
    pub unstable: usize,
    pub dense_bytes: u64,
    pub retained_bytes: u64,
    pub compute_work: u64,
}

impl Facts {
    /// Give a large dense state the tier's whole budget rather than splitting
    /// its construction time across two objectives. Routing uses row geometry.
    pub(super) fn class_limit(self) -> usize {
        if self.dense_bytes >= SINGLE_CLASS_BYTES {
            1
        } else {
            MAX_CLASSES
        }
    }

    pub(super) fn from_root(net: &TwinNet, root: &RootGates, ybox: &YBox) -> Self {
        let unstable = root
            .layers
            .iter()
            .fold(0usize, |sum, layer| sum.saturating_add(layer.unst.len()))
            .saturating_add(
                ybox.ly
                    .iter()
                    .zip(&ybox.uy)
                    .filter(|(l, u)| **l < 0.0 && **u > 0.0)
                    .count(),
            );
        let input_gens = root.lo.iter().zip(&root.hi).filter(|(l, u)| l != u).count();
        Self::for_net(net, input_gens, unstable)
    }

    pub(super) fn from_frame(net: &TwinNet, frame: &super::FrozenDualFrame) -> Self {
        Self::for_net(net, frame.n_in_gens, frame.n_unstable)
    }

    fn for_net(net: &TwinNet, input_gens: usize, unstable: usize) -> Self {
        let mut facts = Self::from_counts(input_gens, unstable);
        let one_pass = net.ops.iter().enumerate().fold(0u64, |sum, (k, op)| {
            let terms = match op {
                TwinOp::Conv(conv) => (conv.k_fwd as u64)
                    .saturating_mul(net.tsize.get(k + 1).copied().unwrap_or(usize::MAX) as u64),
                TwinOp::Gemm { weight, .. } => weight.len() as u64,
                TwinOp::ChannelAffine { scale, .. } => scale.len() as u64,
                TwinOp::Relu { .. } | TwinOp::Add { .. } => {
                    net.tsize.get(k + 1).copied().unwrap_or(usize::MAX) as u64
                }
                TwinOp::Flatten { .. } => 0,
            };
            sum.saturating_add(terms)
        });
        facts.compute_work = Self::work_estimate(input_gens, unstable, one_pass);
        facts
    }

    fn work_estimate(input_gens: usize, unstable: usize, one_pass: u64) -> u64 {
        let n = input_gens.saturating_add(unstable);
        let sort_factor = u64::from(usize::BITS - n.max(1).leading_zeros()) + 1;
        let order_work = (unstable as u64)
            .saturating_mul(n as u64)
            .saturating_mul(sort_factor)
            .saturating_mul(2);
        // Limit the class-independent magnitude pass and complete row scoring.
        // State construction already checks its class deadline per chunk/op;
        // charging its batched arithmetic at a guessed scalar rate would
        // reject useful image states without a measured throughput argument.
        one_pass.saturating_add(order_work)
    }

    fn from_counts(input_gens: usize, unstable: usize) -> Self {
        let n = input_gens.saturating_add(unstable) as u64;
        Self {
            unstable,
            dense_bytes: (unstable as u64).saturating_mul(n).saturating_mul(8),
            retained_bytes: (unstable.min(super::FROZEN_DUAL_MAX_DEPTH) as u64)
                .saturating_mul(n)
                .saturating_mul(8),
            compute_work: 0,
        }
    }

    pub(super) fn admit(
        self,
        objectives: usize,
        backend: bool,
        secs: f64,
        budget_bytes: u64,
    ) -> Result<(), Skip> {
        if objectives == 0 {
            Err(Skip::NoObjective)
        } else if !backend {
            Err(Skip::NoBackend)
        } else if !secs.is_finite() || secs < MIN_SECS {
            Err(Skip::Time)
        } else if self.unstable == 0 {
            Err(Skip::NoUnstable)
        } else if self.retained_bytes > budget_bytes {
            Err(Skip::DenseMemory)
        } else if self.compute_work > MAX_COMPUTE_WORK {
            Err(Skip::ComputeWork)
        } else {
            Ok(())
        }
    }
}

/// Bound the frame's width-dependent pass buffers by choosing a smaller seed
/// chunk before calling the exact existing state estimator.
pub(super) fn chunk_cols(net: &TwinNet) -> usize {
    let max_width = net.tsize.iter().copied().max().unwrap_or(1).max(1) as u64;
    let bytes_per_col = max_width
        .saturating_mul(48)
        .saturating_add((net.n_in as u64).saturating_mul(8));
    // Leave three quarters for the stored rows and all other state data.
    usize::try_from((STATE_BYTES / 4) / bytes_per_col.max(1))
        .unwrap_or(256)
        .clamp(1, super::super::zono_state::FROZEN_DUAL_DEFAULT_CHUNK_COLS)
}

/// Continue beyond the pilot only when a certified refinement pruned a node
/// in this chunk. A tighter bound alone does not show reduced tree work.
pub(super) fn allow_refine(n: usize, depth: usize, stats: &breakpoint::Stats) -> bool {
    depth <= MAX_DEPTH
        && n.saturating_mul(depth) <= MAX_ROW_TERMS
        && (stats.calls < PILOT_CALLS || stats.prunes > 0)
}

pub(super) fn search_deadline(class_deadline: Option<Instant>, now: Instant) -> Option<Instant> {
    let slice = now + SEARCH_SLICE;
    Some(class_deadline.map_or(slice, |deadline| deadline.min(slice)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::margin_row::root::{LayerGates, RootGates};
    use crate::margin_row::{RoundMode, TwinOpSpec, TwinSpec};

    #[test]
    fn automatic_budget_bounds_wall_and_preserves_the_tree_reserve() {
        for left in [0.0, 0.2, 1.0, 1.01, 2.0, 10.0, 40.0, 100.0, 900.0] {
            let budget = budget_secs(Some(left));
            assert!((0.0..=MAX_SECS).contains(&budget));
            assert!(budget <= SHARE * left);
            assert!(budget <= (left - TREE_RESERVE).max(0.0));
        }
        assert_eq!(budget_secs(None), 4.0);
        assert_eq!(budget_secs(Some(100.0)), 4.0);
        assert_eq!(budget_secs(Some(80.0)), 4.0);
        assert_eq!(budget_secs(Some(10.0)), 0.5);
        assert_eq!(budget_secs(Some(f64::NAN)), 0.0);
    }

    #[test]
    fn automatic_admission_rejects_expensive_or_inert_work_before_a_frame() {
        let small = Facts::from_counts(5, 3);
        assert_eq!(small.dense_bytes, 192);
        assert_eq!(small.class_limit(), 2);
        assert_eq!(
            Facts {
                dense_bytes: SINGLE_CLASS_BYTES - 1,
                ..small
            }
            .class_limit(),
            2
        );
        assert_eq!(
            Facts {
                dense_bytes: SINGLE_CLASS_BYTES,
                ..small
            }
            .class_limit(),
            1
        );
        assert_eq!(small.admit(1, true, 1.0, STATE_BYTES), Ok(()));
        assert_eq!(
            small.admit(0, true, 1.0, STATE_BYTES),
            Err(Skip::NoObjective)
        );
        assert_eq!(
            small.admit(1, false, 1.0, STATE_BYTES),
            Err(Skip::NoBackend)
        );
        assert_eq!(small.admit(1, true, 0.01, STATE_BYTES), Err(Skip::Time));
        assert_eq!(small.admit(1, true, 1.0, 191), Err(Skip::DenseMemory));
        assert_eq!(
            Facts::from_counts(usize::MAX, usize::MAX).admit(1, true, 1.0, STATE_BYTES),
            Err(Skip::DenseMemory)
        );
        assert_eq!(
            Facts::from_counts(5, 0).admit(1, true, 1.0, STATE_BYTES),
            Err(Skip::NoUnstable)
        );
        assert_eq!(
            Facts::from_counts(usize::MAX, usize::MAX).dense_bytes,
            u64::MAX
        );
        let expensive = Facts {
            compute_work: MAX_COMPUTE_WORK + 1,
            ..small
        };
        assert_eq!(
            expensive.admit(1, true, 1.0, STATE_BYTES),
            Err(Skip::ComputeWork)
        );
    }

    #[test]
    fn automatic_refinement_skips_large_nodes_and_an_unproductive_pilot() {
        let mut stats = breakpoint::Stats::default();
        assert!(allow_refine(1000, 8, &stats));
        assert!(!allow_refine(1000, 33, &stats));
        assert!(!allow_refine(125_001, 2, &stats));
        assert!(!allow_refine(usize::MAX, 2, &stats));
        stats.calls = PILOT_CALLS;
        assert!(!allow_refine(1000, 8, &stats));
        stats.accepted = 1;
        assert!(!allow_refine(1000, 8, &stats));
        stats.prunes = 1;
        assert!(allow_refine(1000, 8, &stats));
    }

    #[test]
    fn automatic_preflight_admits_representative_image_state_geometry() {
        let mut cifar = Facts::from_counts(3072, 376);
        assert_eq!(cifar.dense_bytes, 10_371_584);
        assert_eq!(cifar.retained_bytes, cifar.dense_bytes);
        cifar.compute_work = Facts::work_estimate(3072, 376, 90_000_000);
        assert_eq!(cifar.admit(2, true, 2.0, STATE_BYTES), Ok(()));
        assert_eq!(cifar.class_limit(), 2);
        let mut tiny = Facts::from_counts(9408, 2010);
        assert_eq!(tiny.dense_bytes, 183_601_440);
        assert_eq!(tiny.retained_bytes, 46_768_128);
        tiny.compute_work = Facts::work_estimate(9408, 2010, 278_000_000);
        assert_eq!(tiny.admit(2, true, 2.0, STATE_BYTES), Ok(()));
        assert_eq!(tiny.class_limit(), 1);
        // Dense storage is eliminated; full row construction work remains.
        // The exact estimator still includes pass buffers and score scratch.
    }

    #[test]
    fn automatic_streaming_admission_uses_retained_rows_but_preserves_work_limits() {
        let mut facts = Facts::from_counts(3072, 4500);
        assert!(facts.dense_bytes > STATE_BYTES);
        assert!(facts.retained_bytes < 32 << 20);
        facts.compute_work = Facts::work_estimate(3072, 4500, 1_000_000);
        assert_eq!(facts.admit(1, true, 2.0, STATE_BYTES), Ok(()));
        // Avoid dividing a costly complete row build across two objectives.
        assert_eq!(facts.class_limit(), 1);
        facts.compute_work = MAX_COMPUTE_WORK + 1;
        assert_eq!(
            facts.admit(1, true, 2.0, STATE_BYTES),
            Err(Skip::ComputeWork)
        );
    }

    #[test]
    fn automatic_from_root_admits_a_synthetic_compiled_cifar_sized_frame() {
        let n_in = 3072;
        let head = 376;
        let mut weights = vec![0.0; n_in * head];
        for j in 0..head {
            weights[j * n_in] = 1.0;
        }
        let net = TwinNet::compile(&TwinSpec {
            n_in,
            ops: vec![
                TwinOpSpec::ChannelAffine {
                    input: 0,
                    scale: vec![1.0; n_in],
                    shift: vec![2.0; n_in],
                    scale_rel_err: 0.0,
                    shift_err: vec![0.0; n_in],
                    shape: (n_in, 1, 1),
                },
                TwinOpSpec::Relu { input: 1 },
                TwinOpSpec::Gemm {
                    input: 2,
                    weight: weights,
                    bias: vec![-2.0; head],
                    bias_err: Vec::new(),
                    weight_rel_err: 0.0,
                    shape: (head, n_in),
                },
                TwinOpSpec::Relu { input: 3 },
                TwinOpSpec::Gemm {
                    input: 4,
                    weight: vec![0.0; 3 * head],
                    bias: vec![0.05, 0.0, 0.0],
                    bias_err: Vec::new(),
                    weight_rel_err: 0.0,
                    shape: (3, head),
                },
            ],
        })
        .expect("compiled dense image fixture");
        // The trunk's x+2 is positive on the input box, so its ReLU is exact.
        // Every head preactivation is (x[0]+2)-2 = x[0], with head box [-1,1].
        // Construct those certified boxes directly to avoid an input identity
        // tableau in this preflight test.
        let root = RootGates {
            mode: RoundMode::Outward,
            lo: vec![-1.0; n_in],
            hi: vec![1.0; n_in],
            mid: vec![0.0; n_in],
            rad: vec![1.0; n_in],
            xabs: vec![1.0; n_in],
            layers: vec![LayerGates {
                op: 1,
                n: n_in,
                l: vec![1.0; n_in],
                u: vec![3.0; n_in],
                alpha: vec![1.0; n_in],
                s: vec![1.0; n_in],
                c: vec![0.0; n_in],
                ms: vec![1.0; n_in],
                unst: Vec::new(),
                clip_rows: None,
            }],
            backward_interm_execution: Default::default(),
            root_precision_policy: Default::default(),
            root_precision_execution: Default::default(),
            spec_alpha_policy: None,
            head_alpha: None,
            root_eval_memo: Default::default(),
            head_group_source: None,
        };
        let ybox = YBox {
            ly: vec![-1.0; head],
            uy: vec![1.0; head],
        };
        let facts = Facts::from_root(&net, &root, &ybox);
        assert_eq!(facts.unstable, 376);
        assert_eq!(facts.dense_bytes, 10_371_584);
        assert_eq!(facts.admit(2, true, 2.0, STATE_BYTES), Ok(()));
        assert_eq!(facts.class_limit(), 2);
        let frame = super::super::FrozenDualFrame::from_root(&net, &root, &ybox, 0)
            .expect("valid exact fixture frame");
        assert_eq!(frame.n_gens, 3448);
        assert!(super::super::estimate_state_bytes(&frame, chunk_cols(&net)) <= STATE_BYTES);
        let policy = super::super::TierPolicy::clamped(u64::MAX, 0.0);
        assert!(policy.automatic());
        assert_eq!(policy.effective_node_cap(), NODE_CAP);
        assert_eq!(policy.budget_secs(Some(100.0)), 4.0);
    }

    #[test]
    fn automatic_search_slice_does_not_change_the_certification_deadline() {
        let now = Instant::now();
        let class_deadline = now + Duration::from_secs(1);
        assert_eq!(
            search_deadline(Some(class_deadline), now),
            Some(now + SEARCH_SLICE)
        );
        assert_eq!(class_deadline, now + Duration::from_secs(1));
        let soon = now + Duration::from_millis(1);
        assert_eq!(search_deadline(Some(soon), now), Some(soon));
        assert_eq!(search_deadline(None, now), Some(now + SEARCH_SLICE));
    }
}
