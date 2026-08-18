// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verification/model-scoped rate limiter for CROWN degradation diagnostics.

use std::sync::atomic::{AtomicU64, Ordering};

/// Admission receipt for a retained CROWN-degradation diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CrownDegradationLogReceipt {
    pub(crate) occurrence: u64,
    pub(crate) suppressed_since_previous_checkpoint: u64,
}

/// Shared by every graph/domain clone belonging to one verification.
///
/// A branch-and-bound run invokes the intermediate collector once per domain,
/// so an expected resource fallback can otherwise emit the same multi-line
/// warning millions of times. Retaining the first observation and powers of
/// two keeps the failure visible, gives an exponential progress signal, and
/// bounds output to O(log(domains)). The two severity classes are independent
/// so a minority degradation cannot hide the first quality-cliff warning.
#[derive(Debug, Default)]
pub(crate) struct CrownDegradationLogScope {
    warning_occurrences: AtomicU64,
    info_occurrences: AtomicU64,
}

impl CrownDegradationLogScope {
    pub(crate) fn warning_receipt(&self) -> Option<CrownDegradationLogReceipt> {
        Self::receipt(&self.warning_occurrences)
    }

    pub(crate) fn info_receipt(&self) -> Option<CrownDegradationLogReceipt> {
        Self::receipt(&self.info_occurrences)
    }

    fn receipt(counter: &AtomicU64) -> Option<CrownDegradationLogReceipt> {
        // Saturation avoids a theoretical wrap back to occurrence one. Relaxed
        // is sufficient: the counter allocates unique report ordinals but
        // protects no verifier data.
        let occurrence = counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_add(1))
            })
            .map_or(u64::MAX, |previous| previous.saturating_add(1));
        if occurrence == 0 || !occurrence.is_power_of_two() {
            return None;
        }
        Some(CrownDegradationLogReceipt {
            occurrence,
            // Between checkpoints N/2 and N, N/2+1..N-1 were suppressed.
            suppressed_since_previous_checkpoint: (occurrence / 2).saturating_sub(1),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::CrownDegradationLogScope;
    use std::sync::{atomic::Ordering, Arc};

    #[test]
    fn retains_first_and_power_of_two_aggregate_reports() {
        let scope = CrownDegradationLogScope::default();
        let retained = (1..=16)
            .filter_map(|_| scope.warning_receipt())
            .map(|receipt| {
                (
                    receipt.occurrence,
                    receipt.suppressed_since_previous_checkpoint,
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(retained, vec![(1, 0), (2, 0), (4, 1), (8, 3), (16, 7)]);
        assert_eq!(scope.warning_occurrences.load(Ordering::Relaxed), 16);
    }

    #[test]
    fn fresh_scopes_each_retain_their_first_warning() {
        let first = CrownDegradationLogScope::default();
        assert_eq!(first.warning_receipt().map(|r| r.occurrence), Some(1));
        assert_eq!(first.warning_receipt().map(|r| r.occurrence), Some(2));
        assert_eq!(first.warning_receipt(), None);

        let later = CrownDegradationLogScope::default();
        assert_eq!(later.warning_receipt().map(|r| r.occurrence), Some(1));
        assert_eq!(first.warning_occurrences.load(Ordering::Relaxed), 3);
        assert_eq!(later.warning_occurrences.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn warning_and_info_streams_are_independent() {
        let scope = CrownDegradationLogScope::default();
        assert_eq!(scope.warning_receipt().map(|r| r.occurrence), Some(1));
        assert_eq!(scope.info_receipt().map(|r| r.occurrence), Some(1));
        assert_eq!(scope.warning_receipt().map(|r| r.occurrence), Some(2));
        assert_eq!(scope.info_receipt().map(|r| r.occurrence), Some(2));
    }

    #[test]
    fn saturated_counter_never_wraps_to_a_first_report() {
        let scope = CrownDegradationLogScope::default();
        scope.warning_occurrences.store(u64::MAX, Ordering::Relaxed);
        assert_eq!(scope.warning_receipt(), None);
        assert_eq!(scope.warning_occurrences.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn concurrent_domain_clones_allocate_each_checkpoint_once() {
        let scope = Arc::new(CrownDegradationLogScope::default());
        let workers = (0..8)
            .map(|_| {
                let scope = Arc::clone(&scope);
                std::thread::spawn(move || {
                    (0..128)
                        .filter_map(|_| scope.warning_receipt())
                        .map(|receipt| {
                            (
                                receipt.occurrence,
                                receipt.suppressed_since_previous_checkpoint,
                            )
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();

        let mut retained = workers
            .into_iter()
            .flat_map(|worker| worker.join().expect("rate-limit worker must not panic"))
            .collect::<Vec<_>>();
        retained.sort_unstable();
        assert_eq!(
            retained,
            vec![
                (1, 0),
                (2, 0),
                (4, 1),
                (8, 3),
                (16, 7),
                (32, 15),
                (64, 31),
                (128, 63),
                (256, 127),
                (512, 255),
                (1024, 511),
            ]
        );
        assert_eq!(scope.warning_occurrences.load(Ordering::Relaxed), 1024);
    }
}
