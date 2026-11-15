// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bounded parallel proof search with a complete retained domain cover.
//!
//! A coordinator dispatches fixed-order LIFO batches to persistent workers.
//! Every original batch domain remains owned by the coordinator until the
//! entire batch succeeds; results are merged in dispatch order. The domain
//! cap is global and reserved BEFORE dispatch. Scheduling cannot increase it
//! or affect the fixed-worker-count tree prefix. LIFO batches retain a small
//! depth-first frontier, rather than materializing an exponential BFS layer.
//! An expiry, refused bound, worker failure or invalid FP environment retains
//! the current batch and every queued domain and returns no proof.

use super::*;
use std::sync::{mpsc, Arc, Mutex};

/// Validated worker count for this std-only search, capped at sixteen.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParallelWorkers(usize);

impl ParallelWorkers {
    pub fn new(count: usize) -> Option<Self> {
        (1..=16).contains(&count).then_some(Self(count))
    }

    pub fn get(self) -> usize {
        self.0
    }
}

#[derive(Clone)]
struct Work {
    domain: BoxDomain,
    depth: usize,
}

enum Answer {
    Bound(Option<BoundReport>),
    Refused(&'static str),
}

/// Runtime operations defeat constant folding. These probes reject directed
/// rounding, DAZ and FTZ in EACH worker, rather than assuming FP state inherited
/// from the initiating thread is suitable. No unsafe FP-control APIs are used.
fn suitable_fp_environment() -> bool {
    use std::hint::black_box;
    let large = black_box(9_007_199_254_740_992.0_f64);
    let one = black_box(1.0_f64);
    let three_halves = black_box(1.5_f64);
    let tiny = black_box(f64::from_bits(1));
    let two = black_box(2.0_f64);
    let normal = black_box(f64::MIN_POSITIVE);
    let half = black_box(0.5_f64);
    black_box(large + one) == large
        && black_box(large + three_halves) == large + 2.0
        && black_box(-large - three_halves) == -large - 2.0
        && black_box(tiny * two).to_bits() == 2
        && black_box(tiny + tiny).to_bits() == 2
        && black_box(normal * half).to_bits() == (1u64 << 51)
}

/// Split scores are heuristics only. Every fourth depth chooses the largest
/// remaining fraction of a root input interval, so a dominant gradient cannot
/// starve all other axes. Adjacent-f64 intervals are skipped while another
/// axis still admits a strict interior midpoint.
pub(super) fn split_axis(
    domain: &BoxDomain,
    root: &BoxDomain,
    bounds: &BoundReport,
    depth: usize,
) -> Result<usize, &'static str> {
    let dim = domain.lo.len();
    if domain.hi.len() != dim
        || root.lo.len() != dim
        || root.hi.len() != dim
        || bounds.split_scores.len() != dim
    {
        return Err("bound_refused");
    }
    let fairness = depth % 4 == 3;
    let mut has_width = false;
    let mut best: Option<(usize, f64)> = None;
    for j in 0..dim {
        if domain.lo[j] >= domain.hi[j] {
            continue;
        }
        has_width = true;
        let mid = domain.lo[j] * 0.5 + domain.hi[j] * 0.5;
        if !(domain.lo[j] < mid && mid < domain.hi[j]) {
            continue;
        }
        let width = domain.hi[j] - domain.lo[j];
        let root_width = root.hi[j] - root.lo[j];
        let score = if fairness && root_width > 0.0 {
            width / root_width
        } else if fairness {
            0.0
        } else if bounds.split_scores[j].is_finite() {
            bounds.split_scores[j]
        } else {
            width
        };
        if best
            .as_ref()
            .is_none_or(|&(old, old_score)| score.total_cmp(&old_score).then(j.cmp(&old)).is_gt())
        {
            best = Some((j, score));
        }
    }
    best.map(|(j, _)| j).ok_or(if has_width {
        "unsplittable"
    } else {
        "point_unclosed"
    })
}

fn children(
    work: &Work,
    bounds: &BoundReport,
    root: &BoxDomain,
) -> Result<[Work; 2], &'static str> {
    if work.depth >= 96 {
        return Err("depth_cap");
    }
    let axis = split_axis(&work.domain, root, bounds, work.depth)?;
    let mid = work.domain.lo[axis] * 0.5 + work.domain.hi[axis] * 0.5;
    if !(work.domain.lo[axis] < mid && mid < work.domain.hi[axis]) {
        return Err("unsplittable");
    }
    let mut left = work.domain.clone();
    let mut right = work.domain.clone();
    left.hi[axis] = mid;
    right.lo[axis] = mid;
    // The midpoint is an exact shared f64 boundary: the two CLOSED boxes
    // cover the original closed box, regardless of midpoint rounding.
    Ok([
        Work {
            domain: right,
            depth: work.depth + 1,
        },
        Work {
            domain: left,
            depth: work.depth + 1,
        },
    ])
}

impl PairedChain {
    /// Prove the same strict property as `prove`, using a bounded native CPU
    /// worker pool. Worker completion order never changes merge order. Cap
    /// accounting is global; queued and in-flight domains are never discarded.
    /// The one-worker case preserves the serial DFS dispatch order.
    pub fn prove_parallel(
        &self,
        root: &BoxDomain,
        epsilon: f64,
        deadline: Instant,
        max_domains: u64,
        paired: bool,
        workers: ParallelWorkers,
    ) -> SearchReport {
        self.run_parallel(root, epsilon, deadline, max_domains, paired, workers, None)
    }

    /// Build rigorous reference bounds ONCE from this same source and root,
    /// then reuse them while proving a complete input split cover. References
    /// cannot be imported by the caller or carried across independent roots.
    /// Refused refinement retains the original root and authorizes no proof.
    /// The absolute deadline and reported elapsed include the refinement.
    pub fn prove_refined_parallel(
        &self,
        root: &BoxDomain,
        epsilon: f64,
        deadline: Instant,
        max_domains: u64,
        paired: bool,
        workers: ParallelWorkers,
    ) -> SearchReport {
        self.prove_refined_parallel_impl(
            root,
            epsilon,
            deadline,
            max_domains,
            paired,
            workers,
            true,
        )
    }

    /// Controlled comparison against the original source refinement, using
    /// the same tree/search implementation and worker accounting.
    pub fn prove_refined_parallel_baseline(
        &self,
        root: &BoxDomain,
        epsilon: f64,
        deadline: Instant,
        max_domains: u64,
        paired: bool,
        workers: ParallelWorkers,
    ) -> SearchReport {
        self.prove_refined_parallel_impl(
            root,
            epsilon,
            deadline,
            max_domains,
            paired,
            workers,
            false,
        )
    }

    fn prove_refined_parallel_impl(
        &self,
        root: &BoxDomain,
        epsilon: f64,
        deadline: Instant,
        max_domains: u64,
        paired: bool,
        workers: ParallelWorkers,
        strong: bool,
    ) -> SearchReport {
        let start = Instant::now();
        let mut report = SearchReport {
            proved: false,
            evaluated: 0,
            closed: 0,
            open: 1,
            max_depth: 0,
            elapsed: Duration::ZERO,
            reason: "invalid",
            worst_closed_bound: 0.0,
        };
        if self.validate().is_err()
            || self.input_state(root).is_none()
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || max_domains == 0
        {
            report.elapsed = start.elapsed();
            return report;
        }
        if !suitable_fp_environment() {
            report.reason = "fp_environment";
            report.elapsed = start.elapsed();
            return report;
        }
        if Instant::now() >= deadline {
            report.reason = "deadline";
            report.elapsed = start.elapsed();
            return report;
        }
        let reference = if strong {
            refine::states_strong(self, root, Some(deadline), paired)
        } else {
            refine::states(self, root, Some(deadline), paired)
        };
        let Some(reference) = reference else {
            report.reason = if Instant::now() >= deadline {
                "deadline"
            } else {
                "refinement_refused"
            };
            report.elapsed = start.elapsed();
            return report;
        };
        let mut report = self.run_parallel(
            root,
            epsilon,
            deadline,
            max_domains,
            paired,
            workers,
            Some(&reference),
        );
        report.elapsed = start.elapsed();
        report
    }

    fn run_parallel(
        &self,
        root: &BoxDomain,
        epsilon: f64,
        deadline: Instant,
        max_domains: u64,
        paired: bool,
        workers: ParallelWorkers,
        reference: Option<&[Enclosure]>,
    ) -> SearchReport {
        let start = Instant::now();
        let mut report = SearchReport {
            proved: false,
            evaluated: 0,
            closed: 0,
            open: 1,
            max_depth: 0,
            elapsed: Duration::ZERO,
            reason: "invalid",
            worst_closed_bound: 0.0,
        };
        if self.validate().is_err()
            || self.input_state(root).is_none()
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || max_domains == 0
        {
            return report;
        }
        let mut queue = vec![Work {
            domain: root.clone(),
            depth: 0,
        }];
        let mut stop = None;
        std::thread::scope(|scope| {
            let (job_tx, job_rx) = mpsc::channel::<(usize, BoxDomain)>();
            let job_rx = Arc::new(Mutex::new(job_rx));
            let (answer_tx, answer_rx) = mpsc::channel::<(usize, Answer)>();
            let mut handles = Vec::new();
            for _ in 0..workers.get() {
                let receiver = Arc::clone(&job_rx);
                let sender = answer_tx.clone();
                let handle = std::thread::Builder::new().spawn_scoped(scope, move || loop {
                    let job = receiver.lock().unwrap_or_else(|p| p.into_inner()).recv();
                    let Ok((slot, domain)) = job else {
                        break;
                    };
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        if !suitable_fp_environment() {
                            Answer::Refused("fp_environment")
                        } else {
                            Answer::Bound(self.bound_with_reference_until(
                                &domain,
                                paired,
                                Some(deadline),
                                Some(epsilon),
                                reference,
                            ))
                        }
                    }))
                    .unwrap_or(Answer::Refused("worker_panicked"));
                    if sender.send((slot, result)).is_err() {
                        break;
                    }
                });
                match handle {
                    Ok(handle) => handles.push(handle),
                    Err(_) => {
                        stop = Some("worker_spawn");
                        break;
                    }
                }
            }
            drop(answer_tx);
            while stop.is_none() && !queue.is_empty() {
                if Instant::now() >= deadline {
                    stop = Some("deadline");
                    break;
                }
                if report.evaluated >= max_domains {
                    stop = Some("domain_cap");
                    break;
                }
                let remaining = max_domains - report.evaluated;
                let count = queue
                    .len()
                    .min(workers.get())
                    .min(remaining.min(16) as usize);
                let batch: Vec<_> = (0..count).map(|_| queue.pop().unwrap()).collect();
                // These reservations include refused/panicked evaluations:
                // every attempted task consumes the one global resource cap.
                report.evaluated += count as u64;
                for (slot, work) in batch.iter().enumerate() {
                    if job_tx.send((slot, work.domain.clone())).is_err() {
                        stop = Some("worker_failed");
                        break;
                    }
                }
                let mut answers: Vec<Option<Answer>> = (0..count).map(|_| None).collect();
                let mut received = 0;
                while stop.is_none() && received < count {
                    let now = Instant::now();
                    if now >= deadline {
                        stop = Some("deadline");
                        break;
                    }
                    let wait = deadline.duration_since(now).min(Duration::from_millis(50));
                    match answer_rx.recv_timeout(wait) {
                        Ok((slot, answer)) if slot < count && answers[slot].is_none() => {
                            answers[slot] = Some(answer);
                            received += 1;
                        }
                        Ok(_) => {
                            stop = Some("worker_protocol");
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            stop = Some("worker_failed");
                        }
                    }
                }
                if stop.is_none() && Instant::now() >= deadline {
                    stop = Some("deadline");
                }
                if stop.is_none() {
                    for answer in &answers {
                        match answer {
                            Some(Answer::Refused(reason)) => {
                                stop = Some(*reason);
                                break;
                            }
                            Some(Answer::Bound(None)) => {
                                stop = Some("bound_refused");
                                break;
                            }
                            None => {
                                stop = Some("worker_protocol");
                                break;
                            }
                            Some(Answer::Bound(Some(_))) => {}
                        }
                    }
                }
                if stop.is_some() {
                    // Reverse dispatch order restores the original LIFO batch
                    // exactly, including tasks still executing in workers.
                    queue.extend(batch.into_iter().rev());
                    break;
                }
                for (slot, work) in batch.iter().enumerate() {
                    if Instant::now() >= deadline {
                        stop = Some("deadline");
                        queue.extend(batch[slot..].iter().rev().cloned());
                        break;
                    }
                    let Some(Answer::Bound(Some(bounds))) = answers[slot].take() else {
                        stop = Some("worker_protocol");
                        queue.extend(batch[slot..].iter().rev().cloned());
                        break;
                    };
                    report.max_depth = report.max_depth.max(work.depth);
                    if bounds.lower.len() != self.output_dim
                        || bounds.upper.len() != self.output_dim
                        || bounds.split_scores.len() != self.input_dim
                        || bounds
                            .lower
                            .iter()
                            .zip(&bounds.upper)
                            .any(|(&l, &h)| !l.is_finite() || !h.is_finite() || l > h)
                    {
                        stop = Some("bound_refused");
                        queue.extend(batch[slot..].iter().rev().cloned());
                        break;
                    }
                    let worst = bounds
                        .lower
                        .iter()
                        .chain(&bounds.upper)
                        .map(|v| v.abs())
                        .fold(0.0, f64::max);
                    if worst < epsilon {
                        report.closed += 1;
                        report.worst_closed_bound = report.worst_closed_bound.max(worst);
                    } else {
                        match children(work, &bounds, root) {
                            Ok(children) => queue.extend(children),
                            Err(reason) => {
                                stop = Some(reason);
                                queue.extend(batch[slot..].iter().rev().cloned());
                                break;
                            }
                        }
                    }
                }
            }
            // Closing the task sender releases every idle worker. Deadline-
            // aware bounds let in-flight workers return before scope teardown.
            drop(job_tx);
            for handle in handles {
                if handle.join().is_err() && stop.is_none() {
                    stop = Some("worker_panicked");
                    // A thread failure can never manufacture exhausted cover.
                    queue.push(Work {
                        domain: root.clone(),
                        depth: 0,
                    });
                }
            }
        });
        if stop.is_none() && Instant::now() >= deadline {
            stop = Some("deadline");
            if queue.is_empty() {
                queue.push(Work {
                    domain: root.clone(),
                    depth: 0,
                });
            }
        }
        report.open = queue.len();
        report.proved = stop.is_none() && queue.is_empty();
        report.reason = if report.proved {
            "covered"
        } else {
            stop.unwrap_or("open")
        };
        report.elapsed = start.elapsed();
        report
    }
}

#[cfg(test)]
#[path = "search_tests.rs"]
mod tests;
