// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deadline-bounded first-party proposals followed by source dual replay.

use super::*;

#[derive(Clone, Debug)]
pub struct NativeLpReport {
    pub proof: LpProofReport,
    pub iterations: usize,
    pub free_variables: usize,
    pub reduced_nonzeros: usize,
    pub elapsed: Duration,
}

impl PairedChain {
    /// Certify the complete root box with a source-authenticated joint
    /// relaxation. Numerical optimizer values never authorize a proof.
    /// Failure or expiration returns no proof; the caller retains the root.
    pub fn prove_lp_until(
        &self,
        domain: &BoxDomain,
        epsilon: f64,
        deadline: Instant,
        steps: usize,
        workers: ParallelWorkers,
    ) -> Option<NativeLpReport> {
        let start = Instant::now();
        if !epsilon.is_finite() || epsilon <= 0.0 || steps > 100_000 || start >= deadline {
            return None;
        }
        let model = lp_model::build_until(self, domain, Some(deadline))?;
        if model
            .output_lo
            .iter()
            .chain(&model.output_hi)
            .all(|v| v.abs() < epsilon)
        {
            let empty = lp_certificate::DualMultipliers {
                equality_multipliers: vec![0.0; model.equalities.len()],
                inequality_multipliers: vec![0.0; model.inequalities.len()],
            };
            let candidates = vec![empty; 2 * model.outputs.len()];
            let proof = lp_model::check_candidates(&model, epsilon, &candidates, Some(deadline))?;
            if Instant::now() >= deadline {
                return None;
            }
            return Some(NativeLpReport {
                proof,
                iterations: 0,
                free_variables: model.bounds.len() - model.equalities.len(),
                reduced_nonzeros: 0,
                elapsed: start.elapsed(),
            });
        }
        // Leave time to join proposals and replay all original source rows.
        let remaining = deadline.saturating_duration_since(Instant::now());
        let verification_tail = Duration::from_secs_f64((remaining.as_secs_f64() * 0.05).min(1.0));
        let proposal_deadline = deadline.checked_sub(verification_tail)?;
        let proposal = lp_optimize::optimize(
            &model,
            lp_optimize::OptimizeOptions {
                steps,
                workers: workers.get(),
                target_lower: Some(-epsilon),
                ..lp_optimize::OptimizeOptions::default()
            },
            proposal_deadline,
        )?;
        if proposal.candidates.iter().enumerate().any(|(i, c)| {
            c.output != i / 2
                || !c.proposal_lower.is_finite()
                || c.sign != if i % 2 == 0 { 1.0 } else { -1.0 }
        }) {
            return None;
        }
        let iterations = proposal
            .candidates
            .iter()
            .map(|c| c.iterations)
            .max()
            .unwrap_or(0);
        let candidates: Vec<_> = proposal.candidates.into_iter().map(|c| c.dual).collect();
        let proof = lp_model::check_candidates(&model, epsilon, &candidates, Some(deadline))?;
        if Instant::now() >= deadline {
            return None;
        }
        Some(NativeLpReport {
            proof,
            iterations,
            free_variables: proposal.free_variables,
            reduced_nonzeros: proposal.reduced_nonzeros,
            elapsed: start.elapsed(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(bias: f64) -> PairedChain {
        PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 1,
                    cols: 1,
                    wf: vec![1.0],
                    wg: vec![1.0],
                    bf: vec![bias],
                    bg: vec![0.0],
                },
                Stage::Relu { width: 1 },
            ],
        }
    }
    fn domain() -> BoxDomain {
        BoxDomain {
            lo: vec![-1.0],
            hi: vec![1.0],
        }
    }
    #[test]
    fn native_proposals_require_source_replay_and_never_close_real_violation() {
        let workers = ParallelWorkers::new(2).unwrap();
        let limit = || Instant::now() + Duration::from_secs(5);
        let safe = chain(0.0)
            .prove_lp_until(&domain(), 0.01, limit(), 100, workers)
            .unwrap();
        assert!(safe.proof.proved);
        let unsafe_band = chain(0.5)
            .prove_lp_until(&domain(), 0.1, limit(), 100, workers)
            .unwrap();
        assert!(!unsafe_band.proof.proved);
    }
    #[test]
    fn native_expired_and_invalid_budgets_do_not_dispatch_a_proof() {
        let c = chain(0.0);
        let workers = ParallelWorkers::new(1).unwrap();
        assert!(c
            .prove_lp_until(&domain(), 0.01, Instant::now(), 100, workers)
            .is_none());
        assert!(c
            .prove_lp_until(
                &domain(),
                f64::NAN,
                Instant::now() + Duration::from_secs(5),
                100,
                workers
            )
            .is_none());
        assert!(c
            .prove_lp_until(
                &domain(),
                0.01,
                Instant::now() + Duration::from_secs(5),
                100_001,
                workers
            )
            .is_none());
    }
}
