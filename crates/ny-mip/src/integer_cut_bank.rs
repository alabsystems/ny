// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integer-premise cut reuse, separate from LP-only `CertifiedRow` authority.
//!
//! A source MILP entails each banked cut by independent exact replay. The
//! augmented MILP therefore has the same integer feasible points. A proof of
//! its infeasibility, independently replayed by NY, composes with those cut
//! proofs to exclude the source MILP. This composition grants no LP-only
//! authority to the integer cuts.

use std::time::Instant;

use ay_milp::{BabSession, CertifiedIntegerCut, Model};
use num_rational::BigRational;
use num_traits::ToPrimitive;

use crate::error::MipError;

type Result<T> = std::result::Result<T, MipError>;

// Keep persistent sibling reuse bounded independently of how many sessions a
// caller visits. Refusing additional rows loses strength, never proof authority.
const MAX_BANKED_CUTS: usize = 128;
const MAX_BANKED_NONZEROS: usize = 65_536;

/// Coverage of the last completed collection, not a claim of all-cut export.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IntegerCutBankReport {
    /// AY reports a finished native tree in its caller frame. Structural or
    /// transformed routes may expose no source-frame harvest.
    pub native_tree_finished: bool,
    /// Actual retained global-root sides counted by AY; a lower bound when
    /// `adoption_scan_complete` is false.
    pub adopted_root_sides: usize,
    /// AY finished counting the adopted root-side denominator.
    pub adoption_scan_complete: bool,
    /// Adopted sides whose derivation AY cannot currently export.
    pub unsupported_sides: usize,
    /// AY artifacts reconstructed for the last finished tree (historical,
    /// including artifacts already drained by an earlier collection).
    pub certified_sides: usize,
    /// Proofs actually drained by this collection, before NY source replay.
    pub received_sides: usize,
    /// Valid source-compatible proofs already present in the bounded bank.
    pub duplicate_sides: usize,
    /// Exported rows depending on an augmented/local source fact were omitted.
    pub source_incompatible: usize,
    /// Valid exported rows omitted because the public f64 row builder cannot
    /// represent their exact coefficients/bound without rounding.
    pub unrepresentable_sides: usize,
    /// New source-replayable rows retained in this bounded bank.
    pub added: usize,
    /// Rows omitted by the persistent bank's fixed resource ceiling.
    pub capacity_omissions: usize,
    /// Collection exhausted the original solve clock; coverage is incomplete.
    pub interrupted: bool,
}

/// Retained integer-cut proofs for reuse across source-compatible MILP domains.
///
/// The bank exposes no unchecked insertion and grants no standalone verdict.
/// Every recipient is rechecked before any row is installed. Source changes,
/// a sibling-only row, or relaxed integrality cannot inherit cut authority.
#[derive(Debug, Clone, Default)]
pub struct IntegerCutBank {
    cuts: Vec<CertifiedIntegerCut>,
    last_report: IntegerCutBankReport,
}

impl IntegerCutBank {
    /// Number of retained proofs, bounded independently of solver count.
    pub fn len(&self) -> usize {
        self.cuts.len()
    }

    /// Whether the bank has no retained integer proofs.
    pub fn is_empty(&self) -> bool {
        self.cuts.is_empty()
    }

    /// Discard all retained source premises and previous coverage.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Actual collection coverage of the last completed intake or banked solve.
    pub fn last_report(&self) -> IntegerCutBankReport {
        self.last_report
    }

    /// Drain an actual finished AY tree into a bank bound to NY's original MILP.
    ///
    /// Both the session frame and NY's unchanged source are independently
    /// replayed. Integer premises confer no LP-only row authority. This explicit
    /// intake leaves NY's structural routing and window-cut recipes unchanged.
    /// The caller supplies the same absolute deadline used for the solve.
    ///
    /// # Errors
    /// Refuses an unfinished native tree, an incompatible existing bank, invalid
    /// solver evidence, unsupported NY lowering, or expiry. The bank changes
    /// only after complete in-budget replay. Session artifacts are drained when
    /// collection starts, including on a later replay error or interruption.
    pub fn harvest_from_finished_session(
        &mut self,
        session: &mut BabSession,
        source: &crate::MilpProblem,
        deadline: Instant,
    ) -> Result<IntegerCutBankReport> {
        poll(Some(deadline))?;
        if !session.cut_harvest_summary().native_tree_finished {
            return Err(MipError::Solver(
                "integer cut intake requires a finished native source-frame harvest".into(),
            ));
        }
        let original = source_model(source, Some(deadline))?;
        poll(Some(deadline))?;
        for cut in &self.cuts {
            poll(Some(deadline))?;
            cut.verify_with_deadline(&original, Some(deadline))
                .map_err(|error| {
                    MipError::Solver(format!("integer cut bank source replay: {error}"))
                })?;
        }
        let mut next = self
            .snapshot_with_deadline(deadline)
            .ok_or_else(|| MipError::Solver("integer cut intake deadline expired".into()))?;
        next.collect(session, &original, Some(deadline))?;
        poll(Some(deadline))?;
        let report = next.last_report;
        *self = next;
        Ok(report)
    }

    /// Snapshot before worker dispatch while retaining the original caller clock.
    pub(crate) fn snapshot_with_deadline(&self, deadline: Instant) -> Option<Self> {
        poll(Some(deadline)).ok()?;
        let mut cuts = Vec::with_capacity(self.cuts.len());
        for cut in &self.cuts {
            poll(Some(deadline)).ok()?;
            let mut coeffs = Vec::with_capacity(cut.coeffs.len());
            for (index, (column, coefficient)) in cut.coeffs.iter().enumerate() {
                if index & 0xff == 0 {
                    poll(Some(deadline)).ok()?;
                }
                coeffs.push((*column, coefficient.clone()));
            }
            poll(Some(deadline)).ok()?;
            cuts.push(CertifiedIntegerCut {
                coeffs,
                lb: cut.lb.clone(),
                proof: cut.proof.clone(),
            });
        }
        poll(Some(deadline)).ok()?;
        Some(Self {
            cuts,
            last_report: self.last_report,
        })
    }

    pub(crate) fn augment(&self, source: &Model, deadline: Option<Instant>) -> Result<Model> {
        // Validate against the unchanged source, never earlier installed cuts.
        // This excludes circular dependencies and integer-to-LP authority loss.
        let mut rows = Vec::with_capacity(self.cuts.len());
        for cut in &self.cuts {
            poll(deadline)?;
            cut.verify_with_deadline(source, deadline)
                .map_err(|error| {
                    MipError::Solver(format!("integer cut recipient replay: {error}"))
                })?;
            let mut entries = Vec::with_capacity(cut.coeffs.len());
            for (index, (column, coefficient)) in cut.coeffs.iter().enumerate() {
                if index & 0xff == 0 {
                    poll(deadline)?;
                }
                let column = source.col_at(*column as usize).ok_or_else(|| {
                    MipError::Encoding("integer cut column missing from recipient".into())
                })?;
                entries.push((column, exact_float(coefficient)?));
            }
            rows.push((entries, exact_float(&cut.lb)?));
        }
        poll(deadline)?;
        let mut augmented = source.clone();
        for (entries, lower) in rows {
            poll(deadline)?;
            augmented.add_row(lower, f64::INFINITY, &entries);
        }
        poll(deadline)?;
        Ok(augmented)
    }

    pub(crate) fn collect(
        &mut self,
        session: &mut BabSession,
        source: &Model,
        deadline: Option<Instant>,
    ) -> Result<()> {
        let coverage = session.cut_harvest_summary();
        self.last_report = IntegerCutBankReport {
            native_tree_finished: coverage.native_tree_finished,
            adopted_root_sides: coverage.adopted_root_sides,
            adoption_scan_complete: coverage.adoption_scan_complete,
            unsupported_sides: coverage.unsupported_sides,
            certified_sides: coverage.certified_sides,
            interrupted: coverage.interrupted,
            ..IntegerCutBankReport::default()
        };
        let mut nonzeros: usize = self.cuts.iter().map(|cut| cut.coeffs.len()).sum();
        let cuts = session.harvest_cuts();
        self.last_report.received_sides = cuts.len();
        for cut in cuts {
            poll(deadline)?;
            // Even AY's exported type is replayed. Invalid solver evidence is
            // a hard error; a valid augmented-source dependency is an omission.
            cut.verify_with_deadline(session.model(), deadline)
                .map_err(|error| {
                    MipError::Solver(format!("harvested integer cut replay: {error}"))
                })?;
            if cut.verify_with_deadline(source, deadline).is_err() {
                poll(deadline)?;
                self.last_report.source_incompatible += 1;
                continue;
            }
            let mut representable = exact_float(&cut.lb).is_ok();
            for (index, (_, coefficient)) in cut.coeffs.iter().enumerate() {
                if index & 0xff == 0 {
                    poll(deadline)?;
                }
                if exact_float(coefficient).is_err() {
                    representable = false;
                    break;
                }
            }
            if !representable {
                self.last_report.unrepresentable_sides += 1;
                continue;
            }
            let mut duplicate = false;
            for existing in &self.cuts {
                poll(deadline)?;
                if existing.lb != cut.lb
                    || existing.proof != cut.proof
                    || existing.coeffs.len() != cut.coeffs.len()
                {
                    continue;
                }
                let mut same = true;
                for (index, (left, right)) in existing.coeffs.iter().zip(&cut.coeffs).enumerate() {
                    if index & 0xff == 0 {
                        poll(deadline)?;
                    }
                    if left != right {
                        same = false;
                        break;
                    }
                }
                if same {
                    duplicate = true;
                    break;
                }
            }
            if duplicate {
                self.last_report.duplicate_sides += 1;
                continue;
            }
            if self.cuts.len() >= MAX_BANKED_CUTS
                || cut.coeffs.len() > MAX_BANKED_NONZEROS.saturating_sub(nonzeros)
            {
                self.last_report.capacity_omissions += 1;
                continue;
            }
            nonzeros += cut.coeffs.len();
            self.cuts.push(cut);
            self.last_report.added += 1;
        }
        poll(deadline)
    }
}

pub(crate) fn source_model(
    source: &crate::MilpProblem,
    deadline: Option<Instant>,
) -> Result<Model> {
    crate::ay_lib::to_ay_model_with_poll(source, || poll(deadline))
}

#[cfg(test)]
thread_local! {
    static TEST_POLLS_LEFT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

fn poll(deadline: Option<Instant>) -> Result<()> {
    #[cfg(test)]
    TEST_POLLS_LEFT.with(|remaining| match remaining.get() {
        Some(0) => Err(MipError::Solver(
            "integer cut replay deadline expired".into(),
        )),
        Some(left) => {
            remaining.set(Some(left - 1));
            Ok(())
        }
        None => Ok(()),
    })?;
    if deadline.is_some_and(|end| Instant::now() >= end) {
        Err(MipError::Solver(
            "integer cut replay deadline expired".into(),
        ))
    } else {
        Ok(())
    }
}

fn exact_float(value: &BigRational) -> Result<f64> {
    // AY's public Model row builder accepts f64. A rounded coefficient can
    // strengthen an integer cut unsoundly, so unsupported conversion refuses.
    value
        .to_f64()
        .filter(|advice| advice.is_finite())
        .filter(|advice| BigRational::from_float(*advice).as_ref() == Some(value))
        .ok_or_else(|| MipError::Encoding("integer cut has no exact f64 row representation".into()))
}

#[cfg(test)]
#[path = "integer_cut_bank/tests.rs"]
mod tests;
