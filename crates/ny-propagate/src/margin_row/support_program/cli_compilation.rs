// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Staged, conditional declaration compilation for an original-source issuer.
//!
//! A successful owner means the genuine compiler owns the moved original
//! parameters. A caller may then release a decoder workspace whose values
//! have actually ceased to be live before constructing root products. This
//! owner does not authenticate ONNX bytes or a VNN-LIB objective, and it never
//! accepts an imported net, arbitrary residency claim or supplied proof.

use std::{fmt, time::Instant};

use ny_core::Result;

use super::{
    cli_proof::{PreparedRootProof, PreparedRootProofOptions, PreparedRootSession},
    compiler::OriginalModel,
    ir::{earliest, invalid, refusal},
};
use crate::{
    certify::work::WorkBudget,
    margin_row::{root_box::ROOT_SOURCE_BYTES_CAP, RootPrecisionRuntimePolicy, TwinSpec},
};

/// Only actual moved-declaration compilation can create this owner. The
/// immutable model retains its exact original ledger, clock and byte claim.
#[derive(Clone)]
pub struct PreparedRootCompilation {
    model: OriginalModel,
    deadline: Option<Instant>,
}

impl fmt::Debug for PreparedRootCompilation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedRootCompilation")
            .field("inputs", &self.n_in())
            .field("outputs", &self.n_out())
            .field("resident_bytes", &self.resident_bytes())
            .field("parameter_contract", &self.model.identity().contract())
            // This includes the compiler's existing fixed owner allowance;
            // it is not a System-allocation, RSS or conservative lease total.
            .field(
                "compiled_capacity_with_owner_allowance_bytes",
                &self.model.actual_bytes(),
            )
            .finish_non_exhaustive()
    }
}

impl PreparedRootCompilation {
    pub fn compile_declared_twin(
        spec: TwinSpec,
        budget: &WorkBudget,
        requested: Option<Instant>,
    ) -> Result<Self> {
        // Include every decoder/source/concurrent claim on this SAME ledger
        // in the unchanged prepared-route ceiling before compiler allocation.
        // Tightening is monotone, and cannot increase any caller allowance.
        budget
            .tighten_total_byte_limit(ROOT_SOURCE_BYTES_CAP)
            .map_err(refusal)?;
        let model = OriginalModel::compile(spec, budget, requested)?;
        let deadline = model.deadline(requested)?;
        let result = Self { model, deadline };
        result.checkpoint(budget, requested)?;
        Ok(result)
    }

    pub fn budget(&self) -> &WorkBudget {
        self.model.budget()
    }

    pub fn n_in(&self) -> usize {
        self.model.net().n_in
    }

    pub fn n_out(&self) -> usize {
        self.model.net().n_out
    }

    /// The actual model owner's conservative paid capacity, not a new credit
    /// for root, decoder or publication allocations.
    pub fn resident_bytes(&self) -> usize {
        self.model.resident_bytes()
    }

    pub fn source_digest(&self) -> &[u8; 32] {
        self.model.identity().source_digest()
    }

    pub fn compiled_digest(&self) -> &[u8; 32] {
        self.model.identity().compiled_digest()
    }

    pub fn deadline(&self, requested: Option<Instant>) -> Result<Option<Instant>> {
        self.model.deadline(earliest(self.deadline, requested))
    }

    pub fn checkpoint(&self, budget: &WorkBudget, requested: Option<Instant>) -> Result<()> {
        if !self.budget().shares_ledger(budget) {
            return Err(invalid("staged compilation requires its original ledger"));
        }
        self.deadline(requested)?;
        ny_core::require_f64_interval_proof_environment()
    }

    /// A consuming source-transfer restriction persists its earliest clock.
    /// Validating a shorter handoff must not let a later caller renew it.
    pub fn restrict_deadline(
        mut self,
        budget: &WorkBudget,
        requested: Option<Instant>,
    ) -> Result<Self> {
        self.checkpoint(budget, requested)?;
        self.deadline = self.deadline(requested)?;
        Ok(self)
    }

    /// Continue from these actual owned parameters without re-extracting,
    /// copying or recompiling them. Numeric proof authority still comes only
    /// from the original root/prefix/terminal producer and class authorizer.
    pub fn prepare_declared_domain(
        self,
        lower: &[f64],
        upper: &[f64],
        policy: RootPrecisionRuntimePolicy,
        options: PreparedRootProofOptions,
        budget: &WorkBudget,
        requested: Option<Instant>,
    ) -> Result<PreparedRootSession> {
        self.checkpoint(budget, requested)?;
        let at = self.deadline(requested)?;
        PreparedRootSession::from_compiled_model(
            self.model,
            super::ir::OriginalInputDomain { lower, upper },
            policy,
            options,
            budget,
            at,
        )
    }

    /// Positive convenience entry. Call `prepare_declared_domain` when an
    /// unresolved root must retain its construction for subsequent search.
    #[allow(clippy::too_many_arguments)]
    pub fn prove_declared_domain(
        self,
        lower: &[f64],
        upper: &[f64],
        target: usize,
        ordered_classes: &[usize],
        policy: RootPrecisionRuntimePolicy,
        options: PreparedRootProofOptions,
        budget: &WorkBudget,
        requested: Option<Instant>,
    ) -> Result<PreparedRootProof> {
        self.checkpoint(budget, requested)?;
        let at = self.deadline(requested)?;
        PreparedRootProof::from_compiled_model(
            self.model,
            super::ir::OriginalInputDomain { lower, upper },
            target,
            ordered_classes,
            policy,
            options,
            budget,
            at,
        )
    }
}

#[cfg(test)]
#[path = "cli_compilation_tests.rs"]
mod tests;
