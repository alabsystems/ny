// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Original-source declarations for the prepared support program.
//!
//! These requests carry no numerical proof or product-delivery authority.
//! Only the compiler can construct the immutable model owner; numerical root
//! and prefix transactions establish their own original-domain induction.

use std::time::Instant;

use ny_core::{NyError, Result};

use crate::certify::work::{LeaseClass, WorkBudget, WorkRefusal};
use crate::margin_row::root::RetainCfg;
use crate::margin_row::RootPrecisionRuntimePolicy;

/// The compiler preserves stored binary64 parameters and their explicit fold
/// uncertainty. This identifies the mathematical model, not an inference
/// kernel's execution or a claim that folded parameters are exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::margin_row) enum OriginalParameterContract {
    Binary64WithFoldUncertainty,
}

/// Declared domain request. Supplied endpoints do not certify a tensor box;
/// the original-root transaction validates, leases and owns its actual input.
pub(super) struct OriginalInputDomain<'a> {
    pub(super) lower: &'a [f64],
    pub(super) upper: &'a [f64],
}

/// Required root products. A declaration is not evidence that a source has
/// supplied them; missing clipping/retention/precision coverage still refuses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RootProductDemand {
    pub(super) precision: RootPrecisionRuntimePolicy,
    pub(super) clipping: bool,
    pub(super) retention: Option<RetainCfg>,
}

/// Shared final-work checkpoint. All callers spend the original ledger and
/// cancellation token; matching limits on another budget confer no authority.
pub(super) fn checkpoint(budget: &WorkBudget, deadline: Option<Instant>) -> Result<()> {
    budget
        .checkpoint_for(LeaseClass::Fallback)
        .map_err(refusal)?;
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(
            "original support program inherited deadline".into(),
        ));
    }
    Ok(())
}

pub(super) fn earliest(left: Option<Instant>, right: Option<Instant>) -> Option<Instant> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (left, right) => left.or(right),
    }
}

pub(super) fn refusal(error: WorkRefusal) -> NyError {
    if error == WorkRefusal::Deadline {
        NyError::DeadlineExceeded("original support program resource deadline".into())
    } else {
        NyError::UnsupportedConfiguration(format!(
            "original support program resource refusal: {error}"
        ))
    }
}

pub(super) fn invalid(message: &str) -> NyError {
    NyError::InvalidSpec(format!("original support program: {message}"))
}
