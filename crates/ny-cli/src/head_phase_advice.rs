// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bootstrap for bounded, untrusted AY phase and global-functional advice.
//!
//! The helper entry precedes normal CLI initialization. Factory installation
//! captures no provider or executable; only an armed source-owned complete
//! cover may ask propagation to construct the provider and launch its child.

use std::ffi::{OsStr, OsString};
use std::sync::Arc;

use anyhow::Result;
use ny_core::head_phase_advice::HeadPhaseAdvice;

pub(crate) const HELPER_COMMAND: &str = ny_mip::head_phase_ay::HELPER_COMMAND;
pub(crate) const FUNCTIONAL_HELPER_COMMAND: &str = ny_mip::head_phase_ay::FUNCTIONAL_HELPER_COMMAND;
pub(crate) const PHASE_COVER_HELPER_COMMAND: &str =
    ny_mip::head_phase_ay::PHASE_COVER_HELPER_COMMAND;

/// Recognize the private transport before platform-stack/GPU/logger setup.
/// Explicit command help continues through Clap; transport accepts no flags,
/// model paths, or argument-supplied payloads.
pub(crate) fn helper_requested(mut args: impl Iterator<Item = OsString>) -> Result<bool> {
    helper_requested_for(&mut args, HELPER_COMMAND)
}

pub(crate) fn functional_helper_requested(
    mut args: impl Iterator<Item = OsString>,
) -> Result<bool> {
    helper_requested_for(&mut args, FUNCTIONAL_HELPER_COMMAND)
}

pub(crate) fn phase_cover_helper_requested(
    mut args: impl Iterator<Item = OsString>,
) -> Result<bool> {
    helper_requested_for(&mut args, PHASE_COVER_HELPER_COMMAND)
}

fn helper_requested_for(args: &mut impl Iterator<Item = OsString>, command: &str) -> Result<bool> {
    if args.next().as_deref() != Some(OsStr::new(command)) {
        return Ok(false);
    }
    match (args.next(), args.next()) {
        (None, None) => Ok(true),
        (Some(flag), None)
            if flag.as_os_str() == OsStr::new("--help") || flag.as_os_str() == OsStr::new("-h") =>
        {
            Ok(false)
        }
        _ => anyhow::bail!("{command} accepts only its bounded stdin JSON request"),
    }
}

pub(crate) fn run_helper() -> Result<()> {
    // The ny-mip service owns JSON caps, the single-threaded pinned float AY
    // call and the bounded hint response. No ordinary NY solver is entered.
    ny_mip::head_phase_ay::run_head_phase_ay_helper().map_err(anyhow::Error::msg)
}

pub(crate) fn run_functional_helper() -> Result<()> {
    ny_mip::head_phase_ay::run_head_functional_ay_helper().map_err(anyhow::Error::msg)
}

pub(crate) fn run_phase_cover_helper() -> Result<()> {
    ny_mip::head_phase_ay::run_head_phase_cover_ay_helper().map_err(anyhow::Error::msg)
}

fn construct_provider() -> Option<Arc<dyn HeadPhaseAdvice>> {
    let executable = std::env::current_exe().ok()?;
    let provider = ny_mip::head_phase_ay::HeadPhaseAyAdvice::new(executable).ok()?;
    Some(Arc::new(provider))
}

pub(crate) fn install_lazy_factory() {
    // First registration wins. The propagation seam invokes this function
    // pointer only after armed typed head-group policy, a complete native
    // cover and remaining original/local deadline admission. Dark startup
    // pays no current_exe lookup, source export, AY construction or spawn.
    ny_propagate::margin_row::register_head_phase_advice_factory(construct_provider);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requested(args: &[&str]) -> Result<bool> {
        helper_requested(args.iter().map(|arg| OsString::from(*arg)))
    }

    #[test]
    fn helper_transport_has_an_exact_early_entry_without_payload_arguments() {
        assert!(requested(&[HELPER_COMMAND]).unwrap());
        assert!(!requested(&[]).unwrap());
        assert!(!requested(&["verify", "model.onnx"]).unwrap());
        assert!(!requested(&[HELPER_COMMAND, "--help"]).unwrap());
        assert!(!requested(&[HELPER_COMMAND, "-h"]).unwrap());
        for extra in ["model.onnx", "{}", "--backend", "--timeout"] {
            assert!(requested(&[HELPER_COMMAND, extra]).is_err());
        }
        assert!(requested(&[HELPER_COMMAND, "--help", "extra"]).is_err());
    }

    #[test]
    fn functional_transport_has_its_own_exact_entry_and_preserves_v1() {
        let functional = |args: &[&str]| {
            functional_helper_requested(args.iter().map(|arg| OsString::from(*arg)))
        };
        assert!(functional(&[FUNCTIONAL_HELPER_COMMAND]).unwrap());
        assert!(!requested(&[FUNCTIONAL_HELPER_COMMAND]).unwrap());
        assert!(!functional(&[HELPER_COMMAND]).unwrap());
        assert!(!functional(&[]).unwrap());
        assert!(!functional(&[FUNCTIONAL_HELPER_COMMAND, "--help"]).unwrap());
        assert!(!functional(&[FUNCTIONAL_HELPER_COMMAND, "-h"]).unwrap());
        for extra in ["model.onnx", "{}", "--backend", "--timeout"] {
            assert!(functional(&[FUNCTIONAL_HELPER_COMMAND, extra]).is_err());
        }
        assert!(functional(&[FUNCTIONAL_HELPER_COMMAND, "--help", "extra"]).is_err());
    }

    #[test]
    fn phase_cover_transport_has_exact_entry_and_preserves_legacy_routes() {
        let cover = |args: &[&str]| {
            phase_cover_helper_requested(args.iter().map(|arg| OsString::from(*arg)))
        };
        assert!(cover(&[PHASE_COVER_HELPER_COMMAND]).unwrap());
        assert!(!cover(&[HELPER_COMMAND]).unwrap());
        assert!(!cover(&[FUNCTIONAL_HELPER_COMMAND]).unwrap());
        assert!(!requested(&[PHASE_COVER_HELPER_COMMAND]).unwrap());
        assert!(!cover(&[PHASE_COVER_HELPER_COMMAND, "--help"]).unwrap());
        assert!(cover(&[PHASE_COVER_HELPER_COMMAND, "{}"]).is_err());
        assert!(cover(&[PHASE_COVER_HELPER_COMMAND, "--timeout", "1"]).is_err());
    }
}
