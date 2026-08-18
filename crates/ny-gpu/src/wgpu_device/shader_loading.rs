// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! THE single seam through which every WgpuDevice compute shader module is
//! created (#rung3-denorm-chase, production arm).
//!
//! Default (env unset or `auto`): use the DenormPreserve passthrough on a device
//! that granted `PASSTHROUGH_SHADERS`, otherwise use plain
//! `create_shader_module`. `NY_GPU_DENORM_PRESERVE=0|1` supplies exact A/B
//! overrides; forced `1` refuses device construction when the capability is
//! absent. On the passthrough route the module is compiled WGSL -> naga SPIR-V,
//! injected with the DenormPreserve float controls, and loaded via passthrough
//! (`ny-gpu-passthrough`, the one workspace crate allowed to make that unsafe
//! call). A module failure falls back to the plain path with a once-per-process
//! warning for non-EFT work and permanently poisons this process's
//! DenormPreserve qualification. The cached rung/EFT gates dynamically observe
//! that poison, and EFT dispatch sites check it after lazy pipeline creation,
//! so a probe cannot attest a different loading path from production.
//!
//! WHY ONE SEAM AND NOT PER-KERNEL OPT-IN: the authority ladder's probes and
//! the production kernels MUST load through the same path, or a passing rung 3
//! attests a configuration production does not run (the #u2b class of hole).
//! Routing every `WgpuDevice` module creation here makes "the ladder measured
//! it" and "production runs it" the same loading-path statement.
//!
//! PER-DEVICE THREADING (#flush-charge admission-config): the seam's loading
//! decision is the CONSTRUCTING DEVICE'S resolved policy, passed explicitly by
//! every call site (`WgpuDevice::denorm_preserve_enabled`), never ambient
//! process state. Env-resolved devices behave exactly as before (their resolved
//! selection is additionally published through
//! [`install_denorm_preserve_selection`], so two env devices with conflicting
//! resolutions still refuse). The charged-flush constructors instead build
//! their device through [`resolve_denorm_preserve_forced_disabled`]: every
//! module is the plain-WGSL (flushing) configuration the oracle charges model,
//! regardless of ambient env — with the one exception that an explicit
//! `NY_GPU_DENORM_PRESERVE=1` pin refuses (env wins, per the repo-wide
//! precedence rule). Modules and pipelines are cached per-`WgpuDevice` only
//! (struct fields / per-device `OnceLock`s — there is no process-global module
//! cache), so a forced-Disabled device can never alias a module built under a
//! different policy.
//!
//! MEASURED BASIS (2026-08-10, GB10/Vulkan, tests/denorm_preserve_probe.rs):
//! plain WGSL path flushes subnormals (rung 3 false); the injected module
//! preserves core add/multiply subnormal lanes bit-exactly, but direct
//! `fma(subnormal, large, 0)` still DAZ-zeroes its operand even when the expected
//! result is normal. Production EFT primary products therefore use the
//! qualified core multiply; the remaining FMA residual forms either
//! conservatively over-charge operand DAZ or use the always-on rung-3 floor.
//! naga 29.0.3 emits no float-control modes.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::OnceLock;

use ny_core::{NyError, Result};

const DENORM_PRESERVE_ENV: &str = "NY_GPU_DENORM_PRESERVE";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DenormPreservePolicy {
    Auto,
    Disabled,
    Required,
    /// #flush-charge admission-config: PROGRAMMATIC per-device force of the
    /// plain-WGSL (flushing) loading path, used by the charged-flush
    /// constructors. Never parsed from env; resolved only through
    /// [`resolve_denorm_preserve_forced_disabled`], which refuses when the
    /// user explicitly pinned `NY_GPU_DENORM_PRESERVE=1` (env wins).
    ForcedDisabled,
}

impl DenormPreservePolicy {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Disabled => "override-disabled",
            Self::Required => "override-required",
            Self::ForcedDisabled => "forced-disabled",
        }
    }
}

fn parse_denorm_preserve_policy(raw: Option<&std::ffi::OsStr>) -> Result<DenormPreservePolicy> {
    match raw.map(std::ffi::OsStr::to_str) {
        None => Ok(DenormPreservePolicy::Auto),
        Some(Some("auto")) => Ok(DenormPreservePolicy::Auto),
        Some(Some("0")) => Ok(DenormPreservePolicy::Disabled),
        Some(Some("1")) => Ok(DenormPreservePolicy::Required),
        Some(_) => Err(NyError::InvalidSpec(format!(
            "{DENORM_PRESERVE_ENV} must be exactly auto, 0, or 1"
        ))),
    }
}

pub(crate) fn resolve_denorm_preserve(
    passthrough_supported: bool,
) -> Result<(DenormPreservePolicy, bool)> {
    let policy = parse_denorm_preserve_policy(std::env::var_os(DENORM_PRESERVE_ENV).as_deref())?;
    match policy {
        DenormPreservePolicy::Auto => Ok((policy, passthrough_supported)),
        DenormPreservePolicy::Disabled => Ok((policy, false)),
        DenormPreservePolicy::Required if passthrough_supported => Ok((policy, true)),
        DenormPreservePolicy::Required => Err(NyError::UnsupportedConfiguration(format!(
            "{DENORM_PRESERVE_ENV}=1 requires WGPU PASSTHROUGH_SHADERS support"
        ))),
        // The parser never produces the programmatic policy; fail closed
        // rather than silently treating it as an env resolution.
        DenormPreservePolicy::ForcedDisabled => Err(NyError::InternalError(format!(
            "DenormPreserve ForcedDisabled is a programmatic device policy and \
             cannot resolve from {DENORM_PRESERVE_ENV}"
        ))),
    }
}

/// #flush-charge admission-config: resolve the loading profile for a device
/// that FORCES the plain-WGSL (flushing) module configuration programmatically
/// — the configuration the oracle flush charges model. Pure on `raw` so the
/// env-precedence contract is unit-testable without touching process env.
///
/// Env still wins, per the repo-wide precedence rule: an explicit
/// `NY_GPU_DENORM_PRESERVE=1` pins the DenormPreserve passthrough path, which
/// the charges cannot cover, so it refuses with a typed message. Unset,
/// `auto`, and `0` all yield the forced plain-WGSL device.
pub(crate) fn resolve_denorm_preserve_forced_disabled_from(
    raw: Option<&std::ffi::OsStr>,
) -> Result<(DenormPreservePolicy, bool)> {
    match parse_denorm_preserve_policy(raw)? {
        DenormPreservePolicy::Required => Err(NyError::UnsupportedConfiguration(format!(
            "charged-flush qualification requires the plain-WGSL (flushing) shader \
             configuration its oracle charges model, but {DENORM_PRESERVE_ENV}=1 \
             explicitly pins the DenormPreserve passthrough loading path; refusing \
             (env wins — unset it, or set `auto`/`0`, to admit the charged device)"
        ))),
        _ => Ok((DenormPreservePolicy::ForcedDisabled, false)),
    }
}

/// Env-reading wrapper over [`resolve_denorm_preserve_forced_disabled_from`].
pub(crate) fn resolve_denorm_preserve_forced_disabled() -> Result<(DenormPreservePolicy, bool)> {
    resolve_denorm_preserve_forced_disabled_from(std::env::var_os(DENORM_PRESERVE_ENV).as_deref())
}

static DENORM_PRESERVE_ENABLED: OnceLock<bool> = OnceLock::new();

/// Publish an ENV-RESOLVED device's choice once. Two env-resolved devices with
/// conflicting resolutions refuse instead of silently authenticating different
/// code (unchanged behavior). Forced-Disabled (charged) devices deliberately
/// do NOT install: their loading path is threaded per-device through
/// [`create_compute_module`], every one of their modules is plain WGSL by
/// construction, and their ladder attests exactly that same per-device flag —
/// so coexisting with an env passthrough device mixes nothing.
pub(crate) fn install_denorm_preserve_selection(enabled: bool) -> Result<()> {
    match DENORM_PRESERVE_ENABLED.set(enabled) {
        Ok(()) => Ok(()),
        Err(_) if DENORM_PRESERVE_ENABLED.get().copied() == Some(enabled) => Ok(()),
        Err(_) => Err(NyError::UnsupportedConfiguration(
            "WGPU DenormPreserve selection conflicts with an existing process device".to_string(),
        )),
    }
}

static PASSTHROUGH_COUNT: AtomicUsize = AtomicUsize::new(0);
static FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);
/// Sticky process-wide poison: once any requested passthrough module falls
/// back, the probe no longer authenticates a uniform production loading path.
/// Never reset — a later success cannot make the already-created plain module
/// DenormPreserve-capable.
struct DenormPreserveFallback {
    occurred: AtomicBool,
}

impl DenormPreserveFallback {
    const fn new() -> Self {
        Self {
            occurred: AtomicBool::new(false),
        }
    }

    fn record(&self) {
        self.occurred.store(true, Ordering::Release);
    }

    fn contract_intact(&self, requested: bool) -> bool {
        !requested || !self.occurred.load(Ordering::Acquire)
    }
}

static DENORM_PRESERVE_FALLBACK: DenormPreserveFallback = DenormPreserveFallback::new();

#[cfg(test)]
const fn denorm_preserve_contract_intact_from(requested: bool, fallback_occurred: bool) -> bool {
    !requested || !fallback_occurred
}

/// Does every requested shader module created so far share the attested
/// DenormPreserve loading path, for a device that resolved `requested`
/// (`WgpuDevice::denorm_preserve_enabled`)?
///
/// This is deliberately dynamic rather than cached in the adapter probes: a
/// lazy production pipeline may be created after a probe has passed. Any
/// passthrough fallback permanently turns later cached authorization reads
/// into refusal for EVERY device that requested the passthrough path. A
/// device that never requested it (`requested == false`) creates plain-WGSL
/// modules only, so the poison cannot make its probes attest a path its
/// production modules do not run.
pub(crate) fn denorm_preserve_contract_intact_for(requested: bool) -> bool {
    DENORM_PRESERVE_FALLBACK.contract_intact(requested)
}

/// Create a compute shader module through the seam. `denorm_preserve` is the
/// CONSTRUCTING DEVICE's resolved loading profile
/// (`WgpuDevice::denorm_preserve_enabled`), threaded explicitly so a device's
/// modules can never be created under another device's (or ambient process)
/// policy.
pub(crate) fn create_compute_module(
    device: &wgpu::Device,
    denorm_preserve: bool,
    label: &str,
    wgsl: &str,
) -> wgpu::ShaderModule {
    if denorm_preserve {
        match ny_gpu_passthrough::create_denorm_preserving_module(device, label, wgsl) {
            Ok(module) => {
                let n = PASSTHROUGH_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                if n <= 3 {
                    // `tracing::debug!`, not `eprintln!`: this is the SUCCESS
                    // path, and an unconditional print bypasses the subscriber
                    // entirely — no log level, no `--json` mode and no `-q`
                    // could silence it, which is what put three lines of
                    // chatter on stderr and broke the empty-stderr contract
                    // (#395). The FAILURE arm below keeps its `eprintln!` on
                    // purpose: that one must be visible even where no
                    // subscriber is installed, as its comment records.
                    tracing::debug!(
                        target: "ny_gpu::wgpu",
                        %label,
                        module_index = n,
                        "denorm-preserve module loaded via passthrough"
                    );
                }
                return module;
            }
            Err(reason) => {
                // Soundness boundary: the probe may already have passed via a
                // DenormPreserve module. Falling back just this later module to
                // plain WGSL would make that cached evidence inapplicable to
                // production. Poison the process before returning the module.
                DENORM_PRESERVE_FALLBACK.record();
                if !FALLBACK_WARNED.swap(true, Ordering::Relaxed) {
                    // The env is a diagnostic opt-in, so the fallback reason must
                    // be visible even where no tracing subscriber is installed
                    // (the ladder report test) — same rationale as the ladder's
                    // own eprintln-based report.
                    eprintln!(
                        "[denorm-preserve] passthrough FAILED for '{label}' ({reason}); \
                         falling back to plain WGSL (fail-closed)"
                    );
                    tracing::warn!(
                        target: "ny_gpu::wgpu",
                        %label,
                        %reason,
                        "NY_GPU_DENORM_PRESERVE requested but passthrough failed; \
                         falling back to plain WGSL for this non-authoritative \
                         module and permanently poisoning the process's rung-3/EFT \
                         qualification. Cached authorization reads and lazy EFT \
                         dispatches now refuse (fail-closed)."
                    );
                }
            }
        }
    }
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        denorm_preserve_contract_intact_from, parse_denorm_preserve_policy,
        resolve_denorm_preserve_forced_disabled_from, DenormPreserveFallback, DenormPreservePolicy,
        DENORM_PRESERVE_ENV,
    };

    /// #flush-charge Fix 1 pin: the forced-Disabled resolution honours env
    /// precedence exactly — unset/`auto`/`0` all force the plain-WGSL device,
    /// an explicit `1` pin refuses with a typed message naming the env, and a
    /// malformed value still refuses at parse.
    #[test]
    fn forced_disabled_resolution_honours_env_precedence() {
        for raw in [None, Some("auto"), Some("0")] {
            let raw = raw.map(std::ffi::OsStr::new);
            let (policy, enabled) = resolve_denorm_preserve_forced_disabled_from(raw)
                .expect("unset/auto/0 must admit the forced plain-WGSL device");
            assert_eq!(policy, DenormPreservePolicy::ForcedDisabled);
            assert_eq!(policy.name(), "forced-disabled");
            assert!(!enabled, "the forced device must never request passthrough");
        }
        let error = resolve_denorm_preserve_forced_disabled_from(Some(std::ffi::OsStr::new("1")))
            .expect_err("an explicit =1 pin must refuse: env wins");
        let message = error.to_string();
        assert!(message.contains(DENORM_PRESERVE_ENV), "message: {message}");
        assert!(message.contains("charged-flush"), "message: {message}");
        assert!(
            resolve_denorm_preserve_forced_disabled_from(Some(std::ffi::OsStr::new("true")))
                .is_err(),
            "malformed env must refuse at parse, not force anything"
        );
    }

    #[test]
    fn denorm_preserve_policy_parser_is_exact_and_fail_closed() {
        assert_eq!(
            parse_denorm_preserve_policy(None).expect("unset is auto"),
            DenormPreservePolicy::Auto
        );
        assert_eq!(
            parse_denorm_preserve_policy(Some(std::ffi::OsStr::new("auto"))).expect("exact auto"),
            DenormPreservePolicy::Auto
        );
        assert_eq!(
            parse_denorm_preserve_policy(Some(std::ffi::OsStr::new("0"))).expect("exact zero"),
            DenormPreservePolicy::Disabled
        );
        assert_eq!(
            parse_denorm_preserve_policy(Some(std::ffi::OsStr::new("1"))).expect("exact one"),
            DenormPreservePolicy::Required
        );
        for invalid in ["", "true", "AUTO", " 1", "1 "] {
            let error = parse_denorm_preserve_policy(Some(std::ffi::OsStr::new(invalid)))
                .expect_err("malformed policy must refuse");
            assert!(error.to_string().contains(DENORM_PRESERVE_ENV));
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let non_unicode = std::ffi::OsString::from_vec(vec![0xff, b'1']);
            assert!(parse_denorm_preserve_policy(Some(&non_unicode)).is_err());
        }
    }

    #[test]
    fn requested_fallback_permanently_invalidates_the_contract() {
        assert!(denorm_preserve_contract_intact_from(false, false));
        assert!(denorm_preserve_contract_intact_from(false, true));
        assert!(denorm_preserve_contract_intact_from(true, false));
        assert!(!denorm_preserve_contract_intact_from(true, true));
    }

    #[test]
    fn fallback_after_a_cached_probe_closes_later_authorization_reads() {
        let fallback = DenormPreserveFallback::new();
        let cached_probe_passed = true;
        assert!(cached_probe_passed && fallback.contract_intact(true));

        // Model a lazy production module failing after the probe cached true.
        fallback.record();
        assert!(
            !(cached_probe_passed && fallback.contract_intact(true)),
            "a post-probe module fallback must invalidate cached authorization"
        );
        fallback.record();
        assert!(
            !fallback.contract_intact(true),
            "the fallback poison must remain sticky after later events"
        );
    }
}
