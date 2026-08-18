// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared CPU dense-materialization policy for CROWN paths (#3515, #3550).
//!
//! Provides upfront memory estimates and budget checks for both sequential
//! (unbatched) and batched dense identity/densification allocations. All CPU
//! dense materialization shares a single budget: `NY_DENSE_BUDGET_MB`.

use ny_core::{checked_shape_product, NyError};

#[cfg(target_os = "linux")]
use std::ffi::OsString;
#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::os::unix::ffi::OsStringExt;
#[cfg(target_os = "linux")]
use std::os::unix::fs::FileExt;
#[cfg(target_os = "linux")]
use std::path::{Component, Path, PathBuf};
#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicU32, Ordering};
#[cfg(target_os = "linux")]
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Default CPU Dense-materialization budget for sequential CROWN in MiB.
pub(crate) const DEFAULT_CROWN_DENSE_BUDGET_MB: usize = 2048;

const CROWN_DENSE_BUDGET_ENV: &str = "NY_DENSE_BUDGET_MB";

/// Upfront estimate for a Dense coefficient-pair materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseMaterializationEstimate {
    pub site: &'static str,
    pub rows: usize,
    pub cols: usize,
    pub required_bytes: usize,
}

impl DenseMaterializationEstimate {
    /// Create a Dense-materialization estimate, saturating to `usize::MAX`
    /// when the exact byte count overflows.
    pub(crate) fn new(site: &'static str, rows: usize, cols: usize) -> Self {
        Self {
            site,
            rows,
            cols,
            required_bytes: dense_pair_bytes(rows, cols).unwrap_or(usize::MAX),
        }
    }

    /// Whether this estimate exceeds the configured budget.
    pub(crate) fn exceeds_budget(self, budget_bytes: usize) -> bool {
        self.required_bytes > budget_bytes
    }

    /// Render a consistent fallback detail string for diagnostics.
    pub(crate) fn budget_exceeded_details(self, budget_bytes: usize) -> String {
        format!(
            "dense materialization at {} requires {} bytes for {}x{} coefficient pair; budget is {} bytes",
            self.site, self.required_bytes, self.rows, self.cols, budget_bytes
        )
    }
}

/// Share of the observed host memory the Dense budget may claim
/// (#adaptive-dense-budget). The observation is already conservative -- on a
/// host without a per-moment "available" probe it is half of physical RAM -- so
/// halving it again keeps a single materialization to roughly a quarter of the
/// machine.
const ADAPTIVE_DENSE_BUDGET_SHARE_DIVISOR: u64 = 2;

/// Share of LIVE cgroup/RLIMIT headroom one advertised Dense coefficient-pair
/// estimate may claim.
///
/// This is deliberately stricter than the host-wide heuristic above.  The
/// 7-D Patches -> Dense conversion can hold eight same-sized matrices while
/// its admission estimate names only the lower/upper f32 pair (two matrices):
/// retained coefficients + first error side + f64 count/accumulator scratch +
/// the second error side peak at four times the advertised pair.  Dividing
/// kernel-enforced headroom by eight leaves half of that headroom after the
/// largest presently known local peak, for graph state, allocator/GEMM
/// overhead, and admissions elsewhere in the process.
const PROCESS_ENVELOPE_DENSE_BUDGET_SHARE_DIVISOR: u64 = 8;

/// Hard ceiling on the ADAPTIVE budget, regardless of host size. An explicit
/// `NY_DENSE_BUDGET_MB` is not subject to it -- an operator stating a number is
/// making an assertion about their machine.
const MAX_ADAPTIVE_DENSE_BUDGET_MB: usize = 12 * 1024;

/// Reclaimable-now figure below which the host is reported as critically low
/// (#dense-budget-valve, withdrawn as a clamp 2026-08-01 -- DIAGNOSTIC ONLY).
///
/// At the instant of `JetsamEvent-2026-07-30-210002` the host had 4668 pages
/// free, 541 speculative and 0 purgeable -- 79 MiB reclaimable-now. A single
/// `ny` run legitimately drives this below 1.5 GiB on a 24 GiB host, which is
/// exactly why this must not gate anything: the process reporting the condition
/// is the one causing it.
#[cfg(not(target_os = "linux"))]
const LOW_MEMORY_REPORT_THRESHOLD_BYTES: u64 = 1536 * 1024 * 1024;

/// How long a live host-availability reading may be reused (#dense-budget-valve).
///
/// Dense admission sits in hot loops -- `backward_step.rs` alone calls
/// `cpu_crown_dense_budget_bytes()` four times in one function -- and the
/// pre-existing macOS probe spawned a `sysctl` CHILD PROCESS on every one of
/// those calls, uncached.
///
/// Deliberately ten times the Linux `MEMORY_PROBE_CACHE_TTL`, because the two
/// probes do not cost the same thing. Linux reads `MemAvailable` from an
/// already-open handle; macOS has to fork `vm_stat`, measured at 1.34 ms a call
/// on this host. At a 100 ms TTL a 2400 s run would spawn up to 24,000 of them
/// (~32 s of overhead, ~1.3% of the budget); at 1 s it is 2,400 (~3 s, ~0.13%).
///
/// One second costs nothing in responsiveness: the valve guards a condition that
/// takes seconds to develop -- allocating the multi-GiB buffers that trigger it
/// is not instantaneous -- so it does not need sub-second resolution.
#[cfg(not(target_os = "linux"))]
const LIVE_MEMORY_PROBE_TTL: Duration = Duration::from_secs(1);

/// Get the sequential CPU Dense-materialization budget in bytes (#3515).
///
/// Priority: `NY_DENSE_BUDGET_MB` env var > host-adaptive share > 2 GiB floor,
/// then the live cgroup/RLIMIT envelope clamps every path.  An operator value
/// overrides a heuristic; it cannot override a limit the kernel will enforce.
pub fn cpu_crown_dense_budget_bytes() -> usize {
    let requested = explicit_cpu_crown_dense_budget_bytes()
        .unwrap_or_else(adaptive_cpu_crown_dense_budget_bytes);
    clamp_dense_budget_to_process_envelope(requested, process_memory_headroom_bytes())
}

fn clamp_dense_budget_to_process_envelope(
    requested_bytes: usize,
    process_headroom_bytes: Option<u64>,
) -> usize {
    process_headroom_bytes.map_or(requested_bytes, |headroom| {
        let share = usize::try_from(headroom / PROCESS_ENVELOPE_DENSE_BUDGET_SHARE_DIVISOR)
            .unwrap_or(usize::MAX);
        requested_bytes.min(share)
    })
}

/// Size the Dense budget from MEASURED host memory instead of a fixed constant
/// (#adaptive-dense-budget).
///
/// The 2 GiB default was a hardcoded number standing in for a measurement, and
/// it is the budget that actually does the refusing: every CROWN Dense
/// materialization gate reads this value, and on a 24 GiB host every refusal
/// logged `budget is 2147483648 bytes` no matter what else was configured.
/// Measured on yolo_2023 (2026-07-29), raising it to 12 GiB improved the only
/// output row that can decide the instance by 5.45x (Y[269] -42784.91 ->
/// -7847.13) with host memory pressure never leaving 80% free -- an improvement
/// no operator would ever have reached, because reaching it required knowing to
/// set an environment variable.
///
/// Three bounds still apply, in this order:
///   * the FLOOR is the historical 2 GiB default, so no host can regress;
///   * the SHARE is half of an already-conservative observation;
///   * `process_memory_headroom_bytes` (cgroup / RLIMIT) is KERNEL-ENFORCED and
///     clamps the result to one eighth of live headroom -- exceeding it is a
///     real OOM, not a policy choice.  The public entry point reapplies that
///     clamp to explicit operator budgets as well.
///
/// With neither an observable host-memory value nor a process-envelope value,
/// this returns exactly the historical default.
fn adaptive_cpu_crown_dense_budget_bytes() -> usize {
    #[cfg(target_os = "linux")]
    {
        let snapshot = linux_memory_probe_snapshot();
        adaptive_cpu_crown_dense_budget_from(
            observed_host_memory_bytes_from_snapshot(&snapshot),
            process_memory_headroom_from_snapshot(&snapshot),
            // Linux already reads `MemAvailable` as its host observation, so
            // the observation IS the live figure; no separate valve is needed.
            None,
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let observed = observed_host_memory_bytes();
        let headroom = process_memory_headroom_bytes();
        let live = live_available_memory_bytes();
        let budget = adaptive_cpu_crown_dense_budget_from(observed, headroom, live);
        // #dense-budget-valve: REPORT a critically low host once per process.
        // This deliberately does not change `budget` -- see the policy function's
        // doc comment for why the clamp was withdrawn.
        if let Some(available) = live {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if available < LOW_MEMORY_REPORT_THRESHOLD_BYTES
                && !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                tracing::warn!(
                    "#dense-budget-valve: host has {available} bytes reclaimable-now, below \
                     {LOW_MEMORY_REPORT_THRESHOLD_BYTES}. The CROWN dense budget is UNCHANGED \
                     at {budget} -- this is a diagnostic, not a throttle. Concurrent heavy runs \
                     on this host are the usual cause. (Logged once per process.)"
                );
            }
        }
        budget
    }
}

/// Pure adaptive-budget policy, separated from the host probes so the
/// kernel-envelope precedence is regression-testable on every platform.
///
/// # `live_available_bytes` -- the low-memory safety valve (#dense-budget-valve)
///
/// `observed_host_bytes` is a STATIC figure on every non-Linux platform: it is
/// `hw.memsize / 2`, i.e. installed RAM, which reads the same whether the
/// machine has 20 GiB free or 60 MiB free. The kernel-envelope clamp that the
/// comment below calls the backstop is `None` on those platforms (there is no
/// cgroup), so before this valve existed macOS had NO term in this function
/// that could respond to the machine actually being full.
///
/// Measured, 2026-07-31 on a 24 GiB M4 Pro: one `ny` run of
/// `yolo_2023/TinyYOLO_prop_000001_eps_1_255` peaks at 14.17 GiB RSS and drives
/// host free memory to 65 MiB -- twice -- while this function keeps returning
/// 6 GiB. On 2026-07-30 that same shape, several copies of it running
/// concurrently in agent worktrees, took the machine into a jetsam kill
/// (`JetsamEvent-2026-07-30-210002`: `ny` at 44 GiB RSS, 4668 pages free).
///
/// # WITHDRAWN 2026-08-01: the valve no longer clamps, and here is why
///
/// This function briefly clamped the budget to 256 MiB when reclaimable-now fell
/// below 1.5 GiB. That was **wrong and is removed**. Two independent reasons,
/// both measured:
///
/// 1. **No benefit.** The premise was that the dense budget governs the
///    process footprint. It does not: cutting `NY_DENSE_BUDGET_MB` 4x moves peak
///    RSS by 5%, and cutting `NY_GPU_MEMORY_BUDGET_MB` 16x moves it 5%. The
///    ~14 GiB is a genuine in-use working set allocated through mimalloc arenas.
///    Clamping this number cannot prevent the OOM it was written to prevent.
/// 2. **Real harm.** The "engages only in a danger band ordinary operation does
///    not enter" claim was verified in unit tests and never end-to-end, and it
///    is false. `ny` is itself the process driving reclaimable memory down, so
///    on a 24 GiB host the valve fires *against the very run it is protecting*,
///    mid-proof. Observed on `yolo_2023` 2026-08-01: it clamped
///    6442450944 -> 268435456, after which Conv2d targets kept their IBP bound,
///    the f64 recompute degraded rows to +/-inf, and the instance returned
///    `unknown`.
///
/// A guard with no demonstrated benefit and a demonstrated cost is not something
/// to re-tune, so the threshold is gone rather than lowered. What survives is
/// worth keeping on its own: `hw.memsize` is cached instead of forking `sysctl`
/// per admission, and a critically low host is reported once per process.
///
/// The lesson worth keeping: a memory guard inside the process that consumes the
/// memory needs its trigger measured END TO END, not in a unit test where the
/// process is not running.
///
/// `live_available_bytes` must be a RECLAIMABLE-NOW figure (free +
/// speculative + purgeable), NOT a Linux-style `MemAvailable` that counts
/// inactive pages. Checked against the jetsam report: at the instant of the
/// kill the MemAvailable-style sum still read 4.29 GiB -- it would not have
/// noticed the machine dying -- while reclaimable-now read 79 MiB.
///
/// An explicit `NY_DENSE_BUDGET_MB` bypasses the host-adaptive heuristic (an
/// operator stating a number is making an assertion about their machine), but
/// the public entry point still clamps it by the kernel-enforced process
/// envelope.
///
/// This is an operational guard, not a soundness mechanism: it changes only how
/// much memory a materialization may claim, never what is proved. A refused
/// materialization takes the existing documented fallback path.
fn adaptive_cpu_crown_dense_budget_from(
    observed_host_bytes: Option<u64>,
    process_headroom_bytes: Option<u64>,
    live_available_bytes: Option<u64>,
) -> usize {
    let floor = DEFAULT_CROWN_DENSE_BUDGET_MB.saturating_mul(1024 * 1024);
    let ceiling = MAX_ADAPTIVE_DENSE_BUDGET_MB.saturating_mul(1024 * 1024);
    let mut budget = observed_host_bytes.map_or(floor, |observed| {
        let share = observed / ADAPTIVE_DENSE_BUDGET_SHARE_DIVISOR;
        usize::try_from(share)
            .unwrap_or(usize::MAX)
            .clamp(floor, ceiling)
    });
    // Kernel-enforced envelope wins over the host-wide observation, and over the
    // floor: if the cgroup says there is less than 2 GiB left, taking 2 GiB is
    // not a conservative default, it is an OOM.
    if let Some(headroom) = process_headroom_bytes {
        let headroom_share =
            usize::try_from(headroom / PROCESS_ENVELOPE_DENSE_BUDGET_SHARE_DIVISOR)
                .unwrap_or(usize::MAX);
        budget = budget.min(headroom_share);
    }
    // #dense-budget-valve: same reasoning as the envelope above, but against a
    // LIVE host-wide reading rather than a per-process kernel limit. Kept as a
    // separate term so `process_memory_headroom_bytes`' "kernel-enforced"
    // contract stays literally true.
    // #dense-budget-valve WITHDRAWN 2026-08-01 -- see the doc comment above.
    // The live reading is still taken and still reported once per process,
    // because "the host is nearly out of memory" is worth saying. It no longer
    // CHANGES the budget.
    let _ = live_available_bytes;
    budget
}

#[cfg(test)]
mod adaptive_budget_tests {
    use super::*;

    #[test]
    fn missing_host_probe_still_honors_process_headroom() {
        let mib = 1024_u64 * 1024;
        assert_eq!(
            adaptive_cpu_crown_dense_budget_from(None, Some(512 * mib), None),
            64 * 1024 * 1024,
            "a missing host-wide probe must not bypass a kernel-enforced envelope"
        );
    }

    #[test]
    fn missing_host_and_process_probes_preserve_historical_floor() {
        assert_eq!(
            adaptive_cpu_crown_dense_budget_from(None, None, None),
            DEFAULT_CROWN_DENSE_BUDGET_MB * 1024 * 1024
        );
    }

    #[test]
    fn process_headroom_can_tighten_an_observed_host_budget_below_the_floor() {
        let gib = 1024_u64 * 1024 * 1024;
        assert_eq!(
            adaptive_cpu_crown_dense_budget_from(Some(128 * gib), Some(gib), None),
            128 * 1024 * 1024
        );
    }

    /// #dense-budget-valve WITHDRAWN: the live reading must never change the
    /// budget. Asserted across the whole range, including the reading recorded
    /// at the 2026-07-30 jetsam and a zero reading, because the withdrawn
    /// version clamped hardest exactly there -- and clamping there is what
    /// degraded a live proof to `unknown`.
    #[test]
    fn live_availability_never_changes_the_budget() {
        let gib = 1024_u64 * 1024 * 1024;
        let at_jetsam = (4668_u64 + 541) * 16384; // 79 MiB reclaimable-now
        let baseline = adaptive_cpu_crown_dense_budget_from(Some(12 * gib), None, None);
        assert_eq!(baseline, 6 * 1024 * 1024 * 1024);
        for reclaimable in [0, at_jetsam, 512 * 1024 * 1024, 2 * gib, 20 * gib] {
            assert_eq!(
                adaptive_cpu_crown_dense_budget_from(Some(12 * gib), None, Some(reclaimable)),
                baseline,
                "reclaimable={reclaimable}: the probe is diagnostic only and must not \
                 move the budget"
            );
        }
    }

    /// The kernel-enforced envelope is a different thing and still binds: it is
    /// a per-process limit the kernel will actually enforce, not a host-wide
    /// observation.
    #[test]
    fn kernel_envelope_still_binds_after_the_valve_withdrawal() {
        let gib = 1024_u64 * 1024 * 1024;
        assert_eq!(
            adaptive_cpu_crown_dense_budget_from(Some(12 * gib), Some(64 * 1024 * 1024), Some(0)),
            8 * 1024 * 1024,
            "a cgroup/RLIMIT envelope must survive, valve or no valve"
        );
    }

    #[test]
    fn explicit_budget_cannot_bypass_kernel_envelope() {
        let gib = 1024_u64 * 1024 * 1024;
        assert_eq!(
            clamp_dense_budget_to_process_envelope(12 * gib as usize, Some(4 * gib)),
            512 * 1024 * 1024
        );
        assert_eq!(
            clamp_dense_budget_to_process_envelope(64 * 1024 * 1024, Some(4 * gib)),
            64 * 1024 * 1024,
            "the envelope narrows an override but never inflates it"
        );
        assert_eq!(
            clamp_dense_budget_to_process_envelope(usize::MAX, Some(0)),
            0
        );
    }
}

#[cfg(all(test, not(target_os = "linux")))]
mod vm_stat_parse_tests {
    use super::*;

    /// Real `vm_stat` output from the 24 GiB M4 Pro this defect was measured on.
    const SAMPLE: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
Pages free:                                   184160.\n\
Pages active:                                 614765.\n\
Pages inactive:                               597542.\n\
Pages speculative:                             16310.\n\
Pages throttled:                                   0.\n\
Pages wired down:                             115640.\n\
Pages purgeable:                                3439.\n\
\"Translation faults\":                        7729185.\n";

    #[test]
    fn parses_reclaimable_pages_at_the_reported_page_size() {
        // free + purgeable + speculative, at 16 KiB pages.
        let expected = (184_160_u64 + 3_439 + 16_310) * 16384;
        assert_eq!(parse_vm_stat_available_bytes(SAMPLE), Some(expected));
    }

    /// The whole point of the metric: inactive pages must NOT be counted.
    /// Including them is what makes a dying host look healthy -- see the
    /// jetsam arithmetic in `adaptive_cpu_crown_dense_budget_from`.
    #[test]
    fn excludes_inactive_wired_and_active_pages() {
        let reclaimable = parse_vm_stat_available_bytes(SAMPLE).unwrap();
        let with_inactive = (184_160_u64 + 3_439 + 16_310 + 597_542) * 16384;
        assert!(
            reclaimable < with_inactive,
            "inactive pages must be excluded: {reclaimable} vs {with_inactive}"
        );
        assert_eq!(reclaimable + 597_542 * 16384, with_inactive);
    }

    #[test]
    fn unparseable_output_is_no_observation_not_zero() {
        assert_eq!(parse_vm_stat_available_bytes(""), None);
        assert_eq!(
            parse_vm_stat_available_bytes("Mach Virtual Memory Statistics: (page size of 0 bytes)"),
            None
        );
        assert_eq!(
            parse_vm_stat_available_bytes(
                "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nSomething else:  1.\n"
            ),
            None,
            "a wholesale format change must read as no observation"
        );
    }

    /// A future OS dropping one field must degrade to the fields that remain,
    /// not to `None` and not to a silent zero.
    #[test]
    fn missing_single_field_degrades_gracefully() {
        let partial = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
Pages free:                                   100.\n";
        assert_eq!(parse_vm_stat_available_bytes(partial), Some(100 * 16384));
    }
}

/// Host memory available to this process, in bytes, or `None` when no probe
/// exists on this platform (#adaptive-dense-budget).
///
/// Linux exposes a per-moment figure via `/proc/meminfo` `MemAvailable`. Other
/// platforms fall back to a conservative FRACTION of measured physical memory --
/// half, deliberately pessimistic since the rest of the machine may be busy --
/// which is still a measurement and strictly better than a fixed constant.
///
/// Dense admission sits in per-domain hot loops, where reopening procfs and
/// rediscovering the same cgroup mount thousands of times can cost more than
/// the proof itself. Linux keeps a short-lived snapshot of open handles but
/// reads `MemAvailable` from that handle at every call, so the value remains
/// live without repeated `openat`/`statx`/`close` traffic.
#[cfg(not(target_os = "linux"))]
fn observed_host_memory_bytes() -> Option<u64> {
    physical_memory_bytes().map(|total| total / 2)
}

/// Parse `MemAvailable:` (reported in kiB) out of `/proc/meminfo` contents.
///
/// Pure so it can be tested without the file. Returns `None` when the field is
/// absent or unparseable, which the caller treats as "no observation" rather
/// than as zero -- a zero would silently clamp the budget to the floor.
#[cfg(target_os = "linux")]
fn parse_meminfo_available_kib(contents: &str) -> Option<u64> {
    contents
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|kib| kib.checked_mul(1024))
        .filter(|bytes| *bytes > 0)
}

/// Total physical memory, where the platform exposes it cheaply.
///
/// Cached: `hw.memsize` is immutable for the life of the process, and this used
/// to spawn a `sysctl` child on every dense-admission check (#dense-budget-valve).
#[cfg(not(target_os = "linux"))]
fn physical_memory_bytes() -> Option<u64> {
    static PHYSICAL: OnceLock<Option<u64>> = OnceLock::new();
    *PHYSICAL.get_or_init(|| {
        if !cfg!(any(
            target_os = "macos",
            target_os = "ios",
            target_vendor = "apple"
        )) {
            return None;
        }
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let value = String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<u64>()
            .ok()?;
        (value > 0).then_some(value)
    })
}

/// Live host-wide RECLAIMABLE-NOW memory in bytes, or `None` where no probe
/// exists (#dense-budget-valve).
///
/// Deliberately NOT the macOS analogue of Linux's `MemAvailable`. INACTIVE pages
/// are excluded: under compressor pressure macOS does not hand them back
/// promptly, and counting them hides exactly the condition this probe exists to
/// detect. Checked against `JetsamEvent-2026-07-30-210002`, which recorded the
/// page counts at the instant of the kill: including inactive reads 4.29 GiB
/// (the machine looks fine while it is being killed), excluding it reads
/// 79 MiB. Wired and active pages are excluded for the obvious reason.
///
/// Read from `vm_stat`, which reports every field in one shot and states its own
/// page size, so the reading cannot silently desynchronize from a host whose
/// page size is not 4 KiB (Apple silicon reports 16384).
#[cfg(not(target_os = "linux"))]
fn live_available_memory_bytes() -> Option<u64> {
    if !cfg!(any(
        target_os = "macos",
        target_os = "ios",
        target_vendor = "apple"
    )) {
        return None;
    }
    static CACHE: OnceLock<Mutex<Option<(Instant, Option<u64>)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    // A poisoned lock must not take down an admission check: fall back to
    // probing directly rather than propagating the panic.
    let Ok(mut slot) = cache.lock() else {
        return parse_vm_stat_available_bytes(&run_vm_stat()?);
    };
    if let Some((at, value)) = *slot {
        if at.elapsed() < LIVE_MEMORY_PROBE_TTL {
            return value;
        }
    }
    let value = run_vm_stat().and_then(|out| parse_vm_stat_available_bytes(&out));
    *slot = Some((Instant::now(), value));
    value
}

#[cfg(not(target_os = "linux"))]
fn run_vm_stat() -> Option<String> {
    let out = std::process::Command::new("vm_stat").output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Parse `vm_stat` output into available bytes.
///
/// Pure so the arithmetic is testable without the binary. Returns `None` when
/// the page size or every availability field is missing -- treated by the caller
/// as "no observation", never as zero, since a zero would clamp the budget to
/// the valve minimum on every call.
#[cfg(not(target_os = "linux"))]
fn parse_vm_stat_available_bytes(contents: &str) -> Option<u64> {
    let page_size = contents
        .lines()
        .next()
        .and_then(|line| line.rsplit_once("page size of "))
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|size| *size > 0)?;

    let field = |name: &str| -> Option<u64> {
        contents
            .lines()
            .find_map(|line| line.trim().strip_prefix(name))
            .and_then(|rest| rest.trim().strip_prefix(':'))
            .and_then(|rest| rest.trim().trim_end_matches('.').parse::<u64>().ok())
    };

    // Any single field may be absent on a future OS revision; require at least
    // one to have parsed so a wholesale format change reads as "no observation".
    // `Pages inactive` is intentionally NOT summed here -- see the doc comment.
    let parts = [
        field("Pages free"),
        field("Pages purgeable"),
        field("Pages speculative"),
    ];
    if parts.iter().all(Option::is_none) {
        return None;
    }
    let pages: u64 = parts.into_iter().flatten().sum();
    pages.checked_mul(page_size)
}

/// Return the operator-configured Dense budget, distinguishing it from the
/// general 2 GiB fallback used by legacy materialization gates.
///
/// Conv CROWN uses this distinction because its independent RAM-adaptive
/// policy was introduced specifically to admit safe multi-GiB result buffers
/// on large hosts. An explicitly set `NY_DENSE_BUDGET_MB` remains authoritative
/// over host heuristics, while callers still enforce cgroup/RLIMIT headroom; an
/// invalid or non-Unicode value preserves the established 2 GiB fallback.
pub(crate) fn explicit_cpu_crown_dense_budget_bytes() -> Option<usize> {
    let raw = std::env::var_os(CROWN_DENSE_BUDGET_ENV)?;
    let parsed = raw
        .to_str()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_CROWN_DENSE_BUDGET_MB);
    Some(parsed.saturating_mul(1024 * 1024))
}

/// Remaining memory in the narrowest kernel-enforced process/container
/// envelope that NY can observe without adding a new configuration claim.
///
/// Linux exposes the two envelopes used by NY's guarded runners:
///
/// - cgroup v2 `memory.max` (or v1 `memory.limit_in_bytes`), including tighter
///   ancestor limits and the corresponding hierarchical current usage;
/// - the process soft `RLIMIT_AS`, installed by `ny-safe-gpu-run` via
///   `ulimit -v`, less the process's current `VmSize`.
///
/// The minimum remaining headroom is returned. An unreadable, malformed, or
/// unlimited source contributes no bound; callers must retain their own
/// conservative fallback. Every resolved cgroup limit/usage pair and both
/// RLIMIT inputs are read at every allocation gate through cached open handles.
/// Only path discovery and handle opening use a short, invalidating snapshot,
/// so dynamic envelope semantics stay live without turning
/// `openat`/`statx`/`close` into the dominant workload.
pub fn process_memory_headroom_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let snapshot = linux_memory_probe_snapshot();
        process_memory_headroom_from_snapshot(&snapshot)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Strict classification of the live kernel-enforced memory envelope.
///
/// Unlike [`process_memory_headroom_bytes`], this API preserves the difference
/// between a process that is conclusively unbounded and one whose cgroup or
/// RLIMIT state could not be observed completely. Optional high-memory work
/// can therefore fail closed without changing the established Dense/Conv
/// budgeting semantics of the legacy `Option<u64>` probe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessMemoryEnvelope {
    /// At least one kernel limit is finite; this is the narrowest live
    /// headroom after accounting for the corresponding usage counters.
    Bounded { headroom_bytes: u64 },
    /// Every applicable kernel source was read successfully and was unlimited.
    Unbounded,
    /// At least one applicable source could not be discovered or read strictly.
    Unavailable,
}

/// Return a strict classification of the process's cgroup/RLIMIT envelope.
///
/// Linux requires complete, internally consistent observations from both
/// cgroups and `RLIMIT_AS`. Any ambiguous source makes the aggregate
/// unavailable. Other operating systems currently have no equivalent probe
/// and report [`ProcessMemoryEnvelope::Unavailable`].
pub fn process_memory_envelope() -> ProcessMemoryEnvelope {
    #[cfg(target_os = "linux")]
    {
        let snapshot = linux_memory_probe_snapshot();
        process_memory_envelope_from_snapshot(&snapshot)
    }
    #[cfg(not(target_os = "linux"))]
    {
        ProcessMemoryEnvelope::Unavailable
    }
}

#[cfg(all(test, not(target_os = "linux")))]
#[test]
fn process_memory_envelope_is_unavailable_without_a_platform_probe() {
    assert_eq!(
        process_memory_envelope(),
        ProcessMemoryEnvelope::Unavailable
    );
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CgroupMemoryVersion {
    V1,
    V2,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParsedMemoryLimit {
    Finite(u64),
    Unlimited,
}

/// Maximum age of cgroup identity/topology validation and cached-handle
/// reopening.
///
/// Resource admission already has an unavoidable time-of-check/time-of-use
/// window. Keeping this at 100 ms preserves prompt response to a changed
/// cgroup identity while reducing a measured 13,500 full discoveries per
/// SafeNLP proof to at most roughly ten per second. Values are never cached:
/// procfs plus every resolved `memory.max`/`memory.current` pair are read live
/// through the open handles on every admission.
#[cfg(target_os = "linux")]
const MEMORY_PROBE_CACHE_TTL: Duration = Duration::from_millis(100);

#[cfg(target_os = "linux")]
const SMALL_MEMORY_FILE_BYTES: usize = 128;
#[cfg(target_os = "linux")]
const PROC_MEMORY_FILE_BYTES: usize = 16 * 1024;

#[cfg(target_os = "linux")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct CgroupProbeIdentity {
    memberships: String,
    mountinfo: String,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct CgroupMemoryCounterPaths {
    version: CgroupMemoryVersion,
    limit_path: PathBuf,
    usage_path: PathBuf,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct CgroupMemoryTopology {
    counters: Vec<CgroupMemoryCounterPaths>,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Debug)]
struct OpenMemoryFile {
    file: Arc<File>,
}

#[cfg(target_os = "linux")]
impl OpenMemoryFile {
    fn open(path: &Path) -> Option<Self> {
        File::open(path).ok().map(|file| Self {
            file: Arc::new(file),
        })
    }

    fn parse<const N: usize, T>(&self, parse: impl FnOnce(&str) -> Option<T>) -> Option<T> {
        let mut bytes = [0_u8; N];
        let len = self.file.read_at(&mut bytes, 0).ok()?;
        // Refuse a possibly truncated pseudo-file instead of parsing a
        // permissive partial value. All selected capacities are far above the
        // corresponding kernel file's normal size.
        if len == bytes.len() {
            return None;
        }
        parse(std::str::from_utf8(&bytes[..len]).ok()?)
    }
}

#[cfg(target_os = "linux")]
#[derive(Clone, Debug)]
struct OpenCgroupMemoryCounter {
    version: CgroupMemoryVersion,
    limit_path: PathBuf,
    usage_path: PathBuf,
    limit: OpenMemoryFile,
    usage: OpenMemoryFile,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Debug, Default)]
struct LinuxMemoryProbeSnapshot {
    cgroup_identity: Option<CgroupProbeIdentity>,
    cgroup_topology: CgroupMemoryTopology,
    cgroup_counters: Vec<OpenCgroupMemoryCounter>,
    membership: Option<OpenMemoryFile>,
    meminfo: Option<OpenMemoryFile>,
    limits: Option<OpenMemoryFile>,
    status: Option<OpenMemoryFile>,
}

#[cfg(target_os = "linux")]
impl LinuxMemoryProbeSnapshot {
    fn cgroup_membership_changed(&self) -> bool {
        let Some(identity) = self.cgroup_identity.as_ref() else {
            return true;
        };
        let Some(membership) = self.membership.as_ref() else {
            return true;
        };
        membership
            .parse::<PROC_MEMORY_FILE_BYTES, _>(|live| Some(live != identity.memberships.as_str()))
            // Missing, unreadable, non-UTF8, or truncated membership data must
            // invalidate. Trusting stale topology on an observation failure
            // could miss migration into a tighter cgroup.
            .unwrap_or(true)
    }
}

#[cfg(target_os = "linux")]
#[derive(Default)]
struct LinuxMemoryProbeCache {
    pid: Option<u32>,
    sampled_at: Option<Instant>,
    snapshot: Option<Arc<LinuxMemoryProbeSnapshot>>,
}

#[cfg(target_os = "linux")]
impl LinuxMemoryProbeCache {
    fn invalidate(&mut self) {
        self.sampled_at = None;
        self.snapshot = None;
    }

    fn get_or_refresh<F>(
        &mut self,
        pid: u32,
        now: Instant,
        ttl: Duration,
        refresh: F,
    ) -> Arc<LinuxMemoryProbeSnapshot>
    where
        F: FnOnce(Option<&LinuxMemoryProbeSnapshot>) -> LinuxMemoryProbeSnapshot,
    {
        // Defense in depth for any non-global use (including deterministic
        // tests): invalidate before the TTL fast path on PID change.
        // `linux_memory_probe_snapshot` performs its OWNER_PID check before
        // acquiring the process-global mutex, which is the fork-deadlock and
        // inherited-`/proc/self/*` safety boundary.
        if self.pid != Some(pid) {
            self.pid = Some(pid);
            self.invalidate();
        }
        if let (Some(sampled_at), Some(snapshot)) = (self.sampled_at, self.snapshot.as_ref()) {
            if now
                .checked_duration_since(sampled_at)
                .is_some_and(|age| age < ttl)
            {
                return Arc::clone(snapshot);
            }
        }

        let refreshed = Arc::new(refresh(self.snapshot.as_deref()));
        self.sampled_at = Some(now);
        self.snapshot = Some(Arc::clone(&refreshed));
        refreshed
    }
}

#[cfg(target_os = "linux")]
fn linux_memory_probe_snapshot() -> Arc<LinuxMemoryProbeSnapshot> {
    static OWNER_PID: AtomicU32 = AtomicU32::new(0);
    static CACHE: OnceLock<Mutex<LinuxMemoryProbeCache>> = OnceLock::new();

    let pid = std::process::id();
    let owner = OWNER_PID.load(Ordering::Acquire);
    let owner = if owner == 0 {
        match OWNER_PID.compare_exchange(0, pid, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => pid,
            Err(installed) => installed,
        }
    } else {
        owner
    };
    if owner != pid {
        // PID validation happens BEFORE touching the global mutex or its TTL
        // fast path. A raw-fork child can inherit the mutex while a vanished
        // sibling held it, and inherited `/proc/self/*` descriptors still
        // expose the parent. Bypass both in that rare pre-exec child. Normal
        // NY subprocesses exec immediately (all Rust-opened descriptors are
        // CLOEXEC); a long-lived raw-fork child stays correct at the cost of
        // uncached discovery.
        return Arc::new(refresh_linux_memory_probe_snapshot(None));
    }

    // One process-wide snapshot is bounded to the resolved hierarchy's six
    // counter pairs plus four procfs handles on this host (16 descriptors
    // total, well below soft RLIMIT_NOFILE=1024) regardless of Rayon width.
    let cache = CACHE.get_or_init(|| Mutex::new(LinuxMemoryProbeCache::default()));
    let mut cache = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let now = Instant::now();
    let snapshot = cache.get_or_refresh(
        pid,
        now,
        MEMORY_PROBE_CACHE_TTL,
        refresh_linux_memory_probe_snapshot,
    );
    if snapshot.cgroup_membership_changed() {
        // A process can be migrated to a tighter cgroup without forking.
        // Compare the tiny membership file on EVERY admission, before
        // trusting cached topology. A mismatch atomically discards every old
        // counter pair and re-resolves the new hierarchy.
        cache.invalidate();
        cache.get_or_refresh(
            pid,
            now,
            MEMORY_PROBE_CACHE_TTL,
            refresh_linux_memory_probe_snapshot,
        )
    } else {
        snapshot
    }
}

#[cfg(target_os = "linux")]
fn refresh_linux_memory_probe_snapshot(
    previous: Option<&LinuxMemoryProbeSnapshot>,
) -> LinuxMemoryProbeSnapshot {
    let observed_identity = std::fs::read_to_string("/proc/self/cgroup")
        .ok()
        .zip(std::fs::read_to_string("/proc/self/mountinfo").ok())
        .map(|(memberships, mountinfo)| CgroupProbeIdentity {
            memberships,
            mountinfo,
        });

    // Never trust a previous identity when fresh discovery fails: the process
    // may have migrated into a tighter hierarchy. `None` reproduces the
    // baseline observation-failure behavior (no fabricated cgroup bound), and
    // membership validation retries discovery on the next admission.
    let cgroup_identity = observed_identity;
    let same_identity = previous
        .is_some_and(|snapshot| snapshot.cgroup_identity.as_ref() == cgroup_identity.as_ref());
    let cgroup_topology = match (&cgroup_identity, previous) {
        (Some(_), Some(previous)) if same_identity => previous.cgroup_topology.clone(),
        (Some(identity), _) => cgroup_memory_topology_from_identity(identity),
        (None, _) => CgroupMemoryTopology::default(),
    };
    let previous_counters: &[OpenCgroupMemoryCounter] = if same_identity {
        previous.map_or(&[], |snapshot| snapshot.cgroup_counters.as_slice())
    } else {
        &[]
    };
    let cgroup_counters = open_cgroup_memory_counters(
        &cgroup_topology,
        previous_counters,
        &mut OpenMemoryFile::open,
    );
    let keep_or_open = |path: &Path, previous: Option<&OpenMemoryFile>| {
        OpenMemoryFile::open(path).or_else(|| previous.cloned())
    };
    let meminfo = keep_or_open(
        Path::new("/proc/meminfo"),
        previous.and_then(|snapshot| snapshot.meminfo.as_ref()),
    );
    let membership = keep_or_open(
        Path::new("/proc/self/cgroup"),
        previous.and_then(|snapshot| snapshot.membership.as_ref()),
    );
    let limits = keep_or_open(
        Path::new("/proc/self/limits"),
        previous.and_then(|snapshot| snapshot.limits.as_ref()),
    );
    let status = keep_or_open(
        Path::new("/proc/self/status"),
        previous.and_then(|snapshot| snapshot.status.as_ref()),
    );

    LinuxMemoryProbeSnapshot {
        cgroup_identity,
        cgroup_topology,
        cgroup_counters,
        membership,
        meminfo,
        limits,
        status,
    }
}

#[cfg(target_os = "linux")]
fn narrowest_headroom(headrooms: &[Option<u64>]) -> Option<u64> {
    headrooms.iter().copied().flatten().min()
}

#[cfg(target_os = "linux")]
fn observed_host_memory_bytes_from_snapshot(snapshot: &LinuxMemoryProbeSnapshot) -> Option<u64> {
    snapshot
        .meminfo
        .as_ref()
        .and_then(|file| file.parse::<PROC_MEMORY_FILE_BYTES, _>(parse_meminfo_available_kib))
}

#[cfg(target_os = "linux")]
fn process_memory_headroom_from_snapshot(snapshot: &LinuxMemoryProbeSnapshot) -> Option<u64> {
    let cgroup = cgroup_memory_headroom_from_open(&snapshot.cgroup_counters);
    // Read both inputs live on every admission. In particular, an unlimited
    // RLIMIT_AS that is lowered mid-proof must become authoritative
    // immediately; the snapshot caches only the open descriptors.
    let address_limit = snapshot.limits.as_ref().and_then(|file| {
        file.parse::<PROC_MEMORY_FILE_BYTES, _>(parse_address_space_soft_limit_bytes)
    });
    let address_used = snapshot
        .status
        .as_ref()
        .and_then(|file| file.parse::<PROC_MEMORY_FILE_BYTES, _>(parse_vm_size_bytes));
    let address_space = address_limit
        .zip(address_used)
        .map(|(limit, used)| limit.saturating_sub(used));
    narrowest_headroom(&[cgroup, address_space])
}

#[cfg(target_os = "linux")]
fn process_memory_envelope_from_snapshot(
    snapshot: &LinuxMemoryProbeSnapshot,
) -> ProcessMemoryEnvelope {
    combine_process_memory_envelopes(&[
        strict_cgroup_memory_envelope_from_snapshot(snapshot),
        strict_address_space_memory_envelope_from_snapshot(snapshot),
    ])
}

#[cfg(target_os = "linux")]
fn combine_process_memory_envelopes(envelopes: &[ProcessMemoryEnvelope]) -> ProcessMemoryEnvelope {
    if envelopes.is_empty() || envelopes.contains(&ProcessMemoryEnvelope::Unavailable) {
        return ProcessMemoryEnvelope::Unavailable;
    }

    envelopes
        .iter()
        .filter_map(|envelope| match envelope {
            ProcessMemoryEnvelope::Bounded { headroom_bytes } => Some(*headroom_bytes),
            ProcessMemoryEnvelope::Unbounded | ProcessMemoryEnvelope::Unavailable => None,
        })
        .min()
        .map_or(ProcessMemoryEnvelope::Unbounded, |headroom_bytes| {
            ProcessMemoryEnvelope::Bounded { headroom_bytes }
        })
}

#[cfg(target_os = "linux")]
fn strict_address_space_memory_envelope_from_snapshot(
    snapshot: &LinuxMemoryProbeSnapshot,
) -> ProcessMemoryEnvelope {
    let Some(limit) = snapshot.limits.as_ref().and_then(|file| {
        file.parse::<PROC_MEMORY_FILE_BYTES, _>(parse_address_space_soft_limit_strict)
    }) else {
        return ProcessMemoryEnvelope::Unavailable;
    };
    match limit {
        ParsedMemoryLimit::Unlimited => ProcessMemoryEnvelope::Unbounded,
        ParsedMemoryLimit::Finite(limit) => {
            let Some(used) = snapshot.status.as_ref().and_then(|file| {
                file.parse::<PROC_MEMORY_FILE_BYTES, _>(parse_vm_size_bytes_strict)
            }) else {
                return ProcessMemoryEnvelope::Unavailable;
            };
            ProcessMemoryEnvelope::Bounded {
                headroom_bytes: limit.saturating_sub(used),
            }
        }
    }
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StrictCgroupMembership<'a> {
    version: CgroupMemoryVersion,
    path: &'a str,
}

#[cfg(target_os = "linux")]
fn strict_cgroup_memberships(contents: &str) -> Option<Vec<StrictCgroupMembership<'_>>> {
    let mut memberships = Vec::new();
    let mut saw_line = false;
    let mut saw_v1_memory = false;
    let mut saw_v2 = false;

    for line in contents.lines() {
        saw_line = true;
        let mut fields = line.splitn(3, ':');
        let hierarchy = fields.next()?;
        let controllers = fields.next()?;
        let path = fields.next()?;
        let hierarchy: u64 = hierarchy.parse().ok()?;
        let non_root_segments_are_normal = path == "/"
            || path
                .strip_prefix('/')?
                .split('/')
                .all(|segment| !segment.is_empty() && segment != "." && segment != "..");
        if !non_root_segments_are_normal
            || Path::new(path).components().any(|component| {
                matches!(
                    component,
                    Component::CurDir | Component::ParentDir | Component::Prefix(_)
                )
            })
        {
            return None;
        }

        if hierarchy == 0 {
            if !controllers.is_empty() || saw_v2 {
                return None;
            }
            saw_v2 = true;
            memberships.push(StrictCgroupMembership {
                version: CgroupMemoryVersion::V2,
                path,
            });
            continue;
        }

        if controllers.is_empty()
            || controllers.split(',').any(|controller| {
                controller.is_empty() || controller.chars().any(char::is_whitespace)
            })
        {
            return None;
        }
        if controllers
            .split(',')
            .any(|controller| controller == "memory")
        {
            if saw_v1_memory {
                return None;
            }
            saw_v1_memory = true;
            memberships.push(StrictCgroupMembership {
                version: CgroupMemoryVersion::V1,
                path,
            });
        }
    }

    saw_line.then_some(memberships)
}

#[cfg(target_os = "linux")]
fn strict_cgroup_memory_envelope_from_snapshot(
    snapshot: &LinuxMemoryProbeSnapshot,
) -> ProcessMemoryEnvelope {
    let Some(identity) = snapshot.cgroup_identity.as_ref() else {
        return ProcessMemoryEnvelope::Unavailable;
    };
    let Some(memberships) = strict_cgroup_memberships(&identity.memberships) else {
        return ProcessMemoryEnvelope::Unavailable;
    };
    if memberships.is_empty() {
        return ProcessMemoryEnvelope::Unbounded;
    }

    let mut narrowest = None;
    for membership in memberships {
        let Some((current, mount)) =
            resolve_cgroup_mount(&identity.mountinfo, membership.path, membership.version)
        else {
            return ProcessMemoryEnvelope::Unavailable;
        };
        let expected = cgroup_memory_counter_paths_at(&current, &mount, membership.version);
        if expected.is_empty() {
            return ProcessMemoryEnvelope::Unavailable;
        }

        for paths in expected {
            let counter = snapshot.cgroup_counters.iter().find(|counter| {
                counter.version == paths.version
                    && counter.limit_path == paths.limit_path
                    && counter.usage_path == paths.usage_path
            });
            let Some(counter) = counter else {
                // A cgroup-v2 namespace mount root may legitimately expose no
                // memory controller files while descendants do. Only paired,
                // proven absence at that exact root is conclusively unbounded;
                // partial absence and every non-root omission are ambiguous.
                let is_v2_mount_root = membership.version == CgroupMemoryVersion::V2
                    && paths.limit_path == mount.join("memory.max")
                    && paths.usage_path == mount.join("memory.current");
                if is_v2_mount_root
                    && matches!(paths.limit_path.try_exists(), Ok(false))
                    && matches!(paths.usage_path.try_exists(), Ok(false))
                {
                    continue;
                }
                return ProcessMemoryEnvelope::Unavailable;
            };

            // Read both counters even for an unlimited limit so a malformed or
            // inaccessible usage file can never be mistaken for an unbounded
            // hierarchy.
            let limit = counter
                .limit
                .parse::<SMALL_MEMORY_FILE_BYTES, _>(|contents| {
                    parse_cgroup_memory_limit_strict(contents, counter.version)
                });
            let usage = counter
                .usage
                .parse::<SMALL_MEMORY_FILE_BYTES, _>(parse_memory_usage_bytes_strict);
            let (Some(limit), Some(usage)) = (limit, usage) else {
                return ProcessMemoryEnvelope::Unavailable;
            };
            if let ParsedMemoryLimit::Finite(limit) = limit {
                let headroom = limit.saturating_sub(usage);
                narrowest = Some(narrowest.map_or(headroom, |old: u64| old.min(headroom)));
            }
        }
    }

    narrowest.map_or(ProcessMemoryEnvelope::Unbounded, |headroom_bytes| {
        ProcessMemoryEnvelope::Bounded { headroom_bytes }
    })
}

#[cfg(target_os = "linux")]
fn parse_address_space_soft_limit_strict(contents: &str) -> Option<ParsedMemoryLimit> {
    let mut matches = contents.lines().filter_map(|line| {
        line.strip_prefix("Max address space")
            .filter(|rest| rest.starts_with(char::is_whitespace))
    });
    let rest = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    let mut fields = rest.split_whitespace();
    let soft = parse_memory_limit_field(fields.next()?)?;
    let hard = parse_memory_limit_field(fields.next()?)?;
    if fields.next()? != "bytes" || fields.next().is_some() {
        return None;
    }
    match (soft, hard) {
        (ParsedMemoryLimit::Unlimited, ParsedMemoryLimit::Finite(_)) => None,
        (ParsedMemoryLimit::Finite(soft), ParsedMemoryLimit::Finite(hard)) if soft > hard => None,
        (soft, _) => Some(soft),
    }
}

#[cfg(target_os = "linux")]
fn parse_memory_limit_field(field: &str) -> Option<ParsedMemoryLimit> {
    if field == "unlimited" {
        Some(ParsedMemoryLimit::Unlimited)
    } else {
        field.parse::<u64>().ok().map(ParsedMemoryLimit::Finite)
    }
}

#[cfg(target_os = "linux")]
fn parse_vm_size_bytes_strict(contents: &str) -> Option<u64> {
    let mut matches = contents
        .lines()
        .filter_map(|line| line.strip_prefix("VmSize:"));
    let rest = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    let mut fields = rest.split_whitespace();
    let kib: u64 = fields.next()?.parse().ok()?;
    if fields.next()? != "kB" || fields.next().is_some() {
        return None;
    }
    kib.checked_mul(1024)
}

#[cfg(target_os = "linux")]
fn parse_cgroup_memory_limit_strict(
    contents: &str,
    version: CgroupMemoryVersion,
) -> Option<ParsedMemoryLimit> {
    let value = contents.trim();
    if value.is_empty() {
        return None;
    }
    if value == "max" {
        return (version == CgroupMemoryVersion::V2).then_some(ParsedMemoryLimit::Unlimited);
    }
    let parsed: u64 = value.parse().ok()?;
    if version == CgroupMemoryVersion::V1 && parsed >= (1_u64 << 60) {
        Some(ParsedMemoryLimit::Unlimited)
    } else {
        Some(ParsedMemoryLimit::Finite(parsed))
    }
}

#[cfg(target_os = "linux")]
fn parse_memory_usage_bytes_strict(contents: &str) -> Option<u64> {
    contents.trim().parse().ok()
}

#[cfg(all(test, target_os = "linux"))]
fn address_space_headroom_from(limits: &str, status: &str) -> Option<u64> {
    let limit = parse_address_space_soft_limit_bytes(limits)?;
    let used = parse_vm_size_bytes(status)?;
    Some(limit.saturating_sub(used))
}

#[cfg(target_os = "linux")]
fn parse_address_space_soft_limit_bytes(contents: &str) -> Option<u64> {
    let line = contents
        .lines()
        .find(|line| line.starts_with("Max address space"))?;
    let rest = line.strip_prefix("Max address space")?;
    let mut fields = rest.split_whitespace();
    let soft = fields.next()?;
    let _hard = fields.next()?;
    let units = fields.next()?;
    if soft == "unlimited" || units != "bytes" {
        return None;
    }
    soft.parse().ok()
}

#[cfg(target_os = "linux")]
fn parse_vm_size_bytes(contents: &str) -> Option<u64> {
    let rest = contents
        .lines()
        .find_map(|line| line.strip_prefix("VmSize:"))?;
    let mut fields = rest.split_whitespace();
    let kib: u64 = fields.next()?.parse().ok()?;
    if fields.next()? != "kB" {
        return None;
    }
    kib.checked_mul(1024)
}

#[cfg(all(test, target_os = "linux"))]
fn cgroup_memory_headroom_from(memberships: &str, mountinfo: &str) -> Option<u64> {
    let identity = CgroupProbeIdentity {
        memberships: memberships.to_owned(),
        mountinfo: mountinfo.to_owned(),
    };
    let topology = cgroup_memory_topology_from_identity(&identity);
    let counters = open_cgroup_memory_counters(&topology, &[], &mut OpenMemoryFile::open);
    cgroup_memory_headroom_from_open(&counters)
}

#[cfg(target_os = "linux")]
fn cgroup_memory_topology_from_identity(identity: &CgroupProbeIdentity) -> CgroupMemoryTopology {
    let mut counters = Vec::new();
    for version in [CgroupMemoryVersion::V2, CgroupMemoryVersion::V1] {
        let Some(membership) = cgroup_membership_path(&identity.memberships, version) else {
            continue;
        };
        let Some((current, mount)) = resolve_cgroup_mount(&identity.mountinfo, membership, version)
        else {
            continue;
        };
        counters.extend(cgroup_memory_counter_paths_at(&current, &mount, version));
    }
    CgroupMemoryTopology { counters }
}

#[cfg(target_os = "linux")]
fn cgroup_memory_counter_paths_at(
    current: &Path,
    mount: &Path,
    version: CgroupMemoryVersion,
) -> Vec<CgroupMemoryCounterPaths> {
    if !current.starts_with(mount) {
        return Vec::new();
    }
    let (limit_name, usage_name) = match version {
        CgroupMemoryVersion::V2 => ("memory.max", "memory.current"),
        CgroupMemoryVersion::V1 => ("memory.limit_in_bytes", "memory.usage_in_bytes"),
    };
    let mut counters = Vec::new();
    let mut directory = current.to_path_buf();
    loop {
        counters.push(CgroupMemoryCounterPaths {
            version,
            limit_path: directory.join(limit_name),
            usage_path: directory.join(usage_name),
        });
        if directory == mount {
            break;
        }
        let Some(parent) = directory.parent() else {
            break;
        };
        if !parent.starts_with(mount) {
            break;
        }
        directory = parent.to_path_buf();
    }
    counters
}

#[cfg(target_os = "linux")]
fn open_cgroup_memory_counters<F>(
    topology: &CgroupMemoryTopology,
    previous: &[OpenCgroupMemoryCounter],
    open: &mut F,
) -> Vec<OpenCgroupMemoryCounter>
where
    F: FnMut(&Path) -> Option<OpenMemoryFile>,
{
    topology
        .counters
        .iter()
        .filter_map(|counter| {
            let old = previous.iter().find(|old| {
                old.version == counter.version
                    && old.limit_path == counter.limit_path
                    && old.usage_path == counter.usage_path
            });
            match (open(&counter.limit_path), open(&counter.usage_path)) {
                (Some(limit), Some(usage)) => Some(OpenCgroupMemoryCounter {
                    version: counter.version,
                    limit_path: counter.limit_path.clone(),
                    usage_path: counter.usage_path.clone(),
                    limit,
                    usage,
                }),
                // Treat the pair atomically. Mixing a newly opened limit with
                // stale usage (or vice versa) could fabricate headroom. Reuse
                // only a complete pair from the SAME cgroup identity; callers
                // pass an empty `previous` slice after identity invalidation.
                _ => old.cloned(),
            }
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn cgroup_memory_headroom_from_open(counters: &[OpenCgroupMemoryCounter]) -> Option<u64> {
    counters
        .iter()
        .filter_map(|counter| {
            // Read BOTH live even when the current limit is unlimited. This
            // preserves the original dynamic-envelope contract exactly:
            // unlimited -> finite, finite tightening, and changing usage all
            // take effect at the next admission. Only opening the files and
            // resolving their paths are cached.
            let limit = counter.limit.parse::<SMALL_MEMORY_FILE_BYTES, _>(|value| {
                parse_cgroup_memory_limit_bytes(value, counter.version)
            });
            let usage = counter
                .usage
                .parse::<SMALL_MEMORY_FILE_BYTES, _>(|value| value.trim().parse::<u64>().ok());
            let (Some(limit), Some(usage)) = (limit, usage) else {
                return None;
            };
            Some(limit.saturating_sub(usage))
        })
        .min()
}

#[cfg(target_os = "linux")]
fn cgroup_membership_path(contents: &str, version: CgroupMemoryVersion) -> Option<&str> {
    contents.lines().find_map(|line| {
        let mut fields = line.splitn(3, ':');
        let hierarchy = fields.next()?;
        let controllers = fields.next()?;
        let path = fields.next()?;
        let matches = match version {
            CgroupMemoryVersion::V2 => hierarchy == "0" && controllers.is_empty(),
            CgroupMemoryVersion::V1 => controllers.split(',').any(|name| name == "memory"),
        };
        matches.then_some(path)
    })
}

#[cfg(target_os = "linux")]
fn resolve_cgroup_mount(
    mountinfo: &str,
    membership: &str,
    version: CgroupMemoryVersion,
) -> Option<(PathBuf, PathBuf)> {
    let membership = Path::new(membership);
    let mut best: Option<(usize, PathBuf, PathBuf)> = None;
    for line in mountinfo.lines() {
        let Some((before_separator, after_separator)) = line.split_once(" - ") else {
            continue;
        };
        let mut after = after_separator.split_whitespace();
        let Some(fs_type) = after.next() else {
            continue;
        };
        let _source = after.next();
        let super_options = after.next().unwrap_or_default();
        let matches = match version {
            CgroupMemoryVersion::V2 => fs_type == "cgroup2",
            CgroupMemoryVersion::V1 => {
                fs_type == "cgroup" && super_options.split(',').any(|name| name == "memory")
            }
        };
        if !matches {
            continue;
        }

        let before: Vec<_> = before_separator.split_whitespace().collect();
        let (Some(raw_root), Some(raw_mount)) = (before.get(3), before.get(4)) else {
            continue;
        };
        let Some(root) = decode_mountinfo_path(raw_root) else {
            continue;
        };
        let Some(mount) = decode_mountinfo_path(raw_mount) else {
            continue;
        };
        let Ok(relative) = membership.strip_prefix(&root) else {
            continue;
        };
        if relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            continue;
        }
        let current = mount.join(relative);
        let specificity = root.components().count();
        if best
            .as_ref()
            .is_none_or(|(best_specificity, _, _)| specificity > *best_specificity)
        {
            best = Some((specificity, current, mount));
        }
    }
    best.map(|(_, current, mount)| (current, mount))
}

#[cfg(target_os = "linux")]
fn decode_mountinfo_path(raw: &str) -> Option<PathBuf> {
    let raw = raw.as_bytes();
    let mut decoded = Vec::with_capacity(raw.len());
    let mut index = 0;
    while index < raw.len() {
        if raw[index] != b'\\' {
            decoded.push(raw[index]);
            index += 1;
            continue;
        }
        let octal = raw.get(index + 1..index + 4)?;
        if !octal.iter().all(|&byte| matches!(byte, b'0'..=b'7')) {
            return None;
        }
        let value = u16::from(octal[0] - b'0') * 64
            + u16::from(octal[1] - b'0') * 8
            + u16::from(octal[2] - b'0');
        decoded.push(u8::try_from(value).ok()?);
        index += 4;
    }
    Some(PathBuf::from(OsString::from_vec(decoded)))
}

#[cfg(all(test, target_os = "linux"))]
fn cgroup_memory_headroom_at(
    current: &Path,
    mount: &Path,
    version: CgroupMemoryVersion,
) -> Option<u64> {
    let topology = CgroupMemoryTopology {
        counters: cgroup_memory_counter_paths_at(current, mount, version),
    };
    let counters = open_cgroup_memory_counters(&topology, &[], &mut OpenMemoryFile::open);
    cgroup_memory_headroom_from_open(&counters)
}

#[cfg(target_os = "linux")]
fn parse_cgroup_memory_limit_bytes(contents: &str, version: CgroupMemoryVersion) -> Option<u64> {
    let value = contents.trim();
    if value.is_empty() || value == "max" {
        return None;
    }
    let parsed: u64 = value.parse().ok()?;
    // Cgroup v1 represents "unlimited" with a huge page-aligned sentinel
    // close to i64::MAX rather than a word such as cgroup v2's `max`.
    if version == CgroupMemoryVersion::V1 && parsed >= (1_u64 << 60) {
        None
    } else {
        Some(parsed)
    }
}

/// Estimate the heap bytes for a Dense lower/upper coefficient pair.
pub(crate) fn dense_pair_bytes(rows: usize, cols: usize) -> Option<usize> {
    rows.checked_mul(cols)?
        .checked_mul(size_of::<f32>())?
        .checked_mul(2)
}

/// Estimate the heap bytes for a Dense identity lower/upper coefficient pair.
pub(crate) fn identity_pair_bytes(dim: usize) -> Option<usize> {
    dense_pair_bytes(dim, dim)
}

/// Upfront estimate for a batched Dense coefficient-pair materialization (#3550).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchedDenseMaterializationEstimate {
    pub site: &'static str,
    pub batch_positions: usize,
    pub rows: usize,
    pub cols: usize,
    pub required_bytes: usize,
}

impl BatchedDenseMaterializationEstimate {
    /// Create estimate, saturating to `usize::MAX` on overflow.
    pub(crate) fn new(
        site: &'static str,
        batch_positions: usize,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self {
            site,
            batch_positions,
            rows,
            cols,
            required_bytes: batched_dense_pair_bytes(batch_positions, rows, cols)
                .unwrap_or(usize::MAX),
        }
    }

    /// Whether this estimate exceeds the configured budget.
    pub(crate) fn exceeds_budget(self, budget_bytes: usize) -> bool {
        self.required_bytes > budget_bytes
    }

    /// Return `CpuMemoryExceeded` error if estimate exceeds budget, `Ok(())` otherwise.
    pub(crate) fn check_budget(self) -> ny_core::Result<()> {
        let budget = cpu_crown_dense_budget_bytes();
        if self.exceeds_budget(budget) {
            Err(NyError::CpuMemoryExceeded {
                required_bytes: self.required_bytes,
                budget_bytes: budget,
                site: self.site,
            })
        } else {
            Ok(())
        }
    }
}

/// Estimate heap bytes for a batched Dense lower/upper coefficient pair.
///
/// `required_bytes = 2 * batch_positions * rows * cols * sizeof(f32)`
pub(crate) fn batched_dense_pair_bytes(
    batch_positions: usize,
    rows: usize,
    cols: usize,
) -> Option<usize> {
    batch_positions
        .checked_mul(rows)?
        .checked_mul(cols)?
        .checked_mul(size_of::<f32>())?
        .checked_mul(2)
}

/// Estimate heap bytes for a batched Dense identity lower/upper pair.
///
/// For shape `[batch..., dim]`, `batch_positions = product(batch_dims)` and
/// `rows = cols = dim`.
pub(crate) fn batched_identity_pair_bytes(shape: &[usize]) -> Option<usize> {
    if shape.is_empty() {
        return Some(0);
    }
    let dim = shape[shape.len() - 1];
    let batch_positions = checked_shape_product(&shape[..shape.len() - 1])?.max(1);
    batched_dense_pair_bytes(batch_positions, dim, dim)
}

/// Check that a batched identity allocation for `shape` fits within the CPU budget.
///
/// Returns `Ok(())` if within budget, `Err(CpuMemoryExceeded)` otherwise.
pub(crate) fn check_batched_identity_budget(
    site: &'static str,
    shape: &[usize],
) -> ny_core::Result<()> {
    if shape.is_empty() {
        return Ok(());
    }
    let dim = shape[shape.len() - 1];
    let batch_positions = checked_shape_product(&shape[..shape.len() - 1])
        .unwrap_or(usize::MAX)
        .max(1);
    let estimate = BatchedDenseMaterializationEstimate::new(site, batch_positions, dim, dim);
    debug_assert_eq!(
        estimate.required_bytes,
        batched_identity_pair_bytes(shape).unwrap_or(usize::MAX)
    );
    estimate.check_budget()
}

#[cfg(all(test, target_os = "linux"))]
mod process_memory_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn explicit_dense_budget_is_distinct_from_the_default() {
        crate::tests::with_serialized_env_vars_removed(&[CROWN_DENSE_BUDGET_ENV], || {
            assert_eq!(explicit_cpu_crown_dense_budget_bytes(), None);
            // The implicit value is host-adaptive. Its deterministic policy is
            // covered by `adaptive_budget_tests`; this test only establishes
            // that no explicit operator override is present.
        });
        crate::tests::with_crown_dense_budget_mb("3072", || {
            let explicit = 3072 * 1024 * 1024;
            assert_eq!(explicit_cpu_crown_dense_budget_bytes(), Some(explicit));
            assert!(
                cpu_crown_dense_budget_bytes() <= explicit,
                "a process envelope may narrow but never inflate an explicit budget"
            );
        });
        crate::tests::with_crown_dense_budget_mb("malformed", || {
            assert_eq!(
                explicit_cpu_crown_dense_budget_bytes(),
                Some(DEFAULT_CROWN_DENSE_BUDGET_MB * 1024 * 1024)
            );
        });
        let overflowing_mb = usize::MAX.to_string();
        crate::tests::with_crown_dense_budget_mb(&overflowing_mb, || {
            assert_eq!(explicit_cpu_crown_dense_budget_bytes(), Some(usize::MAX));
        });
    }

    #[test]
    fn parses_cgroup_memberships_and_mounts() {
        let memberships = concat!(
            "11:cpu,cpuacct:/legacy-cpu\n",
            "10:memory,blkio:/legacy-memory/job\n",
            "0::/user.slice/ny-build.slice/job.service\n",
        );
        assert_eq!(
            cgroup_membership_path(memberships, CgroupMemoryVersion::V2),
            Some("/user.slice/ny-build.slice/job.service")
        );
        assert_eq!(
            cgroup_membership_path(memberships, CgroupMemoryVersion::V1),
            Some("/legacy-memory/job")
        );

        let mountinfo = concat!(
            "39 29 0:33 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
            "40 29 0:34 /legacy-memory /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n",
        );
        let (v2_current, v2_mount) = resolve_cgroup_mount(
            mountinfo,
            "/user.slice/ny-build.slice/job.service",
            CgroupMemoryVersion::V2,
        )
        .expect("v2 mount resolves");
        assert_eq!(v2_mount, Path::new("/sys/fs/cgroup"));
        assert_eq!(
            v2_current,
            Path::new("/sys/fs/cgroup/user.slice/ny-build.slice/job.service")
        );

        let (v1_current, v1_mount) =
            resolve_cgroup_mount(mountinfo, "/legacy-memory/job", CgroupMemoryVersion::V1)
                .expect("v1 mount resolves");
        assert_eq!(v1_mount, Path::new("/sys/fs/cgroup/memory"));
        assert_eq!(v1_current, Path::new("/sys/fs/cgroup/memory/job"));
    }

    #[test]
    fn mountinfo_decoder_handles_kernel_octal_escapes() {
        assert_eq!(
            decode_mountinfo_path(r"/sys/fs/cgroup/with\040space"),
            Some(PathBuf::from("/sys/fs/cgroup/with space"))
        );
        assert!(decode_mountinfo_path(r"/bad\escape").is_none());
        assert!(decode_mountinfo_path(r"/bad\777").is_none());
    }

    #[test]
    fn parses_cgroup_limits_rlimit_and_vm_size() {
        assert_eq!(
            parse_cgroup_memory_limit_bytes("1073741824\n", CgroupMemoryVersion::V2),
            Some(1_073_741_824)
        );
        assert_eq!(
            parse_cgroup_memory_limit_bytes("max\n", CgroupMemoryVersion::V2),
            None
        );
        assert_eq!(
            parse_cgroup_memory_limit_bytes("9223372036854771712\n", CgroupMemoryVersion::V1),
            None,
            "the v1 unlimited sentinel is not a real envelope"
        );

        let limits = concat!(
            "Limit                     Soft Limit           Hard Limit           Units\n",
            "Max cpu time              unlimited            unlimited            seconds\n",
            "Max address space         8589934592           8589934592           bytes\n",
        );
        assert_eq!(
            parse_address_space_soft_limit_bytes(limits),
            Some(8_589_934_592)
        );
        assert_eq!(
            parse_address_space_soft_limit_bytes(
                "Max address space         unlimited            unlimited            bytes\n"
            ),
            None
        );
        assert_eq!(
            parse_vm_size_bytes("Name:\tny\nVmSize:\t   65536 kB\n"),
            Some(64 * 1024 * 1024)
        );
        assert_eq!(parse_vm_size_bytes("VmSize: 64 bytes\n"), None);
        assert_eq!(
            address_space_headroom_from(limits, "VmSize: 2097152 kB\n"),
            Some(6_442_450_944)
        );
        assert_eq!(address_space_headroom_from(limits, "Name:\tny\n"), None);
        assert_eq!(
            address_space_headroom_from(limits, "VmSize: malformed kB\n"),
            None,
            "an unreadable usage counter must not be treated as zero usage"
        );
        assert_eq!(
            address_space_headroom_from(limits, "VmSize: 16777216 kB\n"),
            Some(0),
            "usage above the soft limit saturates to exhausted headroom"
        );
    }

    #[test]
    fn strict_memory_envelope_parsers_preserve_unbounded_and_unavailable() {
        assert_eq!(
            parse_cgroup_memory_limit_strict("4096\n", CgroupMemoryVersion::V2),
            Some(ParsedMemoryLimit::Finite(4096))
        );
        assert_eq!(
            parse_cgroup_memory_limit_strict("max\n", CgroupMemoryVersion::V2),
            Some(ParsedMemoryLimit::Unlimited)
        );
        assert_eq!(
            parse_cgroup_memory_limit_strict("max\n", CgroupMemoryVersion::V1),
            None,
            "the v2 spelling is malformed in a v1 counter"
        );
        assert_eq!(
            parse_cgroup_memory_limit_strict("9223372036854771712\n", CgroupMemoryVersion::V1),
            Some(ParsedMemoryLimit::Unlimited)
        );
        assert_eq!(
            parse_cgroup_memory_limit_strict("malformed\n", CgroupMemoryVersion::V2),
            None
        );

        assert_eq!(
            parse_address_space_soft_limit_strict(
                "Max address space         unlimited            unlimited            bytes\n"
            ),
            Some(ParsedMemoryLimit::Unlimited)
        );
        assert_eq!(
            parse_address_space_soft_limit_strict(
                "Max address space         8192                 unlimited            bytes\n"
            ),
            Some(ParsedMemoryLimit::Finite(8192))
        );
        assert_eq!(
            parse_address_space_soft_limit_strict(
                "Max address space         8192                 unlimited            kbytes\n"
            ),
            None
        );
        assert_eq!(parse_vm_size_bytes_strict("VmSize: 8 kB\n"), Some(8192));
        assert_eq!(parse_vm_size_bytes_strict("VmSize: 8 kB trailing\n"), None);
    }

    #[test]
    fn strict_memory_envelope_fold_is_fail_closed_and_uses_the_narrowest_bound() {
        use ProcessMemoryEnvelope::{Bounded, Unavailable, Unbounded};

        assert_eq!(combine_process_memory_envelopes(&[]), Unavailable);
        assert_eq!(
            combine_process_memory_envelopes(&[Unbounded, Unbounded]),
            Unbounded
        );
        assert_eq!(
            combine_process_memory_envelopes(&[
                Unbounded,
                Bounded { headroom_bytes: 9 },
                Bounded { headroom_bytes: 4 },
            ]),
            Bounded { headroom_bytes: 4 }
        );
        assert_eq!(
            combine_process_memory_envelopes(&[
                Bounded { headroom_bytes: 0 },
                Unavailable,
                Unbounded,
            ]),
            Unavailable,
            "an unobserved source may be tighter than every known bound"
        );
    }

    #[test]
    fn strict_cgroup_membership_distinguishes_absent_ambiguous_and_duplicate_sources() {
        assert_eq!(
            strict_cgroup_memberships("2:cpu,cpuacct:/job\n"),
            Some(Vec::new()),
            "a valid membership with no memory hierarchy is conclusively unbounded"
        );
        for malformed in [
            "",
            "0:memory:/job\n",
            "0::relative\n",
            "0::/parent/../job\n",
            "0::/first\n0::/second\n",
            "7:memory:/first\n8:memory:/second\n",
        ] {
            assert_eq!(
                strict_cgroup_memberships(malformed),
                None,
                "malformed or ambiguous membership must fail closed: {malformed:?}"
            );
        }

        let no_memory = LinuxMemoryProbeSnapshot {
            cgroup_identity: Some(CgroupProbeIdentity {
                memberships: "2:cpu,cpuacct:/job\n".to_owned(),
                mountinfo: String::new(),
            }),
            ..LinuxMemoryProbeSnapshot::default()
        };
        assert_eq!(
            strict_cgroup_memory_envelope_from_snapshot(&no_memory),
            ProcessMemoryEnvelope::Unbounded
        );
        let unresolved = LinuxMemoryProbeSnapshot {
            cgroup_identity: Some(CgroupProbeIdentity {
                memberships: "0::/job\n".to_owned(),
                mountinfo: String::new(),
            }),
            ..LinuxMemoryProbeSnapshot::default()
        };
        assert_eq!(
            strict_cgroup_memory_envelope_from_snapshot(&unresolved),
            ProcessMemoryEnvelope::Unavailable
        );
    }

    #[test]
    fn strict_cgroup_envelope_handles_v2_mount_root_absence_without_hiding_failures() {
        let unique = format!(
            "ny-strict-cgroup-memory-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let mount = std::env::temp_dir().join(unique);
        let parent = mount.join("parent");
        let current = parent.join("job");
        std::fs::create_dir_all(&current).expect("create strict synthetic cgroup tree");
        std::fs::write(parent.join("memory.max"), "1000\n").expect("parent limit");
        std::fs::write(parent.join("memory.current"), "400\n").expect("parent usage");
        std::fs::write(current.join("memory.max"), "800\n").expect("current limit");
        std::fs::write(current.join("memory.current"), "300\n").expect("current usage");

        let identity = CgroupProbeIdentity {
            memberships: "0::/parent/job\n".to_owned(),
            mountinfo: format!("39 29 0:33 / {} rw - cgroup2 cgroup rw\n", mount.display()),
        };
        let topology = cgroup_memory_topology_from_identity(&identity);
        let counters = open_cgroup_memory_counters(&topology, &[], &mut OpenMemoryFile::open);
        let snapshot = LinuxMemoryProbeSnapshot {
            cgroup_identity: Some(identity),
            cgroup_topology: topology,
            cgroup_counters: counters,
            ..LinuxMemoryProbeSnapshot::default()
        };
        assert_eq!(
            strict_cgroup_memory_envelope_from_snapshot(&snapshot),
            ProcessMemoryEnvelope::Bounded {
                headroom_bytes: 500
            },
            "the exact v2 mount root may omit both files without masking descendant bounds"
        );
        assert_eq!(
            process_memory_headroom_from_snapshot(&snapshot),
            Some(500),
            "the legacy Dense/Conv probe must retain its independent Option semantics"
        );
        assert_eq!(
            process_memory_envelope_from_snapshot(&snapshot),
            ProcessMemoryEnvelope::Unavailable,
            "missing RLIMIT observation must dominate only in the strict aggregate"
        );

        std::fs::write(parent.join("memory.max"), "max\n").expect("unbound parent");
        std::fs::write(current.join("memory.max"), "max\n").expect("unbound current");
        assert_eq!(
            strict_cgroup_memory_envelope_from_snapshot(&snapshot),
            ProcessMemoryEnvelope::Unbounded
        );
        std::fs::write(parent.join("memory.max"), "malformed\n").expect("malformed ancestor limit");
        assert_eq!(
            strict_cgroup_memory_envelope_from_snapshot(&snapshot),
            ProcessMemoryEnvelope::Unavailable,
            "a malformed ancestor must dominate an otherwise readable hierarchy"
        );

        std::fs::write(parent.join("memory.max"), "max\n").expect("restore parent");
        std::fs::write(mount.join("memory.max"), "max\n").expect("one-sided root counter");
        assert_eq!(
            strict_cgroup_memory_envelope_from_snapshot(&snapshot),
            ProcessMemoryEnvelope::Unavailable,
            "one missing root counter is ambiguous, not structurally absent"
        );
        std::fs::remove_dir_all(&mount).expect("remove strict synthetic cgroup tree");
    }

    #[test]
    fn strict_rlimit_envelope_needs_vm_size_only_for_a_finite_limit() {
        let unique = format!(
            "ny-strict-rlimit-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let base = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&base).expect("create strict RLIMIT fixture");
        let limits_path = base.join("limits");
        let status_path = base.join("status");
        std::fs::write(
            &limits_path,
            "Max address space         8192                 unlimited            bytes\n",
        )
        .expect("write finite RLIMIT");
        std::fs::write(&status_path, "VmSize: 4 kB\n").expect("write VmSize");
        let snapshot = LinuxMemoryProbeSnapshot {
            limits: OpenMemoryFile::open(&limits_path),
            status: OpenMemoryFile::open(&status_path),
            ..LinuxMemoryProbeSnapshot::default()
        };
        assert_eq!(
            strict_address_space_memory_envelope_from_snapshot(&snapshot),
            ProcessMemoryEnvelope::Bounded {
                headroom_bytes: 4096
            }
        );

        let missing_status = LinuxMemoryProbeSnapshot {
            limits: snapshot.limits,
            ..LinuxMemoryProbeSnapshot::default()
        };
        assert_eq!(
            strict_address_space_memory_envelope_from_snapshot(&missing_status),
            ProcessMemoryEnvelope::Unavailable
        );
        std::fs::write(
            &limits_path,
            "Max address space         unlimited            unlimited            bytes\n",
        )
        .expect("write unlimited RLIMIT");
        assert_eq!(
            strict_address_space_memory_envelope_from_snapshot(&missing_status),
            ProcessMemoryEnvelope::Unbounded,
            "an unlimited address-space limit has no usage subtraction"
        );
        std::fs::remove_dir_all(&base).expect("remove strict RLIMIT fixture");
    }

    #[test]
    fn narrowest_headroom_ignores_unlimited_sources() {
        assert_eq!(
            narrowest_headroom(&[None, Some(9_000), Some(4_000)]),
            Some(4_000)
        );
        assert_eq!(narrowest_headroom(&[None, None]), None);
        assert_eq!(narrowest_headroom(&[Some(0), Some(4_000)]), Some(0));
    }

    #[test]
    fn cached_file_rejects_an_exact_capacity_parseable_prefix() {
        let unique = format!(
            "ny-memory-file-truncation-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let path = std::env::temp_dir().join(unique);
        std::fs::write(&path, "123\n").expect("write exact-capacity fixture");
        let file = OpenMemoryFile::open(&path).expect("open fixture");
        assert_eq!(
            file.parse::<4, u64>(|value| value.trim().parse().ok()),
            None,
            "an exact-size read may be truncated and must never be parsed"
        );
        std::fs::remove_file(&path).expect("remove fixture");
    }

    #[test]
    fn membership_mismatch_or_read_failure_forces_invalidation() {
        let unique = format!(
            "ny-cgroup-membership-live-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let path = std::env::temp_dir().join(unique);
        std::fs::write(&path, "0::/first\n").expect("write membership");
        let membership = OpenMemoryFile::open(&path).expect("open membership");
        let snapshot = LinuxMemoryProbeSnapshot {
            cgroup_identity: Some(CgroupProbeIdentity {
                memberships: "0::/first\n".to_owned(),
                mountinfo: "fixture".to_owned(),
            }),
            membership: Some(membership),
            ..LinuxMemoryProbeSnapshot::default()
        };
        assert!(!snapshot.cgroup_membership_changed());

        std::fs::write(&path, "0::/tighter\n").expect("migrate membership");
        assert!(snapshot.cgroup_membership_changed());

        let missing = LinuxMemoryProbeSnapshot {
            cgroup_identity: snapshot.cgroup_identity.clone(),
            ..LinuxMemoryProbeSnapshot::default()
        };
        assert!(
            missing.cgroup_membership_changed(),
            "missing live membership must not trust cached topology"
        );

        std::fs::write(&path, vec![b'x'; PROC_MEMORY_FILE_BYTES])
            .expect("write exact-capacity membership");
        assert!(
            snapshot.cgroup_membership_changed(),
            "a possibly truncated membership read must invalidate"
        );
        std::fs::remove_file(&path).expect("remove membership fixture");
    }

    #[test]
    fn bounded_probe_cache_invalidates_before_ttl_on_pid_change() {
        let now = Instant::now();
        let ttl = Duration::from_millis(100);
        let refreshes = Cell::new(0);
        let mut cache = LinuxMemoryProbeCache::default();
        let first = cache.get_or_refresh(17, now, ttl, |_| {
            refreshes.set(refreshes.get() + 1);
            LinuxMemoryProbeSnapshot {
                cgroup_identity: Some(CgroupProbeIdentity {
                    memberships: "0::/first\n".to_owned(),
                    mountinfo: "first".to_owned(),
                }),
                ..LinuxMemoryProbeSnapshot::default()
            }
        });
        let reused = cache.get_or_refresh(17, now + Duration::from_millis(99), ttl, |_| {
            panic!("a cache entry younger than the bounded TTL must be reused")
        });
        assert!(Arc::ptr_eq(&first, &reused));
        assert_eq!(refreshes.get(), 1);

        let child = cache.get_or_refresh(18, now + Duration::from_millis(99), ttl, |previous| {
            refreshes.set(refreshes.get() + 1);
            assert!(
                previous.is_none(),
                "a forked child must drop parent handles"
            );
            LinuxMemoryProbeSnapshot {
                cgroup_identity: Some(CgroupProbeIdentity {
                    memberships: "0::/child\n".to_owned(),
                    mountinfo: "child".to_owned(),
                }),
                ..LinuxMemoryProbeSnapshot::default()
            }
        });
        assert!(!Arc::ptr_eq(&first, &child));
        assert_eq!(refreshes.get(), 2);
        assert_eq!(
            child
                .cgroup_identity
                .as_ref()
                .map(|identity| identity.memberships.as_str()),
            Some("0::/child\n")
        );

        let refreshed =
            cache.get_or_refresh(18, now + Duration::from_millis(199), ttl, |previous| {
                refreshes.set(refreshes.get() + 1);
                assert_eq!(
                    previous.and_then(|snapshot| snapshot.cgroup_identity.as_ref()),
                    child.cgroup_identity.as_ref()
                );
                LinuxMemoryProbeSnapshot::default()
            });
        assert!(!Arc::ptr_eq(&child, &refreshed));
        assert_eq!(refreshes.get(), 3);
    }

    #[test]
    fn changed_cgroup_identity_re_resolves_counter_paths() {
        let mountinfo = "39 29 0:33 / /cg rw - cgroup2 cgroup rw\n";
        let first_identity = CgroupProbeIdentity {
            memberships: "0::/first\n".to_owned(),
            mountinfo: mountinfo.to_owned(),
        };
        let first = cgroup_memory_topology_from_identity(&first_identity);
        assert_eq!(
            first
                .counters
                .iter()
                .map(|counter| counter.usage_path.as_path())
                .collect::<Vec<_>>(),
            vec![
                Path::new("/cg/first/memory.current"),
                Path::new("/cg/memory.current")
            ]
        );

        let second_identity = CgroupProbeIdentity {
            memberships: "0::/second\n".to_owned(),
            mountinfo: mountinfo.to_owned(),
        };
        let second = cgroup_memory_topology_from_identity(&second_identity);
        assert_ne!(first_identity, second_identity);
        assert_eq!(
            second
                .counters
                .iter()
                .map(|counter| counter.usage_path.as_path())
                .collect::<Vec<_>>(),
            vec![
                Path::new("/cg/second/memory.current"),
                Path::new("/cg/memory.current")
            ]
        );
    }

    #[test]
    fn cached_handles_keep_unlimited_to_finite_limit_and_usage_live() {
        let unique = format!(
            "ny-cgroup-live-handle-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let base = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&base).expect("create synthetic cgroup");
        let limit_path = base.join("memory.max");
        let usage_path = base.join("memory.current");
        std::fs::write(&limit_path, "max\n").expect("initial unlimited limit");
        std::fs::write(&usage_path, "300\n").expect("initial usage");
        let topology = CgroupMemoryTopology {
            counters: vec![CgroupMemoryCounterPaths {
                version: CgroupMemoryVersion::V2,
                limit_path: limit_path.clone(),
                usage_path: usage_path.clone(),
            }],
        };
        let opens = Cell::new(0);
        let counters = open_cgroup_memory_counters(&topology, &[], &mut |path| {
            opens.set(opens.get() + 1);
            OpenMemoryFile::open(path)
        });
        assert_eq!(opens.get(), 2, "each path is opened exactly once");
        assert_eq!(cgroup_memory_headroom_from_open(&counters), None);

        std::fs::write(&limit_path, "1000\n").expect("tighten unlimited limit");
        assert_eq!(
            cgroup_memory_headroom_from_open(&counters),
            Some(700),
            "unlimited -> finite must become authoritative without reopening"
        );
        std::fs::write(&limit_path, "800\n").expect("tighten finite limit");
        std::fs::write(&usage_path, "700\n").expect("raise live usage");
        assert_eq!(
            cgroup_memory_headroom_from_open(&counters),
            Some(100),
            "finite tightening and changing usage must both remain live"
        );
        assert_eq!(opens.get(), 2, "live reads must reuse both open handles");

        let partial = open_cgroup_memory_counters(&topology, &counters, &mut |path| {
            (path == limit_path)
                .then(|| OpenMemoryFile::open(path))
                .flatten()
        });
        assert_eq!(partial.len(), 1);
        assert!(
            Arc::ptr_eq(&partial[0].limit.file, &counters[0].limit.file)
                && Arc::ptr_eq(&partial[0].usage.file, &counters[0].usage.file),
            "a partial reopen must reuse the old pair atomically"
        );
        assert!(
            open_cgroup_memory_counters(&topology, &[], &mut |path| {
                (path == limit_path)
                    .then(|| OpenMemoryFile::open(path))
                    .flatten()
            })
            .is_empty(),
            "an identity change must never reuse either half of an old pair"
        );
        std::fs::remove_dir_all(&base).expect("remove synthetic cgroup");
    }

    #[test]
    fn cgroup_headroom_includes_tighter_ancestor_and_current_usage() {
        let unique = format!(
            "ny-cgroup-memory-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let mount = std::env::temp_dir().join(unique);
        let parent = mount.join("parent");
        let current = parent.join("job");
        std::fs::create_dir_all(&current).expect("create synthetic cgroup tree");
        std::fs::write(parent.join("memory.max"), "1000\n").expect("parent limit");
        std::fs::write(parent.join("memory.current"), "400\n").expect("parent usage");
        std::fs::write(current.join("memory.max"), "800\n").expect("current limit");
        std::fs::write(current.join("memory.current"), "300\n").expect("current usage");

        assert_eq!(
            cgroup_memory_headroom_at(&current, &mount, CgroupMemoryVersion::V2),
            Some(500),
            "current has 500 bytes free, narrower than the parent's 600"
        );
        std::fs::write(current.join("memory.current"), "900\n").expect("current over limit");
        assert_eq!(
            cgroup_memory_headroom_at(&current, &mount, CgroupMemoryVersion::V2),
            Some(0),
            "usage above a limit saturates to exhausted headroom"
        );
        std::fs::remove_dir_all(&mount).expect("remove synthetic cgroup tree");
    }

    #[test]
    fn cgroup_headroom_requires_a_readable_usage_counter() {
        let unique = format!(
            "ny-cgroup-missing-usage-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let mount = std::env::temp_dir().join(unique);
        let current = mount.join("job");
        std::fs::create_dir_all(&current).expect("create synthetic cgroup tree");
        std::fs::write(current.join("memory.max"), "1000\n").expect("current limit");

        assert_eq!(
            cgroup_memory_headroom_at(&current, &mount, CgroupMemoryVersion::V2),
            None,
            "a missing usage counter must not be treated as zero usage"
        );
        std::fs::write(current.join("memory.current"), "malformed\n")
            .expect("malformed current usage");
        assert_eq!(
            cgroup_memory_headroom_at(&current, &mount, CgroupMemoryVersion::V2),
            None,
            "a malformed usage counter must not be treated as zero usage"
        );
        std::fs::remove_dir_all(&mount).expect("remove synthetic cgroup tree");
    }

    #[test]
    fn cgroup_headroom_uses_v1_when_v2_has_no_memory_envelope() {
        let unique = format!(
            "ny-cgroup-hybrid-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        );
        let base = std::env::temp_dir().join(unique);
        let v2_mount = base.join("v2");
        let v1_mount = base.join("v1");
        let v2_current = v2_mount.join("job");
        let v1_parent = v1_mount.join("parent");
        let v1_current = v1_parent.join("job");
        std::fs::create_dir_all(&v2_current).expect("create synthetic v2 tree");
        std::fs::create_dir_all(&v1_current).expect("create synthetic v1 tree");
        std::fs::write(v1_parent.join("memory.limit_in_bytes"), "1000\n").expect("v1 parent limit");
        std::fs::write(v1_parent.join("memory.usage_in_bytes"), "400\n").expect("v1 parent usage");
        std::fs::write(v1_current.join("memory.limit_in_bytes"), "800\n")
            .expect("v1 current limit");
        std::fs::write(v1_current.join("memory.usage_in_bytes"), "300\n")
            .expect("v1 current usage");

        let memberships = "0::/job\n10:memory,blkio:/parent/job\n";
        let mountinfo = format!(
            "39 29 0:33 / {} rw - cgroup2 cgroup rw\n\
             40 29 0:34 / {} rw - cgroup cgroup rw,memory\n",
            v2_mount.display(),
            v1_mount.display()
        );
        assert_eq!(
            cgroup_memory_headroom_from(memberships, &mountinfo),
            Some(500),
            "a present-but-unbounded v2 hierarchy must not mask v1 memory control"
        );

        std::fs::remove_dir_all(&base).expect("remove synthetic cgroup trees");
    }

    #[test]
    fn equally_specific_cgroup_mounts_resolve_deterministically() {
        let mountinfo = concat!(
            "39 29 0:33 /slice /sys/fs/cgroup/first rw - cgroup2 cgroup rw\n",
            "40 29 0:34 /slice /sys/fs/cgroup/second rw - cgroup2 cgroup rw\n",
        );
        assert_eq!(
            resolve_cgroup_mount(mountinfo, "/slice/job", CgroupMemoryVersion::V2),
            Some((
                PathBuf::from("/sys/fs/cgroup/first/job"),
                PathBuf::from("/sys/fs/cgroup/first")
            )),
            "mountinfo order breaks ties between equally specific roots"
        );
    }
}
