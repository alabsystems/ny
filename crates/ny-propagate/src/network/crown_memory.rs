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
use std::os::unix::ffi::OsStringExt;
#[cfg(target_os = "linux")]
use std::path::{Component, Path, PathBuf};

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

/// Get the sequential CPU Dense-materialization budget in bytes (#3515).
///
/// Priority: `NY_DENSE_BUDGET_MB` env var > 2 GiB default.
pub fn cpu_crown_dense_budget_bytes() -> usize {
    explicit_cpu_crown_dense_budget_bytes().unwrap_or(DEFAULT_CROWN_DENSE_BUDGET_MB * 1024 * 1024)
}

/// Return the operator-configured Dense budget, distinguishing it from the
/// general 2 GiB fallback used by legacy materialization gates.
///
/// Conv CROWN uses this distinction because its independent RAM-adaptive
/// policy was introduced specifically to admit safe multi-GiB result buffers
/// on large hosts. An explicitly set `NY_DENSE_BUDGET_MB` remains authoritative;
/// an invalid or non-Unicode value preserves the established 2 GiB fallback.
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
/// conservative fallback. This is intentionally read at each allocation gate:
/// cgroup usage and limits can change while a long verifier process is alive.
pub(crate) fn process_memory_headroom_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        narrowest_headroom(&[
            cgroup_memory_headroom_bytes(),
            address_space_headroom_bytes(),
        ])
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CgroupMemoryVersion {
    V1,
    V2,
}

#[cfg(target_os = "linux")]
fn narrowest_headroom(headrooms: &[Option<u64>]) -> Option<u64> {
    headrooms.iter().copied().flatten().min()
}

#[cfg(target_os = "linux")]
fn address_space_headroom_bytes() -> Option<u64> {
    let limits = std::fs::read_to_string("/proc/self/limits").ok()?;
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    address_space_headroom_from(&limits, &status)
}

#[cfg(target_os = "linux")]
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

#[cfg(target_os = "linux")]
fn cgroup_memory_headroom_bytes() -> Option<u64> {
    let memberships = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mountinfo = std::fs::read_to_string("/proc/self/mountinfo").ok()?;
    cgroup_memory_headroom_from(&memberships, &mountinfo)
}

#[cfg(target_os = "linux")]
fn cgroup_memory_headroom_from(memberships: &str, mountinfo: &str) -> Option<u64> {
    let headrooms = [CgroupMemoryVersion::V2, CgroupMemoryVersion::V1].map(|version| {
        let membership = cgroup_membership_path(memberships, version)?;
        let (current, mount) = resolve_cgroup_mount(mountinfo, membership, version)?;
        cgroup_memory_headroom_at(&current, &mount, version)
    });
    narrowest_headroom(&headrooms)
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

#[cfg(target_os = "linux")]
fn cgroup_memory_headroom_at(
    current: &Path,
    mount: &Path,
    version: CgroupMemoryVersion,
) -> Option<u64> {
    if !current.starts_with(mount) {
        return None;
    }
    let (limit_name, usage_name) = match version {
        CgroupMemoryVersion::V2 => ("memory.max", "memory.current"),
        CgroupMemoryVersion::V1 => ("memory.limit_in_bytes", "memory.usage_in_bytes"),
    };
    let mut narrowest = None;
    let mut directory = current.to_path_buf();
    loop {
        let limit = std::fs::read_to_string(directory.join(limit_name))
            .ok()
            .and_then(|value| parse_cgroup_memory_limit_bytes(&value, version));
        let usage = std::fs::read_to_string(directory.join(usage_name))
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok());
        if let (Some(limit), Some(usage)) = (limit, usage) {
            let headroom = limit.saturating_sub(usage);
            narrowest = Some(narrowest.map_or(headroom, |best: u64| best.min(headroom)));
        }
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
    narrowest
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

    #[test]
    fn explicit_dense_budget_is_distinct_from_the_default() {
        crate::tests::with_serialized_env_vars_removed(&[CROWN_DENSE_BUDGET_ENV], || {
            assert_eq!(explicit_cpu_crown_dense_budget_bytes(), None);
            assert_eq!(
                cpu_crown_dense_budget_bytes(),
                DEFAULT_CROWN_DENSE_BUDGET_MB * 1024 * 1024
            );
        });
        crate::tests::with_crown_dense_budget_mb("3072", || {
            assert_eq!(
                explicit_cpu_crown_dense_budget_bytes(),
                Some(3072 * 1024 * 1024)
            );
            assert_eq!(cpu_crown_dense_budget_bytes(), 3072 * 1024 * 1024);
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
            assert_eq!(cpu_crown_dense_budget_bytes(), usize::MAX);
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
    fn narrowest_headroom_ignores_unlimited_sources() {
        assert_eq!(
            narrowest_headroom(&[None, Some(9_000), Some(4_000)]),
            Some(4_000)
        );
        assert_eq!(narrowest_headroom(&[None, None]), None);
        assert_eq!(narrowest_headroom(&[Some(0), Some(4_000)]), Some(0));
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
