// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The committed kernel images (ENGINE_PROGRAM §1.9).
//!
//! `kernels/fd_search.cu` (M4, the frozen-dual level search) and
//! `kernels/wave_search.cu` (M6, the margin-row wave search) are compiled
//! OFFLINE by `scripts/build_search_kernels.sh` into one fatbin each — SASS
//! for `sm_86` (A10G) and `sm_121` (GB10) plus `compute_80` PTX for any other
//! device of compute capability 8.0 or newer — and the script records every
//! source's and image's sha256, the entry points, the nvcc version and the
//! flags in `kernels/search_kernels.manifest` (format 2). There is no
//! `build.rs`: the images are embedded as bytes, loaded with cudarc's
//! `nvrtc`-feature `Ptx::from_binary` (`cuModuleLoadData`; libnvrtc itself is
//! never loaded, because nothing is compiled at runtime), so `targo
//! --unverified build --locked --offline` and hosts without a CUDA toolkit are
//! unchanged. The manifest ratchet
//! (`search_kernel_manifest_matches_the_committed_sources`) fails when a
//! source, an image or the embedded bytes drift apart — edit a `.cu`, rerun
//! the script, commit the sources, the images and the manifest together.
//! The two images load into the backend's own context independently: the
//! wave image is loaded (and self-tested) only on the first wave upload.

/// The committed level-search fatbin.
pub(crate) const FD_SEARCH_FATBIN: &[u8] = include_bytes!("../../kernels/fd_search.fatbin");

/// The committed wave-search fatbin.
pub(crate) const WAVE_SEARCH_FATBIN: &[u8] = include_bytes!("../../kernels/wave_search.fatbin");

/// The committed manifest (read by the ratchet).
#[cfg(test)]
pub(crate) const MANIFEST: &str = include_str!("../../kernels/search_kernels.manifest");

/// Entry points, in the order the device loader destructures them.
pub(crate) const KERNELS: [&str; 5] = [
    "fd_level_warm",
    "fd_compact",
    "fd_level_points",
    "fd_level_finish",
    "fd_math_kat",
];

/// The wave image's entry points, in `wave_run::Kernel` order (the loader
/// destructures them in this order).
pub(crate) const WAVE_KERNELS: [&str; 6] = [
    "wave_init",
    "wave_backward",
    "wave_concretize",
    "wave_forward",
    "wave_adam",
    "wave_babsr",
];

/// Flags the manifest must record: IEEE division / square root, no
/// flush-to-zero and no multiply-add contraction (the bit identity with the
/// CPU search depends on them, although every kernel multiply and add is an
/// explicit `__fmul_rn` / `__fadd_rn` as well).
#[cfg(test)]
pub(crate) const REQUIRED_FLAGS: [&str; 4] = [
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
];

/// The value of `key` in the manifest (`key=value` lines, `#` comments).
#[cfg(test)]
#[must_use]
pub(crate) fn manifest_value(key: &str) -> Option<&'static str> {
    MANIFEST
        .lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('#'))
        .find_map(|l| l.strip_prefix(key).and_then(|rest| rest.strip_prefix('=')))
}
