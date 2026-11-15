// Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Scope guard for the `bab.crown_backward_layers` truncation.
//!
//! The knob stops the fixed-slope CROWN backward after N layers and concretizes
//! the remaining trunk with sound forward bounds. Its purpose, stated where it
//! is set (`configs/vnncomp25/relusplitter.yaml`), is to escape the
//! Conv2d backward bottleneck: a single Conv2d backward can eat tens of seconds
//! on this workload, so trading bound tightness for airtime is a win *there*.
//!
//! On a graph with no convolution there is no Conv2d trunk to escape. The
//! truncation then buys nothing and costs the whole backward: the concretized
//! frontier bound is strictly looser than the bound the full backward would
//! have produced, for a backward that was never expensive in the first place.
//!
//! ## How this was found, and what it is NOT
//!
//! `relusplitter` row 34 (`mnist_fc_256x4_RSPLITTER`, `prop_10_0.03`, 90 s) is a
//! pure MLP. `scripts/bisect_regression.py` bisected it over 2446 candidate
//! commits to `1ede1d30`, the WGPU proof quarantine. The `-v` A/B across that
//! commit shows the CPU route logging
//! `CROWN: truncating backward after 6 layers at frontier 9 of 16` nine times
//! where the pre-quarantine WGPU route ran with `crown_backward_layers=None` --
//! which is how the out-of-domain knob surfaced.
//!
//! **The truncation is not why that row died.** Suppressing it was measured
//! directly (commit that introduced this module, 3/3 repeats at the official
//! 90 s budget, both controls passing): the gate fires, the nine truncation
//! log lines disappear, and the deciding clause's root bound moves from
//! `-0.99923617` to `-0.99923426` -- about 2e-6, against a threshold of 0. The
//! row still returns `unknown`.
//!
//! The real gap is upstream: at `1ede1d30^` the intermediate bounds came from
//! `try_gpu_crown_partial_backward` (`CROWN-IBP GPU partial: 256 specs x N
//! layers succeeded`, collection 0.743 s) and the root precheck then verified
//! 9 of 9 objectives in 0.75 s. At HEAD `ComputeDevice::as_gpu_crown_backward()`
//! returns `None` unconditionally (`ny-gpu/src/backend.rs`), so that path is
//! structurally unreachable, the CPU collection finishes in 0.155 s doing less
//! work, and the precheck verifies 0 of 9. That is a capability gap, and
//! closing it needs a *sound* GPU CROWN backward, not this gate.
//!
//! So this module fixes a real scope error with no measured verdict effect. It
//! is kept because the knob is applied outside its documented domain and the
//! change is monotone in the safe direction -- not because it recovers a row.
//!
//! ## Soundness
//!
//! Suppressing the truncation can only tighten bounds: the full backward
//! result is the truncated frontier concretization's refinement, and both are
//! sound. No verdict that was reachable before becomes unreachable, and no
//! bound becomes tighter than the true range.
//!
//! ## Blast radius
//!
//! `configs/vnncomp25/relusplitter.yaml:72` is the only tracked preset that
//! sets `crown_backward_layers`, so this gate can only change the relusplitter
//! category. Within it the conv families (oval21, cifar_biasfield -- 140 of 220
//! rows) keep the truncation exactly; only the 80 mnist_fc rows see the full
//! backward restored.

use tracing::info;

/// The backward-layer cap that should actually be enforced.
///
/// Returns `None` -- i.e. run the backward to completion -- when the caller's
/// graph carries no convolution, because the truncation's only justification is
/// the Conv2d backward cost. `context` names the propagation route for the log.
pub(crate) fn effective_backward_cap(
    configured: Option<usize>,
    has_conv: bool,
    context: &str,
) -> Option<usize> {
    match configured {
        Some(cap) if !has_conv => {
            info!(
                "{context}: ignoring crown_backward_layers={cap} — the graph has no convolution, \
                 so truncating the backward trades bound tightness for a Conv2d cost that is not \
                 present. Running the full backward.",
            );
            None
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::effective_backward_cap;

    #[test]
    fn conv_graphs_keep_the_configured_cap() {
        assert_eq!(effective_backward_cap(Some(6), true, "test"), Some(6));
    }

    #[test]
    fn conv_free_graphs_run_the_full_backward() {
        assert_eq!(effective_backward_cap(Some(6), false, "test"), None);
    }

    #[test]
    fn an_unset_cap_stays_unset_either_way() {
        assert_eq!(effective_backward_cap(None, true, "test"), None);
        assert_eq!(effective_backward_cap(None, false, "test"), None);
    }
}
