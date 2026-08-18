// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::common::assert_finite_and_ordered;
use super::super::graph_support::{first_conv_transpose_node, vocoder_prefix_subgraph};
use super::super::model::{
    bounded_kokoro_features_input_centered, load_kokoro_vocoder_with_fixed_aux,
    KOKORO_VOCODER_MIN_FIXED_AUX_T,
};
use super::*;
use ndarray::Array2;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Spec-guided CROWN on the kokoro vocoder prefix subgraph (#3500 / #3596)
//
// Spec-guided CROWN targeting boundary samples is feasible because: (1) only
// N backward passes for N specs, (2) deadline prevents hangs, and
// (3) Conv1d CROWN backward now threads GemmEngine (#3598).
//
// CROWN-IBP tightening comparison deferred: the prefix has only ~2 nodes so
// tightening room is minimal, and the extra CROWN-IBP + second spec-guided
// CROWN call would push the test past 300s on CPU.
// ---------------------------------------------------------------------------

pub(super) const SMOKE_BOUNDARY_SPEC_SAMPLES: usize = 10;
pub(in crate::tests::core::avoice::kokoro_vocoder) const FULL_BOUNDARY_SPEC_CHUNK_SAMPLES: usize =
    80;
const SPEC_CROWN_DEADLINE_SECS: u64 = 120;

#[derive(Clone)]
pub(in crate::tests::core::avoice::kokoro_vocoder) struct PrefixCrownFixture {
    pub(in crate::tests::core::avoice::kokoro_vocoder) prefix: GraphNetwork,
    pub(in crate::tests::core::avoice::kokoro_vocoder) input: BoundedTensor,
    pub(in crate::tests::core::avoice::kokoro_vocoder) ibp_output: BoundedTensor,
    pub(in crate::tests::core::avoice::kokoro_vocoder) ibp_node_bounds:
        HashMap<String, BoundedTensor>,
}

pub(in crate::tests::core::avoice::kokoro_vocoder) fn build_prefix_crown_fixture_centered(
    center_value: f32,
) -> PrefixCrownFixture {
    let model = load_kokoro_vocoder_with_fixed_aux(KOKORO_VOCODER_MIN_FIXED_AUX_T);
    let graph = model
        .to_graph_network()
        .expect("graph conversion should succeed");
    let cut_node = first_conv_transpose_node(&graph);
    let prefix = vocoder_prefix_subgraph(&graph, &cut_node);
    let input = bounded_kokoro_features_input_centered(
        &model,
        KOKORO_VOCODER_MIN_FIXED_AUX_T,
        center_value,
        1e-3,
    );

    let ibp_output = prefix
        .propagate_ibp(&input)
        .expect("prefix IBP should succeed");
    assert_finite_and_ordered(&ibp_output, "prefix IBP");

    let ibp_node_bounds = prefix
        .collect_node_bounds(&input)
        .expect("IBP node-bound collection should succeed");

    PrefixCrownFixture {
        prefix,
        input,
        ibp_output,
        ibp_node_bounds,
    }
}

fn build_prefix_crown_fixture() -> PrefixCrownFixture {
    build_prefix_crown_fixture_centered(0.0)
}

pub(in crate::tests::core::avoice::kokoro_vocoder) fn cached_prefix_crown_fixture(
) -> PrefixCrownFixture {
    static FIXTURE: OnceLock<PrefixCrownFixture> = OnceLock::new();
    FIXTURE.get_or_init(build_prefix_crown_fixture).clone()
}

pub(in crate::tests::core::avoice::kokoro_vocoder) fn with_kokoro_crown_lock<T>(
    f: impl FnOnce() -> T,
) -> T {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    let _guard = LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    f()
}

pub(super) fn boundary_spec_matrix(flat_len: usize, n: usize) -> Array2<f32> {
    let effective_n = n.min(flat_len);
    let mut spec = Array2::zeros((2 * effective_n, flat_len));
    for i in 0..effective_n {
        spec[[i, i]] = 1.0;
        spec[[effective_n + i, flat_len - effective_n + i]] = 1.0;
    }
    spec
}

pub(in crate::tests::core::avoice::kokoro_vocoder) fn boundary_spec_matrix_range(
    flat_len: usize,
    boundary_size: usize,
    start: usize,
    count: usize,
) -> Array2<f32> {
    assert!(
        boundary_size <= flat_len,
        "boundary_size {boundary_size} exceeds flat output length {flat_len}"
    );
    assert!(
        start < boundary_size,
        "boundary range start {start} exceeds boundary size {boundary_size}"
    );

    let effective_count = count.min(boundary_size - start);
    let mut spec = Array2::zeros((2 * effective_count, flat_len));
    let last_window_start = flat_len - boundary_size;
    for i in 0..effective_count {
        spec[[i, start + i]] = 1.0;
        spec[[effective_count + i, last_window_start + start + i]] = 1.0;
    }
    spec
}

pub(in crate::tests::core::avoice::kokoro_vocoder) fn spec_guided_crown_with_engine(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    spec: &Array2<f32>,
    engine: Option<&dyn ny_core::GemmEngine>,
    label: &str,
) -> (Vec<f32>, Vec<f32>) {
    let deadline = Instant::now() + Duration::from_secs(SPEC_CROWN_DEADLINE_SECS);
    spec_guided_crown_with_engine_and_deadline(
        graph,
        input,
        node_bounds,
        spec,
        engine,
        Some(deadline),
        label,
    )
}

/// Engine-wiring probe for a generic [`ny_core::GemmEngine`].
///
/// Generic GEMM has no cooperative-cancellation contract, so production
/// finite-deadline CROWN must refuse it. Tests that specifically need to
/// observe generic engine dispatch use this explicitly unbounded entry.
pub(in crate::tests::core::avoice::kokoro_vocoder) fn spec_guided_crown_with_unbounded_engine(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    spec: &Array2<f32>,
    engine: &dyn ny_core::GemmEngine,
    label: &str,
) -> (Vec<f32>, Vec<f32>) {
    spec_guided_crown_with_engine_and_deadline(
        graph,
        input,
        node_bounds,
        spec,
        Some(engine),
        None,
        label,
    )
}

fn spec_guided_crown_with_engine_and_deadline(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    spec: &Array2<f32>,
    engine: Option<&dyn ny_core::GemmEngine>,
    deadline: Option<Instant>,
    label: &str,
) -> (Vec<f32>, Vec<f32>) {
    let result = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds_and_deadline(
            input,
            spec,
            engine,
            node_bounds,
            deadline,
        )
        .unwrap_or_else(|err| panic!("{label}: {err}"));
    let flat = result.flatten();
    let lo: Vec<f32> = flat.lower().iter().copied().collect();
    let hi: Vec<f32> = flat.upper().iter().copied().collect();
    assert_eq!(lo.len(), spec.nrows(), "{label}: length mismatch");
    (lo, hi)
}

pub(super) fn spec_guided_crown(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    spec: &Array2<f32>,
    label: &str,
) -> (Vec<f32>, Vec<f32>) {
    spec_guided_crown_with_engine(graph, input, node_bounds, spec, None, label)
}

pub(in crate::tests::core::avoice::kokoro_vocoder) fn assert_crown_no_looser(
    crown_lo: &[f32],
    crown_hi: &[f32],
    ibp_fl: &[f32],
    ibp_fu: &[f32],
    ibp_ll: &[f32],
    ibp_lu: &[f32],
) -> (usize, usize) {
    let n = ibp_fl.len();
    assert_eq!(
        n,
        ibp_fu.len(),
        "first-window IBP lower/upper length mismatch"
    );
    assert_eq!(n, ibp_ll.len(), "first/last-window length mismatch");
    assert_eq!(
        n,
        ibp_lu.len(),
        "last-window IBP lower/upper length mismatch"
    );
    assert_eq!(crown_lo.len(), 2 * n, "CROWN lower length mismatch");
    assert_eq!(crown_hi.len(), 2 * n, "CROWN upper length mismatch");

    let mut tighter = 0usize;
    let mut equal = 0usize;
    for i in 0..(2 * n) {
        let (ilo, ihi) = if i < n {
            (ibp_fl[i], ibp_fu[i])
        } else {
            (ibp_ll[i - n], ibp_lu[i - n])
        };
        let scale = ilo
            .abs()
            .max(ihi.abs())
            .max(crown_lo[i].abs())
            .max(crown_hi[i].abs())
            .max(1.0);
        let tol = 1e-5 * scale;
        let tag = if i < n { "first" } else { "last" };
        let j = i % n;
        assert!(
            crown_lo[i] >= ilo - tol,
            "{tag}[{j}]: CROWN lo looser than IBP"
        );
        assert!(
            crown_hi[i] <= ihi + tol,
            "{tag}[{j}]: CROWN hi looser than IBP"
        );
        if (crown_hi[i] - crown_lo[i]) < (ihi - ilo) - tol {
            tighter += 1;
        } else {
            equal += 1;
        }
    }
    (tighter, equal)
}

pub(super) fn assert_finite_and_ordered_slices(lower: &[f32], upper: &[f32], label: &str) {
    assert_eq!(
        lower.len(),
        upper.len(),
        "{label}: lower/upper length mismatch"
    );
    for (idx, (&lo, &hi)) in lower.iter().zip(upper.iter()).enumerate() {
        assert!(
            lo.is_finite() && hi.is_finite(),
            "{label}[{idx}] non-finite"
        );
        assert!(lo <= hi, "{label}[{idx}] inverted");
    }
}
