// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Atomic caller-supplied specification rows on the deadline-bounded sound
//! CUDA ResNet backend.
//!
//! The backend accepts at most eight rows per call. This module admits the
//! process-global CUDA engine exactly once, executes every source-ordered chunk
//! into private storage, validates the complete vector, and strictly intersects
//! it with an independently certified reference. Once backend execution is
//! committed, no partial row and no ordinary CPU/GPU fallback can escape.

use std::collections::HashMap;
use std::time::Instant;

use ndarray::{s, Array1, Array2};
use ny_core::{GemmEngine, DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS};
use ny_tensor::BoundedTensor;
use tracing::debug;

use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::network::core::GraphNetwork;

use super::resnet_decompose::{
    resnet_gpu_enabled, try_resnet_gpu_suffix_bounded_rows_with_alpha_and_deadline,
    try_resnet_gpu_suffix_single_row_with_alpha_and_deadline,
};
use super::resnet_skeleton::{
    build_resnet_segment_skeleton, extract_skeleton_enabled, ResnetSegmentSkeleton,
};

fn parse_root_alpha_cuda_rows(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Shared exact-dark gate for both the DAG-alpha identity pre-loop and the
/// score-bearing multi-objective root `C`-matrix consumer.
pub(crate) fn root_alpha_cuda_rows_enabled() -> bool {
    parse_root_alpha_cuda_rows(std::env::var("NY_ROOT_ALPHA_CUDA_ROWS").ok().as_deref())
}

/// Typed refusal for an atomic CUDA row transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AtomicCudaRowsRefusal {
    UnsupportedRowCount {
        rows: usize,
    },
    EmptySpec,
    EmptySpecWidth,
    NonFiniteSpec,
    SpecCompressionActive,
    SpecWidthTargetMismatch {
        columns: usize,
        target_elements: usize,
    },
    MissingOutputNode,
    MissingReferenceOutput,
    MissingAlphaState,
    MissingDeadline,
    DeadlineExceeded,
    MissingTargetBounds,
    ReferenceShape {
        elements: usize,
    },
    ReferenceNonFiniteOrInverted,
    ResnetGpuDisabled,
    SkeletonDisabled,
    SkeletonRefused,
    FactoryUnavailable,
    FactoryAdmissionError,
    NoSoundGpuRoute,
    InvalidBackendCapacity {
        capacity: usize,
    },
    ChunkRefused {
        start: usize,
        rows: usize,
    },
    ChunkError {
        start: usize,
        rows: usize,
    },
    ChunkShape {
        start: usize,
        expected: usize,
        elements: usize,
    },
    ChunkNonFiniteOrInverted {
        start: usize,
    },
    CandidateDisjoint {
        row: usize,
    },
    FinalConstruction,
}

impl AtomicCudaRowsRefusal {
    pub(crate) fn telemetry_reason(self) -> &'static str {
        match self {
            Self::UnsupportedRowCount { .. } => "unsupported_row_count",
            Self::EmptySpec => "empty_spec",
            Self::EmptySpecWidth => "empty_spec_width",
            Self::NonFiniteSpec => "nonfinite_spec",
            Self::SpecCompressionActive => "spec_compression_active",
            Self::SpecWidthTargetMismatch { .. } => "spec_width_target_mismatch",
            Self::MissingOutputNode => "missing_output_node",
            Self::MissingReferenceOutput => "missing_reference_output",
            Self::MissingAlphaState => "missing_alpha_state",
            Self::MissingDeadline => "missing_deadline",
            Self::DeadlineExceeded => "deadline_exceeded",
            Self::MissingTargetBounds => "missing_target_bounds",
            Self::ReferenceShape { .. } => "reference_shape",
            Self::ReferenceNonFiniteOrInverted => "reference_nonfinite_or_inverted",
            Self::ResnetGpuDisabled => "resnet_gpu_disabled",
            Self::SkeletonDisabled => "skeleton_disabled",
            Self::SkeletonRefused => "skeleton_refused",
            Self::FactoryUnavailable => "factory_unavailable",
            Self::FactoryAdmissionError => "factory_admission_error",
            Self::NoSoundGpuRoute => "no_sound_gpu_route",
            Self::InvalidBackendCapacity { .. } => "invalid_backend_capacity",
            Self::ChunkRefused { .. } => "chunk_refused",
            Self::ChunkError { .. } => "chunk_error",
            Self::ChunkShape { .. } => "chunk_shape",
            Self::ChunkNonFiniteOrInverted { .. } => "chunk_nonfinite_or_inverted",
            Self::CandidateDisjoint { .. } => "candidate_disjoint",
            Self::FinalConstruction => "final_construction",
        }
    }
}

/// Result after the engine/backend capability boundary has been crossed.
pub(crate) enum AtomicCudaRowsCommit {
    /// Every row completed and the complete CUDA vector intersected the
    /// independent reference without a disjoint element.
    CudaIntersection(Box<BoundedTensor>),
    /// Backend execution was committed, but the complete transaction refused.
    /// This certified reference is authoritative; callers must not fall back.
    ReferenceRetained {
        bounds: Box<BoundedTensor>,
        refusal: AtomicCudaRowsRefusal,
    },
    /// Backend commitment was crossed, but the complete transaction or its
    /// final teardown/publication boundary observed the authority deadline.
    /// No bounds remain publishable.
    DeadlineExceeded,
}

/// The transaction either declined before any backend call, or committed to an
/// atomic CUDA/reference result.
pub(crate) enum AtomicCudaRowsOutcome {
    RefusedBeforeCommit { refusal: AtomicCudaRowsRefusal },
    Committed(AtomicCudaRowsCommit),
}

fn finalize_atomic_rows_outcome_with_clock<F>(
    outcome: AtomicCudaRowsOutcome,
    deadline: Instant,
    mut now: F,
) -> AtomicCudaRowsOutcome
where
    F: FnMut() -> Instant,
{
    if now() < deadline {
        return outcome;
    }
    match outcome {
        AtomicCudaRowsOutcome::Committed(committed) => {
            drop(committed);
            // Publication authority was already observed expired. Poll once
            // more after teardown, but never allow a non-monotonic test clock
            // to make the expired transaction publishable again.
            let _ = now();
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::DeadlineExceeded)
        }
        AtomicCudaRowsOutcome::RefusedBeforeCommit { .. } => {
            AtomicCudaRowsOutcome::RefusedBeforeCommit {
                refusal: AtomicCudaRowsRefusal::DeadlineExceeded,
            }
        }
    }
}

type AssemblyResult<T> = Result<T, AtomicCudaRowsRefusal>;

fn validate_exact_vector(
    bounds: &BoundedTensor,
    expected: usize,
    shape_refusal: impl FnOnce(usize) -> AtomicCudaRowsRefusal,
    numeric_refusal: AtomicCudaRowsRefusal,
) -> AssemblyResult<()> {
    if bounds.shape() != [expected] {
        return Err(shape_refusal(bounds.len()));
    }
    if bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .any(|(&lower, &upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
    {
        return Err(numeric_refusal);
    }
    Ok(())
}

fn chunk_seed(
    spec_matrix: &Array2<f32>,
    start: usize,
    rows: usize,
) -> AssemblyResult<LinearBounds> {
    LinearBounds::from_spec_matrix(spec_matrix.slice(s![start..start + rows, ..]).to_owned())
        .map_err(|_| AtomicCudaRowsRefusal::FinalConstruction)
}

/// Assemble a complete caller-supplied spec matrix through an injectable chunk
/// executor.
///
/// This pure seam is reusable by root-margin `C`, identity-output, and future
/// bounded-row consumers. It never publishes partial storage.
pub(crate) fn assemble_atomic_spec_rows_with<F>(
    reference: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    deadline: Instant,
    max_rows: usize,
    run_chunk: F,
) -> AssemblyResult<BoundedTensor>
where
    F: FnMut(usize, usize, &LinearBounds) -> AssemblyResult<BoundedTensor>,
{
    assemble_atomic_spec_rows_with_clock(
        reference,
        spec_matrix,
        deadline,
        max_rows,
        Instant::now,
        run_chunk,
    )
}

fn assemble_atomic_spec_rows_with_clock<F, N>(
    reference: &BoundedTensor,
    spec_matrix: &Array2<f32>,
    deadline: Instant,
    max_rows: usize,
    mut now: N,
    mut run_chunk: F,
) -> AssemblyResult<BoundedTensor>
where
    F: FnMut(usize, usize, &LinearBounds) -> AssemblyResult<BoundedTensor>,
    N: FnMut() -> Instant,
{
    if now() >= deadline {
        return Err(AtomicCudaRowsRefusal::DeadlineExceeded);
    }
    let total_rows = spec_matrix.nrows();
    if total_rows == 0 {
        return Err(AtomicCudaRowsRefusal::EmptySpec);
    }
    if spec_matrix.ncols() == 0 {
        return Err(AtomicCudaRowsRefusal::EmptySpecWidth);
    }
    for (index, value) in spec_matrix.iter().enumerate() {
        if index.is_multiple_of(4096) && now() >= deadline {
            return Err(AtomicCudaRowsRefusal::DeadlineExceeded);
        }
        if !value.is_finite() {
            return Err(AtomicCudaRowsRefusal::NonFiniteSpec);
        }
    }
    if !(1..=DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&max_rows) {
        return Err(AtomicCudaRowsRefusal::InvalidBackendCapacity { capacity: max_rows });
    }
    validate_exact_vector(
        reference,
        total_rows,
        |elements| AtomicCudaRowsRefusal::ReferenceShape { elements },
        AtomicCudaRowsRefusal::ReferenceNonFiniteOrInverted,
    )?;
    if now() >= deadline {
        return Err(AtomicCudaRowsRefusal::DeadlineExceeded);
    }

    let mut candidate_lower = Vec::with_capacity(total_rows);
    let mut candidate_upper = Vec::with_capacity(total_rows);
    let mut start = 0usize;
    while start < total_rows {
        if now() >= deadline {
            return Err(AtomicCudaRowsRefusal::DeadlineExceeded);
        }
        let remaining = total_rows - start;
        // Keep K>1 on the strict-skeleton bounded API whenever capacity allows:
        // `max_rows + 1` would otherwise produce a one-row tail on the separate
        // single-row extraction path. Splitting it as `(max_rows - 1) + 2`
        // preserves source order and the same <=8 contract.
        let rows = if max_rows > 2 && remaining == max_rows + 1 {
            max_rows - 1
        } else {
            remaining.min(max_rows)
        };
        let seed = chunk_seed(spec_matrix, start, rows)?;
        if now() >= deadline {
            return Err(AtomicCudaRowsRefusal::DeadlineExceeded);
        }
        let chunk = run_chunk(start, rows, &seed)?;
        if now() >= deadline {
            return Err(AtomicCudaRowsRefusal::DeadlineExceeded);
        }
        validate_exact_vector(
            &chunk,
            rows,
            |elements| AtomicCudaRowsRefusal::ChunkShape {
                start,
                expected: rows,
                elements,
            },
            AtomicCudaRowsRefusal::ChunkNonFiniteOrInverted { start },
        )?;
        if now() >= deadline {
            return Err(AtomicCudaRowsRefusal::DeadlineExceeded);
        }
        candidate_lower.extend(chunk.lower().iter().copied());
        candidate_upper.extend(chunk.upper().iter().copied());
        if now() >= deadline {
            return Err(AtomicCudaRowsRefusal::DeadlineExceeded);
        }
        start += rows;
    }
    if now() >= deadline {
        return Err(AtomicCudaRowsRefusal::DeadlineExceeded);
    }
    if candidate_lower.len() != total_rows || candidate_upper.len() != total_rows {
        return Err(AtomicCudaRowsRefusal::FinalConstruction);
    }

    let mut intersected_lower = Vec::with_capacity(total_rows);
    let mut intersected_upper = Vec::with_capacity(total_rows);
    for (row, ((&reference_lower, &reference_upper), (&cuda_lower, &cuda_upper))) in reference
        .lower()
        .iter()
        .zip(reference.upper().iter())
        .zip(candidate_lower.iter().zip(&candidate_upper))
        .enumerate()
    {
        if row.is_multiple_of(4096) && now() >= deadline {
            return Err(AtomicCudaRowsRefusal::DeadlineExceeded);
        }
        let lower = reference_lower.max(cuda_lower);
        let upper = reference_upper.min(cuda_upper);
        if lower > upper {
            return Err(AtomicCudaRowsRefusal::CandidateDisjoint { row });
        }
        intersected_lower.push(lower);
        intersected_upper.push(upper);
    }

    let result = BoundedTensor::new(
        Array1::from_vec(intersected_lower).into_dyn(),
        Array1::from_vec(intersected_upper).into_dyn(),
    )
    .map_err(|_| AtomicCudaRowsRefusal::FinalConstruction)?;
    if now() >= deadline {
        return Err(AtomicCudaRowsRefusal::DeadlineExceeded);
    }
    Ok(result)
}

/// Fully specified generic transaction. `reference` must bound the same
/// source-ordered spec rows as `spec_matrix`.
pub(crate) struct AtomicCudaRowsRequest<'a> {
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    target_node: &'a str,
    node_bounds: &'a HashMap<String, BoundedTensor>,
    alpha_state: &'a GraphAlphaState,
    spec_matrix: &'a Array2<f32>,
    reference: &'a BoundedTensor,
    deadline: Instant,
}

impl<'a> AtomicCudaRowsRequest<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        graph: &'a GraphNetwork,
        input: &'a BoundedTensor,
        target_node: &'a str,
        node_bounds: &'a HashMap<String, BoundedTensor>,
        alpha_state: &'a GraphAlphaState,
        spec_matrix: &'a Array2<f32>,
        reference: &'a BoundedTensor,
        deadline: Instant,
    ) -> Self {
        Self {
            graph,
            input,
            target_node,
            node_bounds,
            alpha_state,
            spec_matrix,
            reference,
            deadline,
        }
    }

    pub(crate) fn run(self) -> AtomicCudaRowsOutcome {
        let refuse_before = |refusal| AtomicCudaRowsOutcome::RefusedBeforeCommit { refusal };
        if self.spec_matrix.nrows() == 0 {
            return refuse_before(AtomicCudaRowsRefusal::EmptySpec);
        }
        if self.spec_matrix.ncols() == 0 {
            return refuse_before(AtomicCudaRowsRefusal::EmptySpecWidth);
        }
        if self.spec_matrix.iter().any(|value| !value.is_finite()) {
            return refuse_before(AtomicCudaRowsRefusal::NonFiniteSpec);
        }
        let Some(target_bounds) = self.node_bounds.get(self.target_node) else {
            return refuse_before(AtomicCudaRowsRefusal::MissingTargetBounds);
        };
        if self.spec_matrix.ncols() != target_bounds.len() {
            return refuse_before(AtomicCudaRowsRefusal::SpecWidthTargetMismatch {
                columns: self.spec_matrix.ncols(),
                target_elements: target_bounds.len(),
            });
        }
        if let Err(refusal) = validate_exact_vector(
            self.reference,
            self.spec_matrix.nrows(),
            |elements| AtomicCudaRowsRefusal::ReferenceShape { elements },
            AtomicCudaRowsRefusal::ReferenceNonFiniteOrInverted,
        ) {
            return refuse_before(refusal);
        }
        if Instant::now() >= self.deadline {
            return refuse_before(AtomicCudaRowsRefusal::DeadlineExceeded);
        }
        if !resnet_gpu_enabled() {
            return refuse_before(AtomicCudaRowsRefusal::ResnetGpuDisabled);
        }
        if !extract_skeleton_enabled() {
            return refuse_before(AtomicCudaRowsRefusal::SkeletonDisabled);
        }
        let Some(skeleton) = build_resnet_segment_skeleton(
            self.graph,
            self.input,
            self.target_node,
            self.node_bounds,
            self.node_bounds,
            Some(self.alpha_state),
            /* allow_pure_chain = */ false,
        ) else {
            return refuse_before(AtomicCudaRowsRefusal::SkeletonRefused);
        };
        if Instant::now() >= self.deadline {
            return refuse_before(AtomicCudaRowsRefusal::DeadlineExceeded);
        }

        let admitted = crate::sound_f64_gemm::with_engine_deadline(self.deadline, |engine| {
            let Some(gpu) = engine
                .as_gpu_crown_backward()
                .filter(|gpu| gpu.provides_sound_gpu_crown())
            else {
                return AtomicCudaRowsOutcome::RefusedBeforeCommit {
                    refusal: AtomicCudaRowsRefusal::NoSoundGpuRoute,
                };
            };
            let max_rows = if self.spec_matrix.nrows() == 1 {
                1
            } else {
                let capacity = gpu.deadline_bounded_resnet_sound_max_rows();
                if !(2..=DEADLINE_BOUNDED_RESNET_SOUND_MAX_ROWS).contains(&capacity) {
                    return AtomicCudaRowsOutcome::RefusedBeforeCommit {
                        refusal: AtomicCudaRowsRefusal::InvalidBackendCapacity { capacity },
                    };
                }
                capacity
            };
            match assemble_on_engine(&self, &skeleton, engine, max_rows) {
                Ok(bounds) if Instant::now() >= self.deadline => {
                    drop(bounds);
                    AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::DeadlineExceeded)
                }
                Ok(bounds) => AtomicCudaRowsOutcome::Committed(
                    AtomicCudaRowsCommit::CudaIntersection(Box::new(bounds)),
                ),
                Err(AtomicCudaRowsRefusal::DeadlineExceeded) => {
                    AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::DeadlineExceeded)
                }
                Err(refusal) => {
                    AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::ReferenceRetained {
                        bounds: Box::new(self.reference.clone()),
                        refusal,
                    })
                }
            }
        });

        match admitted {
            Ok(Some(outcome)) => {
                finalize_atomic_rows_outcome_with_clock(outcome, self.deadline, Instant::now)
            }
            Ok(None) => refuse_before(AtomicCudaRowsRefusal::FactoryUnavailable),
            Err(error) if Instant::now() >= self.deadline => {
                debug!(%error, "atomic CUDA rows factory admission crossed its deadline");
                refuse_before(AtomicCudaRowsRefusal::DeadlineExceeded)
            }
            Err(error) => {
                debug!(%error, "atomic CUDA rows factory admission errored");
                refuse_before(AtomicCudaRowsRefusal::FactoryAdmissionError)
            }
        }
    }
}

fn assemble_on_engine(
    request: &AtomicCudaRowsRequest<'_>,
    skeleton: &ResnetSegmentSkeleton,
    engine: &dyn GemmEngine,
    max_rows: usize,
) -> AssemblyResult<BoundedTensor> {
    assemble_atomic_spec_rows_with(
        request.reference,
        request.spec_matrix,
        request.deadline,
        max_rows,
        |start, rows, seed| {
            let result = if rows == 1 {
                try_resnet_gpu_suffix_single_row_with_alpha_and_deadline(
                    request.graph,
                    request.input,
                    request.target_node,
                    request.node_bounds,
                    request.node_bounds,
                    request.alpha_state,
                    Some(engine),
                    request.deadline,
                    seed,
                )
            } else {
                try_resnet_gpu_suffix_bounded_rows_with_alpha_and_deadline(
                    request.graph,
                    request.input,
                    request.target_node,
                    request.node_bounds,
                    request.node_bounds,
                    request.alpha_state,
                    skeleton,
                    Some(engine),
                    request.deadline,
                    seed,
                )
                .map(|result| result.map(|result| result.bounds))
            };
            match result {
                Ok(Some(bounds)) => Ok(bounds),
                Ok(None) => Err(AtomicCudaRowsRefusal::ChunkRefused { start, rows }),
                Err(error) => {
                    debug!(%error, start, rows, "atomic CUDA rows chunk errored");
                    Err(AtomicCudaRowsRefusal::ChunkError { start, rows })
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::time::Duration;

    use super::*;

    fn vector_bounds(lower: Vec<f32>, upper: Vec<f32>) -> BoundedTensor {
        BoundedTensor::new(
            Array1::from_vec(lower).into_dyn(),
            Array1::from_vec(upper).into_dyn(),
        )
        .expect("test bounds must be valid")
    }

    fn identity(rows: usize) -> Array2<f32> {
        let mut spec = Array2::zeros((rows, rows));
        for row in 0..rows {
            spec[(row, row)] = 1.0;
        }
        spec
    }

    #[test]
    fn exact_gate_only_accepts_one() {
        assert!(!parse_root_alpha_cuda_rows(None));
        for raw in ["", "0", "true", " 1", "1 "] {
            assert!(!parse_root_alpha_cuda_rows(Some(raw)), "raw={raw:?}");
        }
        assert!(parse_root_alpha_cuda_rows(Some("1")));
    }

    #[test]
    fn caller_supplied_ninety_nine_rows_keep_source_order_and_chunk_plan() {
        let mut spec = Array2::<f32>::zeros((99, 5));
        for row in 0..99 {
            for column in 0..5 {
                spec[(row, column)] = row as f32 * 10.0 + column as f32 + 0.25;
            }
        }
        let reference = vector_bounds(vec![-200.0; 99], vec![200.0; 99]);
        let mut calls = Vec::new();
        let result = assemble_atomic_spec_rows_with(
            &reference,
            &spec,
            Instant::now() + Duration::from_secs(10),
            8,
            |start, rows, seed| {
                calls.push((start, rows));
                assert_eq!(
                    seed.lower_a(),
                    &spec.slice(s![start..start + rows, ..]).to_owned()
                );
                assert_eq!(seed.lower_a(), seed.upper_a());
                Ok(vector_bounds(
                    (start..start + rows).map(|row| row as f32).collect(),
                    (start..start + rows).map(|row| row as f32 + 0.5).collect(),
                ))
            },
        )
        .expect("complete atomic assembly");

        assert_eq!(calls.len(), 13);
        assert!(calls[..12]
            .iter()
            .enumerate()
            .all(|(index, &(start, rows))| start == index * 8 && rows == 8));
        assert_eq!(calls[12], (96, 3));
        assert_eq!(
            result.lower().iter().copied().collect::<Vec<_>>(),
            (0..99).map(|row| row as f32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn hundred_identity_rows_are_twelve_eights_plus_four() {
        let spec = identity(100);
        let reference = vector_bounds(vec![-1.0; 100], vec![1.0; 100]);
        let mut calls = Vec::new();
        assemble_atomic_spec_rows_with(
            &reference,
            &spec,
            Instant::now() + Duration::from_secs(10),
            8,
            |start, rows, seed| {
                calls.push((start, rows));
                assert_eq!(
                    seed.lower_a(),
                    &spec.slice(s![start..start + rows, ..]).to_owned()
                );
                Ok(vector_bounds(vec![-0.5; rows], vec![0.5; rows]))
            },
        )
        .expect("complete atomic assembly");
        assert_eq!(calls.len(), 13);
        assert_eq!(calls[12], (96, 4));
    }

    #[test]
    fn arbitrary_positive_row_count_avoids_a_single_tail_when_capacity_allows() {
        let spec = Array2::from_shape_fn((9, 3), |(row, column)| (row * 3 + column) as f32);
        let reference = vector_bounds(vec![-100.0; 9], vec![100.0; 9]);
        let mut calls = Vec::new();
        assemble_atomic_spec_rows_with(
            &reference,
            &spec,
            Instant::now() + Duration::from_secs(10),
            8,
            |start, rows, _seed| {
                calls.push((start, rows));
                Ok(vector_bounds(vec![0.0; rows], vec![1.0; rows]))
            },
        )
        .expect("nine rows are a strict-skeleton seven plus two");
        assert_eq!(calls, vec![(0, 7), (7, 2)]);
    }

    #[test]
    fn chunk_refusal_never_publishes_a_partial_vector() {
        let spec = identity(99);
        let reference = vector_bounds(vec![-1.0; 99], vec![1.0; 99]);
        let mut calls = 0usize;
        let result = assemble_atomic_spec_rows_with(
            &reference,
            &spec,
            Instant::now() + Duration::from_secs(10),
            8,
            |start, rows, _seed| {
                calls += 1;
                if start == 16 {
                    return Err(AtomicCudaRowsRefusal::ChunkRefused { start, rows });
                }
                Ok(vector_bounds(vec![-0.5; rows], vec![0.5; rows]))
            },
        );
        assert_eq!(
            result.err(),
            Some(AtomicCudaRowsRefusal::ChunkRefused { start: 16, rows: 8 })
        );
        assert_eq!(calls, 3, "assembly must stop at the first refusal");
    }

    #[test]
    fn strict_intersection_rejects_disjoint_candidate() {
        let spec = identity(99);
        let reference = vector_bounds(vec![0.0; 99], vec![1.0; 99]);
        let result = assemble_atomic_spec_rows_with(
            &reference,
            &spec,
            Instant::now() + Duration::from_secs(10),
            8,
            |start, rows, _seed| {
                if start == 8 {
                    Ok(vector_bounds(vec![2.0; rows], vec![3.0; rows]))
                } else {
                    Ok(vector_bounds(vec![0.25; rows], vec![0.75; rows]))
                }
            },
        );
        assert_eq!(
            result.err(),
            Some(AtomicCudaRowsRefusal::CandidateDisjoint { row: 8 })
        );
    }

    #[test]
    fn assembly_deadline_matrix_covers_every_post_assembly_publication_boundary() {
        let base = Instant::now();
        let deadline = base + Duration::from_secs(10);
        let spec = identity(1);
        let reference = vector_bounds(vec![-1.0], vec![1.0]);

        // Poll indices for a one-row assembly:
        // 1 spec validation, 2 post-reference validation, 4 post-seed /
        // pre-executor, 5 post-chunk, 9 intersection, and 10
        // post-final-construction.
        for (expire_at, expected_chunks) in [(1, 0), (2, 0), (4, 0), (5, 1), (9, 1), (10, 1)] {
            let polls = Cell::new(0usize);
            let chunks = Cell::new(0usize);
            let result = assemble_atomic_spec_rows_with_clock(
                &reference,
                &spec,
                deadline,
                1,
                || {
                    let poll = polls.get();
                    polls.set(poll + 1);
                    if poll >= expire_at {
                        deadline
                    } else {
                        base
                    }
                },
                |_start, rows, _seed| {
                    chunks.set(chunks.get() + 1);
                    Ok(vector_bounds(vec![-0.5; rows], vec![0.5; rows]))
                },
            );
            assert_eq!(
                result.err(),
                Some(AtomicCudaRowsRefusal::DeadlineExceeded),
                "expire_at={expire_at}"
            );
            assert_eq!(
                chunks.get(),
                expected_chunks,
                "expire_at={expire_at} must not execute another chunk"
            );
            assert_eq!(polls.get(), expire_at + 1, "expire_at={expire_at}");
        }

        let polls = Cell::new(0usize);
        let result = assemble_atomic_spec_rows_with_clock(
            &reference,
            &spec,
            deadline,
            1,
            || {
                polls.set(polls.get() + 1);
                base
            },
            |_start, rows, _seed| Ok(vector_bounds(vec![-0.5; rows], vec![0.5; rows])),
        )
        .expect("a live clock must preserve exact assembly behavior");
        assert_eq!(result.lower()[0].to_bits(), (-0.5_f32).to_bits());
        assert_eq!(result.upper()[0].to_bits(), 0.5_f32.to_bits());
        assert_eq!(polls.get(), 11);
    }

    #[test]
    fn committed_reference_crossing_deadline_is_torn_down_and_never_published() {
        let base = Instant::now();
        let deadline = base + Duration::from_secs(10);
        let calls = Cell::new(0usize);
        let outcome = AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::ReferenceRetained {
            bounds: Box::new(vector_bounds(vec![-1.0], vec![1.0])),
            refusal: AtomicCudaRowsRefusal::ChunkRefused { start: 0, rows: 1 },
        });
        let outcome = finalize_atomic_rows_outcome_with_clock(outcome, deadline, || {
            let call = calls.get();
            calls.set(call + 1);
            if call == 0 {
                deadline
            } else {
                base
            }
        });
        assert!(matches!(
            outcome,
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::DeadlineExceeded)
        ));
        assert_eq!(
            calls.get(),
            2,
            "the retained reference must be destroyed before the final poll"
        );
    }

    #[test]
    fn live_committed_reference_moves_the_same_allocation() {
        let base = Instant::now();
        let deadline = base + Duration::from_secs(10);
        let bounds = Box::new(vector_bounds(vec![-1.0], vec![1.0]));
        let bounds_ptr = std::ptr::from_ref(bounds.as_ref());
        let outcome = finalize_atomic_rows_outcome_with_clock(
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::ReferenceRetained {
                bounds,
                refusal: AtomicCudaRowsRefusal::ChunkRefused { start: 0, rows: 1 },
            }),
            deadline,
            || base,
        );
        let AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::ReferenceRetained {
            bounds,
            refusal,
        }) = outcome
        else {
            panic!("a live committed reference must remain authoritative");
        };
        assert!(std::ptr::eq(bounds_ptr, bounds.as_ref()));
        assert_eq!(
            refusal,
            AtomicCudaRowsRefusal::ChunkRefused { start: 0, rows: 1 }
        );
    }

    #[test]
    fn request_refuses_spec_width_mismatch_before_factory_admission() {
        let graph = GraphNetwork::new();
        let input = vector_bounds(vec![-1.0], vec![1.0]);
        let target = vector_bounds(vec![-1.0; 3], vec![1.0; 3]);
        let reference = vector_bounds(vec![-2.0; 2], vec![2.0; 2]);
        let spec = Array2::zeros((2, 4));
        let alpha_state = GraphAlphaState::new();
        let node_bounds = HashMap::from([("output".to_string(), target)]);

        let outcome = AtomicCudaRowsRequest::new(
            &graph,
            &input,
            "output",
            &node_bounds,
            &alpha_state,
            &spec,
            &reference,
            Instant::now() + Duration::from_secs(10),
        )
        .run();
        assert!(matches!(
            outcome,
            AtomicCudaRowsOutcome::RefusedBeforeCommit {
                refusal: AtomicCudaRowsRefusal::SpecWidthTargetMismatch {
                    columns: 4,
                    target_elements: 3,
                },
            }
        ));
    }
}
