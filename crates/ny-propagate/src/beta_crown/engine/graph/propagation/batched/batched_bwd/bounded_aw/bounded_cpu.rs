// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Private joined-query policy for the ONE historical CPU Linear body.
//! The immutable layer and sealed source relation remain borrowed throughout.
//! Retained graph/layer/seed provenance still requires a complete caller census;
//! this entry does not enable finite traversal or claim diagnostic allocations.

use super::super::super::super::BoundedRelation;
use super::super::super::{claimed_output::ClaimedKernelOutput, Scalar};
use super::*;
use crate::amortise::CooperativeWork;
use crate::layers::linear::bias::finalize_bias_directed_into;
use crate::layers::linear::crown_single::{
    self,
    cpu_storage::{original_weight, CpuLayout, CpuStorage},
    incoming_error_step, scalar_aw,
};
use crate::layers::LinearLayer;
use crate::LinearBounds;
use ndarray::{Array1, Array2, Axis};
use std::ops::{Index, IndexMut};

const FLAGS_FILL: &str = "bounded CPU row flags fill";
const FLAGS_COMMIT: &str = "bounded CPU row flags commit";
const CPU_WORK: &str = "bounded CPU original query work";

struct Fixed<T>(BoundedArray<T>);
impl<T> Index<[usize; 2]> for Fixed<T> {
    type Output = T;
    fn index(&self, at: [usize; 2]) -> &T {
        &self.0.storage[at]
    }
}
impl<T> IndexMut<[usize; 2]> for Fixed<T> {
    fn index_mut(&mut self, at: [usize; 2]) -> &mut T {
        &mut self.0.storage[at]
    }
}
struct Column<T>(BoundedArray<T>);
impl<T> Index<usize> for Column<T> {
    type Output = T;
    fn index(&self, row: usize) -> &T {
        &self.0.storage[[row, 0]]
    }
}
impl<T> IndexMut<usize> for Column<T> {
    fn index_mut(&mut self, row: usize) -> &mut T {
        &mut self.0.storage[[row, 0]]
    }
}

// Data MUST drop before its original capacity claim, including on failure.
struct Flags {
    values: Vec<bool>,
    _charge: Charge,
}
impl Index<usize> for Flags {
    type Output = bool;
    fn index(&self, row: usize) -> &bool {
        &self.values[row]
    }
}
impl IndexMut<usize> for Flags {
    fn index_mut(&mut self, row: usize) -> &mut bool {
        &mut self.values[row]
    }
}

struct OriginalWork<'a> {
    owner: &'a Arc<SoaStorageBudget>,
    poll: Poll,
}
impl CooperativeWork for OriginalWork<'_> {
    fn checkpoint(&mut self) -> Result<()> {
        self.owner.check_live(&mut self.poll)
    }
    fn before(&mut self, units: usize) -> Result<()> {
        if units > POLL_CHUNK_UNITS {
            return Err(invalid("CPU batch exceeds original work quantum"));
        }
        self.checkpoint()?;
        self.poll.admit_elements(units, 1, CPU_WORK)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProductRoute {
    Scalar,
    Native { chunk_macs: usize },
}

// Pure shape classification BEFORE payloads/authority. Do not catch runtime
// UnsupportedOp: original query-work refusal uses that same error variant.
fn product_route(rows: usize, k: usize, cols: usize) -> Result<ProductRoute> {
    LeasedAwStorage::product_units(rows, k, cols)?;
    let Some((chunk_macs, row_chunk)) = scalar_aw::original_native_chunk_plan(k, cols) else {
        return Ok(ProductRoute::Scalar);
    };
    if OutputPlan::new(rows.min(row_chunk), k, cols).is_err() {
        return Ok(ProductRoute::Scalar);
    }
    let tail = rows % row_chunk;
    if tail != 0 && OutputPlan::new(tail, k, cols).is_err() {
        return Ok(ProductRoute::Scalar);
    }
    Ok(ProductRoute::Native { chunk_macs })
}

struct QueryCpuStorage<'a> {
    source: &'a dyn BoundedRelation,
    layer: &'a LinearLayer,
    owner: &'a Arc<SoaStorageBudget>,
    deadline: Option<Instant>,
    poll: Poll,
    route: Option<ProductRoute>,
}
impl QueryCpuStorage<'_> {
    fn array<T: Scalar>(&mut self, rows: usize, cols: usize) -> Result<BoundedArray<T>> {
        self.owner.check_live(&mut self.poll)?;
        BoundedArray::try_from_fn_allow_empty(self.owner, rows, cols, self.deadline, |_, _| {
            T::default()
        })
    }
}

impl CpuStorage for QueryCpuStorage<'_> {
    type Coefficients = Fixed<f32>;
    type Accumulator = Column<f64>;
    type Bias = Column<f32>;
    type RowFlags = Flags;
    type Product = Fixed<f64>;
    type Incoming = Fixed<f32>;
    type Stack = ();
    type Output = ClaimedKernelOutput;

    fn admit<P: CooperativeWork>(
        &mut self,
        layer: &LinearLayer,
        bounds: &LinearBounds,
        layout: &CpuLayout,
        deadline: Option<Instant>,
        allow_engine: bool,
        stack_positions: bool,
        work: &mut P,
    ) -> Result<()> {
        work.checkpoint()?;
        self.owner.check_live(&mut self.poll)?;
        if self.owner.query.is_none()
            || !std::ptr::eq(layer, self.layer)
            || !std::ptr::eq(bounds, self.source.bounds())
            || !Arc::ptr_eq(self.owner, self.source.owner())
        {
            return Err(invalid("CPU source/layer/original joined owner mismatch"));
        }
        if deadline != self.deadline || allow_engine || stack_positions {
            return Err(NyError::UnsupportedOp(
                "bounded CPU: engine and all-position stack ownership require integration".into(),
            ));
        }
        let rows = layout
            .num_outputs
            .checked_mul(2)
            .ok_or_else(|| invalid("stacked CPU rows overflow"))?;
        self.route = Some(product_route(
            rows,
            layout.out_features,
            layout.in_features,
        )?);
        work.checkpoint()
    }
    fn coefficient_zeros(&mut self, rows: usize, cols: usize) -> Result<Self::Coefficients> {
        self.array(rows, cols).map(Fixed)
    }
    fn accumulator_zeros(&mut self, rows: usize) -> Result<Self::Accumulator> {
        self.array(rows, 1).map(Column)
    }
    fn row_flags_zeros(&mut self, rows: usize) -> Result<Self::RowFlags> {
        self.owner.check_live(&mut self.poll)?;
        let mut charge = self.owner.charge(rows, &mut self.poll)?;
        let mut values = reserve::<bool>(rows, &mut charge, &mut self.poll)?;
        self.poll.admit_elements(rows, 1, FLAGS_FILL)?;
        for _ in 0..rows {
            values.push(false);
            self.poll.work(1, FLAGS_FILL)?;
        }
        self.owner.check_live(&mut self.poll)?;
        charge.materialize()?;
        self.poll.checkpoint(FLAGS_COMMIT)?;
        Ok(Flags {
            values,
            _charge: charge,
        })
    }
    fn accumulator_slice_mut(accum: &mut Self::Accumulator) -> Result<&mut [f64]> {
        accum
            .0
            .storage
            .as_slice_mut()
            .ok_or_else(|| invalid("CPU accumulator lost fixed contiguous storage"))
    }
    fn position_stack<P: CooperativeWork>(
        &mut self,
        _: &LinearLayer,
        _: &LinearBounds,
        _: &CpuLayout,
        _: Option<Instant>,
        _: bool,
        _: &mut P,
    ) -> Result<Option<Self::Stack>> {
        Err(invalid("unadmitted bounded CPU all-position stack"))
    }
    fn aw_for_position<P: CooperativeWork>(
        &mut self,
        layer: &LinearLayer,
        bounds: &LinearBounds,
        layout: &CpuLayout,
        position: usize,
        _: Option<Instant>,
        allow_engine: bool,
        legacy_f32: bool,
        stack: Option<&Self::Stack>,
        work: &mut P,
    ) -> Result<(Self::Product, Self::Product)> {
        work.checkpoint()?;
        if allow_engine || legacy_f32 || stack.is_some() {
            return Err(invalid("unadmitted bounded CPU AW dispatch"));
        }
        let rows = layout
            .num_outputs
            .checked_mul(2)
            .ok_or_else(|| invalid("stacked CPU rows overflow"))?;
        let offset = position
            .checked_mul(layout.out_features)
            .ok_or_else(|| invalid("CPU position offset overflow"))?;
        match self
            .route
            .ok_or_else(|| invalid("CPU product before admission"))?
        {
            ProductRoute::Native { chunk_macs } => {
                let input = BoundedMat::try_from_fn(
                    self.owner,
                    rows,
                    layout.out_features,
                    self.deadline,
                    |i, k| {
                        if i < layout.num_outputs {
                            bounds.lower_a()[[i, offset + k]]
                        } else {
                            bounds.upper_a()[[i - layout.num_outputs, offset + k]]
                        }
                    },
                )?;
                let mut storage = OutputTiledLeasedAwStorage::new(self.owner, self.deadline)?;
                let result = crown_single::aw_f64_with_abssum_chunked_poll_with_storage(
                    input.view(),
                    original_weight(layer),
                    None,
                    chunk_macs,
                    &mut || work.checkpoint(),
                    &mut storage,
                )?;
                let (a, s, _) = result.ok_or_else(|| {
                    NyError::UnsupportedOp("bounded CPU: original native AW policy declined".into())
                })?;
                work.checkpoint()?;
                Ok((Fixed(a), Fixed(s)))
            }
            ProductRoute::Scalar => {
                let mut a = self.array::<f64>(rows, layout.in_features)?;
                let mut s = self.array::<f64>(rows, layout.in_features)?;
                scalar_aw::stacked_into_with_work(
                    bounds.lower_a().view(),
                    bounds.upper_a().view(),
                    offset,
                    layout.out_features,
                    original_weight(layer),
                    a.view_mut(),
                    s.view_mut(),
                    work,
                )?;
                Ok((Fixed(a), Fixed(s)))
            }
        }
    }
    fn incoming_error<P: CooperativeWork>(
        &mut self,
        layer: &LinearLayer,
        error: &Array2<f32>,
        offset: usize,
        contraction: usize,
        _: Option<Instant>,
        work: &mut P,
    ) -> Result<Self::Incoming> {
        let mut out = self.array::<f32>(error.nrows(), original_weight(layer).ncols())?;
        incoming_error_step::into_with_work(
            error.view(),
            offset,
            contraction,
            original_weight(layer),
            out.view_mut(),
            work,
        )?;
        Ok(Fixed(out))
    }
    fn finish_biases<P: CooperativeWork>(
        &mut self,
        lower: &Self::Accumulator,
        upper: &Self::Accumulator,
        old_lower: &Array1<f32>,
        old_upper: &Array1<f32>,
        has_bias: bool,
        work: &mut P,
    ) -> Result<(Self::Bias, Self::Bias)> {
        let mut lb = self.array::<f32>(old_lower.len(), 1)?;
        let mut ub = self.array::<f32>(old_upper.len(), 1)?;
        if has_bias {
            finalize_bias_directed_into(
                lower.0.view().index_axis_move(Axis(1), 0),
                upper.0.view().index_axis_move(Axis(1), 0),
                old_lower.view(),
                old_upper.view(),
                lb.view_mut().index_axis_move(Axis(1), 0),
                ub.view_mut().index_axis_move(Axis(1), 0),
                work,
            )?;
        } else {
            work.checkpoint()?;
            for i in 0..old_lower.len() {
                work.before(2)?;
                lb.storage[[i, 0]] = old_lower[i];
            }
            for i in 0..old_upper.len() {
                work.before(2)?;
                ub.storage[[i, 0]] = old_upper[i];
            }
            work.checkpoint()?;
        }
        Ok((Column(lb), Column(ub)))
    }
    fn publish<P: CooperativeWork>(
        &mut self,
        la: Self::Coefficients,
        lb: Self::Bias,
        ua: Self::Coefficients,
        ub: Self::Bias,
        le: Self::Coefficients,
        ue: Self::Coefficients,
        work: &mut P,
    ) -> Result<Self::Output> {
        work.checkpoint()?;
        let output =
            ClaimedKernelOutput::from_parts([la.0, lb.0, ua.0, ub.0, le.0, ue.0], self.deadline)?;
        work.checkpoint()?;
        Ok(output)
    }
}

fn propagate(
    source: &dyn BoundedRelation,
    layer: &LinearLayer,
    requested: Option<Instant>,
) -> Result<ClaimedKernelOutput> {
    propagate_observed(source, layer, requested, |_| Ok(()))
}

fn propagate_observed(
    source: &dyn BoundedRelation,
    layer: &LinearLayer,
    requested: Option<Instant>,
    mut observe: impl FnMut(usize) -> Result<()>,
) -> Result<ClaimedKernelOutput> {
    let owner = source.owner();
    let deadline = narrower(narrower(owner.deadline, source.deadline()), requested);
    let mut original = OriginalWork {
        owner,
        poll: Poll::for_owner(owner, deadline),
    };
    original.checkpoint()?;
    if owner.query.is_none() {
        return Err(invalid("CPU propagation requires original joined query"));
    }
    let mut storage = QueryCpuStorage {
        source,
        layer,
        owner,
        deadline,
        poll: Poll::for_owner(owner, deadline),
        route: None,
    };
    struct Observed<'a, P, F> {
        original: &'a mut P,
        observe: &'a mut F,
        calls: usize,
    }
    impl<P: CooperativeWork, F: FnMut(usize) -> Result<()>> CooperativeWork for Observed<'_, P, F> {
        fn checkpoint(&mut self) -> Result<()> {
            self.calls += 1;
            (self.observe)(self.calls)?;
            self.original.checkpoint()
        }
        fn before(&mut self, units: usize) -> Result<()> {
            self.checkpoint()?;
            self.original.before(units)
        }
    }
    let mut work = Observed {
        original: &mut original,
        observe: &mut observe,
        calls: 0,
    };
    crown_single::propagate_linear_cpu_with_storage(
        layer,
        source.bounds(),
        deadline,
        false,
        false,
        &mut work,
        &mut storage,
    )
}

#[cfg(test)]
#[path = "bounded_cpu_tests.rs"]
mod tests;
