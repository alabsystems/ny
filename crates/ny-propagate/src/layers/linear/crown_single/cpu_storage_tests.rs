// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::amortise::NoPoll;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

// Count ownership guards, not bytes or allocator operations. This models the
// move-only policy contract; it is not a SoaStorageBudget/source-capture proof.
struct Live(Arc<AtomicUsize>);
impl Drop for Live {
    fn drop(&mut self) {
        assert!(self.0.fetch_sub(1, Ordering::SeqCst) > 0);
    }
}

// Struct field order retains the guard until after its payload has dropped.
struct Tracked<T> {
    data: T,
    guard: Live,
}
impl<T, I> Index<I> for Tracked<T>
where
    T: Index<I>,
{
    type Output = <T as Index<I>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        &self.data[index]
    }
}
impl<T, I> IndexMut<I> for Tracked<T>
where
    T: IndexMut<I>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.data[index]
    }
}
impl<T> Tracked<T> {
    fn split(self) -> (T, Live) {
        (self.data, self.guard)
    }
}

struct TrackedOutput {
    bounds: LinearBounds,
    _guards: [Live; 6],
}

#[derive(Default)]
struct Trace {
    calls: Vec<&'static str>,
    fail: Option<usize>,
}
struct TrackingStorage {
    live: Arc<AtomicUsize>,
    trace: Rc<RefCell<Trace>>,
}
impl TrackingStorage {
    fn new(fail: Option<usize>) -> Self {
        Self {
            live: Arc::new(AtomicUsize::new(0)),
            trace: Rc::new(RefCell::new(Trace {
                calls: Vec::new(),
                fail,
            })),
        }
    }
    fn observe(&self, stage: &'static str) -> Result<()> {
        let mut trace = self.trace.borrow_mut();
        trace.calls.push(stage);
        if trace.fail == Some(trace.calls.len()) {
            Err(NyError::DeadlineExceeded(stage.into()))
        } else {
            Ok(())
        }
    }
    fn wrap<T>(&self, data: T) -> Tracked<T> {
        self.live.fetch_add(1, Ordering::SeqCst);
        Tracked {
            data,
            guard: Live(Arc::clone(&self.live)),
        }
    }
}

impl CpuStorage for TrackingStorage {
    type Coefficients = Tracked<Array2<f32>>;
    type Accumulator = Tracked<Array1<f64>>;
    type Bias = Tracked<Array1<f32>>;
    type RowFlags = Tracked<Vec<bool>>;
    type Product = Tracked<Array2<f64>>;
    type Incoming = Tracked<Array2<f32>>;
    type Stack = Tracked<PositionStackedAw>;
    type Output = TrackedOutput;

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
        self.observe("admit")?;
        OwnedCpuStorage.admit(
            layer,
            bounds,
            layout,
            deadline,
            allow_engine,
            stack_positions,
            work,
        )
    }
    fn coefficient_zeros(&mut self, rows: usize, cols: usize) -> Result<Self::Coefficients> {
        self.observe("coefficient")?;
        Ok(self.wrap(OwnedCpuStorage.coefficient_zeros(rows, cols)?))
    }
    fn accumulator_zeros(&mut self, rows: usize) -> Result<Self::Accumulator> {
        self.observe("accumulator")?;
        Ok(self.wrap(OwnedCpuStorage.accumulator_zeros(rows)?))
    }
    fn row_flags_zeros(&mut self, rows: usize) -> Result<Self::RowFlags> {
        self.observe("row-flags")?;
        Ok(self.wrap(OwnedCpuStorage.row_flags_zeros(rows)?))
    }
    fn accumulator_slice_mut(accumulator: &mut Self::Accumulator) -> Result<&mut [f64]> {
        OwnedCpuStorage::accumulator_slice_mut(&mut accumulator.data)
    }
    fn position_stack<P: CooperativeWork>(
        &mut self,
        layer: &LinearLayer,
        bounds: &LinearBounds,
        layout: &CpuLayout,
        deadline: Option<Instant>,
        allow_engine: bool,
        work: &mut P,
    ) -> Result<Option<Self::Stack>> {
        self.observe("position-stack")?;
        Ok(OwnedCpuStorage
            .position_stack(layer, bounds, layout, deadline, allow_engine, work)?
            .map(|stack| self.wrap(stack)))
    }
    fn aw_for_position<P: CooperativeWork>(
        &mut self,
        layer: &LinearLayer,
        bounds: &LinearBounds,
        layout: &CpuLayout,
        position: usize,
        deadline: Option<Instant>,
        allow_engine: bool,
        legacy_f32: bool,
        stack: Option<&Self::Stack>,
        work: &mut P,
    ) -> Result<(Self::Product, Self::Product)> {
        self.observe("aw")?;
        let (a, s) = OwnedCpuStorage.aw_for_position(
            layer,
            bounds,
            layout,
            position,
            deadline,
            allow_engine,
            legacy_f32,
            stack.map(|s| &s.data),
            work,
        )?;
        Ok((self.wrap(a), self.wrap(s)))
    }
    fn incoming_error<P: CooperativeWork>(
        &mut self,
        layer: &LinearLayer,
        error: &Array2<f32>,
        offset: usize,
        contraction: usize,
        deadline: Option<Instant>,
        work: &mut P,
    ) -> Result<Self::Incoming> {
        self.observe("incoming-error")?;
        Ok(self.wrap(OwnedCpuStorage.incoming_error(
            layer,
            error,
            offset,
            contraction,
            deadline,
            work,
        )?))
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
        self.observe("finish-biases")?;
        let (lower, upper) = OwnedCpuStorage.finish_biases(
            &lower.data,
            &upper.data,
            old_lower,
            old_upper,
            has_bias,
            work,
        )?;
        Ok((self.wrap(lower), self.wrap(upper)))
    }
    fn publish<P: CooperativeWork>(
        &mut self,
        lower_a: Self::Coefficients,
        lower_b: Self::Bias,
        upper_a: Self::Coefficients,
        upper_b: Self::Bias,
        lower_err: Self::Coefficients,
        upper_err: Self::Coefficients,
        work: &mut P,
    ) -> Result<Self::Output> {
        self.observe("publish")?;
        let (la, c0) = lower_a.split();
        let (lb, c1) = lower_b.split();
        let (ua, c2) = upper_a.split();
        let (ub, c3) = upper_b.split();
        let (le, c4) = lower_err.split();
        let (ue, c5) = upper_err.split();
        let guards = [c0, c1, c2, c3, c4, c5];
        let bounds = OwnedCpuStorage.publish(la, lb, ua, ub, le, ue, work)?;
        Ok(TrackedOutput {
            bounds,
            _guards: guards,
        })
    }
}

fn fixture(
    positions: usize,
    with_bias: bool,
    error_mask: usize,
    overflow: bool,
) -> (LinearLayer, LinearBounds) {
    let rows = 2;
    let k = 50;
    let p = 5;
    let layer = LinearLayer::new(
        Array2::from_shape_fn((k, p), |(i, j)| {
            ((i * 7 + j * 3) % 13) as f32 * 0.125 - 0.75
        }),
        with_bias.then(|| Array1::from_shape_fn(k, |j| (j % 5) as f32 * 0.125 - 0.25)),
    )
    .unwrap();
    let lower = Array2::from_shape_fn((rows, positions * k), |(i, j)| {
        if overflow && i == 1 {
            1.0e20
        } else {
            ((i * 5 + j * 11) % 17) as f32 * 0.0625 - 0.5
        }
    });
    let mut bounds = LinearBounds::new(
        lower.clone(),
        Array1::from_vec(vec![0.125, -0.25]),
        -lower,
        Array1::from_vec(vec![0.375, -0.5]),
    )
    .unwrap();
    bounds.lower_a_err =
        (error_mask & 1 != 0).then(|| Array2::from_elem((rows, positions * k), 0.03125));
    bounds.upper_a_err =
        (error_mask & 2 != 0).then(|| Array2::from_elem((rows, positions * k), 0.0625));
    (layer, bounds)
}

fn bits(b: &LinearBounds) -> Vec<u32> {
    let mut out = Vec::new();
    for p in [b.lower_a(), b.upper_a()] {
        out.extend(p.iter().map(|v| v.to_bits()));
    }
    for p in [b.lower_b(), b.upper_b()] {
        out.extend(p.iter().map(|v| v.to_bits()));
    }
    for p in [b.lower_a_err(), b.upper_a_err()] {
        out.push(u32::from(p.is_some()));
        if let Some(p) = p {
            out.extend(p.iter().map(|v| v.to_bits()));
        }
    }
    out
}

#[test]
fn cpu_storage_real_sequence_requests_every_payload_stage_and_preserves_source() {
    let (layer, input) = fixture(3, true, 3, false);
    let saved = bits(&input);
    let mut storage = TrackingStorage::new(None);
    let out = propagate_linear_cpu_with_storage(
        &layer,
        &input,
        None,
        false,
        false,
        &mut NoPoll,
        &mut storage,
    )
    .unwrap();
    assert_eq!(
        storage.trace.borrow().calls,
        vec![
            "admit",
            "coefficient",
            "coefficient",
            "coefficient",
            "coefficient",
            "accumulator",
            "accumulator",
            "row-flags",
            "row-flags",
            "aw",
            "incoming-error",
            "incoming-error",
            "aw",
            "incoming-error",
            "incoming-error",
            "aw",
            "incoming-error",
            "incoming-error",
            "finish-biases",
            "publish",
        ]
    );
    assert_eq!(out.bounds.lower_a().dim(), (2, 15));
    assert_eq!(bits(&input), saved);
    assert_eq!(storage.live.load(Ordering::SeqCst), 6);
    drop(out);
    assert_eq!(storage.live.load(Ordering::SeqCst), 0);
}

#[test]
fn cpu_storage_move_only_owners_keep_six_output_guards_through_overflow_and_drop() {
    for with_bias in [false, true] {
        for overflow in [false, true] {
            let (layer, input) = fixture(1, with_bias, 3, overflow);
            let expected = propagate_linear_cpu_with_deadline_stack_and_work(
                &layer,
                &input,
                None,
                false,
                false,
                &mut NoPoll,
            )
            .unwrap();
            let mut storage = TrackingStorage::new(None);
            let out = propagate_linear_cpu_with_storage(
                &layer,
                &input,
                None,
                false,
                false,
                &mut NoPoll,
                &mut storage,
            )
            .unwrap();
            assert_eq!(bits(&out.bounds), bits(&expected));
            assert_eq!(storage.live.load(Ordering::SeqCst), 6);
            if overflow {
                assert!(out.bounds.lower_a().row(1).iter().all(|&v| v == 0.0));
                assert_eq!(out.bounds.lower_b()[1], f32::NEG_INFINITY);
                assert_eq!(out.bounds.upper_b()[1], f32::INFINITY);
            }
            drop(out);
            assert_eq!(storage.live.load(Ordering::SeqCst), 0);
        }
    }
}

#[test]
fn cpu_storage_every_policy_refusal_discards_all_private_storage_and_preserves_input() {
    for stack in [false, true] {
        let (layer, input) = fixture(3, true, 3, false);
        let saved = bits(&input);
        let mut baseline = TrackingStorage::new(None);
        let out = propagate_linear_cpu_with_storage(
            &layer,
            &input,
            None,
            false,
            stack,
            &mut NoPoll,
            &mut baseline,
        )
        .unwrap();
        let calls = baseline.trace.borrow().calls.len();
        assert!(calls >= 20);
        drop(out);
        assert_eq!(baseline.live.load(Ordering::SeqCst), 0);
        for stop in 1..=calls {
            let mut storage = TrackingStorage::new(Some(stop));
            let result = propagate_linear_cpu_with_storage(
                &layer,
                &input,
                None,
                false,
                stack,
                &mut NoPoll,
                &mut storage,
            );
            assert!(matches!(result, Err(NyError::DeadlineExceeded(_))));
            assert_eq!(storage.trace.borrow().calls.len(), stop);
            assert_eq!(storage.live.load(Ordering::SeqCst), 0);
            assert_eq!(bits(&input), saved);
        }
    }
}

#[test]
fn cpu_storage_final_original_checkpoint_refusal_drops_already_published_output() {
    struct Work {
        calls: usize,
        stop: Option<usize>,
    }
    impl CooperativeWork for Work {
        fn before(&mut self, _: usize) -> Result<()> {
            Ok(())
        }
        fn checkpoint(&mut self) -> Result<()> {
            self.calls += 1;
            if self.stop == Some(self.calls) {
                Err(NyError::DeadlineExceeded(
                    "original final CPU authority".into(),
                ))
            } else {
                Ok(())
            }
        }
    }
    let (layer, input) = fixture(3, true, 3, false);
    let saved = bits(&input);
    let mut storage = TrackingStorage::new(None);
    let mut work = Work {
        calls: 0,
        stop: None,
    };
    let out = propagate_linear_cpu_with_storage(
        &layer,
        &input,
        None,
        false,
        false,
        &mut work,
        &mut storage,
    )
    .unwrap();
    assert!(work.calls > 3);
    let stop = work.calls;
    drop(out);
    let mut storage = TrackingStorage::new(None);
    let mut work = Work {
        calls: 0,
        stop: Some(stop),
    };
    let result = propagate_linear_cpu_with_storage(
        &layer,
        &input,
        None,
        false,
        false,
        &mut work,
        &mut storage,
    );
    assert!(
        matches!(result,Err(NyError::DeadlineExceeded(ref s)) if s=="original final CPU authority")
    );
    assert_eq!(storage.trace.borrow().calls.last(), Some(&"publish"));
    assert_eq!(work.calls, stop);
    assert_eq!(storage.live.load(Ordering::SeqCst), 0);
    assert_eq!(bits(&input), saved);
}

#[test]
fn cpu_storage_optional_stack_bias_and_one_sided_error_policies_keep_all_owned_bits() {
    for with_bias in [false, true] {
        for mask in [0, 1, 2, 3] {
            for stack in [false, true] {
                let (layer, input) = fixture(3, with_bias, mask, false);
                let expected = propagate_linear_cpu_with_deadline_stack_and_work(
                    &layer,
                    &input,
                    None,
                    false,
                    stack,
                    &mut NoPoll,
                )
                .unwrap();
                for deadline in [None, Some(Instant::now() + Duration::from_secs(60))] {
                    let mut storage = TrackingStorage::new(None);
                    let out = propagate_linear_cpu_with_storage(
                        &layer,
                        &input,
                        deadline,
                        false,
                        stack,
                        &mut NoPoll,
                        &mut storage,
                    )
                    .unwrap();
                    assert_eq!(bits(&out.bounds), bits(&expected));
                    let trace = storage.trace.borrow();
                    assert_eq!(
                        trace
                            .calls
                            .iter()
                            .filter(|&&s| s == "position-stack")
                            .count(),
                        usize::from(stack)
                    );
                    assert_eq!(trace.calls.iter().filter(|&&s| s == "aw").count(), 3);
                    assert_eq!(
                        trace
                            .calls
                            .iter()
                            .filter(|&&s| s == "incoming-error")
                            .count(),
                        3 * (usize::from(mask & 1 != 0) + usize::from(mask & 2 != 0))
                    );
                    drop(trace);
                    drop(out);
                    assert_eq!(storage.live.load(Ordering::SeqCst), 0);
                }
            }
        }
    }
}

#[test]
fn cpu_storage_abs_error_cache_stays_cold_without_an_incoming_error_plane() {
    for mask in [0, 1, 2, 3] {
        let (layer, input) = fixture(3, true, mask, false);
        assert!(layer.weight_abs_faer.get().is_none());
        let mut storage = TrackingStorage::new(None);
        let out = propagate_linear_cpu_with_storage(
            &layer,
            &input,
            None,
            false,
            false,
            &mut NoPoll,
            &mut storage,
        )
        .unwrap();
        assert_eq!(layer.weight_abs_faer.get().is_some(), mask != 0);
        assert_eq!(
            storage.trace.borrow().calls.contains(&"incoming-error"),
            mask != 0
        );
        drop(out);
        assert_eq!(storage.live.load(Ordering::SeqCst), 0);
    }
}
