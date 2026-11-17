// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Optional ordered compact-root arithmetic. Finite work observes only an
//! already qualified engine; it never initializes or waits for a factory.

use ny_core::packed_root_conv::{check_deadline, Engine, Limits, Output, Request};
use ny_core::Result;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

type Shared = Arc<dyn Engine>;
type Factory = Box<dyn Fn() -> Option<Shared> + Send + Sync>;
static FACTORY: OnceLock<Factory> = OnceLock::new();
static ENGINE: OnceLock<Option<Shared>> = OnceLock::new();

fn published(engine: &OnceLock<Option<Shared>>) -> Option<&Shared> {
    engine.get().and_then(Option::as_ref)
}

/// Register the independently qualified packed-root engine before command work.
pub fn set_factory(factory: impl Fn() -> Option<Shared> + Send + Sync + 'static) {
    let _ = FACTORY.set(Box::new(factory));
}

fn resolve<'a>(
    deadline: Option<Instant>,
    factory: &'a OnceLock<Factory>,
    engine: &'a OnceLock<Option<Shared>>,
) -> Option<&'a Shared> {
    if deadline.is_some() {
        published(engine)
    } else {
        let factory = factory.get()?;
        engine.get_or_init(factory).as_ref()
    }
}

/// Run mandatory module/transport qualification before verifier deadlines exist.
#[must_use]
pub fn prewarm() -> bool {
    resolve(None, &FACTORY, &ENGINE).is_some()
}

/// Observe an already qualified capability without initializing or waiting.
/// Presence permits planning only; the engine still owns resource admission.
pub(super) fn ready() -> bool {
    if crate::sound_f64_gemm::cpu_only_f64_active() {
        return false;
    }
    #[cfg(test)]
    if TEST_ENGINE.with(|slot| slot.borrow().is_some()) {
        return true;
    }
    published(&ENGINE).is_some()
}

pub(super) fn contract(request: &Request<'_>, deadline: Option<Instant>) -> Result<Option<Output>> {
    check_deadline(deadline)?;
    // CPU custody precedes every global or injected-engine observation.
    if crate::sound_f64_gemm::cpu_only_f64_active() {
        return Ok(None);
    }
    #[cfg(test)]
    let injected = TEST_ENGINE.with(|slot| slot.borrow().clone());
    #[cfg(test)]
    let engine = injected
        .as_ref()
        .or_else(|| resolve(deadline, &FACTORY, &ENGINE));
    #[cfg(not(test))]
    let engine = resolve(deadline, &FACTORY, &ENGINE);
    let Some(engine) = engine else {
        return Ok(None);
    };
    // Payload admission uses an amortized, cgroup-clamped headroom estimate.
    // This is an incremental payload budget, not a process RSS guarantee.
    let observed = crate::network::crown_memory::process_memory_headroom_bytes()
        .and_then(|n| usize::try_from(n / 2).ok())
        .unwrap_or(256 * 1024 * 1024);
    let cap = |gib: u64| usize::try_from(gib * (1u64 << 30)).unwrap_or(usize::MAX);
    let limits = Limits {
        deadline,
        max_new_host_bytes: observed.min(cap(4)),
        max_device_bytes: cap(6),
        // Qualified wide owner windows keep each ordered contraction intact.
        // This is a work cap, not a hard wall-time deadline bound.
        max_dispatch_steps: 1 << 28,
    };
    let result = engine.contract(request, limits)?;
    check_deadline(deadline)?;
    Ok(result)
}

#[cfg(test)]
thread_local! {
    static TEST_ENGINE: std::cell::RefCell<Option<Shared>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(super) fn with_test_engine<R>(engine: Shared, run: impl FnOnce() -> R) -> R {
    struct Restore(Option<Shared>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_ENGINE.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(TEST_ENGINE.with(|slot| slot.replace(Some(engine))));
    run()
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// A simple per-coefficient scalar engine, independent of the tiled CPU
    /// implementation. Tests use it to exercise routing and output ownership.
    #[derive(Default)]
    pub(crate) struct Scalar {
        pub(crate) calls: AtomicUsize,
        pub(crate) pointers: Mutex<(usize, usize)>,
    }
    impl Engine for Scalar {
        fn contract(&self, r: &Request<'_>, limits: Limits) -> Result<Option<Output>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let Some(layout) = r.layout(limits)? else {
                return Ok(None);
            };
            let mut output = Output {
                m: vec![0.0; layout.output_len],
                d: vec![0.0; layout.output_len],
            };
            for (i, (m, d)) in output.m.iter_mut().zip(&mut output.d).enumerate() {
                let row = i / r.output_stride;
                let column = i % r.output_stride;
                let (oc, sp) = (row / r.output_spatial, row % r.output_spatial);
                for tap in 0..r.taps {
                    let src_row = r.gather[tap * r.output_spatial + sp];
                    let w = r.weights[oc * r.taps + tap];
                    if src_row == usize::MAX || w == 0.0 {
                        continue;
                    }
                    let src_col = if column + 1 == r.output_stride {
                        Some(r.source_stride - 1)
                    } else {
                        let bucket = sp * r.spatial_taps + tap % r.spatial_taps;
                        let mut source = 0;
                        let mut found = None;
                        for run in &r.runs[r.offsets[bucket]..r.offsets[bucket + 1]] {
                            if (run.dest..run.dest + run.len).contains(&column) {
                                found = Some(source + column - run.dest);
                                break;
                            }
                            source += run.len;
                        }
                        found
                    };
                    if let Some(src_col) = src_col {
                        let source = src_row * r.source_stride + src_col;
                        *m += w * r.m[source];
                        *d += w.abs() * r.d[source];
                    }
                }
            }
            *self.pointers.lock().unwrap() =
                (output.m.as_ptr() as usize, output.d.as_ptr() as usize);
            Ok(Some(output))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Decline;
    impl Engine for Decline {
        fn contract(&self, _: &Request<'_>, _: Limits) -> Result<Option<Output>> {
            Ok(None)
        }
    }
    #[test]
    fn finite_lookup_never_enters_cold_factory_and_prewarm_runs_once() {
        let factory: OnceLock<Factory> = OnceLock::new();
        let engine = OnceLock::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        assert!(published(&engine).is_none());
        let denied: OnceLock<Option<Shared>> = OnceLock::new();
        assert!(denied.set(None).is_ok());
        assert!(published(&denied).is_none());
        assert!(resolve(None, &factory, &engine).is_none());
        assert!(
            engine.get().is_none(),
            "no factory must not cache permanent absence"
        );
        assert!(factory
            .set(Box::new(move || {
                counted.fetch_add(1, Ordering::SeqCst);
                Some(Arc::new(Decline))
            }))
            .is_ok());
        assert!(published(&engine).is_none());
        assert!(resolve(Some(Instant::now()), &factory, &engine).is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let first = resolve(None, &factory, &engine).unwrap();
        assert!(Arc::ptr_eq(first, published(&engine).unwrap()));
        let finite = resolve(Some(Instant::now()), &factory, &engine).unwrap();
        assert!(Arc::ptr_eq(first, finite));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
