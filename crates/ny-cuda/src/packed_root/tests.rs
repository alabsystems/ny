// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn manifest_and_fixture_pack_are_complete_and_pinned() {
    authenticate_contract().unwrap();
    authenticate(SOURCE, "source_sha256").unwrap();
    authenticate(FIXTURES, "fixtures_sha256").unwrap();
    for sm in [86, 120, 121] {
        authenticate(image(sm).unwrap(), &format!("sm_{sm}_sha256")).unwrap();
    }
    assert!(image(90).is_none());
    let fixtures = qualification::fixtures(FIXTURES).unwrap();
    assert_eq!(fixtures.len(), 9);
    assert!(fixtures.iter().any(|f| f.request().output_stride == 1));
    for f in fixtures {
        let request = f.request();
        let layout = request.layout(qualification::limits()).unwrap().unwrap();
        let (gather, inverse) = maps(&request, layout, None).unwrap();
        assert_eq!(gather.len(), request.gather.len());
        assert_eq!(inverse.len(), layout.inverse_len);
        if request.output_stride == 1 {
            assert!(inverse.is_empty());
        }
    }
    for end in [0, 7, 8, 11, 12, FIXTURES.len() - 1] {
        assert!(qualification::fixtures(&FIXTURES[..end]).is_err());
    }
    let mut corrupt = FIXTURES.to_vec();
    corrupt.extend_from_slice(&[0]);
    assert!(qualification::fixtures(&corrupt).is_err());
    corrupt = FIXTURES.to_vec();
    corrupt[36..40].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(qualification::fixtures(&corrupt).is_err());
}

#[cfg(feature = "gpu-tests")]
fn engine() -> PackedRootEngine {
    PackedRootEngine::new(0)
        .expect("packed-root CUDA hardware and adapter qualification required")
        .expect("packed-root native CUDA architecture required")
}

#[cfg(feature = "gpu-tests")]
fn assert_expected(actual: &Output, fixture: &qualification::Fixture) {
    for (lane, values) in [&actual.m, &actual.d].into_iter().enumerate() {
        assert_eq!(
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fixture.expected[lane]
        );
    }
}

#[cfg(feature = "gpu-tests")]
#[test]
fn device_transport_exact_bits_deadline_and_refusal() {
    let engine = engine();
    let before = drv::ctx::get_current().unwrap();
    let fixtures = qualification::fixtures(FIXTURES).unwrap();
    for fixture in &fixtures {
        let request = fixture.request();
        let output = engine
            .contract(&request, qualification::limits())
            .unwrap()
            .unwrap();
        assert_expected(&output, fixture);
        assert_eq!(drv::ctx::get_current().unwrap(), before);
    }
    let request = fixtures[0].request();
    let expired = Limits {
        deadline: Some(Instant::now()),
        ..qualification::limits()
    };
    assert!(matches!(
        engine.contract(&request, expired),
        Err(NyError::DeadlineExceeded(_))
    ));
    let too_small = Limits {
        max_device_bytes: 0,
        ..qualification::limits()
    };
    assert!(engine.contract(&request, too_small).unwrap().is_none());
    assert_eq!(drv::ctx::get_current().unwrap(), before);
    // Teardown must restore a non-null caller context as well as a null one.
    let _restore = RestoreContext::save().unwrap();
    let caller = CudaContext::new_non_primary(0, 0).unwrap();
    let caller_current = drv::ctx::get_current().unwrap();
    assert!(caller_current.is_some());
    let second = PackedRootEngine::new(0).unwrap().unwrap();
    assert_eq!(drv::ctx::get_current().unwrap(), caller_current);
    drop(second);
    assert_eq!(drv::ctx::get_current().unwrap(), caller_current);
    assert_expected(
        &engine
            .contract(&request, qualification::limits())
            .unwrap()
            .unwrap(),
        &fixtures[0],
    );
    assert_eq!(drv::ctx::get_current().unwrap(), caller_current);
    drop(engine);
    assert_eq!(drv::ctx::get_current().unwrap(), caller_current);
    drop(caller);
}

#[cfg(feature = "gpu-tests")]
#[test]
fn device_failed_dispatch_is_drained_and_permanently_denied() {
    let engine = engine();
    let fixtures = qualification::fixtures(FIXTURES).unwrap();
    let request = fixtures[0].request();
    {
        let _restore = RestoreContext::save().unwrap();
        let guard = engine.state.lock().unwrap();
        let state = guard.as_ref().unwrap();
        let audit = Audit {
            invalid_launch: true,
            ..Audit::production()
        };
        assert!(state
            .execute(&request, qualification::limits(), audit)
            .is_err());
        assert!(state.faulted.load(Ordering::Acquire));
    }
    assert!(engine.contract(&request, qualification::limits()).is_err());
}

#[cfg(feature = "gpu-tests")]
#[test]
fn device_post_launch_deadline_discards_output_and_releases_allocations() {
    let engine = engine();
    let fixture = qualification::wide_fixture(2);
    let request = fixture.request();
    assert_eq!(
        request
            .layout(qualification::limits())
            .unwrap()
            .unwrap()
            .owners_per_dispatch,
        116508
    );
    {
        let _restore = RestoreContext::save().unwrap();
        let guard = engine.state.lock().unwrap();
        let state = guard.as_ref().unwrap();
        let audit = Audit {
            expire_after_launch: true,
            ..Audit::production()
        };
        assert!(matches!(
            state.execute(&request, qualification::limits(), audit),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert!(!state.faulted.load(Ordering::Acquire));
    }
    assert_expected(
        &engine
            .contract(&request, qualification::limits())
            .unwrap()
            .unwrap(),
        &fixture,
    );
}

#[cfg(feature = "gpu-tests")]
#[test]
fn device_wide_windows_match_independent_bits() {
    let engine = engine();
    let fixture = qualification::wide_fixture(2);
    for cap in [1 << 24, 1 << 26, 1 << 28] {
        let limits = Limits {
            max_dispatch_steps: cap,
            ..qualification::limits()
        };
        let _restore = RestoreContext::save().unwrap();
        let guard = engine.state.lock().unwrap();
        let state = guard.as_ref().unwrap();
        for chunked in [false, true] {
            let audit = Audit {
                expected: Some([&fixture.expected[0], &fixture.expected[1]]),
                chunked,
                ..Audit::production()
            };
            assert_expected(
                &state
                    .execute(&fixture.request(), limits, audit)
                    .unwrap()
                    .unwrap(),
                &fixture,
            );
        }
    }
}

// Explicit hardware experiment: select benchmarks and NY_PHASE_TELEMETRY=1.
// Qualification/prefix copies and output comparison are outside timed calls.
#[cfg(feature = "benchmarks")]
#[test]
fn device_wide_window_throughput() {
    assert_eq!(std::env::var("NY_PHASE_TELEMETRY").as_deref(), Ok("1"));
    let engine = engine();
    let fixture = qualification::wide_fixture(196);
    let request = fixture.request();
    let limits = Limits {
        max_new_host_bytes: 2 << 30,
        max_device_bytes: 2 << 30,
        ..qualification::limits()
    };
    // Warm each cap; then a balanced Latin square. Same allocation, copy,
    // validation and arithmetic path for all caps, with full exact-bit checks.
    for (round, order) in [[24, 26, 28], [24, 26, 28], [26, 28, 24], [28, 24, 26]]
        .into_iter()
        .enumerate()
    {
        for exponent in order {
            let started = Instant::now();
            let output = engine
                .contract(
                    &request,
                    Limits {
                        max_dispatch_steps: 1 << exponent,
                        ..limits
                    },
                )
                .unwrap()
                .unwrap();
            let elapsed = started.elapsed();
            assert_expected(&output, &fixture);
            eprintln!("NY_PACKED_ROOT_BATCH_BENCH_V1 round={round} cap={} owners={} call_us={} exact=true", 1usize << exponent, output.m.len(), elapsed.as_micros());
        }
    }
}
