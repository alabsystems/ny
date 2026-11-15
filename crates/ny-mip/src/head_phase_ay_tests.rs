// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ny_core::head_phase_advice::{HeadPhaseLpRow, MAX_MAGNITUDE};

fn dyadic_request(mask: usize) -> HeadPhaseLpRequest {
    // x in [-1,1], y1=x, y2=x+1/2. The source margin has exact
    // minimum 1/64; the 13/64 objective constant does not affect AY duals.
    let s1 = if mask & 1 != 0 { 1.0 } else { 0.0 };
    let s2 = if mask & 2 != 0 { 1.0 } else { 0.0 };
    HeadPhaseLpRequest {
        protocol_version: PROTOCOL_VERSION,
        bounds: vec![[-1.0, 1.0], [-1.0, 1.0], [-0.5, 1.5]],
        objective: vec![0.0, s1, -0.375 * s2],
        rows: vec![
            HeadPhaseLpRow {
                coefficients: vec![-1.0, 1.0, 0.0],
                lower: 0.0,
            },
            HeadPhaseLpRow {
                coefficients: vec![1.0, -1.0, 0.0],
                lower: 0.0,
            },
            HeadPhaseLpRow {
                coefficients: vec![0.0, 2.0 * s1 - 1.0, 0.0],
                lower: 0.0,
            },
            HeadPhaseLpRow {
                coefficients: vec![-1.0, 0.0, 1.0],
                lower: 0.5,
            },
            HeadPhaseLpRow {
                coefficients: vec![1.0, 0.0, -1.0],
                lower: -0.5,
            },
            HeadPhaseLpRow {
                coefficients: vec![0.0, 0.0, 2.0 * s2 - 1.0],
                lower: 0.0,
            },
        ],
        phase_rows: vec![2, 5],
    }
}

fn response(request: &HeadPhaseLpRequest, dual: f64) -> Vec<u8> {
    encode_bounded(
        &HelperResponse {
            protocol_version: PROTOCOL_VERSION,
            ay_revision: AY_REVISION.into(),
            hint: HeadPhaseLpHint {
                row_duals: vec![dual; request.rows.len()],
            },
        },
        MAX_RESPONSE_BYTES,
        None,
    )
    .unwrap()
}

#[test]
fn pinned_float_advice_finds_missing_dyadic_phase_candidates() {
    assert!(include_str!("../Cargo.toml").contains(AY_REVISION));
    for mask in [2, 3] {
        let request = dyadic_request(mask);
        let hint = float_advice(&request, Instant::now() + Duration::from_secs(10)).unwrap();
        assert!(hint.validate(&request));
        let mut best = f64::NEG_INFINITY;
        for sign in [1.0, -1.0] {
            let a = (sign * hint.row_duals[2]).max(0.0);
            let b = (sign * hint.row_duals[5]).max(0.0);
            // This is independent analytic replay of the authored dyadic
            // affine source, including its COMPLETE nonzero residual. No
            // solver objective/status or primal is an authority premise.
            let residual = if mask == 3 {
                0.625 - a - b
            } else {
                a - 0.375 - b
            };
            let lower = 13.0 / 64.0 - f64::midpoint(0.375, b) - residual.abs();
            best = best.max(lower);
        }
        assert!(
            best > 0.0,
            "actual cold AY hints must admit a positive source replay"
        );
        assert!(
            best <= 1.0 / 64.0,
            "advice cannot beat this source's exact minimum"
        );
    }
    // Integer arithmetic proves that the complete current native finite
    // grid cannot reach a positive bound on either missing phase.
    for mask in [2, 3] {
        let mut maximum = i64::MIN;
        for a in [0i64, 8, 16, 32, 64, 128, 256] {
            for b in [0i64, 3, 6, 12, 24, 48, 96] {
                let residual = if mask == 3 { 20 - a - b } else { a - 12 - b };
                maximum = maximum.max(13 - (12 + b) - 2 * residual.abs());
            }
        }
        assert_eq!(maximum, -4); // maximum -1/16 in both closed phases
    }
}

#[test]
fn pinned_float_advice_crosses_the_two_coordinate_cancellation_ridge() {
    // Source: y1=x1+x2+1/16, y2=-x1+x2+1/16, stable y3=x2+2;
    // 97/64+ReLU(y1)+ReLU(y2)-3/4*y3 has exact minimum 1/16.
    // Query the all-inactive closed phase; x2 is an INTERIOR optimum so
    // both phase multipliers must balance a continuous cancellation ridge.
    let request = HeadPhaseLpRequest {
        protocol_version: PROTOCOL_VERSION,
        bounds: vec![
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-31.0 / 16.0, 33.0 / 16.0],
            [-31.0 / 16.0, 33.0 / 16.0],
        ],
        objective: vec![0.0, -0.75, 0.0, 0.0],
        rows: vec![
            HeadPhaseLpRow {
                coefficients: vec![-1.0, -1.0, 1.0, 0.0],
                lower: 1.0 / 16.0,
            },
            HeadPhaseLpRow {
                coefficients: vec![1.0, 1.0, -1.0, 0.0],
                lower: -1.0 / 16.0,
            },
            HeadPhaseLpRow {
                coefficients: vec![0.0, 0.0, -1.0, 0.0],
                lower: 0.0,
            },
            HeadPhaseLpRow {
                coefficients: vec![1.0, -1.0, 0.0, 1.0],
                lower: 1.0 / 16.0,
            },
            HeadPhaseLpRow {
                coefficients: vec![-1.0, 1.0, 0.0, -1.0],
                lower: -1.0 / 16.0,
            },
            HeadPhaseLpRow {
                coefficients: vec![0.0, 0.0, 0.0, -1.0],
                lower: 0.0,
            },
        ],
        phase_rows: vec![2, 5],
    };
    let hint = float_advice(&request, Instant::now() + Duration::from_secs(10)).unwrap();
    assert!(hint.validate(&request));
    let mut best_cover = f64::NEG_INFINITY;
    for sign in [1.0, -1.0] {
        let anchor = [
            (sign * hint.row_duals[2]).max(0.0),
            (sign * hint.row_duals[5]).max(0.0),
        ];
        let mut cover = f64::INFINITY;
        for mask in 0..4 {
            let mut residual = [0.0; 2];
            for i in 0..2 {
                let active = mask & (1 << i) != 0;
                let p = if active {
                    (1.0 - anchor[i]).max(0.0)
                } else {
                    anchor[i]
                };
                residual[i] = if active { 1.0 - p } else { p };
            }
            let [a, b] = residual;
            // Source-owned replay separately checks this complete portfolio.
            // Here the analytic affine reduction retains BOTH x residuals;
            // no AY zero reduced-cost or optimum claim grants a bound.
            let lower = 1.0 / 64.0 + (a + b) / 16.0 - (a - b).abs() - (a + b - 0.75).abs();
            cover = cover.min(lower);
        }
        best_cover = best_cover.max(cover);
    }
    assert!(
        best_cover > 0.0,
        "actual float hints must improve the complete source cover"
    );
    assert!(best_cover <= 1.0 / 16.0);
    // The zero proposal and an isolated coordinate step stay negative.
    let native_zero = 1.0 / 64.0 - 0.75;
    assert!(native_zero < 0.0);
    for a in [0.0_f64, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0] {
        let one_coordinate = 1.0 / 64.0 + a / 16.0 - a - (a - 0.75).abs();
        assert!(one_coordinate < 0.0);
    }
}

#[test]
fn bounded_encoding_and_decoding_refuse_invalid_transport() {
    let request = dyadic_request(3);
    assert!(request.validate());
    assert!(workspace_admission(&request).is_some());
    assert!(encode_bounded(&request, 4, None).is_err());
    assert!(encode_bounded(&request, MAX_MESSAGE_BYTES, Some(Instant::now())).is_err());
    let valid = response(&request, -0.625);
    assert_eq!(decode_hint(&valid, &request).unwrap().row_duals[0], -0.625);
    assert!(decode_hint(b"{}", &request).is_none());
    assert!(decode_hint(&vec![b' '; MAX_RESPONSE_BYTES + 1], &request).is_none());
    let mut body: serde_json::Value = serde_json::from_slice(&valid).unwrap();
    body["protocol_version"] = serde_json::json!(99);
    assert!(decode_hint(&serde_json::to_vec(&body).unwrap(), &request).is_none());
    body["protocol_version"] = serde_json::json!(PROTOCOL_VERSION);
    body["ay_revision"] = serde_json::json!("wrong-pin");
    assert!(decode_hint(&serde_json::to_vec(&body).unwrap(), &request).is_none());
    body["ay_revision"] = serde_json::json!(AY_REVISION);
    body["hint"]["row_duals"] = serde_json::json!([0]);
    assert!(decode_hint(&serde_json::to_vec(&body).unwrap(), &request).is_none());
    body["hint"]["row_duals"] = serde_json::json!(vec![2.0 * MAX_MAGNITUDE; request.rows.len()]);
    assert!(decode_hint(&serde_json::to_vec(&body).unwrap(), &request).is_none());
    body["hint"]["row_duals"] = serde_json::json!(vec![0.0; request.rows.len()]);
    body["hint"]["lower"] = serde_json::json!(1.0);
    assert!(decode_hint(&serde_json::to_vec(&body).unwrap(), &request).is_none());
    assert!(read_bounded(&b"12345"[..], 4).is_err());
}

#[test]
fn malformed_and_expired_requests_decline_before_ay_model_work() {
    let request = dyadic_request(3);
    assert!(float_advice(&request, Instant::now()).is_err());
    let mut bad = request.clone();
    bad.rows[0].coefficients.push(0.0);
    assert!(float_advice(&bad, Instant::now() + LOCAL_SLICE).is_err());
    let mut bad = request;
    bad.rows[0].lower = f64::NAN;
    assert!(float_advice(&bad, Instant::now() + LOCAL_SLICE).is_err());
}

#[test]
fn permit_has_one_owner_and_releases_only_on_drop() {
    let active = AtomicBool::new(false);
    let permit = ChildPermit::acquire(&active).unwrap();
    assert!(ChildPermit::acquire(&active).is_none());
    assert!(active.load(Ordering::Acquire));
    drop(permit);
    assert!(!active.load(Ordering::Acquire));
    assert!(ChildPermit::acquire(&active).is_some());
}

#[test]
fn unconfirmed_reap_keeps_admission_occupied_and_confirmed_retry_releases_it() {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    struct FailingChild {
        kill_calls: Arc<AtomicUsize>,
        wait_calls: Arc<AtomicUsize>,
        wait_failures: usize,
    }
    impl ChildControl for FailingChild {
        fn kill(&mut self) -> io::Result<()> {
            self.kill_calls.fetch_add(1, Ordering::Relaxed);
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "injected kill failure",
            ))
        }
        fn wait(&mut self) -> io::Result<()> {
            self.wait_calls.fetch_add(1, Ordering::Relaxed);
            if self.wait_failures > 0 {
                self.wait_failures -= 1;
                Err(io::Error::other("injected unconfirmed reap"))
            } else {
                Ok(())
            }
        }
    }
    for failures in [1, 2] {
        let active = AtomicBool::new(false);
        let mut permit = ChildPermit::acquire(&active).unwrap();
        let kill_calls = Arc::new(AtomicUsize::new(0));
        let wait_calls = Arc::new(AtomicUsize::new(0));
        let child = FailingChild {
            kill_calls: Arc::clone(&kill_calls),
            wait_calls: Arc::clone(&wait_calls),
            wait_failures: failures,
        };
        let mut reaper = ChildReaper::new(child, &mut permit);
        assert!(!reaper.stop());
        assert!(active.load(Ordering::Acquire));
        // Drop retries cleanup once, but never treats an error as evidence
        // of exit. A confirmed wait may release even if kill itself failed.
        drop(reaper);
        drop(permit);
        assert_eq!(kill_calls.load(Ordering::Relaxed), 2);
        assert_eq!(wait_calls.load(Ordering::Relaxed), 2);
        if failures == 2 {
            assert!(active.load(Ordering::Acquire));
            assert!(ChildPermit::acquire(&active).is_none());
        } else {
            assert!(!active.load(Ordering::Acquire));
            assert!(ChildPermit::acquire(&active).is_some());
        }
    }
}

#[test]
fn provider_rejects_other_executable_and_expired_entry() {
    assert!(HeadPhaseAyAdvice::new(PathBuf::from("not-the-current-ny-binary")).is_err());
    let provider = HeadPhaseAyAdvice::new(std::env::current_exe().unwrap()).unwrap();
    assert!(provider.image_unchanged());
    assert!(provider
        .propose(&dyadic_request(3), Instant::now())
        .is_none());
}

#[cfg(unix)]
fn shell(script: &str) -> Command {
    // OS transport fixture, never a solver or a production provider path.
    let mut command = Command::new("/bin/sh");
    command.args(["-c", script]);
    command
}

#[cfg(unix)]
fn exchange_for_test(
    command: &mut Command,
    request: Vec<u8>,
    deadline: Instant,
) -> Option<Vec<u8>> {
    let active = AtomicBool::new(false);
    let mut permit = ChildPermit::acquire(&active).unwrap();
    exchange(command, request, deadline, &mut permit)
}

#[cfg(unix)]
#[test]
fn child_reaper_confirms_exit_before_releasing_ownership() {
    let child = shell("while :; do :; done")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    #[cfg(target_os = "linux")]
    let pid = child.id();
    let active = AtomicBool::new(false);
    let mut permit = ChildPermit::acquire(&active).unwrap();
    let mut process = ChildReaper::new(child, &mut permit);
    assert!(process.stop());
    assert!(process.reaped);
    assert!(process.child.try_wait().unwrap().is_some());
    #[cfg(target_os = "linux")]
    assert!(!PathBuf::from(format!("/proc/{pid}")).exists());
    drop(process);
    drop(permit);
    assert!(!active.load(Ordering::Acquire));
}

#[cfg(unix)]
#[test]
fn transport_drains_bounded_receipt_and_reaps_before_return() {
    let mut command = shell("cat >/dev/null; printf '%s' '{\"ok\":true}'");
    let bytes = exchange_for_test(
        &mut command,
        vec![b'x'; 1024 * 1024],
        Instant::now() + Duration::from_secs(2),
    )
    .unwrap();
    assert_eq!(bytes, b"{\"ok\":true}");
}

#[cfg(unix)]
#[test]
fn transport_kills_expired_pipe_writer_and_oversized_reader() {
    // A shell builtin busy loop retains the stdin pipe without consuming it.
    // Expiry must kill/reap it, unblock the writer and join both I/O threads.
    let active = AtomicBool::new(false);
    let mut permit = ChildPermit::acquire(&active).unwrap();
    let started = Instant::now();
    let mut blocked = shell("while :; do :; done");
    assert!(exchange(
        &mut blocked,
        vec![b'x'; MAX_MESSAGE_BYTES],
        started + Duration::from_millis(30),
        &mut permit,
    )
    .is_none());
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "cleanup must not retain live pipe work"
    );
    assert!(
        active.load(Ordering::Acquire),
        "permit lives through all cleanup"
    );
    drop(permit);
    assert!(!active.load(Ordering::Acquire));
    // Reader refuses at 64 KiB and actively terminates a never-ending sender.
    let mut flooded = shell("while :; do printf '0123456789012345678901234567890123456789012345678901234567890123456789'; done");
    assert!(exchange_for_test(
        &mut flooded,
        Vec::new(),
        Instant::now() + Duration::from_secs(2)
    )
    .is_none());
    let mut failed = shell("exit 7");
    assert!(exchange_for_test(
        &mut failed,
        Vec::new(),
        Instant::now() + Duration::from_secs(2)
    )
    .is_none());
}
