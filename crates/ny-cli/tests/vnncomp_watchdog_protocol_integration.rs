// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Black-box tests for the bidirectional short-grace watchdog protocol.
//!
//! These invoke the real `ny __vnncomp-watchdog` binary. A tiny shell wrapper
//! waits at a pre-exec file gate, letting the parent queue `C` while proving
//! that a pipe write alone produces no ACK and authorizes no preservation.

#![cfg(unix)]

use rustix::time::{clock_gettime, ClockId};
use std::fs;
use std::io::{Read, Write};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

const WATCHDOG_SUBCOMMAND: &str = "__vnncomp-watchdog";
const PARENT_ANCHORED_PROTOCOL: &str = "parent-anchored-monotonic-v2";
const PUBLICATION_COMMIT: &[u8] = b"C";
const PUBLICATION_FINALIZE: &[u8] = b"P";
const PUBLICATION_ACK: u8 = 0x06;
// The debug binary has a large dynamic dependency graph. Keep the semantic
// ordering assertions sub-second, but give a cold exec enough wall-clock room
// on loaded Darwin builders before judging a pre-cutoff ACK absent.
const PRE_CUTOFF_WINDOW: Duration = Duration::from_secs(15);
const ACK_WAIT: Duration = Duration::from_secs(10);

struct ChildGuard {
    child: Option<Child>,
}

impl ChildGuard {
    fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }

    fn child_mut(&mut self) -> &mut Child {
        self.child.as_mut().expect("child already reaped")
    }

    fn id(&self) -> u32 {
        self.child.as_ref().expect("child already reaped").id()
    }

    fn wait(mut self) -> std::io::Result<ExitStatus> {
        self.child.take().expect("child already reaped").wait()
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn ny_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ny"))
}

fn monotonic_deadline(after: Duration) -> u64 {
    let after_nanos = u64::try_from(after.as_nanos()).expect("small test duration");
    let monotonic_now = clock_gettime(ClockId::Monotonic);
    let monotonic_now_nanos = u64::try_from(monotonic_now.tv_sec)
        .expect("positive monotonic seconds")
        .checked_mul(1_000_000_000)
        .and_then(|seconds| {
            seconds.checked_add(u64::try_from(monotonic_now.tv_nsec).expect("positive nanos"))
        })
        .expect("monotonic clock fits u64 nanoseconds");
    monotonic_now_nanos.saturating_add(after_nanos)
}

fn spawn_victim() -> ChildGuard {
    ChildGuard::new(
        Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn disposable verifier victim"),
    )
}

fn spawn_gated_watchdog(
    results_file: &Path,
    parent_pid: u32,
    after: Duration,
) -> (ChildGuard, ChildStdin, PathBuf) {
    let hard_stop_monotonic_nanos = monotonic_deadline(after);
    let ready_file = results_file.with_extension("watchdog-ready");
    let gate_file = results_file.with_extension("watchdog-gate");
    // `$0` is a label. `$1`/`$2` are private readiness/gate paths; after
    // shifting them, `$@` starts with the real ny binary. This deterministic
    // pre-exec gate works on every Unix host without Linux `/proc`. Descriptor
    // 3 keeps the control stream out of the shell's stdin until the immediate
    // exec, so only the real watchdog can consume queued protocol bytes.
    let child = Command::new("sh")
        .arg("-c")
        .arg(
            "exec 3<&0; exec 0</dev/null; ready=$1; gate=$2; shift 2; \
             : > \"$ready\"; while [ ! -e \"$gate\" ]; do sleep 0.01; done; \
             exec 0<&3; exec \"$@\"",
        )
        .arg("ny-watchdog-test")
        .arg(&ready_file)
        .arg(&gate_file)
        .arg(ny_exe())
        .arg(WATCHDOG_SUBCOMMAND)
        .arg(results_file)
        .arg(PARENT_ANCHORED_PROTOCOL)
        .arg(hard_stop_monotonic_nanos.to_string())
        .arg(parent_pid.to_string())
        .arg("preserve-acknowledged")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn gated real watchdog");
    let mut guard = ChildGuard::new(child);
    wait_until_ready(&mut guard, &ready_file);
    let parent_control = guard
        .child_mut()
        .stdin
        .take()
        .expect("watchdog control pipe");
    (guard, parent_control, gate_file)
}

fn wait_until_ready(watchdog: &mut ChildGuard, ready_file: &Path) {
    let deadline = Instant::now() + PRE_CUTOFF_WINDOW;
    loop {
        if ready_file.exists() {
            return;
        }
        assert!(
            watchdog
                .child_mut()
                .try_wait()
                .expect("inspect watchdog wrapper")
                .is_none(),
            "watchdog wrapper exited before opening its pre-exec gate"
        );
        assert!(
            Instant::now() < deadline,
            "watchdog wrapper did not publish pre-exec readiness"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn release_watchdog(gate_file: &Path) {
    fs::write(gate_file, []).expect("release real watchdog from pre-exec gate");
}

fn take_ack_reader(child: &mut ChildGuard) -> ChildStdout {
    child.child_mut().stdout.take().expect("watchdog ACK pipe")
}

fn ack_events(mut reader: ChildStdout) -> Receiver<Instant> {
    let (sender, receiver) = mpsc::sync_channel(4);
    std::thread::spawn(move || {
        let mut bytes = [0u8; 64];
        loop {
            match reader.read(&mut bytes) {
                Ok(0) => return,
                Ok(read) => {
                    for byte in &bytes[..read] {
                        if *byte == PUBLICATION_ACK && sender.send(Instant::now()).is_err() {
                            return;
                        }
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => return,
            }
        }
    });
    receiver
}

#[test]
fn delayed_child_consumption_returns_real_ack_then_post_ack_stall_is_killed() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let results_file = tmp.path().join("result.txt");
    fs::write(&results_file, "unsat\n").expect("write staged verdict");
    let victim = spawn_victim();
    let (mut watchdog, mut control, watchdog_gate) =
        spawn_gated_watchdog(&results_file, victim.id(), PRE_CUTOFF_WINDOW);
    let acknowledgements = ack_events(take_ack_reader(&mut watchdog));

    control
        .write_all(PUBLICATION_COMMIT)
        .expect("queue C while child cannot consume");
    control.flush().expect("flush queued C");
    assert!(
        acknowledgements
            .recv_timeout(Duration::from_millis(75))
            .is_err(),
        "a successful parent pipe write must not synthesize child ACK"
    );

    release_watchdog(&watchdog_gate);
    acknowledgements
        .recv_timeout(ACK_WAIT)
        .expect("real child must ACK after consuming C before cutoff");
    control
        .write_all(PUBLICATION_FINALIZE)
        .expect("place finalize after accepting child ACK");

    // Keep the process-lifetime control lease open. The helper must still fire
    // at its anchored deadline, kill a stalled parent, and preserve only the
    // child-acknowledged verdict.
    let victim_status = victim.wait().expect("reap hard-stopped victim");
    assert_eq!(victim_status.signal(), Some(9));
    assert_eq!(
        fs::read_to_string(&results_file).expect("read preserved verdict"),
        "unsat\n"
    );
    drop(control);
    assert!(watchdog.wait().expect("reap watchdog").success());
}

#[test]
fn child_released_after_cutoff_omits_ack_kills_parent_and_seals_unknown() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let results_file = tmp.path().join("result.txt");
    fs::write(&results_file, "unsat\n").expect("write unacknowledged verdict");
    let victim = spawn_victim();
    let (mut watchdog, mut control, watchdog_gate) =
        spawn_gated_watchdog(&results_file, victim.id(), Duration::from_millis(600));
    let acknowledgements = ack_events(take_ack_reader(&mut watchdog));

    control
        .write_all(PUBLICATION_COMMIT)
        .expect("queue C while child cannot consume");
    std::thread::sleep(Duration::from_millis(800));
    release_watchdog(&watchdog_gate);

    let victim_status = victim.wait().expect("reap hard-stopped victim");
    assert_eq!(victim_status.signal(), Some(9));
    assert!(
        acknowledgements
            .recv_timeout(Duration::from_millis(100))
            .is_err(),
        "a C consumed after the anchored cutoff must never receive ACK"
    );
    assert_eq!(
        fs::read_to_string(&results_file).expect("read fail-closed verdict"),
        "unknown\n"
    );
    drop(control);
    assert!(watchdog.wait().expect("reap watchdog").success());
}

#[test]
fn parent_exit_before_child_ack_cannot_preserve_queued_c() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let results_file = tmp.path().join("result.txt");
    fs::write(&results_file, "unsat\n").expect("write unacknowledged verdict");
    let mut victim = spawn_victim();
    let (mut watchdog, mut control, watchdog_gate) =
        spawn_gated_watchdog(&results_file, victim.id(), PRE_CUTOFF_WINDOW);
    let acknowledgement_reader = take_ack_reader(&mut watchdog);

    control
        .write_all(PUBLICATION_COMMIT)
        .expect("queue C before simulated parent exit");
    // Actual process exit closes both directions. C remains ordered before EOF
    // in the control pipe, but the child cannot return ACK into a closed pipe.
    drop(control);
    drop(acknowledgement_reader);
    release_watchdog(&watchdog_gate);

    assert!(watchdog.wait().expect("reap watchdog").success());
    assert_eq!(
        fs::read_to_string(&results_file).expect("read fail-closed verdict"),
        "unknown\n",
        "queued C without a returned child ACK must not survive parent exit"
    );
    assert!(
        victim
            .child_mut()
            .try_wait()
            .expect("inspect victim")
            .is_none(),
        "EOF retirement before the deadline must not kill an otherwise live parent"
    );
}

#[test]
fn returned_ack_rejected_without_finalize_fails_closed_on_parent_exit() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let results_file = tmp.path().join("result.txt");
    fs::write(&results_file, "unsat\n").expect("write prepared verdict");
    let mut victim = spawn_victim();
    let (mut watchdog, mut control, watchdog_gate) =
        spawn_gated_watchdog(&results_file, victim.id(), PRE_CUTOFF_WINDOW);
    let acknowledgements = ack_events(take_ack_reader(&mut watchdog));

    control
        .write_all(PUBLICATION_COMMIT)
        .expect("queue C while child is at the pre-exec gate");
    release_watchdog(&watchdog_gate);
    acknowledgements
        .recv_timeout(ACK_WAIT)
        .expect("child returns Prepared ACK");

    // Model a parent that rejects the ACK and exits without ever placing P in
    // the ordered pipe. Prepared must not be mistaken for Finalized.
    drop(control);
    assert!(watchdog.wait().expect("reap watchdog").success());
    assert_eq!(
        fs::read_to_string(&results_file).expect("read fail-closed verdict"),
        "unknown\n"
    );
    assert!(
        victim
            .child_mut()
            .try_wait()
            .expect("inspect victim")
            .is_none(),
        "EOF retirement before the deadline must not kill the live victim"
    );
}

#[test]
fn finalize_two_hundred_milliseconds_after_h_seals_unknown() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let results_file = tmp.path().join("result.txt");
    fs::write(&results_file, "unsat\n").expect("write prepared verdict");
    let victim = spawn_victim();
    let (mut watchdog, mut control, watchdog_gate) =
        spawn_gated_watchdog(&results_file, victim.id(), PRE_CUTOFF_WINDOW);
    let acknowledgements = ack_events(take_ack_reader(&mut watchdog));

    control
        .write_all(PUBLICATION_COMMIT)
        .expect("queue C while child is at the pre-exec gate");
    release_watchdog(&watchdog_gate);
    acknowledgements
        .recv_timeout(ACK_WAIT)
        .expect("C must be consumed and ACK returned before the cutoff");

    // The live control lease makes the real helper wait in its ordered
    // post-kill drain after H. Sending P 200 ms after observing the victim's
    // deadline SIGKILL exercises that exact schedule without clock tolerance.
    let victim_status = victim.wait().expect("reap hard-stopped victim at H");
    assert_eq!(victim_status.signal(), Some(9));
    std::thread::sleep(Duration::from_millis(200));
    control
        .write_all(PUBLICATION_FINALIZE)
        .expect("late P still reaches the real helper pipe");
    drop(control);

    assert!(watchdog.wait().expect("reap watchdog").success());
    assert_eq!(
        fs::read_to_string(&results_file).expect("read fail-closed verdict"),
        "unknown\n",
        "P consumed after H must never promote Prepared to Finalized"
    );
}
