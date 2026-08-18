// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// External `ay` process driver: STREAM an SMT-LIB script to stdin from a
// writer thread (so a slow incremental parse can never stall the caller
// past its budget), enforce the wall-clock budget by polling + kill and AY's
// process memory budget through `--memory`, then hand back the collected stdout.

use crate::error::MipError;
use std::io::Write as _;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, MipError>;

/// Per-child AY memory envelope. Each invocation starts one AY worker, whose
/// sibling-blind default may otherwise claim most host RAM. Eight GiB keeps
/// expensive exact solves viable while bounding large-host use and leaving the
/// NY verifier and its model resident.
const AY_MEMORY_LIMIT_MIB: u64 = 8 * 1024;

/// Outcome of one ay invocation.
pub(super) enum AyRun {
    /// Process exited within the budget; stdout captured.
    Completed(String),
    /// Budget exhausted; process killed.
    TimedOut,
}

/// Resolve the ay executable: `$NY_AY` if set, else `ay` on `$PATH` (the
/// same discovery convention as ny's gt/cnf ay routes).
fn ay_command() -> Command {
    match std::env::var_os("NY_AY") {
        Some(path) => Command::new(path),
        None => Command::new("ay"),
    }
}

/// Run ay over `script`, waiting at most `timeout_secs` of wall clock.
pub(super) fn run_ay(script: &str, timeout_secs: f64) -> Result<AyRun> {
    tracing::debug!(
        ay_memory_limit_mib = AY_MEMORY_LIMIT_MIB,
        "launching AY with an explicit per-child memory envelope"
    );
    let mut child = spawn()?;

    // Stream the script from a thread. ay parses stdin INCREMENTALLY, so a
    // slow parse (a 40MB exact-rational MILP script) backpressures the pipe;
    // writing inline would block BEFORE the deadline loop starts — a de-facto
    // unbounded budget (observed: `--timeout 10` on the w2 corpus instance
    // still solving at 99% CPU 30+ minutes later, parent stalled in write).
    // Closing stdin at the end of the thread is what lets batch mode run to
    // EOF; after a deadline kill the write fails with a broken pipe, which is
    // expected and ignored (the run is reported as TimedOut).
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| MipError::Solver("ay: child stdin handle unavailable".to_string()))?;
    let script_owned = script.to_string();
    let writer = std::thread::spawn(move || -> std::io::Result<()> {
        stdin.write_all(script_owned.as_bytes())?;
        stdin.flush()
        // `stdin` drops here, closing the pipe.
    });

    // Drain stdout on a thread so a chatty solver can never fill the pipe and
    // deadlock against our polling loop.
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| MipError::Solver("ay: child stdout handle unavailable".to_string()))?;
    let reader = std::thread::spawn(move || -> std::io::Result<String> {
        use std::io::Read as _;
        let mut out = String::new();
        let mut stdout = stdout;
        stdout.read_to_string(&mut out)?;
        Ok(out)
    });

    let deadline = Instant::now() + Duration::from_secs_f64(timeout_secs.max(0.0));
    let timed_out = loop {
        match child.try_wait() {
            Ok(Some(_status)) => break false,
            Ok(None) => {
                if Instant::now() >= deadline {
                    kill_tree(&mut child);
                    break true;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(e) => {
                kill_tree(&mut child);
                return Err(MipError::Solver(format!("ay: wait failed: {e}")));
            }
        }
    };

    // The kill (or exit) closes both pipes, so both threads always finish.
    let write_res = writer
        .join()
        .map_err(|_| MipError::Solver("ay: stdin writer thread panicked".to_string()))?;
    let output = reader
        .join()
        .map_err(|_| MipError::Solver("ay: stdout reader thread panicked".to_string()))?
        .map_err(|e| MipError::Solver(format!("ay: reading stdout failed: {e}")))?;

    if timed_out {
        // Broken-pipe write errors after the deadline kill are expected.
        return Ok(AyRun::TimedOut);
    }
    if let Err(e) = write_res {
        return Err(MipError::Solver(format!(
            "ay: writing script to stdin failed: {e}"
        )));
    }
    Ok(AyRun::Completed(output))
}

fn spawn() -> Result<Child> {
    let mut cmd = ay_command();
    configure_ay_command(&mut cmd);
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    // Run the child in its OWN process group. The ay CLI RE-EXECS itself at
    // startup (`Command::new(current_exe)` — the worker-thread stack-headroom
    // relaunch), so a plain `child.kill()` only takes the thin outer parent:
    // the re-exec'd solver survives orphaned at 100% CPU holding our pipes
    // (observed: `--timeout 10` on the w2 corpus leaving an `ay -in` running
    // 15+ minutes and the reader thread blocked forever). Making the child a
    // group leader lets the deadline kill take the WHOLE tree.
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
    cmd.spawn().map_err(|e| {
        MipError::Solver(format!(
            "ay: failed to launch (set $NY_AY or put `ay` on $PATH): {e}"
        ))
    })
}

fn configure_ay_command(cmd: &mut Command) {
    cmd.arg("--memory")
        .arg(AY_MEMORY_LIMIT_MIB.to_string())
        .arg("-in");
}

/// Kill the child AND every descendant (ay re-execs itself; see [`spawn`]),
/// then reap. Signals the process group first (the child is its leader), and
/// falls back to the plain child kill for the non-unix / raced cases.
fn kill_tree(child: &mut Child) {
    // The crate forbids `unsafe`, so the group signal goes through /bin/kill
    // instead of libc::kill. A negative pid addresses the process group
    // created by `process_group(0)` in `spawn`.
    #[cfg(unix)]
    {
        let _ = Command::new("/bin/kill")
            .args(["-9", &format!("-{}", child.id())])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ay_child_has_explicit_memory_envelope() {
        let mut command = Command::new("ay");
        configure_ay_command(&mut command);
        let arguments: Vec<_> = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            arguments,
            vec![
                "--memory".to_owned(),
                AY_MEMORY_LIMIT_MIB.to_string(),
                "-in".to_owned(),
            ]
        );
    }
}
