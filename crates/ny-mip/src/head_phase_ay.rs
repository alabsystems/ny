// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bounded AY float advice for a source-owned complete head-phase cover.
//!
//! Only untrusted row multipliers cross this seam. Neither a solver status,
//! objective, primal, model nor an AY certificate can grant neural authority.
//! The caller replays both dual signs against its immutable NY source.
//!
//! The parent bounds the complete helper invocation, including pipe I/O, by
//! the original deadline and a 250 ms legacy / 750 ms V3 batch ceiling. Expiry kills and reaps
//! the helper, then joins both pipe threads before releasing the sole global
//! child permit. If the OS never confirms reap, the invocation declines and
//! global child admission remains occupied for the process lifetime; it must
//! not launch another helper over unresolved cleanup. OS process creation,
//! scheduling, allocation and kill/reap
//! latency are not hard real-time guarantees. The workspace admission and AY
//! memory option are estimates, not an allocator quota at the pinned revision.

use ay_milp::{EngineEconomics, LpSession, Model, Sense, SolveOpts};
use ny_core::head_functional_advice::{
    HeadFunctionalLpHint, HeadFunctionalLpRequest, FUNCTIONAL_MAX_MESSAGE_BYTES,
    FUNCTIONAL_MAX_RESPONSE_BYTES, FUNCTIONAL_PROTOCOL_VERSION,
};
use ny_core::head_phase_advice::{
    HeadPhaseAdvice, HeadPhaseLpHint, HeadPhaseLpRequest, MAX_MESSAGE_BYTES, PROTOCOL_VERSION,
};
use ny_core::head_phase_cover_advice::{
    PhaseCoverLeaf, PhaseCoverLeafHint, PhaseCoverLpHint, PhaseCoverLpRequest,
    PHASE_COVER_MAX_MESSAGE_BYTES, PHASE_COVER_MAX_RESPONSE_BYTES, PHASE_COVER_PROTOCOL_VERSION,
};
use serde::{Deserialize, Serialize};
use std::fs::Metadata;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime};

pub const HELPER_COMMAND: &str = "__head-phase-ay-advice";
pub const FUNCTIONAL_HELPER_COMMAND: &str = "__head-functional-ay-advice";
pub const PHASE_COVER_HELPER_COMMAND: &str = "__head-phase-cover-ay-advice";
// Managed by scripts/bump_ay_pin.py with the dependency manifests and lock.
// This tags untrusted advice; NY's original-source replay grants authority.
const AY_REVISION: &str = "1c059f7b7acf55cddb3e2d09d7cff1167117e186";

/// Immutable AY dependency revision managed with the manifests and lockfile.
pub const fn ay_revision() -> &'static str {
    AY_REVISION
}

const LOCAL_SLICE: Duration = Duration::from_millis(250);
const PHASE_COVER_LOCAL_SLICE: Duration = Duration::from_millis(750);
const POLL_SLICE: Duration = Duration::from_millis(1);
const MAX_RESPONSE_BYTES: usize = 64 * 1024;
const ADVICE_MEMORY_BYTES: usize = 512 * 1024 * 1024;
const FUNCTIONAL_ADVICE_MEMORY_BYTES: usize = 256 * 1024 * 1024;
const PHASE_COVER_ADVICE_MEMORY_BYTES: usize = 256 * 1024 * 1024;
const IO_STACK_BYTES: usize = 256 * 1024;
static ACTIVE_CHILD: AtomicBool = AtomicBool::new(false);

#[derive(Debug, PartialEq, Eq)]
struct ExecutableIdentity {
    len: u64,
    modified: SystemTime,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
}

impl ExecutableIdentity {
    fn from_metadata(metadata: Metadata) -> Result<Self, String> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Ok(Self {
            len: metadata.len(),
            modified: metadata.modified().map_err(|e| e.to_string())?,
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
        })
    }
}

/// Lazy provider for the current first-party NY executable only.
/// On Linux the spawn target is the running image in `/proc`, so replacing
/// the executable's pathname cannot redirect an advice invocation. Other
/// platforms check the snapshotted file identity immediately before spawn.
#[derive(Debug)]
pub struct HeadPhaseAyAdvice {
    executable: PathBuf,
    identity: ExecutableIdentity,
}

impl HeadPhaseAyAdvice {
    pub fn new(current_exe_path: PathBuf) -> Result<Self, String> {
        let current = std::env::current_exe().map_err(|e| e.to_string())?;
        let requested = current_exe_path.canonicalize().map_err(|e| e.to_string())?;
        if requested != current.canonicalize().map_err(|e| e.to_string())? {
            return Err("head-phase advice requires the current NY executable".into());
        }
        #[cfg(target_os = "linux")]
        let executable = PathBuf::from(format!("/proc/{}/exe", std::process::id()));
        #[cfg(not(target_os = "linux"))]
        let executable = requested.clone();
        let identity = ExecutableIdentity::from_metadata(
            std::fs::metadata(&executable).map_err(|e| e.to_string())?,
        )?;
        let requested_identity = ExecutableIdentity::from_metadata(
            std::fs::metadata(&requested).map_err(|e| e.to_string())?,
        )?;
        if identity != requested_identity {
            return Err("head-phase advice executable image changed".into());
        }
        Ok(Self {
            executable,
            identity,
        })
    }

    fn image_unchanged(&self) -> bool {
        std::fs::metadata(&self.executable)
            .ok()
            .and_then(|m| ExecutableIdentity::from_metadata(m).ok())
            .is_some_and(|identity| identity == self.identity)
    }
}

impl HeadPhaseAdvice for HeadPhaseAyAdvice {
    fn propose(&self, request: &HeadPhaseLpRequest, deadline: Instant) -> Option<HeadPhaseLpHint> {
        let started = Instant::now();
        let deadline = deadline.min(started.checked_add(LOCAL_SLICE)?);
        if started >= deadline || !request.validate() || !self.image_unchanged() {
            return None;
        }
        workspace_admission(request)?;
        // Validation precedes every model copy, byte allocation and spawn.
        let encoded = encode_bounded(request, MAX_MESSAGE_BYTES, Some(deadline)).ok()?;
        let mut permit = ChildPermit::acquire(&ACTIVE_CHILD)?;
        if Instant::now() >= deadline || !self.image_unchanged() {
            return None;
        }
        let mut command = Command::new(&self.executable);
        command.arg(HELPER_COMMAND);
        let bytes = exchange(&mut command, encoded, deadline, &mut permit)?;
        if Instant::now() >= deadline {
            return None;
        }
        let hint = decode_hint(&bytes, request)?;
        (Instant::now() < deadline).then_some(hint)
    }

    fn supports_functional_advice(&self) -> bool {
        true
    }

    fn propose_functional(
        &self,
        request: &HeadFunctionalLpRequest,
        deadline: Instant,
    ) -> Option<HeadFunctionalLpHint> {
        let started = Instant::now();
        let deadline = deadline.min(started.checked_add(LOCAL_SLICE)?);
        if started >= deadline || !request.validate() || !self.image_unchanged() {
            return None;
        }
        functional_workspace_admission(request)?;
        let encoded = encode_bounded(request, FUNCTIONAL_MAX_MESSAGE_BYTES, Some(deadline)).ok()?;
        // Both protocols share this single permit, including unresolved reap.
        let mut permit = ChildPermit::acquire(&ACTIVE_CHILD)?;
        if Instant::now() >= deadline || !self.image_unchanged() {
            return None;
        }
        let mut command = Command::new(&self.executable);
        command.arg(FUNCTIONAL_HELPER_COMMAND);
        let bytes = exchange_bounded(
            &mut command,
            encoded,
            deadline,
            &mut permit,
            FUNCTIONAL_MAX_MESSAGE_BYTES,
            FUNCTIONAL_MAX_RESPONSE_BYTES,
        )?;
        if Instant::now() >= deadline {
            return None;
        }
        let hint = decode_functional_hint(&bytes, request)?;
        (Instant::now() < deadline).then_some(hint)
    }

    fn supports_phase_cover_advice(&self) -> bool {
        true
    }

    fn propose_phase_cover(
        &self,
        request: &PhaseCoverLpRequest,
        deadline: Instant,
    ) -> Option<PhaseCoverLpHint> {
        let started = Instant::now();
        let deadline = deadline.min(started.checked_add(PHASE_COVER_LOCAL_SLICE)?);
        if started >= deadline || !request.validate() || !self.image_unchanged() {
            return None;
        }
        phase_cover_workspace_admission(request)?;
        let remaining_micros = deadline
            .saturating_duration_since(Instant::now())
            .as_micros()
            .try_into()
            .ok()?;
        let payload = PhaseCoverHelperPayload {
            remaining_micros,
            request,
        };
        let encoded =
            encode_bounded(&payload, PHASE_COVER_MAX_MESSAGE_BYTES, Some(deadline)).ok()?;
        // V1/V2/V3 share the same permit, including unresolved child cleanup.
        let mut permit = ChildPermit::acquire(&ACTIVE_CHILD)?;
        if Instant::now() >= deadline || !self.image_unchanged() {
            return None;
        }
        let mut command = Command::new(&self.executable);
        command.arg(PHASE_COVER_HELPER_COMMAND);
        let bytes = exchange_bounded(
            &mut command,
            encoded,
            deadline,
            &mut permit,
            PHASE_COVER_MAX_MESSAGE_BYTES,
            PHASE_COVER_MAX_RESPONSE_BYTES,
        )?;
        if Instant::now() >= deadline {
            return None;
        }
        let hint = decode_phase_cover_hint(&bytes, request)?;
        (Instant::now() < deadline).then_some(hint)
    }
}

struct ChildPermit<'a> {
    active: &'a AtomicBool,
    cleanup_confirmed: bool,
}

impl<'a> ChildPermit<'a> {
    fn acquire(active: &'a AtomicBool) -> Option<Self> {
        active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Self {
                active,
                cleanup_confirmed: true,
            })
    }
}

impl Drop for ChildPermit<'_> {
    fn drop(&mut self) {
        if self.cleanup_confirmed {
            self.active.store(false, Ordering::Release);
        }
    }
}

trait ChildControl {
    fn kill(&mut self) -> io::Result<()>;
    fn wait(&mut self) -> io::Result<()>;
}

impl ChildControl for Child {
    fn kill(&mut self) -> io::Result<()> {
        Child::kill(self)
    }

    fn wait(&mut self) -> io::Result<()> {
        Child::wait(self).map(|_| ())
    }
}

struct ChildReaper<'permit, 'counter, C: ChildControl = Child> {
    child: C,
    reaped: bool,
    permit: &'permit mut ChildPermit<'counter>,
}

impl<'permit, 'counter, C: ChildControl> ChildReaper<'permit, 'counter, C> {
    fn new(child: C, permit: &'permit mut ChildPermit<'counter>) -> Self {
        permit.cleanup_confirmed = false;
        Self {
            child,
            reaped: false,
            permit,
        }
    }

    fn stop(&mut self) -> bool {
        if !self.reaped {
            let _ = self.child.kill();
            // A kill error does not preclude a confirmed natural exit. Only
            // successful wait proves cleanup; an unresolved wait keeps global
            // admission occupied even if both pipe threads have finished.
            self.reaped = self.child.wait().is_ok();
            self.permit.cleanup_confirmed = self.reaped;
        }
        self.reaped
    }
}

impl<C: ChildControl> Drop for ChildReaper<'_, '_, C> {
    fn drop(&mut self) {
        self.stop();
    }
}

enum IoEvent {
    Written(io::Result<()>),
    Read(io::Result<Vec<u8>>),
}

/// The private production caller supplies only the fixed current image.
/// Tests can exercise transport faults without routing solver authority.
fn exchange(
    command: &mut Command,
    request: Vec<u8>,
    deadline: Instant,
    permit: &mut ChildPermit<'_>,
) -> Option<Vec<u8>> {
    exchange_bounded(
        command,
        request,
        deadline,
        permit,
        MAX_MESSAGE_BYTES,
        MAX_RESPONSE_BYTES,
    )
}

fn exchange_bounded(
    command: &mut Command,
    request: Vec<u8>,
    deadline: Instant,
    permit: &mut ChildPermit<'_>,
    request_cap: usize,
    response_cap: usize,
) -> Option<Vec<u8>> {
    if Instant::now() >= deadline || request.len() > request_cap {
        return None;
    }
    let child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut process = ChildReaper::new(child, permit);
    let stdin = process.child.stdin.take()?;
    let stdout = process.child.stdout.take()?;
    let received = std::thread::scope(|scope| {
        let (sender, receiver) = mpsc::channel();
        let writer_sender = sender.clone();
        let writer = std::thread::Builder::new()
            .name("ny-head-ay-write".into())
            .stack_size(IO_STACK_BYTES)
            .spawn_scoped(scope, move || {
                let mut stdin = stdin;
                let result = stdin.write_all(&request).and_then(|()| stdin.flush());
                drop(stdin); // EOF is part of the single-message protocol.
                let _ = writer_sender.send(IoEvent::Written(result));
            });
        let Ok(writer) = writer else {
            process.stop();
            return None;
        };
        let reader = std::thread::Builder::new()
            .name("ny-head-ay-read".into())
            .stack_size(IO_STACK_BYTES)
            .spawn_scoped(scope, move || {
                let result = read_bounded(stdout, response_cap);
                let _ = sender.send(IoEvent::Read(result));
            });
        let Ok(reader) = reader else {
            process.stop();
            let _ = writer.join();
            return None;
        };
        let mut written = None;
        let mut received = None;
        let mut status = None;
        let result = loop {
            if Instant::now() >= deadline {
                break None;
            }
            while let Ok(event) = receiver.try_recv() {
                match event {
                    IoEvent::Written(result) => written = Some(result),
                    IoEvent::Read(result) => received = Some(result),
                }
            }
            if written.as_ref().is_some_and(Result::is_err)
                || received.as_ref().is_some_and(Result::is_err)
            {
                break None;
            }
            if status.is_none() {
                match process.child.try_wait() {
                    Ok(Some(exit)) => {
                        process.reaped = true;
                        process.permit.cleanup_confirmed = true;
                        status = Some(exit);
                    }
                    Ok(None) => {}
                    Err(_) => break None,
                }
            }
            if let Some(exit) = status {
                if !exit.success() {
                    break None;
                }
                if written.is_some() && received.is_some() {
                    break received.take().and_then(Result::ok);
                }
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            std::thread::sleep(remaining.min(POLL_SLICE));
        };
        // Our hidden helper spawns no descendants and does not forward either
        // pipe, so killing/reaping it closes the endpoints before these joins.
        let reaped = process.stop();
        let writer_ok = writer.join().is_ok();
        let reader_ok = reader.join().is_ok();
        if reaped && writer_ok && reader_ok && Instant::now() < deadline {
            result
        } else {
            None
        }
    });
    // Reaper drops after scoped I/O joins; the caller's permit drops later.
    received
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct HelperResponse {
    protocol_version: u32,
    ay_revision: String,
    hint: HeadPhaseLpHint,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FunctionalHelperResponse {
    protocol_version: u32,
    ay_revision: String,
    hint: HeadFunctionalLpHint,
}

#[derive(Serialize)]
struct PhaseCoverHelperPayload<'a> {
    remaining_micros: u64,
    request: &'a PhaseCoverLpRequest,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PhaseCoverHelperRequest {
    remaining_micros: u64,
    request: PhaseCoverLpRequest,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PhaseCoverHelperResponse {
    protocol_version: u32,
    ay_revision: String,
    hint: PhaseCoverLpHint,
}

fn decode_phase_cover_hint(
    bytes: &[u8],
    request: &PhaseCoverLpRequest,
) -> Option<PhaseCoverLpHint> {
    if bytes.len() > PHASE_COVER_MAX_RESPONSE_BYTES {
        return None;
    }
    let response: PhaseCoverHelperResponse = serde_json::from_slice(bytes).ok()?;
    if response.protocol_version != PHASE_COVER_PROTOCOL_VERSION
        || response.ay_revision != AY_REVISION
        || !response.hint.validate(request)
    {
        return None;
    }
    Some(response.hint)
}

fn decode_hint(bytes: &[u8], request: &HeadPhaseLpRequest) -> Option<HeadPhaseLpHint> {
    if bytes.len() > MAX_RESPONSE_BYTES {
        return None;
    }
    let response: HelperResponse = serde_json::from_slice(bytes).ok()?;
    if response.protocol_version != PROTOCOL_VERSION
        || response.ay_revision != AY_REVISION
        || !response.hint.validate(request)
    {
        return None;
    }
    Some(response.hint)
}

fn decode_functional_hint(
    bytes: &[u8],
    request: &HeadFunctionalLpRequest,
) -> Option<HeadFunctionalLpHint> {
    if bytes.len() > FUNCTIONAL_MAX_RESPONSE_BYTES {
        return None;
    }
    let response: FunctionalHelperResponse = serde_json::from_slice(bytes).ok()?;
    if response.protocol_version != FUNCTIONAL_PROTOCOL_VERSION
        || response.ay_revision != AY_REVISION
        || !response.hint.validate(request)
    {
        return None;
    }
    Some(response.hint)
}

fn read_bounded(mut reader: impl Read, cap: usize) -> io::Result<Vec<u8>> {
    let mut result = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            return Ok(result);
        }
        if result.len().checked_add(n).is_none_or(|len| len > cap) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "advice message exceeds cap",
            ));
        }
        result.extend_from_slice(&buffer[..n]);
    }
}

struct CappedWriter {
    bytes: Vec<u8>,
    cap: usize,
    deadline: Option<Instant>,
    polled_at: usize,
}

impl Write for CappedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self
            .bytes
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > self.cap)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "advice message exceeds cap",
            ));
        }
        if self.bytes.len().saturating_sub(self.polled_at) >= 4096 {
            if self.deadline.is_some_and(|d| Instant::now() >= d) {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "advice encoding expired",
                ));
            }
            self.polled_at = self.bytes.len();
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn encode_bounded(
    value: &impl Serialize,
    cap: usize,
    deadline: Option<Instant>,
) -> Result<Vec<u8>, String> {
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return Err("advice encoding expired".into());
    }
    let mut writer = CappedWriter {
        bytes: Vec::new(),
        cap,
        deadline,
        polled_at: 0,
    };
    serde_json::to_writer(&mut writer, value).map_err(|e| e.to_string())?;
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return Err("advice encoding expired".into());
    }
    Ok(writer.bytes)
}

fn workspace_admission(request: &HeadPhaseLpRequest) -> Option<usize> {
    let width = request.bounds.len().checked_add(request.rows.len())?;
    let nnz = request.rows.iter().try_fold(0usize, |sum, row| {
        sum.checked_add(row.coefficients.iter().filter(|&&a| a != 0.0).count())
    })?;
    // Include source/request copies, Model/LpSession mirrors, three potential
    // dense float workspaces, bounded pipes, and virtual helper/IO stack
    // reservation. This conservative admission estimate is not an AY quota.
    let estimate = (48usize * 1024 * 1024)
        .checked_add(width.checked_mul(width)?.checked_mul(24)?)?
        .checked_add(nnz.checked_mul(64)?)?
        .checked_add(MAX_MESSAGE_BYTES + MAX_RESPONSE_BYTES + 2 * IO_STACK_BYTES)?;
    (estimate <= ADVICE_MEMORY_BYTES).then_some(estimate)
}

fn functional_workspace_admission(request: &HeadFunctionalLpRequest) -> Option<usize> {
    let width = request.bounds.len().checked_add(request.rows.len())?;
    let nnz = request
        .rows
        .iter()
        .try_fold(0usize, |sum, row| sum.checked_add(row.coefficients.len()))?;
    // Same conservative mirrors/workspace allowance as V1, with the tighter
    // sparse model's 256 MiB admission. This is not an allocator hard limit.
    let estimate = (48usize * 1024 * 1024)
        .checked_add(width.checked_mul(width)?.checked_mul(24)?)?
        .checked_add(nnz.checked_mul(64)?)?
        .checked_add(
            FUNCTIONAL_MAX_MESSAGE_BYTES + FUNCTIONAL_MAX_RESPONSE_BYTES + 2 * IO_STACK_BYTES,
        )?;
    (estimate <= FUNCTIONAL_ADVICE_MEMORY_BYTES).then_some(estimate)
}

fn phase_cover_workspace_admission(request: &PhaseCoverLpRequest) -> Option<usize> {
    let common = request
        .rows
        .iter()
        .try_fold(0usize, |n, r| n.checked_add(r.coefficients.len()))?;
    let mut worst = 0usize;
    for leaf in &request.leaves {
        let rows = request.rows.len().checked_add(leaf.rows.len())?;
        let width = request.bounds.len().checked_add(rows)?;
        let terms = leaf
            .rows
            .iter()
            .try_fold(common, |n, r| n.checked_add(r.coefficients.len()))?;
        let estimate = (48usize * 1024 * 1024)
            .checked_add(width.checked_mul(width)?.checked_mul(24)?)?
            .checked_add(terms.checked_mul(64)?)?
            .checked_add(
                PHASE_COVER_MAX_MESSAGE_BYTES + PHASE_COVER_MAX_RESPONSE_BYTES + 2 * IO_STACK_BYTES,
            )?;
        worst = worst.max(estimate);
    }
    // Sessions are sequential and dropped after each leaf. Aggregate sparse
    // work has a separate wire validator; this estimate is not an AY quota.
    (worst <= PHASE_COVER_ADVICE_MEMORY_BYTES).then_some(worst)
}

fn expired(deadline: Instant) -> Result<(), String> {
    if Instant::now() >= deadline {
        Err("head-phase AY advice expired".into())
    } else {
        Ok(())
    }
}

/// Actual pinned in-process cold float query. No exact/primal/certificate
/// method is called, even when the float lane declines or the model is empty.
fn float_advice(
    request: &HeadPhaseLpRequest,
    deadline: Instant,
) -> Result<HeadPhaseLpHint, String> {
    expired(deadline)?;
    if !request.validate() || workspace_admission(request).is_none() {
        return Err("invalid or oversized head-phase advice request".into());
    }
    let mut model = Model::new();
    let mut cols = Vec::with_capacity(request.bounds.len());
    for (i, &[lower, upper]) in request.bounds.iter().enumerate() {
        if i & 255 == 0 {
            expired(deadline)?;
        }
        cols.push(model.add_col(lower, upper));
    }
    for row in &request.rows {
        expired(deadline)?;
        let mut terms = Vec::new();
        for (i, &a) in row.coefficients.iter().enumerate() {
            if i & 255 == 0 {
                expired(deadline)?;
            }
            if a != 0.0 {
                terms.push((cols[i], a));
            }
        }
        // Exactly one stored coefficient per column: no duplicate-column
        // summation, row transformation, model replacement or hidden phase.
        model.add_row(row.lower, f64::INFINITY, &terms);
    }
    let objective: Vec<_> = request
        .objective
        .iter()
        .enumerate()
        .filter(|&(_, &a)| a != 0.0)
        .map(|(i, &a)| (cols[i], a))
        .collect();
    expired(deadline)?;
    let options = SolveOpts::new()
        .with_deadline(deadline)
        .with_threads(1)
        .with_determinism(true)
        .with_require_certificates(false)
        .with_memory_budget(Some(ADVICE_MEMORY_BYTES))
        .with_tree_cert_leaves(0)
        .with_engine(EngineEconomics::new().with_float_lane(true));
    let mut session = LpSession::new(&model, &options).map_err(|e| e.to_string())?;
    expired(deadline)?;
    let row_duals = session
        .float_dual_for_expr(&objective, Sense::Minimize)
        .map_err(|e| e.to_string())?
        .ok_or("AY float advice declined")?;
    expired(deadline)?;
    let hint = HeadPhaseLpHint { row_duals };
    if !hint.validate(request) {
        return Err("invalid AY row-dual advice".into());
    }
    Ok(hint)
}

/// Sparse global-head search, using the same pinned cold float lane as V1.
/// Row and column order are retained exactly; routing/sign interpretation is
/// owned by NY, and no primal, status or objective crosses the response seam.
fn functional_float_advice(
    request: &HeadFunctionalLpRequest,
    deadline: Instant,
) -> Result<HeadFunctionalLpHint, String> {
    expired(deadline)?;
    if !request.validate() || functional_workspace_admission(request).is_none() {
        return Err("invalid or oversized head-functional advice request".into());
    }
    let mut model = Model::new();
    let mut cols = Vec::with_capacity(request.bounds.len());
    for (i, &[lower, upper]) in request.bounds.iter().enumerate() {
        if i & 255 == 0 {
            expired(deadline)?;
        }
        cols.push(model.add_col(lower, upper));
    }
    for row in &request.rows {
        expired(deadline)?;
        let mut terms = Vec::with_capacity(row.coefficients.len());
        for (i, &(col, coefficient)) in row.coefficients.iter().enumerate() {
            if i & 255 == 0 {
                expired(deadline)?;
            }
            // validate() admitted only sorted unique in-range nonzero terms.
            terms.push((cols[col], coefficient));
        }
        model.add_row(row.lower, f64::INFINITY, &terms);
    }
    let mut objective = Vec::new();
    for (i, &coefficient) in request.objective.iter().enumerate() {
        if i & 255 == 0 {
            expired(deadline)?;
        }
        if coefficient != 0.0 {
            objective.push((cols[i], coefficient));
        }
    }
    expired(deadline)?;
    let options = SolveOpts::new()
        .with_deadline(deadline)
        .with_threads(1)
        .with_determinism(true)
        .with_require_certificates(false)
        .with_memory_budget(Some(FUNCTIONAL_ADVICE_MEMORY_BYTES))
        .with_tree_cert_leaves(0)
        .with_engine(EngineEconomics::new().with_float_lane(true));
    let mut session = LpSession::new(&model, &options).map_err(|e| e.to_string())?;
    expired(deadline)?;
    let row_duals = session
        .float_dual_for_expr(&objective, Sense::Minimize)
        .map_err(|e| e.to_string())?
        .ok_or("AY float functional advice declined")?;
    expired(deadline)?;
    let hint = HeadFunctionalLpHint { row_duals };
    if !hint.validate(request) {
        return Err("invalid AY functional row-dual advice".into());
    }
    Ok(hint)
}

/// Hidden CLI entry, before ordinary NY initialization. Read/write one capped
/// JSON message. Any malformed, expired or declined request returns an error
/// with empty stdout; ordinary CLI diagnostics may go to stderr.
pub fn run_head_phase_ay_helper() -> Result<(), String> {
    let deadline = Instant::now()
        .checked_add(LOCAL_SLICE)
        .ok_or("advice deadline overflow")?;
    let bytes = read_bounded(io::stdin().lock(), MAX_MESSAGE_BYTES).map_err(|e| e.to_string())?;
    expired(deadline)?;
    let request: HeadPhaseLpRequest = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let hint = float_advice(&request, deadline)?;
    let response = HelperResponse {
        protocol_version: PROTOCOL_VERSION,
        ay_revision: AY_REVISION.into(),
        hint,
    };
    let encoded = encode_bounded(&response, MAX_RESPONSE_BYTES, Some(deadline))?;
    expired(deadline)?;
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(&encoded)
        .and_then(|()| stdout.flush())
        .map_err(|e| e.to_string())
}

/// V2 hidden CLI entry, before ordinary initialization. The supervisor owns
/// the original (usually <=125 ms) deadline; this child additionally limits
/// its own cold float query to 250 ms. Malformed input emits no stdout.
pub fn run_head_functional_ay_helper() -> Result<(), String> {
    let deadline = Instant::now()
        .checked_add(LOCAL_SLICE)
        .ok_or("advice deadline overflow")?;
    let bytes = read_bounded(io::stdin().lock(), FUNCTIONAL_MAX_MESSAGE_BYTES)
        .map_err(|e| e.to_string())?;
    expired(deadline)?;
    let request: HeadFunctionalLpRequest =
        serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let hint = functional_float_advice(&request, deadline)?;
    let response = FunctionalHelperResponse {
        protocol_version: FUNCTIONAL_PROTOCOL_VERSION,
        ay_revision: AY_REVISION.into(),
        hint,
    };
    let encoded = encode_bounded(&response, FUNCTIONAL_MAX_RESPONSE_BYTES, Some(deadline))?;
    expired(deadline)?;
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(&encoded)
        .and_then(|()| stdout.flush())
        .map_err(|e| e.to_string())
}

/// One cold AY LP session per leaf, all with the same original helper
/// deadline. No status, primal, objective value or infeasibility certificate
/// is imported; an infeasible/declined/error leaf simply supplies no hint.
fn phase_cover_leaf_float_advice(
    request: &PhaseCoverLpRequest,
    leaf: &PhaseCoverLeaf,
    deadline: Instant,
) -> Result<Option<Vec<f64>>, String> {
    expired(deadline)?;
    let mut model = Model::new();
    let mut cols = Vec::with_capacity(request.bounds.len());
    for (i, &[lower, upper]) in request.bounds.iter().enumerate() {
        if i & 255 == 0 {
            expired(deadline)?;
        }
        cols.push(model.add_col(lower, upper));
    }
    for row in request.rows.iter().chain(&leaf.rows) {
        expired(deadline)?;
        let mut terms = Vec::with_capacity(row.coefficients.len());
        for (i, &(column, coefficient)) in row.coefficients.iter().enumerate() {
            if i & 255 == 0 {
                expired(deadline)?;
            }
            terms.push((cols[column], coefficient));
        }
        model.add_row(row.lower, f64::INFINITY, &terms);
    }
    let mut objective = Vec::new();
    for (i, &coefficient) in request.objective.iter().enumerate() {
        if i & 255 == 0 {
            expired(deadline)?;
        }
        if coefficient != 0.0 {
            objective.push((cols[i], coefficient));
        }
    }
    expired(deadline)?;
    let options = SolveOpts::new()
        .with_deadline(deadline)
        .with_threads(1)
        .with_determinism(true)
        .with_require_certificates(false)
        .with_memory_budget(Some(PHASE_COVER_ADVICE_MEMORY_BYTES))
        .with_tree_cert_leaves(0)
        .with_engine(EngineEconomics::new().with_float_lane(true));
    let mut session = match LpSession::new(&model, &options) {
        Ok(session) => session,
        Err(_) => return Ok(None),
    };
    expired(deadline)?;
    let duals = session
        .float_dual_for_expr(&objective, Sense::Minimize)
        .ok()
        .flatten();
    expired(deadline)?;
    Ok(duals.filter(|values| {
        values.len() == request.rows.len() + leaf.rows.len()
            && values
                .iter()
                .all(|v| v.is_finite() && v.abs() <= ny_core::head_phase_advice::MAX_MAGNITUDE)
    }))
}

fn phase_cover_float_advice(
    request: &PhaseCoverLpRequest,
    deadline: Instant,
) -> Result<PhaseCoverLpHint, String> {
    expired(deadline)?;
    if !request.validate() || phase_cover_workspace_admission(request).is_none() {
        return Err("invalid or oversized head-phase-cover advice request".into());
    }
    // Leave time to encode a partial subset. This never extends the caller's
    // deadline, and every session receives this same absolute solve deadline.
    let solve_deadline = deadline
        .checked_sub(Duration::from_millis(5))
        .unwrap_or(deadline);
    let mut hint = PhaseCoverLpHint { leaves: Vec::new() };
    for leaf in &request.leaves {
        if Instant::now() >= solve_deadline {
            break;
        }
        match phase_cover_leaf_float_advice(request, leaf, solve_deadline) {
            Ok(Some(row_duals)) => hint.leaves.push(PhaseCoverLeafHint {
                mask: leaf.mask,
                row_duals,
            }),
            Ok(None) => {}
            Err(_) => break,
        }
    }
    expired(deadline)?;
    if !hint.validate(request) {
        return Err("invalid AY phase-cover row multipliers".into());
    }
    Ok(hint)
}

/// V3 exact hidden entry. The transmitted remaining allowance is counted
/// from entry, before stdin, decoding and model setup. Parent supervision
/// additionally charges request encoding, process startup, pipes and cleanup
/// against the original deadline; its clock is authoritative for acceptance.
pub fn run_head_phase_cover_ay_helper() -> Result<(), String> {
    let started = Instant::now();
    let local_deadline = started
        .checked_add(PHASE_COVER_LOCAL_SLICE)
        .ok_or("advice deadline overflow")?;
    let bytes = read_bounded(io::stdin().lock(), PHASE_COVER_MAX_MESSAGE_BYTES)
        .map_err(|e| e.to_string())?;
    expired(local_deadline)?;
    let payload: PhaseCoverHelperRequest =
        serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let remaining = Duration::from_micros(payload.remaining_micros);
    if remaining.is_zero() || remaining > PHASE_COVER_LOCAL_SLICE {
        return Err("invalid phase-cover remaining allowance".into());
    }
    let deadline = started
        .checked_add(remaining)
        .ok_or("advice deadline overflow")?
        .min(local_deadline);
    expired(deadline)?;
    let hint = phase_cover_float_advice(&payload.request, deadline)?;
    let response = PhaseCoverHelperResponse {
        protocol_version: PHASE_COVER_PROTOCOL_VERSION,
        ay_revision: AY_REVISION.into(),
        hint,
    };
    let encoded = encode_bounded(&response, PHASE_COVER_MAX_RESPONSE_BYTES, Some(deadline))?;
    expired(deadline)?;
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(&encoded)
        .and_then(|()| stdout.flush())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
#[path = "head_phase_ay_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "head_functional_ay_tests.rs"]
mod functional_tests;

#[cfg(test)]
#[path = "head_phase_cover_ay_tests.rs"]
mod phase_cover_tests;
