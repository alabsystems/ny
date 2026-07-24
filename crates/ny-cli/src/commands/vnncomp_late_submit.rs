// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! VNN-COMP 2026 late-submission automation.
//!
//! Drives the official evaluation platform (<https://vnn.repeatability.cps.cit.tum.de/>)
//! through the same JSON API the web UI uses: Django session + CSRF cookies
//! (kept in a curl cookie jar), `POST /api/signup/`, `POST /api/login/`,
//! `GET /api/toolkit/form-data/` (submission gates + option lists), and the
//! toolkit submission `POST /api/toolkit/submit/`.
//!
//! The 2026 tool-submission window closed on 2026-06-30 AoE and the final
//! evaluation ran on 2026-07-10 (vnncomp2026 issues #9/#13), so the platform
//! is expected to report `can_submit == false` for new toolkits. This command
//! therefore automates everything that is still mechanical — account signup,
//! gate probing, and a fully-formed all-tracks submission attempt — and
//! drafts the late-entry request email to the evaluation chairs, which is the
//! only documented path once the server-side window is closed.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::fs;
use std::io::Read as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Live evaluation platform (the pre-2025 vnncomp.christopher-brix.de host
/// 307-redirects here; GETs follow redirects so either URL works).
const DEFAULT_PLATFORM_URL: &str = "https://vnn.repeatability.cps.cit.tum.de";

/// Environment variable consulted for the platform password before the
/// credentials file.
const PASSWORD_ENV: &str = "NY_VNNCOMP_PASSWORD";

/// Regular-track benchmark ids as hardcoded in the platform SPA bundle
/// (constant year '2026') and announced in vnncomp2026 issue #6 (2026-06-07
/// benchmark-voting results: >=50% of the 8 tool-author votes).
const REGULAR_TRACK_2026: [&str; 24] = [
    "acasxu_2023",
    "cersyve",
    "cgan2026",
    "challenging_certified_training_2026",
    "cifar100_2024",
    "collins_rul_cnn_2022",
    "cora_2024",
    "dist_shift_2023",
    "linearizenn_2024",
    "lsnc_relu",
    "malbeware",
    "metaroom_2023",
    "ml4acopf_2024",
    "nn4sys",
    "relusplitter_2026",
    "safenlp_2024",
    "sat_relu",
    "soundnessbench_2026",
    "tinyimagenet_2024",
    "tllverifybench_2023",
    "traffic_signs_recognition_2023",
    "vggnet16_2022",
    "vit_2023",
    "yolo_2023",
];

/// Extended-track benchmark ids (>=1 vote, not regular; same sources).
const EXTENDED_TRACK_2026: [&str; 6] = [
    "adaptive_cruise_control_non_linear_2026",
    "cctsdb_yolo_2023",
    "collins_aerospace_benchmark",
    "isomorphic_acasxu_2026",
    "monotonic_acasxu_2026",
    "smart_turn_multimodal_2026",
];

/// Evaluation-chair contacts listed on the platform landing page.
const EVALUATION_CHAIRS: [(&str, &str); 2] = [
    ("Tobias Ladner", "tobias.ladner@tum.de"),
    ("Konstantin Kaulen", "kaulen@aim.rwth-aachen.de"),
];

/// 2026 submission-timeline facts (source: vnncomp2026 issues #9, #12, #13).
const TIMELINE_2026: [(&str, &str); 5] = [
    ("2026-06-08", "tool-submission window opened (issue #9)"),
    (
        "2026-06-30",
        "tool-submission window closed, AoE (issue #13)",
    ),
    (
        "2026-07-10",
        "final evaluation completed on a fresh seed (issue #13)",
    ),
    (
        "2026-07-16",
        "counterexample-format fix deadline, 23:59 AoE (issue #13)",
    ),
    ("2026-07-20", "results freeze for FLoC (issue #12)"),
];

/// Actions under `ny vnncomp-late-submit`.
#[derive(clap::Subcommand, Debug)]
pub(crate) enum LateSubmitAction {
    /// Probe platform reachability, session state, and submission gates.
    Status {
        #[command(flatten)]
        platform: PlatformOpts,
    },

    /// Create an evaluation-platform account (organizer activation stays manual).
    Signup {
        /// Account holder name shown to the organizers.
        #[arg(long)]
        name: String,

        /// Account email (also the login username).
        #[arg(long)]
        email: String,

        #[command(flatten)]
        platform: PlatformOpts,
    },

    /// Log in and persist the session cookie jar.
    Login {
        /// Login email (default: stored credentials).
        #[arg(long)]
        email: Option<String>,

        #[command(flatten)]
        platform: PlatformOpts,
    },

    /// Submit the toolkit for the selected tracks (default: both tracks).
    Submit(Box<SubmitArgs>),

    /// Poll the status of a submission task (id from `submit`'s response).
    Task {
        /// Task id returned by the submission POST.
        id: u64,

        #[command(flatten)]
        platform: PlatformOpts,
    },

    /// Draft the late-entry request email to the evaluation chairs.
    RequestEmail(Box<RequestEmailArgs>),
}

/// Options shared by every platform-touching action.
#[derive(clap::Args, Debug)]
pub(crate) struct PlatformOpts {
    /// Evaluation-platform base URL.
    #[arg(long, default_value = DEFAULT_PLATFORM_URL)]
    platform_url: String,

    /// Cookie-jar path holding the Django session (default: ~/.ny/vnncomp2026.cookies).
    #[arg(long)]
    cookie_jar: Option<PathBuf>,

    /// Output as JSON.
    #[arg(long, default_value_t = false)]
    json: bool,
}

/// Which track's benchmarks to select.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrackSelection {
    /// Regular + Extended (all 30 scored benchmarks).
    All,
    /// Regular track only (24 benchmarks).
    Regular,
    /// Extended track only (6 benchmarks).
    Extended,
}

/// Platform evaluation mode (`run_networks` field).
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RunMode {
    /// Smoke mode: the platform samples 10 instances per benchmark.
    Random,
    /// Full evaluation over every instance (organizer-funded AWS time).
    All,
    /// Platform 'different' sampling mode.
    Different,
    /// First instance of each benchmark only.
    First,
}

impl RunMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::All => "all",
            Self::Different => "different",
            Self::First => "first",
        }
    }
}

/// AWS platform choice from rules.md (one platform for all benchmarks).
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InstanceType {
    /// m5.16xlarge: 64 vCPU, 256 GB RAM.
    Cpu,
    /// p3.2xlarge: 8 vCPU, 61 GB RAM, 1x V100.
    Gpu,
    /// g5.8xlarge: 32 vCPU, 128 GB RAM, 1x A10G.
    Balanced,
}

impl InstanceType {
    const fn label_hints(self) -> &'static [&'static str] {
        match self {
            Self::Cpu => &["m5.16xlarge", "cpu"],
            Self::Gpu => &["p3.2xlarge", "gpu"],
            Self::Balanced => &["g5.8xlarge", "balanced"],
        }
    }

    const fn describe(self) -> &'static str {
        match self {
            Self::Cpu => "CPU (m5.16xlarge, 64 vCPU, 256 GB)",
            Self::Gpu => "GPU (p3.2xlarge, 8 vCPU, 61 GB, 1x V100)",
            Self::Balanced => "Balanced (g5.8xlarge, 32 vCPU, 128 GB)",
        }
    }
}

/// Arguments for `submit`.
#[derive(clap::Args, Debug)]
pub(crate) struct SubmitArgs {
    /// Tracks whose benchmarks are submitted.
    #[arg(long, value_enum, default_value_t = TrackSelection::All)]
    tracks: TrackSelection,

    /// Extra benchmark id to include (repeatable), on top of the track selection.
    #[arg(long = "benchmark")]
    benchmarks: Vec<String>,

    /// Skip the tiny 'test' benchmark (included by default as an install check).
    #[arg(long, default_value_t = false)]
    skip_test: bool,

    /// Evaluation mode. 'random' (default) smoke-samples 10 instances per
    /// benchmark; 'all' runs the full evaluation and additionally needs --yes.
    #[arg(long, value_enum, default_value_t = RunMode::Random)]
    mode: RunMode,

    /// AWS platform (rules.md: one platform for all benchmarks).
    #[arg(long, value_enum, default_value_t = InstanceType::Cpu)]
    instance_type: InstanceType,

    /// AMI label substring to prefer (default: an Ubuntu 24.04 option).
    #[arg(long)]
    ami: Option<String>,

    /// Toolkit display name on the platform.
    #[arg(long, default_value = "NY")]
    name: String,

    /// Git clone URL (default: this repo's `origin` remote).
    #[arg(long)]
    repository: Option<String>,

    /// Commit hash to submit (default: current HEAD).
    #[arg(long)]
    commit: Option<String>,

    /// Preferred VNN-LIB version (2.0-only benchmarks fall back automatically).
    #[arg(long, default_value = "1.0")]
    vnnlib_version: String,

    /// Login email (default: stored credentials).
    #[arg(long)]
    email: Option<String>,

    /// Print gates + payload without POSTing (works offline).
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// POST even if the platform reports the submission window closed.
    #[arg(long, default_value_t = false)]
    force: bool,

    /// Acknowledge the cost warning required for --mode all.
    #[arg(long, default_value_t = false)]
    yes: bool,

    #[command(flatten)]
    platform: PlatformOpts,
}

/// Arguments for `request-email`.
#[derive(clap::Args, Debug)]
pub(crate) struct RequestEmailArgs {
    /// Output .eml path (relative paths resolve against the repo root).
    #[arg(short, long, default_value = "dist/vnncomp2026-late-entry-request.eml")]
    output: PathBuf,

    /// Sender name (default: git config user.name).
    #[arg(long)]
    from_name: Option<String>,

    /// Sender email (default: git config user.email).
    #[arg(long)]
    from_email: Option<String>,

    /// Tracks requested in the draft.
    #[arg(long, value_enum, default_value_t = TrackSelection::All)]
    tracks: TrackSelection,

    /// AWS platform requested in the draft.
    #[arg(long, value_enum, default_value_t = InstanceType::Cpu)]
    instance_type: InstanceType,

    /// Git clone URL (default: this repo's `origin` remote).
    #[arg(long)]
    repository: Option<String>,

    /// Commit hash quoted in the draft (default: current HEAD).
    #[arg(long)]
    commit: Option<String>,

    /// Platform account email mentioned in the draft (default: stored credentials).
    #[arg(long)]
    account_email: Option<String>,

    /// Output as JSON.
    #[arg(long, default_value_t = false)]
    json: bool,
}

/// Entry point for `ny vnncomp-late-submit`.
pub(crate) fn handle_vnncomp_late_submit_command(action: LateSubmitAction) -> Result<()> {
    match action {
        LateSubmitAction::Status { platform } => handle_status(&platform),
        LateSubmitAction::Signup {
            name,
            email,
            platform,
        } => handle_signup(&name, &email, &platform),
        LateSubmitAction::Login { email, platform } => handle_login(email.as_deref(), &platform),
        LateSubmitAction::Submit(args) => handle_submit(&args),
        LateSubmitAction::Task { id, platform } => handle_task(id, &platform),
        LateSubmitAction::RequestEmail(args) => handle_request_email(&args),
    }
}

fn handle_task(id: u64, platform: &PlatformOpts) -> Result<()> {
    let client = PlatformClient::new(platform)?;
    let resp = client.get(&format!("/api/toolkit/task-status/{id}/"))?;
    if !resp.ok() {
        bail!(
            "task-status for {id} returned HTTP {}: {}",
            resp.status,
            resp.body.trim()
        );
    }
    let status = resp.json().unwrap_or(Value::Null);
    if platform.json {
        emit_json(&json!({
            "command": "vnncomp-late-submit task",
            "id": id,
            "status": status,
            "submission_url": format!("{}/toolkit/submission/{id}", client.base),
        }))?;
    } else {
        let done = status.get("done").and_then(Value::as_bool);
        println!("Submission task {id}");
        println!(
            "  done:   {}",
            done.map_or_else(|| "?".to_string(), |flag| flag.to_string())
        );
        if let Some(output) = status.get("output").and_then(Value::as_str) {
            println!("  output: {output}");
        }
        println!("  web:    {}/toolkit/submission/{id}", client.base);
    }
    Ok(())
}

fn emit_json(value: &Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

/// Parse a response body as JSON, falling back to the trimmed raw string.
fn body_value(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|_| Value::String(body.trim().to_string()))
}

// ---------------------------------------------------------------------------
// HTTP client (curl + cookie jar)
// ---------------------------------------------------------------------------

struct HttpResponse {
    status: u16,
    body: String,
}

impl HttpResponse {
    fn json(&self) -> Option<Value> {
        serde_json::from_str(&self.body).ok()
    }

    const fn ok(&self) -> bool {
        self.status >= 200 && self.status < 300
    }

    const fn is_redirect(&self) -> bool {
        self.status >= 300 && self.status < 400
    }
}

struct PlatformClient {
    base: String,
    host: String,
    jar: PathBuf,
}

impl PlatformClient {
    fn new(opts: &PlatformOpts) -> Result<Self> {
        let jar = match &opts.cookie_jar {
            Some(path) => path.clone(),
            None => state_dir()?.join("vnncomp2026.cookies"),
        };
        if let Some(parent) = jar.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let base = opts.platform_url.trim_end_matches('/').to_string();
        let host = url_host(&base)
            .ok_or_else(|| anyhow!("cannot extract a host from platform url '{base}'"))?;
        Ok(Self { base, host, jar })
    }

    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<(&Value, &str)>,
    ) -> Result<HttpResponse> {
        let body_file = tempfile::NamedTempFile::new()?;

        let mut cmd = Command::new("curl");
        cmd.arg("-sS")
            .arg("--max-time")
            .arg("60")
            .arg("-c")
            .arg(&self.jar)
            .arg("-b")
            .arg(&self.jar)
            .arg("-o")
            .arg(body_file.path())
            .arg("-w")
            .arg("%{http_code}")
            .arg("-H")
            .arg("Accept: application/json");

        // Follow GET redirects (the legacy host 307s to the live platform).
        // POSTs stay pinned to the configured host so the CSRF header and
        // payload are never re-sent to an unexpected redirect target.
        let payload_file = if let Some((payload, csrf_token)) = body {
            let mut file = tempfile::NamedTempFile::new()?;
            file.write_all(payload.to_string().as_bytes())?;
            file.flush()?;
            cmd.arg("-X")
                .arg(method)
                .arg("-H")
                .arg("Content-Type: application/json")
                .arg("-H")
                .arg(format!("X-CSRFToken: {csrf_token}"))
                .arg("-H")
                .arg(format!("Referer: {}/", self.base))
                .arg("-H")
                .arg(format!("Origin: {}", self.base))
                .arg("--data-binary")
                .arg(format!("@{}", file.path().display()));
            Some(file)
        } else {
            cmd.arg("-L").arg("--max-redirs").arg("5");
            None
        };

        cmd.arg(format!("{}{path}", self.base));

        let output = cmd
            .output()
            .context("running curl (is curl installed and on PATH?)")?;
        drop(payload_file);
        if !output.status.success() {
            bail!(
                "curl {method} {path} failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        restrict_permissions(&self.jar);

        let status: u16 = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .unwrap_or(0);
        let body_text = fs::read_to_string(body_file.path()).unwrap_or_default();
        Ok(HttpResponse {
            status,
            body: body_text,
        })
    }

    fn get(&self, path: &str) -> Result<HttpResponse> {
        self.request("GET", path, None)
    }

    /// POST JSON with the CSRF token (fetching one first if needed).
    fn post(&self, path: &str, payload: &Value) -> Result<HttpResponse> {
        let token = self.ensure_csrf()?;
        self.request("POST", path, Some((payload, &token)))
    }

    /// Read one cookie value scoped to this platform's host from the jar.
    fn cookie(&self, name: &str) -> Option<String> {
        let text = fs::read_to_string(&self.jar).ok()?;
        parse_cookie_jar(&text, name, &self.host)
    }

    /// Make sure the jar holds a csrftoken (Django sets it on GET).
    fn ensure_csrf(&self) -> Result<String> {
        if let Some(token) = self.cookie("csrftoken") {
            return Ok(token);
        }
        let _ = self.get("/")?;
        if let Some(token) = self.cookie("csrftoken") {
            return Ok(token);
        }
        let _ = self.get("/api/user/")?;
        self.cookie("csrftoken")
            .ok_or_else(|| anyhow!("platform did not set a csrftoken cookie; cannot POST safely"))
    }

    /// Current session user, if authenticated. Errors on transport failure.
    fn whoami(&self) -> Result<Option<Value>> {
        let resp = self.get("/api/user/")?;
        if resp.ok() {
            Ok(resp.json())
        } else {
            Ok(None)
        }
    }

    /// Best-effort session user: no network call without a session cookie,
    /// and transport failures degrade to `None` (offline-friendly).
    fn session_user_if_any(&self) -> Option<Value> {
        self.cookie("sessionid")?;
        self.whoami().ok().flatten()
    }

    /// Submission gates + option lists (requires an authenticated session).
    fn form_data(&self) -> Result<Option<Value>> {
        let resp = self.get("/api/toolkit/form-data/")?;
        if resp.ok() {
            Ok(resp.json())
        } else {
            Ok(None)
        }
    }
}

fn url_host(base: &str) -> Option<String> {
    let after_scheme = base.split_once("://").map_or(base, |(_, rest)| rest);
    let host_port = after_scheme.split(['/', '?']).next()?;
    let host = host_port.split(':').next()?;
    if host.is_empty() {
        None
    } else {
        Some(host.to_lowercase())
    }
}

/// Netscape-format jar lookup, honoring the domain column so tokens from a
/// different `--platform-url` never leak into this host's requests.
fn parse_cookie_jar(text: &str, name: &str, host: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.strip_prefix("#HttpOnly_").unwrap_or(line);
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if let [domain, .., cookie_name, cookie_value] = fields.as_slice() {
            let domain = domain.trim_start_matches('.').to_lowercase();
            let domain_matches = host == domain || host.ends_with(&format!(".{domain}"));
            if domain_matches && *cookie_name == name {
                return Some((*cookie_value).to_string());
            }
        }
    }
    None
}

fn restrict_permissions(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = fs::metadata(path) {
            let mut perms = metadata.permissions();
            perms.set_mode(0o600);
            let _ = fs::set_permissions(path, perms);
        }
    }
}

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

fn state_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("NY_STATE_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let home = std::env::var_os("HOME").ok_or_else(|| anyhow!("HOME is not set"))?;
    Ok(PathBuf::from(home).join(".ny"))
}

fn credentials_path() -> Result<PathBuf> {
    Ok(state_dir()?.join("vnncomp2026.credentials"))
}

fn load_credentials() -> Result<Option<(String, String)>> {
    let path = credentials_path()?;
    if !path.is_file() {
        return Ok(None);
    }
    let value: Value = serde_json::from_str(&fs::read_to_string(&path)?)
        .with_context(|| format!("parsing {}", path.display()))?;
    let email = value.get("email").and_then(Value::as_str);
    let password = value.get("password").and_then(Value::as_str);
    match (email, password) {
        (Some(email), Some(password)) => Ok(Some((email.to_string(), password.to_string()))),
        _ => Ok(None),
    }
}

fn store_credentials(email: &str, password: &str) -> Result<PathBuf> {
    let path = credentials_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &path,
        serde_json::to_string_pretty(&json!({"email": email, "password": password}))?,
    )?;
    restrict_permissions(&path);
    Ok(path)
}

/// Password resolution order: env var, then credentials file (matching email).
fn resolve_password(email: &str) -> Result<Option<String>> {
    if let Ok(password) = std::env::var(PASSWORD_ENV) {
        if !password.is_empty() {
            return Ok(Some(password));
        }
    }
    if let Some((stored_email, password)) = load_credentials()? {
        if stored_email == email {
            return Ok(Some(password));
        }
    }
    Ok(None)
}

/// 48 hex chars from /dev/urandom. Deliberately not the `rand` crate: this is
/// the only randomness ny-cli needs and the harness targets are unix-only.
fn generate_password() -> Result<String> {
    let mut file = fs::File::open("/dev/urandom").context("opening /dev/urandom")?;
    let mut buf = [0_u8; 24];
    file.read_exact(&mut buf)?;
    Ok(buf.iter().fold(String::new(), |mut acc, byte| {
        use std::fmt::Write as _;
        let _ = write!(acc, "{byte:02x}");
        acc
    }))
}

// ---------------------------------------------------------------------------
// Git context (anchored to the ny repo root, not the process cwd)
// ---------------------------------------------------------------------------

fn repo_root() -> Result<PathBuf> {
    super::vnncomp_submit::find_repo_root(&std::env::current_dir()?)
}

fn git_stdout(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn default_repository(root: &Path) -> Option<String> {
    git_stdout(root, &["config", "--get", "remote.origin.url"])
}

fn default_commit(root: &Path) -> Option<String> {
    git_stdout(root, &["rev-parse", "HEAD"])
}

/// `Some(false)` when git succeeds with empty output — the commit exists but
/// is on no remote branch. (`git_stdout` can't express that: it folds empty
/// output into `None`.)
fn commit_on_remote(root: &Path, commit: &str) -> Option<bool> {
    let output = Command::new("git")
        .current_dir(root)
        .args(["branch", "-r", "--contains", commit])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(!String::from_utf8_lossy(&output.stdout).trim().is_empty())
}

// ---------------------------------------------------------------------------
// Benchmarks / tracks
// ---------------------------------------------------------------------------

/// Benchmark ids for a track selection, in platform order:
/// 'test' first (unless skipped), then Regular, then Extended, then extras.
fn track_benchmarks(tracks: TrackSelection, extra: &[String], skip_test: bool) -> Vec<String> {
    let mut ids: Vec<String> = Vec::new();
    if !skip_test {
        ids.push("test".to_string());
    }
    if matches!(tracks, TrackSelection::All | TrackSelection::Regular) {
        ids.extend(REGULAR_TRACK_2026.iter().map(ToString::to_string));
    }
    if matches!(tracks, TrackSelection::All | TrackSelection::Extended) {
        ids.extend(EXTENDED_TRACK_2026.iter().map(ToString::to_string));
    }
    for id in extra {
        if !ids.iter().any(|existing| existing == id) {
            ids.push(id.clone());
        }
    }
    ids
}

const fn track_label(tracks: TrackSelection) -> &'static str {
    match tracks {
        TrackSelection::All => "Regular + Extended (all tracks)",
        TrackSelection::Regular => "Regular track",
        TrackSelection::Extended => "Extended track",
    }
}

/// Short form for one-line contexts (email subject).
const fn track_short(tracks: TrackSelection) -> &'static str {
    match tracks {
        TrackSelection::All => "all tracks",
        TrackSelection::Regular => "Regular track",
        TrackSelection::Extended => "Extended track",
    }
}

// ---------------------------------------------------------------------------
// form-data option lists
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ResolvedChoice {
    value: Value,
    label: String,
}

/// Normalize a form-data option list. The platform emits `{value, label}`
/// objects; bare strings are accepted for forward compatibility. Anything
/// else is dropped so an unexpected shape fails loudly downstream.
fn option_choices(options: Option<&Value>) -> Vec<ResolvedChoice> {
    options
        .and_then(Value::as_array)
        .map(|list| {
            list.iter()
                .filter_map(|entry| match entry {
                    Value::String(text) => Some(ResolvedChoice {
                        value: entry.clone(),
                        label: text.clone(),
                    }),
                    Value::Object(map) => Some(ResolvedChoice {
                        value: map.get("value")?.clone(),
                        label: map.get("label").and_then(Value::as_str)?.to_string(),
                    }),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// First choice whose label contains a hint; hints are in priority order.
fn find_choice(choices: &[ResolvedChoice], hints: &[&str]) -> Option<ResolvedChoice> {
    for hint in hints {
        let lowered_hint = hint.to_lowercase();
        if let Some(choice) = choices
            .iter()
            .find(|choice| choice.label.to_lowercase().contains(&lowered_hint))
        {
            return Some(choice.clone());
        }
    }
    None
}

fn labels(choices: &[ResolvedChoice]) -> String {
    choices
        .iter()
        .map(|choice| choice.label.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// status
// ---------------------------------------------------------------------------

fn gate_flag(form: Option<&Value>, key: &str) -> Option<bool> {
    form?.get(key)?.as_bool()
}

fn handle_status(platform: &PlatformOpts) -> Result<()> {
    let client = PlatformClient::new(platform)?;
    // One probe answers both questions: transport failure => unreachable,
    // any HTTP response => reachable (2xx JSON => authenticated).
    let (reachable, user, transport_error) = match client.whoami() {
        Ok(user) => (true, user, None),
        Err(err) => (false, None, Some(err.to_string())),
    };
    let form = if user.is_some() {
        client.form_data()?
    } else {
        None
    };

    let can_submit = gate_flag(form.as_ref(), "can_submit");
    let scheduler_enabled = gate_flag(form.as_ref(), "scheduler_enabled");
    let credentials_email = load_credentials()?.map(|(email, _)| email);

    if platform.json {
        return emit_json(&json!({
            "command": "vnncomp-late-submit status",
            "platform_url": client.base,
            "reachable": reachable,
            "transport_error": transport_error,
            "authenticated": user.is_some(),
            "user": user,
            "stored_credentials_email": credentials_email,
            "can_submit": can_submit,
            "scheduler_enabled": scheduler_enabled,
            "form_data": form,
            "timeline_2026": TIMELINE_2026
                .iter()
                .map(|(date, event)| json!({"date": date, "event": event}))
                .collect::<Vec<_>>(),
        }));
    }

    println!("VNN-COMP 2026 evaluation platform status");
    println!("  url:            {}", client.base);
    match transport_error {
        None => println!("  reachable:      yes"),
        Some(err) => println!("  reachable:      NO ({err})"),
    }
    println!(
        "  authenticated:  {}",
        user.as_ref().map_or_else(
            || "no (login or signup first)".to_string(),
            |value| format!("yes ({})", user_display(value)),
        )
    );
    if let Some(email) = credentials_email {
        println!("  stored account: {email}");
    }
    println!(
        "  can_submit:     {}",
        describe_flag(can_submit, user.is_some())
    );
    println!(
        "  scheduler:      {}",
        describe_flag(scheduler_enabled, user.is_some())
    );
    println!();
    println!("2026 timeline (vnncomp2026 issues #9/#12/#13):");
    for (date, event) in TIMELINE_2026 {
        println!("  {date}  {event}");
    }
    println!();
    println!(
        "Late submissions after 2026-06-30 need evaluation-chair approval: {}",
        chair_list()
    );
    Ok(())
}

fn user_display(user: &Value) -> String {
    ["email", "username", "name"]
        .iter()
        .find_map(|key| user.get(*key).and_then(Value::as_str))
        .map_or_else(|| user.to_string(), ToString::to_string)
}

fn describe_flag(flag: Option<bool>, authenticated: bool) -> String {
    match flag {
        Some(true) => "true".to_string(),
        Some(false) => "false (window closed server-side)".to_string(),
        None if authenticated => "unknown (not reported by form-data)".to_string(),
        None => "unknown (login required to read form-data)".to_string(),
    }
}

fn chair_list() -> String {
    EVALUATION_CHAIRS
        .iter()
        .map(|(name, email)| format!("{name} <{email}>"))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// signup / login
// ---------------------------------------------------------------------------

fn handle_signup(name: &str, email: &str, platform: &PlatformOpts) -> Result<()> {
    let client = PlatformClient::new(platform)?;

    // The credentials file is single-slot; never clobber another account's
    // (possibly auto-generated and otherwise unrecoverable) password.
    if let Some((stored_email, _)) = load_credentials()? {
        if stored_email != email {
            bail!(
                "{} already holds credentials for {stored_email}; refusing to \
                 overwrite them for {email}. Move the file aside first, or log \
                 in to the old account with `ny vnncomp-late-submit login`.",
                credentials_path()?.display()
            );
        }
    }

    let password = match resolve_password(email)? {
        Some(existing) => existing,
        None => generate_password()?,
    };
    // Persist before the POST so a created account's password is never lost.
    let credentials_file = store_credentials(email, &password)?;

    let resp = client.post(
        "/api/signup/",
        &json!({
            "name": name,
            "email": email,
            "password": password,
            "confirm": password,
        }),
    )?;

    let created = resp.ok();
    if platform.json {
        emit_json(&json!({
            "command": "vnncomp-late-submit signup",
            "email": email,
            "created": created,
            "http_status": resp.status,
            "response": body_value(&resp.body),
            "credentials_file": credentials_file,
            "next_step": "Ask the organizers to activate the account (vnncomp2026 issue #9 precedent).",
        }))?;
    } else if created {
        println!("Account created for {email}.");
        println!("  credentials saved: {}", credentials_file.display());
        println!(
            "  NOTE: organizers must activate the account before submission ({}).",
            chair_list()
        );
    } else {
        println!("Signup returned HTTP {}: {}", resp.status, resp.body.trim());
        println!("  credentials kept at: {}", credentials_file.display());
    }
    if !created {
        bail!("signup was not accepted (HTTP {})", resp.status);
    }
    Ok(())
}

fn resolve_login_email(explicit: Option<&str>) -> Result<String> {
    if let Some(email) = explicit {
        return Ok(email.to_string());
    }
    if let Some((email, _)) = load_credentials()? {
        return Ok(email);
    }
    bail!("no login email: pass --email or create an account with `ny vnncomp-late-submit signup`")
}

fn login(client: &PlatformClient, email: &str) -> Result<Value> {
    let Some(password) = resolve_password(email)? else {
        bail!("no password for {email}: set {PASSWORD_ENV} or store credentials via `signup`")
    };
    let resp = client.post(
        "/api/login/",
        &json!({"username": email, "password": password}),
    )?;
    if !resp.ok() {
        bail!(
            "login failed for {email} (HTTP {}): {}",
            resp.status,
            resp.body.trim()
        );
    }
    Ok(resp.json().unwrap_or(Value::Null))
}

fn handle_login(email: Option<&str>, platform: &PlatformOpts) -> Result<()> {
    let client = PlatformClient::new(platform)?;
    let email = resolve_login_email(email)?;
    let user = login(&client, &email)?;
    if platform.json {
        emit_json(&json!({
            "command": "vnncomp-late-submit login",
            "email": email,
            "user": user,
            "cookie_jar": client.jar,
        }))?;
    } else {
        println!(
            "Logged in as {email} (session stored in {}).",
            client.jar.display()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// submit
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct SubmissionPlan {
    repository: String,
    commit: String,
    commit_on_origin: Option<bool>,
    benchmarks: Vec<String>,
    instance: Option<ResolvedChoice>,
    ami: Option<ResolvedChoice>,
    name: String,
    vnnlib_version: String,
    mode: RunMode,
}

impl SubmissionPlan {
    /// The `POST /api/toolkit/submit/` body (exact SPA key parity). `strict`
    /// refuses to build without a form-data-resolved instance type; the
    /// lenient form substitutes placeholders for dry-run display.
    fn payload(&self, strict: bool) -> Result<Value> {
        let aws_instance_type = match (&self.instance, strict) {
            (Some(choice), _) => choice.value.clone(),
            (None, true) => bail!(
                "instance type could not be resolved from the platform form-data; \
                 refusing to POST a guessed value"
            ),
            (None, false) => Value::String("<resolved from form-data at submit time>".to_string()),
        };
        Ok(json!({
            "aws_instance_type": aws_instance_type,
            "name": self.name,
            "ami": self.ami.as_ref().map(|choice| choice.value.clone()),
            "repository": self.repository,
            "hash": self.commit,
            "scripts_dir": ".",
            "manual_installation_step": false,
            "run_installation_script_as_root": false,
            "run_post_installation_script_as_root": false,
            "run_toolkit_as_root": false,
            "vnnlib_version": self.vnnlib_version,
            "benchmarks": self.benchmarks,
            "run_networks": self.mode.as_str(),
            "use_own_eni": false,
        }))
    }
}

fn build_submission_plan(args: &SubmitArgs, form: Option<&Value>) -> Result<SubmissionPlan> {
    let root = repo_root()?;
    let repository = args
        .repository
        .clone()
        .or_else(|| default_repository(&root))
        .ok_or_else(|| anyhow!("no --repository given and no git origin remote found"))?;
    let commit = args
        .commit
        .clone()
        .or_else(|| default_commit(&root))
        .ok_or_else(|| anyhow!("no --commit given and git rev-parse HEAD failed"))?;
    let commit_on_origin = commit_on_remote(&root, &commit);

    let benchmarks = track_benchmarks(args.tracks, &args.benchmarks, args.skip_test);

    let instance_choices = option_choices(form.and_then(|value| value.get("instance_types")));
    let instance = find_choice(&instance_choices, args.instance_type.label_hints());
    if instance.is_none() && !instance_choices.is_empty() {
        bail!(
            "no instance-type option matches {:?}; platform offers: {}",
            args.instance_type.label_hints(),
            labels(&instance_choices)
        );
    }

    let ami_choices = option_choices(form.and_then(|value| value.get("ami_options")));
    let ami = match args.ami.as_deref() {
        Some(hint) => {
            let found = find_choice(&ami_choices, &[hint]);
            if found.is_none() && !ami_choices.is_empty() {
                bail!(
                    "no AMI option matches '{hint}'; platform offers: {}",
                    labels(&ami_choices)
                );
            }
            found
        }
        None => {
            // "ubuntu server" first: plain server images also carry
            // "(Ubuntu 24.04)" inside Deep Learning AMI labels.
            let found = find_choice(
                &ami_choices,
                &["ubuntu server 24.04", "ubuntu 24.04", "24.04", "ubuntu"],
            );
            if found.is_none() && !ami_choices.is_empty() {
                let fallback = ami_choices.first().cloned();
                if let Some(choice) = &fallback {
                    eprintln!(
                        "warning: no Ubuntu AMI option found; falling back to '{}'",
                        choice.label
                    );
                }
                fallback
            } else {
                found
            }
        }
    };

    Ok(SubmissionPlan {
        repository,
        commit,
        commit_on_origin,
        benchmarks,
        instance,
        ami,
        name: args.name.clone(),
        vnnlib_version: args.vnnlib_version.clone(),
        mode: args.mode,
    })
}

struct SubmitOutcome {
    can_submit: Option<bool>,
    scheduler_enabled: Option<bool>,
    authenticated: bool,
    window_closed: bool,
    attempted: bool,
    response: Option<HttpResponse>,
    submission_url: Option<String>,
}

fn handle_submit(args: &SubmitArgs) -> Result<()> {
    if args.mode == RunMode::All && !args.yes {
        bail!(
            "--mode all runs the full evaluation on organizer-funded AWS time \
             (the chairs reported ~$6.5k total costs and no re-runs after the \
             June 30 window; vnncomp2026 issues #9/#12). Re-run with --yes if \
             the chairs approved a full run, or use the default --mode random \
             smoke evaluation."
        );
    }

    let client = PlatformClient::new(&args.platform)?;

    // Dry runs work offline: probe the session only best-effort. Real
    // submissions authenticate for real.
    let user = if args.dry_run {
        client.session_user_if_any()
    } else {
        match client.whoami()? {
            Some(user) => Some(user),
            None => {
                let email = resolve_login_email(args.email.as_deref())?;
                Some(login(&client, &email)?)
            }
        }
    };

    let form = if user.is_some() {
        if args.dry_run {
            client.form_data().ok().flatten()
        } else {
            client.form_data()?
        }
    } else {
        None
    };
    let can_submit = gate_flag(form.as_ref(), "can_submit");
    let scheduler_enabled = gate_flag(form.as_ref(), "scheduler_enabled");
    let plan = build_submission_plan(args, form.as_ref())?;

    if plan.commit_on_origin == Some(false) {
        eprintln!(
            "warning: commit {} is not on any remote branch; the platform clones \
             from the remote and will not see it",
            plan.commit
        );
    }

    let window_closed = can_submit == Some(false) || scheduler_enabled == Some(false);
    let attempted = !args.dry_run && (!window_closed || args.force);
    let mut outcome = SubmitOutcome {
        can_submit,
        scheduler_enabled,
        authenticated: user.is_some(),
        window_closed,
        attempted,
        response: None,
        submission_url: None,
    };

    if attempted {
        let resp = client.post("/api/toolkit/submit/", &plan.payload(true)?)?;
        // redirect_to is a bare task id — a number in practice, but tolerate
        // a string in case the API ever quotes it.
        if let Some(redirect) = resp
            .json()
            .as_ref()
            .and_then(|value| value.get("redirect_to"))
            .map(|value| match value {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            })
        {
            outcome.submission_url = Some(format!("{}/toolkit/submission/{redirect}", client.base));
        }
        outcome.response = Some(resp);
    }

    let display_payload = plan.payload(false)?;
    if args.platform.json {
        emit_json(&json!({
            "command": "vnncomp-late-submit submit",
            "dry_run": args.dry_run,
            "tracks": track_label(args.tracks),
            "benchmark_count": plan.benchmarks.len(),
            "payload": display_payload,
            "instance_type_label": plan.instance.as_ref().map(|choice| choice.label.clone()),
            "ami_label": plan.ami.as_ref().map(|choice| choice.label.clone()),
            "can_submit": outcome.can_submit,
            "scheduler_enabled": outcome.scheduler_enabled,
            "window_closed": outcome.window_closed,
            "attempted": outcome.attempted,
            "http_status": outcome.response.as_ref().map(|resp| resp.status),
            "response": outcome.response.as_ref().map(|resp| body_value(&resp.body)),
            "submission_url": outcome.submission_url,
        }))?;
    } else {
        print_submit_report(args, &plan, &outcome, &display_payload);
    }

    match &outcome.response {
        Some(resp) if !resp.ok() => {
            if resp.is_redirect() {
                bail!(
                    "submission POST redirected (HTTP {}); the platform likely \
                     moved — pass --platform-url with the new host",
                    resp.status
                );
            }
            bail!("submission POST rejected with HTTP {}", resp.status)
        }
        None if !args.dry_run && !attempted => {
            bail!(
                "submission window is closed server-side; not POSTing without --force. \
                 Draft the chair request with `ny vnncomp-late-submit request-email`."
            )
        }
        _ => Ok(()),
    }
}

fn print_submit_report(
    args: &SubmitArgs,
    plan: &SubmissionPlan,
    outcome: &SubmitOutcome,
    display_payload: &Value,
) {
    println!("VNN-COMP 2026 late submission — NY");
    println!("  tracks:        {}", track_label(args.tracks));
    println!(
        "  benchmarks:    {} ({})",
        plan.benchmarks.len(),
        plan.benchmarks.join(", ")
    );
    println!(
        "  instance type: {} -> {}",
        args.instance_type.describe(),
        plan.instance
            .as_ref()
            .map_or("(unresolved: needs form-data)", |choice| &choice.label)
    );
    println!(
        "  ami:           {}",
        plan.ami
            .as_ref()
            .map_or("(unresolved: needs form-data)", |choice| &choice.label)
    );
    println!("  repository:    {}", plan.repository);
    println!(
        "  commit:        {}{}",
        plan.commit,
        match plan.commit_on_origin {
            Some(true) => " (on origin)",
            Some(false) => " (NOT on origin!)",
            None => "",
        }
    );
    println!("  mode:          {}", plan.mode.as_str());
    println!("  vnnlib:        {}", plan.vnnlib_version);
    println!(
        "  gates:         can_submit={} scheduler_enabled={}",
        describe_flag(outcome.can_submit, outcome.authenticated),
        describe_flag(outcome.scheduler_enabled, outcome.authenticated)
    );
    if args.dry_run {
        println!("  dry-run:       payload built, nothing POSTed");
        if let Ok(pretty) = serde_json::to_string_pretty(display_payload) {
            println!("{pretty}");
        }
    } else if outcome.attempted {
        println!(
            "  POST:          HTTP {}",
            outcome
                .response
                .as_ref()
                .map_or_else(|| "?".to_string(), |resp| resp.status.to_string())
        );
        if let Some(url) = &outcome.submission_url {
            println!("  submission:    {url}");
        } else if let Some(resp) = &outcome.response {
            let trimmed = resp.body.trim();
            if !trimmed.is_empty() {
                println!("  response:      {trimmed}");
            }
        }
    } else {
        println!("  POST:          skipped (window closed; use --force to attempt anyway)");
        println!(
            "  next:          `ny vnncomp-late-submit request-email` and contact {}",
            chair_list()
        );
    }
}

// ---------------------------------------------------------------------------
// request-email
// ---------------------------------------------------------------------------

fn handle_request_email(args: &RequestEmailArgs) -> Result<()> {
    let root = repo_root()?;
    let from_name = args
        .from_name
        .clone()
        .or_else(|| git_stdout(&root, &["config", "--get", "user.name"]))
        .unwrap_or_else(|| "NY team".to_string());
    let from_email = args
        .from_email
        .clone()
        .or_else(|| git_stdout(&root, &["config", "--get", "user.email"]))
        .unwrap_or_else(|| "unknown@example.invalid".to_string());
    let repository = args
        .repository
        .clone()
        .or_else(|| default_repository(&root))
        .unwrap_or_else(|| "https://github.com/alabsystems/ny".to_string());
    let commit = args
        .commit
        .clone()
        .or_else(|| default_commit(&root))
        .unwrap_or_else(|| "HEAD".to_string());
    let account_email = match &args.account_email {
        Some(email) => Some(email.clone()),
        None => load_credentials()?.map(|(email, _)| email),
    };

    let eml = render_request_email(
        &from_name,
        &from_email,
        &repository,
        &commit,
        args.tracks,
        args.instance_type,
        account_email.as_deref(),
    );

    // Match vnncomp-submit's convention: relative outputs land in the repo.
    let output = if args.output.is_absolute() {
        args.output.clone()
    } else {
        root.join(&args.output)
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, &eml)?;

    if args.json {
        emit_json(&json!({
            "command": "vnncomp-late-submit request-email",
            "output": output,
            "to": EVALUATION_CHAIRS
                .iter()
                .map(|(_, email)| *email)
                .collect::<Vec<_>>(),
            "from": format!("{from_name} <{from_email}>"),
        }))?;
    } else {
        println!("{eml}");
        println!("---");
        println!(
            "Draft written to {}. Review and send it from your mail client.",
            output.display()
        );
    }
    Ok(())
}

fn render_request_email(
    from_name: &str,
    from_email: &str,
    repository: &str,
    commit: &str,
    tracks: TrackSelection,
    instance_type: InstanceType,
    account_email: Option<&str>,
) -> String {
    let benchmarks = track_benchmarks(tracks, &[], true);
    let account_line = account_email.map_or_else(
        || "- Platform account: (to be created via the signup form)".to_string(),
        |email| format!("- Platform account: {email} (self-registered, pending activation)"),
    );

    format!(
        "From: {from_name} <{from_email}>\n\
         To: {to_header}\n\
         Subject: VNN-COMP 2026: late tool-submission request - NY ({subject_tracks})\n\
         \n\
         Dear VNN-COMP 2026 evaluation chairs,\n\
         \n\
         I am writing to ask whether a late tool submission can still be accepted\n\
         for VNN-COMP 2026, in whatever form is least disruptive to you. I fully\n\
         understand that the tool-submission window closed on June 30 (AoE), that\n\
         the final evaluation was completed on July 10, and that results freeze\n\
         around July 20 for FLoC. An out-of-competition (hors concours) listing,\n\
         an appendix mention in the arXiv report, or exclusion from the official\n\
         ranking are all perfectly fine outcomes for us.\n\
         \n\
         The tool is NY, a Rust neural-network verifier implementing the standard\n\
         v1 script contract. Everything is prepared so that an evaluation requires\n\
         no manual effort on your side:\n\
         \n\
         {account_line}\n\
         - Repository: {repository}\n\
         - Commit: {commit}\n\
         - Scripts: install_tool.sh / prepare_instance.sh / run_instance.sh in the\n\
           repository root (scripts_dir = \".\")\n\
         - Install: single offline cargo build; no external solver licenses\n\
           (no Gurobi/MATLAB), no manual installation steps\n\
         - AWS platform: {instance}\n\
         - AMI: any Ubuntu 24.04 image\n\
         - Preferred VNN-LIB version: 1.0 (the 2.0-only benchmarks, including the\n\
           relational isomorphic/monotonic ACAS Xu pairs, are supported via the\n\
           automatic 2.0 fallback)\n\
         - Benchmarks ({track_label}, {count} total):\n\
           {benchmark_list}\n\
         \n\
         We are aware evaluation costs are a real burden this year and we are happy\n\
         to reimburse the AWS costs of running NY, and/or to start with the 'random'\n\
         smoke mode (10 instances per benchmark) so you can gauge the tool cheaply\n\
         before deciding anything further.\n\
         \n\
         If a late run is not feasible at this point, we completely understand -\n\
         in that case we would appreciate any guidance on participating in the\n\
         report or in VNN-COMP 2027.\n\
         \n\
         Thank you for the enormous work you put into the competition,\n\
         \n\
         {from_name}\n",
        to_header = chair_list(),
        subject_tracks = track_short(tracks),
        instance = instance_type.describe(),
        track_label = track_label(tracks),
        count = benchmarks.len(),
        benchmark_list = benchmarks.join(", "),
    )
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_lists_are_disjoint() {
        for id in EXTENDED_TRACK_2026 {
            assert!(!REGULAR_TRACK_2026.contains(&id));
        }
    }

    #[test]
    fn all_tracks_selects_thirty_benchmarks_plus_test() {
        let ids = track_benchmarks(TrackSelection::All, &[], false);
        assert_eq!(ids.len(), 31);
        assert_eq!(ids.first().map(String::as_str), Some("test"));
        let deduped: std::collections::BTreeSet<_> = ids.iter().collect();
        assert_eq!(deduped.len(), ids.len());
    }

    #[test]
    fn extra_benchmarks_are_appended_once() {
        let extra = vec!["test".to_string(), "custom_bench".to_string()];
        let ids = track_benchmarks(TrackSelection::Extended, &extra, false);
        assert_eq!(ids.iter().filter(|id| id.as_str() == "test").count(), 1);
        assert_eq!(ids.last().map(String::as_str), Some("custom_bench"));
    }

    #[test]
    fn find_choice_prefers_hint_priority_over_option_order() {
        let choices = option_choices(Some(&json!([
            {"value": "ami-1", "label": "Ubuntu 22.04"},
            {"value": "ami-2", "label": "Ubuntu 24.04"},
        ])));
        // "ubuntu" alone matches ami-1 first, but the higher-priority
        // "ubuntu 24.04" hint must win.
        let choice = find_choice(&choices, &["ubuntu 24.04", "24.04", "ubuntu"]).expect("match");
        assert_eq!(choice.value, json!("ami-2"));

        // No hint match: find_choice does NOT silently fall back.
        assert!(find_choice(&choices, &["debian"]).is_none());
    }

    #[test]
    fn option_choices_accepts_objects_and_strings_only() {
        let mixed = json!([
            {"value": 1, "label": "CPU - m5.16xlarge"},
            "bare-string",
            {"unexpected": "shape"},
            42,
        ]);
        let choices = option_choices(Some(&mixed));
        assert_eq!(choices.len(), 2);
        assert_eq!(choices[0].value, json!(1));
        assert_eq!(choices[1].label, "bare-string");
    }

    #[test]
    fn cookie_jar_parsing_is_domain_scoped() {
        let jar = "# Netscape HTTP Cookie File\n\
                   vnn.repeatability.cps.cit.tum.de\tFALSE\t/\tTRUE\t0\tcsrftoken\tabc123\n\
                   #HttpOnly_vnn.repeatability.cps.cit.tum.de\tFALSE\t/\tTRUE\t0\tsessionid\txyz\n\
                   localhost\tFALSE\t/\tTRUE\t0\tcsrftoken\tSTALE\n";
        let host = "vnn.repeatability.cps.cit.tum.de";
        assert_eq!(
            parse_cookie_jar(jar, "csrftoken", host).as_deref(),
            Some("abc123")
        );
        assert_eq!(
            parse_cookie_jar(jar, "sessionid", host).as_deref(),
            Some("xyz")
        );
        assert_eq!(
            parse_cookie_jar(jar, "csrftoken", "localhost").as_deref(),
            Some("STALE")
        );
        assert_eq!(parse_cookie_jar(jar, "missing", host), None);
        // Parent-domain cookies apply to subdomains.
        let parent = ".cit.tum.de\tFALSE\t/\tTRUE\t0\tshared\tval\n";
        assert_eq!(
            parse_cookie_jar(parent, "shared", host).as_deref(),
            Some("val")
        );
    }

    #[test]
    fn url_host_strips_scheme_port_and_path() {
        assert_eq!(
            url_host("https://vnn.repeatability.cps.cit.tum.de").as_deref(),
            Some("vnn.repeatability.cps.cit.tum.de")
        );
        assert_eq!(
            url_host("http://localhost:8000/api").as_deref(),
            Some("localhost")
        );
        assert_eq!(url_host("https://"), None);
    }

    fn submit_args() -> SubmitArgs {
        SubmitArgs {
            tracks: TrackSelection::All,
            benchmarks: Vec::new(),
            skip_test: false,
            mode: RunMode::Random,
            instance_type: InstanceType::Cpu,
            ami: None,
            name: "NY".to_string(),
            repository: Some("https://github.com/alabsystems/ny".to_string()),
            commit: Some("0123456789abcdef".to_string()),
            vnnlib_version: "1.0".to_string(),
            email: None,
            dry_run: true,
            force: false,
            yes: false,
            platform: PlatformOpts {
                platform_url: DEFAULT_PLATFORM_URL.to_string(),
                cookie_jar: None,
                json: false,
            },
        }
    }

    #[test]
    fn submission_payload_matches_platform_contract() {
        let form = json!({
            "instance_types": [
                {"value": 1, "label": "CPU: m5.16xlarge"},
                {"value": 2, "label": "Balanced: g5.8xlarge"},
            ],
            "ami_options": [
                {"value": "ami-1", "label": "Ubuntu 22.04"},
                {"value": "ami-2", "label": "Ubuntu 24.04"},
            ],
        });
        let plan = build_submission_plan(&submit_args(), Some(&form)).expect("plan");
        let payload = plan.payload(true).expect("strict payload");
        let object = payload.as_object().expect("object");

        // Exact key parity with the SPA submit handler (platform bundle).
        for key in [
            "aws_instance_type",
            "name",
            "ami",
            "repository",
            "hash",
            "scripts_dir",
            "manual_installation_step",
            "run_installation_script_as_root",
            "run_post_installation_script_as_root",
            "run_toolkit_as_root",
            "vnnlib_version",
            "benchmarks",
            "run_networks",
            "use_own_eni",
        ] {
            assert!(object.contains_key(key), "missing payload key {key}");
        }
        assert_eq!(object.get("aws_instance_type"), Some(&json!(1)));
        assert_eq!(object.get("ami"), Some(&json!("ami-2")));
        assert_eq!(object.get("scripts_dir"), Some(&json!(".")));
        assert_eq!(object.get("run_networks"), Some(&json!("random")));
        assert_eq!(plan.benchmarks.len(), 31);
    }

    #[test]
    fn strict_payload_refuses_unresolved_instance_type() {
        let plan = build_submission_plan(&submit_args(), None).expect("plan");
        assert!(plan.instance.is_none());
        assert!(plan.payload(true).is_err());
        let lenient = plan.payload(false).expect("lenient payload");
        assert!(lenient
            .get("aws_instance_type")
            .and_then(Value::as_str)
            .is_some_and(|text| text.contains("form-data")));
    }

    #[test]
    fn unknown_explicit_ami_hint_is_an_error() {
        let mut args = submit_args();
        args.ami = Some("no-such-ami".to_string());
        let form = json!({
            "instance_types": [{"value": 1, "label": "CPU: m5.16xlarge"}],
            "ami_options": [{"value": "ami-1", "label": "Ubuntu 24.04"}],
        });
        let err = build_submission_plan(&args, Some(&form)).expect_err("must fail");
        assert!(err.to_string().contains("no-such-ami"));
    }

    #[test]
    fn request_email_lists_all_track_benchmarks() {
        let eml = render_request_email(
            "Test Person",
            "test@example.com",
            "https://github.com/alabsystems/ny",
            "abc123",
            TrackSelection::All,
            InstanceType::Cpu,
            Some("account@example.com"),
        );
        for id in REGULAR_TRACK_2026.iter().chain(EXTENDED_TRACK_2026.iter()) {
            assert!(eml.contains(id), "email draft missing benchmark {id}");
        }
        assert!(eml.contains("tobias.ladner@tum.de"));
        assert!(eml.contains("kaulen@aim.rwth-aachen.de"));
        assert!(eml.contains("hors concours"));
        // The 'test' benchmark is an install check, not a request item.
        assert!(eml.contains("30 total"));
    }

    #[test]
    fn request_email_subject_tracks_the_selection() {
        let eml = render_request_email(
            "Test Person",
            "test@example.com",
            "https://github.com/alabsystems/ny",
            "abc123",
            TrackSelection::Regular,
            InstanceType::Cpu,
            None,
        );
        assert!(eml
            .contains("Subject: VNN-COMP 2026: late tool-submission request - NY (Regular track)"));
        assert!(!eml.contains("all tracks"));
        assert!(eml.contains("24 total"));
    }
}
