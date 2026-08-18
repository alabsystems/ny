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

use super::vnncomp_2026_tracks::{EXTENDED_TRACK_2026, REGULAR_TRACK_2026};

/// Live evaluation platform (the pre-2025 vnncomp.christopher-brix.de host
/// 307-redirects here; GETs follow redirects so either URL works).
const DEFAULT_PLATFORM_URL: &str = "https://vnn.repeatability.cps.cit.tum.de";

/// Environment variable consulted for the platform password before the
/// credentials file.
const PASSWORD_ENV: &str = "NY_VNNCOMP_PASSWORD";

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

    /// Git clone URL. Supply this together with --commit to explicitly
    /// override the clean, live-upstream-verified current branch.
    #[arg(long)]
    repository: Option<String>,

    /// Commit hash. Supply this together with --repository to explicitly
    /// override the clean, live-upstream-verified current branch.
    #[arg(long)]
    commit: Option<String>,

    /// Preferred VNN-LIB version (2.0-only benchmarks fall back automatically).
    #[arg(long, default_value = "1.0")]
    vnnlib_version: String,

    /// Login email (default: stored credentials).
    #[arg(long)]
    email: Option<String>,

    /// Print gates + payload without POSTing. For fully offline use, explicitly
    /// supply both --repository and --commit; implicit source selection queries
    /// the configured live upstream.
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

    /// Git clone URL. Supply this together with --commit to explicitly
    /// override the clean, live-upstream-verified current branch.
    #[arg(long)]
    repository: Option<String>,

    /// Commit hash quoted in the draft. Supply this together with --repository
    /// to explicitly override the clean, live-upstream-verified current branch.
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
            diagnostic_body(&resp.body)
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
            println!("  output: {}", diagnostic_body(output));
        }
        println!(
            "  web:    {}/toolkit/submission/{id}",
            redact_url(&client.base)
        );
    }
    Ok(())
}

fn emit_json(value: &Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&redact_value(value))?);
    Ok(())
}

/// Parse a response body as JSON, falling back to the trimmed raw string.
fn body_value(body: &str) -> Value {
    let parsed =
        serde_json::from_str(body).unwrap_or_else(|_| Value::String(body.trim().to_string()));
    redact_value(&parsed)
}

fn diagnostic_body(body: &str) -> String {
    serde_json::from_str::<Value>(body).map_or_else(
        |_| redact_text(body.trim()),
        |value| {
            serde_json::to_string(&redact_value(&value))
                .unwrap_or_else(|_| "<unprintable response>".to_string())
        },
    )
}

const REDACTED: &str = "[REDACTED]";

fn sensitive_key(key: &str) -> bool {
    let normalized: String = key
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();
    [
        "password",
        "passwd",
        "secret",
        "token",
        "cookie",
        "csrf",
        "authorization",
        "credential",
        "apikey",
        "accesskey",
        "signature",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

/// Remove userinfo and query/fragment values from a URL before it reaches any
/// human- or machine-readable output. The unredacted URL remains available
/// internally for the actual git/curl operation.
fn redact_url(url: &str) -> String {
    let Some(scheme_end) = url.find("://") else {
        return url.to_string();
    };
    let authority_start = scheme_end + 3;
    let authority_end = url[authority_start..]
        .find(['/', '?', '#'])
        .map_or(url.len(), |offset| authority_start + offset);
    let authority = &url[authority_start..authority_end];
    let safe_authority = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);

    let mut redacted = String::with_capacity(url.len());
    redacted.push_str(&url[..authority_start]);
    redacted.push_str(safe_authority);

    let suffix = &url[authority_end..];
    let (before_fragment, has_fragment) = suffix
        .split_once('#')
        .map_or((suffix, false), |(before, _)| (before, true));
    if let Some((path, query)) = before_fragment.split_once('?') {
        redacted.push_str(path);
        redacted.push('?');
        for (index, part) in query.split('&').enumerate() {
            if index != 0 {
                redacted.push('&');
            }
            if let Some((key, _)) = part.split_once('=') {
                redacted.push_str(key);
                redacted.push('=');
                redacted.push_str(REDACTED);
            } else if !part.is_empty() {
                redacted.push_str(REDACTED);
            }
        }
    } else {
        redacted.push_str(before_fragment);
    }
    if has_fragment {
        redacted.push('#');
        redacted.push_str(REDACTED);
    }
    redacted
}

/// Redact URL credentials even when a diagnostic embeds a URL in prose.
fn redact_text(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    while let Some(relative_scheme) = text[cursor..].find("://") {
        let scheme_mark = cursor + relative_scheme;
        let start = text[cursor..scheme_mark]
            .rfind(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '+' | '-' | '.')))
            .map_or(cursor, |index| cursor + index + 1);
        output.push_str(&text[cursor..start]);

        let end = text[scheme_mark + 3..]
            .find(|ch: char| ch.is_ascii_whitespace() || matches!(ch, '"' | '\'' | '<' | '>'))
            .map_or(text.len(), |offset| scheme_mark + 3 + offset);
        output.push_str(&redact_url(&text[start..end]));
        cursor = end;
    }
    output.push_str(&text[cursor..]);
    output
}

fn redact_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| {
                    let value = if sensitive_key(key) {
                        Value::String(REDACTED.to_string())
                    } else {
                        redact_value(value)
                    };
                    (key.clone(), value)
                })
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(redact_value).collect()),
        Value::String(text) => Value::String(redact_text(text)),
        other => other.clone(),
    }
}

fn diagnostic_path(path: &Path) -> String {
    redact_text(&path.display().to_string())
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

fn curl_command() -> Command {
    let mut command = Command::new("curl");
    // curl only honors -q/--disable when it is the first argument.
    command.arg("-q");
    command
}

fn apply_redirect_policy(command: &mut Command, allow_redirects: bool) {
    if allow_redirects {
        command.arg("-L").arg("--max-redirs").arg("5");
    }
}

impl PlatformClient {
    fn new(opts: &PlatformOpts) -> Result<Self> {
        let jar = match &opts.cookie_jar {
            Some(path) => path.clone(),
            None => state_dir()?.join("vnncomp2026.cookies"),
        };
        if let Some(parent) = jar.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", diagnostic_path(parent)))?;
        }
        let base = opts.platform_url.trim_end_matches('/').to_string();
        let host = url_host(&base).ok_or_else(|| {
            anyhow!(
                "cannot extract a host from platform url '{}'",
                redact_url(&base)
            )
        })?;
        Ok(Self { base, host, jar })
    }

    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<(&Value, &str)>,
    ) -> Result<HttpResponse> {
        let body_file = tempfile::NamedTempFile::new()?;
        let staged_jar = stage_private_file(&self.jar, "cookie jar")?;

        let mut cmd = curl_command();
        // This
        // prevents a user or system curlrc from enabling redirects, verbose
        // credential logging, proxying, or other unsafe request mutations.
        cmd.arg("-sS")
            .arg("--max-time")
            .arg("60")
            .arg("-c")
            .arg(staged_jar.path())
            .arg("-b")
            .arg(staged_jar.path())
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
            None
        };
        apply_redirect_policy(&mut cmd, payload_file.is_none());

        cmd.arg(format!("{}{path}", self.base));

        let output = cmd
            .output()
            .context("running curl (is curl installed and on PATH?)")?;
        drop(payload_file);
        if !output.status.success() {
            bail!(
                "curl {method} {path} failed: {}",
                redact_text(String::from_utf8_lossy(&output.stderr).trim())
            );
        }
        persist_private_file(staged_jar, &self.jar, "cookie jar")?;

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
    fn cookie(&self, name: &str) -> Result<Option<String>> {
        let Some(bytes) = read_private_file(&self.jar, "cookie jar")? else {
            return Ok(None);
        };
        let text = String::from_utf8(bytes).context("cookie jar is not valid UTF-8")?;
        Ok(parse_cookie_jar(&text, name, &self.host))
    }

    /// Make sure the jar holds a csrftoken (Django sets it on GET).
    fn ensure_csrf(&self) -> Result<String> {
        if let Some(token) = self.cookie("csrftoken")? {
            return Ok(token);
        }
        let _ = self.get("/")?;
        if let Some(token) = self.cookie("csrftoken")? {
            return Ok(token);
        }
        let _ = self.get("/api/user/")?;
        self.cookie("csrftoken")?
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
    fn session_user_if_any(&self) -> Result<Option<Value>> {
        if self.cookie("sessionid")?.is_none() {
            return Ok(None);
        }
        Ok(self.whoami().ok().flatten())
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
    let authority = after_scheme.split(['/', '?', '#']).next()?;
    let host_port = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    let host = if let Some(bracketed) = host_port.strip_prefix('[') {
        bracketed.split_once(']')?.0
    } else {
        host_port.split(':').next()?
    };
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

fn private_temp_file(path: &Path, label: &str) -> Result<tempfile::NamedTempFile> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("creating {}", diagnostic_path(parent)))?;
    let temporary = tempfile::Builder::new()
        .prefix(".vnncomp-private-")
        .tempfile_in(parent)
        .with_context(|| format!("creating staged {label} in {}", diagnostic_path(parent)))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // NamedTempFile is already created O_EXCL with 0600 on Unix. Set and
        // verify the mode before any sensitive bytes are written, and
        // propagate every failure instead of relying on a write-then-chmod.
        temporary
            .as_file()
            .set_permissions(fs::Permissions::from_mode(0o600))
            .with_context(|| format!("setting staged {label} permissions"))?;
    }
    Ok(temporary)
}

fn open_private_file(path: &Path, label: &str) -> Result<Option<fs::File>> {
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(rustix::fs::OFlags::NOFOLLOW.bits() as i32);
    }
    let file = match options.open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "opening {label} {} without following links",
                    diagnostic_path(path)
                )
            })
        }
    };
    let metadata = file
        .metadata()
        .with_context(|| format!("inspecting {label} {}", diagnostic_path(path)))?;
    if !metadata.is_file() {
        bail!("{label} {} is not a regular file", diagnostic_path(path));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        if metadata.permissions().mode() & 0o777 != 0o600 {
            bail!(
                "{label} {} must have mode 0600 (found {:04o})",
                diagnostic_path(path),
                metadata.permissions().mode() & 0o777
            );
        }
    }
    Ok(Some(file))
}

fn read_private_file(path: &Path, label: &str) -> Result<Option<Vec<u8>>> {
    let Some(mut file) = open_private_file(path, label)? else {
        return Ok(None);
    };
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .with_context(|| format!("reading {label} {}", diagnostic_path(path)))?;
    Ok(Some(bytes))
}

fn stage_private_file(path: &Path, label: &str) -> Result<tempfile::NamedTempFile> {
    let mut staged = private_temp_file(path, label)?;
    if let Some(bytes) = read_private_file(path, label)? {
        staged
            .write_all(&bytes)
            .with_context(|| format!("staging {label} {}", diagnostic_path(path)))?;
        staged
            .flush()
            .with_context(|| format!("flushing staged {label}"))?;
    }
    Ok(staged)
}

fn persist_private_file(staged: tempfile::NamedTempFile, path: &Path, label: &str) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = staged
            .as_file()
            .metadata()
            .with_context(|| format!("inspecting staged {label}"))?
            .permissions()
            .mode()
            & 0o777;
        if mode != 0o600 {
            bail!("staged {label} mode changed from 0600 to {mode:04o}; refusing to persist it");
        }
    }
    staged
        .as_file()
        .sync_all()
        .with_context(|| format!("syncing staged {label}"))?;
    staged
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("atomically replacing {label} {}", diagnostic_path(path)))?;
    Ok(())
}

fn atomic_write_private(path: &Path, bytes: &[u8], label: &str) -> Result<()> {
    let mut staged = private_temp_file(path, label)?;
    staged
        .write_all(bytes)
        .with_context(|| format!("writing staged {label}"))?;
    staged
        .flush()
        .with_context(|| format!("flushing staged {label}"))?;
    persist_private_file(staged, path, label)
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
    let Some(bytes) = read_private_file(&path, "credentials file")? else {
        return Ok(None);
    };
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing {}", diagnostic_path(&path)))?;
    let email = value.get("email").and_then(Value::as_str);
    let password = value.get("password").and_then(Value::as_str);
    match (email, password) {
        (Some(email), Some(password)) => Ok(Some((email.to_string(), password.to_string()))),
        _ => Ok(None),
    }
}

fn store_credentials(email: &str, password: &str) -> Result<PathBuf> {
    let path = credentials_path()?;
    let contents = serde_json::to_vec_pretty(&json!({"email": email, "password": password}))?;
    atomic_write_private(&path, &contents, "credentials file")?;
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

fn checked_git_stdout(root: &Path, args: &[&str], operation: &str) -> Result<String> {
    let output = Command::new("git")
        .current_dir(root)
        .env("GIT_TERMINAL_PROMPT", "0")
        .args(args)
        .output()
        .with_context(|| format!("running git for {operation}"))?;
    if !output.status.success() {
        let diagnostic = redact_text(String::from_utf8_lossy(&output.stderr).trim());
        if diagnostic.is_empty() {
            bail!("git failed while {operation} ({})", output.status);
        }
        bail!("git failed while {operation}: {diagnostic}");
    }
    String::from_utf8(output.stdout)
        .context("git emitted non-UTF-8 output")
        .map(|text| text.trim().to_string())
}

#[derive(Debug)]
struct SubmissionSource {
    repository: String,
    commit: String,
    /// Present only for the implicit path after querying the live remote.
    verified_remote_branch: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrebuiltState {
    /// All three tracked members are present. The installer still validates
    /// their checksum and provenance before use.
    Present,
    /// The locally verified commit exists but contains no complete triplet.
    Absent,
    /// Explicit source override: the selected remote commit was not inspected.
    Unverified,
}

fn prebuilt_state(root: &Path, source: &SubmissionSource) -> PrebuiltState {
    if source.verified_remote_branch.is_none() {
        return PrebuiltState::Unverified;
    }
    let members = [
        "dist/bin/ny-x86_64-linux.xz",
        "dist/bin/ny-x86_64-linux.xz.sha256",
        "dist/bin/ny-x86_64-linux.provenance.txt",
    ];
    let all_present = members.iter().all(|path| {
        let object = format!("{}:{path}", source.commit);
        Command::new("git")
            .current_dir(root)
            .args(["cat-file", "-e", &object])
            .output()
            .is_ok_and(|output| output.status.success())
    });
    if all_present {
        PrebuiltState::Present
    } else {
        PrebuiltState::Absent
    }
}

fn validate_explicit_source_field(value: &str, flag: &str) -> Result<()> {
    if value.is_empty() || value.chars().any(char::is_control) {
        bail!("{flag} must be non-empty and contain no control characters");
    }
    Ok(())
}

/// Resolve the source selected for a platform clone. An explicit override is
/// deliberately all-or-nothing. The implicit path is stricter: it accepts only
/// a clean attached branch whose HEAD exactly matches that branch on the live
/// configured upstream, queried with `ls-remote` rather than a stale local
/// remote-tracking ref.
fn resolve_submission_source(
    root: &Path,
    repository: Option<&str>,
    commit: Option<&str>,
) -> Result<SubmissionSource> {
    match (repository, commit) {
        (Some(repository), Some(commit)) => {
            validate_explicit_source_field(repository, "--repository")?;
            validate_explicit_source_field(commit, "--commit")?;
            return Ok(SubmissionSource {
                repository: repository.to_string(),
                commit: commit.to_string(),
                verified_remote_branch: None,
            });
        }
        (Some(_), None) | (None, Some(_)) => {
            bail!(
                "--repository and --commit are an explicit source override and \
                 must be supplied together"
            );
        }
        (None, None) => {}
    }

    let status = checked_git_stdout(
        root,
        &[
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
            "--ignore-submodules=none",
        ],
        "checking whether the implicit submission worktree is clean",
    )?;
    if !status.is_empty() {
        bail!(
            "refusing implicit repository/commit selection from a dirty worktree; \
             commit and push the intended source, or explicitly provide both \
             --repository and --commit"
        );
    }

    let branch = checked_git_stdout(
        root,
        &["symbolic-ref", "--quiet", "--short", "HEAD"],
        "resolving the current submission branch",
    )?;
    if branch.is_empty() {
        bail!("implicit submission requires an attached branch");
    }
    let remote_key = format!("branch.{branch}.remote");
    let merge_key = format!("branch.{branch}.merge");
    let remote = checked_git_stdout(
        root,
        &["config", "--get", &remote_key],
        "resolving the current branch's upstream remote",
    )?;
    if remote.is_empty() || remote == "." {
        bail!("implicit submission requires a configured non-local upstream remote");
    }
    let remote_ref = checked_git_stdout(
        root,
        &["config", "--get", &merge_key],
        "resolving the current branch's upstream branch",
    )?;
    let Some(remote_branch) = remote_ref.strip_prefix("refs/heads/") else {
        bail!("implicit submission upstream is not a branch under refs/heads/");
    };
    if remote_branch.is_empty() {
        bail!("implicit submission upstream branch is empty");
    }

    let repository = checked_git_stdout(
        root,
        &["remote", "get-url", &remote],
        "resolving the selected upstream repository",
    )?;
    if repository.is_empty() {
        bail!("selected upstream remote has no clone URL");
    }
    let head_commit_object = ["HEAD^", "{", "commit", "}"].concat();
    let commit = checked_git_stdout(
        root,
        &["rev-parse", "--verify", &head_commit_object],
        "resolving the implicit submission commit",
    )?;

    let output = Command::new("git")
        .current_dir(root)
        .env("GIT_TERMINAL_PROMPT", "0")
        .args(["ls-remote", "--exit-code", "--refs", &remote, &remote_ref])
        .output()
        .with_context(|| {
            redact_text(&format!("querying live upstream {remote}/{remote_branch}"))
        })?;
    let safe_upstream = redact_text(&format!("{remote}/{remote_branch}"));
    if !output.status.success() {
        let diagnostic = redact_text(String::from_utf8_lossy(&output.stderr).trim());
        if diagnostic.is_empty() {
            bail!(
                "could not verify live upstream {safe_upstream} ({})",
                output.status
            );
        }
        bail!("could not verify live upstream {safe_upstream}: {diagnostic}");
    }
    let live =
        String::from_utf8(output.stdout).context("git ls-remote emitted non-UTF-8 output")?;
    let mut matching = live.lines().filter_map(|line| {
        let (hash, found_ref) = line.split_once(char::is_whitespace)?;
        (found_ref.trim() == remote_ref).then_some(hash)
    });
    let live_commit = matching
        .next()
        .ok_or_else(|| anyhow!("live upstream branch {safe_upstream} was not found"))?;
    if matching.next().is_some() {
        bail!("live upstream returned duplicate records for {safe_upstream}");
    }
    if !live_commit.eq_ignore_ascii_case(&commit) {
        bail!(
            "implicit HEAD {commit} does not match live upstream \
             {safe_upstream} at {live_commit}; push the selected commit \
             or explicitly provide both --repository and --commit"
        );
    }

    // Close the local check/query race: neither the checked-out commit nor
    // tracked/untracked contents may have changed while ls-remote was running.
    let final_commit = checked_git_stdout(
        root,
        &["rev-parse", "--verify", &head_commit_object],
        "rechecking the implicit submission commit",
    )?;
    let final_status = checked_git_stdout(
        root,
        &[
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
            "--ignore-submodules=none",
        ],
        "rechecking the implicit submission worktree",
    )?;
    if final_commit != commit || !final_status.is_empty() {
        bail!(
            "repository state changed while verifying the implicit submission \
             source; retry from a stable clean worktree"
        );
    }

    Ok(SubmissionSource {
        repository,
        commit,
        verified_remote_branch: Some(format!("{remote}/{remote_branch}")),
    })
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
        .map(|choice| redact_text(&choice.label))
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
        Err(err) => (false, None, Some(redact_text(&err.to_string()))),
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
    println!("  url:            {}", redact_url(&client.base));
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
        println!("  stored account: {}", redact_text(&email));
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
        .map_or_else(|| redact_value(user).to_string(), redact_text)
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
    let safe_email = redact_text(email);

    // The credentials file is single-slot; never clobber another account's
    // (possibly auto-generated and otherwise unrecoverable) password.
    if let Some((stored_email, _)) = load_credentials()? {
        if stored_email != email {
            let safe_stored_email = redact_text(&stored_email);
            bail!(
                "{} already holds credentials for {safe_stored_email}; refusing to \
                 overwrite them for {safe_email}. Move the file aside first, or log \
                 in to the old account with `ny vnncomp-late-submit login`.",
                diagnostic_path(&credentials_path()?)
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
        println!("Account created for {safe_email}.");
        println!(
            "  credentials saved: {}",
            diagnostic_path(&credentials_file)
        );
        println!(
            "  NOTE: organizers must activate the account before submission ({}).",
            chair_list()
        );
    } else {
        println!(
            "Signup returned HTTP {}: {}",
            resp.status,
            diagnostic_body(&resp.body)
        );
        println!(
            "  credentials kept at: {}",
            diagnostic_path(&credentials_file)
        );
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
        bail!(
            "no password for {}: set {PASSWORD_ENV} or store credentials via `signup`",
            redact_text(email)
        )
    };
    let resp = client.post(
        "/api/login/",
        &json!({"username": email, "password": password}),
    )?;
    if !resp.ok() {
        bail!(
            "login failed for {} (HTTP {}): {}",
            redact_text(email),
            resp.status,
            diagnostic_body(&resp.body)
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
            "Logged in as {} (session stored in {}).",
            redact_text(&email),
            diagnostic_path(&client.jar)
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
    verified_remote_branch: Option<String>,
    benchmarks: Vec<String>,
    instance: Option<ResolvedChoice>,
    ami: Option<ResolvedChoice>,
    name: String,
    vnnlib_version: String,
    mode: RunMode,
}

impl SubmissionPlan {
    /// The `POST /api/toolkit/submit/` body (exact SPA key parity). `strict`
    /// refuses to build without form-data-resolved instance and AMI choices;
    /// the lenient form substitutes placeholders for dry-run display.
    fn payload(&self, strict: bool) -> Result<Value> {
        let aws_instance_type = match (&self.instance, strict) {
            (Some(choice), _) if !choice.value.is_null() => choice.value.clone(),
            (Some(_), true) | (None, true) => bail!(
                "instance type could not be resolved from the platform form-data; \
                 refusing to POST a null or guessed value"
            ),
            (Some(_), false) | (None, false) => {
                Value::String("<resolved from form-data at submit time>".to_string())
            }
        };
        let ami = match (&self.ami, strict) {
            (Some(choice), _) if !choice.value.is_null() => choice.value.clone(),
            (Some(_), true) | (None, true) => bail!(
                "AMI could not be resolved from the platform form-data; \
                 refusing to POST a null or guessed value"
            ),
            (Some(_), false) | (None, false) => {
                Value::String("<resolved from form-data at submit time>".to_string())
            }
        };
        Ok(json!({
            "aws_instance_type": aws_instance_type,
            "name": self.name,
            "ami": ami,
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
    let source =
        resolve_submission_source(&root, args.repository.as_deref(), args.commit.as_deref())?;

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
                let safe_hint = redact_text(hint);
                bail!(
                    "no AMI option matches '{safe_hint}'; platform offers: {}",
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
                        redact_text(&choice.label)
                    );
                }
                fallback
            } else {
                found
            }
        }
    };

    Ok(SubmissionPlan {
        repository: source.repository,
        commit: source.commit,
        verified_remote_branch: source.verified_remote_branch,
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

fn post_is_authorized(
    dry_run: bool,
    force: bool,
    can_submit: Option<bool>,
    scheduler_enabled: Option<bool>,
) -> bool {
    !dry_run && (force || (can_submit == Some(true) && scheduler_enabled != Some(false)))
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

    // Dry runs probe the session only best-effort. They are fully offline when
    // paired with an explicit repository+commit; implicit source selection
    // intentionally queries the live upstream. Real submissions authenticate.
    let user = if args.dry_run {
        client.session_user_if_any()?
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

    let window_closed = can_submit == Some(false) || scheduler_enabled == Some(false);
    // A missing/unknown can_submit field is not permission. Only an explicit
    // true authorizes a real POST, unless the caller supplied --force.
    let attempted = post_is_authorized(args.dry_run, args.force, can_submit, scheduler_enabled);
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
                "platform did not explicitly authorize submission \
                 (can_submit must be true and scheduler_enabled must not be false); \
                 not POSTing without --force. \
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
        redact_text(&plan.benchmarks.join(", "))
    );
    println!(
        "  instance type: {} -> {}",
        args.instance_type.describe(),
        plan.instance.as_ref().map_or_else(
            || "(unresolved: needs form-data)".to_string(),
            |choice| redact_text(&choice.label),
        )
    );
    println!(
        "  ami:           {}",
        plan.ami.as_ref().map_or_else(
            || "(unresolved: needs form-data)".to_string(),
            |choice| redact_text(&choice.label),
        )
    );
    println!("  repository:    {}", redact_url(&plan.repository));
    println!(
        "  commit:        {}{}",
        redact_text(&plan.commit),
        plan.verified_remote_branch.as_ref().map_or_else(
            || " (explicit source override)".to_string(),
            |branch| format!(" (live-verified at {})", redact_text(branch))
        )
    );
    println!("  mode:          {}", plan.mode.as_str());
    println!("  vnnlib:        {}", redact_text(&plan.vnnlib_version));
    println!(
        "  gates:         can_submit={} scheduler_enabled={}",
        describe_flag(outcome.can_submit, outcome.authenticated),
        describe_flag(outcome.scheduler_enabled, outcome.authenticated)
    );
    if args.dry_run {
        println!("  dry-run:       payload built, nothing POSTed");
        if let Ok(pretty) = serde_json::to_string_pretty(&redact_value(display_payload)) {
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
            println!("  submission:    {}", redact_url(url));
        } else if let Some(resp) = &outcome.response {
            let safe_body = diagnostic_body(&resp.body);
            let trimmed = safe_body.trim();
            if !trimmed.is_empty() {
                println!("  response:      {trimmed}");
            }
        }
    } else {
        println!(
            "  POST:          skipped (can_submit was not true or scheduler was disabled; \
             use --force to attempt anyway)"
        );
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
    let source =
        resolve_submission_source(&root, args.repository.as_deref(), args.commit.as_deref())?;
    let artifact_state = prebuilt_state(&root, &source);
    let account_email = match &args.account_email {
        Some(email) => Some(email.clone()),
        None => load_credentials()?.map(|(email, _)| email),
    };

    let eml = render_request_email(
        &from_name,
        &from_email,
        &source.repository,
        &source.commit,
        args.tracks,
        args.instance_type,
        account_email.as_deref(),
        artifact_state,
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
            diagnostic_path(&output)
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
    prebuilt_state: PrebuiltState,
) -> String {
    let benchmarks = track_benchmarks(tracks, &[], true);
    let account_line = account_email.map_or_else(
        || "- Platform account: (to be created via the signup form)".to_string(),
        |email| format!("- Platform account: {email} (self-registered, pending activation)"),
    );
    let install_lines = match prebuilt_state {
        PrebuiltState::Present => {
            "- Install artifact: the selected commit contains a prebuilt triplet; the\n\
             installer verifies its checksum and provenance before using it.\n\
             - Fallback: if that artifact is rejected, installation is a networked source\n\
             build requiring crates.io/ORT access and authenticated read access to the\n\
             exact Git-pinned AY revision."
        }
        PrebuiltState::Absent => {
            "- Install artifact: no prebuilt/offline binary is present in the selected\n\
             commit. Installation therefore uses the networked source fallback, which\n\
             requires crates.io/ORT access and authenticated read access to the exact\n\
             Git-pinned AY revision."
        }
        PrebuiltState::Unverified => {
            "- Install artifact: the explicitly selected remote commit was not inspected\n\
             locally; please do not assume offline installation. Unless it contains a\n\
             valid prebuilt triplet, installation uses a networked source build requiring\n\
             crates.io/ORT and authenticated access to the exact Git-pinned AY revision."
        }
    };

    redact_text(&format!(
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
         v1 script contract. The harness entry points are prepared; installation is\n\
         automated but has the artifact/network requirements stated below:\n\
         \n\
         {account_line}\n\
         - Repository: {repository}\n\
         - Commit: {commit}\n\
         - Scripts: install_tool.sh / prepare_instance.sh / run_instance.sh in the\n\
           repository root (scripts_dir = \".\")\n\
         {install_lines}\n\
         - Solver licenses: no external Gurobi/MATLAB license is required\n\
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
    ))
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
    fn strict_payload_refuses_a_null_ami() {
        let form = json!({
            "instance_types": [{"value": 1, "label": "CPU: m5.16xlarge"}],
            "ami_options": [],
        });
        let plan = build_submission_plan(&submit_args(), Some(&form)).expect("plan");
        assert!(plan.instance.is_some());
        assert!(plan.ami.is_none());
        let error = plan.payload(true).expect_err("strict payload needs AMI");
        assert!(error.to_string().contains("AMI"));
        assert!(plan
            .payload(false)
            .expect("display payload")
            .get("ami")
            .is_some_and(Value::is_string));

        let null_form = json!({
            "instance_types": [{"value": 1, "label": "CPU: m5.16xlarge"}],
            "ami_options": [{"value": null, "label": "Ubuntu 24.04"}],
        });
        let null_plan =
            build_submission_plan(&submit_args(), Some(&null_form)).expect("null AMI plan");
        assert!(null_plan.ami.is_some());
        let error = null_plan
            .payload(true)
            .expect_err("null-valued AMI must fail");
        assert!(error.to_string().contains("AMI"));
        assert!(null_plan
            .payload(false)
            .expect("null AMI display payload")
            .get("ami")
            .is_some_and(Value::is_string));
    }

    #[test]
    fn post_requires_an_explicit_true_gate_unless_forced() {
        for can_submit in [None, Some(false)] {
            assert!(!post_is_authorized(false, false, can_submit, Some(true)));
        }
        assert!(post_is_authorized(false, false, Some(true), Some(true)));
        assert!(post_is_authorized(false, false, Some(true), None));
        assert!(!post_is_authorized(false, false, Some(true), Some(false)));
        assert!(post_is_authorized(false, true, None, Some(false)));
        assert!(!post_is_authorized(true, true, Some(true), Some(true)));
    }

    #[test]
    fn curl_disables_curlrc_first_and_pins_post_redirects() {
        let mut post = curl_command();
        apply_redirect_policy(&mut post, false);
        let post_args: Vec<_> = post
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert_eq!(post_args.first().map(String::as_str), Some("-q"));
        assert!(!post_args.iter().any(|arg| arg == "-L"));

        let mut get = curl_command();
        apply_redirect_policy(&mut get, true);
        let get_args: Vec<_> = get
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert_eq!(get_args.first().map(String::as_str), Some("-q"));
        assert!(get_args.iter().any(|arg| arg == "-L"));
        assert!(get_args.iter().any(|arg| arg == "--max-redirs"));
    }

    #[test]
    fn urls_and_secret_json_fields_are_redacted_everywhere() {
        let text = "first https://alice:hunter2@example.test/repo?token=abc#raw \
                    second https://bob:secret@other.test/x?password=pw";
        let safe = redact_text(text);
        for secret in ["alice", "hunter2", "abc", "bob", "secret", "pw", "#raw"] {
            assert!(!safe.contains(secret), "leaked {secret:?}: {safe}");
        }
        assert!(safe.contains("example.test"));
        assert!(safe.contains("other.test"));

        let safe_json = redact_value(&json!({
            "password": "cleartext",
            "csrf_token": "csrf-value",
            "repository": "https://git-user:git-pass@example.test/ny?key=value",
            "nested": [{"authorization": "Bearer value"}],
        }));
        let rendered = safe_json.to_string();
        for secret in [
            "cleartext",
            "csrf-value",
            "git-user",
            "git-pass",
            "value\"",
            "Bearer",
        ] {
            assert!(!rendered.contains(secret), "leaked {secret:?}: {rendered}");
        }

        let response = diagnostic_body(
            r#"{"password":"server-echo","url":"https://u:p@example.test/?token=raw"}"#,
        );
        assert!(!response.contains("server-echo"));
        assert!(!response.contains("u:p"));
        assert!(!response.contains("raw"));
    }

    #[test]
    fn email_redacts_repository_credentials_and_states_source_build_requirements() {
        let eml = render_request_email(
            "Test Person",
            "test@example.com",
            "https://git-user:git-password@example.test/ny?token=raw#secret",
            "abc123",
            TrackSelection::Regular,
            InstanceType::Cpu,
            None,
            PrebuiltState::Absent,
        );
        for secret in ["git-user", "git-password", "raw", "#secret"] {
            assert!(!eml.contains(secret), "email leaked {secret:?}");
        }
        assert!(eml.contains("no prebuilt/offline binary is present"));
        assert!(eml.contains("networked source fallback"));
        assert!(eml.contains("authenticated read access"));
        assert!(!eml.contains("single offline cargo build"));
        assert!(!eml.contains("no manual effort"));
    }

    #[cfg(unix)]
    #[test]
    fn private_files_are_atomic_nofollow_and_mode_0600() {
        use std::os::unix::fs::{symlink, MetadataExt as _, PermissionsExt as _};

        let directory = tempfile::tempdir().expect("tempdir");
        let victim = directory.path().join("victim");
        fs::write(&victim, b"do not overwrite").expect("victim");
        let destination = directory.path().join("credentials");
        symlink(&victim, &destination).expect("destination symlink");

        atomic_write_private(&destination, b"private", "test credentials").expect("atomic write");
        assert_eq!(fs::read(&victim).expect("victim read"), b"do not overwrite");
        assert_eq!(
            read_private_file(&destination, "test credentials")
                .expect("private read")
                .expect("present"),
            b"private"
        );
        let metadata = fs::symlink_metadata(&destination).expect("metadata");
        assert!(metadata.is_file());
        assert_eq!(metadata.mode() & 0o777, 0o600);

        let malicious_link = directory.path().join("linked-cookie");
        symlink(&victim, &malicious_link).expect("read symlink");
        assert!(read_private_file(&malicious_link, "test cookie jar").is_err());

        fs::set_permissions(&victim, fs::Permissions::from_mode(0o644)).expect("permissions");
        assert!(read_private_file(&victim, "unsafe credentials").is_err());
    }

    #[test]
    fn private_file_failures_are_propagated() {
        let directory = tempfile::tempdir().expect("tempdir");
        let not_a_directory = directory.path().join("plain-file");
        fs::write(&not_a_directory, b"x").expect("plain file");
        let impossible = not_a_directory.join("credentials");
        assert!(atomic_write_private(&impossible, b"secret", "test credentials").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn staged_cookie_jar_is_replaced_atomically_with_private_mode() {
        use std::os::unix::fs::PermissionsExt as _;

        let directory = tempfile::tempdir().expect("tempdir");
        let jar = directory.path().join("cookies");
        atomic_write_private(&jar, b"old-cookie", "test cookie jar").expect("old jar");

        let staged = stage_private_file(&jar, "test cookie jar").expect("stage jar");
        assert_eq!(fs::read(staged.path()).expect("staged read"), b"old-cookie");
        fs::write(staged.path(), b"new-cookie").expect("simulate curl write");
        persist_private_file(staged, &jar, "test cookie jar").expect("persist jar");

        assert_eq!(fs::read(&jar).expect("jar read"), b"new-cookie");
        assert_eq!(
            fs::metadata(&jar)
                .expect("jar metadata")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }

    fn run_git(root: &Path, args: &[&str]) {
        let output = Command::new("git")
            .current_dir(root)
            .env("GIT_TERMINAL_PROMPT", "0")
            .args(args)
            .output()
            .expect("run git");
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn upstream_fixture() -> (tempfile::TempDir, PathBuf) {
        let fixture = tempfile::tempdir().expect("fixture");
        let remote = fixture.path().join("remote.git");
        let work = fixture.path().join("work");
        fs::create_dir(&remote).expect("remote dir");
        fs::create_dir(&work).expect("work dir");
        run_git(&remote, &["init", "--bare", "--quiet"]);
        run_git(&work, &["init", "--quiet"]);
        run_git(&work, &["config", "user.name", "Test"]);
        run_git(&work, &["config", "user.email", "test@example.com"]);
        fs::write(work.join("tracked"), b"one").expect("tracked");
        run_git(&work, &["add", "tracked"]);
        run_git(&work, &["commit", "--quiet", "-m", "initial"]);
        run_git(&work, &["branch", "-M", "main"]);
        run_git(
            &work,
            &[
                "remote",
                "add",
                "origin",
                remote.to_str().expect("remote path"),
            ],
        );
        run_git(&work, &["push", "--quiet", "-u", "origin", "main"]);
        (fixture, work)
    }

    #[test]
    fn implicit_source_requires_clean_live_upstream_tip() {
        let (_fixture, work) = upstream_fixture();
        let source =
            resolve_submission_source(&work, None, None).expect("verified implicit source");
        assert_eq!(
            source.verified_remote_branch.as_deref(),
            Some("origin/main")
        );

        fs::write(work.join("untracked"), b"dirty").expect("untracked");
        let dirty = resolve_submission_source(&work, None, None).expect_err("dirty must fail");
        assert!(dirty.to_string().contains("dirty worktree"));
        fs::remove_file(work.join("untracked")).expect("remove untracked");

        fs::write(work.join("tracked"), b"two").expect("modify");
        run_git(&work, &["add", "tracked"]);
        run_git(&work, &["commit", "--quiet", "-m", "not pushed"]);
        // Simulate a stale/forged local remote-tracking tip. A check based on
        // `git branch -r --contains` would now accept HEAD even though the live
        // remote still points at the previous commit.
        run_git(&work, &["update-ref", "refs/remotes/origin/main", "HEAD"]);
        let stale =
            resolve_submission_source(&work, None, None).expect_err("unpushed tip must fail");
        assert!(stale.to_string().contains("does not match live upstream"));
    }

    #[test]
    fn explicit_source_override_is_all_or_nothing_and_bypasses_local_state() {
        let (_fixture, work) = upstream_fixture();
        fs::write(work.join("dirty"), b"intentional").expect("dirty");
        assert!(resolve_submission_source(&work, Some("https://example.test/ny"), None).is_err());
        assert!(resolve_submission_source(&work, None, Some("abc123")).is_err());
        let source = resolve_submission_source(
            &work,
            Some("https://user:password@example.test/ny"),
            Some("abc123"),
        )
        .expect("explicit pair");
        assert!(source.verified_remote_branch.is_none());
        assert_eq!(redact_url(&source.repository), "https://example.test/ny");
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
            PrebuiltState::Absent,
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
            PrebuiltState::Unverified,
        );
        assert!(eml
            .contains("Subject: VNN-COMP 2026: late tool-submission request - NY (Regular track)"));
        assert!(!eml.contains("all tracks"));
        assert!(eml.contains("24 total"));
    }
}
