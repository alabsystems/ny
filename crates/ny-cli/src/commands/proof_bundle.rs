// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ENGINE_PROGRAM §1.7 item B1 at the scored entry point: which proof sink a
//! `ny vnncomp v1` process wires, and the Proof profile's proof bundle
//! (`ny-proof-bundle/v2`, checked offline by `ny proof-check`, B2.1).
//!
//! # What the profile selects here
//!
//! The entry point resolves the proof profile once (`main.rs`) and calls
//! [`wire_scored_run`] once, before any lane runs:
//!
//! * `Competition` (what `run_instance.sh` runs): `NullSink`. Nothing is
//!   written, nothing is hashed, no thread is spawned, and hint capture stays
//!   off, so no lane even builds a hint.
//! * `Proof` (`--proof-profile proof`: banking, moat and checker runs): a
//!   [`BundleSink`] that hands every record to a background writer thread, and
//!   hint capture switched on ([`enable_hint_capture`], called HERE and
//!   nowhere else — ratcheted). The writer streams
//!   `<RESULTS_FILE>.proof.jsonl.partial`:
//!   1. a `header` (schema, the binary's canonical build line, the profile, the
//!      lane, the site and input identity spaces, and the sha256 of the ONNX
//!      and VNN-LIB files, hashed ON the writer thread);
//!   2. the network's `sites` (every Relu node's `input[0]` ONNX name) and the
//!      spec's `clauses` in the checker's canonical form, noted by the lane
//!      ([`note_sites_of`], [`note_clauses`]);
//!   3. `hint.alpha` tables and `node` records the margin-row lane streams for
//!      every AUTHORIZED frame (absolute literals in branch order, β, the
//!      classes each node discharges, retirements, root / frozen-dual /
//!      opaque claims);
//!   4. after RESULTS_FILE is written ([`after_publication_receipt`]): the `verdict`
//!      of the PUBLISHED token, a `published` trailer and the `end` record
//!      (line count and the sha256 of every preceding byte). Only a published
//!      `unsat` renames the stream to `<RESULTS_FILE>.proof.jsonl`; every
//!      other result leaves the `.partial` (there is no proof to check).
//!
//! A bundle that cannot be created wires `NullSink` and says
//! `[proof-artifact] artifact=none(bundle_unwritable: ...)`: proof mode never
//! refuses a certified verdict for want of an artifact.
//!
//! # Why none of this can move a verdict
//!
//! Every [`ny_propagate::certify::ProofSink`] method returns `()`, the sink
//! only ever SENDS into a channel, the lane streams only after its token
//! exists, hint ledgers have no reader, and [`after_publication_receipt`] runs only
//! after the verdict is on disk. Removing the whole module leaves every
//! verdict-bearing value bit-identical; what it costs is wall time, and only
//! in the Proof profile (the writer is off-thread; the final wait is bounded
//! by [`FINALISE_WAIT`]).
//!
//! # B2.2: every other unsat-publishing lane
//!
//! The in-process verifier's lanes — the graph ReLU-split drivers (single
//! objective, precomputed bounds, multi-objective disjunctive/conjunctive),
//! the graph input split and the lowdim tree — keep write-only PROOF TAPES
//! under the Proof profile (`ny_propagate::certify::lane_proof`; the capture
//! is armed around the verifier by [`stash_lane_proofs`]'s caller). After a
//! published `unsat` whose bundle the margin-row lane did not claim,
//! [`after_publication_receipt`] translates them with this module's own readers:
//!
//! * `sites`: each graph activation node the tapes split, resolved through
//!   the ONNX `Relu` node of the same name to its `input[0]`
//!   ([`relu_site_map`]);
//! * `clauses`: the VNN-LIB clauses in the checker's canonical form, exact
//!   rationals of the file's own decimals ([`canonical_clauses`]);
//! * `node`s: split literals with absolute neuron ids and the lane's β, or
//!   input-threshold literals (the box's faces inside the declared box), each
//!   discharging the clauses its objective entails; plus one `opaque` claim
//!   per clause no complete tape discharges;
//! * the `verdict` requires every clause, whatever door the token used.
//!
//! The observable is `[proof-lanes] tapes=<n> used=<n> nodes=<n> opaque=<n>
//! fallback_clauses=<n> sites=<n> lossy=<n>`.
//!
//! # B2.3: the margin-row lane's remaining routes
//!
//! * **Frozen-dual prunes.** A class the frozen-dual tier closed streams one
//!   `hint.fd_state` record (its class, the sha256 of the certified state
//!   the prunes rest on, sizes, root, the split order as `[site id, index]`)
//!   and one `hint.fd_lambda` record per Relu op (the state's slopes and
//!   boxes), then ONE `node` per prune: the split literals in branch order,
//!   `close: bound`, `src: fd_prune`, the class it discharges, and
//!   `hint.fd = {"state", "l0", "l1"}` — the dual point the prune was
//!   certified at. `ny proof-check` rebuilds its OWN frozen-dual state and
//!   re-evaluates each prune (its `fd` rung). A class whose trace is
//!   incomplete (the Proof-only record budget ran out) keeps its
//!   `close: fd` claim. `#fd-compact-checkable`: a class searched on a
//!   #breakpoint-dual RANKED compact state (more than 512 unstable neurons)
//!   streams its prunes only with its CLASS RELAXATION — `hint.fd_lambda`
//!   then carries the slope of every unstable neuron of the frame, the
//!   omitted rows' included, and `hint.fd_state.relaxed` counts them —
//!   recorded and verified under the Proof profile only; without it (table
//!   over its cap, or a table that does not re-derive the state's margin
//!   row) the class keeps its `close: fd` claim.
//! * **Wave leaves.** The wave engine's per-class leaves stream with their
//!   frame (literals in split order, β, one class each); a class whose every
//!   leaf streamed no longer carries the `opaque` wave claim.
//! * **Class trees and nested epoch subtrees** stream their own authorized
//!   frames (`class_part`, `subdomain`), every node under its absolute
//!   prefix, so the checker composes them through the same per-disjunct
//!   cover; a nested subtree's parent leaf is a `subtree` node that claims
//!   nothing.
//!
//! # Every published `unsat` bundle is well-formed (legacy lanes included)
//!
//! A lane with no tape (the sequential engine, GPU DomainList, the
//! input-split multi-clause disjunctive lane, MIP, root-only fast paths of
//! the graph lanes, the non-graph legacy doors: CZ hybrid, relational,
//! nonlinear interval BaB, cctsdb sub-boxes, TLL structure; the isomorphic
//! exact-emptiness door) streams no proof the checker can re-derive. Its
//! bundle used to stop at `header`, `verdict`, `published`, `end`, which
//! `ny proof-check` refuses as `malformed` (exit 30, `missing_sites`): the
//! soundness-emergency code, for a bundle that merely carries less than the
//! lane proved. The WRITER now completes every published `unsat` at its close
//! (one atomic [`Line::Finish`], after every record already queued):
//!
//! * `sites`: an empty `sites` record when no lane named a site;
//! * `clauses`: when no lane wrote them, the instance's clauses in the
//!   checker's canonical form ([`instance_clauses`], computed on the writer
//!   thread at stream start), or — when this module cannot render them (a
//!   dual-network or input-disjunction property, a spec its reader refuses)
//!   — the explicit `ny-proof-bundle/v2.1` form
//!   `{"id": 0, "unavailable": "<why>"}`: clause 0 stands for the whole
//!   property, which nothing in the bundle can check;
//! * claims: one whole-region `opaque` node per required clause the stream
//!   does not already claim, `src: "legacy:<authority>"` for a legacy door
//!   (`"unstreamed:authorized"` for an authorized token whose route streamed
//!   nothing). For a LEGACY token with no lane tape behind it
//!   ([`Completion::WholeRegion`]) only a whole-region node counts as a
//!   claim — a partial stream of ANOTHER lane (a margin-row subtree that
//!   never finished, while the internal verifier published) is not the
//!   verdict's proof; a token whose own proof is the stream
//!   ([`Completion::OwnStream`]: the margin-row lane's authorized frames, the
//!   lane tapes' complete-or-`lane_claim` translation, the input-head route's
//!   own claim) is completed only when it streamed NOTHING (an authorized
//!   route with no emitter: the isomorphic exact-emptiness door), so a class
//!   an emitter drops stays the cover gap it is (an emitter finding,
//!   `invalid`, never masked as `unconfirmed`);
//! * `verdict`: `required` is EVERY clause id of the `clauses` record (a
//!   published `unsat` claims the whole property, whatever door its token
//!   used; a legacy token carries no classes) and `source` names the token's
//!   authority.
//!
//! The checker re-derives nothing from an `opaque` claim, so such a bundle is
//! `unconfirmed` (exit 10, the classic sub-moat), never `verified`: the
//! completion adds claims, never certificates, and cannot make the checker
//! accept anything it could not prove before. A header input the writer
//! cannot hash as one regular file (a VNN-LIB 2.0 dual-network ONNX field)
//! is the v2.1 form `{"path": p, "unavailable": "<why>"}`, which forces the
//! unavailable clause form. The schema is v2.1 exactly when one of the two
//! forms is WRITTEN (review(legacy-bundle): a property this module cannot
//! render keeps the header undecided until a lane writes its own `clauses`
//! — then v2, an ordinary checkable bundle certlib can accept — or the close
//! writes the unavailable clause; the header is still the first line, see
//! `Stream`); every other
//! bundle stays `ny-proof-bundle/v2`, and a stream that already claimed every
//! clause under a token naming every clause is written byte-for-byte as
//! before.
//!
//! # Not yet in the bundle
//!
//! Per-column β, the domain-α carry, the wave's per-domain α and per-node
//! intermediate-bound lemmas are not emitted.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::Duration;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use ny_core::proof_profile::{ProofProfile, ResolvedProofProfile};
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::certify::{
    enable_hint_capture, install_proof_sink, AlphaRecord, FdStateRecord, GraphInstanceProperty,
    LaneProofCapture, LaneProofInstance, LeafRecord, NodeClose, NodeRecord, NullSink, ProofSink,
    SiteTensorRecord, SitesRecord, VerdictRecord,
};
use serde_json::json;
use sha2::{Digest, Sha256};

use super::unsat_receipt::PublishedUnsatSource;

#[cfg(test)]
#[path = "proof_bundle_retained_tests.rs"]
mod retained_tests;

#[path = "proof_bundle_prepared.rs"]
mod prepared_replay;

#[cfg(test)]
#[path = "proof_bundle_prepared_tests.rs"]
pub(crate) mod prepared_replay_tests;

/// The bundle's schema token (first line, `schema=`).
pub(crate) const BUNDLE_SCHEMA: &str = "ny-proof-bundle/v2";

/// The schema minor of a bundle that may carry the explicit `unavailable`
/// forms (a header input that is not one regular file, the whole property as
/// one unrenderable clause): see the module docs.
pub(crate) const BUNDLE_SCHEMA_V2_1: &str = "ny-proof-bundle/v2.1";

/// The node-site identity space: a Relu literal names the ONNX Relu node's
/// `input[0]` (through the `sites` record) and a C-order flat index.
pub(crate) const SITE_SPACE: &str = "onnx-relu-input/v1";

/// The input-literal identity space: `x_i` is the flat C-order index of the
/// ONNX graph's first non-initializer input, batch dimension stripped.
pub(crate) const INPUT_SPACE: &str = "onnx-input0-flat/v1";

/// The header's `lane`: the entry point that wrote the bundle (B2.2: several
/// lanes stream into one bundle; each node's `src` names its producer).
pub(crate) const BUNDLE_LANE: &str = "vnncomp_v1";

/// Longest the entry point waits, after publication, for the writer thread to
/// drain; past it the `.partial` is left as is (the verdict is already
/// published and nothing here can touch it). Only ever paid under Proof.
pub(crate) const FINALISE_WAIT: Duration = Duration::from_secs(60);

/// Which sink a profile wires at the scored entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SinkPlan {
    /// `NullSink`: nothing written.
    Null,
    /// The streaming proof-hint bundle next to RESULTS_FILE.
    Bundle,
}

impl SinkPlan {
    /// The profile's plan. `Competition` never writes an artifact.
    #[must_use]
    pub(crate) const fn for_profile(mode: ProofProfile) -> Self {
        if mode.writes_artifacts() {
            Self::Bundle
        } else {
            Self::Null
        }
    }
}

/// How the writer completes a published `unsat`'s claims (module docs,
/// "Every published `unsat` bundle is well-formed").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Completion {
    /// The stream is the published token's own proof (the margin-row lane's
    /// authorized frames, the lane tapes' complete-or-`lane_claim`
    /// translation, a route that streamed its own claim): it is completed
    /// only when it streamed NO node at all (an authorized route with no
    /// emitter), so a class an emitter drops stays a cover gap for the
    /// checker to report rather than an `unconfirmed` claim.
    OwnStream,
    /// A legacy door's word with no lane tape behind it: every clause no
    /// WHOLE-REGION node claims gets a whole-region claim (a partial stream
    /// of another lane is never taken for the verdict's proof).
    WholeRegion,
}

/// The verdict the writer closes the stream with (the node count is the
/// writer's own; `required` is every clause id of the bundle's `clauses`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VerdictInfo {
    /// Token source (the verdict record's `source`, and the claims' route).
    pub(crate) source: PublishedUnsatSource,
    /// The token's minimum certified margin.
    pub(crate) min_margin: f64,
    /// How the writer completes the stream's claims.
    pub(crate) completion: Completion,
}

/// The `src` of a whole-region claim the writer adds for `source`: the
/// legacy door's authority, or `unstreamed:<source>` for a non-legacy token
/// whose route streamed nothing (`unstreamed:authorized` for the spine's
/// authorized token; the original-owner sources name their own token).
#[must_use]
pub(crate) fn claim_route(source: PublishedUnsatSource) -> String {
    match source.legacy() {
        Some(authority) => format!("legacy:{}", authority.token()),
        None => format!("unstreamed:{}", source.token()),
    }
}

/// What one streamed node claims, as the writer's completion reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Claim {
    /// No literal at all: the node's region is the whole input region.
    whole_region: bool,
    /// Claimed clause ids; `None` = every clause (an emptiness node).
    ids: Option<Vec<usize>>,
}

impl Claim {
    fn of(node: &NodeRecord) -> Self {
        Self {
            whole_region: node.lits.is_empty() && node.xlits.is_empty(),
            ids: match (node.close, &node.dis) {
                (NodeClose::Subtree, _) => Some(Vec::new()),
                (_, None) => None,
                (_, Some(v)) => Some(v.clone()),
            },
        }
    }
}

enum Line {
    /// A `node` record and what it claims.
    Node(String, Claim),
    /// A hint record.
    Record(String),
    /// A `sites` record.
    Sites(String),
    /// A region-0 `clauses` record and its clause ids.
    Clauses(String, Vec<usize>),
    /// Write the instance's canonical clauses now (the writer's own
    /// rendering, ids by position), unless a stream already wrote its own.
    #[cfg_attr(
        not(feature = "mip"),
        expect(
            dead_code,
            reason = "canonical instance clauses are requested by the MIP hull route"
        )
    )]
    InstanceClauses,
    /// A genuine prepared export (its own `sites`, `clauses` and complete
    /// cover nodes), retained with its original owner.
    PreparedReplay(prepared_replay::ReplayPayload),
    /// The close, written atomically after every line queued before it: the
    /// completion and `verdict` (for a published `unsat`), the `published`
    /// trailer and the `end` record. Nothing queued after it is written.
    Finish(Option<VerdictPayload>, String),
}

/// Field drop order releases the actual class buffer before its byte claim,
/// on success, cancellation, channel failure and I/O error alike.
struct VerdictPayload {
    info: VerdictInfo,
    owner: Option<Arc<RetainedBundleOwner>>,
}

/// A queued verdict cannot outlive its original source or its paid buffers.
/// The sender retains another alias through the actual final bundle rename.
struct RetainedBundleOwner {
    receipt: super::unsat_receipt::UnsatReceipt,
    _storage: ny_propagate::certify::work::PersistentReservation,
    replay_quote: Option<prepared_replay::ReplayQuote>,
}

/// The Proof profile's sink: every record goes to the writer thread.
pub(crate) struct BundleSink {
    tx: mpsc::Sender<Line>,
}

impl ProofSink for BundleSink {
    /// Positional (frame-relative) leaves cannot be resolved offline: v2
    /// writes the lane's `node` records instead.
    fn leaf(&mut self, _leaf: &LeafRecord) {}

    fn node(&mut self, node: &NodeRecord) {
        let _ = self.tx.send(Line::Node(node_line(node), Claim::of(node)));
    }

    fn alpha(&mut self, alpha: &AlphaRecord) {
        let _ = self.tx.send(Line::Record(alpha_line(alpha)));
    }

    fn sites(&mut self, sites: &SitesRecord) {
        let _ = self.tx.send(Line::Sites(sites_line(sites)));
    }

    /// B2.3: a frozen-dual state's `hint.fd_state` record, then one
    /// `hint.fd_lambda` record per Relu op of its slopes.
    fn fd_state(&mut self, state: &FdStateRecord) {
        let _ = self.tx.send(Line::Record(fd_state_line(state)));
        for line in fd_lambda_lines(state) {
            let _ = self.tx.send(Line::Record(line));
        }
    }

    /// The verdict record is written by [`BundleWriter::finalise`], which
    /// knows the published token's source.
    fn finish(&mut self, _verdict: &VerdictRecord) {}
}

/// What the header records (hashed on the writer thread, never on the verdict
/// path).
pub(crate) struct Header {
    pub(crate) profile: ResolvedProofProfile,
    pub(crate) onnx: PathBuf,
    pub(crate) vnnlib: PathBuf,
}

/// The schema a bundle is written under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Schema {
    /// `ny-proof-bundle/v2`.
    V2,
    /// `ny-proof-bundle/v2.1`: the explicit `unavailable` forms may occur.
    V2_1,
}

impl Schema {
    /// The header's `schema` token.
    #[must_use]
    pub(crate) const fn token(self) -> &'static str {
        match self {
            Self::V2 => BUNDLE_SCHEMA,
            Self::V2_1 => BUNDLE_SCHEMA_V2_1,
        }
    }
}

/// What the writer thread reports when it drains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Drained {
    /// Node records written before the verdict (the claims included).
    nodes: u64,
    /// Whole-region claims the completion added.
    claims: u64,
    /// Duplicate `sites` / `clauses` records refused (a second one would make
    /// the bundle malformed).
    duplicates: u64,
    schema: Schema,
}

/// The open bundle: its channel, its writer thread and its two paths.
pub(crate) struct BundleWriter {
    tx: mpsc::Sender<Line>,
    done: mpsc::Receiver<Result<Drained, String>>,
    thread: Option<JoinHandle<()>>,
    partial: PathBuf,
    final_path: PathBuf,
}

/// What finalisation did (the `[proof-artifact] artifact=` value).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Artifact {
    /// The bundle was finalised at this path with this many node records,
    /// `claims` of them whole-region claims the completion added.
    Written {
        path: PathBuf,
        nodes: u64,
        claims: u64,
        schema: Schema,
    },
    /// No checkable artifact, and why (a `.partial` may remain).
    None(String),
}

impl Artifact {
    /// The observable value: `written path=<p> nodes=<n> schema=<s>[ claims=<k>]`
    /// or `none(<why>)`.
    #[must_use]
    pub(crate) fn render(&self) -> String {
        match self {
            // The schema is spelled as ONE literal per arm (not a `{}`
            // placeholder) so the `schema=ny-proof-bundle/v2` marker survives
            // in the binary's strings; the bundle test pins it to
            // BUNDLE_SCHEMA.
            Self::Written {
                path,
                nodes,
                claims,
                schema,
            } => {
                let added = if *claims > 0 {
                    format!(" claims={claims}")
                } else {
                    String::new()
                };
                match schema {
                    Schema::V2 => format!(
                        "written path={} nodes={nodes} schema=ny-proof-bundle/v2{added}",
                        path.display()
                    ),
                    Schema::V2_1 => format!(
                        "written path={} nodes={nodes} schema=ny-proof-bundle/v2.1{added}",
                        path.display()
                    ),
                }
            }
            Self::None(why) => format!("none({why})"),
        }
    }
}

/// `<results>.proof.jsonl` (the finalised bundle).
#[must_use]
pub(crate) fn bundle_path(results_file: &Path) -> PathBuf {
    let mut name = results_file.as_os_str().to_owned();
    name.push(".proof.jsonl");
    PathBuf::from(name)
}

/// `<results>.proof.jsonl.partial` (the stream before publication).
#[must_use]
pub(crate) fn partial_path(results_file: &Path) -> PathBuf {
    let mut name = bundle_path(results_file).into_os_string();
    name.push(".partial");
    PathBuf::from(name)
}

impl BundleWriter {
    /// Create the `.partial` stream (synchronously, so an unwritable directory
    /// is known when the sink is wired) and start the writer thread, which
    /// writes the header first.
    pub(crate) fn create(
        results_file: &Path,
        header: Header,
    ) -> std::io::Result<(Self, BundleSink)> {
        let final_path = bundle_path(results_file);
        let partial = partial_path(results_file);
        let file = File::create(&partial)?;
        let (tx, rx) = mpsc::channel::<Line>();
        let (done_tx, done) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("ny-proof-bundle".to_string())
            .spawn(move || {
                let result = write_stream(file, &header, &rx).map_err(|e| e.to_string());
                let _ = done_tx.send(result);
            })?;
        let sink = BundleSink { tx: tx.clone() };
        Ok((
            Self {
                tx,
                done,
                thread: Some(thread),
                partial,
                final_path,
            },
            sink,
        ))
    }

    /// Another sink into this bundle's stream (the close streams the lane
    /// tapes through it, exactly as the installed process sink would).
    #[must_use]
    pub(crate) fn sink(&self) -> BundleSink {
        BundleSink {
            tx: self.tx.clone(),
        }
    }

    /// Send the spec's clauses for a margin-row robustness property (true
    /// class `t`, adversarial classes `adv`): region 0, clause id = class.
    pub(crate) fn clauses(&self, t: usize, adv: &[usize]) {
        let _ = self
            .tx
            .send(Line::Clauses(margin_clauses_line(t, adv), adv.to_vec()));
    }

    /// Send a general `clauses` record: region 0, `(id, canonical atoms)`.
    pub(crate) fn clauses_list(&self, list: &[(usize, Vec<String>)]) {
        let ids = list.iter().map(|(id, _)| *id).collect();
        let _ = self.tx.send(Line::Clauses(clauses_line(list), ids));
    }

    /// Ask the writer for the instance's canonical `clauses` record now (its
    /// own rendering, computed at stream start off the verdict path; ids by
    /// position). A property it cannot render keeps the close's unavailable
    /// form (clause 0).
    #[cfg_attr(
        not(feature = "mip"),
        expect(
            dead_code,
            reason = "MIP hull routes request positional instance clauses"
        )
    )]
    pub(crate) fn request_instance_clauses(&self) {
        let _ = self.tx.send(Line::InstanceClauses);
    }

    /// After publication: close the stream (for a published `unsat`, the
    /// completion and the `verdict` when `verdict` is given; always the
    /// `published` trailer and the `end` record), wait at most `wait` for the
    /// writer, and rename the `.partial` to the final bundle iff
    /// `published == "unsat"`.
    pub(crate) fn finalise(
        self,
        published: &str,
        verdict: Option<VerdictInfo>,
        wait: Duration,
    ) -> Artifact {
        self.finalise_owned(published, verdict, None, wait)
    }

    fn finalise_owned(
        mut self,
        published: &str,
        verdict: Option<VerdictInfo>,
        owner: Option<Arc<RetainedBundleOwner>>,
        wait: Duration,
    ) -> Artifact {
        let verdict = verdict.map(|info| VerdictPayload {
            info,
            owner: owner.clone(),
        });
        let _ = self.tx.send(Line::Finish(verdict, published.to_string()));
        let drained = match self.done.recv_timeout(wait) {
            Ok(Ok(drained)) => drained,
            Ok(Err(e)) => return Artifact::None(format!("bundle_write_failed: {e}")),
            Err(_) => {
                // Detach: the verdict is published; the writer may finish the
                // `.partial` on its own, or not.
                drop(self.thread.take());
                return Artifact::None(format!("writer_timeout_{}s", wait.as_secs()));
            }
        };
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        if published != "unsat" {
            return Artifact::None(format!("verdict={published}"));
        }
        if let Some(owner) = &owner {
            if let Err(error) = owner.receipt.checkpoint_owner(None) {
                return Artifact::None(format!("retained_owner_before_bundle_rename: {error}"));
            }
        }
        if let Err(e) = std::fs::rename(&self.partial, &self.final_path) {
            return Artifact::None(format!("bundle_rename_failed: {e}"));
        }
        if drained.duplicates > 0 {
            eprintln!(
                "[proof-artifact] refused {} duplicate sites/clauses record(s)",
                drained.duplicates
            );
        }
        Artifact::Written {
            path: self.final_path.clone(),
            nodes: drained.nodes,
            claims: drained.claims,
            schema: drained.schema,
        }
    }
}

/// A line writer that keeps the running sha256 and line count the `end`
/// record certifies.
struct Digesting<W: Write> {
    w: W,
    sha: Sha256,
    lines: u64,
    // Declaration order drops the actual stream buffer before its claim,
    // including early I/O errors and detached writer completion.
    retained_owner: Option<Arc<RetainedBundleOwner>>,
}

impl<W: Write> Digesting<W> {
    fn line(&mut self, text: &str) -> std::io::Result<()> {
        self.w.write_all(text.as_bytes())?;
        self.w.write_all(b"\n")?;
        self.sha.update(text.as_bytes());
        self.sha.update(b"\n");
        self.lines += 1;
        Ok(())
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// What the stream claimed so far, per clause id (bounded by the clause
/// count, never by the node count).
#[derive(Debug, Default)]
struct Claimed {
    /// Some node was streamed (any region, any claim).
    streamed: bool,
    /// Some WHOLE-REGION node claims the id (or every id: an emptiness node).
    whole: BTreeSet<usize>,
    whole_all: bool,
}

impl Claimed {
    fn note(&mut self, claim: &Claim) {
        self.streamed = true;
        if !claim.whole_region {
            return;
        }
        match &claim.ids {
            None => self.whole_all = true,
            Some(ids) => self.whole.extend(ids.iter().copied()),
        }
    }

    /// A genuine prepared export's cover node (`classes`: the original
    /// objective it closes; `every_clause`: an emptiness close).
    fn note_replay(&mut self, whole_region: bool, every_clause: bool, classes: &[usize]) {
        self.streamed = true;
        if !whole_region {
            return;
        }
        if every_clause {
            self.whole_all = true;
        } else {
            self.whole.extend(classes.iter().copied());
        }
    }

    /// Whether clause `k` still needs the writer's claim under `completion`:
    /// a legacy word (or an unavailable bundle, where nothing else can
    /// count) needs a WHOLE-REGION claim of `k`; a token's own stream is
    /// completed only when it streamed nothing.
    fn needs(&self, k: usize, completion: Completion, unavailable: bool) -> bool {
        if unavailable || completion == Completion::WholeRegion {
            !(self.whole_all || self.whole.contains(&k))
        } else {
            !self.streamed
        }
    }
}

/// One whole-region `opaque` claim of clause `clause` (the writer's
/// completion): no literal, no hint the checker could use.
fn claim_line(id: u64, clause: usize, src: &str) -> String {
    json!({
        "kind": "node",
        "id": id,
        "region": 0,
        "lits": [],
        "close": NodeClose::Opaque.token(),
        "src": src,
        "dis": [clause],
        "hint": {
            "beta": [],
            "alpha": null,
            "lift": false,
            "margin": null,
            "empty_site": null,
        },
    })
    .to_string()
}

/// The v2.1 `clauses` record of a property this module cannot render: one
/// clause, id 0, standing for the whole property.
fn unavailable_clauses_line(why: &str) -> String {
    json!({
        "kind": "clauses",
        "region": 0,
        "list": [{"id": 0, "unavailable": why}],
    })
    .to_string()
}

/// The bundle's lines in stream order, the header FIRST: lines queued before
/// the schema is decided are held and written right after the header.
/// review(legacy-bundle): the header names `ny-proof-bundle/v2.1` only when
/// the bundle WRITES an `unavailable` form. An unavailable input decides
/// v2.1 at once and a property this module renders decides v2 at once; a
/// property it cannot render stays undecided until a lane writes its OWN
/// `clauses` record (v2: the fallback is then never used, so the bundle is
/// an ordinary checkable one — a margin-row stream certlib can ACCEPT, which
/// it never does for a v2.1 header) or the close needs the unavailable
/// fallback clause (v2.1). In practice nothing but a lane's `sites` record
/// precedes its `clauses`, so what is held is a line or two.
struct Stream<'h, W: Write> {
    w: Digesting<W>,
    header: &'h Header,
    onnx: InputRef,
    vnnlib: InputRef,
    schema: Option<Schema>,
    held: Vec<String>,
}

impl<W: Write> Stream<'_, W> {
    /// Decide the schema (once; a later call is a no-op): write the header,
    /// then every held line in order.
    fn open(&mut self, schema: Schema) -> std::io::Result<()> {
        if self.schema.is_some() {
            return Ok(());
        }
        self.schema = Some(schema);
        self.w
            .line(&header_line(self.header, schema, &self.onnx, &self.vnnlib))?;
        for text in std::mem::take(&mut self.held) {
            self.w.line(&text)?;
        }
        Ok(())
    }

    /// One line after the header (held until the schema is decided).
    fn line(&mut self, text: String) -> std::io::Result<()> {
        if self.schema.is_some() {
            self.w.line(&text)
        } else {
            self.held.push(text);
            Ok(())
        }
    }
}

#[allow(clippy::too_many_lines)]
fn write_stream(
    file: File,
    header: &Header,
    rx: &mpsc::Receiver<Line>,
) -> std::io::Result<Drained> {
    // Off the verdict path: hash the inputs and render the instance's
    // clauses (the completion's fallback) before the header, which names the
    // schema they decide ([`Stream`]).
    let onnx = input_ref(&header.onnx);
    let vnnlib = input_ref(&header.vnnlib);
    let inputs_unavailable = matches!(onnx, InputRef::Unavailable { .. })
        || matches!(vnnlib, InputRef::Unavailable { .. });
    let fallback = if inputs_unavailable {
        Err("inputs_unavailable".to_string())
    } else {
        instance_clauses(&header.vnnlib)
    };
    let mut w = Stream {
        w: Digesting {
            w: BufWriter::new(file),
            sha: Sha256::new(),
            lines: 0,
            retained_owner: None,
        },
        header,
        onnx,
        vnnlib,
        schema: None,
        held: Vec::new(),
    };
    if inputs_unavailable {
        w.open(Schema::V2_1)?;
    } else if fallback.is_ok() {
        w.open(Schema::V2)?;
    }
    let mut drained = Drained {
        nodes: 0,
        claims: 0,
        duplicates: 0,
        schema: Schema::V2,
    };
    let mut sites_written = false;
    let mut clause_ids: Option<Vec<usize>> = None;
    let mut claimed = Claimed::default();
    for line in rx {
        match line {
            Line::Node(text, claim) => {
                w.line(text)?;
                claimed.note(&claim);
                drained.nodes += 1;
            }
            Line::Record(text) => w.line(text)?,
            Line::Sites(text) => {
                if sites_written {
                    drained.duplicates += 1;
                } else {
                    w.line(text)?;
                    sites_written = true;
                }
            }
            Line::Clauses(text, ids) => {
                if clause_ids.is_some() {
                    drained.duplicates += 1;
                } else {
                    // A lane's own clauses: the unavailable fallback is never
                    // written, so an undecided schema is v2.
                    w.open(Schema::V2)?;
                    w.line(text)?;
                    clause_ids = Some(ids);
                }
            }
            Line::InstanceClauses => {
                // A route whose own claim names the instance's clauses BY
                // POSITION (the original-hull close): the property's canonical
                // clauses now, unless a stream already wrote its own.
                if clause_ids.is_none() {
                    if let Ok(list) = &fallback {
                        w.open(Schema::V2)?;
                        w.line(clauses_line(list))?;
                        clause_ids = Some(list.iter().map(|(id, _)| *id).collect());
                    }
                }
            }
            Line::PreparedReplay(payload) => {
                // Retain both original source and complete export admission
                // until actual buffered bytes flush/sync and bundle rename.
                w.w.retained_owner = Some(payload.owner.clone());
                owner_checkpoint(&payload.owner)?;
                payload
                    .owner
                    .receipt
                    .validate_original_prepared_asset_hashes(
                        input_digest(&w.onnx)?,
                        input_digest(&w.vnnlib)?,
                        None,
                    )
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                let authority = payload
                    .owner
                    .receipt
                    .original_prepared_authority()
                    .ok_or_else(|| {
                        std::io::Error::other("prepared replay lacks genuine original owner")
                    })?;
                // The export is the published token's own stream: its OWN
                // `sites` and `clauses` (clause id = original class) and its
                // complete cover. Behind another stream's `sites`/`clauses` it
                // could only be a duplicate (malformed) or a foreign-identity
                // bundle, so it refuses the artifact instead (merge of the
                // legacy-bundle writer with the prepared export: no final
                // bundle; the published result is unchanged).
                if sites_written || clause_ids.is_some() {
                    return Err(std::io::Error::other(
                        "prepared replay behind another stream's sites/clauses",
                    ));
                }
                // A lane's own clauses: an undecided schema is v2.
                w.open(Schema::V2)?;
                for record in &payload.records {
                    owner_checkpoint(&payload.owner)?;
                    w.w.line(&record.text)?;
                    match record.kind {
                        prepared_replay::ReplayKind::Sites => sites_written = true,
                        prepared_replay::ReplayKind::Clauses => {
                            clause_ids = Some(authority.classes().to_vec());
                        }
                        prepared_replay::ReplayKind::Node {
                            whole_region,
                            every_clause,
                        } => {
                            claimed.note_replay(whole_region, every_clause, authority.classes());
                            drained.nodes += 1;
                        }
                    }
                }
                drop(authority);
                owner_checkpoint(&payload.owner)?;
            }
            Line::Finish(verdict, published) => {
                // v2.1 exactly when the close writes the unavailable clause.
                let unavailable_clause =
                    verdict.is_some() && clause_ids.is_none() && fallback.is_err();
                w.open(if unavailable_clause {
                    Schema::V2_1
                } else {
                    Schema::V2
                })?;
                if let Some(VerdictPayload { info: v, owner }) = verdict {
                    w.w.retained_owner.clone_from(&owner);
                    if let Some(owner) = &owner {
                        owner_checkpoint(owner)?;
                        if owner.receipt.original_prepared_authority().is_some() {
                            owner
                                .receipt
                                .validate_original_prepared_asset_hashes(
                                    input_digest(&w.onnx)?,
                                    input_digest(&w.vnnlib)?,
                                    None,
                                )
                                .map_err(|error| std::io::Error::other(error.to_string()))?;
                        }
                    }
                    if !sites_written {
                        w.line(sites_line(&SitesRecord {
                            tensors: Vec::new(),
                        }))?;
                    }
                    // Nothing in a bundle with an unavailable input or
                    // clause is checkable: only whole-region claims count.
                    let mut unavailable = inputs_unavailable;
                    let ids = match clause_ids.take() {
                        Some(ids) => ids,
                        None => match &fallback {
                            Ok(list) => {
                                w.line(clauses_line(list))?;
                                list.iter().map(|(id, _)| *id).collect()
                            }
                            Err(why) => {
                                w.line(unavailable_clauses_line(why))?;
                                unavailable = true;
                                vec![0]
                            }
                        },
                    };
                    let mut required: Vec<usize> = ids;
                    required.sort_unstable();
                    required.dedup();
                    let src = claim_route(v.source);
                    for &k in &required {
                        if claimed.needs(k, v.completion, unavailable) {
                            let id = ny_propagate::certify::reserve_node_id();
                            w.line(claim_line(id, k, &src))?;
                            drained.nodes += 1;
                            drained.claims += 1;
                        }
                    }
                    let text = verdict_line(&v, &required, drained.nodes);
                    if let Some(owner) = &owner {
                        let actual = required
                            .capacity()
                            .checked_mul(size_of::<usize>())
                            .and_then(|bytes| bytes.checked_mul(3))
                            .and_then(|bytes| bytes.checked_add(text.capacity()))
                            .ok_or_else(|| {
                                std::io::Error::other("retained verdict capacity overflow")
                            })?;
                        if actual > owner._storage.bytes() {
                            return Err(std::io::Error::other(
                                "retained verdict allocator capacity exceeds admission",
                            ));
                        }
                    }
                    w.line(text)?;
                    if let Some(owner) = &owner {
                        owner_checkpoint(owner)?;
                    }
                }
                w.line(json!({"kind": "published", "result": published}).to_string())?;
                let digest = hex(&w.w.sha.clone().finalize());
                let end = json!({
                    "kind": "end",
                    "lines": w.w.lines + 1,
                    "sha256": digest,
                })
                .to_string();
                w.line(end)?;
                break;
            }
        }
    }
    // A stream dropped without its close still leaves its lines (a
    // `.partial` is never a bundle).
    w.open(Schema::V2)?;
    drained.schema = w.schema.unwrap_or(Schema::V2);
    w.w.w.flush()?;
    w.w.w.get_ref().sync_all()?;
    if let Some(owner) = &w.w.retained_owner {
        owner_checkpoint(owner)?;
    }
    Ok(drained)
}

/// The retained owner's live check, as a writer I/O error.
fn owner_checkpoint(owner: &RetainedBundleOwner) -> std::io::Result<()> {
    owner
        .receipt
        .checkpoint_owner(None)
        .map_err(|error| std::io::Error::other(error.to_string()))
}

/// The original source digest one header input names (a retained owner's
/// asset-hash binding): the writer's own hash of the regular file.
fn input_digest(input: &InputRef) -> std::io::Result<[u8; 32]> {
    match input {
        InputRef::File { sha256, .. } => digest_bytes(Some(sha256)),
        InputRef::Unavailable { .. } => digest_bytes(None),
    }
}

/// [`input_digest`] read back from a written header record.
#[cfg(test)]
fn header_asset_digest(record: &serde_json::Value, asset: &str) -> std::io::Result<[u8; 32]> {
    digest_bytes(record[asset]["sha256"].as_str())
}

fn digest_bytes(encoded: Option<&str>) -> std::io::Result<[u8; 32]> {
    let encoded = encoded
        .filter(|value| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .ok_or_else(|| std::io::Error::other("retained bundle has no original source digest"))?;
    let mut digest = [0u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16)
            .map_err(|error| std::io::Error::other(error.to_string()))?;
    }
    Ok(digest)
}

/// One header input as the writer hashed it: a regular file's digest, or why
/// there is none (the v2.1 `unavailable` form).
enum InputRef {
    File {
        path: String,
        sha256: String,
        bytes: u64,
    },
    Unavailable {
        path: String,
        why: String,
    },
}

fn input_ref(path: &Path) -> InputRef {
    match crate::build_id::sha256_regular_file_nofollow(path) {
        Ok((canonical, sha256, bytes)) => InputRef::File {
            path: canonical.display().to_string(),
            sha256,
            bytes,
        },
        Err(e) => InputRef::Unavailable {
            path: path.display().to_string(),
            why: format!("{e:#}"),
        },
    }
}

impl InputRef {
    fn json(&self) -> serde_json::Value {
        match self {
            Self::File {
                path,
                sha256,
                bytes,
            } => json!({"path": path, "sha256": sha256, "bytes": bytes}),
            Self::Unavailable { path, why } => json!({"path": path, "unavailable": why}),
        }
    }
}

fn header_line(header: &Header, schema: Schema, onnx: &InputRef, vnnlib: &InputRef) -> String {
    json!({
        "kind": "header",
        "schema": schema.token(),
        "build": crate::build_id::current_line(),
        "profile": header.profile.mode.token(),
        "profile_source": header.profile.source.token(),
        "lane": BUNDLE_LANE,
        "site_space": SITE_SPACE,
        "input_space": INPUT_SPACE,
        "onnx": onnx.json(),
        "vnnlib": vnnlib.json(),
    })
    .to_string()
}

/// A positional leaf path as `t<layer>.<pos><+|->` / `h<index><+|->` tokens
/// (`+` is the `>= 0` half-space), in the path's canonical (site-sorted)
/// order; graph-lane sites are `g<node>.<neuron><+|->` (a ReLU split at zero)
/// and `p<node>.<input>.<neuron>@<point bits, hex><+|->` (a GenBaB point).
/// Diagnostic only: frame-relative positions are not a checkable schema (the
/// v2 bundle carries absolute `node` literals instead). Only the bundle tests
/// read it.
#[cfg(test)]
#[must_use]
pub(crate) fn path_tokens(leaf: &LeafRecord) -> Vec<String> {
    use ny_propagate::certify::CoverSite;
    leaf.path
        .literals()
        .iter()
        .map(|lit| {
            let sign = if lit.positive { '+' } else { '-' };
            match lit.site {
                CoverSite::Trunk { layer, pos } => format!("t{layer}.{pos}{sign}"),
                CoverSite::Head { index } => format!("h{index}{sign}"),
                // #graph-bab-cover-ledger: graph-lane split sites (the graph
                // lane streams no leaves yet; the tokens are fixed now so a
                // later stream needs no schema change).
                CoverSite::GraphRelu { node, neuron } => format!("g{node}.{neuron}{sign}"),
                CoverSite::GraphPoint {
                    node,
                    input,
                    neuron,
                    point_bits,
                } => format!("p{node}.{input}.{neuron}@{point_bits:08x}{sign}"),
            }
        })
        .collect()
}

/// A JSON number, or `null` for a non-finite value.
fn num(x: f64) -> serde_json::Value {
    if x.is_finite() {
        json!(x)
    } else {
        serde_json::Value::Null
    }
}

/// A v2 `node` record: `[site id, flat index, side]` literals (the site id is
/// the Relu op index the `sites` record names), branch order.
pub(crate) fn node_line(node: &NodeRecord) -> String {
    let mut lits: Vec<serde_json::Value> = node
        .lits
        .iter()
        .map(|l| json!([l.site.op, l.site.index, u8::from(l.positive)]))
        .collect();
    // B2.2 input-threshold literals: `["x", index, "<decimal>", side]`, the
    // threshold as its shortest round-trip decimal (the checker reads it back
    // with `f64::from_str`, exactly).
    lits.extend(node.xlits.iter().map(|l| {
        json!([
            "x",
            l.index,
            format!("{:?}", l.threshold),
            u8::from(l.positive)
        ])
    }));
    let dis = match &node.dis {
        None => json!("*"),
        Some(v) => json!(v),
    };
    // No β (a lane that streams none, or a box leaf): `null`, not a
    // misaligned empty list the checker would count as an invalid hint. The
    // checker aligns β with ALL of a node's literals, input thresholds
    // included (review(b2-lanes): a box leaf's `[]` against its `x` literals
    // was counted `hints_invalid` once per box node).
    let beta = if node.beta.is_empty() && !(node.lits.is_empty() && node.xlits.is_empty()) {
        serde_json::Value::Null
    } else {
        serde_json::Value::Array(node.beta.iter().map(|b| num(*b)).collect())
    };
    let mut hint = json!({
        "beta": beta,
        "alpha": node.alpha,
        "lift": node.lift,
        "margin": num(node.margin),
        "empty_site": node.empty_site.map(|s| json!([s.op, s.index])),
    });
    // B2.3: a frozen-dual prune's dual point, aligned with `lits` (a
    // non-finite multiplier is `null`, which the checker reads as 0).
    if let (Some(fd), Some(h)) = (&node.fd, hint.as_object_mut()) {
        let nums = |v: &[f64]| -> Vec<serde_json::Value> { v.iter().map(|x| num(*x)).collect() };
        h.insert(
            "fd".to_string(),
            json!({"state": fd.state, "l0": nums(&fd.l0), "l1": nums(&fd.l1)}),
        );
    }
    json!({
        "kind": "node",
        "id": node.id,
        "region": node.region,
        "lits": lits,
        "close": node.close.token(),
        "src": node.src,
        "dis": dis,
        "hint": hint,
    })
    .to_string()
}

/// B2.3: a frozen-dual state's identity (`hint.fd_state`): its class, the
/// sha256 of the certified state its prunes rest on, sizes, root, the split
/// order as `[site id, index]` pairs and the prune count.
/// `#fd-compact-checkable`: `relaxed` counts the relaxation symbols its
/// `hint.fd_lambda` table covers (`unstable` rows for a full state; every
/// unstable neuron of the frame for a ranked compact state, whose omitted
/// rows' class relaxation the table carries).
pub(crate) fn fd_state_line(s: &FdStateRecord) -> String {
    let order: Vec<serde_json::Value> = s.order.iter().map(|a| json!([a.op, a.index])).collect();
    json!({
        "kind": "hint.fd_state",
        "id": s.id,
        "class": s.class,
        "t": s.t,
        "state_sha256": s.state_sha256,
        "n_gens": s.n_gens,
        "in_gens": s.in_gens,
        "unstable": s.unstable,
        "relaxed": s.relaxed,
        "root": num(s.root),
        "order": order,
        "prunes": s.prunes,
    })
    .to_string()
}

/// B2.3: one `hint.fd_lambda` record per Relu op of a frozen-dual state:
/// the slope of every unstable neuron (and its certified box, diagnostic).
pub(crate) fn fd_lambda_lines(s: &FdStateRecord) -> Vec<String> {
    s.lambda
        .iter()
        .map(|l| {
            let f = |v: &[f64]| -> Vec<serde_json::Value> { v.iter().map(|x| num(*x)).collect() };
            json!({
                "kind": "hint.fd_lambda",
                "id": s.id,
                "t": l.op,
                "idx": l.idx,
                "val": f(&l.val),
                "lo": f(&l.lo),
                "hi": f(&l.hi),
            })
            .to_string()
        })
        .collect()
}

fn alpha_line(a: &AlphaRecord) -> String {
    let val: Vec<serde_json::Value> = a.val.iter().map(|v| num(*v)).collect();
    json!({
        "kind": "hint.alpha",
        "id": a.id,
        "t": a.op,
        "idx": a.idx,
        "val": val,
        "start": "*",
    })
    .to_string()
}

fn sites_line(s: &SitesRecord) -> String {
    let tensors: Vec<serde_json::Value> = s
        .tensors
        .iter()
        .map(|t| json!({"id": t.id, "name": t.name, "size": t.size, "op": t.op}))
        .collect();
    json!({"kind": "sites", "tensors": tensors}).to_string()
}

/// The canonical clause atom of `Y_t - Y_j <= 0` (the checker's form:
/// coefficients sorted by output index, lowest-terms rationals).
#[must_use]
pub(crate) fn margin_clause(t: usize, j: usize) -> String {
    if t < j {
        format!("Y{t}:1,Y{j}:-1<=0")
    } else {
        format!("Y{j}:-1,Y{t}:1<=0")
    }
}

/// A general `clauses` record: region 0, one clause per `(id, atoms)`.
pub(crate) fn clauses_line(list: &[(usize, Vec<String>)]) -> String {
    let list: Vec<serde_json::Value> = list
        .iter()
        .map(|(id, atoms)| json!({"id": id, "atoms": atoms}))
        .collect();
    json!({"kind": "clauses", "region": 0, "list": list}).to_string()
}

/// The `clauses` record of a margin-row robustness property: region 0, one
/// single-atom clause per adversarial class, clause id = the class index.
pub(crate) fn margin_clauses_line(t: usize, adv: &[usize]) -> String {
    let list: Vec<serde_json::Value> = adv
        .iter()
        .map(|&j| json!({"id": j, "atoms": [margin_clause(t, j)]}))
        .collect();
    json!({"kind": "clauses", "region": 0, "list": list}).to_string()
}

/// The `verdict` record: the token's authority (`source`), EVERY clause id
/// of the bundle (`required`: a published `unsat` claims the whole
/// property), its minimum margin and the writer's node count.
fn verdict_line(v: &VerdictInfo, required: &[usize], nodes: u64) -> String {
    json!({
        "kind": "verdict",
        "source": v.source.token(),
        "required": required,
        "min_margin": num(v.min_margin),
        "nodes": nodes,
    })
    .to_string()
}

/// The `sites` record of a net: `(relu op index, ONNX name of the Relu node's
/// input[0], flat size)` per Relu op.
#[must_use]
pub(crate) fn sites_record(names: &[(usize, String, usize)]) -> SitesRecord {
    SitesRecord {
        tensors: names
            .iter()
            .map(|(op, name, size)| {
                let id = u32::try_from(*op).unwrap_or(u32::MAX);
                SiteTensorRecord {
                    id,
                    name: name.clone(),
                    size: *size,
                    op: id,
                }
            })
            .collect(),
    }
}

/// The process's wired bundle (Proof only), taken by [`after_publication_receipt`].
static WIRED: OnceLock<Mutex<Option<BundleWriter>>> = OnceLock::new();

static SITES_NOTED: AtomicBool = AtomicBool::new(false);
static CLAUSES_NOTED: AtomicBool = AtomicBool::new(false);

/// B2.2: the lane proof tapes the in-process verifier's lanes deposited
/// (Proof only), streamed by [`after_publication_receipt`].
static LANE_TAPES: Mutex<Option<LaneProofCapture>> = Mutex::new(None);

/// The wired bundle's inputs (Proof only), for the B2.2 translation.
static BUNDLE_INPUTS: OnceLock<(PathBuf, PathBuf)> = OnceLock::new();

/// B2.2: keep the lanes' proof tapes for [`after_publication_receipt`] (a no-op —
/// the capture is dropped — unless a Proof bundle is wired; several verifier
/// runs in one process accumulate).
pub(crate) fn stash_lane_proofs(capture: LaneProofCapture) {
    if capture.is_empty() || BUNDLE_INPUTS.get().is_none() {
        return;
    }
    let mut slot = LANE_TAPES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match slot.as_mut() {
        Some(held) => held.absorb(capture),
        None => *slot = Some(capture),
    }
}

/// B2.2: every ONNX `Relu` node's graph-node name (the ONNX node name, or its
/// first output when unnamed — the loader's naming) mapped to the ONNX name
/// of its `input[0]`, the checker's site tensor. Read with this module's own
/// decode of the file; an unreadable file maps nothing (every split literal
/// then streams as an `opaque` claim).
pub(crate) fn relu_site_map(onnx: &Path) -> HashMap<String, String> {
    use prost::Message;
    let mut out = HashMap::new();
    let Ok(bytes) = std::fs::read(onnx) else {
        return out;
    };
    let Ok(model) = ny_onnx::onnx_proto::ModelProto::decode(bytes.as_slice()) else {
        return out;
    };
    let Some(g) = model.graph.as_ref() else {
        return out;
    };
    for n in &g.node {
        if n.op_type != "Relu" || !(n.domain.is_empty() || n.domain == "ai.onnx") {
            continue;
        }
        let (Some(input), Some(output)) = (n.input.first(), n.output.first()) else {
            continue;
        };
        let name = if n.name.is_empty() {
            output.clone()
        } else {
            n.name.clone()
        };
        out.entry(name).or_insert_with(|| input.clone());
    }
    out
}

/// An exact rational `num/den` (`den > 0`, lowest terms) of a decimal literal
/// (`-12.5`, `3`, `1e-3`, `.5`); `None` on overflow or a non-decimal. This
/// module's own reader (the checker has its own).
pub(crate) fn decimal_rational(text: &str) -> Option<(i128, i128)> {
    let s = text.trim();
    let (neg, body) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    let (mant, exp) = match body.find(['e', 'E']) {
        Some(p) => (&body[..p], body[p + 1..].parse::<i32>().ok()?),
        None => (body, 0),
    };
    let (ip, fp) = match mant.find('.') {
        Some(p) => (&mant[..p], &mant[p + 1..]),
        None => (mant, ""),
    };
    if (ip.is_empty() && fp.is_empty())
        || !ip.bytes().all(|b| b.is_ascii_digit())
        || !fp.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let mut num: i128 = 0;
    for b in ip.bytes().chain(fp.bytes()) {
        num = num.checked_mul(10)?.checked_add(i128::from(b - b'0'))?;
    }
    let scale = exp.checked_sub(i32::try_from(fp.len()).ok()?)?;
    let mut den: i128 = 1;
    if scale >= 0 {
        for _ in 0..scale {
            num = num.checked_mul(10)?;
        }
    } else {
        for _ in 0..(-scale) {
            den = den.checked_mul(10)?;
        }
    }
    if neg {
        num = num.checked_neg()?;
    }
    Some(reduce(num, den))
}

fn reduce(num: i128, den: i128) -> (i128, i128) {
    let (mut a, mut b) = (num.unsigned_abs(), den.unsigned_abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    let g = i128::try_from(a.max(1)).unwrap_or(1);
    (num / g, den / g)
}

fn rational_add(a: (i128, i128), b: (i128, i128)) -> Option<(i128, i128)> {
    let num = a.0.checked_mul(b.1)?.checked_add(b.0.checked_mul(a.1)?)?;
    let den = a.1.checked_mul(b.1)?;
    Some(reduce(num, den))
}

fn rational_text((num, den): (i128, i128)) -> String {
    if den == 1 {
        format!("{num}")
    } else {
        format!("{num}/{den}")
    }
}

/// Every numeric token of a VNN-LIB text keyed by the bits of its f64 value:
/// an atom constant is rendered from the FILE's own decimal (the checker's
/// clause is the exact rational of that decimal), and only from the shortest
/// round-trip decimal of the parsed value when no token matches.
fn vnnlib_decimals(text: &str) -> HashMap<u64, String> {
    let mut out = HashMap::new();
    for line in text.lines() {
        let line = line.split(';').next().unwrap_or("");
        for tok in line.split(|c: char| c == '(' || c == ')' || c.is_whitespace()) {
            if tok.is_empty() || !tok.bytes().any(|b| b.is_ascii_digit()) {
                continue;
            }
            if let Ok(v) = tok.parse::<f64>() {
                out.entry(v.to_bits()).or_insert_with(|| tok.to_string());
            }
        }
    }
    out
}

/// The checker's canonical string of one atom (`Σ c_j Y_j <= d`, `<` when
/// strict; coefficients by index, exact lowest-terms rationals), or `None`
/// for an atom with no output term or a constant the rational reader cannot
/// hold.
pub(crate) fn canonical_atom(c: &OutputConstraint, dec: &HashMap<u64, String>) -> Option<String> {
    enum T {
        Y(usize),
        C(f64),
    }
    let (strict, pos, neg) = match *c {
        OutputConstraint::LessEq(i, j) => (false, T::Y(i), T::Y(j)),
        OutputConstraint::GreaterEq(i, j) => (false, T::Y(j), T::Y(i)),
        OutputConstraint::LessThan(i, j) => (true, T::Y(i), T::Y(j)),
        OutputConstraint::GreaterThan(i, j) => (true, T::Y(j), T::Y(i)),
        OutputConstraint::LessEqConst(i, v) => (false, T::Y(i), T::C(v)),
        OutputConstraint::GreaterEqConst(i, v) => (false, T::C(v), T::Y(i)),
        OutputConstraint::LessThanConst(i, v) => (true, T::Y(i), T::C(v)),
        OutputConstraint::GreaterThanConst(i, v) => (true, T::C(v), T::Y(i)),
        // A form this renderer does not know has no canonical string: the
        // lane stream is skipped rather than guessed.
        _ => return None,
    };
    let rat = |v: f64| -> Option<(i128, i128)> {
        if !v.is_finite() {
            return None;
        }
        match dec.get(&v.to_bits()) {
            Some(text) => decimal_rational(text),
            None => decimal_rational(&format!("{v:?}")),
        }
    };
    let mut coeffs: BTreeMap<usize, i128> = BTreeMap::new();
    let mut d: (i128, i128) = (0, 1);
    match pos {
        T::Y(j) => *coeffs.entry(j).or_insert(0) += 1,
        T::C(v) => {
            let (n, m) = rat(v)?;
            d = rational_add(d, (n.checked_neg()?, m))?;
        }
    }
    match neg {
        T::Y(j) => *coeffs.entry(j).or_insert(0) -= 1,
        T::C(v) => d = rational_add(d, rat(v)?)?,
    }
    let lhs: Vec<String> = coeffs
        .iter()
        .filter(|(_, c)| **c != 0)
        .map(|(j, c)| format!("Y{j}:{c}"))
        .collect();
    if lhs.is_empty() {
        return None;
    }
    Some(format!(
        "{}{}{}",
        lhs.join(","),
        if strict { "<" } else { "<=" },
        rational_text(d)
    ))
}

/// The instance's clauses (the same grouping the graph instance door reads:
/// one clause per disjunct, or one conjunction), each as canonical atoms.
/// `None` when an atom cannot be rendered.
pub(crate) fn canonical_clauses(spec: &VnnLibSpec, text: &str) -> Option<Vec<Vec<String>>> {
    let dec = vnnlib_decimals(text);
    let groups: Vec<&[OutputConstraint]> = if spec.is_disjunction {
        spec.output_constraint_clauses
            .iter()
            .map(Vec::as_slice)
            .collect()
    } else {
        vec![spec.output_constraints.as_slice()]
    };
    groups
        .into_iter()
        .map(|g| g.iter().map(|c| canonical_atom(c, &dec)).collect())
        .collect()
}

/// The instance's property and its clauses in the checker's canonical form
/// (id = position: the grouping the graph instance door reads), or why this
/// module cannot render them (a dual-network relation, per-clause input
/// boxes, an atom with no canonical string, ...). The ONE rendering both the
/// lane-tape translation and the writer's completion use.
pub(crate) fn instance_clauses_of(
    spec: &VnnLibSpec,
    text: &str,
) -> Result<(GraphInstanceProperty, Vec<(usize, Vec<String>)>), String> {
    let property = crate::commands::graph_cover_publish::graph_instance_property(spec)?;
    let clauses =
        canonical_clauses(spec, text).ok_or_else(|| "clause_canonical_form".to_string())?;
    if clauses.len() != property.clause_count() {
        return Err("clause_count".to_string());
    }
    Ok((property, clauses.into_iter().enumerate().collect()))
}

/// [`instance_clauses_of`] of a VNN-LIB file (the writer thread's fallback
/// `clauses`, computed at stream start, off the verdict path).
pub(crate) fn instance_clauses(vnnlib: &Path) -> Result<Vec<(usize, Vec<String>)>, String> {
    let text = std::fs::read_to_string(vnnlib).map_err(|_| "vnnlib_unreadable".to_string())?;
    let spec = ny_onnx::vnnlib::parse_vnnlib(&text).map_err(|_| "vnnlib_reparse".to_string())?;
    instance_clauses_of(&spec, &text).map(|(_, list)| list)
}

/// B2.2 after a published `unsat`: stream the lane tapes (sites, clauses,
/// nodes) into `writer`'s bundle unless the margin-row lane claimed it
/// (`margin_row_claimed`). Returns the clause ids the tapes were translated
/// against, or `None` when nothing was streamed. Prints `[proof-lanes] ...`.
fn stream_lane_tapes(
    writer: &BundleWriter,
    capture: Option<LaneProofCapture>,
    margin_row_claimed: bool,
    inputs: Option<&(PathBuf, PathBuf)>,
) -> Option<Vec<usize>> {
    let capture = capture?;
    if capture.is_empty() {
        return None;
    }
    if margin_row_claimed {
        eprintln!("[proof-lanes] skipped reason=margin_row_claimed_the_bundle");
        return None;
    }
    let (onnx, vnnlib) = inputs?;
    let refuse = |why: &str| {
        eprintln!("[proof-lanes] skipped reason={why}");
        None
    };
    let Ok(text) = std::fs::read_to_string(vnnlib) else {
        return refuse("vnnlib_unreadable");
    };
    let Ok(spec) = ny_onnx::vnnlib::parse_vnnlib(&text) else {
        return refuse("vnnlib_reparse");
    };
    let (property, list) = match instance_clauses_of(&spec, &text) {
        Ok(v) => v,
        Err(why) => return refuse(&why),
    };
    let inst = LaneProofInstance::new(&property, relu_site_map(onnx));
    writer.clauses_list(&list);
    let summary = ny_propagate::certify::stream_lane_proofs(&capture, &inst, &mut writer.sink());
    eprintln!(
        "[proof-lanes] tapes={} used={} nodes={} opaque={} fallback_clauses={} sites={} lossy={}",
        summary.tapes,
        summary.tapes_used,
        summary.nodes,
        summary.opaque,
        summary.fallback_clauses,
        summary.sites,
        summary.lossy
    );
    Some(inst.clause_ids())
}

/// The lane's Relu sites (read from `onnx` by the lane's own extractor,
/// `extract_twin_spec_named`), streamed ONCE per process into the wired
/// bundle. A no-op — not even a parse — unless the Proof profile switched
/// hint capture on.
pub(crate) fn note_sites_of(onnx: &Path) {
    if !ny_propagate::certify::hint_capture_enabled() || SITES_NOTED.swap(true, Ordering::AcqRel) {
        return;
    }
    if let Some((_, names)) = super::margin_row_bab::extract_twin_spec_named(onnx) {
        ny_propagate::certify::stream_sites(&sites_record(&names));
    }
}

/// The lane's spec clauses, written ONCE per process into the wired bundle
/// (a no-op unless a bundle is wired).
pub(crate) fn note_clauses(t: usize, adv: &[usize]) {
    let Some(slot) = WIRED.get() else {
        return;
    };
    if CLAUSES_NOTED.swap(true, Ordering::AcqRel) {
        return;
    }
    if let Ok(w) = slot.lock() {
        if let Some(writer) = w.as_ref() {
            writer.clauses(t, adv);
        }
    }
}

/// B2 (ROUTER_PROGRAM 10.12): the property's canonical `clauses` record,
/// written ONCE per process into the wired bundle, for a route whose own
/// whole-region claim names the instance's clauses BY POSITION (the
/// original-hull close: clause 0 of its single-clause property). It is the
/// writer's own rendering ([`instance_clauses`], the one the lane-tape
/// translation and the close's completion share), and it marks the bundle
/// claimed, so no other lane's tape is mixed into the route's stream.
///
/// Returns whether the route may stream its claim: `false` only when another
/// stream already wrote its clauses (the margin-row lane's class-id clauses),
/// where a positional id could name a foreign clause; the route then streams
/// nothing and its verdict's cover gap stays the checker's to report. With no
/// wired writer (Competition's empty slot, a test binary) it writes nothing
/// and returns `true`.
#[cfg_attr(
    not(feature = "mip"),
    expect(
        dead_code,
        reason = "instance-clause streaming is used by the MIP hull route"
    )
)]
pub(crate) fn note_instance_clauses() -> bool {
    let Some(slot) = WIRED.get() else {
        return true;
    };
    let Ok(w) = slot.lock() else {
        return true;
    };
    let Some(writer) = w.as_ref() else {
        return true;
    };
    if CLAUSES_NOTED.swap(true, Ordering::AcqRel) {
        return false;
    }
    writer.request_instance_clauses();
    true
}

/// Wire the process's proof sink for the scored entry point: ONCE per process
/// (a later call wires nothing and reports `already_wired`), before any lane
/// runs. Prints `[proof-sink] sink=<null|bundle|already_wired> ...` and
/// returns the sink kind for the flight record.
pub(crate) fn wire_scored_run(
    profile: ResolvedProofProfile,
    results_file: &Path,
    onnx: &Path,
    vnnlib: &Path,
) -> &'static str {
    let mut kind = "already_wired";
    WIRED.get_or_init(|| {
        let (writer, wired) = match SinkPlan::for_profile(profile.mode) {
            SinkPlan::Null => {
                install_proof_sink(Box::new(NullSink));
                (None, "null")
            }
            SinkPlan::Bundle => {
                let header = Header {
                    profile,
                    onnx: onnx.to_path_buf(),
                    vnnlib: vnnlib.to_path_buf(),
                };
                match BundleWriter::create(results_file, header) {
                    Ok((writer, sink)) => {
                        let _ = BUNDLE_INPUTS.set((onnx.to_path_buf(), vnnlib.to_path_buf()));
                        if install_proof_sink(Box::new(sink)) {
                            // The ONE place hint capture is switched on
                            // (ratchet: `enable_hint_capture(` appears only
                            // in this file).
                            enable_hint_capture();
                            (Some(writer), "bundle")
                        } else {
                            let why = writer.finalise("none", None, FINALISE_WAIT);
                            eprintln!(
                                "[proof-artifact] artifact={} (a sink was already installed)",
                                why.render()
                            );
                            (None, "preinstalled")
                        }
                    }
                    Err(e) => {
                        eprintln!("[proof-artifact] artifact=none(bundle_unwritable: {e})");
                        install_proof_sink(Box::new(NullSink));
                        (None, "null")
                    }
                }
            }
        };
        kind = wired;
        Mutex::new(writer)
    });
    let target = if kind == "bundle" {
        partial_path(results_file).display().to_string()
    } else {
        "-".to_string()
    };
    eprintln!(
        "[proof-sink] sink={kind} mode={} target={target}",
        profile.mode.token()
    );
    kind
}

/// Select the actual Proof writer before copying any class summary. A
/// Competition/NullSink call performs no summary allocation. Genuine retained
/// sources prepay the verdict buffers and stay owned through queued writing
/// and the final rename, including a detached timed-out writer.
pub(crate) fn after_publication_receipt(
    published: &str,
    receipt: Option<&super::unsat_receipt::UnsatReceipt>,
) {
    let Some(slot) = WIRED.get() else {
        return;
    };
    let Some(writer) = slot.lock().ok().and_then(|mut w| w.take()) else {
        return;
    };
    let owner = receipt
        .filter(|receipt| receipt.has_retained_owner())
        .map(|receipt| retained_bundle_owner(&writer, receipt))
        .transpose();
    let owner = match owner {
        Ok(owner) => owner,
        Err(error) => {
            // End only the partial artifact; an unaffordable/late export must
            // not create a final bundle or alter the already published result.
            let _ = writer.finalise("artifact_refused", None, FINALISE_WAIT);
            eprintln!("[proof-artifact] artifact=none(retained_owner_refused: {error})");
            return;
        }
    };
    let summary = receipt.map(|receipt| receipt.published_summary());
    finish_publication(writer, published, summary, owner);
}

fn retained_bundle_owner(
    writer: &BundleWriter,
    receipt: &super::unsat_receipt::UnsatReceipt,
) -> anyhow::Result<Arc<RetainedBundleOwner>> {
    let replay_quote = receipt
        .original_prepared_authority()
        .as_ref()
        .map(prepared_replay::quote)
        .transpose()?;
    let classes = receipt
        .original_prepared_authority()
        .map_or(1, |authority| authority.classes().len());
    let bytes = classes
        .checked_mul(512)
        .and_then(|n| {
            writer
                .partial
                .capacity()
                .checked_add(writer.final_path.capacity())
                .and_then(|paths| paths.checked_mul(8))
                .and_then(|paths| n.checked_add(paths))
        })
        .and_then(|n| n.checked_add(32768))
        .and_then(|n| n.checked_add(replay_quote.as_ref().map_or(0, |quote| quote.bytes)))
        .ok_or_else(|| anyhow::anyhow!("retained bundle publication capacity overflow"))?;
    let storage = receipt
        .reserve_owner_publication(bytes, None)?
        .ok_or_else(|| anyhow::anyhow!("retained bundle lost its genuine owner claim"))?;
    Ok(Arc::new(RetainedBundleOwner {
        receipt: receipt.clone(),
        _storage: storage,
        replay_quote,
    }))
}

/// The process-global close: a genuine prepared export first (its own
/// stream; the lane tapes are then not streamed), then [`close_bundle_owned`] with
/// the process's lane tapes and bundle inputs, the retained owner held
/// through the writer's queue and the final rename. Prints
/// `[proof-artifact] artifact=...`.
fn finish_publication(
    writer: BundleWriter,
    published: &str,
    unsat: Option<(Vec<usize>, f64, PublishedUnsatSource)>,
    owner: Option<Arc<RetainedBundleOwner>>,
) {
    let replay = owner
        .as_ref()
        .is_some_and(|owner| owner.replay_quote.is_some());
    if let Some(held) = owner.as_ref().filter(|owner| owner.replay_quote.is_some()) {
        let exported = prepared_replay::build(held.clone()).and_then(|payload| {
            writer
                .tx
                .send(Line::PreparedReplay(payload))
                .map_err(|_| anyhow::anyhow!("prepared replay writer is closed"))
        });
        if let Err(error) = exported {
            // Proof export failure cannot create a final artifact. The result
            // was already published; its verdict is unchanged, and this
            // partial stream remains unavailable for independent banking.
            let _ = writer.finalise_owned("artifact_refused", None, owner, FINALISE_WAIT);
            eprintln!("[proof-artifact] artifact=none(prepared_replay_refused: {error})");
            return;
        }
    }
    // B2.2: the lanes' proof tapes are the stream only when no prepared
    // export is (a genuine export is the published token's whole cover).
    let tapes = if replay {
        None
    } else {
        LANE_TAPES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    };
    let margin_row_claimed =
        CLAUSES_NOTED.load(Ordering::Acquire) || SITES_NOTED.load(Ordering::Acquire);
    let artifact = close_bundle_owned(
        writer,
        published,
        unsat,
        tapes,
        margin_row_claimed,
        BUNDLE_INPUTS.get(),
        owner,
    );
    eprintln!("[proof-artifact] artifact={}", artifact.render());
}

/// The close of [`after_publication_receipt`] with every process-global input passed
/// in (the in-process lane harnesses drive it per lane): stream the lane
/// tapes unless the margin-row lane claimed the bundle, choose the claim
/// completion from the token's source and whether a tape stood behind it,
/// and finalise. A legacy token with no lane tape behind it is the lane's
/// word alone ([`Completion::WholeRegion`]); every other stream is the
/// token's own proof ([`Completion::OwnStream`]).
///
/// review(eng15): a TEST face since the merge — production closes through
/// `finish_publication` -> [`close_bundle_owned`] (it carries origin's
/// retained owner), so outside the test build this face had no caller.
#[cfg(test)]
pub(crate) fn close_bundle(
    writer: BundleWriter,
    published: &str,
    unsat: Option<(Vec<usize>, f64, PublishedUnsatSource)>,
    tapes: Option<LaneProofCapture>,
    margin_row_claimed: bool,
    inputs: Option<&(PathBuf, PathBuf)>,
) -> Artifact {
    close_bundle_owned(
        writer,
        published,
        unsat,
        tapes,
        margin_row_claimed,
        inputs,
        None,
    )
}

/// The lane close behind the test face `close_bundle`, holding a retained
/// owner through the writer's queue and the final rename.
fn close_bundle_owned(
    writer: BundleWriter,
    published: &str,
    unsat: Option<(Vec<usize>, f64, PublishedUnsatSource)>,
    tapes: Option<LaneProofCapture>,
    margin_row_claimed: bool,
    inputs: Option<&(PathBuf, PathBuf)>,
    owner: Option<Arc<RetainedBundleOwner>>,
) -> Artifact {
    let mut verdict = None;
    if let Some((mut classes, min_margin, source)) = unsat {
        // B2.2: the lanes' proof tapes, when the margin-row lane did not
        // claim the bundle. A published `unsat` claims EVERY clause of the
        // instance, whatever door its token used (the writer's `required`).
        let tape_ids = stream_lane_tapes(&writer, tapes, margin_row_claimed, inputs);
        let completion = match (source.legacy(), &tape_ids) {
            (Some(_), None) => Completion::WholeRegion,
            _ => Completion::OwnStream,
        };
        if let Some(ids) = tape_ids {
            if classes.is_empty() {
                classes = ids;
            }
        }
        ny_propagate::certify::finish_stream(classes, min_margin);
        verdict = Some(VerdictInfo {
            source,
            min_margin,
            completion,
        });
    }
    writer.finalise_owned(published, verdict, owner, FINALISE_WAIT)
}

/// Test-only: switch hint capture on in THIS process (the test binary), the
/// way the Proof arm of [`wire_scored_run`] does, so an in-process harness can
/// produce a real bundle.
#[cfg(test)]
pub(crate) fn enable_hint_capture_for_test() {
    enable_hint_capture();
}

#[cfg(test)]
mod tests {
    use super::{
        bundle_path, margin_clause, margin_clauses_line, node_line, partial_path, path_tokens,
        sites_record, Artifact, BundleWriter, Completion, Header, Schema, SinkPlan, VerdictInfo,
        FINALISE_WAIT,
    };
    use ny_core::proof_profile::{EntryPoint, ProofProfile, ResolvedProofProfile};
    use ny_propagate::certify::{
        AbsLit, AbsSite, CoverLiteral, CoverSite, LeafClose, LeafPath, LeafRecord,
        LegacyUnsatAuthority, NodeClose, NodeRecord, ProofSink, UnsatSource,
    };
    use sha2::{Digest, Sha256};

    fn leaf(lits: Vec<CoverLiteral>, margin: f64, close: LeafClose) -> LeafRecord {
        LeafRecord {
            path: LeafPath::from_literals(lits).expect("canonical path"),
            margin,
            close,
        }
    }

    fn node(
        id: u64,
        lits: &[(u32, u32, bool)],
        close: NodeClose,
        dis: Option<Vec<usize>>,
    ) -> NodeRecord {
        NodeRecord {
            id,
            region: 0,
            lits: lits
                .iter()
                .map(|&(op, index, positive)| AbsLit {
                    site: AbsSite { op, index },
                    positive,
                })
                .collect(),
            xlits: Vec::new(),
            close,
            src: "tree",
            dis,
            beta: vec![0.5; lits.len()],
            alpha: None,
            lift: false,
            margin: 0.25,
            empty_site: None,
            fd: None,
        }
    }

    fn lines(path: &std::path::Path) -> Vec<serde_json::Value> {
        std::fs::read_to_string(path)
            .expect("bundle readable")
            .lines()
            .map(|l| serde_json::from_str(l).expect("every bundle line is JSON"))
            .collect()
    }

    /// ENGINE_PROGRAM §1.7: `Competition` wires the null sink and `Proof`
    /// the bundle, for either provenance (entry or `--proof-profile`).
    #[test]
    fn only_the_proof_profile_wires_the_bundle_sink() {
        for entry in [EntryPoint::VnncompV1, EntryPoint::Interactive] {
            for (mode, plan) in [
                (ProofProfile::Proof, SinkPlan::Bundle),
                (ProofProfile::Competition, SinkPlan::Null),
            ] {
                let resolved = ResolvedProofProfile::resolve(entry, Some(mode));
                assert_eq!(SinkPlan::for_profile(resolved.mode), plan);
            }
        }
        assert_eq!(
            SinkPlan::for_profile(ResolvedProofProfile::resolve(EntryPoint::VnncompV1, None).mode),
            SinkPlan::Null,
            "the scored entry point writes nothing by default"
        );
    }

    /// A 1-input, 3-output robustness property of true class 0 (clauses
    /// `Y0 - Y1 <= 0`, `Y0 - Y2 <= 0`): the instance the margin-row clauses
    /// `(t = 0, adv = [1, 2])` name, and one this module renders (schema v2).
    const PROP3: &str = "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n\
        (declare-const Y_1 Real)\n(declare-const Y_2 Real)\n\
        (assert (>= X_0 0))\n(assert (<= X_0 1))\n\
        (assert (or (and (>= Y_1 Y_0)) (and (>= Y_2 Y_0))))\n";

    fn header_for(dir: &std::path::Path) -> Header {
        header_with(dir, PROP3)
    }

    fn header_with(dir: &std::path::Path, vnnlib_text: &str) -> Header {
        let onnx = dir.join("net.onnx");
        let vnnlib = dir.join(format!("prop{}.vnnlib", vnnlib_text.len()));
        std::fs::write(&onnx, b"onnx-bytes").expect("onnx");
        std::fs::write(&vnnlib, vnnlib_text).expect("vnnlib");
        Header {
            profile: ResolvedProofProfile::resolve(
                EntryPoint::VnncompV1,
                Some(ProofProfile::Proof),
            ),
            onnx,
            vnnlib,
        }
    }

    /// The v2 bundle streams off-thread and is finalised only AFTER
    /// publication and only for a published `unsat`: header (schema v2,
    /// build, profile, lane, identity spaces, input digests), the sites and
    /// clauses, one `node` line per streamed node (positional leaves write
    /// nothing), the `verdict` of the published token with the writer's own
    /// node count, the `published` trailer and the `end` record; the
    /// `.partial` is renamed away. A non-`unsat` publication leaves the
    /// `.partial` and no final bundle.
    #[test]
    fn proof_bundle_streams_leaves_and_finalises_only_after_an_unsat() {
        let dir = tempfile::tempdir().expect("tempdir");
        let results = dir.path().join("unsat.txt");
        let (writer, mut sink) =
            BundleWriter::create(&results, header_for(dir.path())).expect("create bundle");
        assert!(
            partial_path(&results).is_file(),
            "the stream starts as .partial"
        );
        sink.sites(&sites_record(&[(1, "z".into(), 4), (7, "y".into(), 3)]));
        writer.clauses(0, &[1, 2]);
        let trunk = |layer, pos, positive| CoverLiteral {
            site: CoverSite::Trunk { layer, pos },
            positive,
        };
        sink.leaf(&leaf(vec![trunk(0, 3, true)], 0.25, LeafClose::Bound));
        sink.node(&node(
            0,
            &[(1, 3, true)],
            NodeClose::Bound,
            Some(vec![1, 2]),
        ));
        sink.node(&node(
            1,
            &[(1, 3, false), (7, 2, false)],
            NodeClose::Empty,
            None,
        ));
        let artifact = writer.finalise(
            "unsat",
            Some(VerdictInfo {
                source: UnsatSource::Authorized.into(),
                min_margin: 1e-3,
                completion: Completion::OwnStream,
            }),
            FINALISE_WAIT,
        );
        let final_path = bundle_path(&results);
        assert_eq!(
            artifact,
            Artifact::Written {
                path: final_path.clone(),
                nodes: 2,
                claims: 0,
                schema: Schema::V2,
            }
        );
        assert!(artifact
            .render()
            .contains(&format!("schema={}", super::BUNDLE_SCHEMA)));
        assert!(!partial_path(&results).exists(), "the .partial was renamed");
        let body = lines(&final_path);
        assert_eq!(
            body.len(),
            8,
            "header, sites, clauses, 2 nodes, verdict, published, end: {body:?}"
        );
        assert_eq!(body[0]["kind"], "header");
        assert_eq!(body[0]["schema"], super::BUNDLE_SCHEMA);
        assert_eq!(body[0]["profile"], "proof");
        assert_eq!(body[0]["profile_source"], "cli");
        assert_eq!(body[0]["lane"], "vnncomp_v1");
        assert_eq!(body[0]["site_space"], super::SITE_SPACE);
        assert_eq!(body[0]["input_space"], super::INPUT_SPACE);
        assert_eq!(body[0]["onnx"]["bytes"], 10);
        assert_eq!(body[0]["onnx"]["sha256"].as_str().map(str::len), Some(64));
        assert_eq!(body[1]["kind"], "sites");
        assert_eq!(body[1]["tensors"][1]["name"], "y");
        assert_eq!(body[2]["kind"], "clauses");
        assert_eq!(body[2]["list"][0]["atoms"][0], "Y0:1,Y1:-1<=0");
        assert_eq!(body[3]["kind"], "node");
        assert_eq!(body[3]["lits"], serde_json::json!([[1, 3, 1]]));
        assert_eq!(body[3]["dis"], serde_json::json!([1, 2]));
        assert_eq!(body[4]["close"], "empty");
        assert_eq!(body[4]["dis"], "*");
        assert_eq!(body[5]["kind"], "verdict");
        assert_eq!(body[5]["source"], "authorized");
        assert_eq!(body[5]["required"], serde_json::json!([1, 2]));
        assert_eq!(body[5]["nodes"], 2);
        assert_eq!(
            body[6],
            serde_json::json!({"kind": "published", "result": "unsat"})
        );
        assert_eq!(body[7]["kind"], "end");
        assert_eq!(body[7]["lines"], 8);

        // A published `sat` has no proof: the stream stays `.partial`.
        let results = dir.path().join("sat.txt");
        let (writer, mut sink) =
            BundleWriter::create(&results, header_for(dir.path())).expect("create bundle");
        sink.node(&node(0, &[(1, 0, true)], NodeClose::Bound, Some(vec![1])));
        let artifact = writer.finalise("sat", None, FINALISE_WAIT);
        assert_eq!(artifact, Artifact::None("verdict=sat".to_string()));
        assert!(!bundle_path(&results).exists());
        let body = lines(&partial_path(&results));
        assert_eq!(body[body.len() - 2]["result"], "sat");
        assert_eq!(
            body.iter().filter(|l| l["kind"] == "verdict").count(),
            0,
            "a non-unsat close writes no verdict and no completion: {body:?}"
        );

        // An unreadable input is recorded (v2.1 `unavailable`), never fatal.
        let results = dir.path().join("legacy.txt");
        let mut h = header_for(dir.path());
        h.vnnlib = dir.path().join("missing.vnnlib");
        let (writer, _sink) = BundleWriter::create(&results, h).expect("create bundle");
        let artifact = writer.finalise(
            "unsat",
            Some(legacy_verdict(LegacyUnsatAuthority::BetaCrownVerdict)),
            FINALISE_WAIT,
        );
        assert!(
            matches!(
                artifact,
                Artifact::Written {
                    schema: Schema::V2_1,
                    claims: 1,
                    ..
                }
            ),
            "{artifact:?}"
        );
        let body = lines(&bundle_path(&results));
        assert!(
            body[0]["vnnlib"]["unavailable"].is_string() && body[0]["vnnlib"]["sha256"].is_null(),
            "an unreadable input is recorded, never fatal: {:?}",
            body[0]
        );
    }

    fn legacy_verdict(authority: LegacyUnsatAuthority) -> VerdictInfo {
        VerdictInfo {
            source: UnsatSource::LegacyUncovered(authority).into(),
            min_margin: f64::NAN,
            completion: Completion::WholeRegion,
        }
    }

    fn of_kind<'a>(body: &'a [serde_json::Value], kind: &str) -> Vec<&'a serde_json::Value> {
        body.iter().filter(|l| l["kind"] == kind).collect()
    }

    /// LEGACY WELL-FORMEDNESS (certified-moat review, 2026-10-08): a legacy
    /// door's `unsat` with NO stream (the sequential engine, MIP, a root-only
    /// graph fast path, CZ hybrid, cctsdb, TLL, ...) used to be finalised as
    /// `header, verdict(required=[]), published, end`, which the checker
    /// refuses `malformed` (`missing_sites`, exit 30). The writer now
    /// completes it: an empty `sites`, the instance's canonical `clauses`
    /// (rendered from the VNN-LIB on the writer thread; ids by position), ONE
    /// whole-region `opaque` claim per clause naming the authority
    /// (`legacy:<authority>`), and a `verdict` requiring EVERY clause.
    #[test]
    fn proof_bundle_legacy_unsat_without_a_stream_is_completed_well_formed() {
        let dir = tempfile::tempdir().expect("tempdir");
        for authority in LegacyUnsatAuthority::ALL {
            let results = dir.path().join(format!("{}.txt", authority.token()));
            let (writer, _sink) =
                BundleWriter::create(&results, header_for(dir.path())).expect("create bundle");
            let artifact = writer.finalise("unsat", Some(legacy_verdict(authority)), FINALISE_WAIT);
            assert_eq!(
                artifact,
                Artifact::Written {
                    path: bundle_path(&results),
                    nodes: 2,
                    claims: 2,
                    schema: Schema::V2,
                },
                "{}",
                authority.token()
            );
            assert!(
                artifact.render().ends_with(" claims=2"),
                "{}",
                artifact.render()
            );
            let body = lines(&bundle_path(&results));
            let kinds: Vec<&str> = body
                .iter()
                .map(|l| l["kind"].as_str().unwrap_or("?"))
                .collect();
            assert_eq!(
                kinds,
                [
                    "header",
                    "sites",
                    "clauses",
                    "node",
                    "node",
                    "verdict",
                    "published",
                    "end"
                ],
                "{body:?}"
            );
            assert_eq!(body[0]["schema"], super::BUNDLE_SCHEMA);
            assert_eq!(body[1]["tensors"], serde_json::json!([]));
            assert_eq!(
                body[2]["list"],
                serde_json::json!([
                    {"id": 0, "atoms": [margin_clause(0, 1)]},
                    {"id": 1, "atoms": [margin_clause(0, 2)]},
                ])
            );
            let route = format!("legacy:{}", authority.token());
            for (k, n) in [&body[3], &body[4]].into_iter().enumerate() {
                assert_eq!(n["close"], "opaque", "{n}");
                assert_eq!(n["src"], route.as_str(), "{n}");
                assert_eq!(n["lits"], serde_json::json!([]), "{n}");
                assert_eq!(n["dis"], serde_json::json!([k]), "{n}");
            }
            assert_ne!(body[3]["id"], body[4]["id"]);
            assert_eq!(body[5]["source"], authority.token());
            assert_eq!(body[5]["required"], serde_json::json!([0, 1]));
            assert_eq!(body[5]["nodes"], 2);
        }
    }

    /// A property this module cannot render in the checker's canonical form
    /// (here: per-clause input boxes, which the graph instance door refuses)
    /// is the explicit `ny-proof-bundle/v2.1` form: the header names v2.1,
    /// the `clauses` record is ONE clause `{"id": 0, "unavailable": <why>}`
    /// standing for the whole property, one whole-region claim names it, and
    /// the verdict requires it. A header input that is not one regular file
    /// (a VNN-LIB 2.0 dual-network ONNX field) is `{"path", "unavailable"}`
    /// and forces the same clause form.
    #[test]
    fn proof_bundle_unrenderable_property_uses_the_v2_1_unavailable_form() {
        let dir = tempfile::tempdir().expect("tempdir");
        let boxes = "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n\
            (assert (or (and (>= X_0 0) (<= X_0 1) (<= Y_0 -5)) \
            (and (>= X_0 2) (<= X_0 3) (<= Y_0 -5))))\n";
        let mut tuple = header_with(dir.path(), PROP3);
        tuple.onnx = std::path::PathBuf::from("[('f', 'a.onnx'), ('g', 'b.onnx')]");
        for (tag, header, why) in [
            (
                "boxes",
                header_with(dir.path(), boxes),
                "per_clause_input_boxes",
            ),
            ("tuple", tuple, "inputs_unavailable"),
        ] {
            let results = dir.path().join(format!("{tag}.txt"));
            let (writer, _sink) = BundleWriter::create(&results, header).expect("create bundle");
            let artifact = writer.finalise(
                "unsat",
                Some(legacy_verdict(LegacyUnsatAuthority::RelationalImplication)),
                FINALISE_WAIT,
            );
            assert!(
                matches!(
                    artifact,
                    Artifact::Written {
                        nodes: 1,
                        claims: 1,
                        schema: Schema::V2_1,
                        ..
                    }
                ),
                "{tag}: {artifact:?}"
            );
            assert!(
                artifact.render().contains("schema=ny-proof-bundle/v2.1 "),
                "{}",
                artifact.render()
            );
            let body = lines(&bundle_path(&results));
            assert_eq!(body[0]["schema"], super::BUNDLE_SCHEMA_V2_1, "{tag}");
            let clauses = of_kind(&body, "clauses");
            assert_eq!(clauses.len(), 1, "{tag}");
            assert_eq!(
                clauses[0]["list"],
                serde_json::json!([{"id": 0, "unavailable": why}]),
                "{tag}"
            );
            let nodes = of_kind(&body, "node");
            assert_eq!(nodes.len(), 1, "{tag}");
            assert_eq!(nodes[0]["src"], "legacy:relational_implication");
            assert_eq!(nodes[0]["dis"], serde_json::json!([0]));
            assert_eq!(
                of_kind(&body, "verdict")[0]["required"],
                serde_json::json!([0])
            );
            if tag == "tuple" {
                assert!(body[0]["onnx"]["unavailable"].is_string(), "{:?}", body[0]);
                assert!(body[0]["onnx"].get("sha256").is_none(), "{:?}", body[0]);
                assert_eq!(body[0]["vnnlib"]["sha256"].as_str().map(str::len), Some(64));
            }
        }
    }

    /// review(legacy-bundle): the header names v2.1 only when the bundle
    /// WRITES an `unavailable` form. A property this module cannot render
    /// (per-clause input boxes) under a lane that writes its OWN `clauses`
    /// (the margin-row lane's `note_clauses`) never uses the fallback, so its
    /// bundle is an ordinary checkable v2 one — certlib refuses to ACCEPT any
    /// v2.1 header, so the builder's eager stream-start decision (v2.1
    /// whenever the fallback failed) cost such a stream its certificate. The
    /// held lines (here `sites`, sent before the `clauses`) still follow the
    /// header in order, and the `end` record's chain covers them; a stream
    /// that closes WITHOUT a lane's clauses still writes the unavailable
    /// clause under a v2.1 header.
    #[test]
    fn proof_bundle_schema_is_v2_1_only_when_an_unavailable_form_is_written() {
        let dir = tempfile::tempdir().expect("tempdir");
        let boxes = "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n\
            (declare-const Y_1 Real)\n(declare-const Y_2 Real)\n\
            (assert (or (and (>= X_0 0) (<= X_0 1) (>= Y_1 Y_0)) \
            (and (>= X_0 2) (<= X_0 3) (>= Y_2 Y_0))))\n";
        let results = dir.path().join("lane_clauses.txt");
        let (writer, mut sink) =
            BundleWriter::create(&results, header_with(dir.path(), boxes)).expect("create bundle");
        sink.sites(&sites_record(&[(1, "z".into(), 4)]));
        writer.clauses(0, &[1, 2]);
        sink.node(&node(0, &[], NodeClose::Root, Some(vec![1, 2])));
        let artifact = writer.finalise(
            "unsat",
            Some(VerdictInfo {
                source: UnsatSource::Authorized.into(),
                min_margin: 0.5,
                completion: Completion::OwnStream,
            }),
            FINALISE_WAIT,
        );
        assert!(
            matches!(
                artifact,
                Artifact::Written {
                    nodes: 1,
                    claims: 0,
                    schema: Schema::V2,
                    ..
                }
            ),
            "a lane's own clauses keep the bundle v2: {artifact:?}"
        );
        let path = bundle_path(&results);
        let body = lines(&path);
        let kinds: Vec<&str> = body
            .iter()
            .map(|l| l["kind"].as_str().unwrap_or("?"))
            .collect();
        assert_eq!(
            kinds,
            [
                "header",
                "sites",
                "clauses",
                "node",
                "verdict",
                "published",
                "end"
            ],
            "{body:?}"
        );
        assert_eq!(body[0]["schema"], super::BUNDLE_SCHEMA);
        assert!(
            !std::fs::read_to_string(&path)
                .expect("bundle")
                .contains("\"unavailable\""),
            "no unavailable form in a v2 bundle"
        );
        let bytes = std::fs::read(&path).expect("bundle");
        let cut = bytes[..bytes.len() - 1]
            .iter()
            .rposition(|&b| b == b'\n')
            .map_or(0, |i| i + 1);
        assert_eq!(
            body.last().expect("end")["sha256"].as_str(),
            Some(super::hex(&Sha256::digest(&bytes[..cut])).as_str()),
            "the end record's chain covers the held lines in file order"
        );
        assert_eq!(body.last().expect("end")["lines"], body.len());
        // The same unrenderable property with NO lane clauses: v2.1.
        let results = dir.path().join("no_lane_clauses.txt");
        let (writer, mut sink) =
            BundleWriter::create(&results, header_with(dir.path(), boxes)).expect("create bundle");
        sink.sites(&sites_record(&[(1, "z".into(), 4)]));
        let artifact = writer.finalise(
            "unsat",
            Some(legacy_verdict(LegacyUnsatAuthority::BetaCrownVerdict)),
            FINALISE_WAIT,
        );
        assert!(
            matches!(
                artifact,
                Artifact::Written {
                    claims: 1,
                    schema: Schema::V2_1,
                    ..
                }
            ),
            "{artifact:?}"
        );
        let body = lines(&bundle_path(&results));
        assert_eq!(body[0]["schema"], super::BUNDLE_SCHEMA_V2_1);
        assert_eq!(body[1]["kind"], "sites", "the held line follows the header");
        assert_eq!(
            of_kind(&body, "clauses")[0]["list"][0]["unavailable"],
            "per_clause_input_boxes"
        );
    }

    /// The partial-record shape (`challenging_certified_training_2026`
    /// idx5683: the margin-row lane named the sites and clauses, streamed
    /// nothing whole-region for a class, and the internal verifier's legacy
    /// door published): a LEGACY token with no lane tape behind it gets a
    /// whole-region claim for every clause no WHOLE-REGION node claims — a
    /// partial subtree of another lane is not the verdict's proof — while a
    /// clause a whole-region node already claims (a root close) gets none.
    #[test]
    fn proof_bundle_legacy_completion_ignores_a_partial_stream() {
        let dir = tempfile::tempdir().expect("tempdir");
        let results = dir.path().join("partial.txt");
        let (writer, mut sink) =
            BundleWriter::create(&results, header_for(dir.path())).expect("create bundle");
        sink.sites(&sites_record(&[(1, "z".into(), 4)]));
        writer.clauses(0, &[1, 2]);
        // A subtree leaf of class 1 (one side of a split only) and a root
        // close of class 2.
        sink.node(&node(0, &[(1, 3, true)], NodeClose::Bound, Some(vec![1])));
        sink.node(&node(1, &[], NodeClose::Root, Some(vec![2])));
        let artifact = writer.finalise(
            "unsat",
            Some(legacy_verdict(LegacyUnsatAuthority::BetaCrownVerdict)),
            FINALISE_WAIT,
        );
        assert!(
            matches!(
                artifact,
                Artifact::Written {
                    nodes: 3,
                    claims: 1,
                    schema: Schema::V2,
                    ..
                }
            ),
            "{artifact:?}"
        );
        let body = lines(&bundle_path(&results));
        let nodes = of_kind(&body, "node");
        let added: Vec<&&serde_json::Value> = nodes
            .iter()
            .filter(|n| n["src"] == "legacy:beta_crown_verdict")
            .collect();
        assert_eq!(added.len(), 1, "{nodes:?}");
        assert_eq!(added[0]["dis"], serde_json::json!([1]));
        assert_eq!(added[0]["lits"], serde_json::json!([]));
        assert_eq!(
            of_kind(&body, "verdict")[0]["required"],
            serde_json::json!([1, 2])
        );
    }

    /// An AUTHORIZED token's stream is its own proof: the writer completes it
    /// only when it streamed NOTHING (an authorized route with no emitter,
    /// the isomorphic exact-emptiness door: one `unstreamed:authorized`
    /// claim per clause) and never papers over an own stream that dropped a
    /// class — that cover gap is the checker's to report (`invalid`), not a
    /// claim to make `unconfirmed`.
    #[test]
    fn proof_bundle_authorized_completion_claims_only_an_empty_stream() {
        let dir = tempfile::tempdir().expect("tempdir");
        let authorized = || {
            Some(VerdictInfo {
                source: UnsatSource::Authorized.into(),
                min_margin: 0.5,
                completion: Completion::OwnStream,
            })
        };
        // An own stream that names class 1 only (class 2 dropped).
        let results = dir.path().join("authorized.txt");
        let (writer, mut sink) =
            BundleWriter::create(&results, header_for(dir.path())).expect("create bundle");
        sink.sites(&sites_record(&[(1, "z".into(), 4)]));
        writer.clauses(0, &[1, 2]);
        sink.node(&node(0, &[(1, 3, true)], NodeClose::Bound, Some(vec![1])));
        let artifact = writer.finalise("unsat", authorized(), FINALISE_WAIT);
        assert!(
            matches!(
                artifact,
                Artifact::Written {
                    nodes: 1,
                    claims: 0,
                    ..
                }
            ),
            "{artifact:?}"
        );
        let body = lines(&bundle_path(&results));
        assert_eq!(of_kind(&body, "node").len(), 1, "{body:?}");
        assert_eq!(
            of_kind(&body, "verdict")[0]["required"],
            serde_json::json!([1, 2]),
            "the verdict still requires the dropped class: the checker sees the gap"
        );
        // An authorized route that streamed nothing at all.
        let results = dir.path().join("unstreamed.txt");
        let (writer, _sink) =
            BundleWriter::create(&results, header_for(dir.path())).expect("create bundle");
        let artifact = writer.finalise("unsat", authorized(), FINALISE_WAIT);
        assert!(
            matches!(
                artifact,
                Artifact::Written {
                    nodes: 2,
                    claims: 2,
                    ..
                }
            ),
            "{artifact:?}"
        );
        let body = lines(&bundle_path(&results));
        let nodes = of_kind(&body, "node");
        assert!(
            nodes
                .iter()
                .all(|n| n["src"] == "unstreamed:authorized" && n["lits"] == serde_json::json!([])),
            "{nodes:?}"
        );
        assert_eq!(of_kind(&body, "sites")[0]["tensors"], serde_json::json!([]));
    }

    /// B2 (ROUTER_PROGRAM 10.12): the original-hull close's bundle. The route
    /// asks for the instance's canonical clauses (the writer's own rendering,
    /// ids by position) and streams ONE whole-region `opaque` `original_hull`
    /// node over its one clause; the close adds only the empty `sites` record
    /// (no completion claim: the stream is the token's own), and the verdict
    /// names the hull's source and requires exactly that clause.
    #[cfg(feature = "mip")]
    #[test]
    fn proof_bundle_original_hull_claim_carries_the_instance_clauses_and_no_completion() {
        let dir = tempfile::tempdir().expect("tempdir");
        let one = "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n\
            (declare-const Y_1 Real)\n(assert (>= X_0 -1))\n(assert (<= X_0 1))\n\
            (assert (<= Y_0 Y_1))\n";
        let results = dir.path().join("hull.txt");
        let (writer, mut sink) =
            BundleWriter::create(&results, header_with(dir.path(), one)).expect("create bundle");
        writer.request_instance_clauses();
        let mut claim = node(0, &[], NodeClose::Opaque, Some(vec![0]));
        claim.src = "original_hull";
        claim.beta = Vec::new();
        sink.node(&claim);
        let artifact = writer.finalise(
            "unsat",
            Some(VerdictInfo {
                source: crate::commands::unsat_receipt::PublishedUnsatSource::OriginalHull,
                min_margin: 0.25,
                completion: Completion::OwnStream,
            }),
            FINALISE_WAIT,
        );
        assert!(
            matches!(
                artifact,
                Artifact::Written {
                    nodes: 1,
                    claims: 0,
                    schema: Schema::V2,
                    ..
                }
            ),
            "{artifact:?}"
        );
        let body = lines(&bundle_path(&results));
        let kinds: Vec<&str> = body
            .iter()
            .map(|l| l["kind"].as_str().unwrap_or("?"))
            .collect();
        assert_eq!(
            kinds,
            [
                "header",
                "clauses",
                "node",
                "sites",
                "verdict",
                "published",
                "end"
            ],
            "{body:?}"
        );
        assert_eq!(body[0]["schema"], super::BUNDLE_SCHEMA);
        let list = body[1]["list"].as_array().expect("clauses list");
        assert_eq!(list.len(), 1, "{list:?}");
        assert_eq!(list[0]["id"], 0);
        assert_eq!(body[2]["src"], "original_hull");
        assert_eq!(body[2]["close"], "opaque");
        assert_eq!(body[2]["lits"], serde_json::json!([]));
        assert_eq!(body[2]["dis"], serde_json::json!([0]));
        assert_eq!(body[4]["source"], "original_hull_replay");
        assert_eq!(body[4]["required"], serde_json::json!([0]));
    }

    /// A second `sites` or `clauses` record (which the checker refuses as a
    /// duplicate) is refused at the writer, and nothing queued after the close
    /// reaches the file: the close is one atomic message.
    #[test]
    fn proof_bundle_refuses_duplicates_and_writes_nothing_after_the_close() {
        let dir = tempfile::tempdir().expect("tempdir");
        let results = dir.path().join("dup.txt");
        let (writer, mut sink) =
            BundleWriter::create(&results, header_for(dir.path())).expect("create bundle");
        sink.sites(&sites_record(&[(1, "z".into(), 4)]));
        sink.sites(&sites_record(&[(1, "z".into(), 4)]));
        writer.clauses(0, &[1, 2]);
        writer.clauses_list(&[(0, vec![margin_clause(0, 1)])]);
        sink.node(&node(0, &[], NodeClose::Root, Some(vec![1, 2])));
        let mut late = writer.sink();
        let artifact = writer.finalise(
            "unsat",
            Some(legacy_verdict(LegacyUnsatAuthority::BetaCrownVerdict)),
            FINALISE_WAIT,
        );
        late.node(&node(9, &[], NodeClose::Root, Some(vec![1])));
        assert!(
            matches!(
                artifact,
                Artifact::Written {
                    nodes: 1,
                    claims: 0,
                    ..
                }
            ),
            "{artifact:?}"
        );
        let body = lines(&bundle_path(&results));
        assert_eq!(of_kind(&body, "sites").len(), 1);
        assert_eq!(of_kind(&body, "clauses").len(), 1);
        assert_eq!(of_kind(&body, "clauses")[0]["list"][0]["id"], 1);
        assert_eq!(of_kind(&body, "node").len(), 1);
        assert_eq!(body.last().map(|l| l["kind"].clone()), Some("end".into()));
    }

    /// The `end` record certifies the whole stream: its `lines` is the file's
    /// line count and its `sha256` the digest of every preceding byte, so a
    /// cut (or edited) bundle no longer matches it.
    #[test]
    fn proof_bundle_v2_end_record_detects_truncation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let results = dir.path().join("r.txt");
        let (writer, mut sink) =
            BundleWriter::create(&results, header_for(dir.path())).expect("create bundle");
        for k in 0..5 {
            sink.node(&node(
                k,
                &[(1, k as u32, k % 2 == 0)],
                NodeClose::Bound,
                Some(vec![1]),
            ));
        }
        let _ = writer.finalise(
            "unsat",
            Some(VerdictInfo {
                source: UnsatSource::Authorized.into(),
                min_margin: 0.5,
                completion: Completion::OwnStream,
            }),
            FINALISE_WAIT,
        );
        let bytes = std::fs::read(bundle_path(&results)).expect("bundle");
        let text = String::from_utf8(bytes.clone()).expect("utf8");
        let all: Vec<&str> = text.lines().collect();
        let end: serde_json::Value = serde_json::from_str(all[all.len() - 1]).expect("end json");
        assert_eq!(end["lines"].as_u64(), Some(all.len() as u64));
        let body_len = bytes.len() - all[all.len() - 1].len() - 1;
        let digest = format!("{:x}", Sha256::digest(&bytes[..body_len]));
        assert_eq!(end["sha256"].as_str(), Some(digest.as_str()));
        // Dropping a node line keeps every line well-formed but breaks both.
        let mut cut: Vec<&str> = all.clone();
        cut.remove(2);
        let recount = format!(
            "{:x}",
            Sha256::digest(cut[..cut.len() - 1].join("\n").as_bytes())
        );
        assert_ne!(end["sha256"].as_str(), Some(recount.as_str()));
        assert_ne!(end["lines"].as_u64(), Some(cut.len() as u64));
    }

    /// B2.3: a frozen-dual state streams as `hint.fd_state` (class, state
    /// sha256, sizes, root, the split order as `[site id, index]`, prune
    /// count) plus one `hint.fd_lambda` per Relu op, and a prune node carries
    /// its dual point as `hint.fd = {state, l0, l1}` (non-finite as `null`)
    /// with `beta: null`: the documented B2.3 schema the checker parses.
    #[test]
    fn proof_bundle_fd_state_and_prune_lines_are_the_documented_schema() {
        use ny_propagate::certify::{FdLambdaRecord, FdNodeHint, FdStateRecord};
        let st = FdStateRecord {
            id: 3,
            class: 2,
            t: 0,
            state_sha256: "ab".repeat(32),
            n_gens: 9,
            in_gens: 4,
            unstable: 5,
            relaxed: 5,
            root: -0.5,
            order: vec![AbsSite { op: 1, index: 7 }, AbsSite { op: 4, index: 2 }],
            lambda: vec![FdLambdaRecord {
                op: 1,
                idx: vec![7, 8],
                val: vec![0.25, 1.0],
                lo: vec![-1.0, -2.0],
                hi: vec![1.0, 0.5],
            }],
            prunes: 4,
        };
        let s: serde_json::Value =
            serde_json::from_str(&super::fd_state_line(&st)).expect("fd_state json");
        assert_eq!(s["kind"], "hint.fd_state");
        assert_eq!(s["id"], 3);
        assert_eq!(s["class"], 2);
        assert_eq!(s["order"], serde_json::json!([[1, 7], [4, 2]]));
        assert_eq!(s["prunes"], 4);
        assert_eq!(s["unstable"], 5);
        assert_eq!(s["relaxed"], 5);
        assert_eq!(s["state_sha256"].as_str().map(str::len), Some(64));
        let lam = super::fd_lambda_lines(&st);
        assert_eq!(lam.len(), 1);
        let l: serde_json::Value = serde_json::from_str(&lam[0]).expect("fd_lambda json");
        assert_eq!(l["kind"], "hint.fd_lambda");
        assert_eq!(l["id"], 3);
        assert_eq!(l["t"], 1);
        assert_eq!(l["idx"], serde_json::json!([7, 8]));
        assert_eq!(l["val"], serde_json::json!([0.25, 1.0]));
        let mut n = node(
            9,
            &[(1, 7, true), (4, 2, false)],
            NodeClose::Bound,
            Some(vec![2]),
        );
        n.beta = Vec::new();
        n.src = "fd_prune";
        n.fd = Some(FdNodeHint {
            state: 3,
            l0: vec![0.5, f64::NAN],
            l1: vec![0.0, 2.0],
        });
        let v: serde_json::Value = serde_json::from_str(&node_line(&n)).expect("node json");
        assert_eq!(v["src"], "fd_prune");
        assert_eq!(v["hint"]["beta"], serde_json::Value::Null);
        assert_eq!(
            v["hint"]["fd"],
            serde_json::json!({"state": 3, "l0": [0.5, null], "l1": [0.0, 2.0]})
        );
        // Every other node carries no `fd` hint at all.
        let plain: serde_json::Value = serde_json::from_str(&node_line(&node(
            1,
            &[(1, 0, true)],
            NodeClose::Bound,
            None,
        )))
        .expect("node json");
        assert!(plain["hint"].get("fd").is_none());
    }

    /// The node literals and the clause atoms are the documented schema:
    /// `[site id, flat index, side]`, `Yi:c,...<=d` sorted by index.
    #[test]
    fn proof_bundle_path_tokens_are_the_documented_schema() {
        let rec = leaf(
            vec![
                CoverLiteral {
                    site: CoverSite::Head { index: 7 },
                    positive: true,
                },
                CoverLiteral {
                    site: CoverSite::Trunk { layer: 2, pos: 11 },
                    positive: false,
                },
            ],
            0.1,
            LeafClose::Subtree,
        );
        assert_eq!(path_tokens(&rec), vec!["t2.11-", "h7+"]);
        assert_eq!(
            (path_tokens(&leaf(Vec::new(), 1.0, LeafClose::Bound))).len(),
            0
        );
        assert_eq!(LeafClose::Subtree.token(), "subtree");
        // #graph-bab-cover-ledger: the graph-lane sites.
        let graph = leaf(
            vec![
                CoverLiteral {
                    site: CoverSite::GraphPoint {
                        node: 4,
                        input: 1,
                        neuron: 2,
                        point_bits: 0.5f32.to_bits(),
                    },
                    positive: false,
                },
                CoverLiteral {
                    site: CoverSite::GraphRelu { node: 3, neuron: 9 },
                    positive: true,
                },
            ],
            0.2,
            LeafClose::Lane,
        );
        assert_eq!(path_tokens(&graph), vec!["g3.9+", "p4.1.2@3f000000-"]);
        assert_eq!(LeafClose::Lane.token(), "lane");
        let n: serde_json::Value = serde_json::from_str(&node_line(&node(
            4,
            &[(3, 9, false), (5, 1, true)],
            NodeClose::Retire,
            Some(vec![2]),
        )))
        .expect("node json");
        assert_eq!(n["lits"], serde_json::json!([[3, 9, 0], [5, 1, 1]]));
        assert_eq!(n["close"], "retire");
        assert_eq!(n["hint"]["beta"], serde_json::json!([0.5, 0.5]));
        assert_eq!(margin_clause(3, 7), "Y3:1,Y7:-1<=0");
        assert_eq!(margin_clause(7, 3), "Y3:-1,Y7:1<=0");
        let c: serde_json::Value =
            serde_json::from_str(&margin_clauses_line(2, &[0, 5])).expect("clauses json");
        assert_eq!(c["list"][0]["id"], 0);
        assert_eq!(c["list"][0]["atoms"][0], "Y0:-1,Y2:1<=0");
        assert_eq!(c["list"][1]["atoms"][0], "Y2:1,Y5:-1<=0");
    }
}
