// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The only door to a published `unsat` (#soundness-spine M2b).
//!
//! `VnncompResult::Unsat` carries an [`UnsatReceipt`], and an `UnsatReceipt`
//! can be built through these private doors:
//!
//! * [`UnsatReceipt::from_token`] — from an instance-scoped
//!   `ny_propagate::certify::UnsatToken` that `certify::authorize_unsat`
//!   minted from covers the facade checked: the margin-row lane's path-checked
//!   tree covers, the isomorphic lane's exact specification-emptiness
//!   certificate (`certify::authorize_scalar_emptiness`), and — for the
//!   in-process β-CROWN verifier's `verified` — the graph lane's completeness
//!   ledger (`certify::authorize_graph_instance`: every clause entailed by a
//!   gate-checked cover a ledgered driver proved; #graph-bab-cover-ledger);
//! * [`UnsatReceipt::legacy_uncovered`] — a LEGACY-UNCOVERED lane: its own
//!   machinery decides `unsat` but it produces no cover, so the token rests on
//!   the lane's word (`LegacyUnsatAuthority` documents, per lane, what
//!   certifies it today and why that is not yet a cover). Every production call
//!   site is frozen per file and per authority by
//!   `unsat_receipt_legacy_sites_are_frozen`, and their COUNT sits under
//!   [`LEGACY_UNCOVERED_CEILING_HISTORY`], which may only go DOWN
//!   (`unsat_receipt_legacy_uncovered_ceiling_only_goes_down`). Each use prints
//!   `[cover-gate] legacy-uncovered authority=<lane>`, so an uncovered `unsat`
//!   is countable in every run log;
//! * `UnsatReceipt::from_original_hull` — with `mip`, from an immutable
//!   original-model replay owner whose scope, deadline, cancellation and paid
//!   storage remain live through the actual writer; this is distinct from a
//!   `ny_propagate` token and cannot be reconstructed from verdict JSON;
//! * [`UnsatReceipt::from_original_prepared`] — from the actual original
//!   model/property source issuer and complete prepared-root producer. Its
//!   immutable source, ordered complete objective, original ledger and earliest
//!   deadline stay owned through publication; no supplied token or JSON can
//!   manufacture this authority;
//! * [`UnsatReceipt::for_test`] — `#[cfg(test)]` only.
//!
//! The receipt compares equal to every other receipt: two `unsat` verdicts are
//! the same verdict whatever lane produced them. Its provenance is readable
//! through [`UnsatReceipt::published_summary`].

use ny_propagate::certify::work::{LeaseClass, PersistentReservation, WorkRequest};
use ny_propagate::certify::{
    legacy_uncovered_unsat, LegacyUnsatAuthority, UnsatSource, UnsatToken,
};
use std::time::Instant;

use super::prepared_root_publication::OriginalPreparedRootPublication;

/// The production legacy-uncovered door count, one entry per reviewed change,
/// newest LAST. It may only go DOWN: routing a lane through a cover removes its
/// door and appends the smaller count; a new uncovered `unsat` source would
/// have to append a LARGER one, which
/// `unsat_receipt_legacy_uncovered_ceiling_only_goes_down` refuses.
///
/// * 8 — the #soundness-spine M2b census (7 in `vnncomp.rs`, 1 in
///   `tll_structure.rs`);
/// * 7 — #spine-legacy-lanes: the isomorphic empty-unsafe-region shortcut now
///   publishes through `certify::authorize_scalar_emptiness`.
///
/// #graph-bab-cover-ledger did NOT lower the count: the `BetaCrownVerdict`
/// door keeps its one site (`vnncomp.rs::beta_crown_verdict_legacy`), because
/// the verifier's other routes (multi-objective, input-split multi-clause
/// disjunctive, GPU DomainList, sequential, lowdim, MIP, ...) have no ledger
/// yet. What changed is what
/// reaches it: a `verified` from a ledgered driver whose cover entails every
/// clause publishes FROM A TOKEN, and every remaining use prints
/// `[cover-gate] graph-ledger outcome=legacy reason=<why>`, so the uncovered
/// remainder is counted per reason in every run log.
///
/// Read only by this module's ratchet tests (hence the non-test allowance).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const LEGACY_UNCOVERED_CEILING_HISTORY: &[usize] = &[8, 7];

/// Proof that an `unsat` came through a certified door. See the module docs.
#[derive(Debug, Clone)]
pub(crate) struct UnsatReceipt {
    authority: ReceiptAuthority,
}

#[derive(Debug, Clone)]
enum ReceiptAuthority {
    Token(UnsatToken),
    OriginalPrepared(OriginalPreparedRootPublication),
    #[cfg(feature = "mip")]
    OriginalHull(std::sync::Arc<super::beta_crown::hull_publication::HullPublicationAuthority>),
}

/// CLI provenance distinguishes the original source owners from a facade token.
/// It never labels JSON or a foreign certificate as `UnsatSource::Authorized`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PublishedUnsatSource {
    Spine(UnsatSource),
    OriginalPrepared,
    #[cfg(feature = "mip")]
    OriginalHull,
}
impl From<UnsatSource> for PublishedUnsatSource {
    fn from(value: UnsatSource) -> Self {
        Self::Spine(value)
    }
}
impl PublishedUnsatSource {
    pub(crate) fn token(self) -> &'static str {
        match self {
            Self::Spine(source) => source.token(),
            Self::OriginalPrepared => "original_prepared_root",
            #[cfg(feature = "mip")]
            Self::OriginalHull => "original_hull_replay",
        }
    }
    pub(crate) fn legacy(self) -> Option<LegacyUnsatAuthority> {
        match self {
            Self::Spine(UnsatSource::LegacyUncovered(authority)) => Some(authority),
            _ => None,
        }
    }
}

impl PartialEq for UnsatReceipt {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for UnsatReceipt {}

impl UnsatReceipt {
    /// Accept a certifier token. `None` unless it is instance-scoped: a
    /// class-part or sub-domain proof must never reach RESULTS_FILE.
    pub(crate) fn from_token(token: UnsatToken) -> Option<Self> {
        token.publishable().then_some(Self {
            authority: ReceiptAuthority::Token(token),
        })
    }

    /// A LEGACY-UNCOVERED lane's `unsat`: the lane's word, no cover (ratcheted
    /// call sites; the count may only go down). Prints the lane's observable.
    pub(crate) fn legacy_uncovered(authority: LegacyUnsatAuthority) -> Self {
        eprintln!(
            "[cover-gate] legacy-uncovered authority={} (no cover certificate behind this unsat)",
            authority.token()
        );
        Self {
            authority: ReceiptAuthority::Token(legacy_uncovered_unsat(authority)),
        }
    }

    #[cfg(feature = "mip")]
    pub(crate) fn from_original_hull(
        authority: std::sync::Arc<super::beta_crown::hull_publication::HullPublicationAuthority>,
    ) -> anyhow::Result<Self> {
        authority.checkpoint()?;
        Ok(Self {
            authority: ReceiptAuthority::OriginalHull(authority),
        })
    }

    /// Only the genuine original-file issuer can supply this owner. Its
    /// complete source-ordered objective and original input enclosure are
    /// checked before the receipt is admitted; no numeric facts are imported.
    pub(crate) fn from_original_prepared(
        authority: OriginalPreparedRootPublication,
        requested: Option<Instant>,
    ) -> anyhow::Result<Self> {
        // The successful handoff's earlier clock is part of this immutable
        // owner value, including Clone and the read-only authority accessor.
        let authority = authority.restricted(requested)?;
        Ok(Self {
            authority: ReceiptAuthority::OriginalPrepared(authority),
        })
    }

    /// A staged or final writer must retain this receipt and repeat this check
    /// immediately before its publication. Tokens preserve their historical
    /// publication path; only genuine retained owners have a live ledger here.
    pub(crate) fn checkpoint_owner(&self, requested: Option<Instant>) -> anyhow::Result<()> {
        check_requested_deadline(requested)?;
        match &self.authority {
            ReceiptAuthority::Token(_) => {}
            ReceiptAuthority::OriginalPrepared(authority) => authority.checkpoint(requested)?,
            #[cfg(feature = "mip")]
            ReceiptAuthority::OriginalHull(authority) => authority.checkpoint()?,
        }
        check_requested_deadline(requested)
    }

    pub(crate) fn has_retained_owner(&self) -> bool {
        match &self.authority {
            ReceiptAuthority::Token(_) => false,
            ReceiptAuthority::OriginalPrepared(_) => true,
            #[cfg(feature = "mip")]
            ReceiptAuthority::OriginalHull(_) => true,
        }
    }

    /// Admit the actual writer's declared storage before allocation on the
    /// retained owner's original ledger. The returned claim must accompany all
    /// staged paths, rendered buffers and queued metadata until their last use.
    /// This prices only the caller's requested publication frontier; it cannot
    /// authorize other solver allocations or reconstruct source authority.
    pub(crate) fn reserve_owner_publication(
        &self,
        bytes: usize,
        requested: Option<Instant>,
    ) -> anyhow::Result<Option<PersistentReservation>> {
        self.checkpoint_owner(requested)?;
        let resident = match &self.authority {
            ReceiptAuthority::Token(_) => return Ok(None),
            ReceiptAuthority::OriginalPrepared(authority) => {
                let work = u64::try_from(bytes)?
                    .checked_mul(128)
                    .and_then(|n| n.checked_add(8192))
                    .ok_or_else(|| anyhow::anyhow!("prepared publication work overflow"))?;
                let mut lease = authority.budget().lease_for(
                    LeaseClass::Replay,
                    WorkRequest {
                        persistent_bytes: bytes,
                        work,
                        ..WorkRequest::default()
                    },
                )?;
                authority.checkpoint(requested)?;
                lease.mark_materialized()?;
                lease.checkpoint()?;
                authority.checkpoint(requested)?;
                lease.commit_persistent()?
            }
            #[cfg(feature = "mip")]
            ReceiptAuthority::OriginalHull(authority) => authority.reserve_publication(bytes)?,
        };
        self.checkpoint_owner(requested)?;
        Ok(Some(resident))
    }

    /// Clone shares the exact immutable source/proof owner and its last-owner
    /// claims. It never detaches a plain verifier token from the source.
    pub(crate) fn original_prepared_authority(&self) -> Option<OriginalPreparedRootPublication> {
        match &self.authority {
            ReceiptAuthority::OriginalPrepared(authority) => Some(authority.clone()),
            ReceiptAuthority::Token(_) => None,
            #[cfg(feature = "mip")]
            ReceiptAuthority::OriginalHull(_) => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn original_prepared_target(&self) -> Option<usize> {
        match &self.authority {
            ReceiptAuthority::OriginalPrepared(authority) => Some(authority.target()),
            ReceiptAuthority::Token(_) => None,
            #[cfg(feature = "mip")]
            ReceiptAuthority::OriginalHull(_) => None,
        }
    }

    /// Bind an authenticated capture/writer's model and property bytes to the
    /// genuine issuer. Other receipt variants cannot impersonate this source.
    pub(crate) fn validate_original_prepared_asset_hashes(
        &self,
        model: [u8; 32],
        property: [u8; 32],
        requested: Option<Instant>,
    ) -> anyhow::Result<()> {
        match &self.authority {
            ReceiptAuthority::OriginalPrepared(authority) => {
                authority.validate_asset_hashes(model, property, requested)
            }
            _ => anyhow::bail!("receipt has no original prepared model/property authority"),
        }
    }

    /// The exact immutable owner also accompanies pending/staged writers.
    #[cfg(all(test, feature = "mip"))]
    pub(crate) fn hull_authority(
        &self,
    ) -> Option<std::sync::Arc<super::beta_crown::hull_publication::HullPublicationAuthority>> {
        match &self.authority {
            ReceiptAuthority::OriginalHull(authority) => Some(authority.clone()),
            ReceiptAuthority::Token(_) | ReceiptAuthority::OriginalPrepared(_) => None,
        }
    }

    /// Unit-test verdicts.
    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        UnsatReceipt::legacy_uncovered(LegacyUnsatAuthority::TestFixture)
    }

    /// The facade-token provenance used by the existing token tests. The
    /// original replay has its distinct CLI source in `published_summary`.
    #[cfg(test)]
    pub(crate) fn source(&self) -> UnsatSource {
        match &self.authority {
            ReceiptAuthority::Token(token) => token.source(),
            ReceiptAuthority::OriginalPrepared(_) => {
                panic!("original prepared authority is not a facade token")
            }
            #[cfg(feature = "mip")]
            ReceiptAuthority::OriginalHull(_) => panic!("hull authority is not a facade token"),
        }
    }

    /// What a Proof-profile bundle records for this PUBLISHED verdict
    /// (#proof-profile B1): the token's classes, its minimum certified margin
    /// and its source. Read after publication only; nothing verdict-bearing
    /// consumes it.
    pub(crate) fn published_summary(&self) -> (Vec<usize>, f64, PublishedUnsatSource) {
        match &self.authority {
            ReceiptAuthority::Token(token) => (
                token.classes().to_vec(),
                token.min_margin(),
                token.source().into(),
            ),
            ReceiptAuthority::OriginalPrepared(authority) => (
                authority.classes().to_vec(),
                authority.min_margin(),
                PublishedUnsatSource::OriginalPrepared,
            ),
            #[cfg(feature = "mip")]
            ReceiptAuthority::OriginalHull(authority) => (
                vec![0],
                authority.lower_bound(),
                PublishedUnsatSource::OriginalHull,
            ),
        }
    }
}

fn check_requested_deadline(requested: Option<Instant>) -> anyhow::Result<()> {
    anyhow::ensure!(
        requested.is_none_or(|at| Instant::now() < at),
        "retained receipt publication deadline expired"
    );
    Ok(())
}

#[cfg(test)]
#[path = "unsat_receipt_prepared_tests.rs"]
mod prepared_tests;

#[cfg(test)]
mod tests {
    use super::{UnsatReceipt, LEGACY_UNCOVERED_CEILING_HISTORY};
    use ny_propagate::certify::{
        authorize_scalar_emptiness, LegacyUnsatAuthority, ScalarAtom, ScalarRel, TokenScope,
        UnsatSource,
    };
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    /// `from_token` publishes exactly the tokens the certifier marks
    /// publishable (instance scope). This crate can NOT mint an authorized,
    /// class-part or sub-domain token — the margin-row minting adapters are
    /// `pub(crate)` to ny-propagate (adversarial review of #soundness-spine) —
    /// so the scope refusal itself is pinned where tokens are minted
    /// (`certify::tests::unsat_token_scopes_publish_only_at_instance`), and
    /// this test pins the door's side of it on the one token this crate can
    /// hold: a legacy lane's instance-scoped token.
    #[test]
    fn unsat_receipt_accepts_only_instance_scoped_tokens() {
        let token =
            ny_propagate::certify::legacy_uncovered_unsat(LegacyUnsatAuthority::TestFixture);
        assert_eq!(token.scope(), TokenScope::Instance);
        assert!(token.publishable());
        let receipt = UnsatReceipt::from_token(token).expect("instance scope publishes");
        assert_eq!(
            receipt.source(),
            UnsatSource::LegacyUncovered(LegacyUnsatAuthority::TestFixture)
        );
        // Receipts are verdict-equal whatever their provenance.
        assert_eq!(receipt, UnsatReceipt::for_test());
        // Through a function pointer, so the frozen-door census below does not
        // count a test as a production door.
        let door = UnsatReceipt::legacy_uncovered;
        assert_eq!(
            door(LegacyUnsatAuthority::TllStructure).source(),
            UnsatSource::LegacyUncovered(LegacyUnsatAuthority::TllStructure)
        );
        // An exact specification-emptiness token is AUTHORIZED, instance
        // scoped, and publishes through the same door as the margin-row lane.
        let gt = ScalarAtom {
            rel: ScalarRel::Gt,
            constant: 0.05,
        };
        let lt = ScalarAtom {
            rel: ScalarRel::Lt,
            constant: -0.05,
        };
        let exact = authorize_scalar_emptiness(&[vec![gt, lt]]).expect("empty region");
        let receipt = UnsatReceipt::from_token(exact).expect("instance scope publishes");
        assert_eq!(receipt.source(), UnsatSource::Authorized);
    }

    fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {dir:?}: {e}"));
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                rust_files(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    /// The ny-cli source tree, and every legacy-uncovered door in it, per
    /// (file, authority). Needles are assembled at runtime so this module
    /// cannot count itself; comment lines are not code.
    struct DoorCensus {
        doors: BTreeMap<(String, String), usize>,
        raw_sites: Vec<String>,
        files: usize,
    }

    fn door_census() -> DoorCensus {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        rust_files(&root, &mut files);
        let door = format!("{}::{}(", "UnsatReceipt", "legacy_uncovered");
        let authority = format!("{}::", "LegacyUnsatAuthority");
        let raw = format!("{}(", "legacy_uncovered_unsat");
        let mut doors: BTreeMap<(String, String), usize> = BTreeMap::new();
        let mut raw_sites = Vec::new();
        for file in &files {
            let text = std::fs::read_to_string(file).expect("read source");
            let rel = file
                .strip_prefix(&root)
                .expect("under src")
                .to_string_lossy()
                .replace('\\', "/");
            let code: String = text
                .lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n");
            if code.contains(raw.as_str()) {
                raw_sites.push(rel.clone());
            }
            for (at, _) in code.match_indices(door.as_str()) {
                let rest = &code[at + door.len()..];
                let name: String = rest
                    .trim_start()
                    .strip_prefix(authority.as_str())
                    .unwrap_or("?")
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric())
                    .collect();
                *doors.entry((rel.clone(), name)).or_default() += 1;
            }
        }
        raw_sites.sort();
        DoorCensus {
            doors,
            raw_sites,
            files: files.len(),
        }
    }

    /// Every legacy-uncovered `unsat` door in ny-cli, per file and per
    /// authority. A new lane that publishes `unsat` must route through
    /// `certify::authorize_unsat` (a cover) or add a reviewed row here — and
    /// the ceiling test below refuses the latter unless it also REMOVES a door.
    #[test]
    fn unsat_receipt_legacy_sites_are_frozen() {
        let expected: BTreeMap<(&str, &str), usize> = [
            (("commands/vnncomp.rs", "BetaCrownVerdict"), 1),
            (("commands/vnncomp.rs", "CzHybridLane"), 1),
            (("commands/vnncomp.rs", "RelationalImplication"), 2),
            (("commands/vnncomp.rs", "NonlinearIntervalBab"), 1),
            (("commands/vnncomp.rs", "CctsdbSubbox"), 1),
            (("commands/tll_structure.rs", "TllStructure"), 1),
            (("commands/unsat_receipt.rs", "TestFixture"), 1),
        ]
        .into_iter()
        .collect();
        let census = door_census();
        assert!(census.files > 50, "scanned only {} files", census.files);
        let found: BTreeMap<(&str, &str), usize> = census
            .doors
            .iter()
            .map(|((f, a), n)| ((f.as_str(), a.as_str()), *n))
            .collect();
        assert_eq!(
            found, expected,
            "legacy-uncovered unsat doors changed: route new lanes through \
             certify::authorize_unsat or review and update this manifest"
        );
        assert_eq!(
            census.raw_sites,
            vec!["commands/unsat_receipt.rs".to_string()],
            "only UnsatReceipt::legacy_uncovered may call the certifier's legacy-uncovered door"
        );
        // The authority each door names is still a variant (a door naming a
        // routed lane would not compile, but a stale manifest row would).
        let variants: Vec<String> = LegacyUnsatAuthority::ALL
            .iter()
            .map(|a| format!("{a:?}"))
            .collect();
        for (_, authority) in expected.keys() {
            assert!(
                variants.iter().any(|v| v == authority),
                "manifest names `{authority}`, which is not a LegacyUnsatAuthority"
            );
        }
    }

    /// The production legacy-uncovered door count is the LAST entry of
    /// `LEGACY_UNCOVERED_CEILING_HISTORY`, and the history never rises: an
    /// uncovered `unsat` source can be retired, never added. It also pins the
    /// isomorphic lane's routing: its authority is gone from the census.
    #[test]
    fn unsat_receipt_legacy_uncovered_ceiling_only_goes_down() {
        let history = LEGACY_UNCOVERED_CEILING_HISTORY;
        assert_eq!(history.first(), Some(&8), "the M2b census started at 8");
        assert!(
            history.windows(2).all(|w| w[1] < w[0]),
            "the legacy-uncovered ceiling may only go DOWN: {history:?}"
        );
        let census = door_census();
        let production: usize = census
            .doors
            .iter()
            .filter(|((_, authority), _)| authority != "TestFixture")
            .map(|(_, n)| *n)
            .sum();
        let ceiling = *history.last().expect("non-empty history");
        assert_eq!(
            production, ceiling,
            "production legacy-uncovered doors = {production}, ceiling = {ceiling}: lower the \
             ceiling (append the new count) when a lane is routed through a cover; never raise it"
        );
        assert!(
            !census
                .doors
                .keys()
                .any(|(_, authority)| authority.contains("Isomorphic")),
            "the isomorphic emptiness shortcut publishes through a token, not a legacy door"
        );
        // Every production authority still in the census has a door, and no
        // door names an authority outside the census.
        for authority in LegacyUnsatAuthority::ALL {
            let name = format!("{authority:?}");
            let doors: usize = census
                .doors
                .iter()
                .filter(|((_, a), _)| *a == name)
                .map(|(_, n)| *n)
                .sum();
            assert!(
                doors > 0,
                "`{name}` has no door left: delete the variant (the lane is routed)"
            );
        }
    }

    /// The exact specification-emptiness entry returns an AUTHORIZED token for
    /// whatever atoms it is handed (it re-checks the certificate, not that the
    /// atoms are the property), so its callers are a door census of their own:
    /// exactly one production caller, the isomorphic lane, which passes the
    /// parser's REAL atoms. A second caller must be reviewed here, never added
    /// silently — it would publish `unsat` outside the legacy-uncovered census.
    ///
    /// The census spans EVERY workspace crate's `src`, not only ny-cli's: the
    /// entry is `pub`, so a token minted in any crate that reaches a
    /// `from_token` door publishes. And it counts the bare IDENTIFIER, not a
    /// call shape: an import alias (`use ... as other;`) or a function pointer
    /// is a caller that a `name(` needle would miss. Only the facade itself
    /// (`ny-propagate/src/certify/`: definition, re-export, unit tests) is
    /// exempt. (Adversarial review of #spine-legacy-lanes.)
    #[test]
    fn exact_emptiness_entry_is_called_only_by_the_isomorphic_lane() {
        let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the workspace crates directory")
            .to_path_buf();
        let mut crate_srcs: Vec<PathBuf> = std::fs::read_dir(&crates)
            .unwrap_or_else(|e| panic!("read {crates:?}: {e}"))
            .map(|entry| entry.expect("crate entry").path().join("src"))
            .filter(|src| src.is_dir())
            .collect();
        crate_srcs.sort();
        assert!(
            crate_srcs.len() >= 20,
            "found only {} crate src trees under {crates:?}: the census would be VOID",
            crate_srcs.len()
        );
        let name = format!("{}_{}", "authorize_scalar", "emptiness");
        let entry = format!("{name}(");
        let facade = format!("{}/src/{}/", "ny-propagate", "certify");
        let mut found: BTreeMap<String, usize> = BTreeMap::new();
        let mut vnncomp_code = String::new();
        let mut scanned = 0usize;
        for src in &crate_srcs {
            let mut files = Vec::new();
            rust_files(src, &mut files);
            for file in &files {
                let rel = file
                    .strip_prefix(&crates)
                    .expect("under the crates directory")
                    .to_string_lossy()
                    .replace('\\', "/");
                if rel.starts_with(facade.as_str()) {
                    continue;
                }
                scanned += 1;
                let text = std::fs::read_to_string(file).expect("read source");
                let code: String = text
                    .lines()
                    .filter(|line| !line.trim_start().starts_with("//"))
                    .collect::<Vec<_>>()
                    .join("\n");
                let n = code.matches(name.as_str()).count();
                if n > 0 {
                    found.insert(rel.clone(), n);
                }
                if rel == "ny-cli/src/commands/vnncomp.rs" {
                    vnncomp_code = code;
                }
            }
        }
        assert!(
            scanned > 1_000,
            "scanned only {scanned} workspace source files: the census would be VOID"
        );
        let expected: BTreeMap<String, usize> = [
            // The production call, spelled by its full path.
            ("ny-cli/src/commands/vnncomp.rs".to_string(), 1),
            // The `use` and the call in
            // `unsat_receipt_accepts_only_instance_scoped_tokens` above.
            ("ny-cli/src/commands/unsat_receipt.rs".to_string(), 2),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            found, expected,
            "users of the exact-emptiness entry changed: it mints an AUTHORIZED token for \
             whatever atoms it is handed, so every new user is a reviewed door"
        );
        assert!(
            !vnncomp_code.is_empty(),
            "ny-cli/src/commands/vnncomp.rs was not scanned: the lane check would be VOID"
        );
        let head = format!("fn {}(", "try_prove_empty_isomorphic_unsafe_region");
        let start = vnncomp_code
            .find(head.as_str())
            .expect("the isomorphic lane");
        let end = vnncomp_code[start..]
            .find("\n}\n")
            .map_or(vnncomp_code.len(), |e| start + e);
        assert!(
            vnncomp_code[start..end].contains(entry.as_str()),
            "the one production caller must be the isomorphic emptiness lane"
        );
    }

    /// #graph-bab-cover-ledger: the graph instance door returns an AUTHORIZED,
    /// instance-scoped token when captured covers entail every clause of the
    /// property it is handed (it re-checks entailment and containment, not
    /// that the property is the instance's), so its users are a reviewed door
    /// census: exactly one production user, the publication module that builds
    /// the property from the parsed VNN-LIB with the lanes' own objective
    /// encoding. Same census shape as the exact-emptiness entry above: every
    /// workspace crate's `src`, the bare identifier, the facade exempt.
    #[test]
    fn graph_instance_entry_is_called_only_by_the_publication_module() {
        let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the workspace crates directory")
            .to_path_buf();
        let mut crate_srcs: Vec<PathBuf> = std::fs::read_dir(&crates)
            .unwrap_or_else(|e| panic!("read {crates:?}: {e}"))
            .map(|entry| entry.expect("crate entry").path().join("src"))
            .filter(|src| src.is_dir())
            .collect();
        crate_srcs.sort();
        assert!(
            crate_srcs.len() >= 20,
            "found only {} crate src trees under {crates:?}: the census would be VOID",
            crate_srcs.len()
        );
        let name = format!("{}_{}", "authorize_graph", "instance");
        let facade = format!("{}/src/{}/", "ny-propagate", "certify");
        let mut found: BTreeMap<String, usize> = BTreeMap::new();
        let mut publish_code = String::new();
        let mut scanned = 0usize;
        for src in &crate_srcs {
            let mut files = Vec::new();
            rust_files(src, &mut files);
            for file in &files {
                let rel = file
                    .strip_prefix(&crates)
                    .expect("under the crates directory")
                    .to_string_lossy()
                    .replace('\\', "/");
                if rel.starts_with(facade.as_str()) {
                    continue;
                }
                scanned += 1;
                let text = std::fs::read_to_string(file).expect("read source");
                let code: String = text
                    .lines()
                    .filter(|line| !line.trim_start().starts_with("//"))
                    .collect::<Vec<_>>()
                    .join("\n");
                let n = code.matches(name.as_str()).count();
                if n > 0 {
                    found.insert(rel.clone(), n);
                }
                if rel == "ny-cli/src/commands/graph_cover_publish.rs" {
                    publish_code = code;
                }
            }
        }
        assert!(
            scanned > 1_000,
            "scanned only {scanned} workspace source files: the census would be VOID"
        );
        let expected: BTreeMap<String, usize> = [
            // The `use` and the one call in `graph_instance_unsat`.
            ("ny-cli/src/commands/graph_cover_publish.rs".to_string(), 2),
            // TEST-ONLY user (`#[cfg(test)]` module of ny-propagate): the
            // end-to-end completeness pin runs the real ledgered drivers and
            // hands every `Verified` run's capture to the door for exactly its
            // own property (the `use` and the one call in `run_family`). It
            // publishes nothing; a production user would still fail here.
            (
                "ny-propagate/src/beta_crown/engine/graph/cover_ledger_tests.rs".to_string(),
                2,
            ),
            // TEST-ONLY user (#vit-v3-bankable, `#[cfg(test)]` module of
            // ny-propagate): the multi-objective family runs the real driver
            // under `bab.mo_cover_certify: ledger` and hands every `Verified`
            // run's per-row capture to the door for exactly its own two-row
            // property (the `use`, the batched call and the root-terminal
            // call). It publishes nothing. review(vit-v3) adds two more
            // test-only calls (count taken from this census's own failure
            // output): the enumeration-exact family hands each armed
            // `Verified` capture to the door for its own property, and for
            // the FALSE property one ulp past a row's exact minimum, where
            // the door must REFUSE.
            (
                "ny-propagate/src/beta_crown/engine/graph/mo_cover_tests.rs".to_string(),
                5,
            ),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            found, expected,
            "users of the graph instance door changed: it mints an AUTHORIZED token for \
             whatever property it is handed, so every new user is a reviewed door"
        );
        let head = format!("fn {}(", "graph_instance_unsat");
        let start = publish_code
            .find(head.as_str())
            .expect("the publication entry");
        let end = publish_code[start..]
            .find("\n}\n")
            .map_or(publish_code.len(), |e| start + e);
        let body = &publish_code[start..end];
        let property = format!("{}(&spec)", "graph_instance_property");
        assert!(
            body.contains(format!("{name}(").as_str()) && body.contains(property.as_str()),
            "the one production call must hand the door the property parsed from THIS \
             instance's VNN-LIB"
        );
    }

    /// #spine-legacy-lanes M2e: every legacy-uncovered lane that reads a
    /// `GraphNetwork` as PROOF AUTHORITY loads it through the certified
    /// lowering (`OnnxLoadConfig::certified_graph_lowering`: every authored
    /// BatchNormalization kept, so no fold drops its coefficient enclosure).
    /// Their models carry no BatchNormalization today, so the rule is
    /// byte-identical on every banked row and only closes the door for a
    /// future BN model.
    #[test]
    fn legacy_uncovered_lanes_load_authority_graphs_through_the_certified_lowering() {
        fn top_level_fn<'a>(text: &'a str, name: &str) -> &'a str {
            let head = format!("fn {name}(");
            let start = text
                .find(head.as_str())
                .unwrap_or_else(|| panic!("`{head}` not found: the scan would be VOID"));
            let end = text[start..]
                .find("\n}\n")
                .map_or(text.len(), |e| start + e);
            &text[start..end]
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/commands");
        let read = |file: &str| {
            let text = std::fs::read_to_string(root.join(file)).expect("read source");
            text.lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let certified_load = format!("{}(", "load_certified_graph_network");
        let lowering = format!(".{}()", "certified_graph_lowering");
        let plain_load = format!("{}(", "load_graph_network");
        let vnncomp = read("vnncomp.rs");
        let loader = top_level_fn(&vnncomp, "load_certified_graph_network");
        assert!(loader.contains(lowering.as_str()), "{loader}");
        // RelationalImplication: the difference-network pair.
        let relational = top_level_fn(&vnncomp, "load_relational_graphs_or_unknown");
        assert!(relational.contains(certified_load.as_str()), "{relational}");
        assert!(
            !relational.contains(plain_load.as_str()),
            "the relational authority graphs must not use the folding loader"
        );
        // NonlinearIntervalBab: the f64 cell evaluator's graph.
        let nonlinear = top_level_fn(&vnncomp, "try_nonlinear_unsat");
        assert!(nonlinear.contains(lowering.as_str()), "{nonlinear}");
        // CctsdbSubbox: the excised sub-box model the cells are refuted on.
        let cctsdb = read("cctsdb_subbox.rs");
        let run = top_level_fn(&cctsdb, "run");
        assert!(
            run.contains(format!("{certified_load}temp.path())").as_str()),
            "{run}"
        );
    }
}
