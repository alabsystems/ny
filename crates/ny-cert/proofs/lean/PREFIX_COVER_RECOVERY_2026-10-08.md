# Prefix cover reconstruction and qualification boundary

This is a **new replacement** for the missing source described in
`docs/engine_program/HIGH_IMPACT_PREFIX_FORMAL_2026-10-01.md`. It is not a recovery
of that source and inherits none of its reported results or axiom audit. Official
Lean 4.30.0 and 4.31.0 accept the new source. The complete pinned Lean 4.30.0
overlay and its 186 required axiom queries now pass. The separate linked CIC
invocation remains rejected. `AXIOM_AUDIT.txt` is regenerated from the actual
pinned full audit stdout, with its historical headers preserved.

## Recovery evidence

The documented original is 10,738 bytes with SHA256
`81ef034d86addd4304a8d0656a6a52366925f113b765e107ce0584037352803d`.
Read-only recovery on 2026-10-08 found:

- No source path in `git log --all --full-history` or
  `git rev-list --all --reflog --objects`.
- No exact blob in the 110,101 objects exposed by the NY repository and both
  recovered object databases, including unreachable objects. The only blob of
  that length has SHA256 `3e0810597e7ca831e39c329af43ec51a7c21b7a17bf685ccd8ef82e43c11c79f`.
- No Std prefix theorem variant in 36,202 candidate blobs of 3,000–60,000 bytes.
- No exact or variant blob in all 1,314 entries of the retained
  `recovered-candidate.bundle`; its pack checksum was validated. Its advertised
  head is already available in the NY object database.
- No exact blob in 147,036 Clean objects. The older sandbox object database is
  empty. A search of 1,555,061 retained source/archive files and the available NY
  session records found references to the historical claim, without its source.

## New source and scope

`NyProof/PrefixCover.lean` imports only Std. A literal contains an original site
identifier and `Off` or `On`. `PhasePredicates.exhaustive` explicitly supplies
coverage by the original closed predicates; it does not certify a numerical
approximation. `Fits` is conjunction of those predicates along a path.

For parent `P` and chosen path `s` of length `k`, `completeMembers` consists of
`P ++ s` first, followed in increasing `j` by
`P ++ s.take j ++ [opposite(s[j])]`. The new source establishes:

- Every original parent sample is in a complete member, and every member is
  contained in the parent.
- Exactly `k+1` descriptors, destination first, with siblings ordered by the
  first-disagreement index. A sibling has depth `P.length+j+1`, copies only the
  preceding phases and stops at its opposite final phase.
- Safety transfer requires valid safety for **every** complete member. A common
  lower bound must be below every valid member bound. The generic lower-bound
  theorem assumes the comparison relation's transitivity explicitly.
- Binary assignments select exactly one descriptor under the extra disjointness
  premise. Closed real phase predicates can overlap at zero; uniqueness is not
  claimed for them.

Four exact controls demonstrate closed-zero overlap, three distinct descriptors
that all contain zero, valid minimum aggregation, and invalid maximum aggregation
including a positive destination with an unsafe sibling.

### Rust correspondence boundary

The descriptor model is matched to
`crates/ny-propagate/src/margin_row/zono_bab/prefix_cover.rs`:

| Rust source | Lean model or theorem |
| --- | --- |
| `chosen = parent ++ chosen_path` | `parent ++ chosen` |
| `members()` emits destination then `FirstDisagreement(i-1)` | `destination_first`, `canonical_member_order` |
| sibling depth `parent_depth+j+1` | `actual_sibling_depth` |
| earlier `chosen[i]`, final `opposite(chosen[parent_depth+j])` | `descriptorSibling`, `sibling_final_opposite`, `sibling_prefix_stops_at_first_opposite` |
| no reads beyond sibling depth | `sibling_has_no_later_phase` |
| exact ordered original split-row IDs | literal `site` must be mapped from `problem.order()[i]` |

`sibling_descriptor_correspondence` proves equality between two executable Lean
definitions. This source comparison has not been upgraded to an authenticated
translation or a theorem about compiled Rust. Rust's checked original row order,
source provenance, capacities, deadlines, allocation refusal, immutable owners,
and numerical row premises remain separate proof obligations. The structural
theorem supplies no pruning token, network lower bound or runtime authority.

## Actual validation and remaining qualification

The author reviewed the theorem assumptions, then executed official Lean and
`targo trust check` through the unchanged `scripts/ny_box.sh build` lock. Every
attempt uses an exclusive scratch source snapshot and pre/post SHA256 checks of
both that snapshot and the original source. Complete command lines, tool version,
stdout, stderr, actual statuses and source receipts are retained under
`scratchpad/high-impact-recovery-20261007/formal-prefix-e1-20261008-v1` through
`formal-prefix-e1-20261008-v4`. No policy override or toolchain installation was
performed.

| Attempt | Exact source SHA256 | Official Lean 4.31.0 | Actual linked CIC |
| --- | --- | --- | --- |
| v1 | `26029fa2ceab9a13afc372cebaa12df030a8c9738c60db2acdc2793915ae89cb` | FAIL, rc1: unavailable Std `∃!` notation and hidden zero-control decidability | REJECTED, rc1: nested tactic parsing |
| v2 | `f285f1d6b943728cc80ce1a441310c17a6ac8cd5c69fff1b40446035ec349d85` | PASS, rc0 | REJECTED, rc1: multiple `generalizing` binders |
| v3 | `160ec7a17eeaba0139529fc3d326fd8149c838663f43b31e85c12894540a61a7` | PASS, rc0 | REJECTED, rc1: same unsupported syntax after binder renaming |
| v4 | `6d10274535c9f39f7f4441e0fded96e88e7da77327153747f267800e148ec708` | PASS, rc0 | REJECTED, rc1: Std import, lemmas, tactics and audit commands unavailable |

The repairs preserve the theorem's predicates and assumptions: explicit
existence/uniqueness replaces unavailable notation, the zero control unfolds its
exact predicates, and elementary Or/Exists elimination plus a forall conclusion
replaces the rejected induction syntax. The failed v1 run is preserved, including
its error-generated `sorryAx`; it is not accepted proof evidence.

The **current source is v4, 18,302 bytes**. Its ordinary standalone command is
the installed official binary
`<home>/.elan/toolchains/leanprover--lean4---v4.31.0/bin/lean` applied to the
exact snapshot. A separate full-query driver includes those exact source bytes
as its prefix and adds all 27 declaration queries. That driver also returns rc0;
all 27 actual axiom rosters contain only `propext` and/or `Quot.sound`, or no
axioms. It has no `sorryAx`, unknown constant or missing query. The driver, its
distinct SHA256, full logs, receipts and machine-readable rosters are retained in
v4; that run is new standalone evidence. The later pinned overlay run below
supplies the evidence for regenerating the current overlay audit.

The actual linked CIC command is `targo trust check SOURCE --census NEW_PATH`
using measured Trust 9192, source revision
`321aaeda75478038420d97830f14724c33c8dc39`. Its final 59 diagnostics include
`unresolved import Std: no Clean-native prelude provides it`, missing `Or.symm`
and List lemmas, elaboration limitations and unsupported `#print axioms` commands.
The diagnostic census reports accepted0/rejected1, transactional batch rejection,
zero new declarations and unchanged owned-identity count 1,989. That is **no
linked CIC proof**, not a small accepted subset. Installing or changing tools,
copying Clean libraries or treating an external `.olean` as linked-kernel
authority was not attempted. The initially attempted `targo trust version --json`
also failed rc2; actual `trustc -Vv` and the invocation's tool observations are
captured instead. Nothing here claims complete toolchain or loaded-image binding.

## Pinned Lean 4.30.0 overlay qualification

The parent installed official Lean 4.30.0 and prepared the exact pinned
dependencies after the four attempts above. The qualified source snapshot is
commit `cd23acfe5a44e77f770854d9d9fd159fa31386ac`, tree
`29c8a7f1ad41cebb1f42f25fbce0d0fa8e1a458a`. Its plan pins 78 files, including
72 Lean files; all planned hashes and tracked source cleanliness were checked
before and after both commands. The new `PrefixCover.lean` remains the exact
18,302-byte v4 source with SHA256
`6d10274535c9f39f7f4441e0fded96e88e7da77327153747f267800e148ec708`.

| Exact official command | Actual result |
| --- | --- |
| Lean 4.30.0 applied to the frozen `PrefixCover.lean` | PASS, rc0 |
| Lean 4.30.0 applied to the full 27-query standalone driver | PASS, rc0; 26 public declarations and the private helper accounted for |
| `lake build NyProof NyProof.AxiomAudit` under Lean 4.30.0 | PASS, rc0 |
| `lake env lean NyProof/AxiomAudit.lean` under Lean 4.30.0 | PASS, rc0; all 186 required queries, including 18 new PrefixCover queries |

The official version is Lean 4.30.0, commit
`d024af099ca4bf2c86f649261ebf59565dc8c622`. Every required overlay query was
independently matched to its actual stdout roster. The rosters contain only
`propext`, `Classical.choice`, and/or `Quot.sound`, or no axioms; there are no
missing queries, extra axioms, unknown constants or `sorryAx`. The new standalone
PrefixCover queries use only `propext` and/or `Quot.sound`, or no axioms.

All ten dependency Git HEADs equal the exact `lake-manifest.json` revisions and
are tracked-clean. They include Clean
`e7ae5d418cbac2e1a029369a5b5eda690d471c5c` and Mathlib
`c5ea00351c28e24afc9f0f84379aa41082b1188f`; the complete list and independent
verification are in the evidence packet. This qualification applies to the
official Lean overlay, with its stated dependencies and trust base.

Exact commands, immutable logs, plans, source hashes, earlier failed attempts,
the axiom rosters and the historical audit are archived in
the formal evidence packet.
The prior 56-artifact receipt and both standalone packets remain unchanged.
The extended receipt additionally binds the complete pinned overlay run and
the regenerated audit. The actual private helper name differs between the
4.30 `_stdin` and 4.31 filename prefixes; both exact names are recorded, rather
than treating name normalization as proof authority.

`AXIOM_AUDIT.txt` now preserves the historical comment headers, adds the exact
audited source/version/query metadata, and then includes the complete actual
4.30 overlay audit stdout unchanged. Its stdout source has SHA256
`fa00c2bcf40bd9dcbc21eefec78e21350f9cc8dceda19ca307a4d78c72b068ca`.

The pinned overlay gate is qualified. The missing historical `81ef…` source is
still unrecovered and receives no proof credit. Linked CIC remains rejected;
there is no authenticated translation or compiled Rust theorem, no numerical
source pruning authority, and no machine-checked allocation/deadline proof.
These results do not close E1 in full or establish a runtime improvement.
