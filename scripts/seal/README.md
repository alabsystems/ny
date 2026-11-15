# Seal pipeline

The system that turns a BUILT capability into a SEALED one, so the router can route it (ROUTER_PROGRAM §3.5 and R5,
"capabilities routed, not armed"). It derives a pre-registered plan from the capability's registry entry
(`docs/engine_program_capabilities.yaml`), runs it through the m0 runner, and writes evidence that
`scripts/engine_program_audit.py` re-derives into S5–S8 and SR.seal. Nothing here is hand-written into the
registry, and nothing new reads `NY_*`.

| file | what |
|---|---|
| `seal.py` | The CLI: `plan`, `run`, `judge-stage`, `envelope`, `record`, `verify-record`, `show` (+ runner helpers). |
| `seallib.py` | Every decision: derivation, overlays, row selection, shards, the stage judge, facts, the envelope, the registry write-back. The auditor imports it. |
| `seal_run.sh` | The one bash runner: step 0, `systemd-run --user`, the m0 runner per stage and shard, heartbeat, DONE line. |
| `certlib.py` | The CERTIFIED S8 mode (#certified-moat, below): row classes, follow-up derivation, the certificate check, the ledger, the replay-record check, the decision. The stage judge (and so the auditor) calls it. |
| `x86_replay.py` | The certified S8's wrong-sat check: every armed-sat witness through the organisers' checker code under emulated x86 (the scratchpad/x86replay + x86parity rig), with witness controls per batch. |
| `seal_selftest.py` | Named unittest tests on a stub world (stub pin, stub `ny_box.sh`, git repo, benchmarks, bank, field results; for the certified mode the stub pin also writes Proof bundles and answers `proof-check`, and a stub x86 rig has the real rig's interfaces). No box. |
| `keyless_moat_plan.py` | #prepared-root-route: the pre-registered S8 moat plan of a KEYLESS, default-on routable capability (`derive` refuses keyless entries): the census-admitted moat rows (+ the superset tail the census refused only for `envelope:ops` / `admit:unknown:*`) on the SHIPPED configs, built with `census_corpus` / `census_admitted` / `s8_rows`, plus the certified-mode B2 replay rule. Writes a plan; runs nothing. |
| `allonlib.py` | VENDORED byte-identical from `exp/all-on-preset` f39aa5bd7 `scripts/allon/allonlib.py` (blob `510c541cb`); a test pins the blob. Re-vendor, never edit. |

```
python3 scripts/seal/seal_selftest.py      # last line: SEAL_SELFTEST_DONE rc=0 ran=<n> ...
```

The selftest pins the vendored file's blob on every host. Re-deriving that blob from its source commit needs the
commit's objects (`exp/all-on-preset` f39aa5bd7, in no ref of the shared remote), so it is the selftest's external
lane: explicitly selected, failing where the commit is absent, never a skip.

```
python3 scripts/seal/seal_selftest.py --external   # every external_* contract; rc=1 if any fails or none ran
```

**Nothing runs that lane for you.** Until 2026-10-08 the cross-check ran inside the default selftest on any clone
that held the commit and was skipped everywhere else; it now runs only when selected. Run it on a host that holds
`exp/all-on-preset` whenever `VENDORED_ALLONLIB` (seallib.py) or `allonlib.py` changes. Pushing f39aa5bd7 to a ref of
the shared remote would let every clone run it.

## The ladder, per capability

| stage | rows | arms | rule (VOID is never a pass) |
|---|---|---|---|
| **S5** arrives | banked-unsat rows whose banked wall is at least 20% of budget, cheapest first, round-robin over the admitted units (`--arrival-rows`, default 2, at least 1: the plan refuses 0 and the judge VOIDs a pre-registered N < 1) | the capability armed **only** by the generated overlay, zero `NY_*`; 1 competition run | PASS when the observable (every `need` regex) shows on at least N rows, and each armed key with a LeverDecl reads `@config` in the flight receipt. Otherwise VOID. |
| **S7** controls | truly-SAT rows from field truth (the registry's `seal.negctl.rows`, else the m0 tier's rows; then seeded field-SAT rows, witnessed first, for EVERY unit those rows leave uncovered -- an m0 tier's rows are 2026-only), plus knife-edge rungs (m0 `knife_edge`) where a banked witness makes one expressible | overlay + every attack key off (m0 `ATTACK_TABLE`, checked against the pin's preset schema); the observable is REQUIRED; 1 competition + 1 proof | Any unsat or alarm is an EMERGENCY and stops everything. PASS when every run is OK, the knife state is `ran` or `not-expressible` (relational-only rows, as documented in the m0 README), and every admitted unit has a clean control. Otherwise PARTIAL or VOID. |
| **S6** measured | strata per unit: target (`seal.s6.rows`), gap (banked undecided, field-decided), near (banked decided at ≥ 70% of budget), sat, uniform; all seeded | armed vs capability-ABSENT control (the keys deleted). The control must never engage (`!need`). Marginal strata (target, gap, near) get 3 reps, the rest 1; competition profile; the first arm alternates by row | Absolute counts only: *converted* means the arm decides in ⌈2n/3⌉ of n reps where the control leaves it undecided in ⌈2n/3⌉; *lost* is the reverse; *closes* are absolute totals when the entry names a closes regex. A conversion on a row where the armed arm never showed the observable is noise (both arms ran the same code) and is reported as `converted_unengaged`, never credited. ADOPT = at least 1 credited conversion and 0 lost; REJECT = any lost; else NULL. VOID when any run is not OK, or when the arm never engaged. |
| **S8** moat (classic) | m0 `moat_rows` on the ARMED configs: every banked sat/unsat row and every field-SAT row of the admitted categories, both years -- with an admission census, EXACTLY the census-admitted moat rows of every scored category (below) | 3 competition + 1 proof reps | Any flip, truth contradiction, `PROFILE-DECIDED-SPLIT` (a Proof verdict that Competition does not reproduce in at least 2 of 3 reps) or alarm is an EMERGENCY. A lost banked verdict is PARTIAL. So is a moat below its FLOOR, whatever the plan options said: fewer than 3 competition + 1 proof reps on a row (`--s8-reps-c 1` makes "2 of 3" vacuous), a plan that left out 2026 or 2025, or an admission scope narrowed by `--categories` (`scope_categories` in the plan). VOID when a run is not OK, or when the capability engaged on no moat run. |
| **S8** moat (certified) | the same rows | 1 competition run per row, then per-row evidence (below) | EMERGENCY / VOID / PARTIAL / PASS as in "Certified S8": every armed unsat certified by an independent `ny proof-check` ACCEPT or covered by a classic 3c+1p sub-moat, every armed sat replayed on the organisers' checker under emulated x86, 0 lost. |
| **envelope** | the S7 and S8 rows on which the capability ENGAGED (excluding any row where the router's own `[router] cap=<id> admit=` line declined) | | Op kinds (OpKind tokens) and spec archetypes over those rows. `other` is never sealed. A row whose router receipt reports op kinds that the offline scan did not find is DISPUTED, and SR.seal refuses. |

Applied to every stage:
- m0's per-run judge. EMERGENCY is sticky.
- m0's profile rules.
- m0's cross-tag INSTANCE-FLIP, applied across ALL stages of the plan.
- A run whose flight `build_id` names another commit than the plan's pin, or carries no `build_id` at all, is VOID.

**SR.seal**, as the auditor derives it (`seallib.sr_seal`, the same function `seal record` applies before it writes
`SR: done` or an envelope):
- S7 and S8 are derived done from seal records. A record derives only if it re-derives to the SAME record: every
  field (rows, engagement, facts re-scanned in memory, the S6 decision, ...) except when and with which snapshot it
  was judged, and every judged.tsv line re-derived from its run's raw log, results file and flight record with m0's
  `judge_run` (an edited verdict re-recorded through `seal judge-stage` carries a fresh sha, so only the artefacts can
  refuse it).
- The record's pin contains the entry's heads.
- `routable.envelope` and `envelope_specs` lie inside the evidence rows' op sets and archetypes.
- **The envelope's closure is moated.** The router routes a sealed capability on any row its own predicate admits
  inside the envelope, whatever the category. With an **admission census** in the plan (`plan --census run|<tsv>`,
  below) the closure is EXACTLY the census-admitted moat rows (`census_closure`): every one must be an S8 row, the
  census must verify (trailer, the plan's pre-registered sha) and bind to this seal (made by the pin's binary, over
  the CURRENT moat corpus, under an envelope at least as wide as the one being sealed); a stale census is re-run on
  the pin by `seal record` and refused by the auditor until it is. A re-run binds only when it was made on the
  plan's census HOST (the router's precondition reads the host), and a census that admits no moat row is no
  closure (review fixes: both passed vacuously before). Without a census, the legacy offline closure
  applies: every m0 moat row of BOTH years and EVERY scored category whose offline op kinds fit the envelope (no
  `other`) and whose spec archetype is in it (an unreadable model or spec counts as inside) must be an S8 row; it
  REFUSES a plan that carries a census. Measured 2026-10-01: a frozen-dual-like envelope (`linear conv relu bn
  residual` x `robustness`) sealed from a cifar100 + tinyimagenet moat admits 2959 moat rows in 13 other categories
  (cora, malbeware, metaroom, safenlp, relusplitter, challenging_certified_training, ...); the check took 44.5 s.

## Certified S8 (#certified-moat)

A classic S8 runs every census-admitted moat row 3 x Competition + 1 x Proof. For `frozen-dual-bab` that is 678 rows /
2712 runs (~75 box-hours), so nothing gets sealed. `plan --s8-mode certified` replaces the REPETITION with
INDEPENDENT EVIDENCE per row, and runs each row once:

| phase | what | where |
|---|---|---|
| **A** (`--stage S8`) | every S8 row (the same census-admitted rows) ONCE in Competition, capability armed by the overlay, zero `NY_*`, official budget, the m0 judge per run (truth contradiction, bank flip, alarms) | `S8/run_<k>of<n>/` |
| **replay** (`--stage S8C`, first) | every row whose phase-A run published `sat`: its results-file witness through the organisers' OWN checker code on emulated x86-64 (qemu `-cpu Haswell-v4`, ORT 1.26.0, python 3.12.13: `checkv1.py` for VNN-LIB 1.0 rows, `driver.py --mode check` for 2026 VNN-LIB 2.0 rows, then `driver.py --mode ort` on the same feeds, bitwise-compared). Each batch first runs the plan's witness CONTROLS through the same code: a good witness must PASS and a bad one must FAIL, or no row of the batch is judged -- and a row's replay counts only beside a PASS and a FAIL control of ITS OWN checker kind (`v1-2025`, `v1-2026`, `v2-2026`), each re-derived from its own checker record (review: a v1 control proves nothing about the v2 code path). One `ny_box.sh build` hold. | `S8C/x86/` |
| **re-proof** | every row whose phase-A run published `unsat`, unless it is banked `unsat` AND the plan's ledger holds a certificate for its instance that re-verifies now: ONE Proof-profile run (tag `S8R_*`, same overlay/flight/alarms plus `^Result: sat$` as an alarm: a sat there is a flip, stopped at once) | `S8C/run_reproof/` |
| **check** | `<pin> proof-check <bundle> --onnx <row onnx> --vnnlib <row vnnlib> --report ... --budget-s <plan>` on every published bundle, one `ny_box.sh build` hold each; the runner appends `CERT_CHECK_RAN rc=<r>` | `S8C/check/` |
| **classic sub-moat** | every re-proof the checker did not ACCEPT: the classic 3 Competition + 1 Proof reps (tag `S8F_*`) under m0's per-row profile rules | `S8C/run_classic/` |

**ACCEPT** (`certlib.verify_certificate`) is ALL of: the report parses with `ny-proof-check-report/v1`; its onnx and
vnnlib sha256 are the ROW's files; its bundle is THIS re-proof's bundle (path and the sha256 of the file on disk); the
bundle's own header names the same input hashes, the `proof` profile and the pin's commit; the checker build names the
pin's commit; the file is byte-identical to the one the checker's `PROOF_CHECK_DONE` line hashed; `verdict=verified`,
`exit_code` 0; a non-empty cover whose every row is claimed AND confirmed; and the checker's OWN process says so too:
its DONE line `rc=0 verdict=verified` and the runner's `CERT_CHECK_RAN rc=0` (`certlib.check_certificate`).
`unconfirmed` (an opaque claim, a budget cut), `invalid`, `malformed`, `unsupported`, a checker fault or crash, a
missing bundle, or a Proof run that does not reproduce the unsat, is NO certificate: that row takes the classic
sub-moat. `refuted` is an EMERGENCY whatever the bindings say -- a report about this instance (its onnx/vnnlib sha256
the row's files) from any bundle or build, or the checker's DONE line / exit 21 with no usable report: a counterexample
is never downgraded to a classic fallback (review fix).

**The certified S8 verdict** (`certlib.decide`, inside the stage judge):
- EMERGENCY: any m0 emergency in phase A, the re-proofs or the sub-moats (truth contradiction, bank flip, alarm,
  VERDICT-FLIP, PROFILE-DECIDED-SPLIT in a sub-moat, a cross-tag INSTANCE-FLIP); a `refuted` check; a Proof `sat`;
  a witness the organisers' checker rejects on the certified platform (a WRONG SAT).
- VOID: any run not OK; a follow-up spec missing or not EXACTLY the re-derivation (from phase A + ledger, and from the
  re-proof outcomes); a check not run; a replay missing or INCONCLUSIVE (rig error, ORT replay mismatch, not the x86
  platform, controls that did not discriminate); the capability engaged on no run.
- PARTIAL: a banked verdict LOST (phase A, a Proof re-run, or any classic sub-moat run) above `LOST_TOLERANCE` = 0 (a
  constant); a floor violation (both years, the census rows, not a smoke plan).
- PASS otherwise. The record carries, per row, its class and evidence: the report path + sha256, the bundle sha256,
  the checker verdict and node counts, the sub-moat verdicts, the replay record sha256 and result; `accepted`,
  `classic_covered`, `replayed`, `ledger_covered`, `lost`.

The auditor's `verify_stage_record` re-derives the whole record through the same judge: every judged.tsv line (phase
A AND the follow-ups) from its run's artefacts, every certificate from the report, bundle and row files on disk, every
replay from its record, the checker's own result record and the controls file. An edited report, log, replay record or
follow-up spec derives nothing (selftest `test_runner_certified_s8_end_to_end_and_the_record_derives`). The checker's
result record must be the replay's OWN (beside `replay.json`; v1: the witness sha256, the model, the property and the
year's scoring directory; v2: this replay's job id, the x86 platform, no injected outputs), and every control is
re-derived from ITS own record (bound to the plan's control witness) rather than read off the controls file's `got`
(review fixes: a passing control's record, or another row's, could stand in for a row's before; selftests
`test_review_replay_counts_only_with_its_own_checker_record`, `test_review_controls_rederive_from_their_own_records`).

### The argument: why independent checking replaces repetition

The moat exists to catch three things. Take them one at a time.

**(1) A false UNSAT the capability makes reachable.** A false unsat is an `unsat` on an instance that has a
counterexample. Repetition in the classic S8 can only see one when something ELSE says sat: field truth, a banked sat,
another rep or the Proof profile stating sat (VERDICT-FLIP / PROFILE-DECIDED-SPLIT). On a row where no run and no
record says sat, a false unsat the binary states consistently is INVISIBLE to the classic moat, however many reps it
runs. The certified mode asks a different question of every unsat the armed binary states: is the INSTANCE unsat? An
ACCEPT from `ny proof-check` -- an independent checker with its own ONNX and VNN-LIB readers and its own
outward-rounded arithmetic, which re-derives every bound the bundle claims and checks that the proved regions cover
the property's whole input region for every disjunct -- is a proof that it is. If the instance is unsat, NO run can
state a false unsat on it: not this rep, not the two classic reps the certified mode does not run, not a nondeterministic
re-run. So for an ACCEPTed row repetition adds nothing to (1), and the certified mode is strictly STRONGER than the
classic one there (it decides the question the classic moat can only sample). The certificate is about the instance,
not the run, which is also why a ledger certificate from an earlier seal (any pin) covers a banked-unsat row: it is
re-verified against today's files at every judgement. The rows the checker cannot ACCEPT -- opaque claims (the
sequential engine, GPU DomainList, MIP, lowdim fold, the legacy doors, frozen-dual classes with a compact state:
b2-lane-emitters / b2-route-emitters gaps), budget cuts, unsupported ops -- get the classic treatment, recorded per row;
nothing is assumed.

**(2) A wrong SAT.** A sat is wrong when its witness does not violate the property on the organisers' checker. The
classic moat never replays a witness (it only compares sat with field truth and the bank), and an arm64 replay is not
the checker: `smart_turn` instance_44 replays to Y = 0.5 on certified x86 and was banked sat on arm64 (memory
`ny-x86-parity-arm64-sat-adjudication-is-not-valid`). The certified mode replays EVERY armed-sat witness on the
organisers' own code on the checker's platform, with controls proving the replay can fail. That is strictly stronger
than the classic moat for every sat it observes.

**(3) A lost banked verdict.** Completeness, not soundness: every banked sat/unsat the armed run does not reproduce is
LOST, reported per row, and blocks PASS above tolerance 0. A classic sub-moat's lost run counts the same way, and so
does a Proof re-run that does not reproduce a banked verdict (review fix: the classic S8 counts its lost Proof rep; the
fallback it triggers must not make the certified mode more lenient).

**Profile divergence is checked where it matters.** On every re-proved row the Proof profile must reproduce the armed
Competition unsat: a Proof `sat` is an EMERGENCY (stopped at once by the flip alarm), a Proof timeout/unknown is a
divergence and sends the row to the classic sub-moat, where m0's PROFILE-DECIDED-SPLIT rule applies with 3
Competition reps.

### What the certified mode does NOT cover (each is a stated gap, not an oversight)

- **Rows the checker cannot ACCEPT fall back to the classic sub-moat** (3c + 1p): they cost MORE than a classic row
  (1 + 1 + 4 runs). With frozen-dual armed on real cifar100/tinyimagenet rows, a class the FD tier closes from a
  compact state streams only an opaque claim (b2-route-emitters gap), so those rows fall back until the full slope
  table is emitted. The saving is exactly the ACCEPTed and the replayed rows; the ledger makes it permanent across
  seals and pins.
- **One sample per row for NONDETERMINISTIC behaviour on rows the armed run did not decide.** A false unsat that
  appears in 1 of k runs on a field-SAT or banked-sat row is sampled once here, 4 times classically (each would be a
  truth contradiction / bank flip, an EMERGENCY, in either mode). The dedicated hunt for false unsats on truly-SAT rows
  is S7 (attacks OFF, the capability engaged, any unsat = EMERGENCY), which S8 still gates on. Likewise a wrong sat
  that appears only in an unrun rep, and a lost verdict that appears 1 in 3 (S6 measures completeness with >= 3 reps on
  marginal rows).
- **Mode identity (section 1.7) only on the re-proved rows.** The Proof profile's own verdicts on rows the armed
  Competition run did not call unsat are not run. Competition is what is scored; a Proof-profile defect elsewhere
  matters to banking (the Proof bundle rule), not to this seal.
- **The checker's own soundness is the trusted base** (B2: own readers, outward rounding, adversarially reviewed and
  mutation-tested; it ships in the same binary). It certifies the REAL-arithmetic property; an f32 ORT counterexample
  inside a sub-ulp margin (the knife edge) is outside it, as it is outside every sound verifier: S7's knife-edge rungs
  and the field truth on every row remain.
- **The x86 rig is trusted as measured** (100/100 smart_turn, 67/67 ACC, 38/38 parity witnesses): its files are
  sha-pinned in the plan, its driver's constants checked against the plan at step 0, and every batch carries
  discriminating controls; a paired-model (relational) row's witness is not replayable by it (INCONCLUSIVE: the seal
  waits).
- **Controls are per checker kind** (review): `certlib.DEFAULT_RIG`'s four scratchpad controls are `v1-2025` and
  `v1-2026` (checkv1), and 13 of the 29 scored 2026 categories -- cifar100_2024 among them -- are VNN-LIB 2.0 rows
  scored by the `v2-2026` path (`driver.py --mode check`, the official `counterexamples_v2`). A plan whose banked-sat
  S8 rows need a kind its rig cannot discriminate is REFUSED (it could never PASS), and S8C's step 0 refuses an
  armed-sat row of such a kind before any box wait. `DEFAULT_RIG` now carries a measured `v2-2026` pair (the two
  witnesses in `scripts/seal/x86_controls/`; the review note below).
- **A ledger certificate is trusted as re-verified** (report, bundle and row files byte-identical, ACCEPT now). The
  ledger DECLARES the checker build that vouched (its `checker_build` column must name the report's `commit=`), and the
  S8 record carries that commit per ledger-covered row (`checker_commit`); the report is NOT re-checked by today's
  checker and is not bound to its DONE line, so a ledger certificate rests on the checker build it names. Re-checking
  ledger bundles with the pin's checker (no verifier run) would close this; it is not built.

### Commands

```bash
python3 $W/scripts/seal/seal.py plan frozen-dual-bab --pin $PIN --out $OUT --census run --s8-mode certified \
    [--cert-ledger $LEDGER] [--x86-rig rig.json] [--checker-budget-s 900] --shards 4
# S5, S7 as usual; then phase A per shard:
for k in 1 2 3 4; do python3 $W/scripts/seal/seal.py run --out $OUT --stage S8 --shard $k/4; done
#   each shard ends SEAL_DONE rc=5 ... status=VOID ("certification pending") until S8C has run
python3 $W/scripts/seal/seal.py run --out $OUT --stage S8C --check   # step 0 only: phase A clean, rig, checker
python3 $W/scripts/seal/seal.py run --out $OUT --stage S8C           # the follow-ups; SEAL_DONE carries the S8 verdict
grep -E '^SEAL_DONE ' $OUT/seal_S8C.log | tail -1                    # cert_accepted= cert_classic= cert_replayed=
python3 $W/scripts/seal/seal.py cert-ledger --ledger $NEW_LEDGER $OUT  # ACCEPTed certificates for the next seal
```

S8C's step 0 refuses (rc 97) before any box wait unless phase A is CLEAN (every row's armed run judged OK, no
emergency), the x86 rig is the plan's (file sha256s, the driver's constants, the control witnesses copied into the plan)
whenever an armed-sat row needs it, every armed-sat row's checker kind has a PASS and a FAIL control, and the pin
carries `proof-check`. A resumed replay batch that finds a FAILED record kept from an earlier batch stops with
`X86_REPLAY_DONE rc=3` (review fix: it reported rc=0 and let the runner spend re-proofs and sub-moats). S8C resumes: a
written follow-up spec must re-derive (refused otherwise), a judged run or a ran check is kept, a replay with a final
record is kept. Its `SEAL_DONE` carries one `rc=` and `phase=replay|reproof|check|classic` on an EMERGENCY.

**Measured (smoke, 2026-10-08, eng13 pin `9cedffdce`, shipped presets).** Three moat rows through the real
runner, box and rig: 2025 cifar100 idx_7704 (banked unsat, margin-row lane), 2025 acasxu run2a_5_3 prop_2 (banked sat)
and run2a_2_7 prop_3 (banked unsat, a legacy lane). Phase A 3/3 OK. The sat witness replayed `correct` on
x86_64/Haswell-v4 with an ORT replay match, all four controls discriminating (24 s). Both re-proofs published unsat and
both were REFUSED by the checker -- cifar100 `unconfirmed`, 98/99 nodes: one frozen-dual prune node `needs_fd_check`
with no FD hint streamed (FD is default-on in the shipped cifar100 preset: the compact-state gap); acasxu `malformed`:
the legacy lane finalises a bundle with no `sites` record -- so both took the classic sub-moat (8/8 runs unsat). The
verdict is PARTIAL with the single reason "a SMOKE plan"; 13/13 runs OK, 0 emergencies, 0 lost; the record re-derives
with 0 differing fields. On REAL data the ACCEPT path was exercised on the one verified report on disk (b2gate, cifar100
idx_6659, 105/105 nodes, 99 cover rows: ACCEPT as a ledger certificate, refused when bound to another pin). So with the
shipped presets a frozen-dual certified S8 saves box time only on ledger-covered, sat and undecided rows until the FD
prune trace is complete under Proof.

**Review (review/certified-moat, 2026-10-08).** The smoke's record above re-derives with 0 differing fields and 0
run mismatches under every review fix: its real `v1-2025` replay record binds to its own `checkv1` record (witness
sha256, model, property, `SCORING-ZERO-TOL`) and its four real controls re-derive from their own records. The real
b2gate ledger certificate still verifies and is recorded with `checker_commit=96e820134`. The two `v2-2026` control
witnesses in `scripts/seal/x86_controls/` (acasxu 2.0 prop_2, run2a_1_2: the scratchpad v1 control's point as a
VNN-LIB 2.0 assignment, and the same with `X_0 = 5.0`, outside the input box) were MEASURED through the real rig
(`x86_replay.replay_core`, kind `v2-2026`, one 9 s `ny_box.sh build` hold, `V2CTL_DONE rc=0`): pos -> `correct` ("all
assertions hold with input_tolerance=0.0, output_tolerance=0.0") with an ORT replay match, neg -> `spec_not_violated`
("assertion 0 failed ... (input)"), both on x86_64 / Haswell-v4 / ORT 1.26.0, and both real records pass the v2
binding (job id, `x86:Haswell-v4`, no injected outputs). They are in `DEFAULT_RIG`, which now discriminates all three
kinds (selftest `test_review_default_rig_discriminates_every_checker_kind`). A plan made before this keeps its pinned
rig (v1 controls only) and so cannot judge a v2 sat.

**Smoke plans.** `plan <cap> --stages S8 --smoke-rows 2025/cifar100_2024/123,...` runs exactly the named moat rows,
S8 UNGATED, to exercise the pipeline on a real pin. A smoke is never evidence: its floor reads "a SMOKE plan"
(PARTIAL at best), `verify_stage_record` derives nothing from it, `seal record` and `sr_seal` refuse it.

**Alarm scoping (fix, found here).** An m0 tier's alarms are "the capability CLOSED this" markers (frozen-dual's
`verdict=all_closed`, the margin-row lane's `UNSAT` line): an emergency on a SAT row, the expected outcome on an
unsat one, and m0's own A/B arms them on sat rows only. The seal applied them to EVERY row, so frozen-dual's S5 (whose
arrival rows are banked-UNSAT rows the prover must close), S6 and S8 raised a false EMERGENCY on its first proof. They
are now `sat_alarms`, carried by rows whose field truth or bank says sat (and every NEGCTL / KNIFE row); the every-row
alarms are the defect signals plus the registry's `seal.alarms`.

## The admission census (#admission-census)

`ny vnncomp census --manifest <corpus> --out <tsv>` (crates/ny-cli/src/router/census.rs) writes, for every row of a
corpus manifest (`# ny-admission-census-corpus/v1`, `row_id<TAB>onnx<TAB>vnnlib<TAB>budget`), CR-RS's OWN decision for
each routable capability as if that capability were sealed: the scored entry point's router facts (the relational
probe, the plan resolver's single-pass model scan, the on-disk size, through `router::policy_input`), the router's
`commit` with ONE capability pretended sealed under the widest recordable envelope and every other route removed
(`ny_core::router::route::census_decision`: precondition, parent admission, envelope; never a slot, pin or seal
reason), and -- where the router defers an inline parent admission -- the parent lane's OWN structural admission
(today the margin-row lane: `lane_model_admission` / `lane_property_admission`, a necessary condition of both lane
entries). Columns: `admit` (the census decision), `router` (the decision without any parent probe), `parent_probe`,
`reasons`, `facts`, `ops`, `spec` (#admission-precision, census schema v2: the row's archetype and routing shape as
the router read them at start — `beta_crown::spec_shape::read_spec`, the lanes' own parser and extractor), `form`.
Gzip files are decompressed (content-addressed, `<tsv>.work/`) so the facts are the competition form's. The header
binds the binary's sha256, the router/census schemas, the registry version, the corpus sha256, the host, the
envelope and the property read (`spec_read`); the trailer is the sha256 of every byte above it. No verification runs.

`seal.py census --pin <bin> --out <dir> [--box scripts/ny_box.sh]` builds the moat corpus (m0 moat rows, both years,
every scored category, one entry per instance) and runs it; `seal.py plan ... --census run|<tsv>` makes S8's rows
exactly the census-admitted moat rows (their categories are armed by the overlay; a category without a preset is a
refusal) and records the census, its corpus and the binding in the pre-registered plan.

## Commands

```bash
W=<home>/ny-wt-<branch>          # a worktree holding the capability's commit
OUT=<home>/ny/scratchpad/seal/<cap>_<pin>   # new dir; never reused across pins
python3 $W/scripts/seal/seal.py plan <capability-id> --pin <home>/ny-pins/ny_<pin> --out $OUT --shards 4
python3 $W/scripts/seal/seal.py run --out $OUT --stage S5 --check       # step 0 only, no box
python3 $W/scripts/seal/seal.py run --out $OUT --stage S5               # launches; SEAL_LAUNCHED unit=... pid=...
grep -E '^SEAL_DONE ' $OUT/seal_S5.log | tail -1                        # anchor chains on '^SEAL_DONE rc=0 '
python3 $W/scripts/seal/seal.py run --out $OUT --stage S7
for k in 1 2 3 4; do python3 $W/scripts/seal/seal.py run --out $OUT --stage S6 --shard $k/4; done
for k in 1 2 3 4; do python3 $W/scripts/seal/seal.py run --out $OUT --stage S8 --shard $k/4; done
python3 $W/scripts/seal/seal.py record --out $OUT --dry-run            # then without --dry-run (in a worktree)
python3 $W/scripts/engine_program_audit.py --tree $W ...               # S5..S8 and SR are now derived
```

Other plan options:
- `--stages`
- `--categories` (only narrows the admission scope, never widens it; a narrowed S8 judges PARTIAL)
- `--years` (an S8 without both years judges PARTIAL)
- `--router-shadow auto|on|off`. `auto` adds `router.mode: shadow` for a routable capability when the pin carries
  `[router] cap=`, for the admission receipt; shadow changes no verdict path.
- `--arrival-rows`
- `--negctl-per-category`
- `--s6-uniform`
- `--s8-reps-c` / `--s8-reps-p` (below 3 / 1 an S8 judges PARTIAL)
- `--seed`

**The plan is pre-registered once.**
- `plan.json` is written with `plan.json.sha256`, and every spec and edited overlay is sha-pinned.
- Step 0 refuses an edited plan or file.
- `configs/` comes from the **pin's commit** (`git archive`), not from the working tree. A newer tree's yaml can carry
  keys the pin refuses: measured on 2026-10-01, the eng7 pin refuses int/eng9's `cifar100_2024.yaml`
  (`unknown field frozen_dual_refine_steps`).

## The runner (`seal_run.sh`)

Step 0 runs before any box wait and fails closed. It checks:
- the tools, and no `$OUT/EMERGENCY`;
- that no `NY_BOX_*` other than `NY_BOX_WAIT_LOCK`/`NY_BOX_WAIT_QUIET`/`NY_BOX_QUIET_GRACE` is set: a raised load cap,
  a test `/proc/stat`, another lock dir or an inherited fcntl lock changes what every measurement means (rc 95);
- the plan and the stage gates. S6 and S8 need S5 PASS and S7 PASS or PARTIAL, re-derived, on the same pin. S7
  needs S5;
- zero `NY_*` and none of `GPU_AVAILABLE`/`RAYON_*`/`AY_MILP_*`/`NYCERT_*`/`CUDA_VISIBLE_DEVICES`/`PROBE_*`;
- that every `memory.max` on the cgroup path is at least 96 GiB, and that `RLIMIT_AS` is unlimited;
- the pin: its sha256 against the plan, a renamed byte-identical copy (`probe_seal`), and `--build-id` (commit,
  features, `dirty=false`, exe). `PIN_LOCK` binds the whole `SEAL_OUT` to one pin;
- the capability's markers in the pin's strings;
- that **every overlay yaml loads through the pin's own preset loader** (`deny_unknown_fields`). A NEGATIVE control,
  an unknown key added next to the first armed key, must be REFUSED **by name**, which proves that the probe
  validates fields;
- the pin's own resolver, through m0's step 0 (`vnncomp plan` per category and configs dir). A NEGATIVE control, an
  empty configs dir, must NOT resolve the overlay;
- m0's whole step 0.

`--launch`:
1. Runs step 0.
2. Snapshots `scripts/seal` and `scripts/m0` into `$OUT/harness_<S>_<stamp>/`, because bash reads a running script
   by offset.
3. Runs `systemd-run --user --property=MemoryMax=96G --property=WorkingDirectory=$SEAL_NY` with
   PATH/HOME/RUSTUP_HOME/CARGO_HOME, `NY_BOX_WAIT_LOCK=21600` and `NY_BOX_WAIT_QUIET=21600`.
4. Verifies MainPID, the unit cgroup, MemoryMax and the environ.

`--run` binds the effective cgroup memory cap (the tightest `memory.max` on its path) to `$OUT/MEM_LOCK` on its first
run and refuses (rc 86) any later run of the same `SEAL_OUT` under another cap: the cap's VALUE moves the CROWN dense
clamp, so an A/B or a moat straddling two caps is confounded (CLAUDE.md). It writes a heartbeat line every 10
minutes. It runs the m0 runner with:
- `M0_STOP_FILE=$OUT/EMERGENCY`, so an emergency in any shard stops every other shard before its next run;
- `M0_OUT_ROOT=$OUT`;
- one `ny_box.sh measure` per run, under the competition backstop, with stdout and stderr kept per run.

When the m0 runner stops on an emergency (exit 3) the runner writes `$OUT/EMERGENCY` BEFORE judging, so every other
shard stops at its next run even if the judge then fails (the DONE line then reads `status=EMERGENCY
judge=failed(rc:<r>)`). It then judges the stage and writes `$OUT/<S>/STAGE.json`. `seal record` refuses to write
anything from a `SEAL_OUT` that holds an `EMERGENCY` file.

**DONE line:** `SEAL_DONE rc=<stage rc> WORST=<max(stage, m0)> stage=<S> shard=<k/n> status=... m0exit=...`. It
carries exactly one `rc=`, and the log is appended, so count markers first.

| rc | meaning |
|---|---|
| 0 | PASS |
| 4 | PARTIAL |
| 5 | VOID or not run |
| 3 | EMERGENCY. `$OUT/EMERGENCY` is written and every stage and shard refuses. |
| 9x | a failed precondition: nothing was measured |
| 10 | an empty shard |

## Known limits

- **The op signature comes from an OFFLINE scan** (`offline-scan/v1`): a dependency-free ONNX protobuf walk with
  constant propagation, and an archetype classifier over m0's VNN-LIB reader. The router's in-binary facts (R1) carry
  only conv presence (`ops_complete=false`) and no spec archetype, so a receipt-only envelope is not possible yet.
  The seal prefers the router's facts once they are complete, and it disputes a row whenever they see more than the
  scan.
- **S8 scope without a census is categories**, not the capability's own admission predicate. With a census it is the
  census-admitted rows. The census is the router's decision PLUS the parent lanes it can ask: a parent whose admission
  is not yet callable (input_split, relu_split.graph) is deferred exactly as the router defers it, and the router has
  no spec archetype at start, so for capabilities under those parents the census admits every row whose facts the
  router can read. A pin built before #admission-census has no `vnncomp census`, so its seal cannot use one.
- **A census is per HOST and per PIN.** The router's precondition reads the host (`AdmitWhen.device`,
  `min_threads`), so a census describes the host it was made on: make it where the router will run, or on a host
  that admits at least as much (today only `wave-beta-bab-cuda` reads the host -- 678 moat rows on the GB10, 0 on a
  declared m5 -- and no routable capability declares `min_threads`). The census also embeds the PIN's parent-lane
  admission (the margin-row probe is that binary's `extract_twin_spec`, compiled-head check and
  `parse_vnnlib_robustness`): a LATER binary that widens the lane's admission starts the lane, and so the routed
  consumers, on rows the pin's census took out of the moat. Re-run the census (and re-seal) on any binary that
  changes one of those functions; nothing enforces this yet.
- **A default-on key** cannot be made absent by deleting it, so its S6 control engages and is VOID (`!need`): fail
  closed, but unmeasurable until the control writes an explicit off value.
- **The knife-edge is not expressible on classification rows** (relational-only clauses), as in the m0 README. Such
  S7s pass on the controls with `knife=not-expressible`.
- **S3 for this pipeline:** the auditor indexes Rust `#[test]` fns only. `seal_selftest.py` has to be added as a gate
  leg, like the m0 and audit selftests.
