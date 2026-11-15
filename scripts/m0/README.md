# M0 arming harness

Tooling for ENGINE_PROGRAM §2 **M0 "arm what is measured"** (`docs/engine_program/ENGINE_PROGRAM.md`), which
gates every arming: tier-engaged negative controls, knife-edge ladders, three pre-registered A/Bs and a
both-year moat in both profiles.

The generators never build, run an ny binary, or touch the box; they write configs copies, VNN-LIB files and
runner specs. Only `moat_both_profiles.sh` measures, on a gated pin.

| file | M0 item | what it does |
|---|---|---|
| `negative_controls.py` | 2 | Writes sat-finding-OFF config overlays with the tier armed through its typed key, plus a spec of truly-SAT control rows. Each row requires the tier's own engagement line. |
| `knife_edge.py` | 3 | Rewrites a VNN-LIB threshold to `f(x*) ∓ δ` for δ ∈ {1e-3, 1e-6, 1e-9, 1e-12}, using a banked witness x*. Every rung is a SAT instance that must never be proven unsat. |
| `ab_arms.py` | 4 | Writes the three pre-registered zero-build A/Bs as exact arms (§3). |
| `moat_rows.py` | 5 | Builds the both-year verdict-row moat. Instances, bank and field truth are aligned by index, with the key asserted at every index. |
| `moat_both_profiles.sh` | 2–5 | Runner: `systemd-run --user` + `ny_box.sh measure`, output captured per run, `--proof-profile competition` and `proof`. EMERGENCY exits 3. |
| `m0_judge.py` | – | Makes every decision the runner acts on: `validate`, `merge`, `markers`, `rows`, `judge`, `profiles`, `prior`, `summary`. |
| `m0lib.py` | – | Shared code: VNN-LIB 1.0/2.0 reader with offsets, witnesses, exact knife-edge arithmetic, verified yaml overlays, the attack-off table, row alignment, the spec, the judge. |
| `m0_selftest.py` | – | 65 named unittest tests: 62 in the default lane, including 18 end-to-end runner tests against a stub pin and a stub `ny_box.sh`, and 3 in the `--real-data` lane. |

```
python3 scripts/m0/m0_selftest.py              # default lane: `test_<name> (...) ... ok` per test; last line M0_SELFTEST_DONE rc=0 ... lane=default
python3 scripts/m0/m0_selftest.py --real-data  # real-data lane: the 3 tests that read the main tree and field results
python3 scripts/m0/m0_selftest.py --bash /bin/bash  # default lane with the runner under the host's stock shell
```

The runner must run under **bash 3.2**, macOS's `/bin/bash`, like ny's other shell tools: no `mapfile`, no
associative arrays (bash 3.2 turns `declare -A` into an indexed array with arithmetic subscripts:
`docs/DEFECT_REGISTER_2026-07-31.md` D1), no bash-4 expansions, and empty arrays expanded with the `+` guard.
Two Cargo checks hold it there. `targo --unverified test -p ny-test-utils --test source_policy m0_` (default lane,
gate check [a]) scans the runner, line by line, for the constructs bash 3.2 cannot run or runs differently and for
the `date` spellings GNU and BSD date do not share (anything but `-u` and a plain `+FORMAT`); it needs no shell, so
it is a red on Linux too, where `/bin/bash` is bash 5, GNU date accepts `-Is`, and the self-test cannot see either.
A plain `"${a[@]}"` passes it only for the arrays on its `M0_NEVER_EMPTY_ARRAYS` list (`ROWS`, `props`, `envs`): a
new array is either guarded or added there with the reason it cannot be empty. The scan reads quoted text as code
and refuses a construct that only appears in a string; reword such a line. `NY_TEST_PYTHON=<venv python> targo
--unverified test -p ny-test-utils --features python-tools --test python_tooling_black_box m0_` (gate check [g])
runs the default lane with `--bash /bin/bash` and requires every test's `... ok` line by name. When the host's own
memory envelope makes step 0 refuse (rc 86, see below), that contract fails naming the refusal instead of reading
as a shell red. `--bash PATH` takes any bash: the runner also passes under bash 5.3 on a BSD userland
(2026-10-09); its first run with GNU date and coreutils `timeout` is the box's.
The self-test needs requirements.txt's PyYAML; the runner's `python3` is the self-test's own interpreter. The
runner's timestamps keep GNU `date -Is` format (`2026-10-04T12:00:00-07:00`) on every host. On a host without
`timeout` (macOS has none) the runner tests use a stand-in that keeps GNU timeout's contract (process group,
SIGTERM at expiry, rc 124); the DONE line's `timeout=` field says which ran.

The 3 real-data tests (class `RealData`) are not part of the default lane, and selecting `--real-data` makes a
missing input a failure that names the path (the source policy forbids skips): all three need
`<home>/ny/benchmarks/vnncomp2026/benchmarks`, and two also need
`<home>/ny/reports/measured-2026/bank_manifest.json` and `~/vnncomp2026_results/SCORING/results.txt`.
16 of the 18 runner tests reach the real step-0 memory check, so on Linux they fail with rc 86 on a host whose
cgroup `memory.max` is unreadable (e.g. cgroup v1). Off Linux the cgroup walk does not apply: there is no cgroup
hierarchy to read and the dense budget's headroom term is `None` there. `RLIMIT_AS` applies on every host: on
macOS ny's strict envelope (`process_memory_envelope`, `#darwin-memory-envelope`) reads it too, so the runner
refuses a finite `RLIMIT_AS` there for the same reason as on Linux. Neither branch reads installed memory: off
Linux the dense budget is a quarter of it, capped at 12 GiB, so a Mac with less than 48 GiB starts below the
ceiling the 96 GiB rule protects on the box.

## 1. The protocol, in commands

All paths are examples. `W` is the arming candidate's isolated worktree (the arming commit), `S` a new directory
under `<home>/ny/scratchpad/m0/` (`M0_OUT` must live under `$M0_OUT_ROOT`, default `$M0_NY/scratchpad/`;
`M0_NY` defaults to `<home>/ny`). `M0_STOP_FILE`, when set, stops the runner before its next run while that
file exists (a caller running several specs points it at its own EMERGENCY record; scripts/seal/ does). The generators read benchmarks and banks from the main tree (`--ny-root`, default
`<home>/ny`) and field results from `--truth-2026` (default `~/vnncomp2026_results/SCORING/results.txt`)
and, in `moat_rows.py`, `--truth-2025` (default `~/vnncomp2025_results/SCORING-ZERO-TOL/results.txt`). The
`scratchpad/w8b` and `scratchpad/w10b` witnesses are untracked run outputs on that host.

```bash
W=<home>/ny-wt-<arming>; S=<home>/ny/scratchpad/m0/<pin>
# (2) tier-engaged negative controls: one sat-finding-OFF configs copy per tier + negctl_spec.tsv (14 rows)
python3 $W/scripts/m0/negative_controls.py --configs $W/configs --out $S/negctl \
    --tier frozen-dual --tier cgan-child-zono --tier fwd-root-screen --tier bi-window-joint-refresh
# (3) knife-edge ladders on the threshold-property controls (each inherits its control's configs and tier)
python3 $W/scripts/m0/knife_edge.py --from-spec $S/negctl/negctl_spec.tsv --row NEG_cz_r4 \
    --witness $W/reports/measured-2026/cgan2026_artifacts/witnesses/0004-cGAN_imgSz32_nCh_3_prop_0_input_eps_0.070_output_eps_0.078.counterexample \
    --out-dir $S/knife/r4 --spec-out $S/knife_spec.tsv
python3 $W/scripts/m0/knife_edge.py --from-spec $S/negctl/negctl_spec.tsv --row NEG_cz_r9 \
    --witness $W/reports/measured-2026/cgan2026_artifacts/witnesses/0009-cGAN_imgSz64_nCh_1_prop_1_input_eps_0.025_output_eps_0.030.counterexample \
    --out-dir $S/knife/r9 --spec-out $S/knife_spec.tsv
python3 $W/scripts/m0/knife_edge.py --from-spec $S/negctl/negctl_spec.tsv --row NEG_frs_14lres_p1 \
    --witness <home>/ny/scratchpad/w8b/v2/FRS_14lres_p1_frs_satctl.res --out-dir $S/knife/14lres --spec-out $S/knife_spec.tsv
#     (likewise NEG_frs_14lnon_p1 / 118lnon_p5 / 300base_p2 / 300lres_p2 with their W8B *_frs_satctl.res witnesses)
# (4) the three pre-registered A/Bs
python3 $W/scripts/m0/ab_arms.py --configs $W/configs --out $S/ab
# (5) the both-year moat, on the ARMED configs (the moat licenses the arming, so it runs what would ship)
python3 $W/scripts/m0/moat_rows.py --configs-dir $W/configs --out $S/moat.tsv --years 2026,2025 \
    --categories cifar100_2024,tinyimagenet_2024,cgan2026,ml4acopf_2024,challenging_certified_training_2026
# one spec, one step-0 check (no box), one launch
python3 $W/scripts/m0/m0_judge.py merge --out $S/m0_spec.tsv --spec $S/negctl/negctl_spec.tsv \
    --spec $S/knife_spec.tsv --spec $S/ab/ab_spec.tsv --spec $S/moat.tsv
export M0_PIN=<home>/ny-pins/ny_<pin> M0_PIN_SHA16=<16 hex> M0_PIN_COMMIT=<40 hex> M0_OUT=$S/run
$W/scripts/m0/moat_both_profiles.sh --check  $S/m0_spec.tsv
$W/scripts/m0/moat_both_profiles.sh --launch $S/m0_spec.tsv    # prints M0MOAT_LAUNCHED unit=... pid=...
# later: the log's own DONE line (never a .res placeholder), then the summary
grep -E '^M0MOAT_DONE ' $S/run/m0moat.log | tail -1
M0_OUT=$S/run $W/scripts/m0/moat_both_profiles.sh --summary $S/m0_spec.tsv
```

A chain that waits on this runner must match the anchored marker `^M0MOAT_DONE rc=0 `, not a bare `rc=`
substring, and must wait for a NEW marker: the log is opened in append mode, so count markers before launching.

**Cost.** The ceiling is Σ budget × runs. The estimate is the same sum using the banked walls.

| spec | rows | runs | ceiling | banked-wall estimate |
|---|---|---|---|---|
| negative controls (4 tiers), 1+1 reps | 14 | 28 | ≤ 3.6 h | – (attacks off: expect most of the ceiling) |
| knife-edge, cgan r4 + ml4acopf 14lres, 1+1 | 8 | 16 | ≤ 3.3 h | – |
| A/Bs (a, b, c), 3 competition reps | 57 | 171 | ≤ 4.8 h | – |
| moat, the 5 armed families, both years, 3+1 | 609 | 2436 | ≤ 151 h | ≈ 27.5 h (≈ 14 h at 1+1) |
| moat, every banked category, both years, 3+1 | 5648 | 22592 | ≤ 920 h | ≈ 148 h |

Moat row counts come from specs generated against `cf2a2469b` configs and banks. The moat takes every banked
sat/unsat row **and every field-SAT row whatever ny banked** (`moat_rows.select_row`): 64 such rows on the 2026
banks (tiny 1831 and 5529, cct 2, soundnessbench 7, smart_turn 50, traffic_signs 2, acc 1), banked
timeout/unknown. On those rows an armed tier runs to the end on a truly-SAT instance with nothing to
short-circuit it: the only place a false unsat can appear that a banked-rows-only moat never visits. Their
banked-wall estimate is the full budget. `--banked-only` drops them and is not a full moat.

## 2. What the runner decides

A run is judged (`m0lib.judge_run`) from its own raw log, its **results file** (the scored verdict of record)
and its `<res>.flight.json`. Every run executes under the competition's OS backstop, `timeout $((budget + 10))`
as in `vnncomp_scripts/run_instance.sh`: an overrun is killed where competition kills it and scored from what
the results file held at that instant (`box_rc=124`, `note:killed-at-backstop`), and a hung run cannot hold the
box. Before each run, files an interrupted earlier attempt left at its paths (raw, results, flight record) are
moved aside (`*.stale.<ns>`), never read as this run's.

- **EMERGENCY**: the runner prints a summary, writes `M0MOAT_DONE rc=3 WORST=3 STOPPED_AT=...`, and exits 3.
  These checks run **before** and **independently of** engagement, so an unsat on a SAT row is an emergency
  even when the tier line is missing. Every decided verdict the run stated counts: each `Result:` line **and**
  the results file.
  - `TRUTH-CONTRADICTION`: a stated verdict contradicts field truth. truth=sat when a counterexample scored,
    unsat when a verification scored and none did. NEGCTL and KNIFE rows are sat by construction.
  - `BANK-FLIP`: a banked sat became unsat, or the reverse.
  - `SELF-CONTRADICTION`: one run stated both sat and unsat (log vs results file, or two `Result:` lines).
  - `ALARM[...]`, on **every** row: a lane certified UNSAT while ONNX Runtime had reconfirmed a counterexample
    (`EARLY EXIT REFUSED — this instance holds…`, ``REFUSING to publish `unsat` — this instance holds…``, or a
    flight `sat_precedence` event reading REFUSED). #sat-precedence then publishes `unknown` or the sat, so the
    verdict hides it, but one of the two answers is wrong whatever the truth.
  - `ALARM[...]`, on every row whose truth or bank is sat (MOAT rows included, since a lane's unsat can lose the
    publication claim and never reach the verdict): `margin-row BaB (concurrent): UNSAT`, frozen-dual
    `verdict=all_closed`, fwd-root `union_refuted == total`. Plus the tier's own alarms and, when a witness is
    supplied, `[frozen-dual] class=<witnessed class> ... verdict=closed`.
  - Across the runs of one row (`judge_profiles`), over **every** judged line that names the row's instance,
    VOID runs and superseded re-judgements included (a run with the tier dark still stated sat or unsat about
    this instance); only a wrong-instance run is excluded:
    - `VERDICT-FLIP`: any sat↔unsat difference, in either profile or any rep.
    - `PROFILE-DECIDED-SPLIT`: §1.7 mode identity. A verdict Proof decides must be decided the same way in
      ≥ ⌈2n/3⌉ of the n valid Competition reps (2 of 3; 1 of 1).
  - Across tags (`judge_instances`): `INSTANCE-FLIP`, a sat on one tag and an unsat on another for the same
    (model, property), e.g. a NEGCTL row, an A/B arm and the moat row of one instance.
  - An EMERGENCY is **sticky**: a later re-judgement of the same run never removes it from the summary, and
    step 0 refuses a relaunch (`prior-emergency`, rc 3) while the record holds one.
- **VOID**: the run proves nothing, and it is never a pass. Causes:
  - no `Result:` line (unless the backstop killed the run: then the results file is the scored verdict);
  - the results file is missing or empty, or names another verdict than the log (`res-log-mismatch`), or the
    log carries two different `Result:` tokens;
  - the `Running VNN-COMP instance:` line is missing or names another vnnlib;
  - `Loading preset:` is missing or names another file;
  - no `[proof-profile] mode=<p> source=cli` line;
  - a `need` regex is absent (the tier line), or a `!` regex is present;
  - a flight check fails: the attack masters and every lane lever must read *off*, e.g. `NY_UPFRONT_ATTACK=false`
    from `config`, `upfront_attack_policy ... master=disabled_by_preset`, and `postbab_wrapper_policy ...
    disabled_by_preset=1 route=proof_tail_by_preset` (or `route=disabled_by_preset`);
  - the run's own flight `ambient_env` snapshot (every `NY_*` plus the thread-count variables the process saw)
    holds anything but `OMP_NUM_THREADS=1`: the per-run zero-`NY_*` proof. The runner strips every `NY_BOX_*`
    from the measured process.

  Lines starting `[ny-box` are excluded from every check, because the `[ny-box] cmd:` echo contains the
  command line.
- **LOST**: a banked verdict was not reproduced. This is reported (rc 4), not a stop. **NOTE**
  (`PROFILE-PROOF-SLOWER`): Competition decided where Proof did not. Proof carries extra audit work, and the
  program's rule is one-directional.
- **Runner rc**: 3 emergency; 5 any VOID or not-run run; 4 any LOST; 0 clean; 2, 86, 9x or 127 a failed
  precondition or harness error (a step-0 failure measures nothing).
  - Step 0 runs before any box wait. It checks: the tools; the pin's sha256 prefix, `--build-id` commit,
    features, `dirty=false` and `exe=`; every marker the spec's tiers and alarms need, present in the pin's
    strings; zero `NY_*` other than the box's own `NY_BOX_*`, and none of the non-`NY_` variables ny reads
    that change what it does (`GPU_AVAILABLE`, `RAYON_*`, `AY_MILP_*`, `NYCERT_*`, `PROBE_*`,
    `CUDA_VISIBLE_DEVICES`); that every `memory.max` on the cgroup path, ancestors included, is unlimited or
    ≥ 96 GiB and `RLIMIT_AS` is unlimited (ny's governor reads the tightest of them); the spec; no EMERGENCY
    already in `judged.tsv`; and the binary's own resolver (`ny vnncomp plan`) on one row per (category,
    configs dir), which is what the resolver's answer depends on.
  - A box refusal (rc 75 or `REFUSING TO MEASURE`) is tried up to 3 times. After that the run is left
    un-judged and counts as not-run; a relaunch (`--launch` again, same `M0_OUT`) runs it.
  - A relaunch skips every run that has a `.judged` marker, **including VOID ones**: resume never re-runs a
    VOID run. To re-run VOID runs, launch a spec of just those rows into a NEW `M0_OUT` (never delete a marker
    or a raw log: the old judgement is evidence, and the flip checks read every line).

## 3. The pre-registered zero-build A/Bs (`ab_arms.py`)

Common to all three:
- Rows: FDB6 = cifar100 `c3494`, `c9585` (resnet_medium) and tiny `t6230`, `t5652`, `t4863`, `t8853`, plus
  `t2712` as a positive control.
- 2026 rows, official 100 s budgets, **3 Competition reps per arm and row**, on one pin.
- Typed yaml keys only. Every arm row requires its own observable; a run without it is VOID and is re-run, not
  counted.
- Bank flips and truth contradictions are emergencies, exactly as in the moat.
- The shipped cifar100/tiny yamls already set `margin_row.frozen_dual_bab: true` (with
  `frozen_dual_refine_steps: 4`), so the shipped-yaml control `bc_ctl` arms frozen-dual too.

| A/B | arms (exact yaml) | observable required (else VOID) | metric |
|---|---|---|---|
| **(a) tier share** | `a_ctl`: `margin_row.frozen_dual_bab: true`. `a_share`: the same plus `margin_row.frozen_dual_bab_secs: 1000000000.0` (capped at the 0.75 share, `zono_bab.rs` `FROZEN_DUAL_MAX_SHARE`) | `[frozen-dual] armed=1 objectives=` on every row. Flight: `NY_MARGIN_ROW_FROZEN_DUAL_BAB_SECS` is `0.0@default` (ctl) or `1000000000.0@config` (share) | conversions per row (unsat in ≥ 2 of 3); summary-line `budget_secs=`, `nodes=` and `closed=` |
| **(b) certify-once** | `bc_ctl`: the shipped yaml. `b_rco`: `margin_row.alpha_opt: false` + `margin_row.root_certify_once: true` (certify-once runs after alpha_opt, so arming it alone adds a phase) | ctl: `[alpha-opt] (done\|refuse)` and no `[root-certify-once] armed=1`. rco: `[root-certify-once] armed=1 iters=` and no `[alpha-opt] done iters=`. Levers: `NY_MARGIN_ROW_ALPHA_OPT=false@config`, `NY_MARGIN_ROW_ROOT_CERTIFY_ONCE=true@config` | conversions; wall on rows unsat in both arms |
| **(c) upfront ordering** | `bc_ctl` (shared). `c_after`: `attack.pgd_order: after` with `attack.ny_pgd_order_compat` deleted, so the attack runs after BaB (`Deferred`). `c_cap1`: `attack.upfront_attack_auto_cap_secs: 1` | after: `upfront_attack_policy ... master=contextual route=disabled_by_preset`. cap1: `NY_UPFRONT_ATTACK_AUTO_CAP=1@config` and `auto_wall_cap_secs=1` | conversions and wall on FDB6; **the verdict on the sat rows** |

Rows for (c) add the field-SAT tiny rows `t1831` and `t5529`, which the program names. Both are banked
`timeout`, so (c) can only gain them. (c) also adds the three banked-SAT rows `t1513`, `c1515` and `c8452`,
which are what (c) can actually lose.

Why (c) is not `upfront_attack_probe_first` / `upfront_attack_cap_secs`, as §2 M0 item 4 names it:
- On these rows the upfront lane is admitted as `auto_disjunction` with `binding=auto_wall_cap`,
  `auto_wall_cap_secs=4` [M: `scratchpad/w10b/v2/FDB_t1513_fdb_satctl.res.flight.json`,
  `upfront_attack_execution`]. So `upfront_attack_cap_secs` (forced/typed admissions only) is inert here.
- `probe_first` adds a 5 s probe consult (`UPFRONT_ATTACK_PROBE_CAP`, `vnncomp.rs`) ahead of the ordinary one,
  so it adds seconds rather than moving them. Its pin test
  (`only_the_traffic_signs_preset_arms_the_probe_first_lane`) scopes it to traffic_signs.

**Decision rule for each A/B** (pre-registered; applied to the summary's per-row table):
1. **VOID gate.** The rule applies only when every arm row has 3 non-VOID runs. Otherwise re-run the VOID runs
   (a spec of those rows into a new `M0_OUT`; resume does not re-run a judged VOID run) before reading anything.
2. **Soundness gate.** 0 EMERGENCY and 0 LOST on banked rows, in either arm. Any EMERGENCY stops everything
   (CLAUDE.md).
3. **Adopt** the treatment for a family when it converts at least one FDB6 row in ≥ 2 of 3 reps that control
   leaves undecided in ≥ 2 of 3, **and** it loses no row that control decides in ≥ 2 of 3.
   - For (c) only: t1513, c1515 and c8452 stay sat in 3 of 3.
   - For (b) only: the wall on rows unsat in both arms may not rise by more than 10% (median).
4. **Otherwise** the result is a null with its observable present. Record it as `S6` in
   `docs/engine_program_capabilities.yaml` with the log reference. A null without the observable is VOID,
   never negative.
5. An adopted A/B is still not armed until the family passes §4 (controls + moat) on the arming pin.

## 4. The arming decision rule (per family, per key)

A key is armed in a shipped yaml only when **all** of these hold, on one gated pin built in an isolated
worktree:

1. **Arrival (S5).** The armed yaml, with zero `NY_*`, shows the lever's own observable on a real row. The
   NEGCTL runs carry the proof: `need` + flight `lever:<NAME>=<v>@config`.
2. **Measured (S6).** Control vs armed × 3 at the official budgets on the pre-registered rows, with conversions
   ≥ 2 of 3 (the W-series or A/B protocol above).
3. **Negative controls clean (S7)**, for every NEGCTL row of the tier:
   - OK (tier engaged, attacks confirmed off) in **both** profiles;
   - 0 unsat and 0 alarms;
   - a VOID control is re-run or replaced by another field-SAT row, never counted.

   Plus, for tiers with threshold properties (cgan-child-zono, fwd-root-screen): the **knife-edge ladder**, all
   four rungs, OK and not unsat. Relational tiers are covered by the gap in §5.
4. **Both-year moat in both profiles (S8)** on the family's categories, on the armed configs:
   `M0MOAT_DONE rc=0`, i.e. 0 EMERGENCY, 0 VOID/not-run, 0 LOST. A LOST row is a regression to explain before
   arming. It is not a soundness stop.
5. **Governance (§1.10).**
   - Arm the resolved file (`ls configs/vnncomp26/` first; a vnncomp26 file shadows vnncomp25).
   - Flip the key's `no_shipped_preset_arms_*` pin test and its
     `crates/ny-levers/tests/data/preset_unarmed_promotions.tsv` row in the same commit.
   - Record the evidence paths in `docs/engine_program_capabilities.yaml` (`evidence.S7` / `S8` / `S9`).

**Kill line.** Any unsat on a field-SAT row, a synthetic sat or a knife-edge rung is an **emergency**, and the
family stays dark.

## 5. Known limits (read before trusting a green)

- **The knife-edge is not expressible for classification rows** (cifar100, tinyimagenet, cct). Those are the
  frozen-dual and bi-window tiers.
  - Their witnessed clauses are relational (`Y_t <= Y_j`), and ny refuses an offset on a relational atom
    (`crates/ny-onnx/src/vnnlib/normalize/convert.rs`: "Relational constraint must have zero constant").
    `knife_edge.py` refuses such rows. `--allow-partial` emits the rung at the natural slack and records
    `knife_edge_exact=false`, which is not a knife-edge.
  - Closing this needs either an ONNX final-bias rewrite (needs the `onnx` Python package) or parser support
    for relational offsets (a spec-translation, U1/G3, TCB change). Until then S7 for those tiers is
    controls-only.
- **f32 vs real semantics of the ladder.** y* in a witness file is ny's ONNX Runtime f32 forward.
  - Rungs below the f32 spacing of y* are marked `sub_f32_resolution` (δ = 1e-9 and 1e-12 on cgan r4). The
    real-valued forward may not reproduce y* to that precision.
  - `--y-values` anchors the ladder on a higher-precision forward at x*.
  - The kill line does not change: an unsat on any rung stops everything. The triage then starts from which
    semantics the rung is sat in.
- **Knife-edge tightness.** A rung is sat with slack exactly δ **at x***. The box maximum can be larger when x*
  is a first-hit attack point rather than a maximiser, so the rung then bites less than δ. Clauses that do not
  hold at x* are left unchanged (`clauses_rewritten` in the manifest).
- **fwd-root per-clause refutations are not observable.** The screen prints counts only, so the alarm catches
  "every clause refuted" on a sat row, not "the witnessed clause refuted".
- **No frozen-dual witness-class alarm by default.** tiny 1831 and 5529 are banked `timeout` and no witness is
  in the tree. Pass `--witness t1513=<sat .res>` to `negative_controls.py` (for example
  `scratchpad/w10b/v2/FDB_t1513_fdb_satctl.res`) to add the per-class alarm.
- **Proof mode has no scored-row baseline before the first both-profile run**, which is itself the first S8
  measurement. Expect NOTE lines (`PROFILE-PROOF-SLOWER`) and read them before blaming the verifier.
- **Paired-network rows** (isomorphic/monotonic acasxu) pass the model literal with absolute members, as the
  sweep does. `ny vnncomp plan` may not load that literal; such rows print `PLAN-UNVERIFIABLE`, and the run's
  own `Loading preset:` line decides.
