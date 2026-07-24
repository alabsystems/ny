---- MODULE EnvLock ----
\* Env-lock protocol for ny's serialized process-environment mutation in tests
\* (mirrors the evidence-contract style of NyVerdictAdmission.tla).
\*
\* WHAT IS MODELED (grounded in the Rust, revision of 2026-07-20):
\*   * the workspace-blessed env-mutation choke point
\*     `ny_test_utils::env` (crates/ny-test-utils/src/env.rs), re-exported to
\*     ny-propagate tests via crates/ny-propagate/src/tests/mod.rs:
\*       - `lock_env()`: ONE process-wide Mutex for all environment mutation
\*         in a test binary;
\*       - `ScopedEnvVar::set/unset`: RAII guard that captures the previous
\*         value and restores it on drop (also on panic);
\*       - `with_serialized_env_vars(vars, f)`: lock -> set guards -> f() ->
\*         guards drop (restore) -> unlock;
\*       - `with_serialized_env_vars_removed(vars, f)`: same with unset guards.
\*   * writer threads = the conv2d #wall-deadwork tests that set the
\*     process-global `NY_CONV_SKIP_DEAD_F32` under `with_serialized_env_vars`
\*     (env cell Bypass = var set to "1", Default = var unset);
\*   * the reader thread = `dispatch_conv2d_with_engine_matches_none_baseline_3959`
\*     (crates/ny-propagate/src/network/backward_dispatch/tests_engine.rs),
\*     whose `gemm_calls() > 0` assertion consults the env cell mid-assertion:
\*     with the var set, the dead-work skip path bypasses the GemmEngine
\*     entirely and the assertion races to a spurious failure.
\*
\* THE HISTORICAL BUG THIS GUARDS (tests_engine.rs, comment at the test head):
\*   the OLD dispatch test read the env cell WITHOUT holding the shared env
\*   lock, so a writer's set..restore window could interleave around the read
\*   (deterministic repro: NY_CONV_SKIP_DEAD_F32=1 fails it in isolation).
\*   THE FIX: hold the same lock with the var scoped to unset
\*   (`with_serialized_env_vars_removed(&["NY_CONV_SKIP_DEAD_F32"], ..)`).
\*   The constant ReaderHoldsLock below selects the variant:
\*     FALSE = the OLD unlocked read  -> ReaderSeesDefault MUST fail
\*             (EnvLock.bug_unlocked_read.cfg, counterexample in the receipt);
\*     TRUE  = the FIXED locked+scoped read -> all invariants MUST hold
\*             exhaustively (EnvLock.cfg).
\*
\* SCOPE / HONESTY: this is a finite-state MODEL theorem (3 threads: 2 writers
\* + 1 reader, one env cell) about the locking protocol, not a proof about the
\* Rust; the correspondence of each transition to the cited code is code-level
\* trust. Mutex acquisition is modeled atomic (std::sync::Mutex guarantees
\* this); ScopedEnvVar's capture-previous/restore is modeled per-thread.
\*
\* HOW TO RUN (manual, like NyVerdictAdmission — no Makefile target; from the
\* ny repo root, ty = the ty TLA+ model checker, https://github.com/alabsystems/ty):
\*   fixed  (must pass exhaustively):
\*     ty check specs/tla/EnvLock.tla --config specs/tla/EnvLock.cfg --workers 1
\*   buggy  (must produce the interleaving counterexample):
\*     ty check specs/tla/EnvLock.tla --config specs/tla/EnvLock.bug_unlocked_read.cfg --workers 1
\* Receipt with verbatim outputs + artifact hashes: EnvLock_receipt.txt.

EXTENDS TLC

CONSTANTS
  Writers,         \* writer-thread ids, e.g. {"w1", "w2"}
  Reader,          \* the reader-thread id, e.g. "r"
  ReaderHoldsLock  \* TRUE = fixed locked+scoped read; FALSE = old unlocked read

ASSUME Writers # {}
ASSUME Reader \notin Writers
ASSUME ReaderHoldsLock \in BOOLEAN

Threads == Writers \union {Reader}

NoOwner == "free"
ASSUME NoOwner \notin Threads

\* The one global env cell: NY_CONV_SKIP_DEAD_F32 unset / set to "1".
EnvValues == {"Default", "Bypass"}

VARIABLES
  env_var,   \* the process-global env cell (Default = unset, Bypass = "1")
  lock,      \* env_mutex() owner: a thread id, or NoOwner when free
  wpc,       \* writer program counters
             \*   "idle" -> "locked" -> "bypass_set" -> "work_done"
             \*   -> "restored" -> "idle"
  wprev,     \* per-writer ScopedEnvVar::set captured previous value
  rpc,       \* reader program counter
             \*   fixed: "idle" -> "locked" -> "scoped" -> "read"
             \*          -> "restored" -> "idle"
             \*   buggy: "idle" -> "read" -> "idle"
  rprev,     \* reader ScopedEnvVar::unset captured previous value
  robs       \* value the reader observed at its gated read ("none" outside)

vars == <<env_var, lock, wpc, wprev, rpc, rprev, robs>>

WriterStages == {"idle", "locked", "bypass_set", "work_done", "restored"}
ReaderStages == {"idle", "locked", "scoped", "read", "restored"}

TypeOK ==
  /\ env_var \in EnvValues
  /\ lock \in Threads \union {NoOwner}
  /\ wpc \in [Writers -> WriterStages]
  /\ wprev \in [Writers -> EnvValues]
  /\ rpc \in ReaderStages
  /\ rprev \in EnvValues
  /\ robs \in EnvValues \union {"none"}

\* ------------------------------------------------------------------------
\* Init: the env cell starts unset (Default), the lock free, every thread
\* idle. (The interesting nondeterminism is the interleaving, not the start.)

Init ==
  /\ env_var = "Default"
  /\ lock = NoOwner
  /\ wpc = [w \in Writers |-> "idle"]
  /\ wprev = [w \in Writers |-> "Default"]
  /\ rpc = "idle"
  /\ rprev = "Default"
  /\ robs = "none"

\* ------------------------------------------------------------------------
\* Writer thread w: with_serialized_env_vars(&[("NY_CONV_SKIP_DEAD_F32","1")], f)
\*   lock_env() -> ScopedEnvVar::set (capture previous, set Bypass) ->
\*   f() (the #wall-deadwork test body) -> guard drops (restore previous) ->
\*   MutexGuard drops (release). Loops to model repeated test executions.

WAcquire(w) ==
  /\ wpc[w] = "idle"
  /\ lock = NoOwner
  /\ lock' = w
  /\ wpc' = [wpc EXCEPT ![w] = "locked"]
  /\ UNCHANGED <<env_var, wprev, rpc, rprev, robs>>

WSet(w) ==
  /\ wpc[w] = "locked"
  /\ wprev' = [wprev EXCEPT ![w] = env_var]  \* ScopedEnvVar captures previous
  /\ env_var' = "Bypass"
  /\ wpc' = [wpc EXCEPT ![w] = "bypass_set"]
  /\ UNCHANGED <<lock, rpc, rprev, robs>>

WWork(w) ==
  /\ wpc[w] = "bypass_set"
  /\ wpc' = [wpc EXCEPT ![w] = "work_done"]   \* critical work under Bypass
  /\ UNCHANGED <<env_var, lock, wprev, rpc, rprev, robs>>

WRestore(w) ==
  /\ wpc[w] = "work_done"
  /\ env_var' = wprev[w]                      \* guard drop restores previous
  /\ wpc' = [wpc EXCEPT ![w] = "restored"]
  /\ UNCHANGED <<lock, wprev, rpc, rprev, robs>>

WRelease(w) ==
  /\ wpc[w] = "restored"
  /\ lock = w
  /\ lock' = NoOwner
  /\ wpc' = [wpc EXCEPT ![w] = "idle"]
  /\ UNCHANGED <<env_var, wprev, rpc, rprev, robs>>

\* ------------------------------------------------------------------------
\* Reader thread, FIXED variant (ReaderHoldsLock = TRUE):
\* with_serialized_env_vars_removed(&["NY_CONV_SKIP_DEAD_F32"], f)
\*   lock_env() -> ScopedEnvVar::unset (capture previous, scope to Default) ->
\*   f() reads the cell at its gated assertion -> guard drops (restore) ->
\*   MutexGuard drops (release).

RAcquire ==
  /\ ReaderHoldsLock
  /\ rpc = "idle"
  /\ lock = NoOwner
  /\ lock' = Reader
  /\ rpc' = "locked"
  /\ UNCHANGED <<env_var, wpc, wprev, rprev, robs>>

RScope ==
  /\ rpc = "locked"
  /\ rprev' = env_var                          \* ScopedEnvVar::unset captures
  /\ env_var' = "Default"                      \* var scoped to unset
  /\ rpc' = "scoped"
  /\ UNCHANGED <<lock, wpc, wprev, robs>>

RReadLocked ==
  /\ rpc = "scoped"
  /\ robs' = env_var                           \* the gemm_calls()>0 assertion
  /\ rpc' = "read"
  /\ UNCHANGED <<env_var, lock, wpc, wprev, rprev>>

RRestore ==
  /\ ReaderHoldsLock
  /\ rpc = "read"
  /\ env_var' = rprev                          \* guard drop restores previous
  /\ rpc' = "restored"
  /\ robs' = "none"
  /\ UNCHANGED <<lock, wpc, wprev, rprev>>

RRelease ==
  /\ rpc = "restored"
  /\ lock = Reader
  /\ lock' = NoOwner
  /\ rpc' = "idle"
  /\ UNCHANGED <<env_var, wpc, wprev, rprev, robs>>

\* ------------------------------------------------------------------------
\* Reader thread, OLD BUGGY variant (ReaderHoldsLock = FALSE): the dispatch
\* test body just runs — no lock, no scoping — and its assertion consults the
\* process-global cell mid-flight, racing any writer's set..restore window.

RReadUnlocked ==
  /\ ~ReaderHoldsLock
  /\ rpc = "idle"
  /\ robs' = env_var                           \* unlocked mid-assertion read
  /\ rpc' = "read"
  /\ UNCHANGED <<env_var, lock, wpc, wprev, rprev>>

RFinishUnlocked ==
  /\ ~ReaderHoldsLock
  /\ rpc = "read"
  /\ rpc' = "idle"
  /\ robs' = "none"
  /\ UNCHANGED <<env_var, lock, wpc, wprev, rprev>>

Next ==
  \/ \E w \in Writers:
       WAcquire(w) \/ WSet(w) \/ WWork(w) \/ WRestore(w) \/ WRelease(w)
  \/ RAcquire \/ RScope \/ RReadLocked \/ RRestore \/ RRelease
  \/ RReadUnlocked \/ RFinishUnlocked

Spec == Init /\ [][Next]_vars

\* ------------------------------------------------------------------------
\* Invariants.

\* A thread is inside the lock-held region of its protocol. The OLD reader
\* never enters one — that IS the bug: its read is not a critical section.
InCritW(w) == wpc[w] # "idle"
InCritR == ReaderHoldsLock /\ rpc # "idle"

\* The lock itself: every thread inside its lock-held region owns the lock...
LockOwnership ==
  /\ \A w \in Writers: InCritW(w) => lock = w
  /\ InCritR => lock = Reader

\* ...hence at most one thread is in a critical section at a time.
MutualExclusion ==
  /\ \A w1, w2 \in Writers:
       (InCritW(w1) /\ InCritW(w2)) => w1 = w2
  /\ \A w \in Writers: ~(InCritW(w) /\ InCritR)

\* THE theorem: whenever the reader performs its gated read (the
\* gemm_calls()>0 assertion moment), it observed Default — no writer's
\* Bypass window leaked into the assertion.
ReaderSeesDefault == (rpc = "read") => (robs = "Default")

====
