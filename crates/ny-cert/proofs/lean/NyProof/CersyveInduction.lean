/-
  CersyveInduction.lean — certified-SAFE-FOREVER for clamped neural control loops.

  The cersyve benchmark (VNN-COMP 2025/2026) ships, per control system, a PAIR of
  networks compiled from the same weights (confirmed against the Cersyve.jl source:
  the value/certificate net V is shared bit-for-bit and the exact ReLU clamp is the
  authors' intended clamped closed-loop):

    * the `inv` query certifies ONE-STEP INDUCTIVENESS of the V-sublevel set on the
      operating domain D under the clamped transition  x' = clamp(f̂(x), D):
          ∀ x ∈ D,  V x ≤ 0  →  V (step x) ≤ 0            (NY verdict: unsat)
    * the `con` query certifies CONSTRAINT SATISFACTION on the sublevel set:
          ∀ x ∈ D,  V x ≤ 0  →  Safe x                    (NY verdict: unsat)

  NY verifies both queries for the finetune systems TODAY (12/12 scored cersyve
  instances solve; the finetune `con`/`inv` legs are the unsat rows).  What VNN-COMP
  scores is the two BOUNDED one-step properties.  This file supplies the missing —
  and mathematically tiny — induction that upgrades them to the UNBOUNDED claim:

      every trajectory started in S = {x ∈ D : V x ≤ 0} remains in S, and is Safe,
      at EVERY time step, forever.

  # Honest scope (state this wherever the result is cited)
  * The theorem is about the ONNX-DEFINED transition `step` — the learned dynamics
    surrogate f̂ composed with the exact clamp — NOT the physical plant.  The
    model-mismatch (surrogate-dynamics) hypothesis is carried by the reader: the
    conclusion transfers to the real system exactly insofar as f̂ models it.  (E.g.
    pendulum's analytic dynamics clamps θ̇ before integrating; the network clamps
    after — the theorem speaks about the network's system, honestly.)
  * `step_mem` (the clamp lands in D) is a hypothesis; for cersyve's box domain and
    coordinatewise clamp it is true by construction of clamp.
  * The one-step hypotheses `hinv`/`hcon` are exactly what NY's unsat verdicts on the
    `inv`/`con` queries certify (exact-rational CROWN path ⇒ certifiable; float path
    ⇒ verdict-grade).  This file adds NO trust: it consumes those facts as hypotheses
    and contributes only the induction, kernel-checked.
-/
import Mathlib.Data.Rat.Defs

namespace Crownproof

namespace Cersyve

variable {State : Type*}

/-- A clamped neural control system: operating domain `D`, the ONNX-defined
    one-step transition `step x = clamp(f̂ x, D)` (hence `step_mem`), the
    certificate/value network `V`, and the safety predicate `Safe`. -/
structure ClampedSystem (State : Type*) where
  /-- The operating domain (the clamp target box). -/
  D : State → Prop
  /-- One step of the clamped closed loop: `x ↦ clamp(f̂ x, D)`. -/
  step : State → State
  /-- The clamp lands in the domain (true by construction for a box clamp). -/
  step_mem : ∀ x, D x → D (step x)
  /-- The certificate (value) network. -/
  V : State → ℚ
  /-- The safety predicate (the `con` query's target). -/
  Safe : State → Prop

/-- The closed-loop trajectory from `x₀`. -/
def trajectory (sys : ClampedSystem State) (x₀ : State) : ℕ → State
  | 0 => x₀
  | k + 1 => sys.step (trajectory sys x₀ k)

/--
**SAFE FOREVER.**  If the two one-step facts NY certifies on the cersyve pair hold —

  * `hinv` (the `inv` query): on `D`, the sublevel set `V ≤ 0` is preserved by one
    clamped step;
  * `hcon` (the `con` query): on `D`, the sublevel set satisfies `Safe` —

then every trajectory started in `S = {x ∈ D : V x ≤ 0}` remains in `D`, remains in
the sublevel set, and is `Safe`, at every time step.  This is the unbounded claim no
bounded-horizon VNN-COMP query states; the entire proof is the induction, so the
trust base is the kernel plus whatever certifies `hinv`/`hcon`.
-/
theorem safe_forever (sys : ClampedSystem State)
    (hinv : ∀ x, sys.D x → sys.V x ≤ 0 → sys.V (sys.step x) ≤ 0)
    (hcon : ∀ x, sys.D x → sys.V x ≤ 0 → sys.Safe x)
    (x₀ : State) (hx₀D : sys.D x₀) (hx₀V : sys.V x₀ ≤ 0) :
    ∀ k : ℕ,
      sys.D (trajectory sys x₀ k) ∧
      sys.V (trajectory sys x₀ k) ≤ 0 ∧
      sys.Safe (trajectory sys x₀ k) := by
  intro k
  induction k with
  | zero =>
      simp only [trajectory]
      exact ⟨hx₀D, hx₀V, hcon x₀ hx₀D hx₀V⟩
  | succ k ih =>
      obtain ⟨hD, hV, _⟩ := ih
      simp only [trajectory]
      have hD' := sys.step_mem _ hD
      have hV' := hinv _ hD hV
      exact ⟨hD', hV', hcon _ hD' hV'⟩

/-- The headline corollary in its quotable form: started in the certified region,
    the clamped closed loop is safe at every time step. -/
theorem safe_at_every_step (sys : ClampedSystem State)
    (hinv : ∀ x, sys.D x → sys.V x ≤ 0 → sys.V (sys.step x) ≤ 0)
    (hcon : ∀ x, sys.D x → sys.V x ≤ 0 → sys.Safe x)
    (x₀ : State) (hx₀D : sys.D x₀) (hx₀V : sys.V x₀ ≤ 0) :
    ∀ k : ℕ, sys.Safe (trajectory sys x₀ k) :=
  fun k => (safe_forever sys hinv hcon x₀ hx₀D hx₀V k).2.2

/-! ## Worked instance — how NY's two verdicts discharge the hypotheses.

For a concrete finetune system (double_integrator / lane_keep / pendulum /
point_mass / unicycle — the five whose `con` AND `inv` queries NY returns UNSAT
on, measured 2026-07-09, `docs/MEASURED_CERSYVE_SAFE_FOREVER.md`), NY's UNSAT
verdicts ARE the two universally-quantified facts `safe_forever` consumes:

  * NY-unsat on the `inv` query  ⇔  `hinv` : ∀ x ∈ D, V x ≤ 0 → V (step x) ≤ 0
  * NY-unsat on the `con` query  ⇔  `hcon` : ∀ x ∈ D, V x ≤ 0 → Safe x

The example below is schematic (the `System`/verdict facts stand for the ONNX
`ClampedSystem` and NY's discharged obligations; the kernel-checked binding of a
specific ONNX graph to a `ClampedSystem` is the v2 `certify_onnx` DAG work). It
demonstrates the composition is DIRECT: given the two verdicts, unbounded safety
is one application, no further proof obligation. -/
example (sys : ClampedSystem State)
    -- the two facts NY's `inv`/`con` UNSAT verdicts establish:
    (ny_inv_unsat : ∀ x, sys.D x → sys.V x ≤ 0 → sys.V (sys.step x) ≤ 0)
    (ny_con_unsat : ∀ x, sys.D x → sys.V x ≤ 0 → sys.Safe x)
    (x₀ : State) (start_certified : sys.D x₀ ∧ sys.V x₀ ≤ 0) :
    -- ⇒ the closed loop is safe for ALL time:
    ∀ k : ℕ, sys.Safe (trajectory sys x₀ k) :=
  safe_at_every_step sys ny_inv_unsat ny_con_unsat x₀ start_certified.1 start_certified.2

end Cersyve

end Crownproof

/-! ## Trust-base check — the induction must reduce to the standard axioms only. -/

#print axioms Crownproof.Cersyve.safe_forever
#print axioms Crownproof.Cersyve.safe_at_every_step
