# Exact-real centered normalization successor

Status: proof successor, awaiting its own successful Lean check. This adds one independent
module, `NyProof/CenteredNormalization.lean`. It changes no existing source,
import registration, axiom audit, theorem assumption, or qualification receipt.

The intended environment is the existing pinned official Lean 4.30.0 overlay
and Mathlib commit `c5ea00351c28e24afc9f0f84379aa41082b1188f`. The source includes
15 explicit axiom queries so the later gate can require every named theorem
and reject missing declarations, `sorryAx`, or unexpected axioms. The v1 run
failed before elaboration because Lake ran outside the Lean project. The v2
run reached all 15 queries under the pinned overlay, but two proof errors
introduced `sorryAx` into four rosters, so that run earns no proof credit.
This successor corrects the root namespace of the erased-coordinate
Cauchy–Schwarz theorem and the final addition ordering. Every theorem
statement and assumption remains unchanged. Both failed source packets and
their complete outputs are preserved; this new source needs its own run.

## Mathematical statements and derivations

All vectors have finite real coordinates. Define

```
E(z) = sum_i z_i^2
dot(a,z) = sum_i a_i*z_i
mean(x) = sum_i x_i / h
P(x)_i = x_i - mean(x)
D(z) = sqrt(epsilon + E(z)/h)
```

For `epsilon > 0` and `h > 0`, the module derives

```
abs(dot(a,z))/D(z) <= sqrt(h)*sqrt(E(a)).
```

Finite-sum Cauchy–Schwarz gives `dot(a,z)^2 <= E(a)*E(z)`. Multiplying by the
positive dimension and using the nonnegative extra term `epsilon*h*E(a)`
bounds the squared numerator by the squared right-hand side times `D(z)^2`.
The denominator is proved strictly positive before dividing. The norm and
desired direction bound are not input premises. This direction theorem holds
for any vector `z`, so it also applies to the original centered vector.

The module derives, rather than assumes, the sharper coordinate bound when
`sum_i z_i = 0` and `h` is the actual finite coordinate cardinality:

```
h*z_k^2 <= (h-1)*E(z)
abs(z_k)/D(z) <= sqrt(h-1).
```

The coordinates other than `k` sum to `-z_k`. Cauchy–Schwarz on precisely that
erased coordinate set gives `z_k^2 <= (h-1)*sum_{i!=k} z_i^2`; adding the selected
coordinate square yields the stated squared inequality. The same positive
denominator argument proves the normalized range. This includes dimension one,
where centering and the coordinate range are exactly zero.

The source also derives `sum(P(x)) = 0`, `dot(P(a),x) = dot(a,P(x))`, and
`dot(a,P(x)) = dot(a,x)` when `sum(a) = 0`. These identities give the projected
direction corollary using the norm of `P(a)`. A final scalar corollary keeps an
additive beta/bias constant explicitly through the triangle inequality.

## Correspondence obligations

The proposed mathematical mapping to the streamed original decoder head is:

- `x`: the genuine original pre-normalization vector;
- `z = P(x)`: its actual centered vector;
- `a_i = (W_target,i - W_competitor,i)*gamma_i`, with exact original real weights;
- the additive constant is the original bias difference plus
  `sum_i (W_target,i - W_competitor,i)*beta_i`;
- `epsilon`: the original declared positive real epsilon;
- `h`: the exact original hidden-coordinate count.

This mapping is a stated correspondence obligation. The new Lean source does
not prove that a Rust object supplies those values or that its interval
endpoints contain the exact values. It does not prove the coefficient import,
directed projection/subtraction, square-root enclosure, scaled L2 arithmetic,
finite-overflow fallback, all-class scan, source/model/clock identity, retained
proof owner, or ledger accounting. It does not grant pruning or UNSAT authority.

The independent Q80 checker has its own directed integer implementation; the
producer's F64 implementation has its own qualification requirements. Neither
can use this source-only module as a replacement for those checks. Existing
formal PrefixCover qualification remains separate and unchanged. The rejected
linked CIC experiment is not credited by these new proof scripts.

## Required validation before registration

1. Run the exact frozen source under the pinned official Lean 4.30 overlay.
2. Require all 15 local axiom-query declarations and an allowed roster of only
   `propext`, `Classical.choice`, and `Quot.sound` (or a subset).
3. Preserve the actual source hash, toolchain/dependency identity, command,
   complete output, status, and pre/post source hashes.
4. Only then consider adding the new module to `NyProof.lean` and the complete
   `AxiomAudit.lean` query set, followed by the full pinned overlay gate.

No new source or theorem is registered by this successor packet.
