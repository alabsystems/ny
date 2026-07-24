//! Kani symbolic (all-inputs) proof harnesses for the soundness lemmas underlying
//! ny-cert's one-hidden-layer ReLU exact-rational CROWN certifier (crown.rs).
//!
//! These are NOT sampling tests. Each `kani::any()` returns a fully symbolic value;
//! `kani::assume` constrains the symbolic domain; Kani's model checker proves the
//! asserted post-condition holds for EVERY value in that domain via the SAT/SMT
//! backend. All arithmetic is integer (rationals scaled to integers) so the solver
//! is exact (no floating-point rounding).
//!
//! Bit-width discipline (for a tractable, yet still fully-exhaustive, proof):
//!   * Symbolic inputs are small fixed-width integers so CBMC bit-blasts a small
//!     multiplier circuit: the two ReLU-envelope harnesses (which COMPARE two
//!     symbolic products) use `i8` inputs; the two Farkas harnesses use `i16`.
//!   * Every input is widened to `i32` and every PRODUCT is computed in `i32`.
//!     Two i16 values multiply to at most 32767^2 = 1_073_676_289 < i32::MAX
//!     (2_147_483_647), and two i8 values to at most 127^2, so no product
//!     overflows i32 -> the i32 arithmetic is exact.
//!   * Where two products are SUMMED (the Farkas harnesses) each product is first
//!     widened to `i64` before adding, so the sum cannot overflow either.
//! Kani additionally emits and discharges arithmetic-overflow VCCs on every op;
//! their SUCCESS in the output is machine confirmation that the integer model is
//! exact over the whole assumed domain.
//!
//! Because the inputs range over the FULL fixed-width domain (constrained only by
//! each lemma's own preconditions), every harness is an exhaustive proof over the
//! entire bounded-integer lattice -- the symbolic analogue of the unbounded
//! rational lemma, which is scale-invariant (it holds for all rationals iff it
//! holds for all integer numerator/denominator pairs).

/// Integer ReLU on the i32 domain.
fn relu(z: i32) -> i32 {
    if z > 0 {
        z
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// (a) ReLU UPPER envelope (the upper chord used in crown.rs).
//
// crown.rs (unstable case): for a pre-activation bound l < 0 < u, the ReLU
// upper relaxation is
//     a <= s * (z - l),   with slope  s = u / (u - l).
// Soundness requirement: this chord dominates the true ReLU on [l, u]:
//     relu(z) <= s*(z - l)   for all z in [l, u].
// Scale by the positive denominator D = (u - l) > 0 to stay exact in integers:
//     (u - l) * relu(z)  <=  u * (z - l).
// ---------------------------------------------------------------------------
#[kani::proof]
fn upper_envelope_dominates_relu() {
    let l: i32 = kani::any::<i8>() as i32;
    let u: i32 = kani::any::<i8>() as i32;
    let z: i32 = kani::any::<i8>() as i32;

    kani::assume(l < 0); // l < 0
    kani::assume(u > 0); // u > 0
    kani::assume(z >= l && z <= u); // z in [l, u]

    let d = u - l; // (u - l) > 0
                   // LHS = D * relu(z), RHS = u * (z - l). Factors are i8-range so the products
                   // fit in i32 exactly (max |product| < 256*256). Claim LHS <= RHS.
    let lhs = d * relu(z);
    let rhs = u * (z - l);
    assert!(lhs <= rhs);
}

// ---------------------------------------------------------------------------
// (b) ReLU LOWER envelope.
//
// crown.rs: lower relaxation  a >= alpha * z  with alpha in [0, 1].
// Soundness requirement: this line is dominated by the true ReLU for ALL z:
//     alpha * z <= relu(z)   for all z, all alpha in [0,1].
// Scale alpha = p/q with integer 0 <= p <= q, q > 0:
//     p * z <= q * relu(z).
// ---------------------------------------------------------------------------
#[kani::proof]
fn lower_envelope_dominated_by_relu() {
    let p: i32 = kani::any::<i8>() as i32; // alpha numerator
    let q: i32 = kani::any::<i8>() as i32; // alpha denominator
    let z: i32 = kani::any::<i8>() as i32;

    kani::assume(q > 0);
    kani::assume(p >= 0 && p <= q); // alpha = p/q in [0, 1]

    let lhs = p * z; // q * (alpha * z)
    let rhs = q * relu(z); // q * relu(z)
    assert!(lhs <= rhs);
}

// ---------------------------------------------------------------------------
// (c) FARKAS combination soundness (the backward pass uses non-negative
//     multipliers to combine the relaxation inequalities).
//
// Lemma: if  e1 <= 0  and  e2 <= 0  are valid, and  m1, m2 >= 0  are the
// non-negative Farkas multipliers, then the combination
//     m1*e1 + m2*e2 <= 0
// is also valid. This is the core soundness property of a non-negative linear
// combination of "<= 0" inequalities (extends to any number of terms by
// induction; two terms is the inductive step).
// ---------------------------------------------------------------------------
#[kani::proof]
fn farkas_nonneg_combination_valid() {
    let e1: i32 = kani::any::<i16>() as i32;
    let e2: i32 = kani::any::<i16>() as i32;
    let m1: i32 = kani::any::<i16>() as i32;
    let m2: i32 = kani::any::<i16>() as i32;

    // Valid inequalities: each expression is <= 0.
    kani::assume(e1 <= 0);
    kani::assume(e2 <= 0);
    // Non-negative multipliers.
    kani::assume(m1 >= 0);
    kani::assume(m2 >= 0);

    // Products in i32 (exact); widen to i64 before summing so the sum is exact.
    let t1 = (m1 * e1) as i64; // (>=0) * (<=0) -> <= 0
    let t2 = (m2 * e2) as i64; // (>=0) * (<=0) -> <= 0
    let combo: i64 = t1 + t2;
    assert!(combo <= 0);
}

// ---------------------------------------------------------------------------
// (c') FARKAS, stronger form actually used by CROWN's backward pass:
//      a non-negative combination of lower-bound rows preserves a TRUE lower
//      bound. If for each i,  row value  v_i  satisfies  v_i <= actual_i
//      (a sound lower bound) and multiplier m_i >= 0, then
//          sum m_i*v_i  <=  sum m_i*actual_i.
//      This is the monotone-combination property that makes the backward
//      substitution sound. Two-term inductive step:
// ---------------------------------------------------------------------------
#[kani::proof]
fn farkas_preserves_lower_bound() {
    let v1: i32 = kani::any::<i16>() as i32;
    let a1: i32 = kani::any::<i16>() as i32; // actual_1, with v1 <= a1
    let v2: i32 = kani::any::<i16>() as i32;
    let a2: i32 = kani::any::<i16>() as i32; // actual_2, with v2 <= a2
    let m1: i32 = kani::any::<i16>() as i32;
    let m2: i32 = kani::any::<i16>() as i32;

    kani::assume(v1 <= a1);
    kani::assume(v2 <= a2);
    kani::assume(m1 >= 0);
    kani::assume(m2 >= 0);

    // Equivalent non-negative-difference (Farkas) form:
    //   sum m_i*v_i <= sum m_i*a_i   <=>   sum m_i*(a_i - v_i) >= 0.
    // Each (a_i - v_i) >= 0 and each m_i >= 0, so each term is a product of two
    // non-negatives. d_i in [0, 2^16] fits i32; m_i*d_i in i32; sum widened to i64.
    let d1: i32 = a1 - v1; // >= 0
    let d2: i32 = a2 - v2; // >= 0
    let slack: i64 = (m1 * d1) as i64 + (m2 * d2) as i64;
    assert!(slack >= 0);
}
