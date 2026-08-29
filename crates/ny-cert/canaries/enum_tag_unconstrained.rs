#![feature(register_tool)]
#![register_tool(trust)]
#![allow(dead_code)]

// R141 EXPERIMENT. The lever wants to model `transmute::<Alignment, usize>` as a
// read of the flattened enum's `__tag` field, cast to the target integer.
// That is only correct if `__tag` holds the REAL discriminant (1, 2, 4) and not a
// variant INDEX (0, 1, 2). ty_convert.rs promises the real tag for
// `variants[i].discriminant`; whether the flattened `__tag` FIELD agrees is a
// separate question, and assuming it would be exactly the class of guess that
// produced false proofs #9 and #11.
//
// DISCRIMINATING TEST: `E`'s discriminants are {1,2,4} but its indices are {0,1,2}.
//   * `disc_ge_one`  is panic-free  <=> the model has the real discriminants.
//   * `disc_ge_zero` is panic-free under EITHER reading (it is the control that
//     proves the harness works at all).
//   * `disc_ge_four` must NOT be provable under either reading (anti-control:
//     if this "proves", the tag is unconstrained and the harness is lying).

#[repr(usize)]
#[derive(Copy, Clone)]
pub enum E {
    A = 1,
    B = 2,
    C = 4,
}

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct W(E);

// CONTROL: provable under both readings.
pub fn disc_ge_zero(e: E) -> usize {
    let v = e as usize;
    assert!(v >= 1 || v < 1);
    v
}

// EXPERIMENT: provable ONLY if __tag carries the real discriminants.
pub fn disc_ge_one(e: E) -> usize {
    let v = e as usize;
    assert!(v >= 1);
    v
}

// ANTI-CONTROL: must stay unprovable. 4 is not a lower bound on {1,2,4}.
pub fn disc_ge_four(e: E) -> usize {
    let v = e as usize;
    assert!(v >= 4);
    v
}

// THE SHAPE THE LEVER IS ABOUT. Expect the CastKind::Transmute refusal today,
// with [src=Some(W) peeled=Some(E)] under TRUST_PEEL_TRACE.
pub fn unwrap_w(w: W) -> usize {
    unsafe { std::mem::transmute(w) }
}

// WRAP DIRECTION — must STAY refused after R141. It carries a real obligation
// (not every usize is a valid E).
pub fn wrap_w(v: usize) -> W {
    unsafe { std::mem::transmute(v) }
}

// R141 FALSE-PROOF ANTI-CONTROL. After the leg lands, `unwrap_w` is admitted as
// an OpaqueScalar — an UNCONSTRAINED usize. Every real `W` holds a discriminant
// in {1,2,4}, so `>= 1` is TRUE of the source program; the model must not know
// that. If this ever reports PROVED, the havoc has been silently constrained and
// the leg is minting a fact it never established. Must stay unknown/failed.
pub fn unwrap_w_ge_one(w: W) -> usize {
    let v: usize = unsafe { std::mem::transmute(w) };
    assert!(v >= 1);
    v
}
