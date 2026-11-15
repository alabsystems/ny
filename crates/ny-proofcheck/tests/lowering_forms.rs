// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The checker's routing and reshaping forms (`crate::lower`): Slice, Gather
//! with constant indices, Concat, any static Reshape / Flatten, MatMul with a
//! constant on EITHER side (numpy promotion and batch broadcasting), and graph
//! inputs of any rank (rank 1, non-batch leading dims).
//!
//! Every positive test compares the checker against an EXACT RATIONAL
//! reference evaluator written here from the ONNX operator definitions, with
//! its own N-d arrays and its own index walk (it shares no code with
//! `lower.rs`): routing ops must reproduce the reference BIT-EXACTLY (they copy,
//! never compute), arithmetic must enclose it. Every admitted condition has a
//! refusal next to it, asserting the precise reason. The certified paths
//! through the new forms — IBP, the α/β back-substitution, the frozen-dual
//! generator forms — are checked against the exact network, and one bundle
//! through all of them verifies end to end (and its mutations do not).
//! `tests/lowering_ort.rs` is the ONNX Runtime oracle on the same forms.

mod support;

use ny_cert::Rat;
use ny_proofcheck::fd::Relu;
use ny_proofcheck::ibp::{eval_point, forward};
use ny_proofcheck::lower::{Net, Op};
use ny_proofcheck::testkit::{alpha_map, fd_state, lower_bound, net_from_onnx};
use ny_proofcheck::Verdict;
use support::*;

// ---------------------------------------------------------------------------
// Exact rationals
// ---------------------------------------------------------------------------

fn rat(x: f64) -> Rat {
    Rat::from_f64_exact(x).expect("finite")
}

fn radd(a: Rat, b: Rat) -> Rat {
    a.add(b).expect("rat add")
}

fn rsub(a: Rat, b: Rat) -> Rat {
    a.sub(b).expect("rat sub")
}

fn rmul(a: Rat, b: Rat) -> Rat {
    a.mul(b).expect("rat mul")
}

fn rdiv(a: Rat, b: Rat) -> Rat {
    a.mul(b.inv().expect("nonzero")).expect("rat mul")
}

// ---------------------------------------------------------------------------
// The reference evaluator: N-d arrays of exact rationals
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Nd {
    shape: Vec<usize>,
    v: Vec<Rat>,
}

/// Every multi-index of `shape` in C order (last axis fastest), built by
/// extending prefixes — a different walk from the lowering's odometer.
fn walk(shape: &[usize]) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = vec![Vec::new()];
    for &d in shape {
        let mut next = Vec::with_capacity(out.len() * d);
        for p in &out {
            for i in 0..d {
                let mut q = p.clone();
                q.push(i);
                next.push(q);
            }
        }
        out = next;
    }
    out
}

/// Horner's flat index of `idx` in `shape`.
fn flat(shape: &[usize], idx: &[usize]) -> usize {
    idx.iter().zip(shape).fold(0, |a, (&i, &d)| a * d + i)
}

impl Nd {
    fn input(shape: &[usize], x: &[f64]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), x.len());
        Self {
            shape: shape.to_vec(),
            v: x.iter().map(|&a| rat(a)).collect(),
        }
    }

    fn get(&self, idx: &[usize]) -> Rat {
        self.v[flat(&self.shape, idx)]
    }

    fn with_shape(&self, shape: Vec<usize>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), self.v.len());
        Self {
            shape,
            v: self.v.clone(),
        }
    }
}

fn axis_of(axis: i64, r: usize) -> usize {
    (if axis < 0 { axis + r as i64 } else { axis }) as usize
}

enum SliceRef {
    Ok(Nd),
    /// Some axis keeps no element.
    Empty,
    /// A negative-step start before the first element (numpy and the
    /// specification's clamp disagree).
    Ambiguous,
}

/// ONNX Slice by explicit per-axis enumeration of the kept indices.
fn ref_slice(
    a: &Nd,
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
) -> SliceRef {
    let n = starts.len();
    let axes: Vec<i64> = axes.map_or_else(|| (0..n as i64).collect(), <[i64]>::to_vec);
    let steps: Vec<i64> = steps.map_or_else(|| vec![1; n], <[i64]>::to_vec);
    let mut pick: Vec<Vec<usize>> = a.shape.iter().map(|&d| (0..d).collect()).collect();
    for j in 0..n {
        let ax = axis_of(axes[j], a.shape.len());
        let dim = a.shape[ax] as i128;
        let st = i128::from(steps[j]);
        let mut s = i128::from(starts[j]);
        let mut e = i128::from(ends[j]);
        if s < 0 {
            s += dim;
        }
        if e < 0 {
            e += dim;
        }
        let mut list = Vec::new();
        if st > 0 {
            s = s.max(0).min(dim);
            e = e.max(0).min(dim);
            let mut i = s;
            while i < e {
                list.push(i as usize);
                i += st;
            }
        } else {
            if s < 0 {
                return SliceRef::Ambiguous;
            }
            s = s.min(dim - 1);
            e = e.max(-1).min(dim - 1);
            let mut i = s;
            while i > e {
                list.push(i as usize);
                i += st;
            }
        }
        if list.is_empty() {
            return SliceRef::Empty;
        }
        pick[ax] = list;
    }
    let shape: Vec<usize> = pick.iter().map(Vec::len).collect();
    let v = walk(&shape)
        .iter()
        .map(|o| {
            let src: Vec<usize> = o.iter().enumerate().map(|(d, &i)| pick[d][i]).collect();
            a.get(&src)
        })
        .collect();
    SliceRef::Ok(Nd { shape, v })
}

fn ref_gather(a: &Nd, idims: &[usize], idx: &[i64], axis: i64) -> Nd {
    let ax = axis_of(axis, a.shape.len());
    let dim = a.shape[ax] as i64;
    let mut shape = a.shape[..ax].to_vec();
    shape.extend_from_slice(idims);
    shape.extend_from_slice(&a.shape[ax + 1..]);
    let q = idims.len();
    let v = walk(&shape)
        .iter()
        .map(|o| {
            let mut k = idx[flat(idims, &o[ax..ax + q])];
            if k < 0 {
                k += dim;
            }
            let mut src = o[..ax].to_vec();
            src.push(k as usize);
            src.extend_from_slice(&o[ax + q..]);
            a.get(&src)
        })
        .collect();
    Nd { shape, v }
}

fn ref_concat(xs: &[&Nd], axis: i64) -> Nd {
    let ax = axis_of(axis, xs[0].shape.len());
    let mut shape = xs[0].shape.clone();
    shape[ax] = xs.iter().map(|x| x.shape[ax]).sum();
    let v = walk(&shape)
        .iter()
        .map(|o| {
            let mut t = o[ax];
            let mut j = 0;
            while t >= xs[j].shape[ax] {
                t -= xs[j].shape[ax];
                j += 1;
            }
            let mut src = o.clone();
            src[ax] = t;
            xs[j].get(&src)
        })
        .collect();
    Nd { shape, v }
}

fn ref_reshape(a: &Nd, target: &[i64]) -> Nd {
    let total = a.v.len() as i64;
    let mut dims: Vec<i64> = target
        .iter()
        .enumerate()
        .map(|(k, &d)| if d == 0 { a.shape[k] as i64 } else { d })
        .collect();
    if let Some(p) = dims.iter().position(|&d| d == -1) {
        let known: i64 = dims.iter().filter(|&&d| d != -1).product();
        dims[p] = total / known;
    }
    a.with_shape(dims.iter().map(|&d| d as usize).collect())
}

fn ref_flatten(a: &Nd, axis: i64) -> Nd {
    let r = a.shape.len() as i64;
    let ax = (if axis < 0 { axis + r } else { axis }) as usize;
    a.with_shape(vec![
        a.shape[..ax].iter().product(),
        a.shape[ax..].iter().product(),
    ])
}

/// numpy `matmul` on exact rationals.
fn ref_matmul(a: &Nd, b: &Nd) -> Nd {
    let av = a.shape.len() == 1;
    let bv = b.shape.len() == 1;
    let ash = if av {
        vec![1, a.shape[0]]
    } else {
        a.shape.clone()
    };
    let bsh = if bv {
        vec![b.shape[0], 1]
    } else {
        b.shape.clone()
    };
    let (m, k) = (ash[ash.len() - 2], ash[ash.len() - 1]);
    let n = bsh[bsh.len() - 1];
    assert_eq!(k, bsh[bsh.len() - 2], "reference: inner dims");
    let ab = ash[..ash.len() - 2].to_vec();
    let bb = bsh[..bsh.len() - 2].to_vec();
    let rank = ab.len().max(bb.len());
    let pad = |s: &[usize]| {
        let mut v = vec![1usize; rank - s.len()];
        v.extend_from_slice(s);
        v
    };
    let (pa, pb) = (pad(&ab), pad(&bb));
    let batch: Vec<usize> = pa.iter().zip(&pb).map(|(&x, &y)| x.max(y)).collect();
    let mut full = batch.clone();
    full.push(m);
    full.push(n);
    let mut v = Vec::new();
    for o in walk(&full) {
        let bo = &o[..rank];
        let (mi, ni) = (o[rank], o[rank + 1]);
        let pick = |p: &[usize], orig: &[usize]| -> Vec<usize> {
            // The operand's own batch index: right-aligned, 1-dims pinned to 0.
            let full_idx: Vec<usize> = bo
                .iter()
                .zip(p)
                .map(|(&i, &d)| if d == 1 { 0 } else { i })
                .collect();
            full_idx[rank - orig.len()..].to_vec()
        };
        let mut s = Rat::ZERO;
        for kk in 0..k {
            let ai = if av {
                vec![kk]
            } else {
                let mut i = pick(&pa, &ab);
                i.extend([mi, kk]);
                i
            };
            let bi = if bv {
                vec![kk]
            } else {
                let mut i = pick(&pb, &bb);
                i.extend([kk, ni]);
                i
            };
            s = radd(s, rmul(a.get(&ai), b.get(&bi)));
        }
        v.push(s);
    }
    let mut shape = batch;
    if !av {
        shape.push(m);
    }
    if !bv {
        shape.push(n);
    }
    Nd { shape, v }
}

/// `y = x Wᵀ + b` for `x: [1, K]`, `w` row-major `[N, K]`.
fn ref_gemm_tb(x: &Nd, w: &[f64], b: &[f64]) -> Nd {
    let k = x.v.len();
    let n = b.len();
    let v = (0..n)
        .map(|j| {
            let mut s = rat(b[j]);
            for i in 0..k {
                s = radd(s, rmul(rat(w[j * k + i]), x.v[i]));
            }
            s
        })
        .collect();
    Nd {
        shape: vec![1, n],
        v,
    }
}

/// `x + c` with `c` broadcast along the last axis.
fn ref_add_last(x: &Nd, c: &[f64]) -> Nd {
    let d = *x.shape.last().expect("rank");
    assert_eq!(d, c.len());
    let v =
        x.v.iter()
            .enumerate()
            .map(|(i, &a)| radd(a, rat(c[i % d])))
            .collect();
    Nd {
        shape: x.shape.clone(),
        v,
    }
}

fn ref_relu(x: &Nd) -> Nd {
    Nd {
        shape: x.shape.clone(),
        v: x.v
            .iter()
            .map(|&a| if a.is_negative() { Rat::ZERO } else { a })
            .collect(),
    }
}

fn nd_const(shape: &[usize], v: &[f64]) -> Nd {
    Nd::input(shape, v)
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const MIN: i64 = i64::MIN;
const MAX: i64 = i64::MAX;

/// `x: in_dims -> nodes -> Reshape(last, [1, -1]) -> y` (the checker's graph
/// output must be `[1, N]`).
fn onnx_of(in_dims: &[i64], mut nodes: Vec<N>, mut inits: Vec<T>, last: &str) -> Vec<u8> {
    nodes.push(node("Reshape", &[last, "flat_shape"], &["y"], vec![]));
    inits.push(ti("flat_shape", &[2], &[1, -1]));
    model("x", in_dims, &nodes, &inits, "y", &[1, 1])
}

fn net_of(in_dims: &[i64], nodes: Vec<N>, inits: Vec<T>, last: &str) -> Result<Net, String> {
    net_from_onnx(&onnx_of(in_dims, nodes, inits, last))
}

/// Distinct, exactly representable values (a routing error moves a value).
fn distinct(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 - 31.0) / 8.0).collect()
}

/// Seeded xorshift dyadics.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A multiple of 1/8 in [-2, 2].
    fn dy(&mut self) -> f64 {
        ((self.next() % 33) as f64 - 16.0) / 8.0
    }

    /// A weight in {-2, -1.5, ..., 2} (multiples of 1/2).
    fn w(&mut self) -> f64 {
        ((self.next() % 9) as f64 - 4.0) / 2.0
    }
}

/// The network output must equal `want` BIT-EXACTLY (pure routing).
fn assert_routes_exactly(net: &Net, x: &[f64], want: &Nd, what: &str) {
    let b = eval_point(net, x);
    let (lo, hi) = (&b.lo[net.output], &b.hi[net.output]);
    assert_eq!(lo.len(), want.v.len(), "{what}: output size");
    for (j, w) in want.v.iter().enumerate() {
        assert!(
            lo[j] == hi[j] && rat(lo[j]) == *w,
            "{what}: y{j} = [{}, {}] but the reference is {}",
            lo[j],
            hi[j],
            w.to_f64_approx()
        );
    }
}

/// The certified point enclosure must contain `want`.
fn assert_encloses(net: &Net, x: &[f64], want: &Nd, what: &str) {
    let b = eval_point(net, x);
    let (lo, hi) = (&b.lo[net.output], &b.hi[net.output]);
    assert_eq!(lo.len(), want.v.len(), "{what}: output size");
    for (j, w) in want.v.iter().enumerate() {
        assert!(
            rat(lo[j]) <= *w && *w <= rat(hi[j]),
            "{what}: y{j} enclosure [{}, {}] misses the reference {}",
            lo[j],
            hi[j],
            w.to_f64_approx()
        );
        assert!(
            hi[j] - lo[j] <= 1e-9 * (1.0 + hi[j].abs()),
            "{what}: y{j} enclosure [{}, {}] is not tight",
            lo[j],
            hi[j]
        );
    }
}

fn refusal(r: Result<Net, String>, needle: &str, what: &str) {
    match r {
        Ok(_) => panic!("{what}: lowered, expected a refusal containing `{needle}`"),
        Err(e) => assert!(
            e.contains(needle),
            "{what}: refusal `{e}` does not name `{needle}`"
        ),
    }
}

fn slice_net(
    in_dims: &[i64],
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
) -> Result<Net, String> {
    let v = |name: &str, d: &[i64]| ti(name, &[d.len() as i64], d);
    let mut inputs = vec!["x", "s", "e"];
    let mut inits = vec![v("s", starts), v("e", ends)];
    match (axes, steps) {
        (None, None) => {}
        (Some(a), None) => {
            inputs.push("a");
            inits.push(v("a", a));
        }
        (a, Some(st)) => {
            if let Some(a) = a {
                inputs.push("a");
                inits.push(v("a", a));
            } else {
                inputs.push("");
            }
            inputs.push("st");
            inits.push(v("st", st));
        }
    }
    net_of(
        in_dims,
        vec![node("Slice", &inputs, &["sl"], vec![])],
        inits,
        "sl",
    )
}

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

/// Every (axis, step, start, end) of a grid that includes negative steps,
/// negative indices, out-of-range indices and the INT64 extremes: the
/// checker and the reference agree on the elements kept, and on WHICH
/// configurations are refused (empty, or the implementation-defined
/// negative-step start).
#[test]
fn slice_routes_exactly_over_a_grid_of_axes_steps_starts_and_ends() {
    let dims = [1usize, 3, 4, 5];
    let x = distinct(60);
    let xr = Nd::input(&dims, &x);
    let (mut ok, mut empty, mut ambiguous) = (0usize, 0usize, 0usize);
    for axis in [1i64, 2, 3, -1, -3] {
        for step in [1i64, 2, 3, -1, -2, -4] {
            for start in [-9i64, -5, -1, 0, 1, 2, 4, 5, 9, MIN, MAX] {
                for end in [-9i64, -5, -1, 0, 1, 3, 5, 9, MIN, MAX] {
                    let what = format!("axis {axis} step {step} start {start} end {end}");
                    let got = slice_net(
                        &[1, 3, 4, 5],
                        &[start],
                        &[end],
                        Some(&[axis]),
                        Some(&[step]),
                    );
                    match ref_slice(&xr, &[start], &[end], Some(&[axis]), Some(&[step])) {
                        SliceRef::Ok(want) => {
                            let net = got.unwrap_or_else(|e| panic!("{what}: refused: {e}"));
                            assert_routes_exactly(&net, &x, &want, &what);
                            ok += 1;
                        }
                        SliceRef::Empty => {
                            refusal(got, "selects no element", &what);
                            empty += 1;
                        }
                        SliceRef::Ambiguous => {
                            refusal(got, "implementation-defined", &what);
                            ambiguous += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!("[slice-grid] ok={ok} empty={empty} ambiguous={ambiguous}");
    assert!(
        ok >= 1000 && empty >= 100 && ambiguous >= 50,
        "grid coverage"
    );
}

#[test]
fn slice_multi_axis_and_default_operands_route_exactly() {
    let dims = [1usize, 3, 4, 5];
    let x = distinct(60);
    let xr = Nd::input(&dims, &x);
    let cases: [(&[i64], &[i64], Option<&[i64]>, Option<&[i64]>); 6] = [
        (&[2, -1], &[MIN, 0], Some(&[1, 3]), Some(&[-1, -2])),
        (&[0, 1, 3], &[1, 3, 99], None, None),
        (&[1], &[MAX], Some(&[-2]), None),
        (&[0, 1, 1], &[1, 3, 4], None, Some(&[1, 1, 2])),
        (&[-2, 4], &[-100, -6], Some(&[2, 3]), Some(&[-3, -1])),
        (&[5, -4], &[0, 9], Some(&[3, 1]), Some(&[-2, 2])),
    ];
    for (k, (s, e, a, st)) in cases.iter().enumerate() {
        let SliceRef::Ok(want) = ref_slice(&xr, s, e, *a, *st) else {
            panic!("case {k}: the reference refuses");
        };
        let net = slice_net(&[1, 3, 4, 5], s, e, *a, *st)
            .unwrap_or_else(|err| panic!("case {k}: refused: {err}"));
        assert_routes_exactly(&net, &x, &want, &format!("multi-axis case {k}"));
        assert!(net.nodes.iter().any(|n| n.op.kind() == "Select"));
    }
}

#[test]
fn slice_refuses_outside_its_form_with_a_precise_reason() {
    let d = [1i64, 3, 4, 5];
    refusal(
        slice_net(&d, &[0, 0], &[1, 1], Some(&[1, -3]), None),
        "repeated",
        "repeated axis",
    );
    refusal(
        slice_net(&d, &[0], &[1], Some(&[4]), None),
        "out of range",
        "axis out of range",
    );
    refusal(
        slice_net(&d, &[0], &[1], Some(&[1]), Some(&[0])),
        "zero step",
        "zero step",
    );
    refusal(
        slice_net(&d, &[0, 0], &[1], None, None),
        "lengths differ",
        "length mismatch",
    );
    refusal(slice_net(&d, &[], &[], None, None), "empty starts", "empty");
    // starts computed from the input.
    refusal(
        net_of(
            &[1, 4],
            vec![node("Slice", &["x", "x", "e"], &["sl"], vec![])],
            vec![ti("e", &[1], &[2])],
            "sl",
        ),
        "computed from the input",
        "activation starts",
    );
    // FLOAT starts.
    refusal(
        net_of(
            &[1, 4],
            vec![node("Slice", &["x", "s", "e"], &["sl"], vec![])],
            vec![tf("s", &[1], &[0.0]), ti("e", &[1], &[2])],
            "sl",
        ),
        "is not INT64",
        "float starts",
    );
    // 2-D starts.
    refusal(
        net_of(
            &[1, 4],
            vec![node("Slice", &["x", "s", "e"], &["sl"], vec![])],
            vec![ti("s", &[1, 1], &[0]), ti("e", &[1], &[2])],
            "sl",
        ),
        "must be 1-D",
        "2-D starts",
    );
    // Slicing a constant is constant folding.
    refusal(
        net_of(
            &[1, 4],
            vec![
                node("Slice", &["k", "s", "e"], &["ks"], vec![]),
                node("Add", &["x", "ks"], &["sl"], vec![]),
            ],
            vec![
                tf("k", &[1, 8], &[1.0; 8]),
                ti("s", &[1], &[0]),
                ti("e", &[1], &[4]),
            ],
            "sl",
        ),
        "constant folding",
        "constant data",
    );
    // The opset-1 attribute form.
    refusal(
        net_of(
            &[1, 4],
            vec![node(
                "Slice",
                &["x"],
                &["sl"],
                vec![A::Is("starts", vec![0]), A::Is("ends", vec![2])],
            )],
            vec![],
            "sl",
        ),
        "attribute form",
        "attribute form",
    );
}

// ---------------------------------------------------------------------------
// Gather
// ---------------------------------------------------------------------------

#[test]
fn gather_routes_constant_indices_exactly_on_every_axis() {
    let dims = [1usize, 3, 4];
    let x = distinct(12);
    let xr = Nd::input(&dims, &x);
    let mut checked = 0;
    for axis in [-3i64, -2, -1, 0, 1, 2] {
        let dim = dims[axis_of(axis, 3)] as i64;
        let variants: Vec<(Vec<usize>, Vec<i64>)> = vec![
            (vec![], vec![0]),
            (vec![], vec![-1]),
            (vec![1], vec![dim - 1]),
            (vec![3], vec![0, -1, 0]),
            (vec![2, 2], vec![dim - 1, 0, -dim, dim - 1]),
        ];
        for (idims, idx) in variants {
            let what = format!("axis {axis} indices {idims:?} {idx:?}");
            let idims_i: Vec<i64> = idims.iter().map(|&d| d as i64).collect();
            let net = net_of(
                &[1, 3, 4],
                vec![node(
                    "Gather",
                    &["x", "i"],
                    &["g"],
                    vec![A::I("axis", axis)],
                )],
                vec![ti("i", &idims_i, &idx)],
                "g",
            )
            .unwrap_or_else(|e| panic!("{what}: refused: {e}"));
            let want = ref_gather(&xr, &idims, &idx, axis);
            assert_routes_exactly(&net, &x, &want, &what);
            checked += 1;
        }
    }
    assert_eq!(checked, 30);
}

#[test]
fn gather_refuses_outside_its_form_with_a_precise_reason() {
    let g = |idx: &[i64], axis: i64| {
        net_of(
            &[1, 3, 4],
            vec![node(
                "Gather",
                &["x", "i"],
                &["g"],
                vec![A::I("axis", axis)],
            )],
            vec![ti("i", &[idx.len() as i64], idx)],
            "g",
        )
    };
    refusal(g(&[3], 1), "out of range", "index past the end");
    refusal(g(&[-4], 1), "out of range", "negative index past the start");
    refusal(g(&[0], 3), "out of range", "axis out of range");
    // Negative indices are opset >= 11 semantics.
    let at = |opset: i64| {
        net_from_onnx(&model_ext(
            "x",
            &[Dm::V(1), Dm::V(3)],
            &[
                node("Gather", &["x", "i"], &["g"], vec![A::I("axis", 1)]),
                node("Reshape", &["g", "fs"], &["y"], vec![]),
            ],
            &[ti("i", &[1], &[-1]), ti("fs", &[2], &[1, -1])],
            "y",
            &[1, 1],
            opset,
        ))
    };
    refusal(at(9), "needs opset >= 11", "negative index at opset 9");
    // review(checker-lowering): the boundary itself (Gather-1 indices are
    // `[0, s-1]`; negative indices are Gather-11 semantics).
    refusal(at(10), "needs opset >= 11", "negative index at opset 10");
    let net = at(11).expect("a negative index lowers at opset 11");
    let x = [0.5, -1.25, 2.0];
    assert_routes_exactly(&net, &x, &Nd::input(&[1, 1], &[2.0]), "opset 11");
    // Indices computed from the input, constant data, FLOAT indices.
    refusal(
        net_of(
            &[1, 3],
            vec![node("Gather", &["k", "x"], &["g"], vec![])],
            vec![tf("k", &[3], &[1.0, 2.0, 3.0])],
            "g",
        ),
        "computed from the input",
        "activation indices",
    );
    refusal(
        net_of(
            &[1, 3],
            vec![
                node("Gather", &["k", "i"], &["kg"], vec![]),
                node("Add", &["x", "kg"], &["g"], vec![]),
            ],
            vec![tf("k", &[3], &[1.0, 2.0, 3.0]), ti("i", &[3], &[0, 1, 2])],
            "g",
        ),
        "constant folding",
        "constant data",
    );
    refusal(
        net_of(
            &[1, 3],
            vec![node("Gather", &["x", "i"], &["g"], vec![A::I("axis", 1)])],
            vec![tf("i", &[1], &[0.0])],
            "g",
        ),
        "is not INT64",
        "float indices",
    );
}

// ---------------------------------------------------------------------------
// Concat
// ---------------------------------------------------------------------------

#[test]
fn concat_routes_activations_exactly_on_every_axis() {
    let dims = [1usize, 2, 3];
    let x = distinct(6);
    let xr = Nd::input(&dims, &x);
    let SliceRef::Ok(a) = ref_slice(&xr, &[0], &[2], Some(&[2]), None) else {
        panic!("reference")
    };
    let SliceRef::Ok(b) = ref_slice(&xr, &[1], &[3], Some(&[2]), None) else {
        panic!("reference")
    };
    let pre = || {
        vec![
            node("Slice", &["x", "s0", "e0", "ax"], &["a"], vec![]),
            node("Slice", &["x", "s1", "e1", "ax"], &["b"], vec![]),
        ]
    };
    let inits = || {
        vec![
            ti("s0", &[1], &[0]),
            ti("e0", &[1], &[2]),
            ti("s1", &[1], &[1]),
            ti("e1", &[1], &[3]),
            ti("ax", &[1], &[2]),
        ]
    };
    let cases: Vec<(Vec<&str>, i64, Nd)> = vec![
        (vec!["a", "b", "a"], 2, ref_concat(&[&a, &b, &a], 2)),
        (vec!["a", "b"], -2, ref_concat(&[&a, &b], -2)),
        (vec!["b", "x", "a"], 2, ref_concat(&[&b, &xr, &a], 2)),
        (vec!["x", "x"], 0, ref_concat(&[&xr, &xr], 0)),
        (vec!["b"], 1, ref_concat(&[&b], 1)),
        (vec!["a", "a"], -3, ref_concat(&[&a, &a], -3)),
    ];
    for (k, (ops, axis, want)) in cases.into_iter().enumerate() {
        let mut nodes = pre();
        nodes.push(node("Concat", &ops, &["c"], vec![A::I("axis", axis)]));
        let net = net_of(&[1, 2, 3], nodes, inits(), "c")
            .unwrap_or_else(|e| panic!("case {k}: refused: {e}"));
        assert_routes_exactly(&net, &x, &want, &format!("concat case {k}"));
    }
}

#[test]
fn concat_refuses_outside_its_form_with_a_precise_reason() {
    let c = |ops: &[&str], attrs: Vec<A>, extra: Vec<N>, inits: Vec<T>| {
        let mut nodes = extra;
        nodes.push(node("Concat", ops, &["c"], attrs));
        net_of(&[1, 2, 3], nodes, inits, "c")
    };
    refusal(
        c(
            &["x", "k"],
            vec![A::I("axis", 2)],
            vec![],
            vec![tf("k", &[1, 2, 1], &[1.0, 2.0])],
        ),
        "constant operand",
        "constant operand",
    );
    refusal(
        c(&["x", "x"], vec![], vec![], vec![]),
        "axis attribute",
        "no axis",
    );
    refusal(
        c(&["x", "x"], vec![A::I("axis", 3)], vec![], vec![]),
        "out of range",
        "axis out of range",
    );
    refusal(
        c(
            &["x", "f"],
            vec![A::I("axis", 1)],
            vec![node("Flatten", &["x"], &["f"], vec![A::I("axis", 1)])],
            vec![],
        ),
        "does not match",
        "rank mismatch",
    );
    refusal(
        c(
            &["x", "t"],
            vec![A::I("axis", 2)],
            vec![node("Slice", &["x", "s", "e", "a"], &["t"], vec![])],
            vec![
                ti("s", &[1], &[0]),
                ti("e", &[1], &[1]),
                ti("a", &[1], &[1]),
            ],
        ),
        "does not match",
        "off-axis dim mismatch",
    );
}

// ---------------------------------------------------------------------------
// Reshape and Flatten
// ---------------------------------------------------------------------------

#[test]
fn reshape_and_flatten_keep_flat_order_for_every_static_target() {
    let dims = [1usize, 2, 3, 4];
    let x = distinct(24);
    let xr = Nd::input(&dims, &x);
    let targets: [&[i64]; 8] = [
        &[1, 24],
        &[0, -1, 2],
        &[4, 0, 3],
        &[24],
        &[2, 3, 4],
        &[-1],
        &[1, 1, 24, 1],
        &[0, 0, 0, 0],
    ];
    for t in targets {
        // Reshape, then reverse the leading axis with a Slice so a wrong
        // shape (not only a wrong order) changes the routed values.
        let want_r = ref_reshape(&xr, t);
        let SliceRef::Ok(want) = ref_slice(&want_r, &[-1], &[MIN], Some(&[0]), Some(&[-1])) else {
            panic!("reference")
        };
        let net = net_of(
            &[1, 2, 3, 4],
            vec![
                node("Reshape", &["x", "t"], &["r"], vec![]),
                node("Slice", &["r", "s", "e", "a", "st"], &["sl"], vec![]),
            ],
            vec![
                ti("t", &[t.len() as i64], t),
                ti("s", &[1], &[-1]),
                ti("e", &[1], &[MIN]),
                ti("a", &[1], &[0]),
                ti("st", &[1], &[-1]),
            ],
            "sl",
        )
        .unwrap_or_else(|e| panic!("target {t:?}: refused: {e}"));
        assert_routes_exactly(&net, &x, &want, &format!("reshape {t:?}"));
        assert_eq!(
            net.tensors[net.by_name["r"]].shape, want_r.shape,
            "reshape {t:?}: the exact ONNX shape is kept"
        );
    }
    for axis in [0i64, 1, 2, 3, 4, -1, -4] {
        let want_f = ref_flatten(&xr, axis);
        let SliceRef::Ok(want) = ref_slice(&want_f, &[-1], &[MIN], Some(&[0]), Some(&[-1])) else {
            panic!("reference")
        };
        let net = net_of(
            &[1, 2, 3, 4],
            vec![
                node("Flatten", &["x"], &["f"], vec![A::I("axis", axis)]),
                node("Slice", &["f", "s", "e", "a", "st"], &["sl"], vec![]),
            ],
            vec![
                ti("s", &[1], &[-1]),
                ti("e", &[1], &[MIN]),
                ti("a", &[1], &[0]),
                ti("st", &[1], &[-1]),
            ],
            "sl",
        )
        .unwrap_or_else(|e| panic!("flatten axis {axis}: refused: {e}"));
        assert_routes_exactly(&net, &x, &want, &format!("flatten {axis}"));
        assert_eq!(net.tensors[net.by_name["f"]].shape, want_f.shape);
    }
}

#[test]
fn reshape_and_flatten_refuse_outside_their_form_with_a_precise_reason() {
    let r = |shape: T, attrs: Vec<A>| {
        net_of(
            &[1, 2, 3, 4],
            vec![node("Reshape", &["x", "t"], &["r"], attrs)],
            vec![shape],
            "r",
        )
    };
    refusal(r(ti("t", &[2], &[-1, -1]), vec![]), "two -1", "two -1");
    refusal(
        r(ti("t", &[2], &[5, 5]), vec![]),
        "element count changes",
        "count change",
    );
    refusal(
        r(ti("t", &[5], &[0, 0, 0, 0, 0]), vec![]),
        "is 0 (copy)",
        "0 past the rank",
    );
    refusal(r(ti("t", &[2], &[-2, 12]), vec![]), "negative dim", "-2");
    refusal(r(ti("t", &[2], &[7, -1]), vec![]), "cannot infer", "infer");
    refusal(
        r(tf("t", &[2], &[1.0, 24.0]), vec![]),
        "is not INT64",
        "float shape",
    );
    refusal(
        r(ti("t", &[1, 2], &[1, 24]), vec![]),
        "must be 1-D",
        "2-D shape",
    );
    refusal(
        r(ti("t", &[2], &[1, 24]), vec![A::I("allowzero", 1)]),
        "allowzero",
        "allowzero",
    );
    refusal(
        net_of(
            &[1, 2],
            vec![node("Reshape", &["x", "x"], &["r"], vec![])],
            vec![],
            "r",
        ),
        "computed from the input",
        "computed shape",
    );
    refusal(
        net_of(
            &[1, 2, 3, 4],
            vec![node("Flatten", &["x"], &["f"], vec![A::I("axis", 5)])],
            vec![],
            "f",
        ),
        "out of range",
        "flatten axis",
    );
}

// ---------------------------------------------------------------------------
// MatMul with a constant operand
// ---------------------------------------------------------------------------

/// One MatMul case: the activation is the input reshaped to `xshape`, the
/// constant has `cshape`, `x_left` puts the activation first.
fn matmul_case(rng: &mut Rng, xshape: &[i64], cshape: &[i64], x_left: bool, x: &[f64]) {
    let cn: usize = cshape.iter().product::<i64>() as usize;
    let c: Vec<f64> = (0..cn).map(|_| rng.w()).collect();
    let ops: [&str; 2] = if x_left { ["xr", "c"] } else { ["c", "xr"] };
    let what = format!("x{xshape:?} c{cshape:?} x_left={x_left}");
    let net = net_of(
        &[1, x.len() as i64],
        vec![
            node("Reshape", &["x", "xs"], &["xr"], vec![]),
            node("MatMul", &ops, &["m"], vec![]),
        ],
        vec![
            ti("xs", &[xshape.len() as i64], xshape),
            tf("c", cshape, &c),
        ],
        "m",
    )
    .unwrap_or_else(|e| panic!("{what}: refused: {e}"));
    let xu: Vec<usize> = xshape.iter().map(|&d| d as usize).collect();
    let cu: Vec<usize> = cshape.iter().map(|&d| d as usize).collect();
    let xa = Nd::input(&xu, x);
    let ca = nd_const(&cu, &c);
    let want = if x_left {
        ref_matmul(&xa, &ca)
    } else {
        ref_matmul(&ca, &xa)
    };
    assert_eq!(
        net.tensors[net.by_name["m"]].shape, want.shape,
        "{what}: output shape"
    );
    assert_encloses(&net, x, &want, &what);
}

#[test]
fn matmul_with_a_constant_on_either_side_matches_numpy_semantics() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let x: Vec<f64> = (0..24).map(|_| rng.dy()).collect();
    let cases: [(&[i64], &[i64], bool); 14] = [
        (&[1, 24], &[24, 3], true),
        (&[1, 1, 24], &[24, 2], true),
        (&[4, 6], &[3, 4], false),
        (&[24], &[24, 2], true),
        (&[2, 3, 4], &[4, 2], true),
        (&[1, 6, 4], &[2, 3, 6], false),
        (&[4, 6], &[4], false),
        (&[4, 6], &[6], true),
        (&[2, 1, 3, 4], &[5, 4, 2], true),
        (&[24], &[24], true),
        (&[24], &[24], false),
        (&[3, 8], &[2, 1, 4, 3], false),
        (&[6, 4], &[4, 6], true),
        (&[2, 12], &[12, 2], true),
    ];
    for (xs, cs, left) in cases {
        matmul_case(&mut rng, xs, cs, left, &x);
    }
}

/// The previously admitted row-vector form lowers to the same Gemm as before
/// (`W` transposed, zero bias): the generalization changes no admitted net.
#[test]
fn matmul_row_vector_lowering_is_the_transposed_weight() {
    let w: Vec<f64> = (0..12).map(|i| f64::from(i) * 0.5 - 3.0).collect();
    let net = net_from_onnx(&model(
        "x",
        &[1, 4],
        &[node("MatMul", &["x", "W"], &["y"], vec![])],
        &[tf("W", &[4, 3], &w)],
        "y",
        &[1, 3],
    ))
    .expect("lowers");
    let Op::Gemm(g) = &net.nodes[0].op else {
        panic!("MatMul lowers to a Gemm");
    };
    assert_eq!((g.n_out, g.n_in), (3, 4));
    for j in 0..3 {
        for i in 0..4 {
            assert_eq!(g.w[j * 4 + i].to_bits(), w[i * 3 + j].to_bits());
        }
    }
    assert!(g.b.iter().all(|&b| b == 0.0));
    assert_eq!(net.tensors[net.output].shape, vec![1, 3]);
}

#[test]
fn matmul_refuses_outside_its_form_with_a_precise_reason() {
    let mm = |xshape: &[i64], cshape: &[i64], x_left: bool, n_in: i64| {
        let cn: usize = cshape.iter().product::<i64>() as usize;
        let ops: [&str; 2] = if x_left { ["xr", "c"] } else { ["c", "xr"] };
        net_of(
            &[1, n_in],
            vec![
                node("Reshape", &["x", "xs"], &["xr"], vec![]),
                node("MatMul", &ops, &["m"], vec![]),
            ],
            vec![
                ti("xs", &[xshape.len() as i64], xshape),
                tf("c", cshape, &vec![0.5; cn.max(1)]),
            ],
            "m",
        )
    };
    refusal(mm(&[4, 6], &[5, 2], true, 24), "inner dims differ", "inner");
    refusal(
        mm(&[2, 3, 4], &[3, 4, 2], true, 24),
        "do not broadcast",
        "batch",
    );
    refusal(mm(&[24], &[], true, 24), "rank-0", "scalar constant");
    // 4160 x 4160 dense entries > 2^24 (and > the 64 x 64 constant).
    refusal(
        mm(&[65, 64], &[64, 64], true, 4160),
        "dense lowering needs",
        "dense cap",
    );
    refusal(
        net_of(
            &[1, 2],
            vec![
                node("MatMul", &["k", "w"], &["kw"], vec![]),
                node("Add", &["x", "kw"], &["m"], vec![]),
            ],
            vec![tf("k", &[1, 2], &[1.0, 2.0]), tf("w", &[2, 2], &[1.0; 4])],
            "m",
        ),
        "both operands are constants",
        "constant folding",
    );
    refusal(
        net_of(
            &[1, 2],
            vec![
                node("Reshape", &["x", "c"], &["xc"], vec![]),
                node("MatMul", &["xc", "x"], &["m"], vec![]),
            ],
            vec![ti("c", &[2], &[2, 1])],
            "m",
        ),
        "activation · activation",
        "bilinear",
    );
}

// ---------------------------------------------------------------------------
// Graph inputs and the batch-requiring forms
// ---------------------------------------------------------------------------

#[test]
fn inputs_of_any_rank_lower_in_flat_c_order() {
    // Rank 1: slice, reshape to [1, 4], Gemm, Relu.
    let w = [1.0, -0.5, 2.0, 0.0, 0.5, 0.5, -1.0, 1.5];
    let b = [0.25, -0.75];
    let x = [0.5, -1.0, 1.25, 2.0, -0.375, 1.0];
    let net = net_of(
        &[6],
        vec![
            node("Slice", &["x", "s", "e"], &["sl"], vec![]),
            node("Reshape", &["sl", "r"], &["row"], vec![]),
            node("Gemm", &["row", "W", "B"], &["z"], vec![A::I("transB", 1)]),
            node("Relu", &["z"], &["h"], vec![]),
        ],
        vec![
            ti("s", &[1], &[1]),
            ti("e", &[1], &[5]),
            ti("r", &[2], &[1, 4]),
            tf("W", &[2, 4], &w),
            tf("B", &[2], &b),
        ],
        "h",
    )
    .expect("a rank-1 input lowers");
    assert_eq!(net.tensors[net.input].shape, vec![6]);
    assert_eq!(net.n_in(), 6);
    let xr = Nd::input(&[6], &x);
    let SliceRef::Ok(sl) = ref_slice(&xr, &[1], &[5], None, None) else {
        panic!("reference")
    };
    let want = ref_relu(&ref_gemm_tb(&sl.with_shape(vec![1, 4]), &w, &b));
    assert_encloses(&net, &x, &want, "rank-1 input");
    // A non-batch leading dim [3, 2]: Gather on axis 0, then Concat.
    let x2 = [0.5, -1.0, 1.25, 2.0, -0.375, 1.0];
    let net = net_of(
        &[3, 2],
        vec![
            node("Gather", &["x", "i"], &["g"], vec![A::I("axis", 0)]),
            node("Concat", &["g", "x"], &["c"], vec![A::I("axis", 0)]),
        ],
        vec![ti("i", &[2], &[2, 0])],
        "c",
    )
    .expect("a [3, 2] input lowers");
    let xr2 = Nd::input(&[3, 2], &x2);
    let want = ref_concat(&[&ref_gather(&xr2, &[2], &[2, 0], 0), &xr2], 0);
    assert_routes_exactly(&net, &x2, &want, "[3, 2] input");
    // A symbolic leading dim of a rank >= 2 input is the batch (resolves to 1).
    let net = net_from_onnx(&model_ext(
        "x",
        &[Dm::P("batch"), Dm::V(3)],
        &[node("Relu", &["x"], &["y"], vec![])],
        &[],
        "y",
        &[1, 3],
        13,
    ))
    .expect("a symbolic batch lowers");
    assert_eq!(net.tensors[net.input].shape, vec![1, 3]);
}

#[test]
fn inputs_and_batch_forms_refuse_outside_their_form_with_a_precise_reason() {
    let relu = |dims: &[Dm]| {
        net_from_onnx(&model_ext(
            "x",
            dims,
            &[node("Relu", &["x"], &["y"], vec![])],
            &[],
            "y",
            &[1, 3],
            13,
        ))
    };
    refusal(relu(&[]), "scalar", "rank-0 input");
    refusal(
        relu(&[Dm::P("n")]),
        "symbolic or non-positive",
        "rank-1 symbolic",
    );
    refusal(
        relu(&[Dm::V(1), Dm::P("n")]),
        "symbolic or non-positive",
        "non-leading symbolic",
    );
    refusal(
        relu(&[Dm::V(1), Dm::V(0)]),
        "symbolic or non-positive",
        "zero dim",
    );
    // The output must be [1, N].
    refusal(
        net_from_onnx(&model(
            "x",
            &[6],
            &[node("Relu", &["x"], &["y"], vec![])],
            &[],
            "y",
            &[6],
        )),
        "graph output must be [1, N]",
        "rank-1 output",
    );
    // Gemm needs [1, K]; Conv needs [1, C, H, W]; BN needs [1, C, ...].
    refusal(
        net_of(
            &[2, 3],
            vec![node("Gemm", &["x", "W"], &["z"], vec![A::I("transB", 1)])],
            vec![tf("W", &[2, 3], &[1.0; 6])],
            "z",
        ),
        "Gemm: input must be [1, K]",
        "gemm batch",
    );
    refusal(
        net_of(
            &[2, 1, 3, 3],
            vec![node("Conv", &["x", "W"], &["z"], vec![])],
            vec![tf("W", &[1, 1, 2, 2], &[1.0; 4])],
            "z",
        ),
        "Conv: input must be [1, C, H, W]",
        "conv batch",
    );
    refusal(
        net_of(
            &[3],
            vec![node(
                "BatchNormalization",
                &["x", "g", "b", "m", "v"],
                &["z"],
                vec![],
            )],
            vec![
                tf("g", &[3], &[1.0; 3]),
                tf("b", &[3], &[0.0; 3]),
                tf("m", &[3], &[0.0; 3]),
                tf("v", &[3], &[1.0; 3]),
            ],
            "z",
        ),
        "BatchNormalization: input must be [1, C, ...]",
        "bn batch",
    );
}

/// The backward charge of a Select is the LARGEST number of outputs that read
/// one input element (a repeated Gather index, a Concat of one tensor with
/// itself): pinned directly, since Higham's slack would hide an undercount.
#[test]
fn select_fan_in_is_the_largest_per_element_multiplicity() {
    let net = net_of(
        &[1, 3],
        vec![
            node("Gather", &["x", "i"], &["g"], vec![A::I("axis", 1)]),
            node("Concat", &["g", "x", "g"], &["c"], vec![A::I("axis", 1)]),
        ],
        vec![ti("i", &[5], &[2, 0, 2, -1, 1])],
        "c",
    )
    .expect("lowers");
    let sel = |name: &str| {
        let t = net.by_name[name];
        let k = net.topo(t).expect("produced");
        match &net.nodes[k].op {
            Op::Select(s) => s.sources(&net.tensors),
            other => panic!("{name}: {other:?}"),
        }
    };
    let (x, g) = (net.by_name["x"], net.by_name["g"]);
    // x[2] is read three times (indices 2, 2, -1).
    assert_eq!(sel("g"), vec![(x, 3)]);
    // Concat(g, x, g): every element of g twice, every element of x once.
    assert_eq!(sel("c"), vec![(g, 2), (x, 1)]);
}

/// A constant that would GROW the activation under ONNX broadcasting is
/// refused (the elementwise lowerings map each activation element once).
#[test]
fn broadcasting_never_grows_the_activation() {
    let add = |kdims: &[i64]| {
        let n: i64 = kdims.iter().product();
        net_of(
            &[1, 3],
            vec![node("Add", &["x", "k"], &["a"], vec![])],
            vec![tf("k", kdims, &vec![0.5; n as usize])],
            "a",
        )
    };
    refusal(add(&[2, 3]), "without growing", "constant grows a dim");
    refusal(
        add(&[2, 1, 3]),
        "would grow",
        "constant adds a non-1 leading dim",
    );
    refusal(add(&[3, 1]), "without growing", "constant column");
    // Right-aligned, non-growing broadcasts lower (with a leading 1 added).
    let net = add(&[1, 1, 3]).expect("a leading 1 broadcasts");
    assert_eq!(net.tensors[net.by_name["a"]].shape, vec![1, 1, 3]);
    let net = add(&[1]).expect("a scalar broadcasts");
    assert_eq!(net.tensors[net.by_name["a"]].shape, vec![1, 3]);
}

/// review(checker-lowering): below opset 7, Add / Sub / Mul / Div broadcast
/// by the LEGACY `broadcast` / `axis` attributes — `B` aligned at `axis`, not
/// right-aligned — so a `[3]` constant against `[1, 3, 2, 3]` with
/// `broadcast = 1, axis = 1` is PER-CHANNEL there, while the right-aligned
/// rule reads it along the last dim: the same bytes, two different networks.
/// The checker must refuse rather than pick one; and from opset 7 on the
/// attributes do not exist (a model carrying them is not well formed).
#[test]
fn legacy_broadcast_attributes_and_pre_opset_7_constant_operands_refuse() {
    let at = |op: &'static str, opset: i64, attrs: Vec<A>| {
        net_from_onnx(&model_ext(
            "x",
            &[Dm::V(1), Dm::V(3), Dm::V(2), Dm::V(3)],
            &[
                node(op, &["x", "k"], &["a"], attrs),
                node("Reshape", &["a", "fs"], &["y"], vec![]),
            ],
            &[tf("k", &[3], &[1.0, 2.0, 4.0]), ti("fs", &[2], &[1, -1])],
            "y",
            &[1, 18],
            opset,
        ))
    };
    for op in ["Add", "Sub", "Mul", "Div"] {
        let legacy = || vec![A::I("broadcast", 1), A::I("axis", 1)];
        refusal(
            at(op, 6, legacy()),
            "legacy broadcast",
            &format!("{op} legacy axis broadcast at opset 6"),
        );
        refusal(
            at(op, 6, vec![]),
            "opset 7",
            &format!("{op} constant operand at opset 6"),
        );
        refusal(
            at(op, 13, legacy()),
            "legacy broadcast",
            &format!("{op} legacy attributes at opset 13"),
        );
        // The same graph without the legacy attributes at opset 7+ is the
        // right-aligned broadcast, and lowers.
        at(op, 7, vec![]).unwrap_or_else(|e| panic!("{op} at opset 7: {e}"));
        at(op, 13, vec![]).unwrap_or_else(|e| panic!("{op} at opset 13: {e}"));
    }
    // Activation + activation of one shape means the same at every opset.
    let same = net_from_onnx(&model_ext(
        "x",
        &[Dm::V(1), Dm::V(3)],
        &[node("Add", &["x", "x"], &["y"], vec![])],
        &[],
        "y",
        &[1, 3],
        6,
    ));
    same.expect("a same-shape activation Add lowers at any opset");
}

/// A Select's scatter is an accumulation like any other: a repeated Gather
/// index whose coefficients cancel catastrophically (`2^54`, 998 ones,
/// `-2^54`: every one is lost in f64) must be charged at the Select's own
/// fan-in, or the certified bound rises above the exact network. The
/// producer-side (Gemm) charge alone is ~24 against a lost mass of 998, so
/// dropping the Select's magnitude term, or counting its fan-in as 1, is
/// caught here — in the back-substitution and in the frozen-dual form.
#[test]
fn select_backward_charges_a_cancelling_fan_in_accumulation() {
    let n = 1000usize;
    let big = 2f64.powi(54);
    let mut w = vec![1.0f64; n];
    w[0] = big;
    w[n - 1] = -big;
    let net = net_from_onnx(&model(
        "x",
        &[1, 1],
        &[
            node("Gather", &["x", "i"], &["g"], vec![A::I("axis", 1)]),
            node("Gemm", &["g", "W"], &["y"], vec![A::I("transB", 1)]),
        ],
        &[
            ti("i", &[n as i64], &vec![0; n]),
            tf("W", &[1, n as i64], &w),
        ],
        "y",
        &[1, 1],
    ))
    .expect("lowers");
    // Σ w = 998 exactly, so y = 998 · x0.
    let exact = |x0: f64| rmul(rat(x0), rat(998.0));
    let (lo, hi) = ([-1.0], [-0.5]);
    let (l, _, _) = lower_bound(&net, &lo, &hi, &[], &[], None, &[(0, 1.0)], 0.0).expect("a bound");
    assert!(l.is_finite(), "{l}");
    for x0 in [-1.0, -0.75, -0.5] {
        assert!(
            rat(l) <= exact(x0),
            "certified {l} above the exact {}",
            998.0 * x0
        );
    }
    let (frame, margin, _, _) =
        fd_state(&net, &lo, &hi, &[], &[(0, 1.0)], 0.0, 0.0, &[]).expect("state");
    assert!(margin.ok);
    assert_eq!((frame.n_gens, frame.in_col[0]), (1, Some(0)));
    for x0 in [-1.0, -0.75, -0.5] {
        let e = rdiv(rsub(rat(x0), rat(frame.mid[0])), rat(frame.rad[0]));
        let ge = rmul(rat(margin.g[0]), e);
        let v = exact(x0);
        assert!(
            rsub(radd(rat(margin.c_lo), ge), rat(margin.err)) <= v
                && v <= radd(radd(rat(margin.c_hi), ge), rat(margin.err)),
            "the frozen-dual form misses the exact {} (err {})",
            998.0 * x0,
            margin.err
        );
    }
}

// ---------------------------------------------------------------------------
// The certified paths through the new forms
// ---------------------------------------------------------------------------

struct RoutingNet {
    onnx: Vec<u8>,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    w3: Vec<f64>,
    b3: Vec<f64>,
}

const PERM: [i64; 9] = [7, 0, 3, 3, 5, 1, 6, 2, 4];

/// `x: [1, 6]`; `lo = x[:, 0:3]`, `hi = x[:, 3:]`; `za = (W1 · loᵀ)ᵀ + b1`
/// (constant on the LEFT of a column), `zb = hi · W2 + b2`; two Relus;
/// `cat = [relu(zb), relu(za)]`; `perm = cat[:, PERM]` (a repeated index);
/// `y = perm · W3ᵀ + b3`. Sites: `za`, `zb`.
fn routing_net(seed: u64) -> RoutingNet {
    let mut rng = Rng(seed | 1);
    let w1: Vec<f64> = (0..12).map(|_| rng.w()).collect();
    let b1: Vec<f64> = (0..4).map(|_| rng.w()).collect();
    let w2: Vec<f64> = (0..12).map(|_| rng.w()).collect();
    let b2: Vec<f64> = (0..4).map(|_| rng.w()).collect();
    let w3: Vec<f64> = (0..18).map(|_| rng.w()).collect();
    let b3: Vec<f64> = (0..2).map(|_| rng.w()).collect();
    let onnx = model(
        "x",
        &[1, 6],
        &[
            node("Slice", &["x", "s0", "e0", "a1"], &["lo"], vec![]),
            node("Slice", &["x", "s1", "e1", "a1"], &["hi"], vec![]),
            node("Reshape", &["lo", "col"], &["loc"], vec![]),
            node("MatMul", &["W1", "loc"], &["m1"], vec![]),
            node("Reshape", &["m1", "row"], &["m1r"], vec![]),
            node("Add", &["m1r", "B1"], &["za"], vec![]),
            node("MatMul", &["hi", "W2"], &["m2"], vec![]),
            node("Add", &["m2", "B2"], &["zb"], vec![]),
            node("Relu", &["za"], &["ha"], vec![]),
            node("Relu", &["zb"], &["hb"], vec![]),
            node("Concat", &["hb", "ha"], &["cat"], vec![A::I("axis", 1)]),
            node("Gather", &["cat", "perm"], &["pg"], vec![A::I("axis", 1)]),
            node("Gemm", &["pg", "W3", "B3"], &["y"], vec![A::I("transB", 1)]),
        ],
        &[
            ti("s0", &[1], &[0]),
            ti("e0", &[1], &[3]),
            ti("s1", &[1], &[3]),
            ti("e1", &[1], &[MAX]),
            ti("a1", &[1], &[1]),
            ti("col", &[2], &[3, 1]),
            ti("row", &[2], &[1, 4]),
            ti("perm", &[9], &PERM),
            tf("W1", &[4, 3], &w1),
            tf("B1", &[4], &b1),
            tf("W2", &[3, 4], &w2),
            tf("B2", &[4], &b2),
            tf("W3", &[2, 9], &w3),
            tf("B3", &[2], &b3),
        ],
        "y",
        &[1, 2],
    );
    RoutingNet {
        onnx,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
    }
}

/// Exact `(za, zb, y)` of [`routing_net`] at `x`, by the reference ops.
fn routing_exact(r: &RoutingNet, x: &[f64]) -> (Nd, Nd, Nd) {
    let xr = Nd::input(&[1, 6], x);
    let SliceRef::Ok(lo) = ref_slice(&xr, &[0], &[3], Some(&[1]), None) else {
        panic!("reference")
    };
    let SliceRef::Ok(hi) = ref_slice(&xr, &[3], &[MAX], Some(&[1]), None) else {
        panic!("reference")
    };
    let m1 = ref_matmul(&nd_const(&[4, 3], &r.w1), &lo.with_shape(vec![3, 1]));
    let za = ref_add_last(&m1.with_shape(vec![1, 4]), &r.b1);
    let zb = ref_add_last(&ref_matmul(&hi, &nd_const(&[3, 4], &r.w2)), &r.b2);
    let cat = ref_concat(&[&ref_relu(&zb), &ref_relu(&za)], 1);
    let pg = ref_gather(&cat, &[9], &PERM, 1);
    let y = ref_gemm_tb(&pg, &r.w3, &r.b3);
    (za, zb, y)
}

/// IBP and the α/β back-substitution through Slice, MatMul (constant left),
/// Concat and a Gather with a repeated index never exceed the exact network:
/// the IBP box contains every sampled exact value, and every certified lower
/// bound (with and without Relu literals, β and α hints) is below the exact
/// margin at every sampled point of its region, as is its linear form.
#[test]
fn certified_bounds_through_routing_forms_never_exceed_the_exact_network() {
    let mut checked = 0usize;
    for seed in 1..=24u64 {
        let r = routing_net(seed * 7919);
        let net = net_from_onnx(&r.onnx).expect("the routing net lowers");
        let (ta, tb) = (net.by_name["za"], net.by_name["zb"]);
        assert!(net.is_site(ta) && net.is_site(tb));
        let mut rng = Rng(seed * 104_729);
        let c: Vec<f64> = (0..6).map(|_| rng.dy() * 0.5).collect();
        let lo: Vec<f64> = c.iter().map(|v| v - 0.75).collect();
        let hi: Vec<f64> = c.iter().map(|v| v + 0.75).collect();
        let pts: Vec<Vec<f64>> = (0..48)
            .map(|_| {
                (0..6)
                    .map(|i| lo[i] + 1.5 * ((rng.next() % 9) as f64) / 8.0)
                    .collect()
            })
            .collect();
        // (1) IBP containment.
        let fwd = forward(&net, &lo, &hi);
        for x in &pts {
            let (za, zb, y) = routing_exact(&r, x);
            for (t, nd) in [(ta, &za), (tb, &zb), (net.output, &y)] {
                for (j, v) in nd.v.iter().enumerate() {
                    assert!(
                        rat(fwd.lo[t][j]) <= *v && *v <= rat(fwd.hi[t][j]),
                        "seed {seed}: IBP misses an exact value"
                    );
                }
            }
        }
        // (2) Root bound and form.
        let obj = [(0usize, 1.0), (1usize, -1.0)];
        let margin = |x: &[f64]| {
            let (_, _, y) = routing_exact(&r, x);
            rsub(y.v[0], y.v[1])
        };
        if let Some((l, lin, cst)) = lower_bound(&net, &lo, &hi, &[], &[], None, &obj, 0.0) {
            for x in &pts {
                let m = margin(x);
                if l.is_finite() {
                    assert!(rat(l) <= m, "seed {seed}: root bound above the network");
                }
                let mut f = rat(cst);
                for (a, xi) in lin.iter().zip(x) {
                    f = radd(f, rmul(rat(*a), rat(*xi)));
                }
                assert!(f <= m, "seed {seed}: root linear form above the network");
                checked += 1;
            }
        }
        // (3) Literal regions through both sites, β and an α hint.
        let (za0, zb0, _) = routing_exact(&r, &pts[0]);
        let (ia, ib) = ((seed % 4) as usize, ((seed / 4) % 4) as usize);
        let lits = [
            ("za", ia, !za0.v[ia].is_negative()),
            ("zb", ib, !zb0.v[ib].is_negative()),
        ];
        let alpha = alpha_map(
            &net,
            &[("za", (ia + 1) % 4, 0.5), ("zb", (ib + 2) % 4, 0.25)],
        );
        if let Some((l, _, _)) =
            lower_bound(&net, &lo, &hi, &lits, &[0.5, 0.25], Some(&alpha), &obj, 0.0)
        {
            if l.is_finite() {
                for x in &pts {
                    let (za, zb, _) = routing_exact(&r, x);
                    let inside =
                        [(&za, ia, lits[0].2), (&zb, ib, lits[1].2)]
                            .iter()
                            .all(|(z, i, pos)| {
                                if *pos {
                                    !z.v[*i].is_negative()
                                } else {
                                    !z.v[*i].is_positive()
                                }
                            });
                    if inside {
                        assert!(rat(l) <= margin(x), "seed {seed}: leaf bound above");
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked >= 500, "only {checked} comparisons");
}

/// B2.3: the frozen-dual generator forms through the routing forms are
/// two-sided enclosures of the exact network (the same charges as the
/// back-substitution, with the relaxation identity), and every generator
/// stays in [-1, 1] on the box.
#[test]
fn fd_forms_through_routing_forms_enclose_the_exact_network() {
    for seed in 1..=12u64 {
        let r = routing_net(seed * 31_337);
        let net = net_from_onnx(&r.onnx).expect("lowers");
        let mut rng = Rng(seed * 65_537);
        let lo: Vec<f64> = (0..6).map(|_| rng.dy() * 0.5 - 0.5).collect();
        let hi: Vec<f64> = lo.iter().map(|v| v + 1.0).collect();
        let sites = [("za", 0usize), ("za", 3), ("zb", 1), ("zb", 2)];
        let (frame, margin, rows, _) = fd_state(
            &net,
            &lo,
            &hi,
            &[("za", 1, 0.5)],
            &[(0, 1.0), (1, -1.0)],
            0.0,
            0.0,
            &sites,
        )
        .expect("state");
        assert!(margin.ok && rows.iter().all(|f| f.ok), "seed {seed}");
        for _ in 0..24 {
            let x: Vec<f64> = (0..6)
                .map(|i| lo[i] + ((rng.next() % 9) as f64) / 8.0)
                .collect();
            let (za, zb, y) = routing_exact(&r, &x);
            let mut e = vec![Rat::ZERO; frame.n_gens];
            for (i, col) in frame.in_col.iter().enumerate() {
                if let Some(col) = *col {
                    e[col] = rdiv(rsub(rat(x[i]), rat(frame.mid[i])), rat(frame.rad[i]));
                }
            }
            for (name, z) in [("za", &za), ("zb", &zb)] {
                for (i, rl) in frame.relax[net.by_name[name]].iter().enumerate() {
                    if let Relu::Sym { col, lam, mu } = *rl {
                        let zi = z.v[i];
                        let h = if zi.is_negative() { Rat::ZERO } else { zi };
                        e[col] = rdiv(rsub(rsub(h, rmul(rat(lam), zi)), rat(mu)), rat(mu));
                    }
                }
            }
            for ej in &e {
                assert!(
                    ej.abs() <= Rat::ONE,
                    "seed {seed}: a generator leaves [-1, 1]"
                );
            }
            let dot = |g: &[f64]| {
                g.iter()
                    .zip(&e)
                    .fold(Rat::ZERO, |a, (gi, ej)| radd(a, rmul(rat(*gi), *ej)))
            };
            let m = rsub(y.v[0], y.v[1]);
            let ge = dot(&margin.g);
            assert!(
                rsub(radd(rat(margin.c_lo), ge), rat(margin.err)) <= m
                    && m <= radd(radd(rat(margin.c_hi), ge), rat(margin.err)),
                "seed {seed}: the objective form misses the network"
            );
            for (k, &(name, i)) in sites.iter().enumerate() {
                let z = if name == "za" { &za } else { &zb };
                let f = &rows[k];
                let c = radd(rat(f.c_lo), dot(&f.g));
                let rr = radd(rat(f.err), rsub(rat(f.c_hi), rat(f.c_lo)));
                assert!(
                    rsub(z.v[i], c).abs() <= rr,
                    "seed {seed}: site row {k} escapes its remainder"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// End to end
// ---------------------------------------------------------------------------

// The routed network is `support::routed_mlp_onnx` (shared with the
// legacy-forms cross test in `tests/legacy.rs`).

/// The base MLP's four-leaf bundle verifies through the routed network (its
/// sites resolve by ONNX name, its input literals by the flat index of the
/// rank-1 input), the margin is certified below the exact minimum, and the
/// base suite's mutations still fail.
#[test]
fn a_bundle_through_the_routing_forms_verifies_end_to_end() {
    const B0: f64 = 1.0 / 1_048_576.0;
    let fx = Fixture::new("routed_accept", routed_mlp_onnx(B0, 0.0), &mlp_vnnlib());
    let o = fx.check("full", &finish(&mlp_bundle_lines(&fx, &mlp_leaves())));
    assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
    assert_eq!(o.exit_code, 0);
    let r = report(&o);
    assert_eq!(
        r["nodes"]["confirmed"].as_u64(),
        Some(4),
        "{}",
        o.report_json
    );
    let ops = &r["network"]["ops"];
    assert_eq!(ops["Select"].as_u64(), Some(4), "{}", o.report_json);
    // A flipped literal leaves a cover gap; a dropped leaf too.
    let mut leaves = mlp_leaves();
    leaves[0].0[1] = L::R(0, 1, false);
    let o = fx.check("flip", &finish(&mlp_bundle_lines(&fx, &leaves)));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    let mut leaves = mlp_leaves();
    leaves.remove(3);
    let o = fx.check("drop", &finish(&mlp_bundle_lines(&fx, &leaves)));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    // A FALSE property (b0 < b1) through the same forms is never verified.
    let bad = Fixture::new("routed_false", routed_mlp_onnx(-0.25, 0.0), &mlp_vnnlib());
    let o = bad.check("false", &finish(&mlp_bundle_lines(&bad, &mlp_leaves())));
    assert_ne!(o.verdict, Verdict::Verified, "{}", o.report_json);
    assert_ne!(o.exit_code, 0);
}

/// A refused form reaches the report as exit 31 `onnx_lowering` with the
/// lowering's own reason.
#[test]
fn a_refused_form_is_unsupported_with_its_reason_in_the_report() {
    let onnx = onnx_of(
        &[1, 2],
        vec![node("Slice", &["x", "s", "e", "a", "st"], &["sl"], vec![])],
        vec![
            ti("s", &[1], &[-9]),
            ti("e", &[1], &[MIN]),
            ti("a", &[1], &[1]),
            ti("st", &[1], &[-1]),
        ],
        "sl",
    );
    let fx = Fixture::new(
        "routed_refused",
        onnx,
        &robustness_vnnlib(&["-1", "-1"], &["1", "1"], 2, 0, &[1]),
    );
    let o = fx.check("b", &finish(&mlp_bundle_lines(&fx, &mlp_leaves())));
    assert_eq!(o.exit_code, 31, "{}", o.report_json);
    assert_eq!(o.verdict, Verdict::Unsupported);
    assert!(reason_codes(&o).contains(&"onnx_lowering".to_string()));
    assert!(
        o.report_json.contains("implementation-defined"),
        "{}",
        o.report_json
    );
}
