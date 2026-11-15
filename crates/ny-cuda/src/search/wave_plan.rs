// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The device WAVE search's PURE side (#wave-beta-bab, ENGINE_PROGRAM M6;
//! `margin_row.tree_mode: wave_cuda`): the lowering of a [`WaveProblem`] into
//! the flat tables `kernels/wave_search.cu` reads, the launch ABI shared with
//! the kernels ([`WaveArgs`], [`AdamConsts`] and the `#define`s below), the
//! lease accounting (every device buffer, from ONE layout the planner and the
//! allocator both read), and a host known-answer oracle for the init
//! self-test. Nothing here touches the driver, so all of it is tested on hosts
//! without a GPU; it names no other module of this crate, so a standalone
//! harness can compile it next to ny-bbe's CPU reference.
//!
//! # The reference
//!
//! ny-bbe's `wave.rs` (`upload::<f32>`, `backward_batch::<f32>`,
//! `branch_scores::<f32>`) is the FUNCTIONAL REFERENCE. The lowering below
//! reproduces its `WaveUpload::<f32>::lower` value for value (`Program::lower`'s
//! f64 -> f32 roundings, `SiteState::<f32>::new`'s chords and adaptive slopes,
//! the warm slopes, the box midpoints and radii) and the kernels reproduce its
//! arithmetic operation for operation, in the same order, in IEEE binary32 /
//! binary64 round-to-nearest without contraction — so every output the lane
//! reads is the CPU backend's, bit for bit (the sign of a zero and a NaN's
//! payload excepted: no output reads either). ny-cuda may not depend on ny-bbe
//! (`crate_dag_ratchet`), so the arithmetic both sides must share — Adam's
//! constants and bias corrections, the learning-rate decay and ingest, the
//! warm-hint ingest — lives ONCE in `ny_core::search` (`WAVE_ADAM_*`,
//! `wave_adam_bias`, `wave_lr_decay`, `wave_ingest_rates`, `wave_warm_alpha` /
//! `wave_warm_beta`), read by the reference and by this host side alike; ny-cli's
//! bit-identity KATs (the only crate that sees both backends) pin the kernels'
//! own arithmetic against it.
//!
//! # Trust
//!
//! UNTRUSTED end to end. Every output is a hint the lane certifies with its
//! own Outward evaluator (hints clamped at ingest) or refutes and splits; THE
//! DEVICE NEVER CERTIFIES.

use ny_core::search::{Refusal, SearchOp, WaveProblem, GATHER_PAD, WAVE_NONE};

/// "No position" / padding sentinel of every `u32` table (`NONE` in the kernels).
pub(crate) const NONE: u32 = u32::MAX;
const _: () = assert!(NONE == WAVE_NONE && NONE == GATHER_PAD);

/// Threads per CTA of the element kernels (`WAVE_THREADS`).
pub(crate) const WAVE_THREADS: u32 = 256;
/// Threads per CTA of `wave_babsr` (one thread per domain; `WAVE_BABSR_THREADS`).
pub(crate) const WAVE_BABSR_THREADS: u32 = 128;
/// Most branch candidates per domain (`WAVE_MAX_K`; `ny_core::search`'s
/// `WAVE_MAX_BRANCH_K`, the CPU search's `MAX_BRANCH_K`).
pub(crate) const WAVE_MAX_K: usize = ny_core::search::WAVE_MAX_BRANCH_K;
/// `i32` fields per op descriptor (`OP_FIELDS`).
pub(crate) const OP_FIELDS: usize = 16;

/// Op kinds (`OP_*`).
pub(crate) const OP_CONV: i32 = 0;
pub(crate) const OP_AFFINE: i32 = 1;
pub(crate) const OP_RELU: i32 = 2;
pub(crate) const OP_ADD: i32 = 3;
pub(crate) const OP_COPY: i32 = 4;
pub(crate) const OP_GEMM: i32 = 5;

/// Op descriptor fields (`F_*`). Tensor fields are float offsets inside one
/// domain's tensor block; parameter fields are offsets into `fp` / `ip`.
pub(crate) const F_KIND: usize = 0;
pub(crate) const F_OUT: usize = 1;
pub(crate) const F_NOUT: usize = 2;
pub(crate) const F_IN: usize = 3;
pub(crate) const F_NIN: usize = 4;
pub(crate) const F_RHS: usize = 5;
pub(crate) const F_A: usize = 6;
pub(crate) const F_B: usize = 7;
pub(crate) const F_C: usize = 8;
pub(crate) const F_W: usize = 9;
pub(crate) const F_BIAS: usize = 10;
pub(crate) const F_G: usize = 11;
pub(crate) const F_INV_START: usize = 12;
pub(crate) const F_INV_ENT: usize = 13;
pub(crate) const F_WT: usize = 14;
pub(crate) const F_SITE: usize = 15;

/// Per-domain state bits (`ST_*`). `ST_ACTIVE` and `ST_HAS_BEST` are read only
/// by the kernel source; the host names them for the ABI ratchet (tests).
#[cfg(test)]
pub(crate) const ST_ACTIVE: u32 = 1;
pub(crate) const ST_GRAD: u32 = 2;
pub(crate) const ST_IMPROVED: u32 = 4;
pub(crate) const ST_EARLY: u32 = 8;
#[cfg(test)]
pub(crate) const ST_HAS_BEST: u32 = 16;

/// Per-domain float scalars (`DOM_F`, `DF_*`). `DF_BIAS`, `DF_BIAS_ABS`,
/// `DF_LB` and `DF_SCALE` are read only by the kernel source; the host names
/// them for the ABI ratchet (tests).
pub(crate) const DOM_F: usize = 6;
#[cfg(test)]
pub(crate) const DF_BIAS: usize = 0;
#[cfg(test)]
pub(crate) const DF_BIAS_ABS: usize = 1;
#[cfg(test)]
pub(crate) const DF_LB: usize = 2;
#[cfg(test)]
pub(crate) const DF_SCALE: usize = 3;
pub(crate) const DF_BEST_LB: usize = 4;
pub(crate) const DF_BEST_SCALE: usize = 5;
/// Per-domain integer scalars (`DOM_U`, `DU_*`).
pub(crate) const DOM_U: usize = 2;
pub(crate) const DU_STATE: usize = 0;
pub(crate) const DU_EVALS: usize = 1;

/// `wave_init` phases (`PHASE_*`).
pub(crate) const PHASE_DOMAINS: i32 = 0;
pub(crate) const PHASE_LITERALS: i32 = 1;
pub(crate) const PHASE_SEED: i32 = 2;

// The wave search's constants: Adam's moments, its step floor, the
// per-iteration learning-rate decay (abc's BaB `lr_decay`) and the fallback
// learning rates of a non-finite / non-positive configured one — ONE copy in
// `ny_core::search`, the one the CPU reference (ny-bbe `wave.rs`) reads.
pub(crate) use ny_core::search::{
    wave_adam_bias, wave_ingest_rates, wave_lr_decay, WAVE_ADAM_B1 as ADAM_B1,
    WAVE_ADAM_B2 as ADAM_B2, WAVE_ADAM_EPS as ADAM_EPS,
};

/// Domains of one slab at most (the lane's `WAVE_BATCH_MAX`; well inside the
/// grid's `y` limit of 65,535).
pub(crate) const MAX_SLAB_DOMAINS: usize = 16_384;
/// Literal capacity of a slab per domain (`nl = max(stride, nd x this)`): a
/// tree `LITS_PER_DOMAIN` deep fills a slab by domains; a deeper wave cuts
/// its slabs by literals instead (a single domain always fits: its literals
/// are distinct universe positions, so at most `stride`).
pub(crate) const LITS_PER_DOMAIN: usize = 32;
/// Work of one iteration of one slab (a backward walk, a forward walk and an
/// Adam step over every domain), in multiply-add units: the slab is sized so
/// one iteration stays near this, and the deadline is polled between
/// iterations. UNMEASURED — an informal, non-quiesced harness run on the GB10
/// (2026-09-29: 256 cifar-shaped domains x 9 passes in 0.085 s) puts 2^32 at
/// roughly 20 ms per iteration. It bounds the cancellation latency and the
/// slab's width (a large net's slab holds `2^32 / pass_work` domains), never
/// correctness: a slab's outputs do not depend on its size.
pub(crate) const PASS_WORK_BUDGET: u64 = 1 << 32;

// ---------------------------------------------------------------------------
//  The device buffers: ONE table (names, element types, lengths, ABI order)
// ---------------------------------------------------------------------------

/// Declares every device buffer of the wave search ONCE: its [`WaveBuf`]
/// variant, its name (the kernel's `WaveArgs` pointer field, ratcheted by
/// `wave_kernel_abi_matches_the_rust_side`), its element type and its length
/// class. The launch ABI's pointer order, the planner's layout, the
/// allocator, the executor's pointer and length lookups and the launch bounds
/// check all read this table; nothing else lists the buffers.
macro_rules! wave_buffers {
    ($($var:ident = $name:literal: $elem:ident, $len:ident;)*) => {
        /// A device buffer of the wave search, in the kernel ABI's pointer
        /// order (see `wave_buffers!`).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub(crate) enum WaveBuf {
            $($var,)*
        }

        impl WaveBuf {
            /// Every buffer, in the ABI's pointer order.
            pub(crate) const ALL: [WaveBuf; WAVE_ARGS_POINTERS] = [$(WaveBuf::$var,)*];

            /// The buffer's name (the kernel's `WaveArgs` field).
            #[must_use]
            pub(crate) const fn name(self) -> &'static str {
                match self {
                    $(WaveBuf::$var => $name,)*
                }
            }

            /// The buffer's element type.
            #[must_use]
            pub(crate) const fn elem(self) -> Elem {
                match self {
                    $(WaveBuf::$var => Elem::$elem,)*
                }
            }

            const fn len_class(self) -> BufLen {
                match self {
                    $(WaveBuf::$var => BufLen::$len,)*
                }
            }
        }
    };
}

/// How a buffer's length follows from the problem and the slab's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BufLen {
    /// The upload's op descriptors (`prog.ops`).
    Ops,
    /// The upload's float tables (`prog.fp_len`).
    Fp,
    /// The upload's integer tables (`prog.ip_len`).
    Ip,
    /// The upload's objective classes (`prog.classes`).
    Classes,
    /// `nd x t`: one tensor block per domain.
    DomTensor,
    /// `nd x stride`: one slope per universe position per domain.
    DomUniverse,
    /// `nl`: one per literal.
    Literal,
    /// `nd`: one per domain.
    Domain,
    /// `nd x DOM_F`.
    DomFloats,
    /// `nd x DOM_U`.
    DomWords,
    /// `nd x WAVE_MAX_K`: the branch candidates.
    DomCandidates,
}

wave_buffers! {
    Ops = "ops": I32, Ops;
    Fp = "fp": F32, Fp;
    Ip = "ip": U32, Ip;
    Classes = "classes": U32, Classes;
    Coef = "coef": F32, DomTensor;
    Val = "val": F32, DomTensor;
    Alpha = "alpha": F32, DomUniverse;
    BestAlpha = "best_alpha": F32, DomUniverse;
    GAlpha = "g_alpha": F32, DomUniverse;
    Ma = "ma": F64, DomUniverse;
    Va = "va": F64, DomUniverse;
    Pin = "pin": U32, DomUniverse;
    LitPos = "lit_pos": U32, Literal;
    LitAct = "lit_act": U32, Literal;
    LitDom = "lit_dom": U32, Literal;
    Beta = "beta": F32, Literal;
    BestBeta = "best_beta": F32, Literal;
    GBeta = "g_beta": F32, Literal;
    Mb = "mb": F64, Literal;
    Vb = "vb": F64, Literal;
    Rows = "rows": U32, Domain;
    DomF = "dom_f": F32, DomFloats;
    DomU = "dom_u": U32, DomWords;
    Bpos = "bpos": U32, DomCandidates;
    Bscore = "bscore": F32, DomCandidates;
}

/// The upload's buffers: the first [`UPLOAD_BUFFERS`] of [`WaveBuf::ALL`]
/// (the problem's tables); the rest are one slab's.
pub(crate) const UPLOAD_BUFFERS: usize = 4;
/// One slab's buffers.
pub(crate) const SLAB_BUFFERS: usize = WAVE_ARGS_POINTERS - UPLOAD_BUFFERS;

const _: () = {
    let mut i = 0;
    while i < WAVE_ARGS_POINTERS {
        let upload = matches!(
            WaveBuf::ALL[i].len_class(),
            BufLen::Ops | BufLen::Fp | BufLen::Ip | BufLen::Classes
        );
        assert!(
            upload == (i < UPLOAD_BUFFERS),
            "the upload's tables come first"
        );
        assert!(WaveBuf::ALL[i] as usize == i, "ALL is in declaration order");
        i += 1;
    }
};

impl WaveBuf {
    /// The buffer called `name`, if any.
    #[must_use]
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.name() == name)
    }

    /// Elements of this buffer in an upload of `prog` (0 for a slab buffer).
    #[must_use]
    fn upload_len(self, prog: &WaveProg) -> usize {
        match self.len_class() {
            BufLen::Ops => prog.ops.len(),
            BufLen::Fp => prog.fp_len,
            BufLen::Ip => prog.ip_len,
            BufLen::Classes => prog.classes.len(),
            _ => 0,
        }
    }

    /// Elements of this buffer in a slab of `nd` domains and `nl` literals over
    /// a tensor block of `t` floats and `stride` universe positions (0 for an
    /// upload buffer).
    #[must_use]
    fn slab_len(self, t: usize, stride: usize, nd: usize, nl: usize) -> usize {
        match self.len_class() {
            BufLen::Ops | BufLen::Fp | BufLen::Ip | BufLen::Classes => 0,
            BufLen::DomTensor => nd.saturating_mul(t),
            BufLen::DomUniverse => nd.saturating_mul(stride),
            BufLen::Literal => nl,
            BufLen::Domain => nd,
            BufLen::DomFloats => nd.saturating_mul(DOM_F),
            BufLen::DomWords => nd.saturating_mul(DOM_U),
            BufLen::DomCandidates => nd.saturating_mul(WAVE_MAX_K),
        }
    }
}

// ---------------------------------------------------------------------------
//  The launch ABI (one struct, passed BY VALUE to every kernel)
// ---------------------------------------------------------------------------

/// Every kernel's first argument, field for field `struct WaveArgs` of
/// `kernels/wave_search.cu` (the ABI ratchet
/// `wave_kernel_abi_matches_the_rust_side` compares the two): the device
/// address of every buffer in [`WaveBuf::ALL`] order (the kernel's
/// `WAVE_ARGS_POINTERS` pointer fields; an array of `u64` has their exact
/// `repr(C)` layout), then the sizes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct WaveArgs {
    /// Device addresses, indexed by [`WaveBuf`].
    pub(crate) ptrs: [u64; WAVE_ARGS_POINTERS],
    pub(crate) t: i64,
    pub(crate) nops: i32,
    pub(crate) n_in: i32,
    pub(crate) out_off: i32,
    pub(crate) target: i32,
    pub(crate) stride: i32,
    pub(crate) nd: i32,
    pub(crate) nl: i32,
    pub(crate) has_alpha: i32,
    pub(crate) s_l: i32,
    pub(crate) s_u: i32,
    pub(crate) s_slope: i32,
    pub(crate) s_icpt: i32,
    pub(crate) s_adapt: i32,
    pub(crate) s_pos: i32,
    pub(crate) alpha0: i32,
    pub(crate) mid: i32,
    pub(crate) rad: i32,
    pub(crate) uni_flat: i32,
    pub(crate) uni_lam: i32,
    pub(crate) k: i32,
}

/// `sizeof(WaveArgs)` on both sides (`static_assert` in the kernel source).
pub(crate) const WAVE_ARGS_BYTES: usize = 288;
const _: () = assert!(size_of::<WaveArgs>() == WAVE_ARGS_BYTES);
const _: () = assert!(align_of::<WaveArgs>() == 8);

/// The number of pointer fields of [`WaveArgs`] (they come first): one per
/// [`WaveBuf`].
pub(crate) const WAVE_ARGS_POINTERS: usize = 25;

/// The field names of [`WaveArgs`] in declaration order, with their C kind
/// (`p` pointer, `l` 64-bit integer, `i` 32-bit integer): what the ABI ratchet
/// compares against the kernel source. The pointer names ARE the buffer
/// table's ([`WaveBuf::name`]); the size fields follow, in declaration order.
/// Only the ABI ratchet (tests) reads it.
#[cfg(test)]
pub(crate) const WAVE_ARGS_FIELDS: [(&str, char); WAVE_ARGS_POINTERS + 21] = {
    const SCALARS: [(&str, char); 21] = [
        ("t", 'l'),
        ("nops", 'i'),
        ("n_in", 'i'),
        ("out_off", 'i'),
        ("target", 'i'),
        ("stride", 'i'),
        ("nd", 'i'),
        ("nl", 'i'),
        ("has_alpha", 'i'),
        ("s_l", 'i'),
        ("s_u", 'i'),
        ("s_slope", 'i'),
        ("s_icpt", 'i'),
        ("s_adapt", 'i'),
        ("s_pos", 'i'),
        ("alpha0", 'i'),
        ("mid", 'i'),
        ("rad", 'i'),
        ("uni_flat", 'i'),
        ("uni_lam", 'i'),
        ("k", 'i'),
    ];
    let mut out = [("", 'i'); WAVE_ARGS_POINTERS + 21];
    let mut i = 0;
    while i < WAVE_ARGS_POINTERS {
        out[i] = (WaveBuf::ALL[i].name(), 'p');
        i += 1;
    }
    let mut j = 0;
    while j < SCALARS.len() {
        out[WAVE_ARGS_POINTERS + j] = SCALARS[j];
        j += 1;
    }
    out
};

impl WaveArgs {
    /// The size fields of a launch over `nd` domains and `nl` literals of
    /// `prog` (`has_alpha`: the slab's slopes were uploaded; `k`: branch
    /// candidates per domain). Pointers are zero until [`Self::with_ptrs`].
    #[must_use]
    pub(crate) fn sized(prog: &WaveProg, nd: usize, nl: usize, has_alpha: bool, k: usize) -> Self {
        // Every value below was bounded by `WaveProg::lower` / the plan
        // (offsets and sizes fit i32, nd <= MAX_SLAB_DOMAINS, k <= 16).
        let i = |v: usize| i32::try_from(v).unwrap_or(i32::MAX);
        let s = &prog.sec;
        Self {
            t: i64::try_from(prog.t).unwrap_or(i64::MAX),
            nops: i(prog.nops()),
            n_in: i(prog.n_in),
            out_off: i(prog.out_off),
            target: i(prog.target),
            stride: i(prog.stride),
            nd: i(nd),
            nl: i(nl),
            has_alpha: i32::from(has_alpha),
            s_l: i(s.s_l),
            s_u: i(s.s_u),
            s_slope: i(s.s_slope),
            s_icpt: i(s.s_icpt),
            s_adapt: i(s.s_adapt),
            s_pos: i(s.s_pos),
            alpha0: i(s.alpha0),
            mid: i(s.mid),
            rad: i(s.rad),
            uni_flat: i(s.uni_flat),
            uni_lam: i(s.uni_lam),
            k: i(k),
            ..Self::default()
        }
    }

    /// Fill every pointer field with `ptr(<buffer>)`.
    #[must_use]
    pub(crate) fn with_ptrs(mut self, ptr: impl Fn(WaveBuf) -> u64) -> Self {
        for (slot, buf) in self.ptrs.iter_mut().zip(WaveBuf::ALL) {
            *slot = ptr(buf);
        }
        self
    }

    /// The pointer fields in declaration order (test and launch-check view).
    #[must_use]
    pub(crate) fn pointers(&self) -> [u64; WAVE_ARGS_POINTERS] {
        self.ptrs
    }
}

/// `wave_adam`'s second argument, field for field `struct AdamConsts` of the
/// kernel source: the CPU search's f64 Adam constants for ONE iteration,
/// computed on the host by the reference's own expressions ([`Self::at`]), so
/// the device never evaluates a `powi`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct AdamConsts {
    pub(crate) b1: f64,
    pub(crate) omb1: f64,
    pub(crate) b2: f64,
    pub(crate) omb2: f64,
    pub(crate) eps: f64,
    pub(crate) c1: f64,
    pub(crate) c2: f64,
    pub(crate) lr_a: f64,
    pub(crate) lr_b: f64,
}

/// `sizeof(AdamConsts)`.
pub(crate) const ADAM_CONSTS_BYTES: usize = 72;
const _: () = assert!(size_of::<AdamConsts>() == ADAM_CONSTS_BYTES);

/// [`AdamConsts`]' field names in declaration order (all `double`). Only the
/// ABI ratchet (tests) reads it.
#[cfg(test)]
pub(crate) const ADAM_CONSTS_FIELDS: [&str; 9] = [
    "b1", "omb1", "b2", "omb2", "eps", "c1", "c2", "lr_a", "lr_b",
];

impl AdamConsts {
    /// The constants of iteration `it` (0-based) for configured learning rates
    /// `lr_alpha` / `lr_beta`, exactly as the CPU search forms them:
    /// `t = it + 1`, `c1 = 1 - B1^t`, `c2 = 1 - B2^t`, step `lr * 0.98^it`
    /// (a non-finite or non-positive rate takes the reference's default).
    #[must_use]
    pub(crate) fn at(it: usize, lr_alpha: f64, lr_beta: f64) -> Self {
        // `ny_core::search`'s ingest and schedule: the CPU reference's own.
        let (lr_a, lr_b) = wave_ingest_rates(lr_alpha, lr_beta);
        let t = i32::try_from(it.saturating_add(1)).unwrap_or(i32::MAX);
        let decay = wave_lr_decay(it);
        let (c1, c2) = wave_adam_bias(t);
        Self {
            b1: ADAM_B1,
            omb1: 1.0 - ADAM_B1,
            b2: ADAM_B2,
            omb2: 1.0 - ADAM_B2,
            eps: ADAM_EPS,
            c1,
            c2,
            lr_a: lr_a * decay,
            lr_b: lr_b * decay,
        }
    }
}

// ---------------------------------------------------------------------------
//  The lowering (WaveUpload::<f32>::lower, flattened for the kernels)
// ---------------------------------------------------------------------------

/// Offsets of the lowering's fixed sections: `s_*` / `alpha0` / `mid` / `rad`
/// into `fp`; `s_pos` / `uni_flat` / `uni_lam` into `ip`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Sections {
    pub(crate) s_l: usize,
    pub(crate) s_u: usize,
    pub(crate) s_slope: usize,
    pub(crate) s_icpt: usize,
    pub(crate) s_adapt: usize,
    pub(crate) s_pos: usize,
    pub(crate) alpha0: usize,
    pub(crate) mid: usize,
    pub(crate) rad: usize,
    pub(crate) uni_flat: usize,
    pub(crate) uni_lam: usize,
}

/// A lowered wave problem: what the upload copies to the device, plus the
/// host's own view of it (the op table drives the launch sequence).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WaveProg {
    /// `nops x OP_FIELDS` op descriptors.
    pub(crate) ops: Vec<i32>,
    /// Float tables: the site sections, the warm slopes, the box midpoints and
    /// radii, then every op's parameters.
    pub(crate) fp: Vec<f32>,
    /// Integer tables: the site universe positions, the universe's site and
    /// coefficient offsets, then every conv's gather and inverse gather.
    pub(crate) ip: Vec<u32>,
    /// The objective classes (row `r` is `y_target - y_classes[r]`).
    pub(crate) classes: Vec<u32>,
    /// Floats of one domain's tensor block (every tensor, the input first).
    pub(crate) t: usize,
    /// Input width (tensor 0 sits at offset 0).
    pub(crate) n_in: usize,
    /// Offset of the objective tensor (the last op's output).
    pub(crate) out_off: usize,
    /// The target class.
    pub(crate) target: usize,
    /// Universe positions (the slope stride).
    pub(crate) stride: usize,
    /// Neurons over every ReLU site.
    pub(crate) sites: usize,
    /// The fixed sections.
    pub(crate) sec: Sections,
    /// Multiply-add units of one iteration of one domain.
    pub(crate) pass_work: u64,
    /// Elements of `fp` / `ip` as lowered (they stay the device buffers'
    /// sizes after [`Self::strip`] drops the host copies).
    pub(crate) fp_len: usize,
    pub(crate) ip_len: usize,
}

fn too_large(what: &str) -> Refusal {
    Refusal::Invalid(format!(
        "cuda wave search: {what} exceeds the kernels' i32 index range"
    ))
}

/// `v as i32`, refusing past `i32::MAX`.
fn i32_of(v: usize, what: &str) -> Result<i32, Refusal> {
    i32::try_from(v).map_err(|_| too_large(what))
}

/// ny-bbe `SiteState::<f32>::new` for one neuron: `(slope, intercept,
/// adaptive)` of the upper chord and CROWN's adaptive lower slope, in f32.
#[must_use]
pub(crate) fn site_neuron(l: f32, u: f32) -> (f32, f32, f32) {
    let (sl, ic) = if !(l < 0.0 && u > 0.0) {
        (0.0, 0.0)
    } else if l.is_finite() && u.is_finite() {
        let sl = u / (u - l);
        (sl, -(sl * l))
    } else if u.is_finite() {
        (0.0, u)
    } else {
        (0.0, f32::INFINITY)
    };
    (sl, ic, if u >= -l { 1.0 } else { 0.0 })
}

impl WaveProg {
    /// Ops.
    #[must_use]
    pub(crate) fn nops(&self) -> usize {
        self.ops.len() / OP_FIELDS
    }

    /// Op `k`'s descriptor.
    #[must_use]
    pub(crate) fn op(&self, k: usize) -> &[i32] {
        &self.ops[k * OP_FIELDS..(k + 1) * OP_FIELDS]
    }

    /// Drop the host copies of the parameter tables once they are on the
    /// device (the launch sequence reads only the descriptors and sizes).
    pub(crate) fn strip(&mut self) {
        self.fp = Vec::new();
        self.ip = Vec::new();
    }

    /// Lower and validate `problem` exactly as ny-bbe's
    /// `WaveUpload::<f32>::lower` (and `upload`'s ReLU check before it), into
    /// the flat tables the kernels read.
    pub(crate) fn lower(problem: &WaveProblem) -> Result<Self, Refusal> {
        let net = &problem.net;
        if !net.ops.iter().any(|op| matches!(op, SearchOp::Relu { .. })) {
            return Err(Refusal::Nothing("no ReLU to relax"));
        }
        let tsize = problem.validate().map_err(Refusal::Invalid)?;
        if tsize.iter().any(|&n| n >= NONE as usize) {
            return Err(Refusal::Invalid(
                "a tensor is too wide for u32 neuron indices".into(),
            ));
        }
        let mut toff = Vec::with_capacity(tsize.len());
        let mut t = 0usize;
        for &n in &tsize {
            toff.push(t);
            t = t
                .checked_add(n)
                .ok_or_else(|| too_large("the tensor block"))?;
        }
        i32_of(t, "the tensor block")?;
        let nops = net.ops.len();

        // ReLU sites in op order, their boxes and flat neuron offsets.
        let mut site_of_op = vec![usize::MAX; nops];
        let mut soff: Vec<usize> = Vec::new();
        let mut boxes: Vec<(&[f64], &[f64])> = Vec::new();
        let mut sites = 0usize;
        for (k, op) in net.ops.iter().enumerate() {
            if let SearchOp::Relu { input, .. } = op {
                let b = problem
                    .boxes
                    .iter()
                    .find(|b| b.op == k)
                    .ok_or_else(|| Refusal::Invalid(format!("no box for ReLU op {k}")))?;
                site_of_op[k] = soff.len();
                soff.push(sites);
                boxes.push((&b.l, &b.u));
                sites = sites
                    .checked_add(tsize[*input])
                    .ok_or_else(|| too_large("the site neurons"))?;
            }
        }

        let mut fp: Vec<f32> = Vec::new();
        let mut ip: Vec<u32> = Vec::new();
        let mut sec = Sections::default();
        // The site sections: l, u, chord slope, chord intercept, adaptive.
        let (mut sl, mut su, mut slope, mut icpt, mut adapt) = (
            Vec::with_capacity(sites),
            Vec::with_capacity(sites),
            Vec::with_capacity(sites),
            Vec::with_capacity(sites),
            Vec::with_capacity(sites),
        );
        for (l, u) in &boxes {
            for (&l, &u) in l.iter().zip(u.iter()) {
                let (l, u) = (l as f32, u as f32);
                let (s, i, a) = site_neuron(l, u);
                sl.push(l);
                su.push(u);
                slope.push(s);
                icpt.push(i);
                adapt.push(a);
            }
        }
        for (field, table) in [
            (&mut sec.s_l, &sl),
            (&mut sec.s_u, &su),
            (&mut sec.s_slope, &slope),
            (&mut sec.s_icpt, &icpt),
            (&mut sec.s_adapt, &adapt),
        ] {
            *field = fp.len();
            fp.extend_from_slice(table);
        }
        // The universe: per site neuron its position, per position its flat
        // site index and its coefficient offset (the ReLU's OUTPUT tensor,
        // whose backward coefficient is the reference's `lam`).
        sec.s_pos = ip.len();
        ip.resize(ip.len() + sites, NONE);
        let stride = problem.universe.len();
        let mut uni_flat = Vec::with_capacity(stride);
        let mut uni_lam = Vec::with_capacity(stride);
        for (p, site) in problem.universe.iter().enumerate() {
            let op = site.op as usize;
            let s = site_of_op
                .get(op)
                .copied()
                .filter(|&s| s != usize::MAX)
                .ok_or_else(|| Refusal::Invalid(format!("universe op {op} is not a ReLU")))?;
            let flat = soff[s] + site.neuron as usize;
            ip[sec.s_pos + flat] = u32::try_from(p).map_err(|_| too_large("the universe"))?;
            uni_flat.push(u32::try_from(flat).map_err(|_| too_large("the site neurons"))?);
            let lam = toff[op + 1] + site.neuron as usize;
            uni_lam.push(u32::try_from(lam).map_err(|_| too_large("the tensor block"))?);
        }
        sec.uni_flat = ip.len();
        ip.extend_from_slice(&uni_flat);
        sec.uni_lam = ip.len();
        ip.extend_from_slice(&uni_lam);
        // Warm slopes: a finite alpha0 clamped into [0, 1], else adaptive.
        sec.alpha0 = fp.len();
        for (p, &a) in problem.alpha0.iter().enumerate() {
            fp.push(if a.is_finite() {
                a.clamp(0.0, 1.0) as f32
            } else {
                adapt[uni_flat[p] as usize]
            });
        }
        sec.mid = fp.len();
        fp.extend(
            problem
                .lo
                .iter()
                .zip(&problem.hi)
                .map(|(&l, &h)| f64::midpoint(l, h) as f32),
        );
        sec.rad = fp.len();
        fp.extend(
            problem
                .lo
                .iter()
                .zip(&problem.hi)
                .map(|(&l, &h)| ((h - l) * 0.5) as f32),
        );

        // The ops.
        let cvt = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let mut ops = vec![0i32; nops * OP_FIELDS];
        let mut work: u64 = problem.net.n_in as u64;
        for (k, op) in net.ops.iter().enumerate() {
            let d = &mut ops[k * OP_FIELDS..(k + 1) * OP_FIELDS];
            let out = k + 1;
            d[F_OUT] = i32_of(toff[out], "the tensor block")?;
            d[F_NOUT] = i32_of(tsize[out], "a tensor")?;
            let n_out = tsize[out] as u64;
            match op {
                SearchOp::Conv(c) => {
                    let n_src = tsize[c.input];
                    d[F_KIND] = OP_CONV;
                    d[F_IN] = i32_of(toff[c.input], "the tensor block")?;
                    d[F_NIN] = i32_of(n_src, "a tensor")?;
                    d[F_A] = i32_of(c.cout, "a conv")?;
                    d[F_B] = i32_of(c.k, "a conv")?;
                    d[F_C] = i32_of(c.p, "a conv")?;
                    d[F_W] = i32_of(fp.len(), "the float tables")?;
                    fp.extend(cvt(&c.weight));
                    d[F_BIAS] = i32_of(fp.len(), "the float tables")?;
                    fp.extend(cvt(&c.bias));
                    // The forward reads the gather in its own `[k][p]` layout.
                    d[F_G] = i32_of(ip.len(), "the integer tables")?;
                    ip.extend(
                        c.gather
                            .iter()
                            .map(|&g| if g == GATHER_PAD { NONE } else { g }),
                    );
                    // The backward GATHERS: per input element, its (sp, t)
                    // taps in (sp, t) order, so the kernel adds them in the
                    // reference's scatter order (output index, then tap).
                    let mut count = vec![0u32; n_src + 1];
                    for &g in &c.gather {
                        if g != GATHER_PAD {
                            count[g as usize + 1] += 1;
                        }
                    }
                    for j in 0..n_src {
                        count[j + 1] = count[j]
                            .checked_add(count[j + 1])
                            .ok_or_else(|| too_large("a conv's taps"))?;
                    }
                    let entries = count[n_src] as usize;
                    let mut ent = vec![0u32; 2 * entries];
                    let mut fill = count.clone();
                    for sp in 0..c.p {
                        for tap in 0..c.k {
                            let g = c.gather[tap * c.p + sp];
                            if g == GATHER_PAD {
                                continue;
                            }
                            let e = fill[g as usize] as usize;
                            ent[2 * e] = sp as u32;
                            ent[2 * e + 1] = tap as u32;
                            fill[g as usize] += 1;
                        }
                    }
                    d[F_INV_START] = i32_of(ip.len(), "the integer tables")?;
                    ip.extend_from_slice(&count);
                    d[F_INV_ENT] = i32_of(ip.len(), "the integer tables")?;
                    ip.extend_from_slice(&ent);
                    let macs = (c.cout as u64)
                        .saturating_mul(c.k as u64)
                        .saturating_mul(c.p as u64);
                    work = work.saturating_add(macs.saturating_mul(2).saturating_add(n_out));
                }
                SearchOp::Affine {
                    input,
                    scale,
                    shift,
                } => {
                    d[F_KIND] = OP_AFFINE;
                    d[F_IN] = i32_of(toff[*input], "the tensor block")?;
                    d[F_NIN] = i32_of(tsize[*input], "a tensor")?;
                    d[F_W] = i32_of(fp.len(), "the float tables")?;
                    fp.extend(cvt(scale));
                    d[F_BIAS] = i32_of(fp.len(), "the float tables")?;
                    fp.extend(cvt(shift));
                    work = work.saturating_add(n_out.saturating_mul(3));
                }
                SearchOp::Relu { input, .. } => {
                    d[F_KIND] = OP_RELU;
                    d[F_IN] = i32_of(toff[*input], "the tensor block")?;
                    d[F_NIN] = i32_of(tsize[*input], "a tensor")?;
                    d[F_SITE] = i32_of(soff[site_of_op[k]], "the site neurons")?;
                    work = work.saturating_add(n_out.saturating_mul(3));
                }
                SearchOp::Add { lhs, rhs } => {
                    d[F_KIND] = OP_ADD;
                    d[F_IN] = i32_of(toff[*lhs], "the tensor block")?;
                    d[F_NIN] = i32_of(tsize[*lhs], "a tensor")?;
                    d[F_RHS] = i32_of(toff[*rhs], "the tensor block")?;
                    work = work.saturating_add(n_out.saturating_mul(3));
                }
                SearchOp::Flatten { input } => {
                    d[F_KIND] = OP_COPY;
                    d[F_IN] = i32_of(toff[*input], "the tensor block")?;
                    d[F_NIN] = i32_of(tsize[*input], "a tensor")?;
                    work = work.saturating_add(n_out.saturating_mul(2));
                }
                SearchOp::Gemm {
                    input,
                    n_out,
                    n_in,
                    weight,
                    bias,
                } => {
                    d[F_KIND] = OP_GEMM;
                    d[F_IN] = i32_of(toff[*input], "the tensor block")?;
                    d[F_NIN] = i32_of(*n_in, "a tensor")?;
                    d[F_A] = i32_of(*n_out, "a gemm")?;
                    d[F_B] = i32_of(*n_in, "a gemm")?;
                    d[F_W] = i32_of(fp.len(), "the float tables")?;
                    fp.extend(cvt(weight));
                    d[F_BIAS] = i32_of(fp.len(), "the float tables")?;
                    fp.extend(cvt(bias));
                    // The forward reads the TRANSPOSE (`[n_in][n_out]`), so a
                    // warp's loads of one input column coalesce.
                    d[F_WT] = i32_of(fp.len(), "the float tables")?;
                    let w = &fp[d[F_W] as usize..d[F_W] as usize + n_out * n_in].to_vec();
                    for i in 0..*n_in {
                        for o in 0..*n_out {
                            fp.push(w[o * n_in + i]);
                        }
                    }
                    let macs = (*n_out as u64).saturating_mul(*n_in as u64);
                    work =
                        work.saturating_add(macs.saturating_mul(2).saturating_add(*n_out as u64));
                }
            }
        }
        i32_of(fp.len(), "the float tables")?;
        i32_of(ip.len(), "the integer tables")?;
        i32_of(stride, "the universe")?;
        let prog = Self {
            ops,
            fp,
            ip,
            classes: problem
                .classes
                .iter()
                .map(|&c| u32::try_from(c).map_err(|_| too_large("a class")))
                .collect::<Result<_, _>>()?,
            t,
            n_in: net.n_in,
            out_off: toff[nops],
            target: problem.target,
            stride,
            sites,
            sec,
            pass_work: work.saturating_add((stride as u64).saturating_mul(4)),
            fp_len: 0,
            ip_len: 0,
        };
        let prog = Self {
            fp_len: prog.fp.len(),
            ip_len: prog.ip.len(),
            ..prog
        };
        prog.check()?;
        Ok(prog)
    }

    /// Defence in depth: every descriptor's reads and writes lie inside the
    /// tables and the tensor block (a lowering bug refuses here instead of
    /// becoming an out-of-bounds device access).
    pub(crate) fn check(&self) -> Result<(), Refusal> {
        let bad = |why: String| Err(Refusal::Invalid(format!("cuda wave lowering: {why}")));
        let (nf, ni, t) = (self.fp.len(), self.ip.len(), self.t);
        let within = |off: i32, len: usize, cap: usize| {
            off >= 0
                && (off as usize)
                    .checked_add(len)
                    .is_some_and(|end| end <= cap)
        };
        let s = &self.sec;
        for (off, len) in [
            (s.s_l, self.sites),
            (s.s_u, self.sites),
            (s.s_slope, self.sites),
            (s.s_icpt, self.sites),
            (s.s_adapt, self.sites),
            (s.alpha0, self.stride),
            (s.mid, self.n_in),
            (s.rad, self.n_in),
        ] {
            if off.checked_add(len).is_none_or(|end| end > nf) {
                return bad("a float section is out of range".into());
            }
        }
        for (off, len) in [
            (s.s_pos, self.sites),
            (s.uni_flat, self.stride),
            (s.uni_lam, self.stride),
        ] {
            if off.checked_add(len).is_none_or(|end| end > ni) {
                return bad("an integer section is out of range".into());
            }
        }
        for p in 0..self.stride {
            if self.ip[s.uni_flat + p] as usize >= self.sites
                || self.ip[s.uni_lam + p] as usize >= t
            {
                return bad(format!("universe position {p} is out of range"));
            }
        }
        // The ReLU kernels index slopes and pins at `d * stride + pos` for
        // every site neuron's position that is not NONE.
        if self.ip[s.s_pos..s.s_pos + self.sites]
            .iter()
            .any(|&p| p != NONE && p as usize >= self.stride)
        {
            return bad("a site neuron's universe position is out of range".into());
        }
        if self.n_in > t || self.out_off >= t || self.target >= t - self.out_off {
            return bad("the input or the objective is out of range".into());
        }
        let n_out_obj = t - self.out_off;
        if self.classes.iter().any(|&c| c as usize >= n_out_obj) {
            return bad("an objective class is out of range".into());
        }
        for k in 0..self.nops() {
            let d = self.op(k);
            let (out, nout, inp, nin) = (d[F_OUT], d[F_NOUT] as usize, d[F_IN], d[F_NIN] as usize);
            if !within(out, nout, t) || !within(inp, nin, t) || d[F_NOUT] < 0 || d[F_NIN] < 0 {
                return bad(format!("op {k}: a tensor is out of the block"));
            }
            let ok = match d[F_KIND] {
                OP_CONV => {
                    let (cout, taps, p) = (d[F_A] as usize, d[F_B] as usize, d[F_C] as usize);
                    let gather_ok = within(d[F_G], taps * p, ni)
                        && self.ip[d[F_G] as usize..d[F_G] as usize + taps * p]
                            .iter()
                            .all(|&g| g == NONE || (g as usize) < nin);
                    let starts_ok = within(d[F_INV_START], nin + 1, ni);
                    let entries = if starts_ok {
                        self.ip[d[F_INV_START] as usize + nin] as usize
                    } else {
                        0
                    };
                    let ent_ok = starts_ok
                        && within(d[F_INV_ENT], 2 * entries, ni)
                        && self.ip[d[F_INV_START] as usize..=d[F_INV_START] as usize + nin]
                            .windows(2)
                            .all(|w| w[0] <= w[1])
                        && self.ip[d[F_INV_ENT] as usize..d[F_INV_ENT] as usize + 2 * entries]
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .all(|e| (e[0] as usize) < p && (e[1] as usize) < taps);
                    cout * p == nout
                        && within(d[F_W], cout * taps, nf)
                        && within(d[F_BIAS], cout, nf)
                        && gather_ok
                        && ent_ok
                }
                OP_AFFINE => nin == nout && within(d[F_W], nout, nf) && within(d[F_BIAS], nout, nf),
                OP_RELU => {
                    nin == nout
                        && d[F_SITE] >= 0
                        && (d[F_SITE] as usize)
                            .checked_add(nout)
                            .is_some_and(|end| end <= self.sites)
                }
                OP_ADD => nin == nout && within(d[F_RHS], nout, t),
                OP_COPY => nin == nout,
                OP_GEMM => {
                    let (n_out, n_in) = (d[F_A] as usize, d[F_B] as usize);
                    n_out == nout
                        && n_in == nin
                        && within(d[F_W], n_out * n_in, nf)
                        && within(d[F_BIAS], n_out, nf)
                        && within(d[F_WT], n_out * n_in, nf)
                }
                _ => false,
            };
            if !ok {
                return bad(format!("op {k}: a descriptor is out of range"));
            }
            // A tensor never feeds its own op (SearchNet validation) and the
            // block lays tensors out in id order, so every input tensor ENDS
            // at or before the op's output starts: the backward's reads of
            // the output (the last CTA's bias chain) and writes of the input
            // (the gather CTAs) never overlap within one launch. `out > inp`
            // alone would admit a partial overlap (a data race).
            let precedes = |off: i32, len: usize| {
                (off as usize)
                    .checked_add(len)
                    .is_some_and(|end| end <= out as usize)
            };
            if !precedes(inp, nin) || (d[F_KIND] == OP_ADD && !precedes(d[F_RHS], nout)) {
                return bad(format!("op {k}: its input does not precede its output"));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
//  The device buffers' layout (one table, read by the planner AND the
//  allocator; see `wave_buffers!`)
// ---------------------------------------------------------------------------

/// Element type of a device buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Elem {
    F32,
    F64,
    I32,
    U32,
}

impl Elem {
    #[must_use]
    pub(crate) const fn bytes(self) -> u64 {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 => 8,
        }
    }
}

/// Every buffer of one upload: `(name, elem, len)` (from the one buffer
/// table, `wave_buffers!`); the allocator allocates exactly these (`max(len,
/// 1)` elements).
#[must_use]
pub(crate) fn upload_buffers(prog: &WaveProg) -> [(&'static str, Elem, usize); UPLOAD_BUFFERS] {
    std::array::from_fn(|i| {
        let b = WaveBuf::ALL[i];
        (b.name(), b.elem(), b.upload_len(prog))
    })
}

/// Every buffer of one slab of `nd` domains and `nl` literals over a tensor
/// block of `t` floats and `stride` universe positions (from the one buffer
/// table, `wave_buffers!`). A launch over `c <= nd` domains and `m <= nl`
/// literals indexes at most `launch_needs(t, stride, c, m)` (see
/// [`launch_needs`]), which is this table's lengths at `(c, m)`.
#[must_use]
pub(crate) fn slab_buffers(
    t: usize,
    stride: usize,
    nd: usize,
    nl: usize,
) -> [(&'static str, Elem, usize); SLAB_BUFFERS] {
    std::array::from_fn(|i| {
        let b = WaveBuf::ALL[UPLOAD_BUFFERS + i];
        (b.name(), b.elem(), b.slab_len(t, stride, nd, nl))
    })
}

/// The elements of each slab buffer a launch over `nd` domains and `nl`
/// literals may index (every kernel indexes a domain's rows at `d < nd` and a
/// literal at `g < nl`; `wave_babsr` writes `nd x k <= nd x WAVE_MAX_K`
/// candidates). Checked against the buffers ACTUALLY allocated before every
/// launch.
#[must_use]
pub(crate) fn launch_needs(
    t: usize,
    stride: usize,
    nd: usize,
    nl: usize,
) -> [(&'static str, usize); SLAB_BUFFERS] {
    slab_buffers(t, stride, nd, nl).map(|(name, _, len)| (name, len))
}

fn bytes_of(buffers: &[(&'static str, Elem, usize)]) -> u64 {
    buffers
        .iter()
        .map(|&(_, e, len)| (len.max(1) as u64).saturating_mul(e.bytes()))
        .fold(0u64, u64::saturating_add)
}

/// Device bytes of one upload.
#[must_use]
pub(crate) fn upload_bytes(prog: &WaveProg) -> u64 {
    bytes_of(&upload_buffers(prog))
}

/// Device bytes of one slab.
#[must_use]
pub(crate) fn slab_bytes(t: usize, stride: usize, nd: usize, nl: usize) -> u64 {
    bytes_of(&slab_buffers(t, stride, nd, nl))
}

/// Literal capacity of a slab of `nd` domains.
#[must_use]
pub(crate) fn literals_for(stride: usize, nd: usize) -> usize {
    stride.max(nd.saturating_mul(LITS_PER_DOMAIN)).max(1)
}

/// One upload's admission: the bytes it holds and its slab capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WavePlan {
    /// Bytes of the uploaded problem.
    pub(crate) upload: u64,
    /// Domains a slab holds.
    pub(crate) nd: usize,
    /// Literals a slab holds.
    pub(crate) nl: usize,
    /// Bytes of the slab buffers.
    pub(crate) slab: u64,
}

impl WavePlan {
    /// Every device byte the upload holds.
    #[must_use]
    pub(crate) fn total(&self) -> u64 {
        self.upload.saturating_add(self.slab)
    }
}

/// Domains of one slab within the per-iteration work budget (at least one,
/// at most [`MAX_SLAB_DOMAINS`]), before the lease caps it.
#[must_use]
pub(crate) fn budget_domains(prog: &WaveProg) -> usize {
    usize::try_from(PASS_WORK_BUDGET / prog.pass_work.max(1))
        .unwrap_or(usize::MAX)
        .clamp(1, MAX_SLAB_DOMAINS)
}

/// Admit an upload of `prog` under `cap` bytes: the problem plus a slab of
/// the most domains (and their literal capacity) the work budget asks for
/// that fits. The minimum is the problem plus a one-domain slab; below it the
/// refusal is [`Refusal::Memory`] with that minimum need.
pub(crate) fn plan(prog: &WaveProg, cap: u64) -> Result<WavePlan, Refusal> {
    let upload = upload_bytes(prog);
    let (t, s) = (prog.t, prog.stride);
    let need = |nd: usize| upload.saturating_add(slab_bytes(t, s, nd, literals_for(s, nd)));
    let min_need = need(1);
    if min_need > cap {
        return Err(Refusal::Memory {
            need: min_need,
            cap,
        });
    }
    // The largest domain capacity that fits (monotone in `nd`).
    let (mut lo, mut hi) = (1usize, budget_domains(prog));
    if need(hi) <= cap {
        lo = hi;
    }
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if need(mid) <= cap {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let nl = literals_for(s, lo);
    Ok(WavePlan {
        upload,
        nd: lo,
        nl,
        slab: slab_bytes(t, s, lo, nl),
    })
}

/// Blocks of [`WAVE_THREADS`] covering `n` elements (at least one).
#[must_use]
pub(crate) fn blocks(n: usize) -> u32 {
    u32::try_from(n.div_ceil(WAVE_THREADS as usize).max(1)).unwrap_or(u32::MAX)
}

// ---------------------------------------------------------------------------
//  The init self-test's fixture and its exact oracle
// ---------------------------------------------------------------------------

/// A search scalar of the oracle (`f32` for the sequential transcription the
/// exactness test compares, `f64` for the oracle itself).
pub(crate) trait OracleReal:
    Copy
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    fn of(v: f32) -> Self;
    fn abs(self) -> Self;
}

impl OracleReal for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn of(v: f32) -> Self {
        v
    }
    fn abs(self) -> Self {
        f32::abs(self)
    }
}

impl OracleReal for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn of(v: f32) -> Self {
        f64::from(v)
    }
    fn abs(self) -> Self {
        f64::abs(self)
    }
}

/// One domain of the oracle: its objective row, literals `(position,
/// active, multiplier)` and slopes (`stride` values, already sanitised).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OracleDomain {
    pub(crate) row: usize,
    pub(crate) lits: Vec<(u32, bool, f32)>,
    pub(crate) alpha: Vec<f32>,
}

/// The oracle's answer for one domain: the bound, its absolute-term sum and
/// the backward coefficient at every universe position (the BaBSR `lam`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OraclePass<R> {
    pub(crate) lb: R,
    pub(crate) scale: R,
    pub(crate) lam: Vec<R>,
}

/// ONE backward pass (a `Bound` pass of the CPU reference) over the LOWERED
/// program, in the reference's scatter order: the known-answer oracle of the
/// device self-test. On the dyadic fixture every partial sum is exact, so the
/// f32 and f64 instantiations agree bit for bit
/// (`wave_self_test_fixture_is_exact`) and the device has exactly one correct
/// answer, whatever its summation order.
#[must_use]
pub(crate) fn oracle_pass<R: OracleReal>(prog: &WaveProg, dom: &OracleDomain) -> OraclePass<R> {
    let s = &prog.sec;
    let mut coef = vec![R::ZERO; prog.t];
    // pin per universe position: 0 free, else (1 + literal index).
    let mut pin = vec![0usize; prog.stride];
    for (g, lit) in dom.lits.iter().enumerate() {
        pin[lit.0 as usize] = g + 1;
    }
    let out = prog.out_off;
    coef[out + prog.target] = coef[out + prog.target] + R::ONE;
    let c = prog.classes[dom.row] as usize;
    coef[out + c] = coef[out + c] - R::ONE;
    let (mut bias, mut babs) = (R::ZERO, R::ZERO);
    let fpv = |i: usize| R::of(prog.fp[i]);
    for k in (0..prog.nops()).rev() {
        let d = prog.op(k);
        let (o0, n, i0) = (d[F_OUT] as usize, d[F_NOUT] as usize, d[F_IN] as usize);
        match d[F_KIND] {
            OP_CONV => {
                let (taps, p) = (d[F_B] as usize, d[F_C] as usize);
                let (w, b, g) = (d[F_W] as usize, d[F_BIAS] as usize, d[F_G] as usize);
                for idx in 0..n {
                    let v = coef[o0 + idx];
                    if v == R::ZERO {
                        continue;
                    }
                    let (o, sp) = (idx / p, idx % p);
                    let tb = fpv(b + o) * v;
                    bias = bias + tb;
                    babs = babs + tb.abs();
                    for tap in 0..taps {
                        let src = prog.ip[g + tap * p + sp];
                        let wt = fpv(w + o * taps + tap);
                        if src == NONE || wt == R::ZERO {
                            continue;
                        }
                        coef[i0 + src as usize] = coef[i0 + src as usize] + wt * v;
                    }
                }
            }
            OP_AFFINE => {
                let (w, b) = (d[F_W] as usize, d[F_BIAS] as usize);
                for j in 0..n {
                    let v = coef[o0 + j];
                    if v == R::ZERO {
                        continue;
                    }
                    let tb = fpv(b + j) * v;
                    bias = bias + tb;
                    babs = babs + tb.abs();
                    coef[i0 + j] = coef[i0 + j] + fpv(w + j) * v;
                }
            }
            OP_RELU => {
                let sb = d[F_SITE] as usize;
                for j in 0..n {
                    let v = coef[o0 + j];
                    let (l, u) = (prog.fp[s.s_l + sb + j], prog.fp[s.s_u + sb + j]);
                    let pos = prog.ip[s.s_pos + sb + j];
                    let lit = if pos == NONE { 0 } else { pin[pos as usize] };
                    let pj = if lit == 0 {
                        0
                    } else if dom.lits[lit - 1].1 {
                        1
                    } else {
                        -1
                    };
                    let mut cc = R::ZERO;
                    if pj > 0 || (pj == 0 && l >= 0.0) {
                        cc = v;
                    } else if pj < 0 || u <= 0.0 {
                    } else if v >= R::ZERO {
                        let a = if pos == NONE {
                            prog.fp[s.s_adapt + sb + j]
                        } else {
                            dom.alpha[pos as usize]
                        };
                        cc = R::of(a) * v;
                    } else {
                        cc = fpv(s.s_slope + sb + j) * v;
                        let tb = fpv(s.s_icpt + sb + j) * v;
                        bias = bias + tb;
                        babs = babs + tb.abs();
                    }
                    if lit != 0 {
                        let (_, active, beta) = dom.lits[lit - 1];
                        cc = if active {
                            cc - R::of(beta)
                        } else {
                            cc + R::of(beta)
                        };
                    }
                    if cc != R::ZERO {
                        coef[i0 + j] = coef[i0 + j] + cc;
                    }
                }
            }
            OP_ADD => {
                let r0 = d[F_RHS] as usize;
                for j in 0..n {
                    let v = coef[o0 + j];
                    if v == R::ZERO {
                        continue;
                    }
                    coef[i0 + j] = coef[i0 + j] + v;
                    coef[r0 + j] = coef[r0 + j] + v;
                }
            }
            OP_COPY => {
                for j in 0..n {
                    let v = coef[o0 + j];
                    if v != R::ZERO {
                        coef[i0 + j] = coef[i0 + j] + v;
                    }
                }
            }
            OP_GEMM => {
                let (n_in, w, b) = (d[F_B] as usize, d[F_W] as usize, d[F_BIAS] as usize);
                for o in 0..n {
                    let v = coef[o0 + o];
                    if v == R::ZERO {
                        continue;
                    }
                    let tb = fpv(b + o) * v;
                    bias = bias + tb;
                    babs = babs + tb.abs();
                    for i in 0..n_in {
                        let wi = fpv(w + o * n_in + i);
                        if wi != R::ZERO {
                            coef[i0 + i] = coef[i0 + i] + wi * v;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    let (mut lb, mut scale) = (bias, babs);
    for i in 0..prog.n_in {
        let (c, m, r) = (coef[i], fpv(s.mid + i), fpv(s.rad + i));
        lb = lb + (c * m - c.abs() * r);
        scale = scale + c.abs() * (m.abs() + r);
    }
    let lam = (0..prog.stride)
        .map(|p| coef[prog.ip[s.uni_lam + p] as usize])
        .collect();
    OraclePass { lb, scale, lam }
}

/// The BaBSR candidates of one domain from its oracle `lam` (the CPU
/// reference's `score_domain`): unsplit universe neurons unstable in the
/// frozen box, by descending `|lam| * (-u l / (u - l))` (f64, cast to f32; a
/// non-finite score is `f32::MAX`), ties to the lower position.
#[must_use]
pub(crate) fn oracle_scores(
    prog: &WaveProg,
    dom: &OracleDomain,
    lam: &[f64],
    k: usize,
) -> Vec<(u32, f32)> {
    let s = &prog.sec;
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    for p in 0..prog.stride {
        if dom.lits.iter().any(|l| l.0 as usize == p) {
            continue;
        }
        let flat = prog.ip[s.uni_flat + p] as usize;
        let (l, u) = (prog.fp[s.s_l + flat], prog.fp[s.s_u + flat]);
        if !(l < 0.0 && u > 0.0) {
            continue;
        }
        let (l, u) = (f64::from(l), f64::from(u));
        let area = if l.is_finite() && u.is_finite() {
            -u * l / (u - l)
        } else {
            1.0
        };
        let mut score = (lam[p].abs() * area) as f32;
        if !score.is_finite() {
            score = f32::MAX;
        }
        let at = top
            .iter()
            .position(|&(q, sc)| score > sc || (score == sc && (p as u32) < q))
            .unwrap_or(top.len());
        if at < k {
            top.insert(at, (p as u32, score));
            top.truncate(k);
        }
    }
    top
}

/// splitmix64 step.
fn mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One of `choices`, uniformly.
fn pick<T: Copy>(state: &mut u64, choices: &[T]) -> T {
    choices[(mix(state) % choices.len() as u64) as usize]
}

/// The self-test's DYADIC problem: a 1x4x4 input, a padded 3x3 conv to two
/// channels, a ReLU, a per-neuron affine, a residual add, a second ReLU, a
/// flatten, a dense layer into a 6-wide head ReLU and a 3-class output, with
/// every parameter, box side and box width a small dyadic rational (every
/// chord slope `u / (u - l)` too: every box width is 2 or 4). Every sum the
/// oracle forms is then exact in f32, so the device has ONE correct answer.
#[must_use]
pub(crate) fn kat_problem(seed: u64) -> WaveProblem {
    use ny_core::search::{ReluBox, SearchConv, SearchNet, WaveSite};
    let mut s = seed ^ 0x3A7E_5EED;
    let w = [-0.5, -0.25, 0.0, 0.25, 0.5];
    let small = [-0.25, -0.125, 0.0, 0.125, 0.25];
    let (h, cin, cout) = (4usize, 1usize, 2usize);
    let (taps, p) = (cin * 9, h * h);
    let mut gather = vec![GATHER_PAD; taps * p];
    for oy in 0..h {
        for ox in 0..h {
            let sp = oy * h + ox;
            for ky in 0..3 {
                for kx in 0..3 {
                    let (iy, ix) = (oy + ky, ox + kx);
                    if (1..=h).contains(&iy) && (1..=h).contains(&ix) {
                        gather[(ky * 3 + kx) * p + sp] = ((iy - 1) * h + (ix - 1)) as u32;
                    }
                }
            }
        }
    }
    let n1 = cout * p;
    let boxes_of = |s: &mut u64, n: usize| -> (Vec<f64>, Vec<f64>) {
        let kinds = [
            (-1.0, 1.0),
            (-1.0, 3.0),
            (-3.0, 1.0),
            (-2.0, 2.0),
            (-0.5, 1.5),
            (1.0, 3.0),
            (-3.0, -1.0),
        ];
        (0..n).map(|_| pick(s, &kinds)).unzip()
    };
    let (l1, u1) = boxes_of(&mut s, n1);
    let (l2, u2) = boxes_of(&mut s, n1);
    let (l3, u3) = boxes_of(&mut s, 6);
    let net = SearchNet {
        n_in: h * h,
        ops: vec![
            SearchOp::Conv(Box::new(SearchConv {
                input: 0,
                cout,
                k: taps,
                p,
                gather,
                weight: (0..cout * taps).map(|_| pick(&mut s, &w)).collect(),
                bias: (0..cout).map(|_| pick(&mut s, &small)).collect(),
            })),
            SearchOp::Relu {
                input: 1,
                head: false,
            },
            SearchOp::Affine {
                input: 2,
                scale: (0..n1)
                    .map(|_| pick(&mut s, &[0.5, 1.0, -0.5, 0.0]))
                    .collect(),
                shift: (0..n1).map(|_| pick(&mut s, &small)).collect(),
            },
            SearchOp::Add { lhs: 3, rhs: 2 },
            SearchOp::Relu {
                input: 4,
                head: false,
            },
            SearchOp::Flatten { input: 5 },
            SearchOp::Gemm {
                input: 6,
                n_out: 6,
                n_in: n1,
                weight: (0..6 * n1).map(|_| pick(&mut s, &w)).collect(),
                bias: (0..6).map(|_| pick(&mut s, &small)).collect(),
            },
            SearchOp::Relu {
                input: 7,
                head: true,
            },
            SearchOp::Gemm {
                input: 8,
                n_out: 3,
                n_in: 6,
                weight: (0..18).map(|_| pick(&mut s, &w)).collect(),
                bias: (0..3).map(|_| pick(&mut s, &small)).collect(),
            },
        ],
    };
    let mut universe = Vec::new();
    for (op, (l, u)) in [(1u32, (&l1, &u1)), (4, (&l2, &u2)), (7, (&l3, &u3))] {
        for j in 0..l.len() {
            if l[j] < 0.0 && u[j] > 0.0 && (!mix(&mut s).is_multiple_of(4) || op == 7) {
                universe.push(WaveSite {
                    op,
                    neuron: j as u32,
                });
            }
        }
    }
    let alpha0 = (0..universe.len())
        .map(|q| match q % 5 {
            0 => f64::NAN,
            1 => 0.25,
            2 => 1.5,
            3 => -0.5,
            _ => 0.75,
        })
        .collect();
    let lo: Vec<f64> = (0..h * h)
        .map(|_| pick(&mut s, &[-0.5, -0.25, 0.0, 0.25]))
        .collect();
    let hi = lo.iter().map(|l| l + 0.5).collect();
    WaveProblem {
        net,
        boxes: vec![
            ReluBox {
                op: 1,
                l: l1,
                u: u1,
            },
            ReluBox {
                op: 4,
                l: l2,
                u: u2,
            },
            ReluBox {
                op: 7,
                l: l3,
                u: u3,
            },
        ],
        universe,
        alpha0,
        lo,
        hi,
        target: 0,
        classes: vec![1, 2],
    }
}

/// The self-test's wave over [`kat_problem`]: `count` domains with dyadic
/// warm slopes (some NaN / out of range, which the ingest maps) and dyadic
/// multipliers (some dirty), literals on distinct positions of both sides.
/// Returns `(rows, offsets, splits, beta, alpha)` in `WaveBatch` layout.
#[must_use]
pub(crate) fn kat_wave(
    stride: usize,
    count: usize,
    seed: u64,
) -> (
    Vec<u32>,
    Vec<u32>,
    Vec<ny_core::search::WaveSplit>,
    Vec<f32>,
    Vec<f32>,
) {
    let mut s = seed ^ 0x0BAD_5EED;
    let (mut rows, mut offsets, mut splits, mut beta, mut alpha) =
        (Vec::new(), vec![0u32], Vec::new(), Vec::new(), Vec::new());
    for d in 0..count {
        rows.push((d % 2) as u32);
        let n = if stride == 0 { 0 } else { d % 4 };
        let mut used: Vec<u32> = Vec::new();
        while used.len() < n.min(stride) {
            let pos = (mix(&mut s) % stride as u64) as u32;
            if used.contains(&pos) {
                continue;
            }
            used.push(pos);
            splits.push(ny_core::search::WaveSplit {
                pos,
                active: mix(&mut s).is_multiple_of(2),
            });
            beta.push(pick(
                &mut s,
                &[0.0, 0.125, 0.25, 0.5, -0.25, f32::NAN, f32::INFINITY],
            ));
        }
        offsets.push(splits.len() as u32);
        for _ in 0..stride {
            alpha.push(pick(
                &mut s,
                &[0.0, 0.25, 0.5, 0.75, 1.0, f32::NAN, 2.0, -1.0],
            ));
        }
    }
    (rows, offsets, splits, beta, alpha)
}

/// The CPU search's warm-slope ingest (`ny_core::search::wave_warm_alpha`):
/// a finite slope clamped into `[0, 1]`, else `fall` (the upload's `alpha0`).
#[must_use]
pub(crate) fn warm_alpha(a: f32, fall: f32) -> f32 {
    ny_core::search::wave_warm_alpha(a).map_or(fall, |x| x as f32)
}

/// The CPU search's warm-multiplier ingest (`ny_core::search::wave_warm_beta`):
/// finite and positive, else 0.
#[must_use]
pub(crate) fn warm_beta(b: f32) -> f32 {
    ny_core::search::wave_warm_beta(b) as f32
}
