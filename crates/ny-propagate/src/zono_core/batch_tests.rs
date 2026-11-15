// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #m8-cgan-leaf-batch pins for the batched certified zonotope forward.
//!
//! The batch evaluator changes SCHEDULING only: the kernels' own row blocks
//! become rayon tasks on disjoint output slices, independent rows and column
//! blocks are processed in parallel, and a tranche's children run
//! concurrently. Every pin here is BITWISE (`to_bits`), never a tolerance:
//! the proof obligation is that no arithmetic changed. Each bit-identity pin
//! also asserts that the parallel arm actually ran parallel work (row tasks,
//! several blocks), so a schedule that silently fell back to the reference
//! is VOID, not a pass.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::ids::SymbolId;
use super::kernels::{ConvGeom, LinearKernel, RowPar, CHUNK_ELEMS};
use super::program::{StepOp, ZonoBatchChild, ZonoForwardOptions, ZonoForwardOutcome, ZonoProgram};
use super::state::{ZonoState, COL_BLOCK};
use super::tests::{random_child, sample_points, small_cgan_graph, Rng};
use super::ZonoRefusal;
use crate::layers::{
    Conv2dLayer, ConvTranspose2dLayer, FlattenLayer, Layer, LinearLayer, ReLULayer, ReshapeLayer,
};
use crate::{GraphNetwork, GraphNode};

#[allow(clippy::cast_possible_truncation)]
fn rand_f32(rng: &mut Rng, n: usize, scale: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.sym() * scale) as f32).collect()
}

fn rand_f64(rng: &mut Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.sym()).collect()
}

fn abs_of(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| x.abs()).collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn opts(row_par: RowPar) -> ZonoForwardOptions {
    ZonoForwardOptions {
        max_generators: 16_384,
        record_trace: false,
        row_par,
    }
}

fn root_box() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![-0.6, -0.4, 0.1, -0.9, 0.3]).expect("lo"),
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.5, 0.35, 0.8, -0.2, 0.9]).expect("hi"),
    )
    .expect("root box")
}

/// A convolution-heavy chain whose generator rows span SEVERAL reference
/// blocks at the production [`CHUNK_ELEMS`]: `Gemm(6 -> 256) -> Reshape
/// [4, 8, 8] -> ConvTranspose(4 -> 8, k4 s2 p1) -> ReLU -> Conv(8 -> 16, k3 s1
/// p1) -> ReLU -> Flatten -> Gemm(4096 -> 2)`. The `Conv` block is
/// `CHUNK_ELEMS / (256 * 72) = 227` rows, so its `K`-row product is
/// multi-block once the first ReLU has a few hundred unstable neurons, and
/// the 4096-wide second ReLU spans two column blocks (`COL_BLOCK`).
fn conv_heavy_graph(seed: u64) -> GraphNetwork {
    let mut rng = Rng::new(seed);
    let w1 = Array2::from_shape_vec((256, 6), rand_f32(&mut rng, 256 * 6, 0.9)).expect("w1");
    let b1 = Array1::from_vec(rand_f32(&mut rng, 256, 0.1));
    let convt = ConvTranspose2dLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[4, 8, 4, 4]), rand_f32(&mut rng, 4 * 8 * 16, 0.4))
            .expect("convt kernel"),
        Some(Array1::from_vec(rand_f32(&mut rng, 8, 0.05))),
        (2, 2),
        (1, 1),
        8,
        8,
    )
    .expect("convt");
    let conv = Conv2dLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[16, 8, 3, 3]), rand_f32(&mut rng, 16 * 8 * 9, 0.3))
            .expect("conv kernel"),
        Some(Array1::from_vec(rand_f32(&mut rng, 16, 0.05))),
        (1, 1),
        (1, 1),
        16,
        16,
    )
    .expect("conv");
    let w2 = Array2::from_shape_vec((2, 4096), rand_f32(&mut rng, 8192, 0.05)).expect("w2");
    let mut graph = GraphNetwork::new();
    let chain: Vec<(&str, Layer)> = vec![
        (
            "gemm0",
            Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("gemm0")),
        ),
        ("reshape", Layer::Reshape(ReshapeLayer::new(vec![4, 8, 8]))),
        ("convt", Layer::ConvTranspose2d(convt)),
        ("relu1", Layer::ReLU(ReLULayer)),
        ("conv", Layer::Conv2d(conv)),
        ("relu2", Layer::ReLU(ReLULayer)),
        ("flat", Layer::Flatten(FlattenLayer::new(0))),
        (
            "gemm1",
            Layer::Linear(LinearLayer::new(w2, None).expect("gemm1")),
        ),
    ];
    let mut prev: Option<String> = None;
    for (name, layer) in chain {
        match prev {
            None => graph.add_node(GraphNode::from_input(name, layer)),
            Some(ref input) => graph.add_node(GraphNode::new(name, layer, vec![input.clone()])),
        }
        prev = Some(name.to_string());
    }
    graph.set_output("gemm1");
    graph
}

fn heavy_root() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[6]), vec![-1.0; 6]).expect("lo"),
        ArrayD::from_shape_vec(IxDyn(&[6]), vec![1.0; 6]).expect("hi"),
    )
    .expect("heavy root")
}

/// Every verdict-bearing field of two outcomes is bitwise equal (stats other
/// than `elapsed` and the `row_tasks` telemetry included).
fn assert_outcomes_bit_identical(a: &ZonoForwardOutcome, b: &ZonoForwardOutcome, what: &str) {
    let mut keys: Vec<&String> = a.published.keys().collect();
    keys.sort();
    let mut keys_b: Vec<&String> = b.published.keys().collect();
    keys_b.sort();
    assert_eq!(keys, keys_b, "{what}: published nodes differ");
    for key in keys {
        let (x, y) = (&a.published[key], &b.published[key]);
        assert_eq!(x.shape(), y.shape(), "{what}: `{key}` shape");
        let lx: Vec<u32> = x.lower().iter().map(|v| v.to_bits()).collect();
        let ly: Vec<u32> = y.lower().iter().map(|v| v.to_bits()).collect();
        let ux: Vec<u32> = x.upper().iter().map(|v| v.to_bits()).collect();
        let uy: Vec<u32> = y.upper().iter().map(|v| v.to_bits()).collect();
        assert_eq!(lx, ly, "{what}: `{key}` lower");
        assert_eq!(ux, uy, "{what}: `{key}` upper");
    }
    assert_eq!(
        bits(&a.output_lo),
        bits(&b.output_lo),
        "{what}: output lower"
    );
    assert_eq!(
        bits(&a.output_hi),
        bits(&b.output_hi),
        "{what}: output upper"
    );
    assert_eq!(a.relus, b.relus, "{what}: relu summaries");
    assert_eq!(a.input_symbols.len(), b.input_symbols.len());
    for (s, t) in a.input_symbols.iter().zip(&b.input_symbols) {
        assert_eq!((s.id, s.dim), (t.id, t.dim), "{what}: input symbol");
        assert_eq!(s.center.to_bits(), t.center.to_bits());
        assert_eq!(s.radius.to_bits(), t.radius.to_bits());
    }
    let (sa, sb) = (&a.stats, &b.stats);
    assert_eq!(
        (sa.gens_final, sa.gens_peak, sa.folded),
        (sb.gens_final, sb.gens_peak, sb.folded),
        "{what}: generator stats"
    );
    assert_eq!(
        sa.max_remainder.to_bits(),
        sb.max_remainder.to_bits(),
        "{what}: max remainder"
    );
    assert_eq!(sa.macs.to_bits(), sb.macs.to_bits(), "{what}: macs");
}

/// A random dense / conv / conv-transpose kernel.
fn kernels(rng: &mut Rng) -> Vec<(&'static str, LinearKernel)> {
    let wd = rand_f64(rng, 7 * 13);
    let dense = LinearKernel::Dense {
        n_in: 13,
        n_out: 7,
        w: wd.clone(),
        w_abs: abs_of(&wd),
    };
    let geom = ConvGeom::new(3, 6, 6, 3, 3, 1, 1, 1, 1).expect("conv geom");
    let kc = rand_f64(rng, 4 * geom.patch_len());
    let conv = LinearKernel::Conv2d {
        geom,
        out_channels: 4,
        k: kc.clone(),
        k_abs: abs_of(&kc),
    };
    // ConvTranspose: output image 3 x 7 x 7 over a 4 x 4 grid (k3 s2 p1).
    let tgeom = ConvGeom::new(3, 7, 7, 3, 3, 2, 2, 1, 1).expect("convt geom");
    let kt = rand_f64(rng, 4 * tgeom.patch_len());
    let convt = LinearKernel::ConvTranspose2d {
        geom: tgeom,
        in_channels: 4,
        k: kt.clone(),
        k_abs: abs_of(&kt),
    };
    vec![
        ("dense", dense),
        ("conv2d", conv),
        ("conv_transpose2d", convt),
    ]
}

/// Rows of the reference's block for `kernel` at `chunk_elems` (restated
/// independently of the kernel code).
fn block_rows(kernel: &LinearKernel, chunk_elems: usize) -> usize {
    match kernel {
        LinearKernel::Dense { n_in, n_out, .. } => (chunk_elems / (*n_in).max(*n_out)).max(1),
        LinearKernel::Conv2d {
            geom, out_channels, ..
        } => (chunk_elems / (geom.grid_len() * geom.patch_len().max(*out_channels))).max(1),
        LinearKernel::ConvTranspose2d {
            geom, in_channels, ..
        } => (chunk_elems / (geom.grid_len() * geom.patch_len().max(*in_channels))).max(1),
    }
}

/// Dense, Conv2d and ConvTranspose2d under `RowPar::Rayon` are bitwise the
/// sequential kernels at rows 1, block - 1, block, block + 1 and
/// 3 * block + 5, for both `A` and `|A|`, with small scratch targets (many
/// blocks) and once at the production [`CHUNK_ELEMS`]; a passed deadline
/// refuses under both schedules.
#[test]
fn zono_row_parallel_kernels_are_bit_identical_to_sequential() {
    let mut rng = Rng::new(901);
    let mut multi_block_calls = 0usize;
    for (name, kernel) in kernels(&mut rng) {
        let (n_in, n_out) = (kernel.n_in(), kernel.n_out());
        for chunk_elems in [n_in.max(n_out) * 3 + 1, 2_000, 5_000] {
            let block = block_rows(&kernel, chunk_elems);
            for rows in [
                1,
                block.saturating_sub(1).max(1),
                block,
                block + 1,
                3 * block + 5,
            ] {
                let x = rand_f64(&mut rng, rows * n_in);
                for abs in [false, true] {
                    let mut seq = vec![0.0; rows * n_out];
                    let mut par = vec![f64::NAN; rows * n_out];
                    let t_seq = kernel
                        .forward_rows_chunked(
                            &x,
                            rows,
                            abs,
                            &mut seq,
                            None,
                            RowPar::Seq,
                            chunk_elems,
                        )
                        .expect("sequential");
                    let t_par = kernel
                        .forward_rows_chunked(
                            &x,
                            rows,
                            abs,
                            &mut par,
                            None,
                            RowPar::Rayon,
                            chunk_elems,
                        )
                        .expect("row-parallel");
                    assert_eq!(t_seq, rows.div_ceil(block), "{name}: block count");
                    assert_eq!(t_seq, t_par, "{name}: the schedules ran different blocks");
                    assert_eq!(
                        bits(&seq),
                        bits(&par),
                        "{name}: rows={rows} block={block} abs={abs} differ"
                    );
                    if t_par > 1 {
                        multi_block_calls += 1;
                    }
                }
            }
        }
        // A passed deadline refuses under both schedules.
        let rows = 3 * block_rows(&kernel, 2_000) + 5;
        let x = rand_f64(&mut rng, rows * n_in);
        for par in [RowPar::Seq, RowPar::Rayon] {
            let mut out = vec![0.0; rows * n_out];
            let got = kernel.forward_rows_chunked(
                &x,
                rows,
                false,
                &mut out,
                Some(Instant::now()),
                par,
                2_000,
            );
            assert_eq!(got, Err(ZonoRefusal::Deadline), "{name} {par:?}");
        }
    }
    assert!(
        multi_block_calls >= 30,
        "only {multi_block_calls} multi-block calls: VOID"
    );

    // Production chunk: a dense kernel whose rows span two blocks.
    let (n_in, n_out) = (8_192usize, 3usize);
    let w = rand_f64(&mut rng, n_in * n_out);
    let dense = LinearKernel::Dense {
        n_in,
        n_out,
        w: w.clone(),
        w_abs: abs_of(&w),
    };
    let block = CHUNK_ELEMS / n_in;
    let rows = block + 1;
    let x = rand_f64(&mut rng, rows * n_in);
    let (mut seq, mut par) = (vec![0.0; rows * n_out], vec![0.0; rows * n_out]);
    dense
        .forward_rows_par(&x, rows, false, &mut seq, None, RowPar::Seq)
        .expect("sequential");
    let tasks = dense
        .forward_rows_par(&x, rows, false, &mut par, None, RowPar::Rayon)
        .expect("row-parallel");
    assert_eq!(tasks, 2, "the production-chunk case must span two blocks");
    assert_eq!(bits(&seq), bits(&par), "production-chunk dense rows differ");
}

/// The column-parallel `abs_generator_sum` and `magnitude` are bitwise the
/// sequential ones on a state whose column sums are ORDER-SENSITIVE
/// (adversarial magnitudes; the negative control shows a reversed-row sum
/// differs), with several column blocks.
#[test]
fn zono_column_sums_are_bit_identical_under_row_parallelism() {
    let n = 3 * COL_BLOCK + 17;
    let k = 40usize;
    let mut rng = Rng::new(902);
    let mut gens = vec![0.0f64; k * n];
    for (row, chunk) in gens.chunks_exact_mut(n).enumerate() {
        for value in chunk.iter_mut() {
            let sign = if rng.unit() < 0.5 { -1.0 } else { 1.0 };
            *value = if row == 0 {
                sign * 1e16
            } else {
                sign * 10f64.powf(32.0 * rng.unit() - 16.0)
            };
        }
    }
    // Negative control: the fixture is order-sensitive, so a reduction that
    // was split or reordered could not pass.
    let differs = (0..n).any(|j| {
        let forward: f64 = (0..k).fold(0.0, |acc, r| acc + gens[r * n + j].abs());
        let backward: f64 = (0..k).rev().fold(0.0, |acc, r| acc + gens[r * n + j].abs());
        forward.to_bits() != backward.to_bits()
    });
    assert!(
        differs,
        "the column sums are order-insensitive: VOID fixture"
    );
    let center = rand_f64(&mut rng, n);
    let rem: Vec<f64> = (0..n).map(|_| rng.unit() * 1e-3).collect();
    let ids: Vec<SymbolId> = (0..k as u64).map(SymbolId).collect();
    let state = ZonoState::from_parts(n, center, gens, ids, rem).expect("state");
    let seq = state.abs_generator_sum();
    let par = state.abs_generator_sum_par(RowPar::Rayon);
    assert_eq!(bits(&seq), bits(&par), "abs_generator_sum differs");
    assert_eq!(
        bits(&state.magnitude(&seq)),
        bits(&state.magnitude_par(&seq, RowPar::Rayon)),
        "magnitude differs"
    );
}

/// Children of `root` with random inherited maps: none, the root map, or the
/// root map and the child's own IBP map.
fn children_with_maps(
    graph: &GraphNetwork,
    root: &BoundedTensor,
    count: usize,
    seed: u64,
) -> (
    Vec<BoundedTensor>,
    HashMap<String, BoundedTensor>,
    Vec<Option<HashMap<String, BoundedTensor>>>,
    Vec<usize>,
) {
    let root_map = graph
        .collect_node_bounds_with_engine(root, None)
        .expect("root map");
    let mut rng = Rng::new(seed);
    let mut boxes = Vec::new();
    let mut own = Vec::new();
    let mut kinds = Vec::new();
    for c in 0..count {
        let child = random_child(root, 1 + c % 5, &mut rng);
        #[allow(clippy::cast_possible_truncation)]
        let kind = (rng.next_u64() % 3) as usize;
        own.push((kind == 2).then(|| {
            graph
                .collect_node_bounds_with_engine(&child, None)
                .expect("child map")
        }));
        boxes.push(child);
        kinds.push(kind);
    }
    (boxes, root_map, own, kinds)
}

fn batch_children<'a>(
    boxes: &'a [BoundedTensor],
    root_map: &'a HashMap<String, BoundedTensor>,
    own: &'a [Option<HashMap<String, BoundedTensor>>],
    kinds: &[usize],
) -> Vec<ZonoBatchChild<'a>> {
    boxes
        .iter()
        .zip(own)
        .zip(kinds)
        .map(|((input, own), &kind)| ZonoBatchChild {
            input,
            inherited: match kind {
                0 => Vec::new(),
                1 => vec![root_map],
                _ => vec![root_map, own.as_ref().expect("own map")],
            },
            deadline: None,
        })
        .collect()
}

/// `forward_batch` at widths 1, 2, 5 and 16 (random inherited maps) is, per
/// child, bitwise the sequential `forward`; on the convolution-heavy fixture
/// the batch ran real multi-block row tasks (VOID guard).
#[test]
fn zono_forward_batch_is_bit_identical_to_per_child_forward() {
    let graph = small_cgan_graph(903, 1);
    let root = root_box();
    let program = ZonoProgram::compile_with_ibp_shapes(&graph, &root).expect("compile");
    for (width, seed) in [(1usize, 904u64), (2, 905), (5, 906), (16, 907)] {
        let (boxes, root_map, own, kinds) = children_with_maps(&graph, &root, width, seed);
        let children = batch_children(&boxes, &root_map, &own, &kinds);
        let batch = program.forward_batch(&children, &opts(RowPar::Seq));
        assert_eq!(batch.len(), width);
        for (c, (child, (result, _nanos))) in children.iter().zip(&batch).enumerate() {
            let reference = program
                .forward(
                    child.input,
                    &opts(RowPar::Seq),
                    &child.inherited,
                    None,
                    None,
                )
                .expect("reference forward");
            let got = result.as_ref().expect("batched forward");
            assert_eq!(reference.stats.row_tasks, 0);
            assert!(
                got.stats.row_tasks > 0,
                "width {width} child {c}: no row task ran"
            );
            assert_outcomes_bit_identical(&reference, got, &format!("width {width} child {c}"));
        }
    }

    let heavy = conv_heavy_graph(908);
    let hroot = heavy_root();
    let hprog = ZonoProgram::compile_with_ibp_shapes(&heavy, &hroot).expect("compile heavy");
    let (boxes, root_map, mut own, _) = children_with_maps(&heavy, &hroot, 2, 909);
    // Child 0 inherits the root map, child 1 the root map and its own IBP map.
    own[1] = Some(
        heavy
            .collect_node_bounds_with_engine(&boxes[1], None)
            .expect("heavy child map"),
    );
    let kinds = vec![1usize, 2];
    let children = batch_children(&boxes, &root_map, &own, &kinds);
    let affine_steps = hprog
        .steps()
        .iter()
        .filter(|s| matches!(s.op, StepOp::Affine(_)))
        .count();
    let batch = hprog.forward_batch(&children, &opts(RowPar::Seq));
    for (c, (child, (result, _))) in children.iter().zip(&batch).enumerate() {
        let reference = hprog
            .forward(
                child.input,
                &opts(RowPar::Seq),
                &child.inherited,
                None,
                None,
            )
            .expect("heavy reference");
        let got = result.as_ref().expect("heavy batched");
        assert!(
            got.stats.row_tasks > affine_steps,
            "heavy child {c}: {} row tasks over {affine_steps} affine steps — no multi-block \
             product ran (VOID)",
            got.stats.row_tasks
        );
        assert_outcomes_bit_identical(&reference, got, &format!("heavy child {c}"));
    }
}

/// Pools of 1, 3 and 8 workers publish bitwise-identical batches.
#[test]
fn zono_forward_batch_is_thread_count_independent() {
    let graph = small_cgan_graph(911, 0);
    let root = root_box();
    let program = ZonoProgram::compile_with_ibp_shapes(&graph, &root).expect("compile");
    let (boxes, root_map, own, kinds) = children_with_maps(&graph, &root, 6, 912);
    let children = batch_children(&boxes, &root_map, &own, &kinds);
    let heavy = conv_heavy_graph(913);
    let hroot = heavy_root();
    let hprog = ZonoProgram::compile_with_ibp_shapes(&heavy, &hroot).expect("compile heavy");
    let (hboxes, hroot_map, hown, _) = children_with_maps(&heavy, &hroot, 2, 914);
    let hkinds = vec![1usize, 1];
    let hchildren = batch_children(&hboxes, &hroot_map, &hown, &hkinds);
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        pool.install(|| {
            (
                program.forward_batch(&children, &opts(RowPar::Seq)),
                hprog.forward_batch(&hchildren, &opts(RowPar::Seq)),
            )
        })
    };
    let (base, hbase) = run(1);
    for threads in [3usize, 8] {
        let (other, hother) = run(threads);
        for (c, ((a, _), (b, _))) in base.iter().zip(&other).enumerate() {
            let (a, b) = (
                a.as_ref().expect("1 thread"),
                b.as_ref().expect("n threads"),
            );
            assert_outcomes_bit_identical(a, b, &format!("{threads} threads child {c}"));
        }
        for (c, ((a, _), (b, _))) in hbase.iter().zip(&hother).enumerate() {
            let (a, b) = (
                a.as_ref().expect("1 thread"),
                b.as_ref().expect("n threads"),
            );
            assert_outcomes_bit_identical(a, b, &format!("{threads} threads heavy child {c}"));
        }
    }
}

/// A non-finite child box and a past-deadline child refuse ALONE; their
/// tranche-mates publish exactly their sequential outcomes.
#[test]
fn zono_forward_batch_refuses_only_the_bad_child() {
    let graph = small_cgan_graph(915, 0);
    let root = root_box();
    let program = ZonoProgram::compile_with_ibp_shapes(&graph, &root).expect("compile");
    let root_map = graph
        .collect_node_bounds_with_engine(&root, None)
        .expect("root map");
    let mut rng = Rng::new(916);
    let good_a = random_child(&root, 2, &mut rng);
    let good_b = random_child(&root, 3, &mut rng);
    let late = random_child(&root, 1, &mut rng);
    // BoundedTensor cannot carry a NaN; an infinite endpoint is the
    // non-finite child the zonotope must refuse.
    let non_finite = BoundedTensor::new_allow_infinite(
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![f32::NEG_INFINITY, -0.4, 0.1, -0.9, 0.3])
            .expect("lo"),
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.5, 0.35, 0.8, -0.2, 0.9]).expect("hi"),
    )
    .expect("non-finite box");
    let children = vec![
        ZonoBatchChild {
            input: &good_a,
            inherited: vec![&root_map],
            deadline: None,
        },
        ZonoBatchChild {
            input: &non_finite,
            inherited: vec![&root_map],
            deadline: None,
        },
        ZonoBatchChild {
            input: &good_b,
            inherited: Vec::new(),
            deadline: None,
        },
        ZonoBatchChild {
            input: &late,
            inherited: vec![&root_map],
            deadline: Some(Instant::now()),
        },
    ];
    let batch = program.forward_batch(&children, &opts(RowPar::Seq));
    assert!(
        matches!(batch[1].0, Err(ZonoRefusal::Shape(_))),
        "{:?}",
        batch[1].0.as_ref().err()
    );
    assert_eq!(batch[3].0.as_ref().err(), Some(&ZonoRefusal::Deadline));
    for c in [0usize, 2] {
        let reference = program
            .forward(
                children[c].input,
                &opts(RowPar::Seq),
                &children[c].inherited,
                None,
                None,
            )
            .expect("reference");
        let got = batch[c]
            .0
            .as_ref()
            .expect("a good child must not be refused");
        assert_outcomes_bit_identical(&reference, got, &format!("good child {c}"));
    }
}

/// The admission reserve covers the row-task scratch: for every thread count
/// and row count, `row_task_scratch_bytes(t)` is at least `t` tasks of the
/// largest per-task scratch any step allocates, the per-task formula matches
/// an independent restatement of the conv kernels' buffers, and a dense
/// block allocates nothing.
#[test]
fn zono_forward_batch_estimate_covers_row_task_scratch() {
    let heavy = conv_heavy_graph(917);
    let hprog = ZonoProgram::compile_with_ibp_shapes(&heavy, &heavy_root()).expect("heavy");
    let small = small_cgan_graph(918, 1);
    let sprog = ZonoProgram::compile_with_ibp_shapes(&small, &root_box()).expect("small");
    let mut checked_conv = 0usize;
    for program in [&hprog, &sprog] {
        let mut previous = 0u64;
        for threads in [1usize, 3, 8, 20] {
            let reserve = program.row_task_scratch_bytes(threads);
            assert!(reserve >= previous, "the reserve shrank with more threads");
            previous = reserve;
            for step in program.steps() {
                let StepOp::Affine(op) = &step.op else {
                    continue;
                };
                for rows in [1usize, 7, 100, 1_000, 100_000] {
                    let per_task = op.kernel.task_scratch_elems(rows);
                    let expected = match &op.kernel {
                        LinearKernel::Dense { .. } => 0,
                        LinearKernel::Conv2d {
                            geom, out_channels, ..
                        } => {
                            let (hw, pl) = (geom.grid_len(), geom.patch_len());
                            let chunk = (CHUNK_ELEMS / (hw * pl.max(*out_channels)))
                                .max(1)
                                .min(rows);
                            checked_conv += 1;
                            chunk * hw * pl + chunk * hw * out_channels
                        }
                        LinearKernel::ConvTranspose2d {
                            geom, in_channels, ..
                        } => {
                            let (hw, pl) = (geom.grid_len(), geom.patch_len());
                            let chunk =
                                (CHUNK_ELEMS / (hw * pl.max(*in_channels))).max(1).min(rows);
                            checked_conv += 1;
                            chunk * hw * in_channels + chunk * hw * pl
                        }
                    };
                    assert_eq!(
                        per_task, expected,
                        "`{}`: per-task scratch formula",
                        step.node
                    );
                    let need = (threads as u64) * 8 * per_task as u64;
                    assert!(
                        reserve >= need,
                        "`{}` rows={rows} threads={threads}: reserve {reserve} < {need}",
                        step.node
                    );
                }
            }
        }
        assert!(previous > 0, "a program with convolutions reserved nothing");
    }
    assert!(checked_conv > 0, "no convolution step was checked: VOID");
}

/// Every published pre-activation box of a batched tranche contains the
/// sampled pre-activations of ITS child (corners plus uniform samples), on
/// the small cGAN and the convolution-heavy fixture.
#[test]
fn zono_forward_batch_published_boxes_contain_sampled_preactivations() {
    let cases: Vec<(GraphNetwork, BoundedTensor, usize, usize)> = vec![
        (small_cgan_graph(921, 1), root_box(), 5, 2_000),
        (conv_heavy_graph(922), heavy_root(), 2, 300),
    ];
    for (case, (graph, root, width, samples)) in cases.into_iter().enumerate() {
        let program = ZonoProgram::compile_with_ibp_shapes(&graph, &root).expect("compile");
        let (boxes, root_map, own, kinds) = children_with_maps(&graph, &root, width, 923);
        let children = batch_children(&boxes, &root_map, &own, &kinds);
        let batch = program.forward_batch(&children, &opts(RowPar::Seq));
        let steps = program.steps();
        for (c, (child, (result, _))) in children.iter().zip(&batch).enumerate() {
            let outcome = result.as_ref().expect("batched forward");
            assert!(
                !outcome.published.is_empty(),
                "case {case} child {c}: nothing published"
            );
            let (points, rows) = sample_points(child.input, samples, 924 + c as u64);
            let mut checked = 0usize;
            program
                .evaluate_points(&points, rows, |step, values| {
                    let Some(s) = step else { return };
                    let Some(entry) = outcome.published.get(&steps[s].node) else {
                        return;
                    };
                    let lo: Vec<f32> = entry.lower().iter().copied().collect();
                    let hi: Vec<f32> = entry.upper().iter().copied().collect();
                    let n = lo.len();
                    for r in 0..rows {
                        for j in 0..n {
                            let v = values[r * n + j];
                            assert!(
                                f64::from(lo[j]) <= v && v <= f64::from(hi[j]),
                                "SOUNDNESS: case {case} child {c}: `{}`[{j}] = {v} outside [{}, {}]",
                                steps[s].node,
                                lo[j],
                                hi[j]
                            );
                            checked += 1;
                        }
                    }
                })
                .expect("evaluate");
            assert!(checked > 0, "case {case} child {c}: nothing checked (VOID)");
        }
    }
}

/// The batched certified path stays unfused: the bit-identity of every row
/// schedule needs the reference's separate multiply and add, and Rust never
/// fuses them, so only an explicit fused call could break it. Needles built
/// from pieces so this file does not match itself.
#[test]
fn zono_batch_has_no_fma() {
    let fused = [concat!("mul", "_add"), concat!("f", "ma")];
    assert!("x.mul_add(y, z)".contains(fused[0]), "VOID needle");
    assert!("vfmaq_f64".contains(fused[1]), "VOID needle");
    for (name, src) in [
        ("kernels.rs", include_str!("kernels.rs")),
        ("affine.rs", include_str!("affine.rs")),
        ("relu.rs", include_str!("relu.rs")),
        ("state.rs", include_str!("state.rs")),
        ("program.rs", include_str!("program.rs")),
    ] {
        assert!(src.len() > 1_000, "{name} is empty: VOID");
        for needle in fused {
            assert!(
                !src.contains(needle),
                "{name} contains `{needle}`: the certified zonotope path must stay unfused"
            );
        }
    }
}

/// Sanity for the deadline type in this file (keeps `Duration` honest).
#[test]
fn zono_forward_batch_children_carry_their_own_deadline() {
    let graph = small_cgan_graph(925, 0);
    let root = root_box();
    let program = ZonoProgram::compile_with_ibp_shapes(&graph, &root).expect("compile");
    let far = Instant::now() + Duration::from_secs(600);
    let children = vec![
        ZonoBatchChild {
            input: &root,
            inherited: Vec::new(),
            deadline: Some(far),
        },
        ZonoBatchChild {
            input: &root,
            inherited: Vec::new(),
            deadline: Some(Instant::now()),
        },
    ];
    let batch = program.forward_batch(&children, &opts(RowPar::Seq));
    assert!(batch[0].0.is_ok(), "a distant deadline must not refuse");
    assert_eq!(batch[1].0.as_ref().err(), Some(&ZonoRefusal::Deadline));
}
