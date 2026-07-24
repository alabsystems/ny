#!/usr/bin/env python3
"""Create immutable provenance records for a NY VNN-COMP measurement run.

The start record is deliberately content-oriented: it binds the solver binary,
the NY worktree (including every non-ignored untracked file), the dependency
pins, benchmark checkout, host, and effective sweep configuration without
copying source patches or arbitrary environment variables into the artifact.
The completion record is separate so the start evidence can never be rewritten.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
V1_INPUT_ASSIGNMENT = re.compile(
    r"\(\s*X_\d+\s+[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*\)"
)
V2_ASSIGNMENT_HEADER = re.compile(r"^(\S+)\s+(\S+)\s+\[([0-9,\s]*)\]$")
STANDARD_SOLVER_VERDICTS = frozenset({"sat", "unsat", "unknown", "timeout", "error"})
PROCESS_SNAPSHOT_LIMIT = 128
PROCESS_ARGS_LIMIT = 384
SNAPSHOT_OUTPUT_LIMIT = 64 * 1024

_SENSITIVE_ARGUMENT = (
    r"(?:api[-_]?key|token|password|passwd|secret|credential|authorization|"
    r"auth|cookie|private[-_]?key|access[-_]?key)"
)

# This is intentionally an explicit allowlist, not an NY_* or process-environment
# sweep. Unknown NY_* settings, every externally inherited AY_* setting, and
# mimalloc runtime controls make capture fail so a solver-affecting knob can
# neither disappear from the record nor accidentally leak a new secret. NY
# configures its reviewed AY policy only inside the solver process, after this
# caller has captured the launch inputs. Competition wrappers sanitize mimalloc;
# a local allocator A/B must explicitly remove or provenance-review those knobs.
ENV_ALLOWLIST = frozenset(
    {
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_MODULE_LOADING",
        "CUDA_VISIBLE_DEVICES",
        "DYLD_LIBRARY_PATH",
        "GPU_AVAILABLE",
        "LD_LIBRARY_PATH",
        "MKL_NUM_THREADS",
        "NY_ACASXU_PROF",
        "NY_ADAPTIVE_MICROBATCH_CONTROLLER",
        "NY_ALPHA_FINAL_BOUND_ONLY",
        "NY_ALPHA_REFRESH_FRACTION",
        "NY_AY",
        "NY_AY_BRANCH_HINTS",
        "NY_AY_MARGIN_REFRAME",
        "NY_AY_MILP_TALL_FLIP_CAP",
        "NY_ATTACK_EXTEND",
        "NY_ATTACK_EXTEND_FRAC",
        "NY_ATTACK_EXTEND_MARGIN",
        "NY_BAB_QUEUE_MEM_MB",
        "NY_BAB_RESNET_REFOLD_GUARD",
        "NY_BN_FOLD_EXT",
        "NY_BRANCH_KFSB_CHILDSIM",
        "NY_BRANCH_RESCORE",
        "NY_BRANCH_STEM",
        "NY_BRANCH_STEM_K",
        "NY_BRANCH_STEM_NODES",
        "NY_BRANCH_STEM_PROBE",
        "NY_BRANCH_TRACE",
        "NY_BROOT",
        "NY_BUILD_DISK_PATH",
        "NY_BUILD_FEATURES",
        "NY_BUILD_MIN_FREE_KIB",
        "NY_CLIP_INTERM_RESNET",
        "NY_CONV_SKIP_DEAD_F32",
        "NY_CONVTRANSPOSE_SOUND_F64_GPU",
        "NY_CROWN_CUT_SEGMENT",
        "NY_CUDA_CROWN",
        "NY_CUDA_DGEMM_TRIPLET",
        "NY_CUDA_WIDE",
        "NY_CUDA_WIDE_MAX_BYTES",
        "NY_CROWN_MEM_CAP_MB",
        "NY_CROWN_OBJ_CHUNK",
        "NY_DENSE_BUDGET_MB",
        "NY_DISABLE_CROWN_COLLECTION_CACHE",
        "NY_F64_BLAS",
        "NY_F64_LEAF",
        "NY_F64_LINEAGE_RECOVER",
        "NY_F64_TAIL",
        "NY_FACET_BANK",
        "NY_FACET_BANK_MAX_BYTES",
        "NY_FACET_BANK_PLANES",
        "NY_FCHEAD_GRACE_SECS",
        "NY_FCHEAD_TIGHTEN",
        "NY_GPU_MEMORY_BUDGET_MB",
        "NY_GPU_LOCK_PATH",
        "NY_GPU_LOCK_WAIT_SECS",
        "NY_GPU_VMEM_LIMIT_KIB",
        "NY_GRAPH_MIP",
        "NY_GRAPH_MIP_LEAF",
        "NY_GRAPH_MIP_LEAF_MAX_BINARIES",
        "NY_GRAPH_MIP_LEAF_MAX_NNZ",
        "NY_GRAPH_MIP_LEAF_MIN_DEPTH",
        "NY_GRAPH_MIP_LEAF_PIN",
        "NY_GRAPH_MIP_LEAF_SLICE_S",
        "NY_GRAPH_MIP_LEAF_TOTAL_FRAC",
        "NY_GRAPH_MIP_MAX_BINARIES",
        "NY_GRAPH_MIP_MAX_NNZ",
        "NY_GRAPH_MIP_MIN_SLICE_S",
        "NY_GRAPH_MIP_SERIAL",
        "NY_HYDRA_CROWN",
        # Certificate-backed Input-Manifold Bound canary controls. Keep this
        # list explicit: any newly introduced NY_IMB_* knob still fails closed
        # until it receives the same provenance review.
        "NY_IMB",
        "NY_IMB_AY_REGION_PROOF",
        "NY_IMB_BATCHED_REPLAY",
        "NY_IMB_BUDGET_S",
        "NY_IMB_EARLY",
        "NY_IMB_LEAF_MODE",
        "NY_IMB_OBJ",
        "NY_IMB_REGION_K",
        "NY_IMB_REPLAY_ONLY",
        "NY_IMB_REPLAY_ONLY_LEAF",
        "NY_IMB_TAIL_ALPHA",
        "NY_IMB_TAIL_CERT_AY",
        "NY_IMB_WIRE",
        "NY_INTERM_REFINE",
        "NY_INTERM_REFINE_ALPHA",
        "NY_INTERM_REFINE_ALPHA_ITERS",
        "NY_INTERM_REFINE_ALPHA_LR",
        "NY_INTERM_REFINE_ALPHA_MAX_ROWS",
        "NY_INTERM_REFINE_ALPHA_REOPT",
        "NY_INTERM_REFINE_LAYERS",
        "NY_INTERM_REFINE_MAX_DIM",
        "NY_INTERM_REFINE_MIN_DEPTH",
        "NY_INTERM_REFINE_PRUNE",
        "NY_INTERM_REFINE_PRUNE_TOL",
        "NY_INTERM_REFINE_ROWS",
        "NY_INTERM_REFINE_SEEDS",
        "NY_INTERM_REFINE_SELECTIVE_TOPK",
        "NY_INPUT_SPLIT_WARM_PARALLEL",
        "NY_INVPROP",
        "NY_INVPROP_LR",
        "NY_INVPROP_OPTIMIZE",
        "NY_INVPROP_SPLIT_LIFT",
        "NY_KFSB_LAYER_QUOTA",
        "NY_MARGIN_ROW_ADAPTIVE_RESERVE",
        "NY_MARGIN_ROW_DOMAIN_STACK",
        "NY_MARGIN_ROW_LRU",
        "NY_MARGIN_ROW_CLASSWISE",
        "NY_MARGIN_ROW_PARALLEL",
        "NY_MARGIN_ROW_PROFILE",
        "NY_MARGIN_ROW_RESERVE_SECS",
        "NY_MEASURE_ARTIFACTS",
        "NY_MEASURE_BIN",
        "NY_MEASURE_CAP",
        "NY_MEASURE_CATS",
        "NY_MEASURE_CONFIGS_DIR",
        "NY_MEASURE_INSTANCE_INDEX",
        "NY_MEASURE_MAX_ROWS_PER_CATEGORY",
        "NY_MEASURE_OUTPUT_DIR",
        "NY_MEASURE_RUN_ID",
        "NY_MEASURE_VNNLIB_VERSION",
        "NY_MIP_STABILITY_HINTS",
        "NY_MIP_WINDOW_TIMEOUT_SECS",
        "NY_MO_ADAPTIVE_DEPTH_SELECT",
        "NY_MO_ADAPTIVE_DEPTH_SHADOW",
        "NY_MO_BAB_TRACE",
        "NY_MO_GPU_CHUNK",
        "NY_MO_KFSB",
        "NY_MO_KFSB_CACHED_LA",
        "NY_MO_KFSB_CHUNK",
        "NY_MO_KFSB_K",
        "NY_MO_KFSB_PROBE",
        "NY_MO_KFSB_REDUCE",
        "NY_MO_KFSB_WINNER_PROBE",
        "NY_MO_KFSB_WINNER_PROBE_DOMAINS",
        "NY_MOAT_SECS",
        "NY_MULTIOBJ_JOINT_ALPHA",
        "NY_MULTIOBJ_JOINT_ALPHA_GPU",
        "NY_NO_ALPHA_BRIDGE",
        "NY_NO_CNF_ROUTE",
        "NY_NO_CUDA",
        "NY_NO_CUDA_F32",
        "NY_NO_FRAC_HEAD",
        "NY_NO_PGD_TIME_CAP",
        "NY_ORACLE_FRONTIER",
        "NY_ORT_ATTACK",
        "NY_ORT_ACTIVE_SET_REPAIR",
        "NY_ORT_REFINE_GRAD",
        "NY_ORT_SESSION_CACHE",
        "NY_PACKED_GRAPH_ALPHA_QUEUE",
        "NY_PATCHES_BUDGET_SECS",
        "NY_PATCHES_GPU",
        "NY_PHASE_TELEMETRY",
        "NY_PGD_DIAG",
        "NY_PGD_EXACT_BATCHED",
        "NY_PGD_GAMA",
        "NY_PGD_GAMA_LAMBDA",
        "NY_PGD_GAMA_LIN_FRAC",
        "NY_PGD_VJP_BATCH",
        "NY_POSTBAB_ATTACK",
        "NY_POSTBAB_BAB_SEEDS",
        "NY_POSTBAB_BAB_SEEDS_K",
        "NY_POSTBAB_FRONTIER_FASTLANE",
        "NY_POSTBAB_FRONTIER_FASTLANE_SECS",
        "NY_POSTBAB_RESERVE_SECS",
        "NY_REL_BAB_DEADLINE_MULT",
        "NY_REL_DIFF_COUPLING",
        "NY_REL_JOINT_RELU_CUTS",
        "NY_REL_JOINT_RELU_CUTS_SUM",
        "NY_RELATIONAL_BAB",
        "NY_RELATIONAL_UNSAT",
        "NY_REL_EDGE_ALPHA",
        "NY_REL_EDGE_ALPHA_ITERS",
        "NY_REL_EDGE_ALPHA_TOP",
        "NY_REL_EDGE_MILP",
        "NY_REL_EDGE_MILP_DEPTH",
        "NY_REL_EDGE_MILP_GAP",
        "NY_REL_WHOLE_MIP",
        "NY_REL_WHOLE_MIP_OBBT",
        "NY_REL_WHOLE_MIP_OBBT_CHUNK",
        "NY_REL_WHOLE_MIP_OBBT_COND",
        "NY_REL_WHOLE_MIP_OBBT_COND_FRAC",
        "NY_REL_WHOLE_MIP_OBBT_MAXN",
        "NY_REL_WHOLE_MIP_OBBT_OUTER",
        "NY_REL_WHOLE_MIP_OBBT_ROUNDS",
        "NY_REL_WHOLE_MIP_OBBT_S",
        "NY_REL_WHOLE_MIP_OBBT_WIDTH",
        "NY_REL_WHOLE_MIP_SLICE_S",
        "NY_RESNET_GPU",
        "NY_RESNET_GPU_MAX_OBJECTIVES",
        "NY_RESNET_GPU_MAX_SEED",
        "NY_RESNET_GPU_TIME_BUDGET_MS",
        "NY_RESNET_WARMUP_GPU",
        "NY_RNG_RESTARTS",
        "NY_RNG_SEED",
        "NY_ROOT",
        "NY_ROOT_ALPHA_GPU",
        "NY_ROOT_ALPHA_ITERS",
        "NY_ROOT_ALPHA_TRUE",
        "NY_ROOT_ALPHA_TRUE_MAXROWS",
        "NY_ROOT_BLAS",
        "NY_ROOT_BLAS_TILE",
        "NY_ROOT_GEMM",
        "NY_ROOT_CROWN_INTERM",
        "NY_ROOT_CROWN_INTERM_LAYERS",
        "NY_ROOT_CROWN_INTERM_MAXDIM",
        "NY_ROOT_CROWN_INTERM_OPTALPHA",
        "NY_ROOT_INTERM_ALPHA",
        "NY_ROOT_INTERM_ALPHA_SECS",
        "NY_ROOT_JOINT_INTERM_ALPHA",
        "NY_ROOT_JOINT_INTERM_ALPHA_ITERS",
        "NY_ROOT_JOINT_INTERM_ALPHA_LR",
        "NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM",
        "NY_ROOT_JOINT_INTERM_ALPHA_SECS",
        "NY_ROOT_POST_C_SURVIVOR",
        "NY_ROOT_SPARSE_INTERM_CROWN",
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_DIM",
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_ROWS",
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_TARGETS",
        "NY_ROOT_SPARSE_INTERM_CROWN_SECS",
        "NY_ROOT_SPEC_PRUNE",
        "NY_RUMP_F64_ENGINE",
        "NY_SCRATCH",
        # Schedule-only screen knobs still affect which clauses close before
        # a deadline. Preserve their raw spellings as launch evidence; do not
        # normalize fallback-equivalent values into a stronger identity.
        "NY_SCREEN_CELL_CHUNK",
        "NY_SCREEN_CROWN_MS",
        "NY_SCREEN_MVF_CHUNK",
        "NY_SCREEN_WAVE_SIZE",
        "NY_SKIP_DISJ_PGD",
        "NY_SPEC_ALPHA_DIRECT",
        "NY_SPEC_ROOT_ALPHA",
        "NY_SPEC_ROOT_GPU",
        "NY_SPEC_ROOT_MARGIN",
        "NY_STATE_DIR",
        "NY_STRICT_IBP",
        "NY_TLL_STRUCTURE_BOUND",
        "NY_UPFRONT_ATTACK",
        "NY_UPFRONT_ATTACK_AUTO_CAP",
        "NY_UPFRONT_ATTACK_CAP",
        "NY_UPFRONT_ATTACK_FRAC",
        "NY_UNSTABLE_COUNT",
        "NY_VNNLIB_CACHE",
        "NY_WARMUP_ITERS",
        "NY_WIDE_ALPHA_CPU",
        "NY_WIDE_ALPHA_LOCAL",
        "NY_WIDE_ALPHA_LR",
        "NY_WIDE_ALPHA_NOBIAS",
        "NY_WIDE_ALPHA_TRUE",
        "NY_WIDE_ALPHA_TRUE_DOMS",
        "NY_WIDE_ALPHA_TRUE_EVERY",
        "NY_WIDE_ALPHA_TRUE_STEP",
        "NY_WIDE_ALPHA_UNSHARED",
        "NY_WIDE_MAX_STACKED_ROWS",
        "NY_WITNESS_DEEPEN",
        "NY_WITNESS_DEEPEN_TARGET",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "ORT_LOG_SEVERITY_LEVEL",
        "RAYON_NUM_THREADS",
        "RUST_BACKTRACE",
        "RUST_LOG",
        "RUSTFLAGS",
        "RUSTUP_TOOLCHAIN",
        "TMPDIR",
    }
)

TYPED_NONNEGATIVE_INTEGER_ENV = frozenset(
    {
        "NY_CROWN_CUT_SEGMENT",
        "NY_FACET_BANK_MAX_BYTES",
        "NY_FACET_BANK_PLANES",
        "NY_IMB_OBJ",
        "NY_IMB_REGION_K",
        "NY_IMB_REPLAY_ONLY_LEAF",
        "NY_MO_KFSB_K",
        "NY_WARMUP_ITERS",
    }
)

TYPED_POSITIVE_INTEGER_ENV = frozenset(
    {
        "NY_BRANCH_STEM_K",
        "NY_BUILD_MIN_FREE_KIB",
        "NY_CUDA_WIDE_MAX_BYTES",
        "NY_GPU_LOCK_WAIT_SECS",
        "NY_GPU_VMEM_LIMIT_KIB",
        "NY_MO_GPU_CHUNK",
        "NY_MO_KFSB_CHUNK",
        "NY_MO_KFSB_WINNER_PROBE_DOMAINS",
    }
)

TYPED_BOUNDED_FLOAT_ENV = (("NY_REL_BAB_DEADLINE_MULT", 1.0, 10.0),)

TYPED_ENUM_ENV = (
    (
        "NY_IMB_AY_REGION_PROOF",
        frozenset({"affine", "reachability", "residual", "shared"}),
    ),
)

TYPED_STRICT_BOOLEAN_ENV = frozenset({"GPU_AVAILABLE", "NY_AY_BRANCH_HINTS"})

# These sealed gates have exact-string runtime semantics. Keep measurement
# syntax equally exact: unlike GPU_AVAILABLE, an explicitly empty value is
# malformed rather than a false spelling, and only reviewed "0"/"1" launch
# values are accepted.
TYPED_EXACT_BOOLEAN_ENV = frozenset(
    {
        "NY_ADAPTIVE_MICROBATCH_CONTROLLER",
        "NY_ALPHA_FINAL_BOUND_ONLY",
        "NY_BN_FOLD_EXT",
        "NY_BRANCH_STEM",
        "NY_BRANCH_STEM_PROBE",
        "NY_BRANCH_TRACE",
        "NY_CONVTRANSPOSE_SOUND_F64_GPU",
        "NY_CUDA_DGEMM_TRIPLET",
        "NY_IMB_BATCHED_REPLAY",
        "NY_IMB_REPLAY_ONLY",
        "NY_IMB_TAIL_CERT_AY",
        "NY_MO_BAB_TRACE",
        "NY_PACKED_GRAPH_ALPHA_QUEUE",
        "NY_ROOT_ALPHA_GPU",
        "NY_ROOT_POST_C_SURVIVOR",
        "NY_UNSTABLE_COUNT",
    }
)


class ProvenanceError(RuntimeError):
    """Evidence could not be captured completely and reproducibly."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _stable_file_hash(path: Path) -> tuple[str, dict[str, int]]:
    before = _file_fingerprint(path)
    digest = _sha256_file(path)
    after = _file_fingerprint(path)
    if before != after:
        raise ProvenanceError(f"file changed while provenance was captured: {path}")
    return digest, after


def _capture_executable_identity(
    declared_path: str,
    *,
    base_dir: Path,
    label: str,
) -> dict[str, object]:
    path = Path(declared_path)
    candidate = path if path.is_absolute() else base_dir / path
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(
            f"{label} executable does not exist: {declared_path}"
        ) from error
    if not resolved.is_file():
        raise ProvenanceError(f"{label} executable is not a file: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise ProvenanceError(f"{label} executable is not executable: {resolved}")
    digest, fingerprint = _stable_file_hash(resolved)
    try:
        resolved_after = candidate.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(
            f"{label} executable changed during provenance capture: {declared_path}"
        ) from error
    if resolved_after != resolved or _file_fingerprint(resolved) != fingerprint:
        raise ProvenanceError(
            f"{label} executable changed during provenance capture: {declared_path}"
        )
    return {
        "declared_path": declared_path,
        "resolved_path": str(resolved),
        "size_bytes": fingerprint["size_bytes"],
        "sha256": digest,
        "fingerprint": fingerprint,
    }


def _capture_ay_executable(repo_root: Path) -> dict[str, object] | None:
    declared_path = os.environ.get("NY_AY", "")
    if not declared_path:
        return None
    return _capture_executable_identity(
        declared_path,
        base_dir=repo_root,
        label="AY",
    )


def _config_tree_entries(root: Path) -> list[dict[str, object]]:
    """Return a deterministic, content-addressed manifest of a configs tree."""
    evidence: list[dict[str, object]] = []

    def visit(directory: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda entry: os.fsencode(entry.name))
        except OSError as error:
            raise ProvenanceError(
                f"could not enumerate measurement configs directory {directory}: {error}"
            ) from error
        for child in children:
            path = Path(child.path)
            relative = path.relative_to(root).as_posix()
            try:
                if child.is_symlink():
                    target = os.readlink(path)
                    item: dict[str, object] = {
                        "kind": "symlink",
                        "path": relative,
                        "target": target,
                        "target_sha256": _sha256(os.fsencode(target)),
                    }
                    if path.is_file():
                        digest, fingerprint = _stable_file_hash(path)
                        item.update(
                            {
                                "resolved_kind": "file",
                                "resolved_size_bytes": fingerprint["size_bytes"],
                                "resolved_sha256": digest,
                            }
                        )
                    elif path.is_dir():
                        raise ProvenanceError(
                            "measurement configs must not contain directory "
                            f"symlinks: {path}"
                        )
                    elif path.exists():
                        raise ProvenanceError(
                            f"measurement configs symlink targets a special file: {path}"
                        )
                    else:
                        item["resolved_kind"] = "missing"
                    evidence.append(item)
                elif child.is_dir(follow_symlinks=False):
                    evidence.append({"kind": "directory", "path": relative})
                    visit(path)
                elif child.is_file(follow_symlinks=False):
                    digest, fingerprint = _stable_file_hash(path)
                    evidence.append(
                        {
                            "kind": "file",
                            "path": relative,
                            "size_bytes": fingerprint["size_bytes"],
                            "sha256": digest,
                        }
                    )
                else:
                    raise ProvenanceError(
                        f"measurement configs contain a special file: {path}"
                    )
            except OSError as error:
                raise ProvenanceError(
                    f"could not capture measurement config input {path}: {error}"
                ) from error

    visit(root)
    return evidence


def _capture_config_inputs(configs_dir: Path) -> dict[str, object]:
    if not configs_dir.is_absolute():
        raise ProvenanceError(
            f"measurement configs directory must be absolute: {configs_dir}"
        )
    declared_path = str(configs_dir)
    try:
        resolved = configs_dir.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(
            f"measurement configs directory does not exist: {configs_dir}"
        ) from error
    if not resolved.is_dir():
        raise ProvenanceError(
            f"measurement configs path is not a directory: {configs_dir}"
        )
    first = _config_tree_entries(resolved)
    second = _config_tree_entries(resolved)
    try:
        resolved_after = configs_dir.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(
            f"measurement configs directory changed while provenance was captured: {configs_dir}"
        ) from error
    if first != second or resolved_after != resolved:
        raise ProvenanceError(
            f"measurement configs directory changed while provenance was captured: {configs_dir}"
        )
    manifest = {
        "schema": "ny_measurement_config_inputs_v1",
        "entries": first,
    }
    return {
        "schema": "ny_measurement_config_inputs_v1",
        "declared_path": declared_path,
        "resolved_path": str(resolved),
        "entry_count": len(first),
        "manifest_sha256": _sha256(_json_bytes(manifest)),
        "entries": first,
    }


def _seal_file(
    source: Path,
    destination: Path,
    *,
    executable: bool,
    expected_sha256: str | None = None,
    expected_fingerprint: dict[str, int] | None = None,
) -> dict[str, object]:
    """Copy one stable source into a read-only, content-addressed execution file."""
    try:
        resolved_source = source.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(
            f"cannot resolve file selected for sealing: {source}"
        ) from error
    if not resolved_source.is_file():
        raise ProvenanceError(
            f"file selected for sealing is not regular: {resolved_source}"
        )
    source_digest, source_fingerprint = _stable_file_hash(resolved_source)
    if expected_sha256 is not None and source_digest != expected_sha256:
        raise ProvenanceError(
            f"file changed before it could be sealed: {resolved_source}"
        )
    if expected_fingerprint is not None and source_fingerprint != expected_fingerprint:
        raise ProvenanceError(
            f"file fingerprint changed before it could be sealed: {resolved_source}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ProvenanceError(f"sealed-file destination is unsafe: {destination}")
        sealed_digest, sealed_fingerprint = _stable_file_hash(destination)
        if sealed_digest != source_digest:
            raise ProvenanceError(
                f"existing sealed file has different content: {destination}"
            )
    else:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        copied_digest = hashlib.sha256()
        try:
            with (
                resolved_source.open("rb") as input_file,
                os.fdopen(descriptor, "wb") as output_file,
            ):
                for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                    copied_digest.update(chunk)
                    output_file.write(chunk)
                output_file.flush()
                os.fsync(output_file.fileno())
            if (
                copied_digest.hexdigest() != source_digest
                or _file_fingerprint(resolved_source) != source_fingerprint
            ):
                raise ProvenanceError(
                    f"file changed while it was sealed: {resolved_source}"
                )
            destination.chmod(0o555 if executable else 0o444)
            sealed_digest, sealed_fingerprint = _stable_file_hash(destination)
            if sealed_digest != source_digest:
                raise ProvenanceError(
                    f"sealed copy does not match its source: {destination}"
                )
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    if executable and not os.access(destination, os.X_OK):
        raise ProvenanceError(f"sealed executable is not executable: {destination}")
    return {
        "schema": "ny_measurement_sealed_file_v1",
        "source_resolved_path": str(resolved_source),
        "source_sha256": source_digest,
        "source_fingerprint": source_fingerprint,
        "path": str(destination.resolve()),
        "size_bytes": sealed_fingerprint["size_bytes"],
        "sha256": sealed_digest,
        "fingerprint": sealed_fingerprint,
        "mode": "executable_read_only" if executable else "read_only",
    }


def _seal_config_inputs(
    original: dict[str, object], run_dir: Path
) -> dict[str, object]:
    """Materialize the captured config tree without live symlink dependencies."""
    source_value = original.get("resolved_path")
    manifest_digest = original.get("manifest_sha256")
    entries = original.get("entries")
    if (
        not isinstance(source_value, str)
        or not isinstance(manifest_digest, str)
        or not isinstance(entries, list)
    ):
        raise ProvenanceError("captured config identity is incomplete")
    source_root = Path(source_value)
    sealed_root = run_dir / "sealed" / "configs" / manifest_digest
    sealed_root.mkdir(parents=True, exist_ok=True)
    directories = [sealed_root]
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ProvenanceError("captured config entry is invalid")
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ProvenanceError(f"unsafe captured config path: {relative}")
        source = source_root / relative
        destination = sealed_root / relative
        kind = item.get("kind")
        if kind == "directory":
            destination.mkdir(parents=True, exist_ok=True)
            directories.append(destination)
        elif kind == "file":
            expected_digest = item.get("sha256")
            if not isinstance(expected_digest, str):
                raise ProvenanceError(f"captured config digest is invalid: {source}")
            _seal_file(
                source,
                destination,
                executable=False,
                expected_sha256=expected_digest,
            )
        elif kind == "symlink" and item.get("resolved_kind") == "file":
            expected_digest = item.get("resolved_sha256")
            if not isinstance(expected_digest, str):
                raise ProvenanceError(f"captured config symlink is invalid: {source}")
            _seal_file(
                source,
                destination,
                executable=False,
                expected_sha256=expected_digest,
            )
        else:
            raise ProvenanceError(f"cannot safely seal non-file config entry: {source}")
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        directory.chmod(0o555)
    sealed = _capture_config_inputs(sealed_root.resolve())
    if _capture_config_inputs(Path(str(original["declared_path"]))) != original:
        raise ProvenanceError("measurement configs changed while they were sealed")
    return sealed


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _write_immutable(path: Path, data: bytes) -> None:
    """Create evidence exactly once; a reused run ID is always an error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to replace immutable evidence: {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout: int = 30,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProvenanceError(f"could not run {command[0]!r}: {error}") from error
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ProvenanceError(
            f"command failed ({result.returncode}): {' '.join(command)}: {detail}"
        )
    return result


def _find_executable(name: str) -> str | None:
    """Find a command, including rustup's conventional user-local install."""
    found = shutil.which(name)
    if found is not None:
        return found
    cargo_candidate = Path.home() / ".cargo" / "bin" / name
    if cargo_candidate.is_file() and os.access(cargo_candidate, os.X_OK):
        return str(cargo_candidate)
    return None


def _git(repo: Path, *args: str, check: bool = True) -> bytes:
    return _run(["git", "-C", str(repo), *args], check=check).stdout


def _decode_nul_records(data: bytes) -> list[str]:
    return [os.fsdecode(item) for item in data.split(b"\0") if item]


def _untracked_evidence(repo: Path) -> list[dict[str, object]]:
    names = _git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    evidence: list[dict[str, object]] = []
    for raw_name in (item for item in names.split(b"\0") if item):
        relative = os.fsdecode(raw_name)
        path = repo / relative
        if path.is_symlink():
            target = os.fsencode(os.readlink(path))
            evidence.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "size_bytes": len(target),
                    "sha256": _sha256(target),
                }
            )
        elif path.is_file():
            evidence.append(
                {
                    "path": relative,
                    "kind": "file",
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        else:
            raise ProvenanceError(
                f"cannot hash non-file untracked worktree entry: {relative!r}"
            )
    return evidence


def _capture_worktree_once(repo: Path) -> dict[str, object]:
    commit = _git(repo, "rev-parse", "HEAD").decode().strip()
    status = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    tracked_diff = _git(repo, "diff", "--binary", "HEAD", "--")
    untracked = _untracked_evidence(repo)
    digest_payload = {
        "commit": commit,
        "status_sha256": _sha256(status),
        "tracked_diff_sha256": _sha256(tracked_diff),
        "untracked_files": untracked,
    }
    return {
        "commit": commit,
        "clean": not status,
        "status_porcelain_v1_z_entries": _decode_nul_records(status),
        "status_sha256": _sha256(status),
        "tracked_diff_bytes": len(tracked_diff),
        "tracked_diff_sha256": _sha256(tracked_diff),
        "untracked_files": untracked,
        "worktree_evidence_sha256": _sha256(
            json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ),
    }


def _capture_worktree(repo: Path) -> dict[str, object]:
    first = _capture_worktree_once(repo)
    second = _capture_worktree_once(repo)
    if first != second:
        raise ProvenanceError("NY worktree changed while provenance was captured")
    branch_result = _run(
        ["git", "-C", str(repo), "symbolic-ref", "--short", "-q", "HEAD"],
        check=False,
    )
    first["branch"] = (
        branch_result.stdout.decode("utf-8", "replace").strip()
        if branch_result.returncode == 0
        else None
    )
    first["repo_root"] = str(repo)
    return first


def _parse_toolchain(repo: Path) -> dict[str, object]:
    path = repo / "rust-toolchain.toml"
    if not path.is_file():
        raise ProvenanceError(f"missing pinned toolchain file: {path}")
    data = path.read_bytes()
    text = data.decode("utf-8")
    channel_match = re.search(r'^\s*channel\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if channel_match is None:
        raise ProvenanceError(f"could not parse toolchain channel from {path}")
    components_match = re.search(
        r"^\s*components\s*=\s*\[([^]]*)\]", text, re.MULTILINE
    )
    components = (
        re.findall(r'"([^"]+)"', components_match.group(1))
        if components_match is not None
        else []
    )
    channel = channel_match.group(1)
    rustup = _find_executable("rustup")
    if rustup is not None:
        version_result = _run(
            [rustup, "run", channel, "rustc", "--version", "--verbose"],
            check=False,
        )
        version_command = [rustup, "run", channel, "rustc", "--version", "--verbose"]
    else:
        rustc = _find_executable("rustc")
        if rustc is None:
            raise ProvenanceError("neither rustup nor rustc is available")
        version_result = _run([rustc, "--version", "--verbose"], check=False)
        version_command = [rustc, "--version", "--verbose"]
    return {
        "path": str(path),
        "sha256": _sha256(data),
        "channel": channel,
        "components": components,
        "version_command": version_command,
        "version_returncode": version_result.returncode,
        "version_stdout": version_result.stdout.decode("utf-8", "replace").strip(),
        "version_stderr": version_result.stderr.decode("utf-8", "replace").strip(),
    }


def _parse_ay_pin(repo: Path) -> dict[str, str]:
    lock_path = repo / "Cargo.lock"
    if not lock_path.is_file():
        raise ProvenanceError(f"missing Cargo.lock: {lock_path}")
    data = lock_path.read_bytes()
    matches = re.findall(
        rb"git\+https://github\.com/alabsystems/ay\.git"
        rb"\?rev=([0-9a-f]{40})#([0-9a-f]{40})",
        data,
    )
    pins = {(requested.decode(), resolved.decode()) for requested, resolved in matches}
    if len(pins) != 1:
        raise ProvenanceError(
            f"expected one exact AY revision in Cargo.lock, found {len(pins)}"
        )
    requested, resolved = next(iter(pins))
    if requested != resolved:
        raise ProvenanceError(
            f"AY requested revision {requested} resolved to different commit {resolved}"
        )
    return {
        "git_revision": resolved,
        "requested_revision": requested,
        "cargo_lock_sha256": _sha256(data),
    }


def _sanitize_remote_url(url: str) -> str:
    """Remove credentials, query strings, and fragments from a Git remote."""
    value = url.strip()
    if "://" in value:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname
        if parsed.port is not None:
            netloc += f":{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    value = value.split("?", 1)[0].split("#", 1)[0]
    scp_match = re.match(r"(?:[^@/:]+@)?([^:]+):(.+)", value)
    if scp_match is not None:
        return f"{scp_match.group(1)}:{scp_match.group(2)}"
    return value


def _capture_benchmark(benchmark_root: Path) -> dict[str, object]:
    top = _git(benchmark_root, "rev-parse", "--show-toplevel").decode().strip()
    repo = Path(top).resolve()
    worktree = _capture_worktree(repo)
    remotes: list[dict[str, str]] = []
    for name in _git(repo, "remote").decode("utf-8", "replace").splitlines():
        remote = _git(repo, "remote", "get-url", name).decode("utf-8", "replace")
        remotes.append({"name": name, "fetch_url": _sanitize_remote_url(remote)})
    return {
        **worktree,
        "benchmark_root": str(benchmark_root),
        "remotes": remotes,
    }


def _read_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"ID", "VERSION_ID", "PRETTY_NAME", "BUILD_ID"}:
            values[key] = value.strip().strip('"')
    return values


def _optional_command(command: list[str]) -> dict[str, object] | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    actual = [executable, *command[1:]]
    result = _run(actual, check=False, timeout=15)
    return {
        "command": actual,
        "returncode": result.returncode,
        "stdout": result.stdout.decode("utf-8", "replace").strip(),
        "stderr": result.stderr.decode("utf-8", "replace").strip(),
    }


def _bounded_decode(
    data: bytes, limit: int = SNAPSHOT_OUTPUT_LIMIT
) -> tuple[str, bool]:
    truncated = len(data) > limit
    return data[:limit].decode("utf-8", "replace").strip(), truncated


def _snapshot_command(command: list[str], *, timeout: int = 5) -> dict[str, object]:
    """Run a fixed diagnostic command without making provenance capture fatal."""
    result: dict[str, object] = {
        "command": command,
        "status": "unavailable",
        "returncode": None,
        "stdout": "",
        "stdout_truncated": False,
        "stderr": "",
        "stderr_truncated": False,
    }
    executable = shutil.which(command[0])
    if executable is None:
        return result
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        stdout, stdout_truncated = _bounded_decode(error.stdout or b"")
        stderr, stderr_truncated = _bounded_decode(error.stderr or b"")
        result.update(
            {
                "status": "timeout",
                "stdout": stdout,
                "stdout_truncated": stdout_truncated,
                "stderr": stderr,
                "stderr_truncated": stderr_truncated,
            }
        )
        return result
    except OSError as error:
        result.update({"status": "error", "stderr": str(error)[:1024]})
        return result
    stdout, stdout_truncated = _bounded_decode(completed.stdout)
    stderr, stderr_truncated = _bounded_decode(completed.stderr)
    result.update(
        {
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "stdout": stdout,
            "stdout_truncated": stdout_truncated,
            "stderr": stderr,
            "stderr_truncated": stderr_truncated,
        }
    )
    return result


def _redact_process_args(args: str) -> str:
    """Bound process arguments and redact common credential-bearing forms."""
    safe = " ".join(args.replace("\x00", " ").split())
    safe = re.sub(
        r"(?i)\b([a-z][a-z0-9+.-]*://)[^/@\s]+@",
        r"\1[REDACTED]@",
        safe,
    )
    safe = re.sub(
        rf"(?i)(\b(?=[A-Za-z_])[A-Za-z0-9_]*{_SENSITIVE_ARGUMENT}"
        rf"[A-Za-z0-9_]*\s*=\s*)(?:\"[^\"]*\"|'[^']*'|\S+)",
        r"\1[REDACTED]",
        safe,
    )
    safe = re.sub(
        rf"(?i)((?:^|\s)--?[A-Za-z0-9_-]*{_SENSITIVE_ARGUMENT}[A-Za-z0-9_-]*"
        rf"(?:\s*=\s*|\s+))(?:\"[^\"]*\"|'[^']*'|\S+)",
        r"\1[REDACTED]",
        safe,
    )
    safe = re.sub(r"(?i)(\bBearer\s+)\S+", r"\1[REDACTED]", safe)
    if len(safe) > PROCESS_ARGS_LIMIT:
        return safe[: PROCESS_ARGS_LIMIT - 1] + "…"
    return safe


def _process_snapshot() -> dict[str, object]:
    command = [
        "ps",
        "-eo",
        "pid=,ppid=,stat=,etimes=,pcpu=,pmem=,comm=,args=",
    ]
    captured = _snapshot_command(command)
    entries: list[dict[str, object]] = []
    if captured["status"] == "ok":
        for line in str(captured["stdout"]).splitlines():
            fields = line.strip().split(None, 7)
            if len(fields) < 7:
                continue
            try:
                pid = int(fields[0])
                ppid = int(fields[1])
                elapsed_seconds = int(fields[3])
                cpu_percent = float(fields[4])
                memory_percent = float(fields[5])
            except ValueError:
                continue
            entries.append(
                {
                    "pid": pid,
                    "ppid": ppid,
                    "state": fields[2][:16],
                    "elapsed_seconds": elapsed_seconds,
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "command": fields[6][:64],
                    "args_redacted": _redact_process_args(
                        fields[7] if len(fields) == 8 else fields[6]
                    ),
                }
            )
    entries.sort(
        key=lambda entry: (
            -float(entry["cpu_percent"]),
            -float(entry["memory_percent"]),
            int(entry["pid"]),
        )
    )
    total_observed = len(entries)
    return {
        "source": "ps",
        "status": captured["status"],
        "returncode": captured["returncode"],
        "stderr": captured["stderr"],
        "stderr_truncated": captured["stderr_truncated"],
        "max_entries": PROCESS_SNAPSHOT_LIMIT,
        "max_args_chars": PROCESS_ARGS_LIMIT,
        "sort": ["cpu_percent_desc", "memory_percent_desc", "pid_asc"],
        "total_observed": total_observed,
        "truncated": total_observed > PROCESS_SNAPSHOT_LIMIT
        or bool(captured["stdout_truncated"]),
        "entries": entries[:PROCESS_SNAPSHOT_LIMIT],
    }


def _load_average_snapshot() -> dict[str, object]:
    try:
        one, five, fifteen = os.getloadavg()
    except (AttributeError, OSError):
        return {
            "available": False,
            "one_minute": None,
            "five_minutes": None,
            "fifteen_minutes": None,
        }
    return {
        "available": True,
        "one_minute": one,
        "five_minutes": five,
        "fifteen_minutes": fifteen,
    }


def _gpu_activity_snapshot() -> dict[str, object]:
    return {
        "utilization": _snapshot_command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,utilization.memory,"
                "memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ]
        ),
        "compute_processes": _snapshot_command(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        ),
    }


def _capture_host_state() -> dict[str, object]:
    return {
        "schema": "ny_measurement_host_state_v1",
        "captured_at_utc": _utc_now(),
        "load_average": _load_average_snapshot(),
        "processes": _process_snapshot(),
        "gpu": _gpu_activity_snapshot(),
    }


def _cpu_identity() -> dict[str, object]:
    identity: dict[str, object] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    if hasattr(os, "sched_getaffinity"):
        identity["allowed_logical_cpus"] = sorted(os.sched_getaffinity(0))
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        selected: dict[str, list[str]] = {}
        accepted = {
            "model name",
            "hardware",
            "processor",
            "cpu implementer",
            "cpu part",
        }
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, value = (item.strip() for item in line.split(":", 1))
            if key.lower() in accepted and value and value not in selected.get(key, []):
                selected.setdefault(key, []).append(value)
        identity["proc_cpuinfo_identity"] = selected
    identity["lscpu"] = _optional_command(["lscpu", "--json"])
    return identity


def _memory_identity() -> dict[str, object]:
    identity: dict[str, object] = {}
    if hasattr(os, "sysconf"):
        try:
            identity["physical_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf(
                "SC_PHYS_PAGES"
            )
        except (OSError, ValueError):
            pass
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(("MemTotal:", "SwapTotal:")):
                key, value = line.split(":", 1)
                identity[key] = value.strip()
    return identity


def _gpu_identity() -> dict[str, object]:
    query = _optional_command(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "nvidia_smi_query": query,
        "nvcc_version": _optional_command(["nvcc", "--version"]),
        "display_pci_devices": _optional_command(["lspci", "-nn"]),
        "macos_displays": _optional_command(
            ["system_profiler", "SPDisplaysDataType", "-json"]
        ),
    }


def _resource_limits() -> dict[str, list[int]]:
    names = [
        "RLIMIT_AS",
        "RLIMIT_CORE",
        "RLIMIT_CPU",
        "RLIMIT_DATA",
        "RLIMIT_FSIZE",
        "RLIMIT_MEMLOCK",
        "RLIMIT_NOFILE",
        "RLIMIT_NPROC",
        "RLIMIT_STACK",
    ]
    limits: dict[str, list[int]] = {}
    for name in names:
        if hasattr(resource, name):
            soft, hard = resource.getrlimit(getattr(resource, name))
            limits[name] = [soft, hard]
    return limits


def _capture_environment() -> dict[str, object]:
    unknown_solver = sorted(
        key
        for key in os.environ
        if key.startswith(("NY_", "AY_", "MIMALLOC_")) and key not in ENV_ALLOWLIST
    )
    if unknown_solver:
        raise ProvenanceError(
            "unrecorded NY_*/AY_*/MIMALLOC_* environment variables are set; "
            "add a reviewed non-secret key to the fixed allowlist or unset it: "
            f"{', '.join(unknown_solver)}"
        )
    values: dict[str, str] = {
        key: os.environ[key] for key in sorted(ENV_ALLOWLIST) if key in os.environ
    }
    typed_values: dict[str, dict[str, object]] = {}
    usize_max = (sys.maxsize * 2) + 1
    for key in sorted(TYPED_NONNEGATIVE_INTEGER_ENV):
        if key not in values:
            continue
        raw = values[key]
        if re.fullmatch(r"[0-9]+", raw) is None:
            raise ProvenanceError(
                f"{key} must be a nonnegative integer for measurement provenance"
            )
        value = int(raw)
        if value > usize_max:
            raise ProvenanceError(
                f"{key} is outside the native usize range for measurement provenance"
            )
        typed_values[key] = {
            "type": "nonnegative_integer",
            "value": value,
        }
    for key in sorted(TYPED_POSITIVE_INTEGER_ENV):
        if key not in values:
            continue
        raw = values[key]
        if re.fullmatch(r"[0-9]+", raw) is None:
            raise ProvenanceError(
                f"{key} must be a positive integer for measurement provenance"
            )
        value = int(raw)
        if value > usize_max:
            raise ProvenanceError(
                f"{key} is outside the native usize range for measurement provenance"
            )
        if value == 0:
            raise ProvenanceError(
                f"{key} must be a positive integer for measurement provenance"
            )
        typed_values[key] = {
            "type": "positive_integer",
            "value": value,
        }
    decimal_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    for key, minimum, maximum in TYPED_BOUNDED_FLOAT_ENV:
        if key not in values:
            continue
        raw = values[key]
        if re.fullmatch(decimal_pattern, raw) is None:
            raise ProvenanceError(
                f"{key} must be a decimal number for measurement provenance"
            )
        value = float(raw)
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ProvenanceError(
                f"{key} must be finite and within [{minimum}, {maximum}] "
                "for measurement provenance"
            )
        typed_values[key] = {
            "type": "bounded_float",
            "value": value,
            "minimum": minimum,
            "maximum": maximum,
        }
    for key, allowed_values in TYPED_ENUM_ENV:
        if key not in values:
            continue
        raw = values[key]
        if raw not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise ProvenanceError(
                f"{key} must be unset or one of [{allowed}] "
                "for measurement provenance"
            )
        typed_values[key] = {
            "type": "enum",
            "value": raw,
            "allowed_values": sorted(allowed_values),
        }
    for key in sorted(TYPED_STRICT_BOOLEAN_ENV):
        if key not in values:
            continue
        raw = values[key]
        # The runtime gives an explicitly empty value the same false meaning as
        # "0". Measurement syntax deliberately narrows the runtime's broader
        # u32/fallback parser to the reviewed boolean spellings below so a typo
        # cannot silently disable GPU routing in scored evidence.
        if raw not in {"", "0", "1"}:
            raise ProvenanceError(
                f"{key} must be unset, empty, '0', or '1' for measurement provenance"
            )
        typed_values[key] = {
            "type": "boolean",
            "value": raw == "1",
        }
    for key in sorted(TYPED_EXACT_BOOLEAN_ENV):
        if key not in values:
            continue
        raw = values[key]
        if raw not in {"0", "1"}:
            raise ProvenanceError(
                f"{key} must be unset, '0', or '1' for measurement provenance"
            )
        typed_values[key] = {
            "type": "boolean",
            "value": raw == "1",
        }
    return {
        "allowlist_schema": "ny_measurement_environment_v1",
        "values": values,
        "typed_values": typed_values,
    }


def _declared_build_features() -> tuple[str, list[str]]:
    raw = os.environ.get("NY_BUILD_FEATURES", "").strip()
    if not raw:
        raise ProvenanceError(
            "NY_BUILD_FEATURES is required; provenance cannot infer Cargo "
            "features from an already-built binary"
        )
    features = [feature.strip() for feature in raw.split(",")]
    if any(
        not feature or re.fullmatch(r"[A-Za-z0-9_+./?:-]+", feature) is None
        for feature in features
    ):
        raise ProvenanceError(
            "NY_BUILD_FEATURES must be a comma-separated list of Cargo feature names"
        )
    if len(set(features)) != len(features):
        raise ProvenanceError("NY_BUILD_FEATURES contains a duplicate feature")
    return raw, features


def _resolve_from(root: Path, path: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_mutation_root(path: Path, repo_root: Path, label: str) -> None:
    filesystem_root = Path(path.anchor)
    if path == filesystem_root or path == repo_root or path in repo_root.parents:
        raise ProvenanceError(f"unsafe broad {label}: {path}")
    git_dir = (repo_root / ".git").resolve()
    if path == git_dir or _is_within(path, git_dir):
        raise ProvenanceError(f"{label} must not be inside the NY Git metadata: {path}")
    if path.exists() and not path.is_dir():
        raise ProvenanceError(f"{label} exists but is not a directory: {path}")
    if _is_within(path, repo_root):
        relative = path.relative_to(repo_root)
        tracked = _git(repo_root, "ls-files", "-z", "--", str(relative))
        if tracked:
            raise ProvenanceError(
                f"{label} contains tracked NY paths and would invalidate its own "
                f"measurement: {path}"
            )
        ignored = _run(
            [
                "git",
                "-C",
                str(repo_root),
                "check-ignore",
                "-q",
                "--no-index",
                "--",
                str(relative),
            ],
            check=False,
        )
        if ignored.returncode != 0:
            raise ProvenanceError(
                f"{label} inside the NY worktree must be Git-ignored: {path}"
            )


def capture_start_manifest(
    *,
    repo_root: Path,
    binary: Path,
    benchmark_root: Path,
    artifact_root: Path,
    run_id: str,
    output_dir: Path,
    scratch_dir: Path,
    result_file: Path,
    solver_log_file: Path,
    categories_raw: str,
    timeout_cap_seconds: int,
    watchdog_grace_seconds: int,
    max_rows_per_category: int,
    instance_index: int,
    vnnlib_version: str,
    sweep_script: Path,
    configs_dir: Path | None = None,
) -> Path:
    """Capture and immutably create one run's start manifest."""
    if SAFE_COMPONENT.fullmatch(run_id) is None:
        raise ValueError(f"unsafe run ID: {run_id!r}")
    repo_root = repo_root.resolve()
    binary = _resolve_from(repo_root, binary)
    benchmark_root = _resolve_from(repo_root, benchmark_root)
    artifact_root = _resolve_from(repo_root, artifact_root)
    output_dir = _resolve_from(repo_root, output_dir)
    scratch_dir = _resolve_from(repo_root, scratch_dir)
    result_file = _resolve_from(repo_root, result_file)
    solver_log_file = _resolve_from(repo_root, solver_log_file)
    sweep_script = _resolve_from(repo_root, sweep_script)
    if not binary.is_file():
        raise ProvenanceError(f"solver binary does not exist: {binary}")
    if not benchmark_root.is_dir():
        raise ProvenanceError(f"benchmark root does not exist: {benchmark_root}")
    if not sweep_script.is_file():
        raise ProvenanceError(f"sweep script does not exist: {sweep_script}")
    if timeout_cap_seconds <= 0 or watchdog_grace_seconds < 0:
        raise ValueError("timeout cap must be positive and watchdog grace nonnegative")
    if max_rows_per_category < 0:
        raise ValueError("maximum rows per category must be nonnegative")
    if instance_index < 0:
        raise ValueError("instance index must be nonnegative")
    if vnnlib_version not in {"", "1.0", "2.0"}:
        raise ValueError("VNN-LIB version selection must be empty, 1.0, or 2.0")
    _validate_mutation_root(output_dir, repo_root, "measurement output directory")
    _validate_mutation_root(artifact_root, repo_root, "measurement artifact root")
    _validate_mutation_root(scratch_dir, repo_root, "measurement scratch directory")
    if not _is_within(result_file, scratch_dir):
        raise ProvenanceError(
            "result scratch file must be inside the scratch directory"
        )
    if not _is_within(solver_log_file, scratch_dir):
        raise ProvenanceError("solver log file must be inside the scratch directory")
    if result_file == solver_log_file:
        raise ProvenanceError("result and solver-log scratch files must be distinct")

    start_path = artifact_root / "runs" / run_id / "start.json"
    run_dir = start_path.parent
    config_inputs = (
        _capture_config_inputs(configs_dir) if configs_dir is not None else None
    )
    environment = _capture_environment()
    build_features_raw, build_features = _declared_build_features()
    ay_dependency: dict[str, object] = dict(_parse_ay_pin(repo_root))
    ay_dependency["executable"] = _capture_ay_executable(repo_root)
    binary_digest, binary_fingerprint = _stable_file_hash(binary)
    sealed_binary = _seal_file(
        binary,
        run_dir / "sealed" / "solver" / binary_digest / binary.name,
        executable=True,
        expected_sha256=binary_digest,
        expected_fingerprint=binary_fingerprint,
    )
    ay_executable = ay_dependency["executable"]
    if isinstance(ay_executable, dict):
        ay_path = ay_executable.get("resolved_path")
        ay_digest = ay_executable.get("sha256")
        ay_fingerprint = ay_executable.get("fingerprint")
        if (
            not isinstance(ay_path, str)
            or not isinstance(ay_digest, str)
            or not isinstance(ay_fingerprint, dict)
        ):
            raise ProvenanceError("captured AY executable identity is incomplete")
        ay_dependency["sealed_executable"] = _seal_file(
            Path(ay_path),
            run_dir / "sealed" / "ay" / ay_digest / Path(ay_path).name,
            executable=True,
            expected_sha256=ay_digest,
            expected_fingerprint=ay_fingerprint,
        )
    else:
        ay_dependency["sealed_executable"] = None
    sealed_config_inputs = (
        _seal_config_inputs(config_inputs, run_dir)
        if config_inputs is not None
        else None
    )
    sealed_binary_path = str(sealed_binary["path"])
    binary_version = _run([sealed_binary_path, "--version"], check=False, timeout=15)
    if _file_fingerprint(binary) != binary_fingerprint:
        raise ProvenanceError(
            f"solver binary changed during provenance capture: {binary}"
        )
    categories = categories_raw.split()
    if not categories:
        raise ValueError("measurement category list is empty")
    solver_command_template = [
        sealed_binary_path,
        "vnncomp",
        "v1",
        "<category>",
        "<onnx>",
        "<vnnlib>",
        str(result_file),
        "<capped_timeout_seconds>",
    ]
    if sealed_config_inputs is not None:
        solver_command_template.extend(
            ["--configs-dir", str(sealed_config_inputs["resolved_path"])]
        )
    payload = {
        "schema": "ny_measurement_start_v1",
        "run_id": run_id,
        "started_at_utc": _utc_now(),
        "ny": _capture_worktree(repo_root),
        "solver_binary": {
            "path": str(binary),
            "size_bytes": binary_fingerprint["size_bytes"],
            "sha256": binary_digest,
            "fingerprint": binary_fingerprint,
            "sealed_execution": sealed_binary,
            "version_returncode": binary_version.returncode,
            "version_stdout": binary_version.stdout.decode("utf-8", "replace").strip(),
            "version_stderr": binary_version.stderr.decode("utf-8", "replace").strip(),
            "declared_build_features": build_features,
            "declared_build_features_raw": build_features_raw,
        },
        "dependencies": {"ay": ay_dependency},
        "rust_toolchain": _parse_toolchain(repo_root),
        "benchmark": _capture_benchmark(benchmark_root),
        "measurement": {
            "sweep_invocation": [str(sweep_script)],
            "categories_raw": categories_raw,
            "categories": categories,
            "timeout_cap_seconds": timeout_cap_seconds,
            "watchdog_grace_seconds": watchdog_grace_seconds,
            "max_rows_per_category": (
                max_rows_per_category if max_rows_per_category > 0 else None
            ),
            "instance_index": instance_index if instance_index > 0 else None,
            "vnnlib_version_selection": vnnlib_version or None,
            "config_inputs": config_inputs,
            "sealed_config_inputs": sealed_config_inputs,
            "benchmark_root": str(benchmark_root),
            "output_dir": str(output_dir),
            "artifact_root": str(artifact_root),
            "scratch_dir": str(scratch_dir),
            "result_file": str(result_file),
            "solver_log_file": str(solver_log_file),
            "solver_output_capture": "combined_stdout_stderr_exact_bytes",
            "csv_columns": [
                "category",
                "onnx",
                "vnnlib",
                "prepare_seconds",
                "result",
                "runtime_seconds",
                "run_id",
            ],
            "solver_command_template": solver_command_template,
            "solver_environment_overrides": {"RUST_LOG": "error"},
        },
        "environment": environment,
        "host": {
            "platform": platform.platform(),
            "uname": list(platform.uname()),
            "os_release": _read_os_release(),
            "cpu": _cpu_identity(),
            "memory": _memory_identity(),
            "gpu": _gpu_identity(),
            "resource_limits": _resource_limits(),
        },
        "host_state": _capture_host_state(),
    }
    _write_immutable(start_path, _json_bytes(payload))
    return start_path


def _add_integrity_violation(
    violations: list[dict[str, str]],
    code: str,
    detail: str,
) -> None:
    violations.append({"code": code, "detail": detail})


def _identity_sha256(identity: object) -> str:
    return _sha256(_json_bytes(identity))


def _validate_solver_binary(
    start: dict[str, object],
    violations: list[dict[str, str]],
) -> dict[str, object]:
    violation_count = len(violations)
    check: dict[str, object] = {"status": "invalid"}
    expected = start.get("solver_binary")
    if not isinstance(expected, dict):
        _add_integrity_violation(
            violations,
            "solver_binary_start_identity_invalid",
            "start manifest solver-binary identity is missing or invalid",
        )
        return check
    path_value = expected.get("path")
    expected_digest = expected.get("sha256")
    expected_fingerprint = expected.get("fingerprint")
    if (
        not isinstance(path_value, str)
        or not isinstance(expected_digest, str)
        or not isinstance(expected_fingerprint, dict)
    ):
        _add_integrity_violation(
            violations,
            "solver_binary_start_identity_invalid",
            "start manifest solver-binary identity is incomplete",
        )
        return check
    check.update(
        {
            "path": path_value,
            "expected_sha256": expected_digest,
            "expected_fingerprint": expected_fingerprint,
        }
    )
    try:
        path = Path(path_value)
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise ProvenanceError(
                f"solver binary is not an executable file: {resolved}"
            )
        observed_digest, observed_fingerprint = _stable_file_hash(resolved)
    except (OSError, ProvenanceError) as error:
        _add_integrity_violation(
            violations,
            "solver_binary_unavailable",
            str(error),
        )
        return check
    check.update(
        {
            "resolved_path": str(resolved),
            "observed_sha256": observed_digest,
            "observed_fingerprint": observed_fingerprint,
        }
    )
    if str(resolved) != path_value:
        _add_integrity_violation(
            violations,
            "solver_binary_resolved_path_mismatch",
            f"expected {path_value}, observed {resolved}",
        )
    if observed_digest != expected_digest:
        _add_integrity_violation(
            violations,
            "solver_binary_sha256_mismatch",
            f"expected {expected_digest}, observed {observed_digest}",
        )
    if observed_fingerprint != expected_fingerprint:
        _add_integrity_violation(
            violations,
            "solver_binary_fingerprint_mismatch",
            "solver-binary stat fingerprint changed after start capture",
        )
    check["status"] = "valid" if len(violations) == violation_count else "invalid"
    return check


def _validate_ay_executable(
    start: dict[str, object],
    violations: list[dict[str, str]],
) -> dict[str, object]:
    violation_count = len(violations)
    check: dict[str, object] = {"status": "invalid"}
    dependencies = start.get("dependencies")
    ay = dependencies.get("ay") if isinstance(dependencies, dict) else None
    if not isinstance(ay, dict) or "executable" not in ay:
        _add_integrity_violation(
            violations,
            "ay_executable_start_identity_invalid",
            "start manifest AY executable identity is missing",
        )
        return check
    expected = ay["executable"]
    if expected is None:
        return {"status": "not_configured"}
    if not isinstance(expected, dict):
        _add_integrity_violation(
            violations,
            "ay_executable_start_identity_invalid",
            "start manifest AY executable identity is invalid",
        )
        return check
    declared_path = expected.get("declared_path")
    ny = start.get("ny")
    repo_root = ny.get("repo_root") if isinstance(ny, dict) else None
    if not isinstance(declared_path, str) or not isinstance(repo_root, str):
        _add_integrity_violation(
            violations,
            "ay_executable_start_identity_invalid",
            "start manifest AY executable path or NY repository root is invalid",
        )
        return check
    check["expected_identity_sha256"] = _identity_sha256(expected)
    try:
        observed = _capture_executable_identity(
            declared_path,
            base_dir=Path(repo_root),
            label="AY",
        )
    except (OSError, ProvenanceError) as error:
        _add_integrity_violation(
            violations,
            "ay_executable_unavailable",
            str(error),
        )
        return check
    check["observed_identity_sha256"] = _identity_sha256(observed)
    check["resolved_path"] = observed["resolved_path"]
    if observed != expected:
        _add_integrity_violation(
            violations,
            "ay_executable_identity_mismatch",
            "resolved AY executable identity changed after start capture",
        )
    check["status"] = "valid" if len(violations) == violation_count else "invalid"
    return check


def _validate_recaptured_identity(
    *,
    name: str,
    expected: object,
    capture: Callable[[], dict[str, object]],
    violations: list[dict[str, str]],
) -> dict[str, object]:
    violation_count = len(violations)
    check: dict[str, object] = {"status": "invalid"}
    if not isinstance(expected, dict):
        _add_integrity_violation(
            violations,
            f"{name}_start_identity_invalid",
            f"start manifest {name.replace('_', ' ')} identity is missing or invalid",
        )
        return check
    check["expected_identity_sha256"] = _identity_sha256(expected)
    try:
        observed = capture()
    except (OSError, ValueError, ProvenanceError) as error:
        _add_integrity_violation(
            violations,
            f"{name}_unavailable",
            str(error),
        )
        return check
    check["observed_identity_sha256"] = _identity_sha256(observed)
    if observed != expected:
        _add_integrity_violation(
            violations,
            f"{name}_identity_mismatch",
            f"{name.replace('_', ' ')} identity changed after start capture",
        )
    check["status"] = "valid" if len(violations) == violation_count else "invalid"
    return check


def _validate_config_inputs(
    start: dict[str, object],
    violations: list[dict[str, str]],
) -> dict[str, object]:
    measurement = start.get("measurement")
    if not isinstance(measurement, dict) or "config_inputs" not in measurement:
        _add_integrity_violation(
            violations,
            "config_inputs_start_identity_invalid",
            "start manifest config-input identity is missing",
        )
        return {"status": "invalid"}
    expected = measurement["config_inputs"]
    if expected is None:
        return {"status": "not_configured"}
    if not isinstance(expected, dict):
        _add_integrity_violation(
            violations,
            "config_inputs_start_identity_invalid",
            "start manifest config-input identity is invalid",
        )
        return {"status": "invalid"}
    declared_path = expected.get("declared_path")
    if not isinstance(declared_path, str):
        _add_integrity_violation(
            violations,
            "config_inputs_start_identity_invalid",
            "start manifest config-input path is invalid",
        )
        return {"status": "invalid"}
    return _validate_recaptured_identity(
        name="config_inputs",
        expected=expected,
        capture=lambda: _capture_config_inputs(Path(declared_path)),
        violations=violations,
    )


def _validate_sealed_file(
    *,
    expected: object,
    name: str,
    executable: bool,
    violations: list[dict[str, str]],
) -> dict[str, object]:
    violation_count = len(violations)
    check: dict[str, object] = {"status": "invalid"}
    if expected is None:
        return {"status": "not_configured"}
    if not isinstance(expected, dict):
        _add_integrity_violation(
            violations,
            f"{name}_start_identity_invalid",
            f"start manifest {name.replace('_', ' ')} identity is invalid",
        )
        return check
    path_value = expected.get("path")
    expected_digest = expected.get("sha256")
    expected_fingerprint = expected.get("fingerprint")
    if (
        expected.get("schema") != "ny_measurement_sealed_file_v1"
        or not isinstance(path_value, str)
        or not isinstance(expected_digest, str)
        or not isinstance(expected_fingerprint, dict)
    ):
        _add_integrity_violation(
            violations,
            f"{name}_start_identity_invalid",
            f"start manifest {name.replace('_', ' ')} identity is incomplete",
        )
        return check
    try:
        path = Path(path_value)
        if path.is_symlink():
            raise ProvenanceError(f"sealed file is a symlink: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or (executable and not os.access(resolved, os.X_OK)):
            raise ProvenanceError(f"sealed file is unavailable: {resolved}")
        observed_digest, observed_fingerprint = _stable_file_hash(resolved)
    except (OSError, ProvenanceError) as error:
        _add_integrity_violation(violations, f"{name}_unavailable", str(error))
        return check
    check.update(
        {
            "path": str(resolved),
            "expected_sha256": expected_digest,
            "observed_sha256": observed_digest,
            "expected_fingerprint": expected_fingerprint,
            "observed_fingerprint": observed_fingerprint,
        }
    )
    if str(resolved) != path_value:
        _add_integrity_violation(
            violations,
            f"{name}_resolved_path_mismatch",
            f"expected {path_value}, observed {resolved}",
        )
    if observed_digest != expected_digest:
        _add_integrity_violation(
            violations,
            f"{name}_sha256_mismatch",
            f"expected {expected_digest}, observed {observed_digest}",
        )
    if observed_fingerprint != expected_fingerprint:
        _add_integrity_violation(
            violations,
            f"{name}_fingerprint_mismatch",
            f"{name.replace('_', ' ')} fingerprint changed after sealing",
        )
    check["status"] = "valid" if len(violations) == violation_count else "invalid"
    return check


def _validate_sealed_config_inputs(
    start: dict[str, object], violations: list[dict[str, str]]
) -> dict[str, object]:
    measurement = start.get("measurement")
    expected = (
        measurement.get("sealed_config_inputs")
        if isinstance(measurement, dict)
        else object()
    )
    if expected is None:
        return {"status": "not_configured"}
    if not isinstance(expected, dict) or not isinstance(
        expected.get("declared_path"), str
    ):
        _add_integrity_violation(
            violations,
            "sealed_config_inputs_start_identity_invalid",
            "start manifest sealed config-input identity is invalid",
        )
        return {"status": "invalid"}
    return _validate_recaptured_identity(
        name="sealed_config_inputs",
        expected=expected,
        capture=lambda: _capture_config_inputs(Path(str(expected["declared_path"]))),
        violations=violations,
    )


def _input_cache_key(path: str, fingerprint: dict[str, int]) -> str:
    identity = json.dumps(
        [path, fingerprint], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256(identity)


def _stable_file_bytes(path: Path) -> tuple[bytes, str, dict[str, int]]:
    """Read a regular non-symlink file and reject concurrent replacement."""
    if path.is_symlink() or not path.is_file():
        raise ProvenanceError(
            f"evidence path is not a regular non-symlink file: {path}"
        )
    before = _file_fingerprint(path)
    data = path.read_bytes()
    after = _file_fingerprint(path)
    if before != after:
        raise ProvenanceError(f"file changed while evidence was captured: {path}")
    return data, _sha256(data), after


def _balanced_parentheses(text: str) -> bool:
    depth = 0
    for character in text:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _structured_sat_assignment(lines: list[bytes]) -> bool:
    """Recognize the same complete VNN-LIB 1.x/2.0 witness as the archiver."""
    try:
        payload = [line.decode("utf-8").strip() for line in lines if line.strip()]
    except UnicodeDecodeError:
        return False
    if not payload:
        return False
    legacy = "\n".join(payload)
    if legacy.startswith("("):
        return (
            _balanced_parentheses(legacy)
            and V1_INPUT_ASSIGNMENT.search(legacy) is not None
        )

    position = 0
    declarations = 0
    while position < len(payload):
        match = V2_ASSIGNMENT_HEADER.fullmatch(payload[position])
        if match is None:
            return False
        dimensions = match.group(3).strip()
        try:
            shape = (
                []
                if not dimensions
                else [int(value.strip()) for value in dimensions.split(",")]
            )
        except ValueError:
            return False
        if any(dimension <= 0 for dimension in shape):
            return False
        value_count = 1
        for dimension in shape:
            value_count *= dimension
        position += 1
        if position + value_count > len(payload):
            return False
        for value in payload[position : position + value_count]:
            if len(value.split()) != 1:
                return False
            try:
                float(value)
            except ValueError:
                if value.lower() not in {"true", "false"}:
                    return False
        position += value_count
        declarations += 1
    return declarations > 0


def _expected_metadata_config_identity(start: dict[str, object]) -> object:
    measurement = start.get("measurement")
    config_inputs = (
        measurement.get("config_inputs") if isinstance(measurement, dict) else None
    )
    if config_inputs is None:
        return None
    if not isinstance(config_inputs, dict):
        return object()
    required = (
        "schema",
        "declared_path",
        "resolved_path",
        "entry_count",
        "manifest_sha256",
    )
    if any(key not in config_inputs for key in required):
        return object()
    return {key: config_inputs[key] for key in required}


def _discover_run_artifacts(
    *,
    artifact_root: Path,
    run_id: str,
    violations: list[dict[str, str]],
) -> dict[Path, dict[str, Path]]:
    """Discover exact current-run artifact names without trusting metadata paths."""
    discovered: dict[Path, dict[str, Path]] = {}
    expected_names = {
        f"{run_id}.json": "metadata",
        f"{run_id}.results": "result",
        f"{run_id}.solver.log": "solver_log",
        f"{run_id}.preflight.json": "preflight",
    }
    try:
        for directory, directory_names, file_names in os.walk(
            artifact_root, followlinks=False
        ):
            root = Path(directory)
            if root == artifact_root and "runs" in directory_names:
                directory_names.remove("runs")
            for name in list(directory_names):
                child = root / name
                if child.is_symlink():
                    _add_integrity_violation(
                        violations,
                        "run_artifact_symlink_directory",
                        f"artifact tree contains a directory symlink: {child}",
                    )
                    directory_names.remove(name)
            for name in file_names:
                kind = expected_names.get(name)
                if kind is None:
                    continue
                path = root / name
                discovered.setdefault(root, {})[kind] = path
    except OSError as error:
        _add_integrity_violation(
            violations,
            "run_artifact_discovery_failed",
            str(error),
        )
    return discovered


def _run_artifact_namespace(
    discovered: dict[Path, dict[str, Path]], artifact_root: Path
) -> list[tuple[str, str]]:
    return sorted(
        (
            path.relative_to(artifact_root).as_posix(),
            kind,
        )
        for artifacts in discovered.values()
        for kind, path in artifacts.items()
    )


def _revalidate_run_evidence_snapshot(
    *,
    snapshot: dict[str, object],
    run_id: str,
    violations: list[dict[str, str]],
) -> None:
    """Reject artifact/CSV mutations that race the cache postflight."""
    artifact_root = snapshot.get("artifact_root")
    output_dir = snapshot.get("output_dir")
    categories = snapshot.get("categories")
    expected_namespace = snapshot.get("artifact_namespace")
    file_fingerprints = snapshot.get("file_fingerprints")
    expected_csv_paths = snapshot.get("csv_paths")
    if (
        not isinstance(artifact_root, Path)
        or not isinstance(output_dir, Path)
        or not isinstance(categories, list)
        or not isinstance(expected_namespace, list)
        or not isinstance(file_fingerprints, dict)
        or not isinstance(expected_csv_paths, set)
    ):
        _add_integrity_violation(
            violations,
            "run_evidence_snapshot_invalid",
            "internal run-evidence snapshot is missing or malformed",
        )
        return

    discovery_violations: list[dict[str, str]] = []
    observed = _discover_run_artifacts(
        artifact_root=artifact_root,
        run_id=run_id,
        violations=discovery_violations,
    )
    if (
        discovery_violations
        or _run_artifact_namespace(observed, artifact_root) != expected_namespace
    ):
        _add_integrity_violation(
            violations,
            "run_artifact_namespace_changed_during_completion",
            "current-run artifact namespace changed during completion validation",
        )

    candidate_csvs: set[Path] = {
        output_dir / f"{category}.csv"
        for category in categories
        if isinstance(category, str)
    }
    if output_dir.is_dir():
        candidate_csvs.update(output_dir.glob("*.csv"))
    observed_csv_paths = {path for path in candidate_csvs if path.exists()}
    if observed_csv_paths != expected_csv_paths:
        _add_integrity_violation(
            violations,
            "run_csv_namespace_changed_during_completion",
            "measurement CSV namespace changed during completion validation",
        )

    for path, expected_fingerprint in file_fingerprints.items():
        if not isinstance(path, Path) or not isinstance(expected_fingerprint, dict):
            _add_integrity_violation(
                violations,
                "run_evidence_snapshot_invalid",
                "internal evidence fingerprint is malformed",
            )
            continue
        try:
            if path.is_symlink() or not path.is_file():
                raise ProvenanceError(f"evidence path is unavailable: {path}")
            observed_fingerprint = _file_fingerprint(path)
        except (OSError, ProvenanceError) as error:
            _add_integrity_violation(
                violations,
                "run_evidence_file_changed_during_completion",
                str(error),
            )
            continue
        if observed_fingerprint != expected_fingerprint:
            _add_integrity_violation(
                violations,
                "run_evidence_file_changed_during_completion",
                f"evidence file changed during completion validation: {path}",
            )


def _metadata_input_reference(
    *,
    metadata: dict[str, object],
    label: str,
    metadata_path: Path,
    references: dict[str, dict[str, object]],
    violations: list[dict[str, str]],
) -> tuple[str, str, str, int] | None:
    value = metadata.get(label)
    if not isinstance(value, dict):
        _add_integrity_violation(
            violations,
            "run_metadata_input_invalid",
            f"{metadata_path} has invalid {label} evidence",
        )
        return None
    declared_path = value.get("declared_path")
    resolved_path = value.get("resolved_path")
    digest = value.get("sha256")
    size_bytes = value.get("size_bytes")
    cache_key = value.get("hash_cache_key")
    cache_hit = value.get("hash_cache_hit")
    if (
        not isinstance(declared_path, str)
        or not isinstance(resolved_path, str)
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or type(size_bytes) is not int
        or size_bytes < 0
        or not isinstance(cache_key, str)
        or re.fullmatch(r"[0-9a-f]{64}", cache_key) is None
        or type(cache_hit) is not bool
    ):
        _add_integrity_violation(
            violations,
            "run_metadata_input_invalid",
            f"{metadata_path} has incomplete {label} evidence",
        )
        return None
    reference = {
        "path": resolved_path,
        "sha256": digest,
        "size_bytes": size_bytes,
    }
    previous = references.get(cache_key)
    if previous is not None and previous != reference:
        _add_integrity_violation(
            violations,
            "run_metadata_cache_reference_conflict",
            f"cache key {cache_key} has conflicting row evidence",
        )
        return None
    references[cache_key] = reference
    return declared_path, resolved_path, cache_key, size_bytes


def _validate_row_preflight(
    *,
    preflight_path: Path,
    metadata_path: Path,
    metadata: dict[str, object],
    artifact_root: Path,
    start_manifest: Path,
    start_digest: str,
    category: str,
    instance_index: int,
    onnx: str,
    vnnlib: str,
    file_fingerprints: dict[Path, dict[str, int]],
    violations: list[dict[str, str]],
) -> dict[str, object] | None:
    """Validate and rehash a row's before-execution inputs and sealed copies."""
    try:
        data, digest, fingerprint = _stable_file_bytes(preflight_path)
        preflight = json.loads(data)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ProvenanceError,
    ) as error:
        _add_integrity_violation(
            violations,
            "run_preflight_unreadable",
            f"could not validate {preflight_path}: {error}",
        )
        return None
    file_fingerprints[preflight_path] = fingerprint
    try:
        start_artifact = start_manifest.relative_to(artifact_root).as_posix()
        preflight_artifact = preflight_path.relative_to(artifact_root).as_posix()
    except ValueError:
        _add_integrity_violation(
            violations,
            "run_preflight_path_invalid",
            f"preflight is outside the artifact root: {preflight_path}",
        )
        return None
    metadata_link = metadata.get("input_preflight")
    if (
        not isinstance(preflight, dict)
        or preflight.get("schema") != "ny_measurement_input_preflight_v1"
        or preflight.get("run_id") != metadata.get("run_id")
        or preflight.get("category") != category
        or preflight.get("instance_index") != instance_index
        or preflight.get("start_manifest") != start_artifact
        or preflight.get("start_manifest_sha256") != start_digest
        or not isinstance(metadata_link, dict)
        or metadata_link.get("schema") != "ny_measurement_input_preflight_v1"
        or metadata_link.get("artifact") != preflight_artifact
        or metadata_link.get("sha256") != digest
    ):
        _add_integrity_violation(
            violations,
            "run_preflight_identity_mismatch",
            f"preflight identity or metadata link does not match: {preflight_path}",
        )
        return None
    inputs = preflight.get("inputs")
    execution_inputs = metadata.get("execution_inputs")
    if not isinstance(inputs, dict) or not isinstance(execution_inputs, dict):
        _add_integrity_violation(
            violations,
            "run_preflight_inputs_invalid",
            f"preflight execution inputs are missing: {preflight_path}",
        )
        return None
    evidence: dict[str, object] = {
        "artifact": preflight_artifact,
        "sha256": digest,
        "size_bytes": fingerprint["size_bytes"],
        "inputs": {},
    }
    evidence_inputs = evidence["inputs"]
    assert isinstance(evidence_inputs, dict)
    for label, declared_name in (("onnx", onnx), ("vnnlib", vnnlib)):
        value = inputs.get(label)
        metadata_original = metadata.get(label)
        if (
            not isinstance(value, dict)
            or value.get("declared_name") != declared_name
            or not isinstance(metadata_original, dict)
        ):
            _add_integrity_violation(
                violations,
                "run_preflight_inputs_invalid",
                f"preflight {label} declaration is invalid: {preflight_path}",
            )
            continue
        original = value.get("original")
        sealed = value.get("sealed")
        if not isinstance(original, dict) or not isinstance(sealed, dict):
            _add_integrity_violation(
                violations,
                "run_preflight_inputs_invalid",
                f"preflight {label} identities are invalid: {preflight_path}",
            )
            continue
        original_path_value = original.get("resolved_path")
        original_digest = original.get("sha256")
        original_size = original.get("size_bytes")
        original_fingerprint = original.get("fingerprint")
        sealed_path_value = sealed.get("resolved_path")
        sealed_digest = sealed.get("sha256")
        sealed_size = sealed.get("size_bytes")
        sealed_fingerprint = sealed.get("fingerprint")
        sealed_artifact = sealed.get("artifact")
        if (
            not isinstance(original_path_value, str)
            or not isinstance(original_digest, str)
            or type(original_size) is not int
            or not isinstance(original_fingerprint, dict)
            or not isinstance(sealed_path_value, str)
            or not isinstance(sealed_digest, str)
            or type(sealed_size) is not int
            or not isinstance(sealed_fingerprint, dict)
            or not isinstance(sealed_artifact, str)
        ):
            _add_integrity_violation(
                violations,
                "run_preflight_inputs_invalid",
                f"preflight {label} identity is incomplete: {preflight_path}",
            )
            continue
        if (
            metadata_original.get("declared_path") != declared_name
            or metadata_original.get("resolved_path") != original_path_value
            or metadata_original.get("sha256") != original_digest
            or metadata_original.get("size_bytes") != original_size
            or execution_inputs.get(label) != sealed
        ):
            _add_integrity_violation(
                violations,
                "run_preflight_metadata_mismatch",
                f"metadata does not bind the {label} preflight: {metadata_path}",
            )
        original_path = Path(original_path_value)
        sealed_path = Path(sealed_path_value)
        expected_sealed = (
            start_manifest.parent
            / "sealed"
            / "inputs"
            / category
            / preflight_path.parent.name
            / label
            / original_digest
            / original_path.name
        )
        try:
            if original_path.is_symlink():
                raise ProvenanceError(
                    f"original input became a symlink: {original_path}"
                )
            observed_original_digest, observed_original_fingerprint = _stable_file_hash(
                original_path
            )
            if sealed_path.is_symlink():
                raise ProvenanceError(f"sealed input became a symlink: {sealed_path}")
            observed_sealed_digest, observed_sealed_fingerprint = _stable_file_hash(
                sealed_path
            )
        except (OSError, ProvenanceError) as error:
            _add_integrity_violation(
                violations,
                "run_preflight_input_unavailable",
                str(error),
            )
            continue
        file_fingerprints[original_path] = observed_original_fingerprint
        file_fingerprints[sealed_path] = observed_sealed_fingerprint
        if (
            observed_original_digest != original_digest
            or observed_original_fingerprint != original_fingerprint
            or observed_sealed_digest != sealed_digest
            or observed_sealed_fingerprint != sealed_fingerprint
            or original_digest != sealed_digest
            or original_size != sealed_size
            or sealed_path.resolve() != expected_sealed.resolve()
            or (artifact_root / sealed_artifact).resolve() != sealed_path.resolve()
        ):
            _add_integrity_violation(
                violations,
                "run_preflight_input_drift",
                f"pre-run {label} binding changed: {preflight_path}",
            )
        evidence_inputs[label] = {
            "original_sha256": observed_original_digest,
            "sealed_sha256": observed_sealed_digest,
            "sealed_artifact": sealed_artifact,
        }
    return evidence


def _validate_run_evidence(
    *,
    start: dict[str, object],
    start_manifest: Path,
    start_digest: str,
    run_id: str,
    violations: list[dict[str, str]],
) -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    bool,
    dict[str, object],
]:
    """Bind each current-run CSV row to one complete, content-checked trio."""
    measurement = start.get("measurement")
    ny = start.get("ny")
    if not isinstance(measurement, dict) or not isinstance(ny, dict):
        _add_integrity_violation(
            violations,
            "run_evidence_start_identity_invalid",
            "start manifest lacks measurement or NY identity",
        )
        return (
            {
                "schema": "ny_measurement_run_evidence_v1",
                "status": "invalid",
            },
            {},
            False,
            {},
        )
    artifact_root_value = measurement.get("artifact_root")
    output_dir_value = measurement.get("output_dir")
    repo_root_value = ny.get("repo_root")
    categories_value = measurement.get("categories")
    if (
        not isinstance(artifact_root_value, str)
        or not isinstance(output_dir_value, str)
        or not isinstance(repo_root_value, str)
        or not isinstance(categories_value, list)
        or any(
            not isinstance(category, str) or SAFE_COMPONENT.fullmatch(category) is None
            for category in categories_value
        )
    ):
        _add_integrity_violation(
            violations,
            "run_evidence_start_identity_invalid",
            "start manifest has invalid evidence roots or categories",
        )
        return (
            {
                "schema": "ny_measurement_run_evidence_v1",
                "status": "invalid",
            },
            {},
            False,
            {},
        )

    artifact_root = Path(artifact_root_value)
    output_dir = Path(output_dir_value)
    repo_root = Path(repo_root_value)
    categories = list(categories_value)
    category_order = {category: index for index, category in enumerate(categories)}
    initial_violation_count = len(violations)
    discovered = _discover_run_artifacts(
        artifact_root=artifact_root,
        run_id=run_id,
        violations=violations,
    )
    metadata_paths: list[Path] = []
    result_count = 0
    solver_log_count = 0
    preflight_count = 0
    for instance_dir, artifacts in sorted(
        discovered.items(), key=lambda item: str(item[0])
    ):
        missing = {"metadata", "result", "solver_log", "preflight"} - set(artifacts)
        if missing:
            _add_integrity_violation(
                violations,
                "run_artifact_trio_incomplete",
                f"{instance_dir} is missing current-run artifacts: {', '.join(sorted(missing))}",
            )
        metadata_path = artifacts.get("metadata")
        if metadata_path is not None:
            metadata_paths.append(metadata_path)
        result_count += int("result" in artifacts)
        solver_log_count += int("solver_log" in artifacts)
        preflight_count += int("preflight" in artifacts)

    file_fingerprints: dict[Path, dict[str, int]] = {}
    references: dict[str, dict[str, object]] = {}
    records: list[dict[str, object]] = []
    seen_instances: set[tuple[str, int]] = set()
    expected_config_identity = _expected_metadata_config_identity(start)
    expected_execution_config_identity = (
        measurement.get("sealed_config_inputs")
        if isinstance(measurement, dict)
        else object()
    )
    try:
        expected_start_artifact = start_manifest.relative_to(artifact_root).as_posix()
        expected_cache_artifact = (
            start_manifest.with_name("input_hash_cache.json")
            .relative_to(artifact_root)
            .as_posix()
        )
    except ValueError:
        expected_start_artifact = ""
        expected_cache_artifact = ""
        _add_integrity_violation(
            violations,
            "run_evidence_start_path_invalid",
            "start manifest is outside its declared artifact root",
        )

    for metadata_path in sorted(metadata_paths):
        instance_dir = metadata_path.parent
        artifacts = discovered[instance_dir]
        result_path = artifacts.get("result")
        solver_log_path = artifacts.get("solver_log")
        preflight_path = artifacts.get("preflight")
        if result_path is None or solver_log_path is None or preflight_path is None:
            continue
        try:
            metadata_data, metadata_digest, metadata_fingerprint = _stable_file_bytes(
                metadata_path
            )
            result_data, result_digest, result_fingerprint = _stable_file_bytes(
                result_path
            )
            solver_log_data, solver_log_digest, solver_log_fingerprint = (
                _stable_file_bytes(solver_log_path)
            )
            metadata = json.loads(metadata_data)
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ProvenanceError,
        ) as error:
            _add_integrity_violation(
                violations,
                "run_artifact_unreadable",
                f"could not validate {metadata_path}: {error}",
            )
            continue
        file_fingerprints[metadata_path] = metadata_fingerprint
        file_fingerprints[result_path] = result_fingerprint
        file_fingerprints[solver_log_path] = solver_log_fingerprint
        if not isinstance(metadata, dict) or metadata.get("schema") != (
            "ny_measurement_result_v2"
        ):
            _add_integrity_violation(
                violations,
                "run_metadata_schema_invalid",
                f"unsupported result metadata: {metadata_path}",
            )
            continue
        category = metadata.get("category")
        instance_index = metadata.get("instance_index")
        verdict = metadata.get("solver_verdict")
        elapsed_seconds = metadata.get("elapsed_seconds")
        timeout_seconds = metadata.get("timeout_seconds")
        solver_exit_status = metadata.get("solver_exit_status")
        witness_present = metadata.get("witness_present")
        counterexample_validation = metadata.get("counterexample_validation")
        source_csv = metadata.get("source_csv")
        solver_log = metadata.get("solver_log")
        if (
            metadata.get("schema_version") != 2
            or not isinstance(category, str)
            or category not in category_order
            or type(instance_index) is not int
            or instance_index <= 0
            or not isinstance(verdict, str)
            or verdict not in STANDARD_SOLVER_VERDICTS
            or type(elapsed_seconds) is not int
            or elapsed_seconds < 0
            or type(timeout_seconds) is not int
            or timeout_seconds <= 0
            or type(solver_exit_status) is not int
            or not 0 <= solver_exit_status <= 255
            or (verdict in {"sat", "unsat"} and solver_exit_status != 0)
            or type(witness_present) is not bool
            or witness_present != (verdict == "sat")
            or not isinstance(counterexample_validation, dict)
            or counterexample_validation.get("status")
            != ("not_checked" if verdict == "sat" else "not_applicable")
            or not isinstance(source_csv, str)
            or not isinstance(solver_log, dict)
        ):
            _add_integrity_violation(
                violations,
                "run_metadata_fields_invalid",
                f"result metadata fields are invalid: {metadata_path}",
            )
            continue
        identity = (category, instance_index)
        if identity in seen_instances:
            _add_integrity_violation(
                violations,
                "run_metadata_instance_duplicate",
                f"duplicate current-run instance identity: {identity}",
            )
            continue
        seen_instances.add(identity)
        onnx_reference = _metadata_input_reference(
            metadata=metadata,
            label="onnx",
            metadata_path=metadata_path,
            references=references,
            violations=violations,
        )
        vnnlib_reference = _metadata_input_reference(
            metadata=metadata,
            label="vnnlib",
            metadata_path=metadata_path,
            references=references,
            violations=violations,
        )
        if onnx_reference is None or vnnlib_reference is None:
            continue
        onnx, _onnx_resolved, onnx_cache_key, _onnx_size = onnx_reference
        vnnlib, _vnnlib_resolved, vnnlib_cache_key, _vnnlib_size = vnnlib_reference
        instance_identity = json.dumps(
            [category, instance_index, onnx, vnnlib],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected_instance_dir = (
            f"{instance_index:05d}-{_sha256(instance_identity)[:16]}"
        )
        if instance_dir.name != expected_instance_dir or instance_dir.parent != (
            artifact_root / category
        ):
            _add_integrity_violation(
                violations,
                "run_artifact_identity_path_mismatch",
                f"metadata is outside its content-addressed instance path: {metadata_path}",
            )
        expected_result_artifact = result_path.relative_to(artifact_root).as_posix()
        expected_log_artifact = solver_log_path.relative_to(artifact_root).as_posix()
        if metadata.get("run_id") != run_id:
            _add_integrity_violation(
                violations,
                "run_metadata_run_id_mismatch",
                f"metadata run ID mismatch: {metadata_path}",
            )
        if (
            metadata.get("start_manifest") != expected_start_artifact
            or metadata.get("start_manifest_sha256") != start_digest
            or metadata.get("input_hash_cache") != expected_cache_artifact
        ):
            _add_integrity_violation(
                violations,
                "run_metadata_provenance_link_mismatch",
                f"metadata provenance link mismatch: {metadata_path}",
            )
        if metadata.get("config_inputs") != expected_config_identity:
            _add_integrity_violation(
                violations,
                "run_metadata_config_identity_mismatch",
                f"metadata config identity mismatch: {metadata_path}",
            )
        if (
            metadata.get("execution_config_inputs")
            != expected_execution_config_identity
        ):
            _add_integrity_violation(
                violations,
                "run_metadata_execution_config_identity_mismatch",
                f"metadata sealed config identity mismatch: {metadata_path}",
            )
        if (
            metadata.get("result_artifact") != expected_result_artifact
            or metadata.get("result_sha256") != result_digest
            or metadata.get("raw_result_sha256") != result_digest
        ):
            _add_integrity_violation(
                violations,
                "run_result_artifact_mismatch",
                f"raw result does not match metadata: {metadata_path}",
            )
        if (
            solver_log.get("artifact") != expected_log_artifact
            or solver_log.get("sha256") != solver_log_digest
            or solver_log.get("size_bytes") != len(solver_log_data)
        ):
            _add_integrity_violation(
                violations,
                "run_solver_log_artifact_mismatch",
                f"solver log does not match metadata: {metadata_path}",
            )
        result_lines = result_data.splitlines()
        first_line = (
            b"".join(result_lines[0].split()).decode("utf-8", "replace").lower()
            if result_lines
            else ""
        )
        verdict_matches = (
            first_line == verdict
            if verdict != "timeout"
            else first_line in {"", "timeout"}
        )
        if not verdict_matches:
            _add_integrity_violation(
                violations,
                "run_result_verdict_mismatch",
                f"raw result verdict does not match metadata: {metadata_path}",
            )
        if verdict == "sat" and not _structured_sat_assignment(result_lines[1:]):
            _add_integrity_violation(
                violations,
                "run_sat_witness_invalid",
                f"SAT result has no structured assignment: {metadata_path}",
            )
        preflight_evidence = _validate_row_preflight(
            preflight_path=preflight_path,
            metadata_path=metadata_path,
            metadata=metadata,
            artifact_root=artifact_root,
            start_manifest=start_manifest,
            start_digest=start_digest,
            category=category,
            instance_index=instance_index,
            onnx=onnx,
            vnnlib=vnnlib,
            file_fingerprints=file_fingerprints,
            violations=violations,
        )
        declared_source = Path(source_csv)
        resolved_source = (
            declared_source
            if declared_source.is_absolute()
            else repo_root / declared_source
        ).resolve()
        expected_csv = (output_dir / f"{category}.csv").resolve()
        if resolved_source != expected_csv:
            _add_integrity_violation(
                violations,
                "run_metadata_source_csv_mismatch",
                f"metadata source CSV does not match the declared output: {metadata_path}",
            )
        records.append(
            {
                "category": category,
                "instance_index": instance_index,
                "onnx": onnx,
                "vnnlib": vnnlib,
                "solver_verdict": verdict,
                "solver_exit_status": solver_exit_status,
                "timeout_seconds": timeout_seconds,
                "elapsed_seconds": elapsed_seconds,
                "input_hash_cache_keys": [onnx_cache_key, vnnlib_cache_key],
                "metadata": {
                    "artifact": metadata_path.relative_to(artifact_root).as_posix(),
                    "sha256": metadata_digest,
                    "size_bytes": metadata_fingerprint["size_bytes"],
                },
                "result": {
                    "artifact": expected_result_artifact,
                    "sha256": result_digest,
                    "size_bytes": result_fingerprint["size_bytes"],
                },
                "solver_log": {
                    "artifact": expected_log_artifact,
                    "sha256": solver_log_digest,
                    "size_bytes": solver_log_fingerprint["size_bytes"],
                },
                "preflight": preflight_evidence,
            }
        )

    records.sort(
        key=lambda record: (
            category_order.get(str(record["category"]), len(category_order)),
            int(record["instance_index"]),
            str(record["metadata"]),
        )
    )
    expected_rows: dict[str, list[list[str]]] = {
        category: [] for category in categories
    }
    for record in records:
        category = str(record["category"])
        expected_rows[category].append(
            [
                category,
                str(record["onnx"]),
                str(record["vnnlib"]),
                "0",
                str(record["solver_verdict"]),
                str(record["elapsed_seconds"]),
                run_id,
            ]
        )

    csv_evidence: list[dict[str, object]] = []
    observed_csv_paths: set[Path] = set()
    actual_rows: dict[str, list[list[str]]] = {category: [] for category in categories}
    candidate_csvs: set[Path] = {
        output_dir / f"{category}.csv" for category in categories
    }
    if output_dir.is_dir():
        candidate_csvs.update(output_dir.glob("*.csv"))
    for csv_path in sorted(candidate_csvs):
        if not csv_path.exists():
            continue
        observed_csv_paths.add(csv_path)
        try:
            data, digest, fingerprint = _stable_file_bytes(csv_path)
            rows = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
        except (OSError, UnicodeDecodeError, csv.Error, ProvenanceError) as error:
            _add_integrity_violation(
                violations,
                "run_csv_unreadable",
                f"could not validate {csv_path}: {error}",
            )
            continue
        current_rows: list[list[str]] = []
        for row in rows:
            if len(row) > 6 and row[6] == run_id:
                current_rows.append(row)
                if len(row) != 7:
                    _add_integrity_violation(
                        violations,
                        "run_csv_row_invalid",
                        f"current-run row does not have seven columns: {csv_path}",
                    )
            elif run_id in row:
                _add_integrity_violation(
                    violations,
                    "run_csv_row_invalid",
                    f"current-run ID appears outside the provenance column: {csv_path}",
                )
        category = csv_path.stem
        if current_rows and category not in actual_rows:
            _add_integrity_violation(
                violations,
                "run_csv_unexpected_category",
                f"current-run rows appear in unexpected CSV: {csv_path}",
            )
        elif category in actual_rows:
            actual_rows[category] = current_rows
        csv_evidence.append(
            {
                "path": str(csv_path.resolve()),
                "sha256": digest,
                "size_bytes": fingerprint["size_bytes"],
                "current_run_row_count": len(current_rows),
                "current_run_rows_sha256": _identity_sha256(current_rows),
            }
        )
        file_fingerprints[csv_path] = fingerprint
    for category in categories:
        if actual_rows[category] != expected_rows[category]:
            _add_integrity_violation(
                violations,
                "run_csv_artifact_bijection_mismatch",
                f"current-run CSV rows and artifacts differ for category {category}",
            )

    csv_row_count = sum(len(rows) for rows in actual_rows.values())
    evidence_required = bool(
        metadata_paths
        or result_count
        or solver_log_count
        or preflight_count
        or csv_row_count
    )
    check = {
        "schema": "ny_measurement_run_evidence_v1",
        "status": (
            "valid" if len(violations) == initial_violation_count else "invalid"
        ),
        "produced_rows": csv_row_count > 0,
        "metadata_count": len(metadata_paths),
        "result_count": result_count,
        "solver_log_count": solver_log_count,
        "preflight_count": preflight_count,
        "validated_record_count": len(records),
        "csv_row_count": csv_row_count,
        "records_sha256": _identity_sha256(records),
        "csv_evidence_sha256": _identity_sha256(csv_evidence),
        "records": records,
        "csv_evidence": csv_evidence,
    }
    snapshot = {
        "artifact_root": artifact_root,
        "output_dir": output_dir,
        "categories": categories,
        "artifact_namespace": _run_artifact_namespace(discovered, artifact_root),
        "file_fingerprints": file_fingerprints,
        "csv_paths": observed_csv_paths,
    }
    return check, references, evidence_required, snapshot


def _validate_input_hash_cache(
    *,
    cache_path: Path,
    run_id: str,
    start_digest: str,
    referenced_entries: dict[str, dict[str, object]],
    evidence_required: bool,
    violations: list[dict[str, str]],
) -> tuple[dict[str, object], dict[str, object]]:
    violation_count = len(violations)
    check: dict[str, object] = {"status": "invalid"}
    if cache_path.is_symlink():
        _add_integrity_violation(
            violations,
            "input_hash_cache_invalid_file",
            f"input hash cache must not be a symlink: {cache_path}",
        )
        return {"present": True, "artifact": cache_path.name}, check
    if not cache_path.exists():
        if evidence_required or referenced_entries:
            _add_integrity_violation(
                violations,
                "input_hash_cache_missing_for_run_artifacts",
                "current-run rows or artifacts exist without their input hash cache",
            )
        return (
            {
                "present": False,
                "artifact": cache_path.name,
                "entry_count": 0,
            },
            {
                "status": (
                    "invalid" if len(violations) != violation_count else "absent"
                ),
                "referenced_entry_count": len(referenced_entries),
            },
        )
    evidence: dict[str, object] = {
        "present": True,
        "artifact": cache_path.name,
    }
    if not cache_path.is_file():
        _add_integrity_violation(
            violations,
            "input_hash_cache_invalid_file",
            f"input hash cache must be a regular non-symlink file: {cache_path}",
        )
        return evidence, check
    try:
        cache_data, cache_digest, cache_fingerprint = _stable_file_bytes(cache_path)
    except (OSError, ProvenanceError) as error:
        _add_integrity_violation(
            violations,
            "input_hash_cache_unreadable",
            str(error),
        )
        return evidence, check
    evidence.update(
        {
            "sha256": cache_digest,
            "size_bytes": cache_fingerprint["size_bytes"],
        }
    )
    try:
        cache = json.loads(cache_data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _add_integrity_violation(
            violations,
            "input_hash_cache_invalid_json",
            str(error),
        )
        return evidence, check
    if not isinstance(cache, dict):
        _add_integrity_violation(
            violations,
            "input_hash_cache_invalid_manifest",
            "input hash cache root must be an object",
        )
        return evidence, check
    entries = cache.get("entries")
    evidence["entry_count"] = len(entries) if isinstance(entries, dict) else None
    if cache.get("schema") != "ny_measurement_input_hash_cache_v1":
        _add_integrity_violation(
            violations,
            "input_hash_cache_schema_mismatch",
            "input hash cache schema does not match ny_measurement_input_hash_cache_v1",
        )
    if cache.get("run_id") != run_id:
        _add_integrity_violation(
            violations,
            "input_hash_cache_run_id_mismatch",
            "input hash cache run ID does not match the start manifest",
        )
    if cache.get("start_manifest_sha256") != start_digest:
        _add_integrity_violation(
            violations,
            "input_hash_cache_start_manifest_mismatch",
            "input hash cache start-manifest digest does not match",
        )
    fingerprint_keys = {"device", "inode", "size_bytes", "mtime_ns", "ctime_ns"}
    entries_well_formed = isinstance(entries, dict)
    if isinstance(entries, dict):
        for key, entry in sorted(entries.items()):
            if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
                entries_well_formed = False
                break
            if not isinstance(entry, dict):
                entries_well_formed = False
                break
            path_value = entry.get("path")
            fingerprint = entry.get("fingerprint")
            digest = entry.get("sha256")
            if (
                not isinstance(path_value, str)
                or not isinstance(fingerprint, dict)
                or set(fingerprint) != fingerprint_keys
                or any(type(value) is not int for value in fingerprint.values())
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or _input_cache_key(path_value, fingerprint) != key
            ):
                entries_well_formed = False
                break
    if not entries_well_formed:
        _add_integrity_violation(
            violations,
            "input_hash_cache_entries_invalid",
            "input hash cache entries are malformed or not content-addressed",
        )
    rehashed_entries: list[dict[str, object]] = []
    if entries_well_formed and isinstance(entries, dict):
        for key, entry_object in sorted(entries.items()):
            assert isinstance(entry_object, dict)
            path_value = entry_object["path"]
            fingerprint = entry_object["fingerprint"]
            digest = entry_object["sha256"]
            assert isinstance(path_value, str)
            assert isinstance(fingerprint, dict)
            assert isinstance(digest, str)
            path = Path(path_value)
            try:
                resolved = path.resolve(strict=True)
                if (
                    not path.is_absolute()
                    or resolved != path
                    or path.is_symlink()
                    or not path.is_file()
                ):
                    raise ProvenanceError(
                        f"cached input path is not canonical: {path_value}"
                    )
                observed_digest, observed_fingerprint = _stable_file_hash(path)
            except (OSError, ProvenanceError) as error:
                _add_integrity_violation(
                    violations,
                    "input_hash_cache_entry_unavailable",
                    f"could not rehash cache entry {key}: {error}",
                )
                continue
            if observed_digest != digest:
                _add_integrity_violation(
                    violations,
                    "input_hash_cache_entry_sha256_mismatch",
                    f"cached input digest changed for {path_value}",
                )
            if observed_fingerprint != fingerprint:
                _add_integrity_violation(
                    violations,
                    "input_hash_cache_entry_fingerprint_mismatch",
                    f"cached input stat identity changed for {path_value}",
                )
            reference = referenced_entries.get(key)
            if reference is None:
                _add_integrity_violation(
                    violations,
                    "input_hash_cache_entry_unreferenced",
                    f"cache entry {key} is not referenced by a current-run artifact",
                )
            elif (
                reference.get("path") != path_value
                or reference.get("sha256") != digest
                or reference.get("size_bytes") != fingerprint.get("size_bytes")
            ):
                _add_integrity_violation(
                    violations,
                    "input_hash_cache_metadata_mismatch",
                    f"cache entry {key} differs from row metadata",
                )
            rehashed_entries.append(
                {
                    "key": key,
                    "path": path_value,
                    "sha256": observed_digest,
                    "size_bytes": observed_fingerprint["size_bytes"],
                }
            )
        missing_references = set(referenced_entries) - set(entries)
        if missing_references:
            _add_integrity_violation(
                violations,
                "input_hash_cache_reference_missing",
                "row metadata references cache keys that are absent: "
                + ", ".join(sorted(missing_references)),
            )
    if evidence_required and isinstance(entries, dict) and not entries:
        _add_integrity_violation(
            violations,
            "input_hash_cache_empty_for_run_artifacts",
            "current-run rows or artifacts exist with an empty input hash cache",
        )
    check["status"] = "valid" if len(violations) == violation_count else "invalid"
    check["sha256"] = evidence.get("sha256")
    check["entry_count"] = evidence.get("entry_count")
    check["referenced_entry_count"] = len(referenced_entries)
    check["rehashed_entry_count"] = len(rehashed_entries)
    check["entries_sha256"] = _identity_sha256(rehashed_entries)
    return evidence, check


def create_completion(*, start_manifest: Path, exit_status: int) -> Path:
    """Create the immutable completion sibling for an existing start record."""
    if not 0 <= exit_status <= 255:
        raise ValueError(f"exit status is outside the shell range: {exit_status}")
    if start_manifest.is_symlink():
        raise ProvenanceError(f"start manifest must not be a symlink: {start_manifest}")
    start_manifest = start_manifest.resolve()
    start_data = start_manifest.read_bytes()
    start = json.loads(start_data)
    if start.get("schema") != "ny_measurement_start_v1":
        raise ProvenanceError(f"unsupported start manifest schema: {start_manifest}")
    run_id = start.get("run_id")
    if not isinstance(run_id, str) or SAFE_COMPONENT.fullmatch(run_id) is None:
        raise ProvenanceError(f"invalid run ID in start manifest: {start_manifest}")
    completion_path = start_manifest.with_name("completion.json")
    start_digest = _sha256(start_data)
    violations: list[dict[str, str]] = []
    solver = start.get("solver_binary")
    dependencies = start.get("dependencies")
    ay = dependencies.get("ay") if isinstance(dependencies, dict) else None
    checks: dict[str, object] = {
        "solver_binary": _validate_solver_binary(start, violations),
        "sealed_solver_binary": _validate_sealed_file(
            expected=(
                solver.get("sealed_execution") if isinstance(solver, dict) else object()
            ),
            name="sealed_solver_binary",
            executable=True,
            violations=violations,
        ),
        "ay_executable": _validate_ay_executable(start, violations),
        "sealed_ay_executable": _validate_sealed_file(
            expected=(
                ay.get("sealed_executable") if isinstance(ay, dict) else object()
            ),
            name="sealed_ay_executable",
            executable=True,
            violations=violations,
        ),
        "config_inputs": _validate_config_inputs(start, violations),
        "sealed_config_inputs": _validate_sealed_config_inputs(start, violations),
    }

    ny = start.get("ny")
    ny_repo_root = ny.get("repo_root") if isinstance(ny, dict) else None
    if isinstance(ny_repo_root, str):
        checks["ny_worktree"] = _validate_recaptured_identity(
            name="ny_worktree",
            expected=ny,
            capture=lambda: _capture_worktree(Path(ny_repo_root)),
            violations=violations,
        )
    else:
        _add_integrity_violation(
            violations,
            "ny_worktree_start_identity_invalid",
            "start manifest NY repository root is missing or invalid",
        )
        checks["ny_worktree"] = {"status": "invalid"}

    benchmark = start.get("benchmark")
    benchmark_root = (
        benchmark.get("benchmark_root") if isinstance(benchmark, dict) else None
    )
    if isinstance(benchmark_root, str):
        checks["benchmark"] = _validate_recaptured_identity(
            name="benchmark",
            expected=benchmark,
            capture=lambda: _capture_benchmark(Path(benchmark_root)),
            violations=violations,
        )
    else:
        _add_integrity_violation(
            violations,
            "benchmark_start_identity_invalid",
            "start manifest benchmark root is missing or invalid",
        )
        checks["benchmark"] = {"status": "invalid"}

    run_evidence_violation_count = len(violations)
    (
        run_evidence,
        referenced_cache_entries,
        evidence_required,
        run_evidence_snapshot,
    ) = _validate_run_evidence(
        start=start,
        start_manifest=start_manifest,
        start_digest=start_digest,
        run_id=run_id,
        violations=violations,
    )
    cache_path = start_manifest.with_name("input_hash_cache.json")
    cache_lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with cache_lock_path.open("a+b") as cache_lock:
        fcntl.flock(cache_lock.fileno(), fcntl.LOCK_EX)
        cache_evidence, checks["input_hash_cache"] = _validate_input_hash_cache(
            cache_path=cache_path,
            run_id=run_id,
            start_digest=start_digest,
            referenced_entries=referenced_cache_entries,
            evidence_required=evidence_required,
            violations=violations,
        )
    _revalidate_run_evidence_snapshot(
        snapshot=run_evidence_snapshot,
        run_id=run_id,
        violations=violations,
    )
    run_evidence["input_hash_cache_entry_count"] = checks["input_hash_cache"].get(
        "entry_count"
    )
    run_evidence["referenced_input_hash_cache_entry_count"] = len(
        referenced_cache_entries
    )
    run_evidence["status"] = (
        "valid" if len(violations) == run_evidence_violation_count else "invalid"
    )
    checks["run_evidence"] = run_evidence
    integrity_status = "valid" if not violations else "invalid"
    payload = {
        "schema": "ny_measurement_completion_v1",
        "run_id": run_id,
        "ended_at_utc": _utc_now(),
        "exit_status": exit_status,
        "completed_successfully": exit_status == 0 and integrity_status == "valid",
        "start_manifest": start_manifest.name,
        "start_manifest_sha256": start_digest,
        "input_hash_cache": cache_evidence,
        "integrity": {
            "schema": "ny_measurement_completion_integrity_v1",
            "status": integrity_status,
            "violations": violations,
            "checks": checks,
        },
        "host_state": _capture_host_state(),
    }
    _write_immutable(completion_path, _json_bytes(payload))
    return completion_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="capture immutable start evidence")
    start.add_argument("--repo-root", type=Path, required=True)
    start.add_argument("--binary", type=Path, required=True)
    start.add_argument("--benchmark-root", type=Path, required=True)
    start.add_argument("--artifact-root", type=Path, required=True)
    start.add_argument("--run-id", required=True)
    start.add_argument("--output-dir", type=Path, required=True)
    start.add_argument("--scratch-dir", type=Path, required=True)
    start.add_argument("--result-file", type=Path, required=True)
    start.add_argument("--solver-log-file", type=Path, required=True)
    start.add_argument("--categories", required=True)
    start.add_argument("--timeout-cap-seconds", type=int, required=True)
    start.add_argument("--watchdog-grace-seconds", type=int, required=True)
    start.add_argument("--max-rows-per-category", type=int, required=True)
    start.add_argument("--instance-index", type=int, required=True)
    start.add_argument("--vnnlib-version", required=True)
    start.add_argument("--configs-dir", type=Path)
    start.add_argument("--sweep-script", type=Path, required=True)
    complete = commands.add_parser("complete", help="record immutable completion")
    complete.add_argument("--start-manifest", type=Path, required=True)
    complete.add_argument("--exit-status", type=int, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    completion_integrity_valid = True
    try:
        if args.command == "start":
            path = capture_start_manifest(
                repo_root=args.repo_root,
                binary=args.binary,
                benchmark_root=args.benchmark_root,
                artifact_root=args.artifact_root,
                run_id=args.run_id,
                output_dir=args.output_dir,
                scratch_dir=args.scratch_dir,
                result_file=args.result_file,
                solver_log_file=args.solver_log_file,
                categories_raw=args.categories,
                timeout_cap_seconds=args.timeout_cap_seconds,
                watchdog_grace_seconds=args.watchdog_grace_seconds,
                max_rows_per_category=args.max_rows_per_category,
                instance_index=args.instance_index,
                vnnlib_version=args.vnnlib_version,
                sweep_script=args.sweep_script,
                configs_dir=args.configs_dir,
            )
        else:
            path = create_completion(
                start_manifest=args.start_manifest,
                exit_status=args.exit_status,
            )
            completion = json.loads(path.read_bytes())
            integrity = completion.get("integrity")
            completion_integrity_valid = (
                isinstance(integrity, dict) and integrity.get("status") == "valid"
            )
    except (OSError, ValueError, ProvenanceError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(path)
    if args.command == "complete" and not completion_integrity_valid:
        print(
            "ERROR: measurement completion integrity validation failed", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
