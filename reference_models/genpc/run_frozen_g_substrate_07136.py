"""Close the frozen-G GenPC substrate gate without rerunning generation.

This batch deliberately starts from the first completed method-valid baseline's
raw registration artifacts.  It runs one uninstrumented control registration,
one read-only boundary-capture registration from the identical frozen inputs,
and a fusion-only replay from the captured O_aligned/G_aligned pair.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch

from run_instrumented_equivalence_07136 import (
    BASELINE_OUTPUT,
    BASELINE_ROOT,
    FLAG,
    INSTANTMESH,
    ROOT,
    STAGE2_FILES,
    UPSTREAM,
    configure_cuda_runtime,
    cuda_memory,
    load_cfg,
    set_seed,
    sha256,
)


CANONICAL_G = BASELINE_OUTPUT / f"{FLAG}_instantmesh.glb"
CANONICAL_O = BASELINE_OUTPUT / "color_point.ply"
GROUND_TRUTH = UPSTREAM / "data" / "GT" / f"{FLAG}.ply"
SUBSTRATE_ROOT = ROOT / "runs" / "frozen_g_substrate_07136_v6"
FROZEN_INPUTS = SUBSTRATE_ROOT / "frozen_inputs"
CONTROL_ROOT = ROOT / "runs" / "frozen_g_control_07136_v6"
INSTRUMENTED_ROOT = ROOT / "runs" / "frozen_g_instrumented_07136_v6"
FUSION_ROOT = ROOT / "runs" / "frozen_g_fusion_replay_07136_v6"
CONTROL_OUTPUT = CONTROL_ROOT / FLAG
INSTRUMENTED_OUTPUT = INSTRUMENTED_ROOT / FLAG
FUSION_OUTPUT = FUSION_ROOT / FLAG
RESULT_PATH = SUBSTRATE_ROOT / "frozen_g_substrate_result.json"


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_array_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def pcd_snapshot(pcd: Any) -> dict[str, Any]:
    points = np.asarray(pcd.points).copy()
    colors = np.asarray(pcd.colors).copy() if pcd.has_colors() else None
    return {
        "points": points,
        "colors": colors,
        "point_count": int(len(points)),
        "point_order_hash": canonical_array_hash(points),
        "coordinate_hash": canonical_array_hash(points),
        "color_hash": canonical_array_hash(colors) if colors is not None else None,
    }


def pcd_snapshot_meta(snapshot: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    points = snapshot["points"]
    colors = snapshot["colors"]
    result: dict[str, Any] = {
        "point_count": snapshot["point_count"],
        "point_order_hash": snapshot["point_order_hash"],
        "coordinate_hash": snapshot["coordinate_hash"],
        "color_hash": snapshot["color_hash"],
        "xyz_min": points.min(axis=0).tolist() if len(points) else [],
        "xyz_max": points.max(axis=0).tolist() if len(points) else [],
        "has_colors": colors is not None,
    }
    if colors is not None and len(colors):
        result["color_min"] = colors.min(axis=0).tolist()
        result["color_max"] = colors.max(axis=0).tolist()
    else:
        result["color_min"] = []
        result["color_max"] = []
    if path is not None:
        result.update({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return result


def compare_snapshots(
    before: dict[str, Any], after: dict[str, Any], include_colors: bool = True
) -> dict[str, Any]:
    points_equal = bool(np.array_equal(before["points"], after["points"]))
    if before["colors"] is None or after["colors"] is None:
        colors_equal = before["colors"] is None and after["colors"] is None
    else:
        colors_equal = bool(np.array_equal(before["colors"], after["colors"]))
    return {
        "point_count_unchanged": before["point_count"] == after["point_count"],
        "point_order_unchanged": before["point_order_hash"] == after["point_order_hash"],
        "coordinates_unchanged": points_equal,
        "colors_checked": include_colors,
        "colors_unchanged": colors_equal if include_colors else None,
        "before_fingerprint": {
            "point_order_hash": before["point_order_hash"],
            "coordinate_hash": before["coordinate_hash"],
            "color_hash": before["color_hash"],
        },
        "after_fingerprint": {
            "point_order_hash": after["point_order_hash"],
            "coordinate_hash": after["coordinate_hash"],
            "color_hash": after["color_hash"],
        },
        "identity_pass": bool(points_equal and (colors_equal if include_colors else True)),
    }


def compare_exported_to_snapshot(
    snapshot: dict[str, Any], path: Path, include_colors: bool = True
) -> dict[str, Any]:
    import open3d as o3d

    exported = pcd_snapshot(o3d.io.read_point_cloud(str(path)))
    comparison = compare_snapshots(snapshot, exported, include_colors=include_colors)
    comparison["exported"] = pcd_snapshot_meta(exported, path)
    comparison["in_memory"] = pcd_snapshot_meta(snapshot)
    comparison["numeric_identity_pass"] = comparison["identity_pass"]
    return comparison


def save_exact_pcd_sidecar(snapshot: dict[str, Any], path: Path) -> dict[str, Any]:
    colors = snapshot["colors"]
    if colors is None:
        colors = np.empty((0, 3), dtype=np.float64)
    np.savez_compressed(path, points=snapshot["points"], colors=colors)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "points_sha256": canonical_array_hash(snapshot["points"]),
        "colors_sha256": canonical_array_hash(colors),
        "point_count": snapshot["point_count"],
        "colors_present": snapshot["colors"] is not None,
    }


def load_exact_pcd_sidecar(path: Path):
    import open3d as o3d

    with np.load(path, allow_pickle=False) as data:
        points = np.asarray(data["points"], dtype=np.float64)
        colors = np.asarray(data["colors"], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if len(colors):
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def mesh_geometry_meta(path: Path) -> dict[str, Any]:
    import trimesh

    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [(str(name), loaded.geometry[name]) for name in sorted(loaded.geometry)]
    else:
        meshes = [(path.name, loaded)]
    digest = hashlib.sha256()
    total_vertices = 0
    total_faces = 0
    parts = []
    for name, mesh in meshes:
        vertices = np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float64))
        faces = np.ascontiguousarray(np.asarray(mesh.faces, dtype=np.int64))
        digest.update(name.encode("utf-8"))
        digest.update(repr(vertices.shape).encode("ascii"))
        digest.update(vertices.tobytes(order="C"))
        digest.update(repr(faces.shape).encode("ascii"))
        digest.update(faces.tobytes(order="C"))
        total_vertices += len(vertices)
        total_faces += len(faces)
        parts.append({"name": name, "vertices": int(len(vertices)), "faces": int(len(faces))})
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "geometry_sha256": digest.hexdigest(),
        "mesh_count": len(meshes),
        "vertex_count": total_vertices,
        "face_count": total_faces,
        "parts": parts,
    }


def raw_pcd_meta(path: Path) -> dict[str, Any]:
    import open3d as o3d

    snapshot = pcd_snapshot(o3d.io.read_point_cloud(str(path)))
    result = pcd_snapshot_meta(snapshot, path)
    result["geometry_sha256"] = hash_bytes(
        json.dumps(
            {
                "point_order_hash": snapshot["point_order_hash"],
                "coordinate_hash": snapshot["coordinate_hash"],
                "color_hash": snapshot["color_hash"],
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    return result


def rng_state_hash(value: Any) -> str:
    return hash_bytes(pickle.dumps(value, protocol=4))


def rng_snapshot() -> dict[str, Any]:
    cuda_states = []
    if torch.cuda.is_available():
        cuda_states = [hash_bytes(state.cpu().numpy().tobytes()) for state in torch.cuda.get_rng_state_all()]
    return {
        "declared_seed": 42,
        "python": {"state_sha256": rng_state_hash(random.getstate())},
        "numpy": {"state_sha256": rng_state_hash(np.random.get_state())},
        "torch_cpu": {
            "initial_seed": int(torch.initial_seed()),
            "state_sha256": hash_bytes(torch.get_rng_state().numpy().tobytes()),
        },
        "torch_cuda": {
            "initial_seed": int(torch.cuda.initial_seed()) if torch.cuda.is_available() else None,
            "state_sha256": cuda_states,
        },
    }


def deterministic_settings() -> dict[str, Any]:
    return {
        "torch_deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def copy_frozen_inputs() -> dict[str, Any]:
    FROZEN_INPUTS.mkdir(parents=True, exist_ok=False)
    copied = {}
    for label, source, target_name in [
        ("O_raw_registration", CANONICAL_O, "O_raw_registration.ply"),
        ("G_raw_canonical", CANONICAL_G, f"G_raw_canonical_{FLAG}.glb"),
        ("GT", GROUND_TRUTH, f"GT_{FLAG}.ply"),
    ]:
        target = FROZEN_INPUTS / target_name
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise RuntimeError(f"frozen copy changed bytes: {label}")
        copied[label] = {"source": str(source), "path": str(target), "sha256": sha256(target)}
    return copied


def stage_input_files(run_root: Path) -> dict[str, Any]:
    output = run_root / FLAG
    output.mkdir(parents=True, exist_ok=False)
    copied = {}
    for source, target_name in [
        (FROZEN_INPUTS / "O_raw_registration.ply", "color_point.ply"),
        (FROZEN_INPUTS / f"G_raw_canonical_{FLAG}.glb", f"{FLAG}_instantmesh.glb"),
    ]:
        target = output / target_name
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise RuntimeError(f"run input copy changed bytes: {target_name}")
        copied[target_name] = {"source": str(source), "path": str(target), "sha256": sha256(target)}
    return copied


def run_control_registration() -> dict[str, Any]:
    import reg_xyz

    cfg = load_cfg(CONTROL_ROOT)
    set_seed(42)
    before_rng = rng_snapshot()
    os.chdir(CONTROL_ROOT)
    torch.cuda.reset_peak_memory_stats()
    reg_xyz.reg(cfg, FLAG, cd_inv_weight=0.5, diff_init=True, reg_fine_xyz=True)
    after_rng = rng_snapshot()
    fused = CONTROL_OUTPUT / f"{FLAG}_fused.ply"
    transform = CONTROL_ROOT / "final_transform.npy"
    return {
        "status": "completed",
        "boundary_export": False,
        "rng_before_registration": before_rng,
        "rng_after_registration": after_rng,
        "memory": cuda_memory(),
        "fused": raw_pcd_meta(fused),
        "final_transform": {
            "path": str(transform),
            "sha256": sha256(transform),
            "matrix": np.load(transform).tolist(),
        },
    }


class ReadOnlyBoundaryHook:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.called = False
        self.capture: dict[str, Any] = {}

    @staticmethod
    def matrix(value: Any) -> list[list[float]] | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        return array.tolist() if array.shape == (4, 4) else None

    def __call__(self, original, source_pcd, target_pcd, *args, **kwargs):
        if self.called:
            raise RuntimeError("boundary hook called more than once")
        self.called = True
        import open3d as o3d

        self.output_dir.mkdir(parents=True, exist_ok=False)
        fusion_numpy_rng_state = np.random.get_state()
        fusion_rng_path = self.output_dir / "numpy_rng_before_fusion.pkl"
        with fusion_rng_path.open("wb") as handle:
            pickle.dump(fusion_numpy_rng_state, handle, protocol=4)
        source_before = pcd_snapshot(source_pcd)
        target_before = pcd_snapshot(target_pcd)
        source_path = self.output_dir / "O_aligned.ply"
        target_path = self.output_dir / "G_aligned.ply"
        source_sidecar = self.output_dir / "O_aligned_exact.npz"
        target_sidecar = self.output_dir / "G_aligned_exact.npz"
        if not o3d.io.write_point_cloud(str(source_path), source_pcd):
            raise RuntimeError("failed to export O_aligned")
        if not o3d.io.write_point_cloud(str(target_path), target_pcd):
            raise RuntimeError("failed to export G_aligned")
        source_sidecar_meta = save_exact_pcd_sidecar(source_before, source_sidecar)
        target_sidecar_meta = save_exact_pcd_sidecar(target_before, target_sidecar)
        source_after = pcd_snapshot(source_pcd)
        target_after = pcd_snapshot(target_pcd)
        source_identity = compare_snapshots(source_before, source_after)
        target_identity = compare_snapshots(target_before, target_after)
        source_export_identity = compare_exported_to_snapshot(source_before, source_path, include_colors=False)
        target_export_identity = compare_exported_to_snapshot(target_before, target_path, include_colors=False)
        source_sidecar_identity = {
            "points_sha256": source_sidecar_meta["points_sha256"] == source_before["coordinate_hash"],
            "colors_sha256": source_sidecar_meta["colors_sha256"] == source_before["color_hash"],
            "exact_numeric_identity_pass": bool(
                source_sidecar_meta["points_sha256"] == source_before["coordinate_hash"]
                and source_sidecar_meta["colors_sha256"] == source_before["color_hash"]
            ),
            "artifact": source_sidecar_meta,
        }
        target_sidecar_identity = {
            "points_sha256": target_sidecar_meta["points_sha256"] == target_before["coordinate_hash"],
            "colors_sha256": target_sidecar_meta["colors_sha256"] == target_before["color_hash"],
            "exact_numeric_identity_pass": bool(
                target_sidecar_meta["points_sha256"] == target_before["coordinate_hash"]
                and target_sidecar_meta["colors_sha256"] == target_before["color_hash"]
            ),
            "artifact": target_sidecar_meta,
        }

        frame = sys._getframe()
        reg_locals: dict[str, Any] = {}
        caller = frame.f_back
        for _ in range(5):
            if caller is None:
                break
            if caller.f_code.co_name == "reg":
                reg_locals = caller.f_locals
                break
            caller = caller.f_back
        matrices = {
            "diff_transform": self.matrix(reg_locals.get("diff_transform")),
            "inverse_diff_transform_at_boundary": self.matrix(reg_locals.get("inv")),
            "coarse_transformation": self.matrix(reg_locals.get("coarse_transformation")),
            "best_scales_transformation": self.matrix(reg_locals.get("best_scales_transformation")),
            "best_transformation_xyz": self.matrix(reg_locals.get("best_transformation_xyz")),
        }
        for name in ["diff_transform", "coarse_transformation", "best_scales_transformation", "best_transformation_xyz"]:
            if matrices[name] is not None:
                matrices[f"inverse_{name}"] = np.linalg.inv(np.asarray(matrices[name], dtype=np.float64)).tolist()
        lineage = {
            "boundary": "immediately before official reg_xyz.remove_close_points(source_pcd, target_pcd, distance_threshold=0.0001)",
            "hook_behavior": "copy/serialize and fingerprint only; original remove_close_points receives unchanged objects",
            "matrices": matrices,
            "distance_threshold": float(kwargs.get("distance_threshold", args[0] if args else 0.0001)),
            "fusion_rng": {
                "numpy_state_sha256": rng_state_hash(fusion_numpy_rng_state),
                "path": str(fusion_rng_path),
                "sha256": sha256(fusion_rng_path),
            },
            "source_semantics": [
                "O_raw_registration",
                "remove_noise_from_point_cloud",
                "diff_transform",
                "coarse_transformation",
                "inverse_coarse_transformation",
                "inverse_diff_transform",
                "O_aligned",
            ],
            "target_semantics": [
                "G_raw_canonical sampled by official glb2point",
                "normalize_numpy(range=0.5)",
                "InstantMesh x-axis +90 degree rotation",
                "InstantMesh y-axis +90 degree rotation",
                "inverse_best_scales_transformation",
                "inverse_best_transformation_xyz",
                "inverse_coarse_transformation",
                "inverse_diff_transform",
                "G_aligned",
            ],
            "source_before": pcd_snapshot_meta(source_before),
            "source_after": pcd_snapshot_meta(source_after),
            "target_before": pcd_snapshot_meta(target_before),
            "target_after": pcd_snapshot_meta(target_after),
            "source_identity": source_identity,
            "target_identity": target_identity,
            "source_export_identity": source_export_identity,
            "target_export_identity": target_export_identity,
            "source_exact_sidecar_identity": source_sidecar_identity,
            "target_exact_sidecar_identity": target_sidecar_identity,
            "identity_pass": bool(
                source_identity["identity_pass"]
                and target_identity["identity_pass"]
                and source_export_identity["numeric_identity_pass"]
                and target_export_identity["numeric_identity_pass"]
                and source_sidecar_identity["exact_numeric_identity_pass"]
                and target_sidecar_identity["exact_numeric_identity_pass"]
            ),
        }
        (self.output_dir / "transform_lineage.json").write_text(
            json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.capture = lineage
        if not lineage["identity_pass"]:
            raise RuntimeError("read-only boundary identity assertion failed")
        return original(source_pcd, target_pcd, *args, **kwargs)


def run_instrumented_registration() -> dict[str, Any]:
    import reg_xyz

    cfg = load_cfg(INSTRUMENTED_ROOT)
    set_seed(42)
    before_rng = rng_snapshot()
    hook = ReadOnlyBoundaryHook(INSTRUMENTED_ROOT / "pre_fusion")
    original_remove_close_points = reg_xyz.remove_close_points

    def hooked_remove_close_points(source_pcd, target_pcd, *args, **kwargs):
        return hook(original_remove_close_points, source_pcd, target_pcd, *args, **kwargs)

    reg_xyz.remove_close_points = hooked_remove_close_points
    try:
        os.chdir(INSTRUMENTED_ROOT)
        torch.cuda.reset_peak_memory_stats()
        reg_xyz.reg(cfg, FLAG, cd_inv_weight=0.5, diff_init=True, reg_fine_xyz=True)
    finally:
        reg_xyz.remove_close_points = original_remove_close_points
    after_rng = rng_snapshot()
    fused = INSTRUMENTED_OUTPUT / f"{FLAG}_fused.ply"
    transform = INSTRUMENTED_ROOT / "final_transform.npy"
    return {
        "status": "completed",
        "boundary_export": True,
        "hook_called": hook.called,
        "rng_before_registration": before_rng,
        "rng_after_registration": after_rng,
        "memory": cuda_memory(),
        "fused": raw_pcd_meta(fused),
        "final_transform": {
            "path": str(transform),
            "sha256": sha256(transform),
            "matrix": np.load(transform).tolist(),
        },
        "pre_fusion": hook.capture,
    }


def fusion_only_replay(input_root: Path) -> dict[str, Any]:
    import open3d as o3d
    from fpsample import fps_sampling
    from utils.dataUtils import numpy2o3d, remove_noise_from_point_cloud
    import reg_xyz

    output_root = FUSION_ROOT
    output_root.mkdir(parents=True, exist_ok=False)
    output = FUSION_OUTPUT
    output.mkdir()
    source_path = input_root / "pre_fusion" / "O_aligned.ply"
    target_path = input_root / "pre_fusion" / "G_aligned.ply"
    source_sidecar = input_root / "pre_fusion" / "O_aligned_exact.npz"
    target_sidecar = input_root / "pre_fusion" / "G_aligned_exact.npz"
    fusion_rng_path = input_root / "pre_fusion" / "numpy_rng_before_fusion.pkl"
    with fusion_rng_path.open("rb") as handle:
        fusion_numpy_rng_state = pickle.load(handle)
    np.random.set_state(fusion_numpy_rng_state)
    source = load_exact_pcd_sidecar(source_sidecar)
    target = load_exact_pcd_sidecar(target_sidecar)
    source_meta = pcd_snapshot_meta(pcd_snapshot(source), source_path)
    target_meta = pcd_snapshot_meta(pcd_snapshot(target), target_path)
    filtered_target = reg_xyz.remove_close_points(source, target, distance_threshold=0.0001)
    filtered_meta = pcd_snapshot_meta(pcd_snapshot(filtered_target))
    fused_pcd = source + filtered_target
    _all_fused_pcd = source + target
    fused_xyz = np.asarray(fused_pcd.points)
    fused_color = np.asarray(fused_pcd.colors)
    pre_fps_count = int(len(fused_xyz))
    fused_indices = fps_sampling(fused_xyz, 20000)
    sampled_xyz = fused_xyz[fused_indices]
    sampled_color = fused_color[fused_indices]
    sampled_count = int(len(sampled_xyz))
    fused_pcd = numpy2o3d(sampled_xyz, sampled_color)
    fused_pcd = remove_noise_from_point_cloud(fused_pcd, std_ratio=2.5)
    output_path = output / f"{FLAG}_fused.ply"
    if not o3d.io.write_point_cloud(str(output_path), fused_pcd):
        raise RuntimeError("fusion-only replay failed to write fused point cloud")
    return {
        "status": "completed",
        "source": source_meta,
        "target": target_meta,
        "lossless_sidecars": {
            "O_aligned_exact": {"path": str(source_sidecar), "sha256": sha256(source_sidecar)},
            "G_aligned_exact": {"path": str(target_sidecar), "sha256": sha256(target_sidecar)},
        },
        "fusion_rng": {"path": str(fusion_rng_path), "sha256": sha256(fusion_rng_path)},
        "filtered_target": filtered_meta,
        "point_accounting": {
            "source_count": int(len(source.points)),
            "target_count": int(len(target.points)),
            "filtered_target_count": int(len(filtered_target.points)),
            "pre_fps_fused_count": pre_fps_count,
            "fps_requested": 20000,
            "fps_selected_count": sampled_count,
            "final_count": int(len(fused_pcd.points)),
        },
        "fused": raw_pcd_meta(output_path),
    }


def metric_result(pred_path: Path) -> dict[str, float]:
    from run_method_valid_baseline_07136 import compute_author_metrics

    return compute_author_metrics(pred_path, GROUND_TRUTH)


def main() -> int:
    configure_cuda_runtime()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    result: dict[str, Any] = {
        "run_id": "frozen_g_substrate_07136_20260913_v6",
        "status": "started",
        "genpc_method_valid_substrate": "not_ready",
        "exact_published_baseline": "blocked",
        "experiment1_run": False,
        "structural_association": "not_defined",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canonical_selection_rule": "first successfully completed method-valid baseline lineage by chronology; no CD/EMD quality selection",
        "canonical_source_run": "baseline_07136",
        "official_upstream_commit": "bac9e7b59f4fea0eaaa936f24ef60f342f349829",
        "seed": 42,
        "deterministic_settings": deterministic_settings(),
    }
    try:
        required = [CANONICAL_G, CANONICAL_O, GROUND_TRUTH, BASELINE_ROOT / "baseline_result.json"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("canonical baseline artifacts missing: " + ", ".join(missing))
        if any(path.exists() for path in [SUBSTRATE_ROOT, CONTROL_ROOT, INSTRUMENTED_ROOT, FUSION_ROOT]):
            raise RuntimeError("refusing to reuse or overwrite a frozen-G run directory")
        SUBSTRATE_ROOT.mkdir(parents=True, exist_ok=False)
        result["frozen_inputs"] = copy_frozen_inputs()
        result["canonical_artifacts"] = {
            "G_raw_canonical": mesh_geometry_meta(CANONICAL_G),
            "O_raw_registration": raw_pcd_meta(CANONICAL_O),
            "GT": raw_pcd_meta(GROUND_TRUTH),
            "source_observation": str(UPSTREAM / "data" / f"{FLAG}.ply"),
            "preprocessing_lineage": [
                "upstream data/07136.ply",
                "DepthPrompting/ControlNet stage output preserved in baseline",
                "ScaleAdapter.colorPoint sampling from preserved img.png",
                "baseline color_point.ply selected as O_raw_registration",
            ],
            "zero123_input_sha256": sha256(BASELINE_OUTPUT / "img_sam.png"),
            "zero123_output_sha256": sha256(BASELINE_OUTPUT / f"{FLAG}_zero123++.png"),
            "instantmesh_configuration": {
                "config_path": str(INSTANTMESH / "configs" / "instant-mesh-base.yaml"),
                "config_sha256": sha256(INSTANTMESH / "configs" / "instant-mesh-base.yaml"),
                "diffusion_steps": 75,
                "seed": 42,
                "generative_model": "instantmesh",
            },
        }
        result["registration_configuration"] = {
            "cd_inv_weight": 0.5,
            "diff_init": True,
            "reg_fine_xyz": True,
            "remove_close_points_distance_threshold": 0.0001,
            "dataset": "redwood",
            "device": "cuda",
        }
        CONTROL_ROOT.mkdir(parents=True, exist_ok=False)
        INSTRUMENTED_ROOT.mkdir(parents=True, exist_ok=False)
        result["control_inputs"] = stage_input_files(CONTROL_ROOT)
        result["instrumented_inputs"] = stage_input_files(INSTRUMENTED_ROOT)
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(UPSTREAM))
        sys.path.insert(0, str(INSTANTMESH))
        result["control_registration"] = run_control_registration()
        result["instrumented_registration"] = run_instrumented_registration()
        result["fusion_only_replay"] = fusion_only_replay(INSTRUMENTED_ROOT)

        control_fused = CONTROL_OUTPUT / f"{FLAG}_fused.ply"
        instrumented_fused = INSTRUMENTED_OUTPUT / f"{FLAG}_fused.ply"
        fusion_fused = FUSION_OUTPUT / f"{FLAG}_fused.ply"
        # The author metric itself uses random FPS starts.  Reset the declared
        # evaluation seed for each output so equal point sets receive equal
        # metric samples; this is not used to select a canonical output.
        result["metric_evaluation"] = {
            "fps_points": 16384,
            "seed_reset_per_output": 42,
            "comparison_policy": "compare CD/EMD on the exact common point arrays; retain independent GPU EMD calls as diagnostics",
        }
        result["metrics"] = {}
        for label, path in [("control", control_fused), ("instrumented", instrumented_fused), ("fusion_only", fusion_fused)]:
            set_seed(42)
            result["metrics"][label] = metric_result(path)
        import open3d as o3d

        instr_final = pcd_snapshot(o3d.io.read_point_cloud(str(instrumented_fused)))
        fusion_final = pcd_snapshot(o3d.io.read_point_cloud(str(fusion_fused)))
        fusion_identity = compare_snapshots(instr_final, fusion_final)
        result["same_run_fusion_comparison"] = {
            "instrumented_final": pcd_snapshot_meta(instr_final, instrumented_fused),
            "fusion_only_final": pcd_snapshot_meta(fusion_final, fusion_fused),
            "numeric_geometry_identity": fusion_identity,
            "point_accounting_match": bool(
                result["fusion_only_replay"]["point_accounting"]["final_count"] == instr_final["point_count"]
            ),
            "cd_abs_diff": abs(result["metrics"]["instrumented"]["cd"] - result["metrics"]["fusion_only"]["cd"]),
            "emd_abs_diff": abs(result["metrics"]["instrumented"]["emd"] - result["metrics"]["fusion_only"]["emd"]),
            "metric_tolerance": None,
            "metric_tolerance_basis": "not used for the readiness gate because independent GPU EMD calls are nondeterministic; exact common point-array identity is the metric-input agreement proof",
            "independent_metric_diagnostic": {
                "cd_abs_diff": abs(result["metrics"]["instrumented"]["cd"] - result["metrics"]["fusion_only"]["cd"]),
                "emd_abs_diff": abs(result["metrics"]["instrumented"]["emd"] - result["metrics"]["fusion_only"]["emd"]),
                "note": "Independent author-metric calls can differ in GPU EMD despite identical point arrays; values are preserved, not quality-selected.",
            },
        }
        result["same_run_fusion_comparison"]["metrics_identity_pass"] = bool(
            result["same_run_fusion_comparison"]["numeric_geometry_identity"]["identity_pass"]
            and result["same_run_fusion_comparison"]["point_accounting_match"]
        )
        result["same_run_fusion_comparison"]["sufficiency_pass"] = bool(
            fusion_identity["identity_pass"]
            and result["same_run_fusion_comparison"]["point_accounting_match"]
            and result["same_run_fusion_comparison"]["metrics_identity_pass"]
        )
        hook_capture = result["instrumented_registration"]["pre_fusion"]
        result["read_only_hook_identity_pass"] = bool(hook_capture["identity_pass"])
        result["registration_reached_boundary"] = bool(result["instrumented_registration"]["hook_called"])
        result["GT_and_evaluation_frozen"] = bool(
            result["canonical_artifacts"]["GT"]["sha256"] == sha256(GROUND_TRUTH)
        )
        result["genpc_method_valid_substrate"] = "ready" if all(
            [
                result["registration_reached_boundary"],
                result["read_only_hook_identity_pass"],
                result["same_run_fusion_comparison"]["sufficiency_pass"],
                result["GT_and_evaluation_frozen"],
                result["experiment1_run"] is False,
            ]
        ) else "not_ready"
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" and result["genpc_method_valid_substrate"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
