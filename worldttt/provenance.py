"""Reproducibility and data-provenance helpers for WorldTTT experiments.

The repository intentionally contains an older ``configs/worldttt`` copy.  This
module is imported from the root package and records the package location,
source tree, reference repositories, resolved data counts, and runtime
versions.  It never mutates experiment state unless ``write_manifest`` is
explicitly called by a CLI or an experiment launcher.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


CANONICAL_PACKAGE = "worldttt"
REFERENCE_REPOSITORIES = {
    "Sana": ("/home/newuser001/huangyu/Sana", "f917874"),
    "RoboTTT": ("/home/newuser001/huangyu/robo_ttt", "98eb883"),
    "Titans": ("/home/newuser001/huangyu/titans-pytorch-Unofficial-implementation", "1d40c44"),
    "MoRAM": ("/home/newuser001/huangyu/MoRAM", "aaa51d8"),
    "YUME": ("/home/newuser001/huangyu/YUME", "111c3fa"),
}


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def repository_fingerprint(repo: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(repo).resolve()
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "commit": _git(path, "rev-parse", "HEAD") if path.is_dir() else None,
        "dirty": bool(_git(path, "status", "--porcelain")) if path.is_dir() else None,
        "tree": _git(path, "rev-parse", "HEAD^{tree}") if path.is_dir() else None,
    }


def _zip_keys(path: Path, suffix: str) -> list[str] | None:
    if not path.is_file():
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            return [Path(name).stem for name in archive.namelist() if name.lower().endswith(suffix)]
    except (OSError, zipfile.BadZipFile):
        return None


def resolved_data_manifest(root: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(root).resolve()
    raw_zip = next(root.glob("data/sekai_game_train_961frames_16fps_ovl640/*.zip"), None)
    latent_zip = next(root.glob("data/vae_cache/**/sekai_game_train_00000000.zip"), None)
    camera = next(root.glob("data/sekai_game_train_961frames_16fps_ovl640/*_camera.npz"), None)
    repo = root.parents[1]
    scenes = next((path for path in (
        repo / "datasets/worldttt/scenes.jsonl",
        repo.parent / "extra/scenes.jsonl",
    ) if path.is_file()), None)
    scene_count = sum(1 for _ in scenes.open(encoding="utf-8")) if scenes is not None else None
    raw_keys = _zip_keys(raw_zip, ".mp4") if raw_zip else None
    latent_keys = _zip_keys(latent_zip, ".npz") if latent_zip else None
    filtered = sorted(set(raw_keys) - set(latent_keys)) if raw_keys is not None and latent_keys is not None else None
    return {
        "root": str(root),
        "raw_zip": str(raw_zip) if raw_zip else None,
        "latent_zip": str(latent_zip) if latent_zip else None,
        "camera_npz": str(camera) if camera else None,
        "raw_clip_count": len(raw_keys) if raw_keys is not None else None,
        "latent_clip_count": len(latent_keys) if latent_keys is not None else None,
        "raw_without_latent": len(filtered) if filtered is not None else None,
        "filtered_raw_keys": filtered,
        "scene_manifest": str(scenes) if scenes is not None else None,
        "scene_manifest_count": scene_count,
        "pixel_frames": 961,
        "latent_frames": 121,
        "temporal_stride": 8,
    }


def _package_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", "installed")


def runtime_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": {name: _package_version(name) for name in ("torch", "diffusers", "triton", "pyrallis", "numpy")},
    }
    try:
        import torch

        snapshot["torch_cuda"] = torch.version.cuda
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        snapshot["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            snapshot["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:  # pragma: no cover - depends on host runtime
        snapshot["torch_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def source_tree_hash(root: str | os.PathLike[str]) -> str:
    """Hash tracked and untracked source files without reading large datasets."""
    root = Path(root).resolve()
    digest = hashlib.sha256()
    excluded = {".git", ".cache", ".pytest_cache", "__pycache__", "runs", "datasets"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        if path.name.startswith("worldttt-provenance") and path.suffix == ".json":
            continue  # generated manifests must not hash themselves
        if path.suffix not in {".py", ".json", ".yaml", ".yml", ".md", ".toml", ".sh"}:
            continue
        rel = path.relative_to(root).as_posix().encode()
        digest.update(rel)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_manifest(root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    root_path = Path(root or Path(__file__).resolve().parents[1]).resolve()
    data_root = root_path / "datasets/sana-wm-example"
    return {
        "schema_version": 1,
        "canonical_package": CANONICAL_PACKAGE,
        "repository": repository_fingerprint(root_path),
        "source_tree_hash": source_tree_hash(root_path),
        "references": {name: repository_fingerprint(root_path.parent / Path(path).name) | {"expected_revision": rev}
                       for name, (path, rev) in REFERENCE_REPOSITORIES.items()},
        "data": resolved_data_manifest(data_root),
        "runtime": runtime_snapshot(),
    }


def write_manifest(path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    payload = build_manifest(root)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    import argparse

    parser = argparse.ArgumentParser(description="Write a WorldTTT provenance manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    print(json.dumps(write_manifest(args.output, args.root), indent=2, sort_keys=True))
