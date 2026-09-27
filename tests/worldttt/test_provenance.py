"""Local and remote research workspaces share one provenance contract."""

from pathlib import Path
from zipfile import ZipFile

from worldttt.provenance import build_manifest, resolved_data_manifest, source_tree_hash


def test_manifest_discovers_reference_repositories_beside_worldttt(tmp_path):
    workspace = tmp_path / "workspace"
    repo = workspace / "WorldTTT"
    repo.mkdir(parents=True)
    (workspace / "Sana").mkdir()
    (workspace / "MoRAM").mkdir()

    manifest = build_manifest(repo)
    assert manifest["references"]["Sana"]["exists"]
    assert manifest["references"]["Sana"]["path"] == str((workspace / "Sana").resolve())
    assert manifest["references"]["MoRAM"]["exists"]


def test_data_manifest_lists_raw_clips_missing_latents(tmp_path):
    raw = tmp_path / "data" / "sekai_game_train_961frames_16fps_ovl640" / "raw.zip"
    latent = tmp_path / "data" / "vae_cache" / "v1" / "sekai_game_train_00000000.zip"
    raw.parent.mkdir(parents=True)
    latent.parent.mkdir(parents=True)
    with ZipFile(raw, "w") as archive:
        for name in ("a.mp4", "b.mp4", "c.mp4"):
            archive.writestr(name, b"")
    with ZipFile(latent, "w") as archive:
        for name in ("a.npz", "c.npz"):
            archive.writestr(name, b"")

    manifest = resolved_data_manifest(tmp_path)
    assert manifest["raw_clip_count"] == 3
    assert manifest["latent_clip_count"] == 2
    assert manifest["raw_without_latent"] == 1
    assert manifest["filtered_raw_keys"] == ["b"]


def test_local_workspace_scene_manifest_is_counted(tmp_path):
    workspace = tmp_path / "workspace"
    data_root = workspace / "WorldTTT" / "datasets" / "sana-wm-example"
    data_root.mkdir(parents=True)
    extra = workspace / "extra"
    extra.mkdir()
    (extra / "scenes.jsonl").write_text('{"key":"a"}\n{"key":"b"}\n', encoding="utf-8")

    manifest = resolved_data_manifest(data_root)
    assert manifest["scene_manifest_count"] == 2
    assert manifest["scene_manifest"] == str((extra / "scenes.jsonl").resolve())


def test_generated_provenance_does_not_change_source_hash(tmp_path):
    (tmp_path / "module.py").write_text("x = 1\n", encoding="utf-8")
    before = source_tree_hash(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "worldttt-provenance-local.json").write_text('{"source_tree_hash":"placeholder"}\n', encoding="utf-8")
    assert source_tree_hash(tmp_path) == before


def test_pytest_cache_is_not_part_of_source_identity(tmp_path):
    (tmp_path / "module.py").write_text("x = 1\n", encoding="utf-8")
    before = source_tree_hash(tmp_path)
    cache = tmp_path / ".pytest_cache"
    cache.mkdir()
    (cache / "README.md").write_text("test runner cache\n", encoding="utf-8")
    assert source_tree_hash(tmp_path) == before
