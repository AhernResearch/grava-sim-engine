from __future__ import annotations

import copy
from pathlib import Path

from grava_sim_engine.adapters.dataset_loader import (
    InternalDatasetLoader,
    get_boost_cache_dir,
    load_scene_context,
    resolve_dataset_loader,
    save_source_scene_context,
    set_boost_cache_dir,
    warmup_boost_cache,
    write_scene_context_metadata,
)


def test_internal_scene_context_cache_loads_and_writes_boost(tmp_path: Path, straight_road_scene):
    source_dir = tmp_path / "source"
    boost_dir = tmp_path / "boost"
    write_scene_context_metadata(source_dir)
    save_source_scene_context(straight_road_scene, source_dir)

    previous_boost_cache_dir = get_boost_cache_dir()
    load_scene_context.cache_clear()
    try:
        set_boost_cache_dir(str(boost_dir))
        scene = load_scene_context(
            source_dir,
            straight_road_scene.log_name,
            straight_road_scene.scene_token,
        )
        assert scene.scene_token == straight_road_scene.scene_token

        boost_path = boost_dir / straight_road_scene.log_name / f"{straight_road_scene.scene_token}.pkl"
        assert boost_path.exists()

        (source_dir / straight_road_scene.log_name / f"{straight_road_scene.scene_token}.pkl").unlink()
        load_scene_context.cache_clear()
        scene_from_boost = load_scene_context(
            source_dir,
            straight_road_scene.log_name,
            straight_road_scene.scene_token,
        )
    finally:
        load_scene_context.cache_clear()
        set_boost_cache_dir(previous_boost_cache_dir)

    assert scene_from_boost.scene_token == straight_road_scene.scene_token


def test_resolve_dataset_loader_detects_internal_metadata(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    assert not isinstance(resolve_dataset_loader(source_dir), InternalDatasetLoader)

    write_scene_context_metadata(source_dir)
    assert isinstance(resolve_dataset_loader(source_dir), InternalDatasetLoader)


def test_internal_warmup_reuses_boost_progress_stats(tmp_path: Path, straight_road_scene):
    source_dir = tmp_path / "source"
    boost_dir = tmp_path / "boost"
    write_scene_context_metadata(source_dir)

    scene_a = copy.deepcopy(straight_road_scene)
    scene_a.log_name = "log_a"
    scene_a.scene_token = "scene_a"
    scene_b = copy.deepcopy(straight_road_scene)
    scene_b.log_name = "log_b"
    scene_b.scene_token = "scene_b"
    save_source_scene_context(scene_a, source_dir)
    save_source_scene_context(scene_b, source_dir)

    previous_boost_cache_dir = get_boost_cache_dir()
    try:
        set_boost_cache_dir(str(boost_dir))
        stats = warmup_boost_cache(str(source_dir), str(boost_dir), num_workers=1)
    finally:
        set_boost_cache_dir(previous_boost_cache_dir)

    assert stats["total"] == 2
    assert stats["converted"] == 2
    assert (boost_dir / "log_a" / "scene_a.pkl").exists()
    assert (boost_dir / "log_b" / "scene_b.pkl").exists()
