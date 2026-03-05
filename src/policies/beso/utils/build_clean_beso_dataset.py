#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a clean LeRobot dataset for BESO training (new dataset version):
1) Merge sub-datasets in a fixed order (reindex episode_index/index/task_index correctly)
2) Split merged raw dataset into train/eval subsets
3) For each split, add standard keys:
   - observation.state = concat(q202, gripper_width_202)   [8D]
   - action = concat(action_q202, action_gripper202)       [8D]
   - observation.goal.tail_q202 = per-episode fixed tail goal, repeated on every frame (shape [1, D_goal])
4) Remove old split keys (do not keep legacy keys)

Notes:
- This script intentionally uses LeRobot official offline dataset tools.
- rename_processor is runtime-only; it does not compute new values or fix episode collisions.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lerobot-src",
        type=str,
        default="/home/zzh/workspace/Robot_learning/lerobot/src",
        help="Path to lerobot source root (contains lerobot/ package).",
    )
    ap.add_argument(
        "--datasets-root",
        type=str,
        default="/home/zzh/workspace/Robot_learning/dataset_new_2",
        help="Root containing sub-datasets (place_1..place_8, freestyle folders).",
    )
    ap.add_argument(
        "--merged-tmp-root",
        type=str,
        default="/home/zzh/workspace/Robot_learning/dataset_new_2/_tmp_beso_merged_raw",
        help="Temporary merged dataset root (will be created).",
    )
    ap.add_argument(
        "--output-root",
        type=str,
        default="/home/zzh/workspace/Robot_learning/dataset_new_2/banana_beso_clean_v1",
        help="Only used for its parent directory, where banana_beso_train/eval are written.",
    )
    ap.add_argument(
        "--output-repo-id",
        type=str,
        default="banana_beso_clean_v1",
        help="Repo id label stored in LeRobot metadata.",
    )
    ap.add_argument(
        "--goal-tail-frames",
        type=int,
        default=10,
        help="Number of final q202 frames used to build goal; flattened to shape [1, goal_tail_frames*7].",
    )
    ap.add_argument(
        "--goal-key",
        type=str,
        default="observation.goal.tail_q202",
        help="Name of the goal feature to add.",
    )
    ap.add_argument(
        "--keep-temp-merged",
        action="store_true",
        help="Keep temporary merged raw dataset instead of deleting it.",
    )
    ap.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used when sampling eval episodes.",
    )
    return ap.parse_args()


def _canonical_source_order(root: Path) -> list[Path]:
    # Fixed order to avoid episode_index ambiguity and to make aggregation deterministic.
    alias_groups = [[f"place_{i}"] for i in range(1, 9)] + [
        ["0-10_freestyle", "0-10 freestyle", "0-10_ freestyle"],
    ]
    paths: list[Path] = []
    missing_groups: list[list[str]] = []
    for aliases in alias_groups:
        matched = None
        for name in aliases:
            p = root / name
            if p.exists():
                matched = p
                break
        if matched is None:
            missing_groups.append(aliases)
        else:
            paths.append(matched)

    if missing_groups:
        missing_str = ["/".join(group) for group in missing_groups]
        raise FileNotFoundError(f"Missing dataset folders (any alias works): {missing_str}")

    return paths


def _tail_flatten(arr: np.ndarray, tail_frames: int) -> np.ndarray:
    # arr: [L, D]
    if arr.ndim != 2:
        raise ValueError(f"Expected [L, D], got {arr.shape}")
    if tail_frames <= 0:
        raise ValueError(f"tail_frames must be > 0, got {tail_frames}")

    L, D = arr.shape
    if L == 0:
        raise ValueError("Empty episode array")

    if L >= tail_frames:
        tail = arr[-tail_frames:]  # [tail_frames, D]
    else:
        # Left pad with the first frame to keep a fixed-length goal when episodes are short.
        pad = np.repeat(arr[:1], tail_frames - L, axis=0)
        tail = np.concatenate([pad, arr], axis=0)

    return tail.reshape(tail_frames * D).astype(np.float32)


def _build_episode_tables(merged_dataset, goal_tail_frames: int):
    """
    Precompute per-episode sequences after aggregation (episode_index already globally reindexed).

    Returns:
      episode_to_action: dict[int, np.ndarray]  # [L, 8]
      episode_to_state:  dict[int, np.ndarray]  # [L, 8]
      episode_to_goal:   dict[int, np.ndarray]  # [goal_tail_frames * 7]
    """
    hf = merged_dataset.hf_dataset.with_format(None)

    q202 = np.asarray(hf["observation.state.q.Panda202"], dtype=np.float32)                    # [N, 7]
    aq202 = np.asarray(hf["action.q.Panda202"], dtype=np.float32)                               # [N, 7]
    g202 = np.asarray(hf["action.gripper_width.PandaGripper202"], dtype=np.float32)             # [N] or [N,1]
    obs_g202 = np.asarray(hf["observation.state.gripper_width.PandaGripper202"], dtype=np.float32)  # [N] or [N,1]
    ep_idx = np.asarray(hf["episode_index"], dtype=np.int64)                                    # [N]
    frame_idx = np.asarray(hf["frame_index"], dtype=np.int64)                                   # [N]

    if g202.ndim == 1:
        g202 = g202[:, None]  # [N, 1]
    elif g202.ndim == 2 and g202.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Unexpected gripper action shape: {g202.shape}")

    if obs_g202.ndim == 1:
        obs_g202 = obs_g202[:, None]  # [N, 1]
    elif obs_g202.ndim == 2 and obs_g202.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Unexpected obs gripper shape: {obs_g202.shape}")

    N = len(ep_idx)
    if not (len(q202) == len(aq202) == len(g202) == len(obs_g202) == len(frame_idx) == N):
        raise ValueError("Column lengths mismatch in merged dataset")

    # Collect rows by episode in global order (after merge, episode_index is already unique/reindexed)
    rows_by_ep: dict[int, list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
    for i in range(N):
        e = int(ep_idx[i])
        f = int(frame_idx[i])
        rows_by_ep.setdefault(e, []).append((f, q202[i], aq202[i], g202[i], obs_g202[i]))

    unique_eps = sorted(rows_by_ep.keys())
    # Expect contiguous after official aggregation
    if unique_eps != list(range(len(unique_eps))):
        raise ValueError(
            f"Aggregated episode_index is not contiguous 0..N-1. Got head={unique_eps[:10]}"
        )

    episode_to_action: dict[int, np.ndarray] = {}
    episode_to_state: dict[int, np.ndarray] = {}
    episode_to_goal: dict[int, np.ndarray] = {}

    for e in unique_eps:
        rows = sorted(rows_by_ep[e], key=lambda x: x[0])

        # Validate frame_index continuity inside each episode
        expected_frames = list(range(len(rows)))
        got_frames = [r[0] for r in rows]
        if got_frames != expected_frames:
            raise ValueError(
                f"Episode {e}: frame_index not contiguous from 0. "
                f"Expected head={expected_frames[:10]}, got head={got_frames[:10]}"
            )

        q_seq = np.stack([r[1] for r in rows], axis=0).astype(np.float32)         # [L, 7]
        aq_seq = np.stack([r[2] for r in rows], axis=0).astype(np.float32)        # [L, 7]
        g_seq = np.stack([r[3] for r in rows], axis=0).astype(np.float32)         # [L, 1]
        obs_g202_seq = np.stack([r[4] for r in rows], axis=0).astype(np.float32)  # [L, 1]

        # action = [action.q.Panda202, action.gripper_width.PandaGripper202]
        action_seq = np.concatenate([aq_seq, g_seq], axis=1)  # [L, 8]

        # observation.state = [observation.state.q.Panda202, observation.state.gripper_width.PandaGripper202]
        state_seq = np.concatenate([q_seq, obs_g202_seq], axis=1)  # [L, 8]

        # goal = flatten(last goal_tail_frames of q202), shape [goal_tail_frames * 7]
        goal_vec = _tail_flatten(q_seq, tail_frames=goal_tail_frames)

        episode_to_action[e] = action_seq
        episode_to_state[e] = state_seq
        episode_to_goal[e] = goal_vec

    print(f"[INFO] Built episode tables for {len(unique_eps)} episodes")
    print(f"[INFO] Example episode 0: action shape={episode_to_action[0].shape}, state shape={episode_to_state[0].shape}, goal shape={episode_to_goal[0].shape}")
    return episode_to_action, episode_to_state, episode_to_goal


def _patch_visual_feature_names_if_missing(dataset_root: Path) -> None:
    """
    Some LeRobot versions expect ft['names'] for image/video features in meta/info.json.
    Older datasets often omit this field for visual features, so patch it here once offline.
    """
    info_path = dataset_root / "meta" / "info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    features = info.get("features", {})
    changed = False
    for key, ft in features.items():
        if not isinstance(ft, dict):
            continue
        if ft.get("dtype") in {"image", "video"} and "names" not in ft:
            # LeRobot visual features are stored in dataset metadata as [H, W, C].
            ft["names"] = ["height", "width", "channel"]
            changed = True

    if changed:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Patched missing visual feature names in {info_path}")


def _recompute_added_feature_stats(clean_dataset, goal_key: str, remove_features: list[str]) -> None:
    """
    Recompute stats.json AND per-episode stats in meta/episodes parquet for
    newly added / reindexed features, and clean up orphaned stats columns
    left over from removed features.
    """
    from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
    from lerobot.datasets.utils import flatten_dict, write_stats

    hf = clean_dataset.hf_dataset.with_format(None)

    # Derived vector features we added
    derived_keys = ["action", "observation.state", goal_key]
    # Scalar/index features whose range changes after split & reindex
    index_keys = [k for k in ["episode_index", "index", "frame_index", "timestamp", "task_index"]
                  if k in clean_dataset.meta.features]
    target_keys = derived_keys + index_keys

    for k in target_keys:
        if k not in clean_dataset.meta.features:
            raise ValueError(f"Missing feature in dataset: {k}")

    feature_desc = {k: clean_dataset.meta.features[k] for k in target_keys}
    ep_stats_list = []
    total_eps = clean_dataset.meta.total_episodes

    for ep in range(total_eps):
        from_idx = int(clean_dataset.meta.episodes["dataset_from_index"][ep])
        to_idx = int(clean_dataset.meta.episodes["dataset_to_index"][ep])
        ep_ds = hf.select(range(from_idx, to_idx))

        ep_data = {}
        for k in target_keys:
            ep_data[k] = np.asarray(ep_ds[k], dtype=np.float32)

        ep_stats = compute_episode_stats(ep_data, feature_desc)
        ep_stats_list.append(ep_stats)

        if ep < 3:
            print(
                f"[debug] ep={ep} lens={to_idx-from_idx} "
                f"action_shape={ep_data['action'].shape} "
                f"state_shape={ep_data['observation.state'].shape} "
                f"goal_shape={ep_data[goal_key].shape}"
            )

    # --- 1) Update meta/stats.json (global aggregate) ---
    new_stats = aggregate_stats(ep_stats_list)
    merged_stats = dict(clean_dataset.meta.stats) if clean_dataset.meta.stats is not None else {}
    merged_stats.update(new_stats)
    write_stats(merged_stats, clean_dataset.root)
    print("[INFO] stats.json updated.")
    print("[INFO] added/updated stats keys:", list(new_stats.keys()))

    # --- 2) Update per-episode stats in meta/episodes parquet ---
    episodes_dir = clean_dataset.root / "meta" / "episodes"
    ep_parquet_files = sorted(episodes_dir.rglob("*.parquet"))
    if not ep_parquet_files:
        print("[WARN] No episode parquet files found, skipping per-episode stats fix.")
        return

    # Pre-flatten all per-episode stats into {col_name: [val_ep0, val_ep1, ...]}
    all_flat_cols: dict[str, list] = {}
    for ep_idx, ep_stat in enumerate(ep_stats_list):
        flat = flatten_dict({"stats": ep_stat})
        for col_name, value in flat.items():
            if col_name not in all_flat_cols:
                all_flat_cols[col_name] = [None] * len(ep_stats_list)
            if isinstance(value, np.ndarray):
                # Match existing parquet convention: store as float64 1-D arrays
                value = value.flatten().astype(np.float64)
            all_flat_cols[col_name][ep_idx] = value

    for ep_file in ep_parquet_files:
        df_ep = pd.read_parquet(ep_file)

        # 2a) Remove orphaned stats columns from deleted features
        orphan_prefixes = [f"stats/{feat}/" for feat in remove_features]
        cols_to_drop = [c for c in df_ep.columns
                        if any(c.startswith(prefix) for prefix in orphan_prefixes)]
        if cols_to_drop:
            df_ep = df_ep.drop(columns=cols_to_drop)
            print(f"[INFO] Dropped {len(cols_to_drop)} orphaned stats columns from {ep_file.name}")

        # 2b) Update / add per-episode stats for target_keys
        ep_indices = df_ep["episode_index"].astype(int).tolist()
        for col_name, values_by_ep in all_flat_cols.items():
            col_values = [values_by_ep[ep] for ep in ep_indices]
            df_ep[col_name] = col_values

        df_ep.to_parquet(ep_file, index=False)
        print(f"[INFO] Updated per-episode stats in {ep_file}")


def _sample_train_eval_episodes(
    source_ep_counts: list[int], seed: int, n_eval: int = 5,
) -> tuple[list[int], list[int]]:
    """Randomly sample n_eval episodes globally for evaluation.

    Args:
        source_ep_counts: number of episodes in each source folder (in merge order).
        seed: random seed.
        n_eval: number of eval episodes to sample globally.
    Returns:
        (train_episodes, eval_episodes) as sorted lists of merged episode indices.
    """
    total_episodes = sum(source_ep_counts)
    if n_eval >= total_episodes:
        raise ValueError(f"n_eval={n_eval} >= total_episodes={total_episodes}")
    rng = random.Random(seed)
    eval_episodes = sorted(rng.sample(range(total_episodes), n_eval))
    eval_set = set(eval_episodes)
    train_episodes = [ep for ep in range(total_episodes) if ep not in eval_set]
    return train_episodes, eval_episodes


def _write_split_manifest(split_root: Path, split_name: str, kept_old_episodes: list[int]) -> None:
    payload = {
        "split": split_name,
        "num_episodes": len(kept_old_episodes),
        "new_to_old_episode_index": kept_old_episodes,
        "old_to_new_episode_index": {str(old): new for new, old in enumerate(kept_old_episodes)},
    }
    with open(split_root / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] wrote split manifest: {split_root / 'split_manifest.json'}")


def _restore_split_video_alignment(
    merged_dataset,
    split_root: Path,
    kept_old_episodes: list[int],
    split_name: str,
) -> None:
    """
    Keep split videos strictly aligned with merged raw dataset:
    - copy original video files from merged dataset (no partial re-encode)
    - restore per-episode from/to timestamps in meta/episodes from merged metadata
    """
    video_keys = list(merged_dataset.meta.video_keys)
    if not video_keys:
        return

    # 1) Copy original video files used by kept episodes.
    rel_video_paths = set()
    for old_ep in kept_old_episodes:
        for video_key in video_keys:
            rel_video_paths.add(merged_dataset.meta.get_video_file_path(old_ep, video_key))

    for rel_path in sorted(rel_video_paths):
        src = merged_dataset.root / rel_path
        dst = split_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # 2) Patch split episode metadata video fields back to merged timestamps.
    new_to_old = {new: old for new, old in enumerate(kept_old_episodes)}
    episodes_dir = split_root / "meta" / "episodes"
    episode_files = sorted(episodes_dir.glob("*/*.parquet"))

    for ep_file in episode_files:
        df = pd.read_parquet(ep_file)
        ep_series = df["episode_index"].astype(int)

        for video_key in video_keys:
            chunk_col = f"videos/{video_key}/chunk_index"
            file_col = f"videos/{video_key}/file_index"
            from_col = f"videos/{video_key}/from_timestamp"
            to_col = f"videos/{video_key}/to_timestamp"

            df[chunk_col] = ep_series.map(
                lambda new_ep: int(merged_dataset.meta.episodes[new_to_old[int(new_ep)]][chunk_col])
            )
            df[file_col] = ep_series.map(
                lambda new_ep: int(merged_dataset.meta.episodes[new_to_old[int(new_ep)]][file_col])
            )
            df[from_col] = ep_series.map(
                lambda new_ep: float(merged_dataset.meta.episodes[new_to_old[int(new_ep)]][from_col])
            )
            df[to_col] = ep_series.map(
                lambda new_ep: float(merged_dataset.meta.episodes[new_to_old[int(new_ep)]][to_col])
            )

        df.to_parquet(ep_file, index=False)

    print(f"[INFO] restored video alignment for split={split_name}")


def _build_clean_dataset(
    source_dataset,
    output_root: Path,
    output_repo_id: str,
    goal_tail_frames: int,
    goal_key: str,
):
    from lerobot.datasets.dataset_tools import modify_features

    episode_to_action, episode_to_state, episode_to_goal = _build_episode_tables(
        source_dataset,
        goal_tail_frames=goal_tail_frames,
    )
    goal_dim = goal_tail_frames * 7

    def action_feature_fn(row: dict, ep_idx: int, frame_in_ep: int):
        return episode_to_action[int(ep_idx)][int(frame_in_ep)].tolist()

    def obs_state_feature_fn(row: dict, ep_idx: int, frame_in_ep: int):
        return episode_to_state[int(ep_idx)][int(frame_in_ep)].tolist()

    def goal_feature_fn(row: dict, ep_idx: int, frame_in_ep: int):
        goal_vec = episode_to_goal[int(ep_idx)]
        return [goal_vec.tolist()]

    add_features = {
        "action": (
            action_feature_fn,
            {
                "dtype": "float32",
                "shape": [8],
                "names": None,
            },
        ),
        "observation.state": (
            obs_state_feature_fn,
            {
                "dtype": "float32",
                "shape": [8],
                "names": None,
            },
        ),
        goal_key: (
            goal_feature_fn,
            {
                "dtype": "float32",
                "shape": [1, goal_dim],
                "names": None,
            },
        ),
    }

    remove_features = [
        "observation.state.q.Panda201",
        "observation.state.gripper_width.PandaGripper201",
        "observation.state.q.Panda202",
        "observation.state.gripper_width.PandaGripper202",
        "action.q.Panda202",
        "action.gripper_width.PandaGripper202",
    ]

    if output_root.exists():
        raise FileExistsError(f"Output root already exists: {output_root}")

    print(f"[INFO] Writing cleaned dataset to: {output_root}")
    clean_dataset = modify_features(
        dataset=source_dataset,
        add_features=add_features,
        remove_features=remove_features,
        output_dir=output_root,
        repo_id=output_repo_id,
    )

    _patch_visual_feature_names_if_missing(clean_dataset.root)
    _recompute_added_feature_stats(clean_dataset, goal_key, remove_features)

    print("[INFO] Clean dataset created.")
    print(f"[INFO] repo_id={clean_dataset.repo_id}")
    print(f"[INFO] root={clean_dataset.root}")
    print(f"[INFO] total_episodes={clean_dataset.meta.total_episodes}, total_frames={clean_dataset.meta.total_frames}")
    print("[INFO] Final feature keys:")
    for k in clean_dataset.meta.features.keys():
        print(f"  - {k}")

    return clean_dataset


def _split_train_eval_raw_datasets(
    merged_dataset, datasets_root: Path, seed: int, source_ep_counts: list[int]
):
    from lerobot.datasets.dataset_tools import delete_episodes

    train_episodes, eval_episodes = _sample_train_eval_episodes(source_ep_counts, seed)
    print(f"[INFO] sampled eval episodes in merged dataset={eval_episodes}")

    train_raw_root = datasets_root / "_tmp_beso_train_raw"
    eval_raw_root = datasets_root / "_tmp_beso_eval_raw"
    if train_raw_root.exists():
        raise FileExistsError(f"Temporary train raw root already exists: {train_raw_root}")
    if eval_raw_root.exists():
        raise FileExistsError(f"Temporary eval raw root already exists: {eval_raw_root}")

    train_raw = delete_episodes(
        dataset=merged_dataset,
        episode_indices=eval_episodes,
        output_dir=train_raw_root,
        repo_id="banana_beso_train_raw_tmp",
    )
    eval_raw = delete_episodes(
        dataset=merged_dataset,
        episode_indices=train_episodes,
        output_dir=eval_raw_root,
        repo_id="banana_beso_eval_raw_tmp",
    )
    return (
        train_raw,
        eval_raw,
        train_raw_root,
        eval_raw_root,
        train_episodes,
        eval_episodes,
    )


def main():
    args = _parse_args()

    # Import lerobot from local source path (avoid relying on global pip install)
    lerobot_src = Path(args.lerobot_src)
    if not lerobot_src.exists():
        raise FileNotFoundError(f"--lerobot-src not found: {lerobot_src}")
    sys.path.insert(0, str(lerobot_src))

    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    datasets_root = Path(args.datasets_root)
    output_root = Path(args.output_root)

    source_paths = _canonical_source_order(datasets_root)

    # 1) Load source datasets in a fixed order
    source_datasets = []
    print("[INFO] Source order:")
    for i, p in enumerate(source_paths):
        # repo_id here is just a local label; root is the actual dataset root.
        repo_id = f"src_{i:02d}_{p.name.replace(' ', '_')}"
        ds = LeRobotDataset(
            repo_id=repo_id,
            root=p,
            download_videos=False,
        )
        source_datasets.append(ds)
        print(f"  {i:02d}: {p.name}  episodes={ds.meta.total_episodes} frames={ds.meta.total_frames}")

    # 2) Merge first (official aggregation) -> this fixes episode/index/task reindexing across folders
    merged_tmp_root = Path(args.merged_tmp_root)
    merged_tmp_repo = "banana_beso_merged_tmp"

    if merged_tmp_root.exists():
        raise FileExistsError(f"Temporary merged output already exists: {merged_tmp_root}")

    print(f"[INFO] Merging datasets into temporary root: {merged_tmp_root}")
    merged_dataset = merge_datasets(
        datasets=source_datasets,
        output_repo_id=merged_tmp_repo,
        output_dir=merged_tmp_root,
    )

    # Sanity check aggregated episode indexing
    hf_merged = merged_dataset.hf_dataset.with_format(None)
    merged_eps = np.asarray(hf_merged["episode_index"], dtype=np.int64)
    uniq_eps = np.unique(merged_eps)
    print(f"[INFO] Aggregated total episodes = {merged_dataset.meta.total_episodes}")
    print(f"[INFO] Aggregated unique episode_index count = {len(uniq_eps)}")
    print(f"[INFO] episode_index head = {uniq_eps[:10].tolist()}, tail = {uniq_eps[-10:].tolist()}")
    if uniq_eps.tolist() != list(range(len(uniq_eps))):
        raise ValueError("Aggregated episode_index is not contiguous after merge; stop and inspect.")

    # Collect per-source episode counts (in merge order) for stratified eval sampling
    source_ep_counts = [int(ds.meta.total_episodes) for ds in source_datasets]

    # 3) Split merged raw dataset first (avoids pandas bug when splitting after derived-feature injection)
    train_raw, eval_raw, train_raw_root, eval_raw_root, train_episodes, eval_episodes = _split_train_eval_raw_datasets(
        merged_dataset=merged_dataset,
        datasets_root=datasets_root,
        seed=args.split_seed,
        source_ep_counts=source_ep_counts,
    )

    _restore_split_video_alignment(
        merged_dataset=merged_dataset,
        split_root=train_raw_root,
        kept_old_episodes=train_episodes,
        split_name="train",
    )
    _restore_split_video_alignment(
        merged_dataset=merged_dataset,
        split_root=eval_raw_root,
        kept_old_episodes=eval_episodes,
        split_name="eval",
    )
    _write_split_manifest(train_raw_root, "train_raw", train_episodes)
    _write_split_manifest(eval_raw_root, "eval_raw", eval_episodes)

    # 4) Build final clean train/eval datasets
    train_root = output_root.parent / "banana_beso_train"
    eval_root = output_root.parent / "banana_beso_eval"
    _build_clean_dataset(
        source_dataset=train_raw,
        output_root=train_root,
        output_repo_id="banana_beso_train",
        goal_tail_frames=args.goal_tail_frames,
        goal_key=args.goal_key,
    )
    _build_clean_dataset(
        source_dataset=eval_raw,
        output_root=eval_root,
        output_repo_id="banana_beso_eval",
        goal_tail_frames=args.goal_tail_frames,
        goal_key=args.goal_key,
    )
    _write_split_manifest(train_root, "train", train_episodes)
    _write_split_manifest(eval_root, "eval", eval_episodes)

    # 5) Remove split raw temporary datasets
    print(f"[INFO] Removing temporary train raw dataset: {train_raw_root}")
    shutil.rmtree(train_raw_root, ignore_errors=True)
    print(f"[INFO] Removing temporary eval raw dataset: {eval_raw_root}")
    shutil.rmtree(eval_raw_root, ignore_errors=True)

    # 6) Optional cleanup of temp merged dataset
    if not args.keep_temp_merged:
        print(f"[INFO] Removing temporary merged dataset: {merged_tmp_root}")
        shutil.rmtree(merged_tmp_root, ignore_errors=True)
    else:
        print(f"[INFO] Kept temporary merged dataset: {merged_tmp_root}")


if __name__ == "__main__":
    main()
