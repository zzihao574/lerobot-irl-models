#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a clean LeRobot dataset for BESO training:
1) Merge sub-datasets in a fixed order (reindex episode_index/index/task_index correctly)
2) Add standard keys:
   - observation.state = concat(q202, prev_gripper_action202)
   - action = concat(action_q202, action_gripper202)
   - observation.goal.tail_q202 = per-episode fixed tail goal, repeated on every frame (shape [1, D_goal])
3) Remove old split keys (do not keep legacy keys)

Notes:
- This script intentionally uses LeRobot official offline dataset tools.
- rename_processor is runtime-only; it does not compute new values or fix episode collisions.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np


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
        default="/home/zzh/workspace/Robot_learning/datasets_robot",
        help="Root containing sub-datasets (place_1..place_8, freestyle folders).",
    )
    ap.add_argument(
        "--merged-tmp-root",
        type=str,
        default="/home/zzh/workspace/Robot_learning/datasets_robot/_tmp_beso_merged_raw",
        help="Temporary merged dataset root (will be created).",
    )
    ap.add_argument(
        "--output-root",
        type=str,
        default="/home/zzh/workspace/Robot_learning/datasets_robot/banana_beso_clean_v1",
        help="Final cleaned dataset root.",
    )
    ap.add_argument(
        "--output-repo-id",
        type=str,
        default="banana_beso_clean_v1",
        help="Repo id label stored in LeRobot metadata.",
    )
    ap.add_argument(
        "--tail-ratio",
        type=float,
        default=0.8,
        help="Use the last (1-tail_ratio) portion of each episode to compute goal.",
    )
    ap.add_argument(
        "--goal-reduce",
        type=str,
        default="median",
        choices=["last", "mean", "median"],
        help="How to reduce tail states into a single goal vector.",
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
    return ap.parse_args()


def _canonical_source_order(root: Path) -> list[Path]:
    # Fixed order to avoid episode_index ambiguity and to make aggregation deterministic.
    names = [f"place_{i}" for i in range(1, 9)] + ["0-9 freestyle", "10-13 freestyle"]
    paths = [root / n for n in names]

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset folders: {missing}")

    return paths


def _tail_reduce(arr: np.ndarray, tail_ratio: float, reduce: str) -> np.ndarray:
    # arr: [L, D]
    if arr.ndim != 2:
        raise ValueError(f"Expected [L, D], got {arr.shape}")
    L = arr.shape[0]
    if L == 0:
        raise ValueError("Empty episode array")

    start = int(math.floor(L * tail_ratio))
    start = min(max(start, 0), L - 1)
    tail = arr[start:]  # [L_tail, D]

    if reduce == "last":
        out = tail[-1]
    elif reduce == "mean":
        out = tail.mean(axis=0)
    elif reduce == "median":
        out = np.median(tail, axis=0)
    else:
        raise ValueError(f"Unknown reduce={reduce}")

    return out.astype(np.float32)


def _build_episode_tables(merged_dataset, tail_ratio: float, goal_reduce: str):
    """
    Precompute per-episode sequences after aggregation (episode_index already globally reindexed).

    Returns:
      episode_to_action: dict[int, np.ndarray]  # [L, 8]
      episode_to_state:  dict[int, np.ndarray]  # [L, 8]
      episode_to_goal:   dict[int, np.ndarray]  # [7]
    """
    hf = merged_dataset.hf_dataset.with_format(None)

    q202 = np.asarray(hf["observation.state.q.Panda202"], dtype=np.float32)          # [N, 7]
    aq202 = np.asarray(hf["action.q.Panda202"], dtype=np.float32)                     # [N, 7]
    g202 = np.asarray(hf["action.gripper_width.PandaGripper202"], dtype=np.float32)   # [N] or [N,1]
    ep_idx = np.asarray(hf["episode_index"], dtype=np.int64)                          # [N]
    frame_idx = np.asarray(hf["frame_index"], dtype=np.int64)                         # [N]

    if g202.ndim == 1:
        g202 = g202[:, None]  # [N, 1]
    elif g202.ndim == 2 and g202.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Unexpected gripper action shape: {g202.shape}")

    N = len(ep_idx)
    if not (len(q202) == len(aq202) == len(g202) == len(frame_idx) == N):
        raise ValueError("Column lengths mismatch in merged dataset")

    # Collect rows by episode in global order (after merge, episode_index is already unique/reindexed)
    rows_by_ep: dict[int, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = {}
    for i in range(N):
        e = int(ep_idx[i])
        f = int(frame_idx[i])
        rows_by_ep.setdefault(e, []).append((f, q202[i], aq202[i], g202[i]))

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

        q_seq = np.stack([r[1] for r in rows], axis=0).astype(np.float32)     # [L, 7]
        aq_seq = np.stack([r[2] for r in rows], axis=0).astype(np.float32)    # [L, 7]
        g_seq = np.stack([r[3] for r in rows], axis=0).astype(np.float32)     # [L, 1]

        # action = [action.q.Panda202, action.gripper_width.PandaGripper202]
        action_seq = np.concatenate([aq_seq, g_seq], axis=1)  # [L, 8]

        # prev gripper action as state feature (boundary: t=0 uses current action)
        prev_g = np.empty_like(g_seq)  # [L, 1]
        prev_g[0] = g_seq[0]
        if len(g_seq) > 1:
            prev_g[1:] = g_seq[:-1]

        # observation.state = [observation.state.q.Panda202, prev_gripper_action202]
        state_seq = np.concatenate([q_seq, prev_g], axis=1)  # [L, 8]

        # goal = tail statistic of q202 only (shape [7])
        goal_vec = _tail_reduce(q_seq, tail_ratio=tail_ratio, reduce=goal_reduce)  # [7]

        episode_to_action[e] = action_seq
        episode_to_state[e] = state_seq
        episode_to_goal[e] = goal_vec

    print(f"[INFO] Built episode tables for {len(unique_eps)} episodes")
    print(f"[INFO] Example episode 0: action shape={episode_to_action[0].shape}, state shape={episode_to_state[0].shape}, goal shape={episode_to_goal[0].shape}")
    return episode_to_action, episode_to_state, episode_to_goal


def main():
    args = _parse_args()

    # Import lerobot from local source path (avoid relying on global pip install)
    lerobot_src = Path(args.lerobot_src)
    if not lerobot_src.exists():
        raise FileNotFoundError(f"--lerobot-src not found: {lerobot_src}")
    sys.path.insert(0, str(lerobot_src))

    from lerobot.datasets.dataset_tools import merge_datasets, modify_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    datasets_root = Path(args.datasets_root)
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

    # 3) Precompute per-episode derived features on the merged dataset
    episode_to_action, episode_to_state, episode_to_goal = _build_episode_tables(
        merged_dataset,
        tail_ratio=args.tail_ratio,
        goal_reduce=args.goal_reduce,
    )

    # 4) Add new features + remove old split keys in ONE pass
    # NOTE: We use callables (not raw [N,D] arrays) because modify_features writes one parquet column
    # and direct 2D numpy assignment to a single pandas column can fail.
    def action_feature_fn(row: dict, ep_idx: int, frame_in_ep: int):
        # Returns [8]
        return episode_to_action[int(ep_idx)][int(frame_in_ep)].tolist()

    def obs_state_feature_fn(row: dict, ep_idx: int, frame_in_ep: int):
        # Returns [8]
        return episode_to_state[int(ep_idx)][int(frame_in_ep)].tolist()

    def goal_feature_fn(row: dict, ep_idx: int, frame_in_ep: int):
        # Returns [1, 7] so batch becomes [B, 1, 7] (clean B,G,D_goal interface)
        goal_vec = episode_to_goal[int(ep_idx)]  # [7]
        return [goal_vec.tolist()]               # [1, 7]

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
        args.goal_key: (
            goal_feature_fn,
            {
                "dtype": "float32",
                "shape": [1, 7],
                "names": None,
            },
        ),
    }

    # Remove legacy split keys (keep images + timestamp/frame/episode/index/task_index)
    remove_features = [
        "observation.state.q.Panda201",
        "observation.state.q.Panda202",
        "observation.state.gripper.width.PandaGripper201",
        "action.q.Panda202",
        "action.gripper_width.PandaGripper202",
    ]

    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"Final output root already exists: {output_root}")

    print(f"[INFO] Writing cleaned dataset to: {output_root}")
    clean_dataset = modify_features(
        dataset=merged_dataset,
        add_features=add_features,
        remove_features=remove_features,
        output_dir=output_root,
        repo_id=args.output_repo_id,
    )

    print("[INFO] Clean dataset created.")
    print(f"[INFO] repo_id={clean_dataset.repo_id}")
    print(f"[INFO] root={clean_dataset.root}")
    print(f"[INFO] total_episodes={clean_dataset.meta.total_episodes}, total_frames={clean_dataset.meta.total_frames}")
    print("[INFO] Final feature keys:")
    for k in clean_dataset.meta.features.keys():
        print(f"  - {k}")

    # 5) Optional cleanup of temp merged dataset
    if not args.keep_temp_merged:
        print(f"[INFO] Removing temporary merged dataset: {merged_tmp_root}")
        shutil.rmtree(merged_tmp_root, ignore_errors=True)
    else:
        print(f"[INFO] Kept temporary merged dataset: {merged_tmp_root}")


if __name__ == "__main__":
    main()
