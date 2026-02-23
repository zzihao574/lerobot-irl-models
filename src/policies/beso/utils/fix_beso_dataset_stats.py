#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lerobot-src",
        type=str,
        default="/home/zzh/workspace/Robot_learning/lerobot/src",
    )
    ap.add_argument(
        "--dataset-root",
        type=str,
        default="/home/zzh/workspace/Robot_learning/datasets_robot/banana_beso_clean_v1",
    )
    ap.add_argument(
        "--repo-id",
        type=str,
        default="banana_beso_clean_v1",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, args.lerobot_src)

    from lerobot.datasets.compute_stats import compute_episode_stats, aggregate_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import write_stats

    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=Path(args.dataset_root),
        download_videos=False,
    )
    hf = ds.hf_dataset.with_format(None)

    target_keys = [
        "action",
        "observation.state",
        "observation.goal.tail_q202",
    ]

    for k in target_keys:
        if k not in ds.meta.features:
            raise ValueError(f"Missing feature in dataset: {k}")

    # Only compute stats for our new vector features (skip videos to save time)
    feature_desc = {k: ds.meta.features[k] for k in target_keys}

    ep_stats_list = []
    total_eps = ds.meta.total_episodes

    for ep in range(total_eps):
        from_idx = int(ds.meta.episodes["dataset_from_index"][ep])
        to_idx = int(ds.meta.episodes["dataset_to_index"][ep])

        ep_ds = hf.select(range(from_idx, to_idx))
        ep_data = {}

        for k in target_keys:
            arr = np.asarray(ep_ds[k], dtype=np.float32)
            ep_data[k] = arr

        ep_stats = compute_episode_stats(ep_data, feature_desc)
        ep_stats_list.append(ep_stats)

        if ep < 3:
            print(f"[debug] ep={ep} lens={to_idx-from_idx} "
                  f"action_shape={ep_data['action'].shape} "
                  f"state_shape={ep_data['observation.state'].shape} "
                  f"goal_shape={ep_data['observation.goal.tail_q202'].shape}")

    new_stats = aggregate_stats(ep_stats_list)

    # Merge into existing stats.json (preserve image / index stats)
    merged_stats = dict(ds.meta.stats) if ds.meta.stats is not None else {}
    merged_stats.update(new_stats)

    write_stats(merged_stats, ds.root)
    print("[ok] stats.json updated.")
    print("[ok] added/updated keys:", list(new_stats.keys()))


if __name__ == "__main__":
    main()
