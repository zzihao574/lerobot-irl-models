#!/usr/bin/env python3
"""
Comprehensive validator for banana_beso_train / banana_beso_eval datasets
produced by build_clean_beso_dataset.py.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/home/zzh/workspace/Robot_learning/dataset_new")
TRAIN_ROOT = ROOT / "banana_beso_train"
EVAL_ROOT = ROOT / "banana_beso_eval"

# ── Source datasets (in canonical merge order) ──
SOURCE_NAMES = [f"place_{i}" for i in range(1, 9)] + ["0-4_freestyle", "5-9_freestyle"]

PASS = 0
FAIL = 0

def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")
    return cond


def load_info(root):
    with open(root / "meta" / "info.json") as f:
        return json.load(f)


def load_stats(root):
    with open(root / "meta" / "stats.json") as f:
        return json.load(f)


def load_episodes_parquet(root):
    ep_dir = root / "meta" / "episodes"
    files = sorted(ep_dir.rglob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_data_parquet(root):
    data_dir = root / "data"
    files = sorted(data_dir.rglob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def get_source_info():
    """Return list of (name, total_episodes, total_frames) for each source."""
    result = []
    for name in SOURCE_NAMES:
        info = load_info(ROOT / name)
        result.append((name, info["total_episodes"], info["total_frames"]))
    return result


# ═══════════════════════════════════════════════════════════════
# 1. Basic metadata checks
# ═══════════════════════════════════════════════════════════════
def check_metadata(root, label):
    print(f"\n{'='*60}")
    print(f"  CHECKING METADATA: {label}")
    print(f"{'='*60}")
    info = load_info(root)

    # Check expected features exist
    expected_feats = {
        "action", "observation.state", "observation.goal.tail_q202",
        "observation.image.centric_cam", "observation.image.wrist_cam",
        "timestamp", "frame_index", "episode_index", "index", "task_index",
    }
    actual_feats = set(info["features"].keys())
    check(expected_feats.issubset(actual_feats),
          f"All expected features present. actual={sorted(actual_feats)}")

    # Check removed features are gone
    removed = {
        "observation.state.q.Panda201",
        "observation.state.gripper_width.PandaGripper201",
        "observation.state.q.Panda202",
        "observation.state.gripper_width.PandaGripper202",
        "action.q.Panda202",
        "action.gripper_width.PandaGripper202",
    }
    leftovers = removed & actual_feats
    check(len(leftovers) == 0, f"Removed features are gone. leftovers={leftovers}")

    # Check shapes
    check(info["features"]["action"]["shape"] == [8],
          f"action shape=[8], got {info['features']['action']['shape']}")
    check(info["features"]["observation.state"]["shape"] == [8],
          f"observation.state shape=[8], got {info['features']['observation.state']['shape']}")
    check(info["features"]["observation.goal.tail_q202"]["shape"] == [1, 7],
          f"goal shape=[1,7], got {info['features']['observation.goal.tail_q202']['shape']}")

    return info


# ═══════════════════════════════════════════════════════════════
# 2. Episode indexing & frame continuity
# ═══════════════════════════════════════════════════════════════
def check_episodes(root, label, info):
    print(f"\n{'='*60}")
    print(f"  CHECKING EPISODES: {label}")
    print(f"{'='*60}")

    df = load_data_parquet(root)
    n_frames = len(df)
    n_episodes = info["total_episodes"]
    total_frames_meta = info["total_frames"]

    check(n_frames == total_frames_meta,
          f"Parquet rows ({n_frames}) == info.total_frames ({total_frames_meta})")

    ep_idx = df["episode_index"].values.astype(int)
    frame_idx = df["frame_index"].values.astype(int)
    idx_col = df["index"].values.astype(int)

    unique_eps = sorted(set(ep_idx))
    check(unique_eps == list(range(n_episodes)),
          f"episode_index contiguous 0..{n_episodes-1}. unique count={len(unique_eps)}")

    # Check global index is contiguous 0..N-1
    expected_idx = list(range(n_frames))
    actual_idx = idx_col.tolist()
    check(actual_idx == expected_idx,
          f"Global 'index' column contiguous 0..{n_frames-1}")

    # Per-episode frame_index continuity
    all_frame_ok = True
    ep_lengths = {}
    for ep in unique_eps:
        mask = ep_idx == ep
        frames = frame_idx[mask]
        expected = list(range(len(frames)))
        if frames.tolist() != expected:
            all_frame_ok = False
            print(f"    Episode {ep}: frame_index NOT contiguous, head={frames[:10].tolist()}")
        ep_lengths[ep] = len(frames)

    check(all_frame_ok, "All episodes have contiguous frame_index starting from 0")

    # Check episodes parquet
    ep_df = load_episodes_parquet(root)
    check(len(ep_df) == n_episodes,
          f"Episodes parquet rows ({len(ep_df)}) == total_episodes ({n_episodes})")

    # Check from/to indices in episodes parquet
    if "dataset_from_index" in ep_df.columns and "dataset_to_index" in ep_df.columns:
        from_to_ok = True
        running_idx = 0
        for i in range(n_episodes):
            from_idx = int(ep_df["dataset_from_index"].iloc[i])
            to_idx = int(ep_df["dataset_to_index"].iloc[i])
            ep_len = ep_lengths.get(i, 0)
            if from_idx != running_idx:
                from_to_ok = False
                print(f"    Episode {i}: from_idx={from_idx} expected={running_idx}")
            if to_idx != running_idx + ep_len:
                from_to_ok = False
                print(f"    Episode {i}: to_idx={to_idx} expected={running_idx + ep_len}")
            running_idx += ep_len
        check(from_to_ok, "Episodes from/to indices are consistent with data")
    else:
        print("  [SKIP] No dataset_from_index/dataset_to_index in episodes parquet")

    return df, ep_lengths


# ═══════════════════════════════════════════════════════════════
# 3. Data value checks (action, state, goal)
# ═══════════════════════════════════════════════════════════════
def check_data_values(root, label, df, ep_lengths, info):
    print(f"\n{'='*60}")
    print(f"  CHECKING DATA VALUES: {label}")
    print(f"{'='*60}")

    n_episodes = info["total_episodes"]
    ep_idx = df["episode_index"].values.astype(int)

    # Check action shape
    actions = np.stack(df["action"].values)
    check(actions.shape == (len(df), 8),
          f"action data shape={actions.shape}, expected ({len(df)}, 8)")

    # Check state shape
    states = np.stack(df["observation.state"].values)
    check(states.shape == (len(df), 8),
          f"observation.state data shape={states.shape}, expected ({len(df)}, 8)")

    # Check goal shape - goals stored as nested object arrays: each is ndarray(shape=(1,), dtype=object)
    # containing one 7-dim float32 array
    raw_goals = df["observation.goal.tail_q202"].values
    sample = raw_goals[0]
    print(f"    Goal sample: shape={getattr(sample, 'shape', 'N/A')}, dtype={getattr(sample, 'dtype', 'N/A')}")
    
    # Extract the inner 7-dim vectors: raw_goals[i] is ndarray([ndarray([7 floats])])
    goals = np.stack([np.array(g[0], dtype=np.float32) for g in raw_goals])  # [N, 7]
    check(goals.shape == (len(df), 7),
          f"goal inner data shape={goals.shape}, expected ({len(df)}, 7)")
    # Reshape for downstream checks that expect [N, 1, 7]
    goals_3d = goals.reshape(len(df), 1, 7)

    # Check no NaN/Inf in these fields
    check(not np.any(np.isnan(actions)), "No NaN in actions")
    check(not np.any(np.isinf(actions)), "No Inf in actions")
    check(not np.any(np.isnan(states)), "No NaN in states")
    check(not np.any(np.isinf(states)), "No Inf in states")
    check(not np.any(np.isnan(goals)), "No NaN in goals")
    check(not np.any(np.isinf(goals)), "No Inf in goals")
    # Use goals_3d for shape-dependent checks below

    # Check goal is constant within each episode
    goal_const_ok = True
    for ep in range(n_episodes):
        mask = ep_idx == ep
        ep_goals = goals[mask]  # [L, 7]
        if ep_goals.shape[0] == 0:
            goal_const_ok = False
            print(f"    Episode {ep}: empty!")
            continue
        first = ep_goals[0]
        if not np.allclose(ep_goals, first[None, :], atol=1e-6):
            goal_const_ok = False
            diffs = np.max(np.abs(ep_goals - first[None, :]), axis=1)
            n_diff = np.sum(diffs > 1e-6)
            print(f"    Episode {ep}: goal NOT constant, {n_diff}/{len(ep_goals)} frames differ, max_diff={diffs.max():.6f}")
    check(goal_const_ok, "Goal vector constant within each episode")

    # Verify goal computation (tail median of q202 = first 7 dims of observation.state)
    goal_compute_ok = True
    tail_ratio = 0.9
    for ep in range(n_episodes):
        mask = ep_idx == ep
        ep_states = states[mask]  # [L, 8]
        ep_q202 = ep_states[:, :7]  # [L, 7] - first 7 dims are q.Panda202
        ep_goal = goals[mask][0]  # [7]

        L = len(ep_q202)
        start = int(math.floor(L * tail_ratio))
        start = min(max(start, 0), L - 1)
        tail = ep_q202[start:]
        expected_goal = np.median(tail, axis=0).astype(np.float32)

        if not np.allclose(ep_goal, expected_goal, atol=1e-5):
            goal_compute_ok = False
            diff = np.max(np.abs(ep_goal - expected_goal))
            print(f"    Episode {ep}: goal mismatch, max_diff={diff:.8f}")
            print(f"      got:      {ep_goal[:4]}...")
            print(f"      expected: {expected_goal[:4]}...")
    check(goal_compute_ok, "Goal = median of tail 10% of q202 (verified)")

    return actions, states, goals, goals_3d


# ═══════════════════════════════════════════════════════════════
# 4. Stats.json validation
# ═══════════════════════════════════════════════════════════════
def check_stats(root, label, df, actions, states, goals, info):
    print(f"\n{'='*60}")
    print(f"  CHECKING STATS.JSON: {label}")
    print(f"{'='*60}")

    stats = load_stats(root)
    n_episodes = info["total_episodes"]
    ep_idx = df["episode_index"].values.astype(int)

    # Check stats for derived features
    for feat_name, data in [("action", actions), ("observation.state", states)]:
        if feat_name not in stats:
            check(False, f"stats.json has key '{feat_name}'")
            continue
        check(True, f"stats.json has key '{feat_name}'")

        fs = stats[feat_name]
        
        # Compute expected stats
        actual_mean = data.mean(axis=0).astype(np.float32)
        actual_std = data.std(axis=0).astype(np.float32)
        actual_min = data.min(axis=0).astype(np.float32)
        actual_max = data.max(axis=0).astype(np.float32)

        stored_mean = np.array(fs["mean"], dtype=np.float32)
        stored_std = np.array(fs["std"], dtype=np.float32)
        stored_min = np.array(fs["min"], dtype=np.float32)
        stored_max = np.array(fs["max"], dtype=np.float32)

        # Note: stats may use per-episode aggregation (not raw global), so tolerate some diff
        # But min/max should be exact
        check(np.allclose(stored_min, actual_min, atol=1e-4),
              f"  {feat_name} min close to actual. max_diff={np.max(np.abs(stored_min - actual_min)):.6f}")
        check(np.allclose(stored_max, actual_max, atol=1e-4),
              f"  {feat_name} max close to actual. max_diff={np.max(np.abs(stored_max - actual_max)):.6f}")
        # Mean and std may diverge slightly with per-episode aggregation
        mean_diff = np.max(np.abs(stored_mean - actual_mean))
        std_diff = np.max(np.abs(stored_std - actual_std))
        check(mean_diff < 0.05,
              f"  {feat_name} mean diff={mean_diff:.6f} (per-ep aggregation tolerance)")
        check(std_diff < 0.05,
              f"  {feat_name} std diff={std_diff:.6f} (per-ep aggregation tolerance)")

    # Check goal stats
    goal_key = "observation.goal.tail_q202"
    if goal_key in stats:
        check(True, f"stats.json has key '{goal_key}'")
        gs = stats[goal_key]
        stored_goal_min = np.array(gs["min"], dtype=np.float32)
        stored_goal_max = np.array(gs["max"], dtype=np.float32)
        actual_goal_min = goals.min(axis=0).astype(np.float32)
        actual_goal_max = goals.max(axis=0).astype(np.float32)
        check(np.allclose(stored_goal_min, actual_goal_min, atol=1e-4),
              f"  {goal_key} min close. diff={np.max(np.abs(stored_goal_min - actual_goal_min)):.6f}")
        check(np.allclose(stored_goal_max, actual_goal_max, atol=1e-4),
              f"  {goal_key} max close. diff={np.max(np.abs(stored_goal_max - actual_goal_max)):.6f}")
    else:
        check(False, f"stats.json has key '{goal_key}'")

    # Check index-type features in stats
    for idx_feat in ["episode_index", "index", "frame_index", "timestamp", "task_index"]:
        if idx_feat in stats:
            fs = stats[idx_feat]
            if idx_feat == "episode_index":
                expected_min = 0.0
                expected_max = float(n_episodes - 1)
                check(float(fs["min"][0]) == expected_min,
                      f"  {idx_feat} min={fs['min'][0]} expected={expected_min}")
                check(float(fs["max"][0]) == expected_max,
                      f"  {idx_feat} max={fs['max'][0]} expected={expected_max}")
            elif idx_feat == "index":
                n_frames = len(df)
                check(float(fs["min"][0]) == 0.0,
                      f"  {idx_feat} min={fs['min'][0]} expected=0")
                check(float(fs["max"][0]) == float(n_frames - 1),
                      f"  {idx_feat} max={fs['max'][0]} expected={n_frames - 1}")

    # Check no removed feature stats remain
    removed_feats = [
        "observation.state.q.Panda201",
        "observation.state.gripper_width.PandaGripper201",
        "observation.state.q.Panda202",
        "observation.state.gripper_width.PandaGripper202",
        "action.q.Panda202",
        "action.gripper_width.PandaGripper202",
    ]
    leftover_stats = [f for f in removed_feats if f in stats]
    check(len(leftover_stats) == 0,
          f"No removed feature stats remain. leftovers={leftover_stats}")

    return stats


# ═══════════════════════════════════════════════════════════════
# 5. Per-episode stats in episodes parquet
# ═══════════════════════════════════════════════════════════════
def check_episode_stats_parquet(root, label, df, actions, states, goals, info):
    print(f"\n{'='*60}")
    print(f"  CHECKING PER-EPISODE STATS (parquet): {label}")
    print(f"{'='*60}")

    ep_df = load_episodes_parquet(root)
    ep_idx = df["episode_index"].values.astype(int)
    n_episodes = info["total_episodes"]

    # Check that per-episode stats columns exist for key features
    for feat in ["action", "observation.state", "observation.goal.tail_q202"]:
        for stat_type in ["mean", "std", "min", "max"]:
            col_name = f"stats/{feat}/{stat_type}"
            check(col_name in ep_df.columns,
                  f"Column '{col_name}' exists in episodes parquet")

    # Check removed feature stats columns are gone
    removed_feats = [
        "observation.state.q.Panda201",
        "observation.state.gripper_width.PandaGripper201", 
        "observation.state.q.Panda202",
        "observation.state.gripper_width.PandaGripper202",
        "action.q.Panda202", 
        "action.gripper_width.PandaGripper202",
    ]
    orphan_cols = []
    for feat in removed_feats:
        for stat_type in ["mean", "std", "min", "max"]:
            col = f"stats/{feat}/{stat_type}"
            if col in ep_df.columns:
                orphan_cols.append(col)
    check(len(orphan_cols) == 0,
          f"No orphaned stats columns for removed features. orphans={orphan_cols}")

    # Validate a sample of per-episode action min/max
    sample_ok = True
    for ep in range(min(n_episodes, 5)):
        mask = ep_idx == ep
        ep_actions = actions[mask]
        actual_min = ep_actions.min(axis=0)
        actual_max = ep_actions.max(axis=0)
        
        stored_min = np.array(ep_df["stats/action/min"].iloc[ep], dtype=np.float32)
        stored_max = np.array(ep_df["stats/action/max"].iloc[ep], dtype=np.float32)
        
        if not np.allclose(stored_min, actual_min, atol=1e-4):
            sample_ok = False
            print(f"    Ep {ep} action min mismatch: {np.max(np.abs(stored_min - actual_min)):.6f}")
        if not np.allclose(stored_max, actual_max, atol=1e-4):
            sample_ok = False
            print(f"    Ep {ep} action max mismatch: {np.max(np.abs(stored_max - actual_max)):.6f}")
    check(sample_ok, f"Per-episode action min/max correct (checked first {min(n_episodes, 5)} eps)")


# ═══════════════════════════════════════════════════════════════
# 6. Train/Eval split correctness
# ═══════════════════════════════════════════════════════════════
def check_split(source_info):
    print(f"\n{'='*60}")
    print(f"  CHECKING TRAIN/EVAL SPLIT")
    print(f"{'='*60}")

    train_info = load_info(TRAIN_ROOT)
    eval_info = load_info(EVAL_ROOT)

    total_src_eps = sum(ep for _, ep, _ in source_info)  # 59
    n_sources = len(source_info)  # 10

    check(eval_info["total_episodes"] == n_sources,
          f"Eval has {eval_info['total_episodes']} episodes (expected {n_sources}, one per source)")
    check(train_info["total_episodes"] == total_src_eps - n_sources,
          f"Train has {train_info['total_episodes']} episodes (expected {total_src_eps - n_sources})")

    total_train_frames = train_info["total_frames"]
    total_eval_frames = eval_info["total_frames"]
    # Can't exactly expect sum because we don't have the merged raw anymore,
    # but train_frames + eval_frames should equal total source frames
    # Actually NO - frames were processed at original fps, no resampling in this script
    print(f"  Train frames: {total_train_frames}")
    print(f"  Eval frames:  {total_eval_frames}")
    total_merged = total_train_frames + total_eval_frames
    total_src_frames = sum(fr for _, _, fr in source_info)
    check(total_merged == total_src_frames,
          f"Train+Eval frames ({total_merged}) == source total ({total_src_frames})")

    # Check split manifests
    for split_root, split_name in [(TRAIN_ROOT, "train"), (EVAL_ROOT, "eval")]:
        manifest_path = split_root / "split_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            check(manifest["num_episodes"] == (train_info if split_name == "train" else eval_info)["total_episodes"],
                  f"{split_name} manifest num_episodes matches info.total_episodes")
            
            # Verify old episodes don't overlap between train and eval
            if split_name == "train":
                train_old_eps = set(manifest["new_to_old_episode_index"])
            else:
                eval_old_eps = set(manifest["new_to_old_episode_index"])
        else:
            check(False, f"{split_name} split_manifest.json exists")

    if 'train_old_eps' in dir() and 'eval_old_eps' in dir():
        overlap = train_old_eps & eval_old_eps
        check(len(overlap) == 0,
              f"No overlap between train/eval old episodes. overlap={overlap}")
        combined = train_old_eps | eval_old_eps
        check(combined == set(range(total_src_eps)),
              f"Train+Eval old episodes cover all {total_src_eps} source episodes")

        # Check eval has exactly 1 episode per source group
        # Reconstruct which source group each merged episode belongs to
        ep_to_source = {}
        offset = 0
        for i, (name, n_eps, _) in enumerate(source_info):
            for j in range(n_eps):
                ep_to_source[offset + j] = i
            offset += n_eps

        eval_sources = [ep_to_source[e] for e in eval_old_eps]
        check(len(eval_sources) == len(set(eval_sources)),
              f"Each source has exactly 1 eval episode (no duplicates). sources={eval_sources}")
        check(set(eval_sources) == set(range(n_sources)),
              f"All {n_sources} sources represented in eval")


# ═══════════════════════════════════════════════════════════════
# 7. Video file existence check
# ═══════════════════════════════════════════════════════════════
def check_videos(root, label, info, ep_lengths):
    print(f"\n{'='*60}")
    print(f"  CHECKING VIDEOS: {label}")
    print(f"{'='*60}")

    video_keys = []
    for feat_name, feat_desc in info["features"].items():
        if isinstance(feat_desc, dict) and feat_desc.get("dtype") in ["video", "image"]:
            video_keys.append(feat_name)

    if not video_keys:
        print("  No video features found, skipping.")
        return

    print(f"  Video keys: {video_keys}")
    video_dir = root / "videos"
    n_episodes = info["total_episodes"]

    for vk in video_keys:
        # Video dirs use the feature key as-is (with dots), e.g. videos/observation.image.centric_cam/
        video_root = root / "videos" / vk
        if video_root.exists():
            vk_files = sorted(video_root.rglob("*.mp4"))
        else:
            vk_files = []
        check(len(vk_files) >= 1,
              f"Video files for '{vk}': found {len(vk_files)} file(s) in {video_root}")


# ═══════════════════════════════════════════════════════════════
# 8. Timestamp monotonicity per episode
# ═══════════════════════════════════════════════════════════════
def check_timestamps(df, label, info):
    print(f"\n{'='*60}")
    print(f"  CHECKING TIMESTAMPS: {label}")
    print(f"{'='*60}")

    ep_idx = df["episode_index"].values.astype(int)
    timestamps = df["timestamp"].values.astype(float)
    n_episodes = info["total_episodes"]

    mono_ok = True
    for ep in range(n_episodes):
        mask = ep_idx == ep
        ep_ts = timestamps[mask]
        if len(ep_ts) < 2:
            continue
        diffs = np.diff(ep_ts)
        if not np.all(diffs > 0):
            mono_ok = False
            neg_count = np.sum(diffs <= 0)
            print(f"    Episode {ep}: {neg_count} non-monotonic timestamp transitions")
    check(mono_ok, "All episode timestamps are strictly monotonically increasing")

    # Check timestamps start from 0 for each episode
    start_zero_ok = True
    for ep in range(n_episodes):
        mask = ep_idx == ep
        ep_ts = timestamps[mask]
        if len(ep_ts) > 0 and abs(ep_ts[0]) > 0.05:
            start_zero_ok = False
            print(f"    Episode {ep}: starts at t={ep_ts[0]:.4f}")
    check(start_zero_ok, "All episodes start with timestamp ~0")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    global PASS, FAIL

    print("=" * 60)
    print("  BESO DATASET VALIDATION")
    print("=" * 60)

    source_info = get_source_info()
    print("\nSource datasets:")
    total_src_eps = 0
    total_src_frames = 0
    for name, n_eps, n_frames in source_info:
        print(f"  {name:20s}  eps={n_eps}  frames={n_frames}")
        total_src_eps += n_eps
        total_src_frames += n_frames
    print(f"  {'TOTAL':20s}  eps={total_src_eps}  frames={total_src_frames}")

    # Check split
    check_split(source_info)

    # Check each dataset
    for root, label in [(TRAIN_ROOT, "TRAIN"), (EVAL_ROOT, "EVAL")]:
        info = check_metadata(root, label)
        df, ep_lengths = check_episodes(root, label, info)
        actions, states, goals, goals_3d = check_data_values(root, label, df, ep_lengths, info)
        check_stats(root, label, df, actions, states, goals_3d, info)
        check_episode_stats_parquet(root, label, df, actions, states, goals_3d, info)
        check_videos(root, label, info, ep_lengths)
        check_timestamps(df, label, info)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    print(f"{'='*60}")
    if FAIL > 0:
        print("  *** SOME CHECKS FAILED ***")
        sys.exit(1)
    else:
        print("  ALL CHECKS PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()
