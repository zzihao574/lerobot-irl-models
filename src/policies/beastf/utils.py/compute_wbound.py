from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

THIS_FILE = Path(__file__).resolve()
IRL_SRC = THIS_FILE.parents[3]
ROBOT_LEARNING_ROOT = THIS_FILE.parents[4].parent
LEROBOT_SRC = ROBOT_LEARNING_ROOT / "lerobot" / "src"
BEAST_CONFIG_ROOT = ROBOT_LEARNING_ROOT / "lerobot-irl-models" / "configs" / "beast"
TRAIN_CONFIG_PATH = BEAST_CONFIG_ROOT / "train_beast.yaml"
MODEL_CONFIG_PATH = BEAST_CONFIG_ROOT / "model" / "beast.yaml"
STRIDE = 10
QUANTILE = 0.001

sys.path.insert(0, str(IRL_SRC))
sys.path.insert(0, str(LEROBOT_SRC))

from policies.beastf.beastf_config import BeastVLAConfig
from policies.beastf.beast_tokenizer.beast import BeastTokenizer


def load_cfg():
    train_cfg = OmegaConf.load(TRAIN_CONFIG_PATH)
    model_cfg_raw = OmegaConf.to_container(OmegaConf.load(MODEL_CONFIG_PATH), resolve=True)
    model_cfg_raw.pop("_target_", None)
    model_cfg = BeastVLAConfig(**model_cfg_raw)
    return train_cfg, model_cfg


def load_dataset(dataset_root: Path) -> tuple[pd.DataFrame, dict]:
    stats = json.loads((dataset_root / "meta" / "stats.json").read_text())
    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    frames = [pd.read_parquet(path, columns=["episode_index", "action"]) for path in data_files]
    data = pd.concat(frames, ignore_index=True)
    return data, stats


def compute_bounds(
    data: pd.DataFrame,
    stats: dict,
    *,
    model_cfg,
) -> dict:
    action_norm = str(model_cfg.normalization_mapping["ACTION"])
    action_mean = torch.tensor(stats["action"]["mean"], dtype=torch.float32)
    action_std = torch.tensor(stats["action"]["std"], dtype=torch.float32)
    tokenizer = BeastTokenizer(
        num_dof=int(model_cfg.num_dof),
        num_basis=int(model_cfg.num_basis),
        seq_len=int(model_cfg.act_window_size),
        vocab_size=256,
        degree_p=int(model_cfg.degree_p),
        gripper_zero_order=bool(model_cfg.gripper_zero_order),
        gripper_dof=int(model_cfg.gripper_dof),
        enforce_init_pos=bool(model_cfg.enforce_init_pos),
        device="cpu",
    )

    params_list = []
    total_windows = 0

    for _, episode in data.groupby("episode_index", sort=True):
        actions = np.stack(episode["action"].to_numpy()).astype("float32")
        actions = torch.from_numpy(actions)
        if action_norm == "MEAN_STD":
            actions = (actions - action_mean) / (action_std + 1e-8)
        elif action_norm != "IDENTITY":
            raise ValueError(f"Unsupported ACTION normalization for w_bound: {action_norm}")

        if actions.shape[0] < int(model_cfg.act_window_size):
            continue

        starts = range(0, actions.shape[0] - int(model_cfg.act_window_size) + 1, STRIDE)
        chunks = torch.stack(
            [actions[start : start + int(model_cfg.act_window_size)] for start in starts],
            dim=0,
        )
        times = tokenizer._get_repeated_times(chunks.shape[0])
        params = tokenizer._learn_trajectory_params(times, chunks)["params"]
        params_list.append(params)
        total_windows += chunks.shape[0]

    all_params = torch.cat(params_list, dim=0)
    flat = all_params.reshape(-1)
    w_min = torch.quantile(all_params, QUANTILE, dim=0)
    w_max = torch.quantile(all_params, 1.0 - QUANTILE, dim=0)

    return {
        "train_config_path": str(TRAIN_CONFIG_PATH),
        "model_config_path": str(MODEL_CONFIG_PATH),
        "action_normalization": action_norm,
        "windows": total_windows,
        "params_shape": list(all_params.shape),
        "quantile": QUANTILE,
        "global_min": float(flat.min()),
        "global_max": float(flat.max()),
        "recommended_global_w_min": float(w_min.min()),
        "recommended_global_w_max": float(w_max.max()),
        "frac_abs_gt_1": float((flat.abs() > 1).float().mean()),
        "frac_abs_gt_1_5": float((flat.abs() > 1.5).float().mean()),
        "frac_abs_gt_2": float((flat.abs() > 2).float().mean()),
        "per_dim_w_min": w_min.tolist(),
        "per_dim_w_max": w_max.tolist(),
    }


def main() -> None:
    train_cfg, model_cfg = load_cfg()
    dataset_root = Path(str(train_cfg.dataset_path)).expanduser().resolve()
    data, stats = load_dataset(dataset_root)
    result = compute_bounds(data, stats, model_cfg=model_cfg)
    result["dataset_root"] = str(dataset_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
