from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from policies.beso.beso_config import BesoConfig  # noqa: F401
from policies.beso.modelling_beso import BesoPolicy

# Hardcoded rollout settings
VIDEO_BACKEND = "torchcodec"
INIT_PREV_GRIPPER = 0.07

log = logging.getLogger(__name__)


def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args():
    ap = argparse.ArgumentParser(description="Offline evaluate BESO rollout on an eval dataset.")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument(
        "--stats_data_dir",
        type=str,
        default=None,
        help="Optional dataset root whose meta.stats is used for normalization. "
        "Default: sibling banana_beso_train.",
    )
    ap.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Single pretrained_model dir, or run/checkpoints dir when --all-checkpoints is set.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all-checkpoints", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    return ap.parse_args()


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )


def _episode_bounds(dataset: LeRobotDataset, ep_idx: int) -> tuple[int, int]:
    start = int(dataset.meta.episodes["dataset_from_index"][ep_idx])
    end = int(dataset.meta.episodes["dataset_to_index"][ep_idx])
    return start, end


def _collect_pretrained_dirs(checkpoint_path: Path, all_checkpoints: bool) -> list[tuple[int | None, Path]]:
    if not all_checkpoints:
        if not (checkpoint_path / "model.safetensors").exists():
            raise FileNotFoundError(f"Expected model.safetensors under: {checkpoint_path}")
        step = int(checkpoint_path.parent.name) if checkpoint_path.parent.name.isdigit() else None
        return [(step, checkpoint_path)]

    if (checkpoint_path / "checkpoints").is_dir():
        checkpoints_root = checkpoint_path / "checkpoints"
    elif checkpoint_path.name == "checkpoints" and checkpoint_path.is_dir():
        checkpoints_root = checkpoint_path
    else:
        raise FileNotFoundError(
            f"When --all-checkpoints is set, pass a run dir containing checkpoints/ or checkpoints dir. "
            f"Got: {checkpoint_path}"
        )

    out: list[tuple[int | None, Path]] = []
    for child in checkpoints_root.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        pretrained_dir = child / "pretrained_model"
        if (pretrained_dir / "model.safetensors").exists():
            out.append((int(child.name), pretrained_dir))

    out.sort(key=lambda x: x[0] if x[0] is not None else -1)
    if not out:
        raise FileNotFoundError(f"No valid checkpoint/pretrained_model dirs found under {checkpoints_root}")
    return out


@dataclass
class RolloutMetricsAccumulator:
    joint_sq_sum: float = 0.0
    joint_count: int = 0
    gripper_sq_sum: float = 0.0
    gripper_count: int = 0

    cold_mse_sum: float = 0.0
    cold_count: int = 0
    steady_mse_sum: float = 0.0
    steady_count: int = 0

    num_frames: int = 0
    num_episodes: int = 0

    def add_step(self, diff: torch.Tensor, t: int, window_size: int) -> None:
        sq = diff.float().pow(2)

        self.joint_sq_sum += float(sq[:7].sum().item())
        self.joint_count += 7

        self.gripper_sq_sum += float(sq[7].item())
        self.gripper_count += 1

        step_mse = float(sq.mean().item())
        if t < window_size:
            self.cold_mse_sum += step_mse
            self.cold_count += 1
        else:
            self.steady_mse_sum += step_mse
            self.steady_count += 1

        self.num_frames += 1

    def add_episode(self):
        self.num_episodes += 1

    def finalize(self) -> dict[str, float]:
        return {
            "rollout/joint_mse": self.joint_sq_sum / self.joint_count,
            "rollout/gripper_mse": self.gripper_sq_sum / self.gripper_count,
            "rollout/mse_cold_start": self.cold_mse_sum / self.cold_count if self.cold_count > 0 else float("nan"),
            "rollout/mse_steady": self.steady_mse_sum / self.steady_count if self.steady_count > 0 else float("nan"),
            "rollout/num_frames": float(self.num_frames),
            "rollout/num_episodes": float(self.num_episodes),
        }


class BesoRolloutStateAdapter:
    def __init__(self, init_prev_gripper: float = INIT_PREV_GRIPPER):
        self.init_prev_gripper = float(init_prev_gripper)
        self.prev_gripper = self.init_prev_gripper

    def reset(self):
        self.prev_gripper = self.init_prev_gripper

    def build_obs_state(self, frame_state: torch.Tensor) -> torch.Tensor:
        out = frame_state.clone().float()
        out[7] = self.prev_gripper
        return out

    def update_from_pred_action(self, pred_action: torch.Tensor):
        self.prev_gripper = float(pred_action[7].item())


def _build_policy_input_for_frame(
    frame: dict,
    policy: BesoPolicy,
    device: torch.device,
    state_adapter: BesoRolloutStateAdapter,
) -> dict[str, torch.Tensor]:
    batch = {}
    for key in policy.config.input_features.keys():
        if key == "observation.state":
            obs_state = state_adapter.build_obs_state(frame["observation.state"])
            batch[key] = obs_state.unsqueeze(0).to(device)
        else:
            batch[key] = frame[key].unsqueeze(0).to(device)
    return batch


@torch.no_grad()
def run_episode_rollout(
    policy: BesoPolicy,
    dataset: LeRobotDataset,
    episode_idx: int,
    state_adapter: BesoRolloutStateAdapter,
    metrics: RolloutMetricsAccumulator,
    device: torch.device,
):
    start, end = _episode_bounds(dataset, episode_idx)

    policy.reset()
    state_adapter.reset()
    window_size = policy.config.window_size

    for t, abs_idx in enumerate(
        tqdm(
            range(start, end),
            total=end - start,
            desc=f"episode {episode_idx}",
            unit="frame",
            leave=False,
        )
    ):
        frame = dataset[abs_idx]
        gt_action = frame["action"].to(device=device, dtype=torch.float32)

        policy_input = _build_policy_input_for_frame(
            frame=frame,
            policy=policy,
            device=device,
            state_adapter=state_adapter,
        )

        pred_action = policy.select_action(policy_input)
        if pred_action.ndim == 2:
            pred_action = pred_action[0]
        pred_action = pred_action.to(device=device, dtype=torch.float32)

        state_adapter.update_from_pred_action(pred_action)
        metrics.add_step(diff=pred_action - gt_action, t=t, window_size=window_size)

    metrics.add_episode()


@torch.no_grad()
def run_rollout_eval(
    policy: BesoPolicy,
    dataset: LeRobotDataset,
    eval_episode_indices: list[int],
    device: torch.device,
) -> dict[str, float]:
    policy.eval()
    metrics = RolloutMetricsAccumulator()
    adapter = BesoRolloutStateAdapter(init_prev_gripper=INIT_PREV_GRIPPER)

    for ep in tqdm(eval_episode_indices, desc="episodes", unit="ep"):
        log.info("Rollout eval episode %d", ep)
        run_episode_rollout(
            policy=policy,
            dataset=dataset,
            episode_idx=ep,
            state_adapter=adapter,
            metrics=metrics,
            device=device,
        )

    return metrics.finalize()


def _load_policy(
    pretrained_dir: Path,
    dataset: LeRobotDataset,
    dataset_stats: dict,
    device: torch.device,
) -> BesoPolicy:
    policy = BesoPolicy.from_pretrained(
        pretrained_dir,
        dataset_meta=dataset.meta,
        dataset_stats=dataset_stats,
    )
    policy = policy.to(device)
    policy.eval()
    return policy


def main():
    args = _parse_args()
    _setup_logging()
    set_seed_everywhere(args.seed)

    data_dir = Path(args.data_dir)
    if args.stats_data_dir is not None:
        stats_data_dir = Path(args.stats_data_dir)
    else:
        stats_data_dir = data_dir.parent / "banana_beso_train"
    checkpoint_path = Path(args.checkpoint_path)
    device = torch.device(args.device)
    ckpts = _collect_pretrained_dirs(checkpoint_path, args.all_checkpoints)
    log.info("Evaluating %d checkpoint(s)", len(ckpts))
    all_results: list[dict] = []

    dataset = LeRobotDataset(
        repo_id=data_dir.name,
        root=data_dir,
        download_videos=True,
        video_backend=VIDEO_BACKEND,
    )
    if not stats_data_dir.exists():
        raise FileNotFoundError(f"stats_data_dir not found: {stats_data_dir}")
    stats_path = stats_data_dir / "meta" / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"stats.json not found: {stats_path}")
    with open(stats_path, "r", encoding="utf-8") as f:
        dataset_stats = json.load(f)
    eval_episodes = list(range(int(dataset.meta.total_episodes)))
    log.info("Loaded eval dataset: %s", data_dir)
    log.info("Loaded normalization stats from: %s", stats_data_dir)
    log.info("eval episodes=%s", eval_episodes)
    print(f"[INFO] normalization stats source: {stats_data_dir}")

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project="beso_eval",
            mode="online",
            group="eval_rollout",
            name="beso_eval_rollout",
            config={
                "data_dir": str(data_dir),
                "checkpoint_path": str(checkpoint_path),
                "device": str(device),
                "seed": args.seed,
                "stats_data_dir": str(stats_data_dir),
                "video_backend": VIDEO_BACKEND,
                "init_prev_gripper": INIT_PREV_GRIPPER,
                "eval_episodes": eval_episodes,
                "all_checkpoints": args.all_checkpoints,
            },
        )

    for step, pretrained_dir in tqdm(ckpts, desc="checkpoints", unit="ckpt"):
        log.info("Loading policy from %s", pretrained_dir)
        policy = _load_policy(
            pretrained_dir,
            dataset=dataset,
            dataset_stats=dataset_stats,
            device=device,
        )

        metrics = run_rollout_eval(
            policy=policy,
            dataset=dataset,
            eval_episode_indices=eval_episodes,
            device=device,
        )

        log.info("Checkpoint %s metrics:", pretrained_dir)
        for k, v in metrics.items():
            log.info("  %s = %.8f", k, v)
        print(metrics)

        if not args.all_checkpoints:
            out_path = pretrained_dir.parent / "eval_rollout_metrics.json"
            payload = {
                "checkpoint_step": step,
                "pretrained_dir": str(pretrained_dir),
                "data_dir": str(data_dir),
                "stats_data_dir": str(stats_data_dir),
                "metrics": metrics,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[INFO] wrote metrics json: {out_path}")
        else:
            all_results.append(
                {
                    "checkpoint_step": step,
                    "pretrained_dir": str(pretrained_dir),
                    "metrics": metrics,
                }
            )

        if wandb_run is not None:
            if step is None:
                wandb_run.log(metrics)
            else:
                wandb_run.log(metrics, step=step)

    if wandb_run is not None:
        wandb_run.finish()

    if args.all_checkpoints:
        if (checkpoint_path / "checkpoints").is_dir():
            checkpoints_root = checkpoint_path / "checkpoints"
        else:
            checkpoints_root = checkpoint_path
        out_path = checkpoints_root / "eval_rollout_metrics_all.json"
        payload = {
            "data_dir": str(data_dir),
            "stats_data_dir": str(stats_data_dir),
            "checkpoints_root": str(checkpoints_root),
            "results": all_results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[INFO] wrote metrics json: {out_path}")


if __name__ == "__main__":
    main()
