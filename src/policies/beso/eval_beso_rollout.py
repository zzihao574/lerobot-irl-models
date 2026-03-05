from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from policies.beso.beso_config import BesoConfig, BESO_CONFIG_NAME  # noqa: F401
from policies.beso.modelling_beso import BesoPolicy

# Hardcoded rollout setting
VIDEO_BACKEND = "torchcodec"

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
    ap.add_argument(
        "--use-non-ema",
        action="store_true",
        help="Load model_non_ema.safetensors instead of model.safetensors.",
    )
    ap.add_argument(
        "--sampling-steps",
        type=int,
        default=None,
        help="Override DDIM sampling steps at eval time (independent of training config). "
             "E.g. --sampling-steps 10 for higher quality inference.",
    )
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


def _collect_pretrained_dirs(
    checkpoint_path: Path, all_checkpoints: bool, use_non_ema: bool
) -> list[tuple[int | None, Path]]:
    weight_name = "model_non_ema.safetensors" if use_non_ema else "model.safetensors"
    if not all_checkpoints:
        if not (checkpoint_path / weight_name).exists():
            raise FileNotFoundError(f"Expected {weight_name} under: {checkpoint_path}")
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
        if (pretrained_dir / weight_name).exists():
            out.append((int(child.name), pretrained_dir))

    out.sort(key=lambda x: x[0] if x[0] is not None else -1)
    if not out:
        raise FileNotFoundError(f"No valid checkpoint/pretrained_model dirs found under {checkpoints_root}")
    return out


def _infer_pos_emb_seq_len_from_checkpoint(weight_file: Path) -> int | None:
    if not weight_file.exists():
        return None
    key = "diffusion.dit_backbone.pos_emb"
    with safe_open(str(weight_file), framework="pt", device="cpu") as f:
        if key not in f.keys():
            return None
        shape = tuple(f.get_tensor(key).shape)
    if len(shape) != 3:
        return None
    return int(shape[1])


def _infer_goal_feature_from_cfg(cfg) -> str | None:
    input_features = getattr(cfg, "input_features", None)
    if input_features is None:
        return None
    for k in input_features.keys():
        if str(k).startswith("observation.goal"):
            return str(k)
    return None


def _align_cfg_with_checkpoint_arch(cfg, weight_file: Path) -> None:
    pos_seq_len = _infer_pos_emb_seq_len_from_checkpoint(weight_file)
    n_obs_steps = getattr(cfg, "n_obs_steps", None)
    if pos_seq_len is None or n_obs_steps is None:
        return

    n_obs_steps = int(n_obs_steps)
    if pos_seq_len <= n_obs_steps:
        if hasattr(cfg, "goal_conditioned"):
            cfg.goal_conditioned = False
        return

    goal_tokens = pos_seq_len - n_obs_steps
    if hasattr(cfg, "goal_conditioned"):
        cfg.goal_conditioned = True
    if getattr(cfg, "goal_feature", None) in (None, ""):
        goal_feature = _infer_goal_feature_from_cfg(cfg)
        if goal_feature is not None:
            cfg.goal_feature = goal_feature

    log.info(
        "Aligned ckpt arch: n_obs_steps=%d, pos_emb_seq_len=%d, goal_tokens=%d, goal_feature=%s",
        n_obs_steps,
        pos_seq_len,
        goal_tokens,
        getattr(cfg, "goal_feature", None),
    )


@dataclass
class RolloutMetricsAccumulator:
    prefix: str
    joint_sq_sum: float = 0.0
    joint_count: int = 0
    gripper_sq_sum: float = 0.0
    gripper_count: int = 0

    cold_mse_sum: float = 0.0
    cold_count: int = 0
    steady_mse_sum: float = 0.0
    steady_count: int = 0

    t0_mse_sum: float = 0.0
    t0_count: int = 0
    t1_mse_sum: float = 0.0
    t1_count: int = 0
    t2_mse_sum: float = 0.0
    t2_count: int = 0
    t3_mse_sum: float = 0.0
    t3_count: int = 0
    t_ge4_mse_sum: float = 0.0
    t_ge4_count: int = 0

    joint_sq_sum_per_dim: list[float] = field(default_factory=lambda: [0.0] * 7)

    pred_sum: list[float] = field(default_factory=lambda: [0.0] * 8)
    pred_sq_sum: list[float] = field(default_factory=lambda: [0.0] * 8)
    gt_sum: list[float] = field(default_factory=lambda: [0.0] * 8)
    gt_sq_sum: list[float] = field(default_factory=lambda: [0.0] * 8)
    var_count: int = 0

    num_frames: int = 0
    num_episodes: int = 0

    def add_step(
        self,
        diff: torch.Tensor,
        t: int,
        window_size: int,
        pred_action: torch.Tensor | None = None,
        gt_action: torch.Tensor | None = None,
    ) -> None:
        sq = diff.float().pow(2)

        self.joint_sq_sum += float(sq[:7].sum().item())
        self.joint_count += 7
        for i in range(7):
            self.joint_sq_sum_per_dim[i] += float(sq[i].item())

        self.gripper_sq_sum += float(sq[7].item())
        self.gripper_count += 1

        step_mse = float(sq.mean().item())
        if t < window_size:
            self.cold_mse_sum += step_mse
            self.cold_count += 1
        else:
            self.steady_mse_sum += step_mse
            self.steady_count += 1
        if t == 0:
            self.t0_mse_sum += step_mse
            self.t0_count += 1
        elif t == 1:
            self.t1_mse_sum += step_mse
            self.t1_count += 1
        elif t == 2:
            self.t2_mse_sum += step_mse
            self.t2_count += 1
        elif t == 3:
            self.t3_mse_sum += step_mse
            self.t3_count += 1
        else:
            self.t_ge4_mse_sum += step_mse
            self.t_ge4_count += 1

        if gt_action is not None:
            gt = gt_action.float()
            pred = pred_action.float() if pred_action is not None else (gt + diff.float())
            for i in range(8):
                pv = float(pred[i].item())
                gv = float(gt[i].item())
                self.pred_sum[i] += pv
                self.pred_sq_sum[i] += pv * pv
                self.gt_sum[i] += gv
                self.gt_sq_sum[i] += gv * gv
            self.var_count += 1

        self.num_frames += 1

    def add_episode(self):
        self.num_episodes += 1

    @staticmethod
    def _safe_div(x: float, n: int) -> float:
        return x / n if n > 0 else float("nan")

    @staticmethod
    def _var(sum_x: float, sum_x2: float, n: int) -> float:
        if n <= 0:
            return float("nan")
        mean = sum_x / n
        return max(0.0, sum_x2 / n - mean * mean)

    def finalize(self) -> dict[str, float]:
        out = {
            f"{self.prefix}/joint_mse": self._safe_div(self.joint_sq_sum, self.joint_count),
            f"{self.prefix}/gripper_mse": self._safe_div(self.gripper_sq_sum, self.gripper_count),
            f"{self.prefix}/mse_cold_start": self._safe_div(self.cold_mse_sum, self.cold_count),
            f"{self.prefix}/mse_steady": self._safe_div(self.steady_mse_sum, self.steady_count),
            f"{self.prefix}/mse_t0": self._safe_div(self.t0_mse_sum, self.t0_count),
            f"{self.prefix}/mse_t1": self._safe_div(self.t1_mse_sum, self.t1_count),
            f"{self.prefix}/mse_t2": self._safe_div(self.t2_mse_sum, self.t2_count),
            f"{self.prefix}/mse_t3": self._safe_div(self.t3_mse_sum, self.t3_count),
            f"{self.prefix}/mse_t_ge4": self._safe_div(self.t_ge4_mse_sum, self.t_ge4_count),
            f"{self.prefix}/num_frames": float(self.num_frames),
            f"{self.prefix}/num_episodes": float(self.num_episodes),
        }
        for i in range(7):
            out[f"{self.prefix}/joint_mse_j{i}"] = self._safe_div(self.joint_sq_sum_per_dim[i], self.num_frames)

        pred_var = [self._var(self.pred_sum[i], self.pred_sq_sum[i], self.var_count) for i in range(8)]
        gt_var = [self._var(self.gt_sum[i], self.gt_sq_sum[i], self.var_count) for i in range(8)]
        out[f"{self.prefix}/pred_var_joint_mean"] = float(np.mean(pred_var[:7]))
        out[f"{self.prefix}/gt_var_joint_mean"] = float(np.mean(gt_var[:7]))
        out[f"{self.prefix}/pred_var_gripper"] = pred_var[7]
        out[f"{self.prefix}/gt_var_gripper"] = gt_var[7]
        return out


def _build_policy_input_for_frame(
    frame: dict,
    policy: BesoPolicy,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: frame[key].unsqueeze(0).to(device) for key in policy.config.input_features.keys()}


@torch.no_grad()
def run_episode_rollout(
    policy: BesoPolicy,
    dataset: LeRobotDataset,
    episode_idx: int,
    metrics: RolloutMetricsAccumulator,
    device: torch.device,
):
    start, end = _episode_bounds(dataset, episode_idx)

    policy.reset()
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

        policy_input = _build_policy_input_for_frame(
            frame=frame,
            policy=policy,
            device=device,
        )

        pred_action = policy.select_action(policy_input)
        if pred_action.ndim == 2:
            pred_action = pred_action[0]
        pred_action = pred_action.to(device=device, dtype=torch.float32)
        gt_action = frame["action"].to(device=device, dtype=torch.float32)
        metrics.add_step(
            diff=pred_action - gt_action,
            t=t,
            window_size=window_size,
            pred_action=pred_action,
            gt_action=gt_action,
        )

    metrics.add_episode()


@torch.no_grad()
def run_rollout_eval(
    policy: BesoPolicy,
    dataset: LeRobotDataset,
    eval_episode_indices: list[int],
    device: torch.device,
) -> dict[str, float]:
    policy.eval()
    metrics = RolloutMetricsAccumulator(prefix="rollout")

    for ep in tqdm(eval_episode_indices, desc="episodes", unit="ep"):
        log.info("Rollout eval episode %d", ep)
        run_episode_rollout(
            policy=policy,
            dataset=dataset,
            episode_idx=ep,
            metrics=metrics,
            device=device,
        )

    return metrics.finalize()


@torch.no_grad()
def _load_policy(
    pretrained_dir: Path,
    dataset: LeRobotDataset,
    dataset_stats: dict,
    device: torch.device,
    use_non_ema: bool = False,
) -> BesoPolicy:
    weight_file = (
        pretrained_dir / "model_non_ema.safetensors"
        if use_non_ema
        else pretrained_dir / "model.safetensors"
    )
    if not weight_file.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_file}")

    cfg = PreTrainedConfig.from_pretrained(pretrained_dir)
    # Load BESO-specific fields that draccus doesn't serialize to config.json.
    beso_cfg_file = pretrained_dir / BESO_CONFIG_NAME
    if beso_cfg_file.exists():
        with open(beso_cfg_file) as _f:
            _extra = json.load(_f)
        for _k, _v in _extra.items():
            if _v is not None or not hasattr(cfg, _k):
                setattr(cfg, _k, _v)
        log.info("Loaded BESO extra config from %s: %s", BESO_CONFIG_NAME,
                 {k: _extra[k] for k in ("sigma_data", "sigma_max", "sigma_min", "sampling_steps") if k in _extra})
    else:
        log.warning("No %s found in %s — BESO sigma/arch params will use BesoConfig defaults", BESO_CONFIG_NAME, pretrained_dir)
    cfg.device = str(device)
    _align_cfg_with_checkpoint_arch(cfg, weight_file)
    # resize_shape fallback: infer from pos_grid in weights for old checkpoints without beso_config.json.
    if not beso_cfg_file.exists():
        with safe_open(str(weight_file), framework="pt") as _sf:
            _pg = next((k for k in _sf.keys() if k.endswith(".pool.pos_grid")), None)
            if _pg is not None:
                _n = _sf.get_tensor(_pg).shape[0]
                _side = round(_n ** 0.5)
                cfg.resize_shape = (_side * 32, _side * 32) if _side * _side == _n else None
            else:
                cfg.resize_shape = None
    log.info(
        "Load cfg resolved: n_obs_steps=%s window_size=%s goal_conditioned=%s goal_feature=%s weights=%s",
        getattr(cfg, "n_obs_steps", None),
        getattr(cfg, "window_size", None),
        getattr(cfg, "goal_conditioned", None),
        getattr(cfg, "goal_feature", None),
        weight_file.name,
    )
    policy = BesoPolicy(config=cfg, dataset_meta=dataset.meta, dataset_stats=dataset_stats)
    policy = BesoPolicy._load_as_safetensor(policy, str(weight_file), str(device), strict=False)
    policy = policy.to(device)
    policy.eval()
    return policy


def _override_sampling_steps(policy: "BesoPolicy", sampling_steps: int) -> None:
    """Override DDIM sampling steps after model load (eval-time only)."""
    policy.config.sampling_steps = sampling_steps
    log.info("[override] sampling_steps → %d", sampling_steps)


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
    ckpts = _collect_pretrained_dirs(checkpoint_path, args.all_checkpoints, args.use_non_ema)
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
            use_non_ema=args.use_non_ema,
        )
        if args.sampling_steps is not None:
            _override_sampling_steps(policy, args.sampling_steps)
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
                "use_non_ema": args.use_non_ema,
                "metrics": metrics,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[INFO] wrote metrics json: {out_path}")
        else:
            result_item = {
                "checkpoint_step": step,
                "pretrained_dir": str(pretrained_dir),
                "use_non_ema": args.use_non_ema,
                "metrics": metrics,
            }
            all_results.append(result_item)

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
            "use_non_ema": args.use_non_ema,
            "results": all_results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[INFO] wrote metrics json: {out_path}")


if __name__ == "__main__":
    main()
