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
    ap.add_argument(
        "--eval-mode",
        type=str,
        default="both",
        choices=["rollout", "one_step", "both"],
    )
    ap.add_argument(
        "--state7-mode",
        type=str,
        default="pred",
        choices=["pred", "gt"],
    )
    ap.add_argument(
        "--gt-shift",
        type=int,
        default=0,
        choices=[-1, 0, 1],
        help="Compare pred at t with gt at t+gt_shift.",
    )
    ap.add_argument(
        "--scan-gt-shift",
        action="store_true",
        help="Run all gt shifts in {-1,0,+1} and report each.",
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


def _shift_tag(gt_shift: int) -> str:
    if gt_shift < 0:
        return f"m{abs(gt_shift)}"
    return f"p{gt_shift}"


def _get_shifted_gt_action(
    dataset: LeRobotDataset,
    start: int,
    end: int,
    abs_idx: int,
    gt_shift: int,
    device: torch.device,
    frame_at_abs_idx: dict | None = None,
) -> torch.Tensor | None:
    gt_idx = abs_idx + gt_shift
    if gt_idx < start or gt_idx >= end:
        return None
    if gt_shift == 0 and frame_at_abs_idx is not None:
        gt_frame = frame_at_abs_idx
    else:
        gt_frame = dataset[gt_idx]
    return gt_frame["action"].to(device=device, dtype=torch.float32)


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


class BesoRolloutStateAdapter:
    def __init__(self, mode: str, init_prev_gripper: float = INIT_PREV_GRIPPER):
        self.mode = mode
        self.init_prev_gripper = float(init_prev_gripper)
        self.prev_gripper = self.init_prev_gripper

    def reset(self):
        self.prev_gripper = self.init_prev_gripper

    def build_obs_state(self, frame_state: torch.Tensor) -> torch.Tensor:
        if self.mode == "gt":
            return frame_state.float()
        out = frame_state.clone().float()
        out[7] = self.prev_gripper
        return out

    def update_from_pred_action(self, pred_action: torch.Tensor):
        if self.mode == "gt":
            return
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
    gt_shift: int,
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
        gt_action = _get_shifted_gt_action(
            dataset=dataset,
            start=start,
            end=end,
            abs_idx=abs_idx,
            gt_shift=gt_shift,
            device=device,
            frame_at_abs_idx=frame,
        )
        if gt_action is None:
            continue
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
    state7_mode: str,
    device: torch.device,
    gt_shift: int,
) -> dict[str, float]:
    policy.eval()
    metrics = RolloutMetricsAccumulator(prefix="rollout")
    adapter = BesoRolloutStateAdapter(mode=state7_mode, init_prev_gripper=INIT_PREV_GRIPPER)

    for ep in tqdm(eval_episode_indices, desc="episodes", unit="ep"):
        log.info("Rollout eval episode %d", ep)
        run_episode_rollout(
            policy=policy,
            dataset=dataset,
            episode_idx=ep,
            state_adapter=adapter,
            metrics=metrics,
            device=device,
            gt_shift=gt_shift,
        )

    return metrics.finalize()


@torch.no_grad()
def run_episode_one_step(
    policy: BesoPolicy,
    dataset: LeRobotDataset,
    episode_idx: int,
    state7_mode: str,
    metrics: RolloutMetricsAccumulator,
    device: torch.device,
    gt_shift: int,
):
    start, end = _episode_bounds(dataset, episode_idx)
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
        hist_start = max(start, abs_idx - window_size + 1)

        policy.reset()
        state_adapter = BesoRolloutStateAdapter(mode=state7_mode, init_prev_gripper=INIT_PREV_GRIPPER)

        pred_action = None
        frame_at_abs_idx = None
        for j in range(hist_start, abs_idx + 1):
            frame = dataset[j]
            if j == abs_idx:
                frame_at_abs_idx = frame
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

        gt_action = _get_shifted_gt_action(
            dataset=dataset,
            start=start,
            end=end,
            abs_idx=abs_idx,
            gt_shift=gt_shift,
            device=device,
            frame_at_abs_idx=frame_at_abs_idx,
        )
        if gt_action is None:
            continue
        metrics.add_step(
            diff=pred_action - gt_action,
            t=t,
            window_size=window_size,
            pred_action=pred_action,
            gt_action=gt_action,
        )

    metrics.add_episode()


@torch.no_grad()
def run_one_step_eval(
    policy: BesoPolicy,
    dataset: LeRobotDataset,
    eval_episode_indices: list[int],
    state7_mode: str,
    device: torch.device,
    gt_shift: int,
) -> dict[str, float]:
    policy.eval()
    metrics = RolloutMetricsAccumulator(prefix="one_step")

    for ep in tqdm(eval_episode_indices, desc="episodes(one_step)", unit="ep"):
        log.info("One-step eval episode %d", ep)
        run_episode_one_step(
            policy=policy,
            dataset=dataset,
            episode_idx=ep,
            state7_mode=state7_mode,
            metrics=metrics,
            device=device,
            gt_shift=gt_shift,
        )

    return metrics.finalize()


@torch.no_grad()
def run_constant_baseline_eval(
    dataset: LeRobotDataset,
    eval_episode_indices: list[int],
    baseline_action: torch.Tensor,
    window_size: int,
    device: torch.device,
    gt_shift: int,
) -> dict[str, float]:
    metrics = RolloutMetricsAccumulator(prefix="baseline")
    baseline_action = baseline_action.to(device=device, dtype=torch.float32)

    for ep in tqdm(eval_episode_indices, desc="episodes(baseline)", unit="ep"):
        start, end = _episode_bounds(dataset, ep)
        for t, abs_idx in enumerate(
            tqdm(
                range(start, end),
                total=end - start,
                desc=f"episode {ep}",
                unit="frame",
                leave=False,
            )
        ):
            frame = dataset[abs_idx]
            gt_action = _get_shifted_gt_action(
                dataset=dataset,
                start=start,
                end=end,
                abs_idx=abs_idx,
                gt_shift=gt_shift,
                device=device,
                frame_at_abs_idx=frame,
            )
            if gt_action is None:
                continue
            pred_action = baseline_action
            metrics.add_step(
                diff=pred_action - gt_action,
                t=t,
                window_size=window_size,
                pred_action=pred_action,
                gt_action=gt_action,
            )
        metrics.add_episode()

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
    gt_shifts = [-1, 0, 1] if args.scan_gt_shift else [args.gt_shift]
    log.info("eval_mode=%s state7_mode=%s gt_shifts=%s", args.eval_mode, args.state7_mode, gt_shifts)
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
    baseline_action = torch.as_tensor(dataset_stats["action"]["mean"], dtype=torch.float32, device=device)
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
                "eval_mode": args.eval_mode,
                "state7_mode": args.state7_mode,
                "gt_shift": args.gt_shift,
                "scan_gt_shift": args.scan_gt_shift,
            },
        )

    baseline_metrics_cache: dict[int, dict[str, float]] = {}
    for step, pretrained_dir in tqdm(ckpts, desc="checkpoints", unit="ckpt"):
        log.info("Loading policy from %s", pretrained_dir)
        policy = _load_policy(
            pretrained_dir,
            dataset=dataset,
            dataset_stats=dataset_stats,
            device=device,
        )
        metrics: dict[str, float] = {}
        metrics_by_shift: dict[str, dict[str, float]] = {}
        for gt_shift in gt_shifts:
            if gt_shift not in baseline_metrics_cache:
                baseline_metrics_cache[gt_shift] = run_constant_baseline_eval(
                    dataset=dataset,
                    eval_episode_indices=eval_episodes,
                    baseline_action=baseline_action,
                    window_size=policy.config.window_size,
                    device=device,
                    gt_shift=gt_shift,
                )

            shift_metrics: dict[str, float] = {}
            if args.eval_mode in {"rollout", "both"}:
                shift_metrics.update(
                    run_rollout_eval(
                        policy=policy,
                        dataset=dataset,
                        eval_episode_indices=eval_episodes,
                        state7_mode=args.state7_mode,
                        device=device,
                        gt_shift=gt_shift,
                    )
                )
            shift_metrics.update(baseline_metrics_cache[gt_shift])
            if args.eval_mode in {"one_step", "both"}:
                shift_metrics.update(
                    run_one_step_eval(
                        policy=policy,
                        dataset=dataset,
                        eval_episode_indices=eval_episodes,
                        state7_mode=args.state7_mode,
                        device=device,
                        gt_shift=gt_shift,
                    )
                )
            metrics_by_shift[str(gt_shift)] = shift_metrics
            if len(gt_shifts) == 1:
                metrics = shift_metrics
            else:
                tag = _shift_tag(gt_shift)
                for k, v in shift_metrics.items():
                    metrics[f"shift_{tag}/{k}"] = v

        if len(gt_shifts) > 1:
            if args.eval_mode in {"rollout", "both"}:
                score_key = "rollout/joint_mse"
            else:
                score_key = "one_step/joint_mse"
            valid_pairs = []
            for s in gt_shifts:
                v = metrics_by_shift[str(s)].get(score_key, float("nan"))
                if not np.isnan(v):
                    valid_pairs.append((s, v))
            if valid_pairs:
                best_shift, best_score = min(valid_pairs, key=lambda x: x[1])
                metrics["summary/best_gt_shift"] = float(best_shift)
                metrics[f"summary/best_{score_key}"] = float(best_score)

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
                "eval_mode": args.eval_mode,
                "state7_mode": args.state7_mode,
                "gt_shift": args.gt_shift,
                "scan_gt_shift": args.scan_gt_shift,
                "metrics": metrics,
            }
            if len(gt_shifts) > 1:
                payload["metrics_by_shift"] = metrics_by_shift
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[INFO] wrote metrics json: {out_path}")
        else:
            result_item = {
                "checkpoint_step": step,
                "pretrained_dir": str(pretrained_dir),
                "metrics": metrics,
            }
            if len(gt_shifts) > 1:
                result_item["metrics_by_shift"] = metrics_by_shift
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
            "eval_mode": args.eval_mode,
            "state7_mode": args.state7_mode,
            "gt_shift": args.gt_shift,
            "scan_gt_shift": args.scan_gt_shift,
            "results": all_results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[INFO] wrote metrics json: {out_path}")


if __name__ == "__main__":
    main()
