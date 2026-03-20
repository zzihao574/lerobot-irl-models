import torch
import torch.nn.functional as F


def random_shifts_aug(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Random spatial shift augmentation (from DrQ-v2, used by beast_calvin)."""
    n, c, h, w = x.size()
    x = F.pad(x, [pad] * 4, mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(
        -1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device, dtype=x.dtype
    )[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(
        0, 2 * pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
    )
    shift *= 2.0 / (h + 2 * pad)
    grid = base_grid + shift
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def build_policy_prompt(
    instruction: str,
    robot_name: str,
    num_arms: int,
    action_space: str,
    include_meta: bool = True,
) -> str:
    instruction = instruction.strip()
    if include_meta:
        return (
            f"Agent Type: {num_arms}-arm {robot_name}, "
            f"Action Space: {action_space}, "
            f"Task Instruction: {instruction}"
        )
    return f"Task Instruction: {instruction}"


def create_bidirectional_mask(batch_size, seq_length, device):
    """
    In a bidirectional mask, every token can attend to every other token,
    allowing full visibility in both directions.
    
    Args:
        batch_size (int): Batch size
        seq_length (int): Sequence length (both target and source length for self-attention)
        device: Device to create tensor on
        
    Returns:
        torch.FloatTensor: Bidirectional mask with shape (batch_size, 1, seq_length, seq_length)
    """
    # For bidirectional attention, we want all positions to be visible
    # This means the mask should be all zeros (allowing attention everywhere)
    
    # Create a tensor with shape (batch_size, 1, seq_length, seq_length) filled with zeros
    # In attention masks, 0.0 means "attend to this position"
    bidirectional_mask = torch.zeros((batch_size, 1, seq_length, seq_length), device=device)
    
    return bidirectional_mask


def token_prediction_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes token-level prediction accuracy for a batch.

    Args:
        logits (torch.Tensor): Model output logits of shape (batch_size, num_classes, seq_len, ...)
        targets (torch.Tensor): Ground-truth class indices of shape (batch_size, seq_len, ...)

    Returns:
        float: Accuracy as a percentage (0-100).
    """

    # Compute accuracy
    correct = (preds == targets).sum().item()
    total = targets.numel()  # Total number of tokens

    return 100.0 * correct / total if total > 0 else 0.0  # Return accuracy as percentage
    
