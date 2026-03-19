from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
import torch


def compute_turn_level_entropy(
    token_entropy: torch.Tensor,
    turn_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> Tuple[List[List[float]], Dict[int, int]]:
    """
    Compute mean token entropy for each valid turn in every trajectory.
    """
    batch_size = token_entropy.shape[0]
    turn_entropies: List[List[float]] = []
    turn_counts: Dict[int, int] = {}

    for i in range(batch_size):
        valid_mask = (turn_ids[i] >= 0) & (response_mask[i] > 0)
        if not valid_mask.any():
            turn_entropies.append([])
            turn_counts[i] = 0
            continue

        valid_turn_ids = turn_ids[i][valid_mask]
        valid_entropy = token_entropy[i][valid_mask]
        unique_turns = torch.unique(valid_turn_ids, sorted=True)

        trajectory_turn_entropies: List[float] = []
        for turn_id in unique_turns:
            turn_mask = valid_turn_ids == turn_id
            if turn_mask.any():
                trajectory_turn_entropies.append(valid_entropy[turn_mask].mean().item())

        turn_entropies.append(trajectory_turn_entropies)
        turn_counts[i] = len(trajectory_turn_entropies)

    return turn_entropies, turn_counts


def compute_turn_level_credit(turn_entropies: List[List[float]]) -> List[List[float]]:
    """
    r_temp[t] = entropy[t - 1] - entropy[t], and r_temp[0] = -entropy[0].
    """
    turn_credits: List[List[float]] = []

    for trajectory_entropies in turn_entropies:
        if not trajectory_entropies:
            turn_credits.append([])
            continue

        credits: List[float] = []
        for t, entropy in enumerate(trajectory_entropies):
            if t == 0:
                credits.append(-entropy)
            else:
                credits.append(trajectory_entropies[t - 1] - entropy)
        turn_credits.append(credits)

    return turn_credits


def normalize_turn_level_credit(
    turn_credits: List[List[float]],
    group_ids: Union[np.ndarray, torch.Tensor, List[int]],
    epsilon: float = 1e-6,
) -> List[List[float]]:
    """
    Group-wise z-score normalization over all turns in the same group.
    """
    if isinstance(group_ids, torch.Tensor):
        gid_list = group_ids.view(-1).detach().cpu().tolist()
    elif isinstance(group_ids, np.ndarray):
        gid_list = group_ids.reshape(-1).tolist()
    else:
        gid_list = list(group_ids)

    group2traj: Dict[str, List[int]] = defaultdict(list)
    for i, gid in enumerate(gid_list):
        group2traj[str(gid)].append(i)

    normalized_credits: List[List[float]] = [[] for _ in range(len(turn_credits))]

    for traj_indices in group2traj.values():
        all_credits: List[float] = []
        for idx in traj_indices:
            all_credits.extend(turn_credits[idx])

        if not all_credits:
            continue

        credits_tensor = torch.tensor(all_credits, dtype=torch.float32)
        mean = credits_tensor.mean()
        std = credits_tensor.std(unbiased=False)

        for idx in traj_indices:
            if not turn_credits[idx]:
                continue
            traj_credits = torch.tensor(turn_credits[idx], dtype=torch.float32)
            normalized = (traj_credits - mean) / (std + epsilon)
            normalized_credits[idx] = normalized.tolist()

    return normalized_credits


def compute_turn_level_metrics(
    turn_entropies: List[List[float]],
    turn_counts: Dict[int, int],
    normalized_credits: List[List[float]],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    if turn_counts:
        metrics["turn_credit/avg_turns_per_trajectory"] = (
            sum(turn_counts.values()) / len(turn_counts)
        )

    all_entropies = [entropy for traj in turn_entropies for entropy in traj]
    if all_entropies:
        metrics["turn_credit/avg_entropy_per_turn"] = sum(all_entropies) / len(all_entropies)

    all_norm_credits = [credit for traj in normalized_credits for credit in traj]
    if all_norm_credits:
        metrics["turn_credit/avg_normalized_credit"] = sum(all_norm_credits) / len(all_norm_credits)

    return metrics


def broadcast_turn_credit_to_tokens(
    turn_credits: List[List[float]],
    turn_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len = turn_ids.shape
    token_credits = torch.zeros((batch_size, seq_len), device=turn_ids.device, dtype=torch.float32)

    for i in range(batch_size):
        if not turn_credits[i]:
            continue

        valid_mask = (turn_ids[i] >= 0) & (response_mask[i] > 0)
        if not valid_mask.any():
            continue

        valid_turn_ids = turn_ids[i][valid_mask]
        valid_positions = torch.where(valid_mask)[0]

        for turn_id, credit in enumerate(turn_credits[i]):
            turn_mask = valid_turn_ids == turn_id
            if turn_mask.any():
                token_credits[i, valid_positions[turn_mask]] = credit

    return token_credits


def compute_turn_level_advantage(
    token_entropy: torch.Tensor,
    turn_ids: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: Union[np.ndarray, torch.Tensor, List[int]],
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute turn-level advantage from entropy reduction and broadcast it to tokens.
    """
    target_len = token_entropy.size(1)
    if response_mask.size(1) != target_len:
        raise ValueError(
            f"response_mask length {response_mask.size(1)} does not match token_entropy length {target_len}"
        )

    if turn_ids.size(1) > target_len:
        turn_ids = turn_ids[:, :target_len]
    elif turn_ids.size(1) < target_len:
        pad = torch.full(
            (turn_ids.size(0), target_len - turn_ids.size(1)),
            -1,
            device=turn_ids.device,
            dtype=turn_ids.dtype,
        )
        turn_ids = torch.cat([turn_ids, pad], dim=1)

    turn_entropies, turn_counts = compute_turn_level_entropy(
        token_entropy=token_entropy,
        turn_ids=turn_ids,
        response_mask=response_mask,
    )
    turn_credits = compute_turn_level_credit(turn_entropies)
    normalized_credits = normalize_turn_level_credit(
        turn_credits=turn_credits,
        group_ids=group_ids,
        epsilon=epsilon,
    )
    token_level_advantage = broadcast_turn_credit_to_tokens(
        turn_credits=normalized_credits,
        turn_ids=turn_ids,
        response_mask=response_mask,
    )
    metrics = compute_turn_level_metrics(
        turn_entropies=turn_entropies,
        turn_counts=turn_counts,
        normalized_credits=normalized_credits,
    )
    return token_level_advantage, metrics
