#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from typing import Any, Dict, List, Iterable, Tuple


def load_records(path: str) -> List[Dict[str, Any]]:
    """
    Support:
      1) JSON Lines: each line is a JSON object
      2) JSON Array: the whole file is a list of objects
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # Try JSON array first
    if text[0] == "[":
        obj = json.loads(text)
        if not isinstance(obj, list):
            raise ValueError("Top-level JSON is '[' but not a list.")
        return obj

    # Otherwise treat as JSONL
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at line {ln}: {e}") from e
            if not isinstance(rec, dict):
                raise ValueError(f"Line {ln} is not a JSON object.")
            records.append(rec)
    return records


def extract_reward(rec: Dict[str, Any], key: str) -> int:
    """
    reward could be at top-level: rec['reward']
    or nested in something like rec['content']['reward'] depending on your dump.
    We support dotted path like: "content.reward"
    """
    cur: Any = rec
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing key path '{key}' (stuck at '{part}').")
        cur = cur[part]

    # Normalize to int 0/1
    if isinstance(cur, bool):
        return int(cur)
    if isinstance(cur, (int, float)):
        if cur == 0 or cur == 1:
            return int(cur)
    raise ValueError(f"Reward value is not 0/1: {cur!r}")


def group_iter(vals: List[int], group_size: int, drop_last: bool) -> Iterable[List[int]]:
    n = len(vals)
    end = n - (n % group_size) if drop_last else n
    for i in range(0, end, group_size):
        g = vals[i:i + group_size]
        if len(g) < group_size and drop_last:
            break
        if len(g) == group_size or not drop_last:
            yield g


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser(description="Compute mean@K and best@K over reward groups (K=8 by default).")
    ap.add_argument("input", help="Path to JSON/JSONL file.")
    ap.add_argument("--group-size", type=int, default=8, help="Group size K (default: 8).")
    ap.add_argument("--reward-key", type=str, default="reward",
                    help="Reward key path, supports dotted path like 'content.reward' (default: reward).")
    ap.add_argument("--drop-last", action="store_true",
                    help="Drop last incomplete group if record count not divisible by group size.")
    ap.add_argument("--print-first", type=int, default=0,
                    help="Print first K group summaries (default: 0).")
    args = ap.parse_args()

    records = load_records(args.input)
    if not records:
        print("No records loaded.")
        return

    rewards: List[int] = []
    bad = 0
    for rec in records:
        try:
            r = extract_reward(rec, args.reward_key)
            rewards.append(r)
        except Exception:
            bad += 1

    if not rewards:
        print(f"Loaded {len(records)} records, but failed to extract any rewards. "
              f"Try --reward-key like 'content.reward'. Bad records: {bad}")
        return

    gs = args.group_size
    groups = list(group_iter(rewards, gs, args.drop_last))
    total_groups = len(groups)
    if total_groups == 0:
        print(f"Not enough rewards ({len(rewards)}) to form any group of size {gs}.")
        return

    # Compute per-group mean@K and best@K
    group_stats: List[Tuple[int, float, int]] = []
    # (group_idx, mean_at_k, best_at_k)

    mean_at_k_list: List[float] = []
    best_at_k_list: List[int] = []

    for gi, g in enumerate(groups):
        m = sum(g) / len(g)
        b = max(g)  # for 0/1 rewards: whether any success in the group
        group_stats.append((gi, m, b))
        mean_at_k_list.append(m)
        best_at_k_list.append(b)

    overall_mean_at_k = mean(mean_at_k_list)
    overall_best_at_k = mean([float(x) for x in best_at_k_list])

    used_rewards = total_groups * gs if args.drop_last else len(rewards)

    print(f"Records loaded: {len(records)}")
    print(f"Rewards extracted: {len(rewards)} (bad/missing: {bad})")
    if len(rewards) != used_rewards:
        print(f"Rewards used for grouping: {used_rewards} (group_size={gs}, drop_last={args.drop_last})")
    else:
        print(f"Rewards used for grouping: {used_rewards} (group_size={gs})")
    print(f"Total groups: {total_groups}\n")

    print("=== mean@K / best@K (K = group_size) ===")
    print(f"mean@{gs}: {overall_mean_at_k:.6f}")
    print(f"best@{gs}: {overall_best_at_k:.6f}")
    print()

    if args.print_first > 0:
        K = min(args.print_first, total_groups)
        print(f"=== First {K} Group Summaries (index, mean@{gs}, best@{gs}) ===")
        for gi, m, b in group_stats[:K]:
            print(f"group={gi:>6}  mean@{gs}={m:7.4f}  best@{gs}={b:d}")


if __name__ == "__main__":
    main()