#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import hashlib
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple


GROUP_SIZE = 8


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"读取 JSONL 失败: {path}, 第 {line_no} 行, 错误: {e}")
    return data


def normalize_reward(x) -> int:
    """
    reward 可能是 0/1, 0.0/1.0, bool
    这里统一转成 int(0或1)
    """
    try:
        v = float(x)
        return 1 if v > 0.5 else 0
    except Exception:
        return 1 if x else 0


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_user_text(text: str) -> str:
    """
    清理 user 段中的噪声：
    - 去掉 /no_think
    - 去掉 <tool_call>...</tool_call>
    - 合并空白
    """
    if not text:
        return ""
    text = re.sub(r"<tool_call>.*?</tool_call>", " ", text, flags=re.DOTALL)
    text = text.replace("/no_think", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_user_messages(input_text: str) -> List[str]:
    """
    从 input 字符串里抽取所有 user 消息文本。
    """
    if not isinstance(input_text, str):
        return []

    text = input_text.replace("\r\n", "\n").replace("\r", "\n")

    pattern = re.compile(
        r"(?:^|\n)user\s*\n(.*?)(?=(?:\nassistant\s*\n)|(?:\nuser\s*\n)|\Z)",
        flags=re.DOTALL
    )
    matches = pattern.findall(text)

    user_msgs = []
    for m in matches:
        cleaned = clean_user_text(m)
        if cleaned:
            user_msgs.append(cleaned)

    return user_msgs


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def build_task_key(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    为任务构造多个可比对的 key。
    重点使用 all_user 作为主 key。
    """
    input_text = sample.get("input", "")
    user_msgs = extract_user_messages(input_text)

    first_user = user_msgs[0] if user_msgs else ""
    all_user = " ||| ".join(user_msgs) if user_msgs else ""

    input_norm = normalize_text(input_text)

    return {
        "first_user": first_user,
        "all_user": all_user,
        "all_user_hash": sha1_text(all_user) if all_user else "",
        "input_norm_prefix": input_norm[:1000],
    }


def split_into_groups(data: List[Dict[str, Any]], group_size: int = GROUP_SIZE) -> List[List[Dict[str, Any]]]:
    if len(data) % group_size != 0:
        raise ValueError(
            f"数据条数不是 {group_size} 的倍数: {len(data)}。"
            f"这说明“相邻{group_size}个一组”的假设可能不成立，或文件不完整。"
        )
    return [data[i:i + group_size] for i in range(0, len(data), group_size)]


def summarize_group(group: List[Dict[str, Any]], group_id: int, source_name: str) -> Dict[str, Any]:
    rewards = [normalize_reward(x.get("reward", 0)) for x in group]
    key_info = build_task_key(group[0])

    same_first_user = True
    same_all_user = True
    inner_keys = []

    for item in group:
        k = build_task_key(item)
        inner_keys.append(k["all_user"])
        if k["first_user"] != key_info["first_user"]:
            same_first_user = False
        if k["all_user"] != key_info["all_user"]:
            same_all_user = False

    return {
        "group_id": group_id,
        "source_name": source_name,
        "num_correct": sum(rewards),
        "success": int(sum(rewards) > 0),
        "rewards": rewards,
        "first_user": key_info["first_user"],
        "all_user": key_info["all_user"],
        "all_user_hash": key_info["all_user_hash"],
        "input_norm_prefix": key_info["input_norm_prefix"],
        "group_internal_consistent_first_user": same_first_user,
        "group_internal_consistent_all_user": same_all_user,
        "group_internal_keys": inner_keys,
    }


def compare_task_alignment_by_order(before_groups: List[Dict[str, Any]], after_groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    仅用于诊断“如果按顺序对齐，会错多少”。
    不参与真正统计。
    """
    mismatches = []

    for i, (b, a) in enumerate(zip(before_groups, after_groups)):
        same_all_user = (b["all_user"] == a["all_user"])
        same_first_user = (b["first_user"] == a["first_user"])

        if not same_all_user and not same_first_user:
            mismatches.append({
                "task_index": i,
                "before_first_user": b["first_user"],
                "after_first_user": a["first_user"],
                "before_all_user": b["all_user"],
                "after_all_user": a["all_user"],
            })

    return {
        "num_tasks": len(before_groups),
        "num_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
    }


def build_group_map(tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    按 all_user_hash 建索引，支持重复任务。
    """
    mp = defaultdict(list)
    for t in tasks:
        key = t["all_user_hash"]
        mp[key].append(t)
    return mp


def analyze_key_distribution(before_tasks: List[Dict[str, Any]], after_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    before_map = build_group_map(before_tasks)
    after_map = build_group_map(after_tasks)

    before_only = []
    after_only = []
    count_mismatch = []

    all_keys = set(before_map.keys()) | set(after_map.keys())

    for k in all_keys:
        b = before_map.get(k, [])
        a = after_map.get(k, [])
        if b and not a:
            before_only.append({
                "key_hash": k,
                "count": len(b),
                "example_all_user": b[0]["all_user"],
            })
        elif a and not b:
            after_only.append({
                "key_hash": k,
                "count": len(a),
                "example_all_user": a[0]["all_user"],
            })
        elif len(b) != len(a):
            count_mismatch.append({
                "key_hash": k,
                "before_count": len(b),
                "after_count": len(a),
                "example_all_user": b[0]["all_user"] if b else a[0]["all_user"],
            })

    return {
        "num_unique_keys_before": len(before_map),
        "num_unique_keys_after": len(after_map),
        "num_keys_before_only": len(before_only),
        "num_keys_after_only": len(after_only),
        "num_keys_count_mismatch": len(count_mismatch),
        "before_only_examples": before_only[:20],
        "after_only_examples": after_only[:20],
        "count_mismatch_examples": count_mismatch[:20],
    }


def match_tasks_by_key(before_tasks: List[Dict[str, Any]], after_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    真正的匹配逻辑：
    - 按 all_user_hash 匹配
    - 支持重复任务：同一个 key 下按出现顺序一一配对
    - 多出来的会记为 unmatched
    """
    before_map = build_group_map(before_tasks)
    after_map = build_group_map(after_tasks)

    matched_pairs = []
    unmatched_before = []
    unmatched_after = []
    duplicate_keys = []

    all_keys = sorted(set(before_map.keys()) | set(after_map.keys()))

    for k in all_keys:
        b_list = before_map.get(k, [])
        a_list = after_map.get(k, [])

        if len(b_list) > 1 or len(a_list) > 1:
            duplicate_keys.append({
                "key_hash": k,
                "before_count": len(b_list),
                "after_count": len(a_list),
                "example_all_user": (b_list[0]["all_user"] if b_list else a_list[0]["all_user"]),
            })

        pair_n = min(len(b_list), len(a_list))

        for i in range(pair_n):
            matched_pairs.append((b_list[i], a_list[i]))

        if len(b_list) > pair_n:
            for x in b_list[pair_n:]:
                unmatched_before.append({
                    "group_id": x["group_id"],
                    "key_hash": k,
                    "all_user": x["all_user"],
                    "num_correct": x["num_correct"],
                })

        if len(a_list) > pair_n:
            for x in a_list[pair_n:]:
                unmatched_after.append({
                    "group_id": x["group_id"],
                    "key_hash": k,
                    "all_user": x["all_user"],
                    "num_correct": x["num_correct"],
                })

    return {
        "matched_pairs": matched_pairs,
        "matching_summary": {
            "num_before_tasks": len(before_tasks),
            "num_after_tasks": len(after_tasks),
            "num_matched_pairs": len(matched_pairs),
            "num_unmatched_before": len(unmatched_before),
            "num_unmatched_after": len(unmatched_after),
            "num_duplicate_keys": len(duplicate_keys),
            "duplicate_key_examples": duplicate_keys[:20],
            "unmatched_before_examples": unmatched_before[:20],
            "unmatched_after_examples": unmatched_after[:20],
        }
    }


def compute_transition_stats_from_pairs(matched_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    n = len(matched_pairs)

    transition_counter = Counter()
    delta_counter = Counter()

    improved_tasks = 0
    worsened_tasks = 0
    unchanged_tasks = 0

    success_before = 0
    success_after = 0

    traj_correct_before = 0
    traj_correct_after = 0

    from_all_wrong_to_any_correct = 0
    from_success_to_all_wrong = 0

    from_multi_correct_to_less = 0
    from_any_correct_to_less = 0
    from_one_correct_to_more = 0
    from_zero_to_multi = 0
    from_zero_to_full = 0
    from_full_to_not_full = 0

    per_task_rows = []

    for pair_idx, (b, a) in enumerate(matched_pairs):
        bc = b["num_correct"]
        ac = a["num_correct"]
        bs = b["success"]
        a_s = a["success"]

        success_before += bs
        success_after += a_s
        traj_correct_before += bc
        traj_correct_after += ac

        transition_counter[(bc, ac)] += 1
        delta_counter[ac - bc] += 1

        if ac > bc:
            improved_tasks += 1
        elif ac < bc:
            worsened_tasks += 1
        else:
            unchanged_tasks += 1

        if bc == 0 and ac >= 1:
            from_all_wrong_to_any_correct += 1

        if bs == 1 and a_s == 0:
            from_success_to_all_wrong += 1

        if bc > 1 and ac < bc:
            from_multi_correct_to_less += 1

        if bc >= 1 and ac < bc:
            from_any_correct_to_less += 1

        if bc == 1 and ac > 1:
            from_one_correct_to_more += 1

        if bc == 0 and ac >= 2:
            from_zero_to_multi += 1

        if bc == 0 and ac == 8:
            from_zero_to_full += 1

        if bc == 8 and ac < 8:
            from_full_to_not_full += 1

        per_task_rows.append({
            "pair_index": pair_idx,
            "before_group_id": b["group_id"],
            "after_group_id": a["group_id"],
            "key_hash": b["all_user_hash"],
            "before_num_correct": bc,
            "after_num_correct": ac,
            "delta_num_correct": ac - bc,
            "before_success": bs,
            "after_success": a_s,
            "first_user": b["first_user"],
            "all_user": b["all_user"],
        })

    summary = {
        "num_tasks": n,

        "success_tasks_before": success_before,
        "success_tasks_after": success_after,
        "success_tasks_delta": success_after - success_before,
        "success_task_rate_before": success_before / n if n else 0.0,
        "success_task_rate_after": success_after / n if n else 0.0,

        "traj_correct_before": traj_correct_before,
        "traj_correct_after": traj_correct_after,
        "traj_correct_delta": traj_correct_after - traj_correct_before,
        "traj_success_rate_before": traj_correct_before / (n * GROUP_SIZE) if n else 0.0,
        "traj_success_rate_after": traj_correct_after / (n * GROUP_SIZE) if n else 0.0,

        "avg_num_correct_per_task_before": traj_correct_before / n if n else 0.0,
        "avg_num_correct_per_task_after": traj_correct_after / n if n else 0.0,

        "num_tasks_correct_count_increased": improved_tasks,
        "num_tasks_correct_count_decreased": worsened_tasks,
        "num_tasks_correct_count_unchanged": unchanged_tasks,

        "num_tasks_from_all_wrong_to_any_correct": from_all_wrong_to_any_correct,
        "num_tasks_from_multi_correct_to_less": from_multi_correct_to_less,

        "num_tasks_from_any_correct_to_less": from_any_correct_to_less,
        "num_tasks_from_success_to_all_wrong": from_success_to_all_wrong,
        "num_tasks_from_one_correct_to_more": from_one_correct_to_more,
        "num_tasks_from_zero_to_multi": from_zero_to_multi,
        "num_tasks_from_zero_to_full": from_zero_to_full,
        "num_tasks_from_full_to_not_full": from_full_to_not_full,

        "delta_num_correct_distribution": {
            str(k): v for k, v in sorted(delta_counter.items(), key=lambda x: x[0])
        },
        "transition_matrix_num_correct": {
            f"{b}->{a}": c for (b, a), c in sorted(transition_counter.items(), key=lambda x: (x[0][0], x[0][1]))
        },
    }

    return {
        "summary": summary,
        "per_task": per_task_rows,
    }


def print_human_readable_report(report: Dict[str, Any]) -> None:
    order_align = report["order_alignment_diagnostic"]
    key_stats = report["key_distribution"]
    matching = report["matching"]["matching_summary"]
    summary = report["stats"]["summary"]
    meta = report["meta"]

    print("=" * 80)
    print("基础信息")
    print("=" * 80)
    print(f"before 任务数: {meta['before_num_tasks']}")
    print(f"after 任务数 : {meta['after_num_tasks']}")
    print(f"实际参与统计的 matched 任务数: {summary['num_tasks']}")

    print("\n" + "=" * 80)
    print("组内一致性检查")
    print("=" * 80)
    print(f"before 组内 query 不一致任务数: {meta['before_group_internal_inconsistent_count']}")
    print(f"after 组内 query 不一致任务数 : {meta['after_group_internal_inconsistent_count']}")

    print("\n" + "=" * 80)
    print("按顺序对齐的诊断（仅用于说明顺序是否乱了）")
    print("=" * 80)
    print(f"若按顺序 zip，对不上的任务数: {order_align['num_mismatches']} / {order_align['num_tasks']}")

    print("\n" + "=" * 80)
    print("按 key 匹配诊断")
    print("=" * 80)
    print(f"before unique key 数: {key_stats['num_unique_keys_before']}")
    print(f"after unique key 数 : {key_stats['num_unique_keys_after']}")
    print(f"只在 before 出现的 key 数: {key_stats['num_keys_before_only']}")
    print(f"只在 after 出现的 key 数 : {key_stats['num_keys_after_only']}")
    print(f"before/after 次数不一致的 key 数: {key_stats['num_keys_count_mismatch']}")
    print(f"重复 key 数: {matching['num_duplicate_keys']}")
    print(f"未匹配 before 任务数: {matching['num_unmatched_before']}")
    print(f"未匹配 after 任务数 : {matching['num_unmatched_after']}")
    print(f"成功匹配任务对数: {matching['num_matched_pairs']}")

    if matching["num_unmatched_before"] > 0:
        print("\n前几个 unmatched_before 示例:")
        for x in report["matching"]["matching_summary"]["unmatched_before_examples"][:5]:
            print(f"- group_id={x['group_id']}, num_correct={x['num_correct']}, all_user={x['all_user'][:120]}")

    if matching["num_unmatched_after"] > 0:
        print("\n前几个 unmatched_after 示例:")
        for x in report["matching"]["matching_summary"]["unmatched_after_examples"][:5]:
            print(f"- group_id={x['group_id']}, num_correct={x['num_correct']}, all_user={x['all_user'][:120]}")

    print("\n" + "=" * 80)
    print("核心统计（仅基于成功匹配的任务）")
    print("=" * 80)
    print(f"成功任务数 before: {summary['success_tasks_before']}")
    print(f"成功任务数 after : {summary['success_tasks_after']}")
    print(f"成功任务数变化    : {summary['success_tasks_delta']}")

    print()
    print(f"轨迹正确总数 before: {summary['traj_correct_before']}")
    print(f"轨迹正确总数 after : {summary['traj_correct_after']}")
    print(f"轨迹正确总数变化    : {summary['traj_correct_delta']}")

    print()
    print(f"每任务平均正确条数 before: {summary['avg_num_correct_per_task_before']:.4f}")
    print(f"每任务平均正确条数 after : {summary['avg_num_correct_per_task_after']:.4f}")

    print()
    print(f"正确条数增加的任务数: {summary['num_tasks_correct_count_increased']}")
    print(f"正确条数减少的任务数: {summary['num_tasks_correct_count_decreased']}")
    print(f"正确条数不变的任务数: {summary['num_tasks_correct_count_unchanged']}")

    print("\n" + "-" * 80)
    print("你特别关心的指标")
    print("-" * 80)
    print(f"从“全错”变成“至少一条对”的任务数: {summary['num_tasks_from_all_wrong_to_any_correct']}")
    print(f"从“原本多条正确(>1)”变成“正确条数减少”的任务数: {summary['num_tasks_from_multi_correct_to_less']}")

    print("\n" + "-" * 80)
    print("补充指标")
    print("-" * 80)
    print(f"从“原本至少一条对”变成“正确条数减少”的任务数: {summary['num_tasks_from_any_correct_to_less']}")
    print(f"从“成功任务”变成“全错任务”的数量: {summary['num_tasks_from_success_to_all_wrong']}")
    print(f"从“1条正确”变成“多条正确”的任务数: {summary['num_tasks_from_one_correct_to_more']}")
    print(f"从“0条正确”变成“2条及以上正确”的任务数: {summary['num_tasks_from_zero_to_multi']}")
    print(f"从“0条正确”直接变成“8条全对”的任务数: {summary['num_tasks_from_zero_to_full']}")
    print(f"从“8条全对”退化为“非全对”的任务数: {summary['num_tasks_from_full_to_not_full']}")

    print("\n" + "=" * 80)
    print("正确条数变化分布（after-before）")
    print("=" * 80)
    for k, v in summary["delta_num_correct_distribution"].items():
        print(f"{k:>3}: {v}")

    print("\n" + "=" * 80)
    print("0~8 正确条数转移矩阵（before->after）")
    print("=" * 80)
    for k, v in summary["transition_matrix_num_correct"].items():
        print(f"{k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="分析加经验前后 BFCL rollout 的 reward 变化（按 query 匹配任务，不按顺序）")
    parser.add_argument("--before", type=str, required=True, help="加经验前 jsonl 文件")
    parser.add_argument("--after", type=str, required=True, help="加经验后 jsonl 文件")
    parser.add_argument("--out", type=str, default=None, help="输出分析结果 json 文件路径")
    parser.add_argument("--save-per-task", type=str, default=None, help="输出每个匹配任务的详细统计 jsonl")
    args = parser.parse_args()

    before_data = read_jsonl(args.before)
    after_data = read_jsonl(args.after)

    before_groups_raw = split_into_groups(before_data, GROUP_SIZE)
    after_groups_raw = split_into_groups(after_data, GROUP_SIZE)

    before_tasks = [summarize_group(g, i, "before") for i, g in enumerate(before_groups_raw)]
    after_tasks = [summarize_group(g, i, "after") for i, g in enumerate(after_groups_raw)]

    inconsistent_before = [x["group_id"] for x in before_tasks if not x["group_internal_consistent_all_user"]]
    inconsistent_after = [x["group_id"] for x in after_tasks if not x["group_internal_consistent_all_user"]]

    order_alignment_diagnostic = compare_task_alignment_by_order(before_tasks, after_tasks)
    key_distribution = analyze_key_distribution(before_tasks, after_tasks)
    matching = match_tasks_by_key(before_tasks, after_tasks)
    stats = compute_transition_stats_from_pairs(matching["matched_pairs"])

    report = {
        "meta": {
            "before_file": os.path.abspath(args.before),
            "after_file": os.path.abspath(args.after),
            "group_size": GROUP_SIZE,
            "before_num_records": len(before_data),
            "after_num_records": len(after_data),
            "before_num_tasks": len(before_tasks),
            "after_num_tasks": len(after_tasks),
            "before_group_internal_inconsistent_count": len(inconsistent_before),
            "after_group_internal_inconsistent_count": len(inconsistent_after),
            "before_group_internal_inconsistent_examples": inconsistent_before[:20],
            "after_group_internal_inconsistent_examples": inconsistent_after[:20],
        },
        "order_alignment_diagnostic": order_alignment_diagnostic,
        "key_distribution": key_distribution,
        "matching": matching,
        "stats": stats,
    }

    # matched_pairs 太大，不适合直接完整写入 report json
    report_for_save = {
        **report,
        "matching": {
            "matching_summary": matching["matching_summary"]
        }
    }

    print_human_readable_report(report)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report_for_save, f, ensure_ascii=False, indent=2)
        print(f"\n已保存汇总结果到: {args.out}")

    if args.save_per_task:
        with open(args.save_per_task, "w", encoding="utf-8") as f:
            for row in stats["per_task"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"已保存每任务详细结果到: {args.save_per_task}")


if __name__ == "__main__":
    main()