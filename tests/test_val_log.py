#!/usr/bin/env python3
"""
分析验证集日志，找出重复 prompt 的原因
- 检查是任务本身重复（数据集中有重复）
- 还是 tokenize/decode 导致的不同 prompt 变成相同文本
"""

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_validation_log(log_file: str) -> List[Dict[str, Any]]:
    """加载验证集日志文件（JSONL 格式）"""
    samples = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                continue
    return samples


def extract_prompt_from_input(input_text: str) -> str:
    """
    从 input 中提取实际的 prompt 文本
    根据你提供的格式，input 包含 system message 和 tools 定义
    需要提取 user 的实际查询
    """
    # 查找 user 标签后的内容
    if "user\n" in input_text:
        user_part = input_text.split("user\n")[-1]
        # 移除 /no_think 等标记
        user_part = user_part.split("/no_think")[0].strip()
        return user_part
    return input_text


def analyze_duplicates(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析重复情况"""
    
    # 1. 统计原始 input 的重复情况
    raw_inputs = [sample.get("input", "") for sample in samples]
    raw_input_counter = Counter(raw_inputs)
    raw_duplicates = {inp: count for inp, count in raw_input_counter.items() if count > 1}
    
    # 2. 提取并统计 prompt 文本的重复情况
    prompts = []
    for sample in samples:
        input_text = sample.get("input", "")
        prompt = extract_prompt_from_input(input_text)
        prompts.append(prompt)
    
    prompt_counter = Counter(prompts)
    prompt_duplicates = {prompt: count for prompt, count in prompt_counter.items() if count > 1}
    
    # 3. 分析：哪些是真正的数据重复，哪些是 decode 导致的
    # 对于每个重复的 prompt，检查对应的原始 input 是否也重复
    analysis = {
        "true_duplicates": [],  # 原始 input 就重复的
        "decode_duplicates": [],  # 原始 input 不同但 decode 后相同的
        "ambiguous": []  # 需要进一步检查的
    }
    
    # 按 prompt 分组
    prompt_to_inputs = defaultdict(list)
    for i, (sample, prompt) in enumerate(zip(samples, prompts)):
        raw_input = sample.get("input", "")
        prompt_to_inputs[prompt].append((i, raw_input))
    
    for prompt, input_list in prompt_to_inputs.items():
        if len(input_list) > 1:  # 这个 prompt 有重复
            # 检查对应的原始 input 是否也重复
            raw_inputs_for_prompt = [inp for _, inp in input_list]
            unique_raw_inputs = set(raw_inputs_for_prompt)
            
            if len(unique_raw_inputs) == 1:
                # 原始 input 完全相同 -> 真正的数据重复
                analysis["true_duplicates"].append({
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "count": len(input_list),
                    "raw_input_hash": hash(raw_inputs_for_prompt[0]) % 1000000,
                    "indices": [idx for idx, _ in input_list]
                })
            elif len(unique_raw_inputs) > 1:
                # 原始 input 不同但 prompt 相同 -> decode 导致的重复
                analysis["decode_duplicates"].append({
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "count": len(input_list),
                    "num_unique_inputs": len(unique_raw_inputs),
                    "indices": [idx for idx, _ in input_list],
                    "input_samples": [inp[:200] + "..." if len(inp) > 200 else inp 
                                     for inp in list(unique_raw_inputs)[:3]]  # 只显示前3个
                })
    
    # 4. 统计信息
    stats = {
        "total_samples": len(samples),
        "unique_raw_inputs": len(set(raw_inputs)),
        "unique_prompts": len(set(prompts)),
        "raw_input_duplicates": len(raw_duplicates),
        "prompt_duplicates": len(prompt_duplicates),
        "true_duplicate_count": len(analysis["true_duplicates"]),
        "decode_duplicate_count": len(analysis["decode_duplicates"]),
        "total_duplicate_samples": sum(count - 1 for count in raw_input_counter.values() if count > 1),
    }
    
    return {
        "stats": stats,
        "analysis": analysis,
        "raw_input_counter": dict(raw_input_counter.most_common(10)),  # 前10个最常见的
        "prompt_counter": dict(prompt_counter.most_common(10)),  # 前10个最常见的
    }


def print_analysis_report(result: Dict[str, Any], output_file: str = None):
    """打印分析报告"""
    stats = result["stats"]
    analysis = result["analysis"]
    
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("验证集重复情况分析报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 统计信息
    report_lines.append("【统计信息】")
    report_lines.append(f"  总样本数: {stats['total_samples']}")
    report_lines.append(f"  唯一原始 input 数: {stats['unique_raw_inputs']}")
    report_lines.append(f"  唯一 prompt 文本数: {stats['unique_prompts']}")
    report_lines.append(f"  原始 input 重复数: {stats['raw_input_duplicates']}")
    report_lines.append(f"  Prompt 文本重复数: {stats['prompt_duplicates']}")
    report_lines.append(f"  总重复样本数: {stats['total_duplicate_samples']}")
    report_lines.append("")
    
    # 重复率
    if stats['total_samples'] > 0:
        duplicate_rate = stats['total_duplicate_samples'] / stats['total_samples'] * 100
        report_lines.append(f"  重复率: {duplicate_rate:.2f}%")
        report_lines.append("")
    
    # 分析结果
    report_lines.append("【重复类型分析】")
    report_lines.append(f"  真正的数据重复（原始 input 相同）: {stats['true_duplicate_count']} 个 prompt")
    report_lines.append(f"  Decode 导致的重复（原始 input 不同但 prompt 相同）: {stats['decode_duplicate_count']} 个 prompt")
    report_lines.append("")
    
    # 详细分析
    if analysis["true_duplicates"]:
        report_lines.append("【真正的数据重复示例】（前5个）")
        for i, dup in enumerate(analysis["true_duplicates"][:5], 1):
            report_lines.append(f"  {i}. Prompt (出现 {dup['count']} 次):")
            report_lines.append(f"     {dup['prompt']}")
            report_lines.append(f"     样本索引: {dup['indices'][:5]}..." if len(dup['indices']) > 5 else f"     样本索引: {dup['indices']}")
            report_lines.append("")
    
    if analysis["decode_duplicates"]:
        report_lines.append("【Decode 导致的重复示例】（前5个）")
        for i, dup in enumerate(analysis["decode_duplicates"][:5], 1):
            report_lines.append(f"  {i}. Prompt (出现 {dup['count']} 次，来自 {dup['num_unique_inputs']} 个不同的原始 input):")
            report_lines.append(f"     {dup['prompt']}")
            report_lines.append(f"     样本索引: {dup['indices'][:5]}..." if len(dup['indices']) > 5 else f"     样本索引: {dup['indices']}")
            report_lines.append("     原始 input 示例:")
            for j, inp_sample in enumerate(dup['input_samples'], 1):
                report_lines.append(f"       {j}. {inp_sample[:150]}...")
            report_lines.append("")
    
    # 最常见的重复
    if result["prompt_counter"]:
        report_lines.append("【最常见的重复 Prompt】（前10个）")
        for prompt, count in list(result["prompt_counter"].items())[:10]:
            report_lines.append(f"  出现 {count} 次: {prompt[:80]}...")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # 打印到控制台
    print(report_text)
    
    # 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n报告已保存到: {output_file}")
    
    return report_text


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python analyze_validation_duplicates.py <validation_log_file> [output_report_file]")
        print("示例: python analyze_validation_duplicates.py validation_log.jsonl report.txt")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(log_file).exists():
        print(f"错误: 文件不存在: {log_file}")
        sys.exit(1)
    
    print(f"正在加载验证集日志: {log_file}")
    samples = load_validation_log(log_file)
    print(f"加载了 {len(samples)} 个样本")
    
    print("正在分析重复情况...")
    result = analyze_duplicates(samples)
    
    print_analysis_report(result, output_file)


if __name__ == "__main__":
    main()

