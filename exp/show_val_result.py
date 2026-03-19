import os
import json
import numpy as np

# 文件夹路径
folder_path = "/gemini/space/gjx/AgentEvolver/experiments/tech_synthetic/bfcl_qwen3-vl-4b_agentevolver_w_nav/validation_log"



# 遍历文件夹下所有 jsonl 文件
for filename in os.listdir(folder_path):
    if not filename.endswith(".jsonl"):
        continue

    file_path = os.path.join(folder_path, filename)
    rewards = []

    # 逐行读取 JSONL
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                rewards.append(item.get("reward", 0))
            except json.JSONDecodeError:
                print(f"解析错误: {filename} 行: {line[:50]}...")
                continue

    # 按每 8 个一组
    best8_list = []
    mean8_list = []
    for i in range(0, len(rewards), 8):
        group = rewards[i:i+8]
        if len(group) < 8:
            continue  # 不满 8 个的最后一组忽略
        best8_list.append(max(group))
        mean8_list.append(np.mean(group))

    if best8_list and mean8_list:
        file_best8 = np.mean(best8_list)
        file_mean8 = np.mean(mean8_list)
        print(f"{filename} -> best@8: {file_best8:.4f}, mean@8: {file_mean8:.4f}")
    else:
        print(f"{filename} -> 数据不足 8 个，无法统计")