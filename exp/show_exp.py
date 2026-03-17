import os
import json
import pandas as pd

# 文件夹路径
folder_path = "/gemini/space/gjx/AgentEvolver/experiments/tech_synthetic/train_multibase_qwen3-vl-8b/llm_evaluation_logs/step_000102"

# 输出文件路径
output_file = "/gemini/space/gjx/AgentEvolver/exp/experience_analysis_step_000102.txt"

# 遍历 JSON 文件并读取
data_list = []
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data_list.append({
                "task_id": data.get("task_id"),
                "used_experience": data.get("used_experience"),
                "overall_score": data.get("overall_score"),
                "original_trajectory_score": data.get("original_trajectory_score")
            })

df = pd.DataFrame(data_list)

# 全局统计（兼容写法）
global_stats = df.groupby("used_experience").agg({
    "overall_score": ["count", "mean", "std", "min", "max"],
    "original_trajectory_score": ["count", "mean", "std", "min", "max"]
}).reset_index()

# task_id 聚合统计（兼容写法）
task_stats = df.groupby(["task_id", "used_experience"]).agg({
    "overall_score": ["mean", "std", "min", "max", "count"],
    "original_trajectory_score": ["mean", "std", "min", "max", "count"]
}).reset_index()

# 写入 TXT 文件
with open(output_file, "w", encoding='utf-8') as f:
    f.write("====== 全局统计（按是否使用经验） ======\n")
    f.write(global_stats.to_string(index=False))
    f.write("\n\n====== 按 task_id 聚合统计 ======\n")
    f.write(task_stats.to_string(index=False))

print(f"统计结果已保存到 {output_file}")