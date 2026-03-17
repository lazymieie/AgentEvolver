import os
import json
import requests
from time import sleep
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

# 文件路径设置
folder_path = "/gemini/space/gjx/AgentEvolver/experiments/tech_synthetic/train_multibase_qwen3-vl-8b/llm_evaluation_logs/step_000042"
url = "http://10.233.45.72:5590/v1/chat/completions"
model_name = "Qwen3.5-397B-A17B-FP8"
output_file = "/gemini/space/gjx/AgentEvolver/exp/experience_used_analysis_enhanced.json"

all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")]

def process_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("used_experience", False):
            return None

        # 构造 prompt
        query_text = f"""
任务: {data.get('task_id')}
问题: {data.get('query')}
模型轨迹: {data.get('rollout')}
检索到经验: {data.get('experience_list')}
原始轨迹评分: {data.get('original_trajectory_score')}

请分析经验遵循情况，分类为：
1. correct_use: 正确遵循经验
2. irrelevant: 经验无关
3. misuse: 已读但误用
4. not_executed: 经验未执行
5. conflict: 经验与环境冲突

请输出 JSON：
{{
  "classification": "...",
  "reasoning": "...",
  "step_level_followed": int,  # 遵循经验步骤数
  "step_level_total_exp_steps": int,  # 总经验相关步骤数
  "exp_compliance_score": float,  # 遵循比例 0-1
  "trajectory_G/B_ratio": float,  # GOOD/BAD step比例
  "experience_type": []  # 每条经验类型，例如 ["operation","constraint"]
}}
"""

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "你是经验遵循分析专家。"},
                {"role": "user", "content": query_text}
            ],
            "chat_template_kwargs": {"enable_thinking": True}
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            resp_json = response.json()
            model_output = resp_json["choices"][0]["message"]["content"]
            try:
                analysis = json.loads(model_output)
            except:
                analysis = {
                    "classification": "parsing_error",
                    "reasoning": model_output,
                    "step_level_followed": 0,
                    "step_level_total_exp_steps": 0,
                    "exp_compliance_score": 0.0,
                    "trajectory_G/B_ratio": 0.0,
                    "experience_type": []
                }
        except Exception as e:
            analysis = {
                "classification": "api_error",
                "reasoning": str(e),
                "step_level_followed": 0,
                "step_level_total_exp_steps": 0,
                "exp_compliance_score": 0.0,
                "trajectory_G/B_ratio": 0.0,
                "experience_type": []
            }

        # 合并结果
        result = {
            "task_id": data.get("task_id"),
            "query": data.get("query"),
            "rollout": data.get("rollout"),
            "experience_list": data.get("experience_list"),
            "original_trajectory_score": data.get("original_trajectory_score")
        }
        result.update(analysis)
        sleep(0.1)
        return result
    except Exception as e:
        return {"task_id": None, "classification": "file_error", "reasoning": str(e)}

# 并行处理
results = []
with ProcessPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(process_file, f): f for f in all_files}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing JSON files"):
        res = future.result()
        if res:
            results.append(res)

# 保存 JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"增强分析完成，结果保存到 {output_file}")

# ----------------------
# 统计汇总
# ----------------------
stats = {
    "classification_counts": defaultdict(int),
    "task_stats": defaultdict(lambda: {"count":0, "mean_compliance":0.0})
}

for r in results:
    cls = r.get("model_classification", "unknown")
    stats["classification_counts"][cls] += 1

    task = r.get("task_id")
    if task:
        stats["task_stats"][task]["count"] += 1
        stats["task_stats"][task]["mean_compliance"] += r.get("exp_compliance_score", 0.0)

# 计算平均遵循率
for task, v in stats["task_stats"].items():
    if v["count"] > 0:
        v["mean_compliance"] /= v["count"]

# 保存统计
summary_file = output_file.replace(".json","_summary.json")
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"统计完成，汇总保存到 {summary_file}")