import os
import json
import requests
import time
from tqdm import tqdm

# =====================
# 配置区
# =====================
FOLDER_PATH = "/gemini/space/gjx/AgentEvolver/experiments/tech_synthetic/bfcl_qwen3-vl-4b_agentevolver_wo_self_attribute_wo_question_base/llm_evaluation_logs/step_000002"  # 你的 JSON 文件目录
MODEL_URL = "http://10.233.45.72:5590/v1/chat/completions"
MODEL_NAME = "Qwen3.5-397B-A17B-FP8"
OUTPUT_FIELD = "experience_analysis"  # 在原 json 上增加这个字段

# 请求参数
MAX_RETRIES = 5
RETRY_WAIT = 5  # 秒

# =====================
# 构建 prompt
# =====================
PROMPT_TEMPLATE = """
你是一个经验分析专家，请分析用户任务轨迹中的经验使用情况。
输入数据：
任务详情（query）:
{query}

经验列表（experience_list）:
{experience_list}

轨迹步骤（steps）:
{steps}

请完成以下分析：
1. experience_applicability: 对每条经验是否适用于该任务，输出 "applicable" 或 "not_applicable"，可附简短理由。
2. experience_followed: 如果经验适用，是否被遵循，输出 "followed", "partially_followed", "not_followed"。
3. first_deviation_step: 如果经验未被完全遵循，请标注轨迹中第一处偏离经验的步骤 index（从 0 开始），如果没有偏离则为 null。

输出要求严格的 JSON 结构，不要输出其他文字，例如：
{{
  "experience_applicability": [...],
  "experience_followed": [...],
  "first_deviation_step": [...]
}}
"""

# =====================
# 调用模型函数
# =====================
def call_model(query, experience_list, steps):
    prompt = PROMPT_TEMPLATE.format(
        query=query,
        experience_list=json.dumps(experience_list, ensure_ascii=False, indent=2),
        steps=json.dumps(steps, ensure_ascii=False, indent=2)
    )
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                MODEL_URL,
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 2000
                },
                timeout=180
            )
            response.raise_for_status()
            data = response.json()
            # Qwen3.5 返回格式通常在 choices[0].message.content
            content = data["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            print(f"调用模型失败 (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_WAIT)
    return None

# =====================
# 遍历文件分析
# =====================
def analyze_json_files(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    for file_name in tqdm(all_files):
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            continue

        if not data.get("used_experience", False):
            continue  # 未使用经验的轨迹跳过

        query = data.get("query", "")
        experience_list = data.get("experience_list", [])
        steps = data.get("steps", [])

        # 调用模型
        model_output = call_model(query, experience_list, steps)
        if model_output is None:
            print(f"模型调用失败，跳过 {file_path}")
            continue

        try:
            # 尝试解析模型返回的 JSON
            analysis_json = json.loads(model_output)
        except Exception as e:
            print(f"模型返回无法解析为 JSON，保存原文，文件: {file_path}, 错误: {e}")
            analysis_json = {"raw_output": model_output}

        # 补充到原 JSON 文件
        data[OUTPUT_FIELD] = analysis_json
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"写入文件失败 {file_path}: {e}")

# =====================
# 主函数
# =====================
if __name__ == "__main__":
    analyze_json_files(FOLDER_PATH)