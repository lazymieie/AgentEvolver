import os
import json
import math
import pandas as pd

# 根目录
base_path = "/gemini/space/gjx/AgentEvolver/experiments/tech_synthetic/train_multibase_qwen3-vl-8b/llm_evaluation_logs"

# step范围
start_step = 33
end_step = 102

# 读取哪个分数
score_key = "original_trajectory_score"
# score_key = "overall_score"   # 如果你以后想切换，改这里就行

results = []

def safe_mean(lst):
    return sum(lst) / len(lst) if lst else None

def safe_std(lst):
    if not lst or len(lst) < 2:
        return None
    s = pd.Series(lst, dtype="float")
    return float(s.std())

def safe_median(lst):
    if not lst:
        return None
    return float(pd.Series(lst, dtype="float").median())

all_exp_scores_global = []
all_noexp_scores_global = []
all_scores_global = []

for step in range(start_step, end_step + 1):
    step_folder = os.path.join(base_path, f"step_{step:06d}")

    if not os.path.exists(step_folder):
        print(f"跳过不存在目录: {step_folder}")
        continue

    scores_all = []
    scores_exp = []
    scores_noexp = []

    for filename in os.listdir(step_folder):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(step_folder, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"读取失败，跳过 {file_path}: {e}")
            continue

        score = data.get(score_key)
        used_exp = data.get("used_experience")

        if score is None:
            continue
        if not isinstance(score, (int, float)):
            continue

        scores_all.append(score)
        all_scores_global.append(score)

        if used_exp is True:
            scores_exp.append(score)
            all_exp_scores_global.append(score)
        else:
            scores_noexp.append(score)
            all_noexp_scores_global.append(score)

    mean_all = safe_mean(scores_all)
    mean_exp = safe_mean(scores_exp)
    mean_noexp = safe_mean(scores_noexp)

    std_all = safe_std(scores_all)
    std_exp = safe_std(scores_exp)
    std_noexp = safe_std(scores_noexp)

    median_all = safe_median(scores_all)
    median_exp = safe_median(scores_exp)
    median_noexp = safe_median(scores_noexp)

    diff_mean = None
    relative_gain = None
    better_flag = None

    if mean_exp is not None and mean_noexp is not None:
        diff_mean = mean_exp - mean_noexp
        if mean_noexp != 0:
            relative_gain = diff_mean / mean_noexp
        if diff_mean > 0:
            better_flag = "exp_better"
        elif diff_mean < 0:
            better_flag = "noexp_better"
        else:
            better_flag = "tie"

    results.append({
        "step": step,
        "total_samples": len(scores_all),
        "exp_samples": len(scores_exp),
        "noexp_samples": len(scores_noexp),

        "overall_mean": mean_all,
        "exp_mean": mean_exp,
        "noexp_mean": mean_noexp,

        "overall_std": std_all,
        "exp_std": std_exp,
        "noexp_std": std_noexp,

        "overall_median": median_all,
        "exp_median": median_exp,
        "noexp_median": median_noexp,

        "exp_minus_noexp": diff_mean,
        "relative_gain": relative_gain,
        "better_flag": better_flag,
        "exp_ratio": len(scores_exp) / len(scores_all) if scores_all else None
    })

# step级结果表
df = pd.DataFrame(results)

# ========= 全局汇总 =========
valid_compare_df = df[df["exp_minus_noexp"].notna()].copy()

num_steps = len(valid_compare_df)
num_exp_better = int((valid_compare_df["exp_minus_noexp"] > 0).sum())
num_noexp_better = int((valid_compare_df["exp_minus_noexp"] < 0).sum())
num_tie = int((valid_compare_df["exp_minus_noexp"] == 0).sum())

avg_step_gain = valid_compare_df["exp_minus_noexp"].mean() if num_steps > 0 else None
median_step_gain = valid_compare_df["exp_minus_noexp"].median() if num_steps > 0 else None
std_step_gain = valid_compare_df["exp_minus_noexp"].std() if num_steps > 1 else None

avg_relative_gain = valid_compare_df["relative_gain"].mean() if "relative_gain" in valid_compare_df and len(valid_compare_df["relative_gain"].dropna()) > 0 else None

# 按step样本数加权的平均提升
weighted_gain = None
if num_steps > 0:
    tmp = valid_compare_df.dropna(subset=["exp_minus_noexp", "total_samples"])
    if len(tmp) > 0 and tmp["total_samples"].sum() > 0:
        weighted_gain = (tmp["exp_minus_noexp"] * tmp["total_samples"]).sum() / tmp["total_samples"].sum()

# 全局样本级均值
global_exp_mean = safe_mean(all_exp_scores_global)
global_noexp_mean = safe_mean(all_noexp_scores_global)
global_all_mean = safe_mean(all_scores_global)

global_exp_std = safe_std(all_exp_scores_global)
global_noexp_std = safe_std(all_noexp_scores_global)
global_all_std = safe_std(all_scores_global)

global_diff = None
global_relative_gain = None
if global_exp_mean is not None and global_noexp_mean is not None:
    global_diff = global_exp_mean - global_noexp_mean
    if global_noexp_mean != 0:
        global_relative_gain = global_diff / global_noexp_mean

# 提升/下降幅度分桶
bucket_stats = {
    "gain_gt_0": int((valid_compare_df["exp_minus_noexp"] > 0).sum()) if num_steps > 0 else 0,
    "gain_ge_0.01": int((valid_compare_df["exp_minus_noexp"] >= 0.01).sum()) if num_steps > 0 else 0,
    "gain_ge_0.05": int((valid_compare_df["exp_minus_noexp"] >= 0.05).sum()) if num_steps > 0 else 0,
    "gain_ge_0.10": int((valid_compare_df["exp_minus_noexp"] >= 0.10).sum()) if num_steps > 0 else 0,
    "drop_le_-0.01": int((valid_compare_df["exp_minus_noexp"] <= -0.01).sum()) if num_steps > 0 else 0,
    "drop_le_-0.05": int((valid_compare_df["exp_minus_noexp"] <= -0.05).sum()) if num_steps > 0 else 0,
    "drop_le_-0.10": int((valid_compare_df["exp_minus_noexp"] <= -0.10).sum()) if num_steps > 0 else 0,
}

summary_lines = []
summary_lines.append("========== 基本设置 ==========")
summary_lines.append(f"score_key: {score_key}")
summary_lines.append(f"step range: {start_step} ~ {end_step}")
summary_lines.append("")

summary_lines.append("========== step级比较汇总 ==========")
summary_lines.append(f"可比较step数: {num_steps}")
summary_lines.append(f"有经验均值 > 没经验均值 的step数: {num_exp_better}")
summary_lines.append(f"有经验均值 < 没经验均值 的step数: {num_noexp_better}")
summary_lines.append(f"持平step数: {num_tie}")
summary_lines.append(f"经验更优占比: {num_exp_better / num_steps:.4f}" if num_steps > 0 else "经验更优占比: None")
summary_lines.append(f"平均step提升(exp_mean - noexp_mean): {avg_step_gain:.6f}" if avg_step_gain is not None else "平均step提升: None")
summary_lines.append(f"中位数step提升: {median_step_gain:.6f}" if median_step_gain is not None else "中位数step提升: None")
summary_lines.append(f"step提升标准差: {std_step_gain:.6f}" if std_step_gain is not None else "step提升标准差: None")
summary_lines.append(f"平均相对提升: {avg_relative_gain:.6f}" if avg_relative_gain is not None else "平均相对提升: None")
summary_lines.append(f"按step样本数加权的平均提升: {weighted_gain:.6f}" if weighted_gain is not None else "按step样本数加权的平均提升: None")
summary_lines.append("")

summary_lines.append("========== 全局样本级汇总 ==========")
summary_lines.append(f"全局总样本数: {len(all_scores_global)}")
summary_lines.append(f"全局有经验样本数: {len(all_exp_scores_global)}")
summary_lines.append(f"全局无经验样本数: {len(all_noexp_scores_global)}")
summary_lines.append(f"全局overall均值: {global_all_mean:.6f}" if global_all_mean is not None else "全局overall均值: None")
summary_lines.append(f"全局exp均值: {global_exp_mean:.6f}" if global_exp_mean is not None else "全局exp均值: None")
summary_lines.append(f"全局noexp均值: {global_noexp_mean:.6f}" if global_noexp_mean is not None else "全局noexp均值: None")
summary_lines.append(f"全局exp std: {global_exp_std:.6f}" if global_exp_std is not None else "全局exp std: None")
summary_lines.append(f"全局noexp std: {global_noexp_std:.6f}" if global_noexp_std is not None else "全局noexp std: None")
summary_lines.append(f"全局exp - noexp: {global_diff:.6f}" if global_diff is not None else "全局exp - noexp: None")
summary_lines.append(f"全局相对提升: {global_relative_gain:.6f}" if global_relative_gain is not None else "全局相对提升: None")
summary_lines.append("")

summary_lines.append("========== 提升/下降幅度分桶 ==========")
for k, v in bucket_stats.items():
    summary_lines.append(f"{k}: {v}")
summary_lines.append("")

# 找最强提升step和最强下降step
if num_steps > 0:
    best_row = valid_compare_df.loc[valid_compare_df["exp_minus_noexp"].idxmax()]
    worst_row = valid_compare_df.loc[valid_compare_df["exp_minus_noexp"].idxmin()]

    summary_lines.append("========== 极值step ==========")
    summary_lines.append(
        f"提升最大step: {int(best_row['step'])}, "
        f"diff={best_row['exp_minus_noexp']:.6f}, "
        f"exp_mean={best_row['exp_mean']:.6f}, "
        f"noexp_mean={best_row['noexp_mean']:.6f}"
    )
    summary_lines.append(
        f"下降最大step: {int(worst_row['step'])}, "
        f"diff={worst_row['exp_minus_noexp']:.6f}, "
        f"exp_mean={worst_row['exp_mean']:.6f}, "
        f"noexp_mean={worst_row['noexp_mean']:.6f}"
    )
    summary_lines.append("")

# 输出文件
output_step_file = "/gemini/space/gjx/AgentEvolver/exp/experience_step_trend_ori_detailed_base_wa_8b.txt"
output_csv_file = "/gemini/space/gjx/AgentEvolver/exp/experience_step_trend_ori_detailed_base_wa_8b.csv"

# 保存step级表
df.to_csv(output_csv_file, index=False, encoding="utf-8-sig")

# 保存txt分析
with open(output_step_file, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
    f.write("\n\n========== 每个step详细统计 ==========\n")
    f.write(df.to_string(index=False))

print("\n".join(summary_lines))
print("\n========== 每个step详细统计 ==========")
print(df)

print(f"\n详细TXT已保存到: {output_step_file}")
print(f"CSV已保存到: {output_csv_file}")