import json
import os
import tempfile

def remove_bad_jsonl_lines(path, max_show=200, backup=True):
    """
    删除JSONL文件中有问题的行
    
    参数:
        path: 文件路径
        max_show: 错误预览的最大长度
        backup: 是否创建备份文件
    """
    bad_lines = []
    
    # 第一步：读取并检查所有行
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 第二步：识别有问题的行
    valid_lines = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s:  # 跳过空行
            valid_lines.append(line)
            continue
            
        try:
            json.loads(s)
            valid_lines.append(line)
        except Exception as e:
            bad_lines.append((i, str(e), s[:max_show]))
            # 不将这一行添加到valid_lines中
    
    # 第三步：创建备份（可选）
    if backup and bad_lines:
        backup_path = path + ".backup"
        with open(backup_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"已创建备份文件: {backup_path}")
    
    # 第四步：重写文件（只保留有效行）
    if bad_lines:
        # 使用临时文件确保原子性操作
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(path))
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as temp_file:
                temp_file.writelines(valid_lines)
            
            # 用临时文件替换原文件
            os.replace(temp_path, path)
            
        except Exception:
            # 如果出错，删除临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    
    return bad_lines

# 使用示例
path = "local_vector_store/bfcl_multiturn_800.jsonl"  # 改成你的实际路径
bad = remove_bad_jsonl_lines(path, backup=True)

print(f"已删除 {len(bad)} 行有问题的数据")
if bad:
    print("被删除的行（前20个）：")
    for i, err, preview in bad[:20]:
        print(f"[line {i}] {err}\n  预览: {preview}\n")
    
    # 统计信息
    with open(path, "r", encoding="utf-8") as f:
        remaining_lines = sum(1 for line in f if line.strip())
    print(f"清理后文件剩余有效行数: {remaining_lines}")
else:
    print("文件格式正确，无需修改")