import json

def find_bad_jsonl(path, max_show=200):
    bad = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                json.loads(s)
            except Exception as e:
                bad.append((i, str(e), s[:max_show]))
    return bad

path = "local_vector_store/bfcl_multiturn_800.jsonl"  # 改成你的实际路径
bad = find_bad_jsonl(path)

print(f"bad lines = {len(bad)}")
for i, err, preview in bad[:20]:
    print(f"[line {i}] {err}\n  {preview}\n")
