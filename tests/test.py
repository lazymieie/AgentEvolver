from transformers import AutoProcessor, AutoModelForVision2Seq

path = "/vepfs-cnbj3fa964354bf4/gjx/AgentEvolver/model/Qwen/Qwen3-VL-8B-Thinking"

# Qwen3-VL 用 Processor（不是 tokenizer）
processor = AutoProcessor.from_pretrained(
    path,
    local_files_only=True,
    trust_remote_code=True,
)

model = AutoModelForVision2Seq.from_pretrained(
    path,
    local_files_only=True,
    trust_remote_code=True,
    device_map="cpu",   # 先用 CPU 验证
)

print("✅ Qwen3-VL-8B-Instruct loaded successfully")
print(type(processor), type(model))
