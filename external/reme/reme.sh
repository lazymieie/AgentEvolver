# ===== Azure LLM 环境变量 =====


export OPENAI_BASE_URL="http://127.0.0.1:3888/v1"
export OPENAI_API_KEY="local"


# 环境变量

export FLOW_EMBEDDING_API_KEY=sk-xxxx
export FLOW_EMBEDDING_BASE_URL=http://127.0.0.1:3888/v1

export AZURE_OPENAI_API_KEY="xxxxx"
export EMBEDDING_API_KEY="sk-xxxxx"


OPENAI_BASE_URL="http://127.0.0.1:3888/v1" \
OPENAI_API_KEY="local" \
reme \
  config=default \
  backend=http \
  http.host="127.0.0.1" \
  http.port=8001 \
  http.limit_concurrency=256 \
  \
  llm.default.provider=openai_compatible \
  llm.default.base_url="https://cv1-gpt4o.openai.azure.com/openai/deployments/gpt-4o-2" \
  llm.default.api_key="$FLOW_LLM_API_KEY" \
  llm.default.model_name="gpt-4o-2" \
  llm.default.api_version="2024-12-01-preview" \
  \
  embedding_model.default.provider=openai_compatible \
  embedding_model.default.base_url="http://127.0.0.1:3888/v1" \
  embedding_model.default.api_key="local" \
  embedding_model.default.model_name="my-embed" \
  embedding_model.default.dimensions=null \
  \
  vector_store.default.backend=local

CUDA_VISIBLE_DEVICES=4 \
vllm serve /vepfs-cnbj3fa964354bf4/gjx/AgentEvolver/model/Qwen/Qwen3-Embedding-0___6B \
  --host 127.0.0.1 \
  --port 3888 \
  --served-model-name my-embed
