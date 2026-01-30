# ---- Start Environment Service ----
# conda activate appworld
# bash env_service/launch_script/appworld.sh


# ---- Start ReMe Service ----
# conda activate reme
# cd external/reme
# reme \
#   config=default \
#   backend=http \
#   thread_pool_max_workers=256 \
#   http.host="127.0.0.1" \
#   http.port=8001 \
#   http.limit_concurrency=256 \
#   llm.default.model_name=qwen-max-2025-01-25 \
#   embedding_model.default.model_name=text-embedding-v4 \
#   vector_store.default.backend=local \
#   op.rerank_memory_op.params.enable_llm_rerank=false


# ---- Start Training ----

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/config"
env_url=http://localhost:8011
em_url=http://localhost:8001
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="logs/run_val_only_qwen3_VL_8B_thinking_linear/log_${current_time}.log"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export RAY_DISABLE_DASHBOARD=1


# ---- Start Environment Service ----
# conda activate appworld
# bash env_service/launch_script/appworld.sh

# ---- Start Training ----
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/config"
current_time=$(date "+%Y%m%d_%H%M%S")

export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export RAY_DISABLE_DASHBOARD=1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m agentevolver.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='script_config' \
    env_service.env_url=$env_url \
    actor_rollout_ref.actor.off_cliprange_high=0.6 \
    attribution_driven_credit_assignment.enable=false \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.max_prompt_length=20480 \
    data.max_response_length=6096 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.use_qwen3=True \
    actor_rollout_ref.rollout.enable_request_id=False \
    actor_rollout_ref.rollout.prompt_length=20480 \
    actor_rollout_ref.rollout.response_length=6096 \
    actor_rollout_ref.rollout.max_model_len=26576 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.model.path=/vepfs-cnbj3fa964354bf4/gjx/AgentEvolver/model/Qwen/Qwen3-VL-8B-Thinking \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.context_template='linear' \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.n_gpus_per_node=4 \
    trainer.critic_warmup=0 \
    trainer.logger="['tensorboard','console']" \
    trainer.project_name="bfcl-qwen3-VL-8B-Thinking_linear" \
    trainer.experiment_name="bfcl_qwen3-VL-8B-Thinking_val_only_linear" \
    trainer.nnodes=1 \
    trainer.save_freq=10000 \
    trainer.test_freq=10 \
    trainer.total_epochs=0 \
    trainer.val_before_train=True \
    trainer.validation_data_dir="experiments/tech_synthetic/${experiment_name}/validation_log" \
    trainer.rollout_data_dir="experiments/tech_synthetic/${experiment_name}/rollout_log" \
    trainer.resume_mode='disable' \
    trainer.val_only=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=27580 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=27580 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=27580 \
    critic.ppo_max_token_len_per_gpu=27580 \
    critic.forward_max_token_len_per_gpu=27580 \
    data.train_files=null \
    data.val_files=null \
    env_service.env_type=bfcl \
    task_manager.n=0 \
    task_manager.mixture.synthetic_data_ratio=0.0 \
    task_manager.mixture.use_original_tasks=True \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    2>&1 | tee "$log_file" \