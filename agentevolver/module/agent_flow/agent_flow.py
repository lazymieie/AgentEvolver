import time
import os

from loguru import logger

from agentevolver.client.em_client import EMClient
from agentevolver.client.env_client import EnvClient
from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from agentevolver.utils.utils import convert_tool_to_user_message
from agentevolver.schema.trajectory import Reward, Trajectory
from best_logger import register_logger, print_dict, print_listofdict
from agentevolver.module.context_manager.cmt_linear import Linear_CMT, ExtendedMessage
from agentevolver.module.context_manager.cmt_linear_think import LinearThinkCMT
from agentevolver.module.context_manager.cmt_context_clip import SelfContextClipCMT
from agentevolver.module.agent_flow.reward_calculator import RewardCalculator
from typing import Any, Dict, List, Union, Optional
import threading
from agentevolver.module.exp_manager.exp_manager import TrajExpConfig, ExperienceWorker


log_generate_lock = threading.Lock()

class AgentFlow(BaseAgentFlow):

    def __init__(self, reward_calculator:Optional[RewardCalculator]=None, **kwargs):
        """
        Initializes an instance of the AgentFlow class.

        Args:
            reward_calculator (Optional[RewardCalculator]): An optional reward calculator object.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)  # ⭐ Call the constructor of the base class
        self._reward_calculator = reward_calculator
        # self._enable_context_generator=self.config.experience_maker.enable_context_generator

        self.instruction_template_ids = self.tokenizer.encode("user\n")  # ⭐ Encode the user instruction template
        self.response_template_ids = self.tokenizer.encode("assistant\n")  # ⭐ Encode the assistant response template
        # self.em_client = EMClient(base_url=self.config.experience_maker.base_url)  # ⭐ Initialize the EMClient
        self.sparse = self.config.actor_rollout_ref.rollout.sparse  # add sparse by ANNI 0723
        # self.experience_template = self.config.hybrid_experience_training.experience_template
        self.cmt: Union[Linear_CMT, LinearThinkCMT] = None
        self.console_debug_mode: bool = self.config.actor_rollout_ref.rollout.debug_llm_io
        self.exp_worker = ExperienceWorker(config=self.config)
        # artifact_recorder will be set from trainer if available
        self.exp_worker.artifact_recorder = None

    def _log_raw_conversation(self, task_id: str, traj_id: str = ""):
        """
        统一记录「未经过 tokenizer 的原始对话轨迹」，适用于所有 CMT。

        优先使用各 CMT 的 prepare_previous_context(mod='raw')，
        若不存在则尝试从 full_context 中恢复。
        """
        try:
            # 优先走统一接口
            if hasattr(self.cmt, "prepare_previous_context"):
                raw_messages = self.cmt.prepare_previous_context(mod="raw")
            # 兜底：从 full_context 里取 ExtendedMessage.content
            elif hasattr(self.cmt, "full_context"):
                raw_messages = [
                    {"role": m.role, "content": m.content}
                    for m in getattr(self.cmt, "full_context", [])
                ]
            else:
                raw_messages = []

            if not raw_messages:
                return

            # DEBUG 日志已关闭：如果需要再次查看原始对话，可恢复下面的打印。
            # header = f"Raw conversation task {task_id}"
            # if traj_id:
            #     header += f", traj {traj_id}"
            #
            # print_listofdict(
            #     raw_messages,
            #     header=header,
            #     mod="conversation_raw",
            #     narrow=False,
            # )
        except Exception as e:
            import logging
            logging.warning(f"Failed to log raw conversation for task {task_id}: {e}")


    def execute(self, context_manager, init_messages: List[dict], env: EnvClient, instance_id: str, tmux, stop, thread_index, task_id, traj_exp_config,data_id="", rollout_id="", query="", **kwargs) -> Linear_CMT:
        """
        Executes the interaction between the AI agent and the environment, managing the context, experience generation, and reward calculation.

        Args:
            context_manager (ContextManager): The context manager for the current task.
            init_messages (List[dict]): Initial messages for the task.
            env (EnvClient): The environment client.
            instance_id (str): The ID of the instance.
            tmux (dict): TMUX dictionary for tracking steps and tokens.
            stop (list): A list indicating whether to stop the thread.
            thread_index (int): The index of the current thread.
            task_id (str): The ID of the task.
            traj_exp_config (TrajExpConfig): Experience Configuration for the trajectory.
            data_id (str, optional): The ID of the data. Defaults to "".
            rollout_id (str, optional): The ID of the rollout. Defaults to "".
            query (str, optional): The query string. Defaults to "".
            **kwargs: Additional keyword arguments.

        Returns:
            Linear_CMT: The context manager after the execution.
        """
        self.cmt = context_manager
        # disable think for qwen3
        add_nothink = self.config.actor_rollout_ref.rollout.use_qwen3 # if qwen3, add /no_think

        # 1. 🚀 Initialize messages
        traj_exp_config.query = query
        init_messages, traj_exp_config = self.exp_worker.manage_rollout_context(
                init_messages=init_messages,
                traj_exp_config=traj_exp_config
                )
        self.cmt.metadata["task_train_exp_mode"] = traj_exp_config.train_mode
        self.cmt.metadata["add_exp"] = traj_exp_config.add_exp
        self.cmt.metadata["experience_list"] = traj_exp_config.experience_list
        # init_messages, metadata = self.add_experience(init_messages, task_id, data_id, rollout_id, query, add_exp)  # ⭐ Initialize messages and metadata
        # self.cmt.metadata = metadata
        self.cmt.save_init_input(init_messages, add_nothink)

        request_id: str = ""
        err_in_generating=False
        err_in_env = False
        for act_step in range(self.max_steps):
            # 2. 🔄 Update thread progress
            tmux['step'][thread_index] = act_step
            if (stop is not None) and stop[thread_index]: # Check if the thread should stop (because other threads have completed, making this thread useless)
                self.cmt.discarded = True
                break

            # 3. ⏮️ get previous steps
            try:
                step_input_message_arr = self.cmt.prepare_next_llm_context()  # ⭐ Prepare the next LLM context
            except Exception as e:
                print_listofdict(self.cmt.to_role_content(self.cmt.full_context), mod='exception', header="Before Crash")
                raise e

            # 4. ⚠️ check token overflow
            is_safe: bool = self.cmt.check_context_token_num_safe(step_input_message_arr)  # ⭐ Check if the context token count is safe
            if not is_safe:
                logger.warning(f"Token overflow detected at step {act_step}. Current token count exceeds the limit.")
                print(
                f"[OVERFLOW] task_id={task_id}, step={act_step}, "
                )

                self.cmt.is_terminated = False # trajectory not finished.
                break
            # print("debug：act_step")
            # print(act_step)
            # 5. 🤖 call llm
            llm_output = self.llm_chat_fn(step_input_message_arr, request_id=request_id)  # ⭐ Call the LLM to generate the next response
            if (stop is not None) and stop[thread_index]:  # Check if the thread should stop (because other threads have completed, making this thread useless)
                self.cmt.discarded = True
                break
            
            # 6. 💾 save llm output
            self.cmt.save_llm_output(llm_output, input_msg_ref=step_input_message_arr)  # ⭐ Save the LLM output
            tmux['token'][thread_index] += self.cmt.generated_token_cnt

            # 7. 🌍 world interaction
            try:
                world_interaction_content = self.cmt.prepare_world_interaction()
                # DEBUG (disabled): Print what is being sent to environment
                # if hasattr(self.cmt, '__class__') and 'Memory' in self.cmt.__class__.__name__:
                #     print_dict({
                #         "DEBUG_AGENTFLOW_ENV_STEP": {
                #             "step": act_step,
                #             "content_to_env": world_interaction_content[:500] if world_interaction_content else "EMPTY",
                #             "content_length": len(world_interaction_content) if world_interaction_content else 0,
                #             "content_is_empty": not world_interaction_content or world_interaction_content.strip() == "",
                #         }
                #     }, mod='memory_debug')
                env_output = env.step(instance_id, {"content": world_interaction_content, "role": "assistant"})  # ⭐ Interact with the environment
                assert len(env_output['state'])==1
                env_output["state"] = env_output["state"][0]
                if env_output["state"]["role"] == "tool":
                    env_output["state"] = convert_tool_to_user_message(env_output["state"], self.tokenizer, format="qwen")
                
                # DEBUG (disabled): Print env response and reward info
                # if hasattr(self.cmt, '__class__') and 'Memory' in self.cmt.__class__.__name__:
                #     print_dict({
                #         "DEBUG_AGENTFLOW_ENV_RESPONSE": {
                #             "step": act_step,
                #             "env_state_role": env_output["state"].get("role", "unknown"),
                #             "env_state_content_preview": env_output["state"].get("content", "")[:300] if env_output["state"].get("content") else "EMPTY",
                #             "is_terminated": env_output.get("is_terminated", False),
                #             "env_reward": env_output.get("reward", None),
                #         }
                #     }, mod='memory_debug')
                
                if self.console_debug_mode:
                    print_listofdict(
                        step_input_message_arr +
                        [{'role': 'llm_latest', 'content': llm_output['content']}] +
                        [{'role': 'env',        'content': env_output["state"]['content']}]
                    , mod='c')
            except Exception as e:
                logger.bind(exception=True).exception(f"call env.step error with {e}")
                err_in_env = True
                self.cmt.is_terminated = False # trajectory not finished.
                state = {"content": str(e), "role": "user"}
                env_output = {
                    "reward": 0,
                    "is_terminated": True,
                    "state": state,
                }

            # 8. 📥 save environment output
            state = env_output["state"]
            state.pop('tool_calls', None)
            self.cmt.save_env_output(state, input_msg_ref=step_input_message_arr, add_nothink=add_nothink)  # ⭐ Save the environment output

            # 9. 🔚 determine if the episode is terminated
            self.cmt.is_terminated = env_output["is_terminated"]
            if self.cmt.is_terminated or err_in_env:
                break

        tmux['step'][thread_index] = -1

        if self._reward_calculator is not None:
            grader_res = self._reward_calculator.calculate_reward(self.cmt, env, instance_id)  # ⭐ Calculate the reward using the reward calculator
            score = grader_res["score"] 
            reason = grader_res["reason"] or "No reason provided."
            # DEBUG (disabled): trace reward when using external grader
            # print_dict(
            #     {
            #         "mode": "reward_calculator",
            #         "score": float(score),
            #         "reason": reason,
            #         "task_id": task_id,
            #         "context_template": getattr(self.config.actor_rollout_ref.rollout, "context_template", "unknown"),
            #     },
            #     mod="reward_debug",
            # )
        else:
            score = env.evaluate(instance_id, params={"sparse": self.sparse})  # ⭐ Evaluate the score from the environment
            reason = "Outcome 1 = success, 0 = failure."
            # DEBUG (disabled): trace reward when using env.evaluate
            # print_dict(
            #     {
            #         "mode": "env_evaluate",
            #         "score": float(score),
            #         "reason": reason,
            #         "task_id": task_id,
            #         "context_template": getattr(self.config.actor_rollout_ref.rollout, "context_template", "unknown"),
            #     },
            #     mod="reward_debug",
            # )

        if score >= 1: success_rate = 1.0
        else: success_rate = 0.0

        self.cmt.reward = Reward(outcome=score, success_rate=success_rate, madness=self.cmt.compute_madness(), description=reason)  # ⭐ Set the reward for the context
        self.cmt.reward = self.cmt.reward_patch(self.cmt.reward)
        
        # DEBUG (disabled): Final reward assignment
        # print_dict({
        #     "DEBUG_AGENTFLOW_FINAL_REWARD": {
        #         "task_id": task_id,
        #         "raw_score": float(score),
        #         "success_rate": float(success_rate),
        #         "madness": float(self.cmt.reward.madness),
        #         "final_outcome": float(self.cmt.reward.outcome),
        #         "description": reason,
        #         "context_template": getattr(self.config.actor_rollout_ref.rollout, "context_template", "unknown"),
        #         "num_groups": len(self.cmt.grouped_steps) if hasattr(self.cmt, 'grouped_steps') else 0,
        #         "is_terminated": self.cmt.is_terminated,
        #     }
        # }, mod='memory_debug')
        
        self.cmt.remove_last_context()

        # Record rollout
        traj_id = f"{data_id}_{rollout_id}" if data_id and rollout_id else f"{task_id}_unknown"

        # 统一记录原始对话轨迹（所有 CMT 通用，包含 linear / linear_think / context_selfclip / memory 等）
        # self._log_raw_conversation(task_id=task_id, traj_id=traj_id)

        if hasattr(self, 'artifact_recorder') and self.artifact_recorder and self.artifact_recorder.enable:
            try:
                # Extract steps from CMT
                steps_list = []
                if hasattr(self.cmt, 'steps') and self.cmt.steps:
                    for i, step_msg in enumerate(self.cmt.steps):
                        if isinstance(step_msg, dict):
                            step_dict = {
                                "t": i,
                                "obs": step_msg.get("content", ""),
                                "action": "",  # Will be filled from next step if available
                            }
                            steps_list.append(step_dict)
                
                # Determine mode from traj_exp_config
                mode = "woexp"
                if hasattr(traj_exp_config, 'add_exp') and traj_exp_config.add_exp:
                    mode = "mixed"
                
                self.artifact_recorder.write_rollouts(
                    task_id=task_id,
                    traj_id=traj_id,
                    mode=mode,
                    n_steps=len(steps_list) if steps_list else 0,
                    steps=steps_list if self.artifact_recorder.dump_steps else None,
                )
                
                # Record reward
                self.artifact_recorder.write_rewards(
                    task_id=task_id,
                    traj_id=traj_id,
                    reward_value=float(score),
                    reward_detail={
                        "success_rate": success_rate,
                        "madness": self.cmt.compute_madness(),
                        "description": reason,
                    },
                    grader_name=getattr(self._reward_calculator, '__class__', {}).__name__ if self._reward_calculator else "env",
                )
            except Exception as e:
                import logging
                logging.warning(f"Failed to record rollout/reward: {e}")

        with log_generate_lock:
            self.cmt.generate_log(task_id=task_id)  # ⭐ Generate the log for the task


        return self.cmt
