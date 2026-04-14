import time
import os
import hashlib

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
        self.exp_worker = ExperienceWorker(config=self.config, tokenizer=self.tokenizer)
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
        self.cmt.metadata["state_experience_tool_calls"] = 0
        self.cmt.metadata["state_experience_tool_call_steps"] = []
        self.cmt.metadata["state_experience_tool_injections"] = 0
        self.cmt.metadata["state_experience_tool_nonempty_retrievals"] = 0
        self.cmt.metadata["state_experience_tool_empty_results"] = 0
        self.cmt.metadata["state_experience_tool_overflow_reverts"] = 0
        self.cmt.metadata["state_experience_tool_repeat_calls_after_injection"] = 0
        self.cmt.metadata["state_experience_tool_retrieved_char_count"] = 0
        self.cmt.metadata["state_experience_tool_retrieved_token_count"] = 0
        self.cmt.metadata["used_state_experience_tool"] = False
        self.cmt.metadata["state_experience_tool_traces"] = []
        self.cmt.metadata["state_tool_ablation_mode"] = self.exp_worker.get_state_tool_ablation_mode()
        self.cmt.metadata["no_tool_second_chance_calls"] = 0
        self.cmt.metadata["no_tool_second_chance_steps"] = []
        # init_messages, metadata = self.add_experience(init_messages, task_id, data_id, rollout_id, query, add_exp)  # ⭐ Initialize messages and metadata
        # self.cmt.metadata = metadata
        self.cmt.save_init_input(init_messages, add_nothink)

        request_id: str = ""
        err_in_generating=False
        err_in_env = False
        completed_env_steps = 0
        stop_reason: str | None = None

        traj_id = f"{data_id}_{rollout_id}" if data_id and rollout_id else f"{task_id}_unknown"

        def mark_generation_failure(
            step: int,
            reason: str,
            stage: str | None = None,
            prompt_with_exp: str | None = None,
            prompt_without_exp: str | None = None,
            llm_output: Dict[str, Any] | None = None,
        ) -> None:
            self.cmt.is_terminated = False
            self.cmt.metadata["generation_failure"] = True
            self.cmt.metadata["generation_failure_step"] = step
            self.cmt.metadata["generation_failure_reason"] = reason
            if stage is not None:
                self.cmt.metadata["generation_failure_stage"] = stage

            if hasattr(self, 'artifact_recorder') and self.artifact_recorder and self.artifact_recorder.enable:
                mode = "mixed" if getattr(traj_exp_config, "add_exp", False) else "woexp"
                self.artifact_recorder.write_generation_failures(
                    task_id=task_id,
                    traj_id=traj_id,
                    mode=mode,
                    step=step,
                    reason=reason,
                    stage=stage,
                    prompt_with_exp=prompt_with_exp,
                    prompt_without_exp=prompt_without_exp,
                    llm_output=llm_output,
                    tool_names=self.exp_worker.get_called_tool_names(llm_output) if llm_output else None,
                )

        def record_tool_call_issue(step: int, stage: str, llm_output: Dict[str, Any]) -> None:
            issue = {
                "step": step,
                "stage": stage,
                "type": "mixed_experience_and_other_tool_calls",
                "tool_names": self.exp_worker.get_called_tool_names(llm_output),
            }
            existing_issues = self.cmt.metadata.get("experience_tool_call_issues")
            if not isinstance(existing_issues, list):
                existing_issues = []
                self.cmt.metadata["experience_tool_call_issues"] = existing_issues
            existing_issues.append(issue)

        def record_state_experience_tool_call(step: int, llm_output: Dict[str, Any], repeated_after_injection: bool = False) -> int:
            tool_call_count = self.exp_worker.get_called_tool_names(llm_output).count("get_experience_guidance")
            if tool_call_count <= 0 and self.exp_worker.has_experience_guidance_tool_call(llm_output):
                tool_call_count = 1
            if tool_call_count <= 0:
                return 0

            self.cmt.metadata["state_experience_tool_calls"] += tool_call_count
            self.cmt.metadata["used_state_experience_tool"] = True

            call_steps = self.cmt.metadata.get("state_experience_tool_call_steps")
            if not isinstance(call_steps, list):
                call_steps = []
                self.cmt.metadata["state_experience_tool_call_steps"] = call_steps
            call_steps.extend([step] * tool_call_count)

            if repeated_after_injection:
                self.cmt.metadata["state_experience_tool_repeat_calls_after_injection"] += tool_call_count

            return tool_call_count

        def record_repeated_state_experience_tool_event(
            step: int,
            retry_idx: int,
            max_retries: int,
            llm_output: Dict[str, Any],
            prompt_with_exp: str,
            prompt_without_exp: str,
            injected_experience: str,
            injection_applied: bool,
        ) -> None:
            if not hasattr(self, 'artifact_recorder') or not self.artifact_recorder or not self.artifact_recorder.enable:
                return

            try:
                self.artifact_recorder.write_state_experience_tool_event(
                    task_id=task_id,
                    traj_id=traj_id,
                    step=step,
                    event_type="repeat_call_after_injection",
                    retry_idx=retry_idx,
                    max_retries=max_retries,
                    injection_applied=injection_applied,
                    tool_names=self.exp_worker.get_called_tool_names(llm_output),
                    prompt_with_exp=prompt_with_exp,
                    prompt_without_exp=prompt_without_exp,
                    injected_experience=injected_experience,
                    llm_output=llm_output,
                )
            except Exception as e:
                logger.warning(f"Failed to record repeated state experience tool event: {e}")

        def write_state_tool_trace(step: int, trace: Dict[str, Any]) -> None:
            stored_traces = self.cmt.metadata.get("state_experience_tool_traces")
            if not isinstance(stored_traces, list):
                stored_traces = []
                self.cmt.metadata["state_experience_tool_traces"] = stored_traces
            stored_traces.append({
                "step": step,
                **trace,
            })
            if not hasattr(self, 'artifact_recorder') or not self.artifact_recorder or not self.artifact_recorder.enable:
                return
            try:
                self.artifact_recorder.write_state_experience_tool_trace(
                    task_id=task_id,
                    traj_id=traj_id,
                    step=step,
                    trace=trace,
                )
            except Exception as e:
                logger.warning(f"Failed to record state experience tool trace: {e}")

        def should_apply_no_tool_second_chance(step: int) -> bool:
            mode = self.exp_worker.get_no_tool_second_chance_mode()
            if mode == "all_second_chance_no_tool":
                return True
            if mode == "random_second_chance_no_tool":
                ratio = self.exp_worker.get_no_tool_second_chance_ratio()
                if ratio <= 0.0:
                    return False
                seed = self.exp_worker.get_no_tool_second_chance_seed()
                key = f"{seed}:{task_id}:{data_id}:{rollout_id}:{thread_index}:{step}"
                digest = hashlib.sha256(key.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)
                return bucket < ratio
            return False

        def record_no_tool_second_chance(step: int, extra_calls: int) -> None:
            self.cmt.metadata["no_tool_second_chance_calls"] += extra_calls
            applied_steps = self.cmt.metadata.get("no_tool_second_chance_steps")
            if not isinstance(applied_steps, list):
                applied_steps = []
                self.cmt.metadata["no_tool_second_chance_steps"] = applied_steps
            applied_steps.append(step)

        for act_step in range(self.max_steps):
            # 2. 🔄 Update thread progress
            tmux['step'][thread_index] = act_step
            if (stop is not None) and stop[thread_index]: # Check if the thread should stop (because other threads have completed, making this thread useless)
                self.cmt.discarded = True
                stop_reason = "discarded_by_parallel_stop"
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
                stop_reason = "context_overflow"
                break
            # print("debug：act_step")
            # print(act_step)
            # 5. 🤖 call llm
            final_step_input_message_arr = step_input_message_arr
            transient_state_experience = ""
            transient_injection_applied = False
            state_tool_trace: Dict[str, Any] | None = None
            llm_output = self.llm_chat_fn(step_input_message_arr, request_id=request_id)  # ⭐ Call the LLM to generate the next response
            if "content" not in llm_output or llm_output["content"] is None:
                llm_output["content"] = ""
            use_state_tool_experience = (
                self.exp_worker._use_state_tool_experience() and (
                    traj_exp_config.add_exp or self.exp_worker.should_force_state_tool_ablation()
                )
            )
            if use_state_tool_experience and self.exp_worker.has_mixed_experience_and_other_tool_calls(llm_output):
                logger.warning(
                    f"Assistant mixed get_experience_guidance with other tool calls in the same response "
                    f"at step {act_step}; continuing as experience-only handling."
                )
                record_tool_call_issue(act_step, "initial_request", llm_output)
            if use_state_tool_experience and self.exp_worker.has_experience_guidance_tool_call(llm_output):
                record_state_experience_tool_call(act_step, llm_output)
                latest_prompt_without_exp = ""
                if step_input_message_arr:
                    latest_prompt_without_exp = str(step_input_message_arr[-1].get("content", ""))
                state_tool_trace = {
                    "tool_call_count": 1,
                    "context_before_tool": step_input_message_arr,
                    "prompt_without_exp": latest_prompt_without_exp,
                    "initial_llm_output": llm_output,
                    "tool_names": self.exp_worker.get_called_tool_names(llm_output),
                    "tool_payload": self.exp_worker.get_experience_tool_payload(llm_output),
                    "retrieval": {},
                    "guided_retries": [],
                }
                retrieval_details = self.exp_worker.retrieve_state_tool_experience_details(
                    llm_output=llm_output,
                    traj_exp_config=traj_exp_config,
                    task_id=task_id,
                )
                transient_state_experience = str(retrieval_details.get("formatted_experience", ""))
                if state_tool_trace is not None:
                    state_tool_trace["retrieval"] = {
                        "retrieval_mode": retrieval_details.get("retrieval_mode", ""),
                        "query": retrieval_details.get("query", ""),
                        "topk": retrieval_details.get("topk", []),
                        "raw_experience": retrieval_details.get("raw_experience", ""),
                        "formatted_experience": transient_state_experience,
                    }
                if transient_state_experience:
                    self.cmt.metadata["state_experience_tool_nonempty_retrievals"] += 1
                    self.cmt.metadata["state_experience_tool_retrieved_char_count"] += len(transient_state_experience)
                    self.cmt.metadata["state_experience_tool_retrieved_token_count"] += len(
                        self.tokenizer.encode(transient_state_experience, add_special_tokens=False)
                    )
                else:
                    self.cmt.metadata["state_experience_tool_empty_results"] += 1
                final_step_input_message_arr = self.exp_worker.prepend_experience_to_latest_message(
                    step_input_message_arr,
                    transient_state_experience,
                )
                transient_injection_applied = final_step_input_message_arr != step_input_message_arr
                if state_tool_trace is not None:
                    latest_prompt_with_exp = ""
                    if final_step_input_message_arr:
                        latest_prompt_with_exp = str(final_step_input_message_arr[-1].get("content", ""))
                    state_tool_trace["prompt_with_exp"] = latest_prompt_with_exp
                    state_tool_trace["injection_applied"] = transient_injection_applied

                is_safe = self.cmt.check_context_token_num_safe(final_step_input_message_arr)
                if not is_safe:
                    logger.warning(f"Token overflow detected after transient state experience injection at step {act_step}.")
                    final_step_input_message_arr = step_input_message_arr
                    transient_injection_applied = False
                    self.cmt.metadata["state_experience_tool_overflow_reverts"] += 1
                if state_tool_trace is not None:
                    state_tool_trace["overflow_reverted"] = not is_safe
                    state_tool_trace["effective_prompt_after_overflow_check"] = (
                        final_step_input_message_arr[-1].get("content", "")
                        if final_step_input_message_arr else ""
                    )

                max_guided_generation_retries = self.exp_worker.get_guided_generation_retries()
                generation_succeeded = False
                for guided_retry_idx in range(max_guided_generation_retries):
                    llm_output = self.llm_chat_fn(final_step_input_message_arr, request_id=request_id)
                    if "content" not in llm_output or llm_output["content"] is None:
                        llm_output["content"] = ""

                    if self.exp_worker.has_mixed_experience_and_other_tool_calls(llm_output):
                        logger.warning(
                            f"Assistant mixed get_experience_guidance with other tool calls after transient injection "
                            f"at step {act_step}; continuing as experience-only handling."
                        )
                        record_tool_call_issue(act_step, f"guided_retry_{guided_retry_idx + 1}", llm_output)

                    if not self.exp_worker.has_experience_guidance_tool_call(llm_output):
                        generation_succeeded = True
                        break

                    record_state_experience_tool_call(
                        act_step,
                        llm_output,
                        repeated_after_injection=True,
                    )
                    latest_prompt_with_exp = ""
                    latest_prompt_without_exp = ""
                    if final_step_input_message_arr:
                        latest_prompt_with_exp = str(final_step_input_message_arr[-1].get("content", ""))
                    if step_input_message_arr:
                        latest_prompt_without_exp = str(step_input_message_arr[-1].get("content", ""))
                    record_repeated_state_experience_tool_event(
                        step=act_step,
                        retry_idx=guided_retry_idx + 1,
                        max_retries=max_guided_generation_retries,
                        llm_output=llm_output,
                        prompt_with_exp=latest_prompt_with_exp,
                        prompt_without_exp=latest_prompt_without_exp,
                        injected_experience=transient_state_experience,
                        injection_applied=transient_injection_applied,
                    )
                    if state_tool_trace is not None:
                        state_tool_trace["tool_call_count"] += 1
                        state_tool_trace["guided_retries"].append({
                            "retry_idx": guided_retry_idx + 1,
                            "tool_names": self.exp_worker.get_called_tool_names(llm_output),
                            "tool_payload": self.exp_worker.get_experience_tool_payload(llm_output),
                            "llm_output": llm_output,
                        })
                    logger.warning(
                        f"Assistant requested get_experience_guidance again after transient injection "
                        f"at step {act_step}, retry {guided_retry_idx + 1}/{max_guided_generation_retries}."
                    )

                if not generation_succeeded:
                    logger.warning(
                        f"Generation failure at step {act_step}: assistant still called "
                        f"get_experience_guidance after {max_guided_generation_retries} retries."
                    )
                    err_in_generating = True
                    latest_prompt_with_exp = ""
                    latest_prompt_without_exp = ""
                    if final_step_input_message_arr:
                        latest_prompt_with_exp = str(final_step_input_message_arr[-1].get("content", ""))
                    if step_input_message_arr:
                        latest_prompt_without_exp = str(step_input_message_arr[-1].get("content", ""))
                    mark_generation_failure(
                        act_step,
                        "experience_guidance_retry_exceeded",
                        stage=f"guided_retry_{max_guided_generation_retries}",
                        prompt_with_exp=latest_prompt_with_exp,
                        prompt_without_exp=latest_prompt_without_exp,
                        llm_output=llm_output,
                    )
                    if state_tool_trace is not None:
                        state_tool_trace["final_status"] = "retry_exceeded"
                        state_tool_trace["final_llm_output"] = llm_output
                        state_tool_trace["stop_reason"] = "experience_guidance_retry_exceeded"
                        write_state_tool_trace(act_step, state_tool_trace)
                    stop_reason = "experience_guidance_retry_exceeded"
                    break

                if generation_succeeded and transient_injection_applied:
                    self.cmt.metadata["state_experience_tool_injections"] += 1
                    latest_prompt_with_exp = ""
                    latest_prompt_without_exp = ""
                    if final_step_input_message_arr:
                        latest_prompt_with_exp = str(final_step_input_message_arr[-1].get("content", ""))
                    if step_input_message_arr:
                        latest_prompt_without_exp = str(step_input_message_arr[-1].get("content", ""))
                    self.exp_worker.record_experience_usage(
                        formatted_experience=transient_state_experience,
                        traj_exp_config=traj_exp_config,
                        task_id=task_id,
                        prompt_with_exp=latest_prompt_with_exp,
                        prompt_without_exp=latest_prompt_without_exp,
                    )
                if state_tool_trace is not None:
                    state_tool_trace["final_status"] = "guided_generation_succeeded"
                    state_tool_trace["final_llm_output"] = llm_output
            elif should_apply_no_tool_second_chance(act_step):
                no_tool_second_chance_retries = self.exp_worker.get_no_tool_second_chance_retries()
                if no_tool_second_chance_retries > 0:
                    record_no_tool_second_chance(act_step, no_tool_second_chance_retries)
                    for _ in range(no_tool_second_chance_retries):
                        llm_output = self.llm_chat_fn(final_step_input_message_arr, request_id=request_id)
                        if "content" not in llm_output or llm_output["content"] is None:
                            llm_output["content"] = ""
            if (stop is not None) and stop[thread_index]:  # Check if the thread should stop (because other threads have completed, making this thread useless)
                self.cmt.discarded = True
                stop_reason = "discarded_by_parallel_stop"
                break
            
            # 6. 💾 save llm output
            self.cmt.save_llm_output(llm_output, input_msg_ref=final_step_input_message_arr)  # ⭐ Save the LLM output
            tmux['token'][thread_index] += self.cmt.generated_token_cnt

            # 7. 🌍 world interaction
            world_interaction_content = ""
            try:
                world_interaction_content = self.cmt.prepare_world_interaction()
                env_output = env.step(instance_id, {"content": world_interaction_content, "role": "assistant"})  # ⭐ Interact with the environment
                assert len(env_output['state'])==1
                env_output["state"] = env_output["state"][0]
                if env_output["state"]["role"] == "tool":
                    env_output["state"] = convert_tool_to_user_message(env_output["state"], self.tokenizer, format="qwen")
                
                if self.console_debug_mode:
                    print_listofdict(
                        final_step_input_message_arr +
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
            completed_env_steps = act_step + 1
            if state_tool_trace is not None:
                state_tool_trace["post_guidance_action"] = world_interaction_content
                state_tool_trace["env_output"] = env_output
                state_tool_trace["context_after_step"] = self.cmt.prepare_previous_context(mod="future")
                state_tool_trace["final_status"] = "env_step_completed"
                write_state_tool_trace(act_step, state_tool_trace)

            # 9. 🔚 determine if the episode is terminated
            self.cmt.is_terminated = env_output["is_terminated"]
            if self.cmt.is_terminated or err_in_env:
                if err_in_env:
                    stop_reason = "env_error"
                elif self.cmt.is_terminated:
                    stop_reason = "env_terminated"
                break
        else:
            stop_reason = "max_steps_reached"

        tmux['step'][thread_index] = -1
        self.cmt.metadata["completed_env_steps"] = completed_env_steps
        self.cmt.metadata["max_allowed_steps"] = self.max_steps
        self.cmt.metadata["hit_max_steps"] = (stop_reason == "max_steps_reached")
        if stop_reason is not None:
            self.cmt.metadata["stop_reason"] = stop_reason

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
        self.cmt.metadata["reward_outcome"] = float(self.cmt.reward.outcome)
        self.cmt.metadata["reward_success_rate"] = float(self.cmt.reward.success_rate)
        self.cmt.metadata["reward_description"] = self.cmt.reward.description
        
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
                self.artifact_recorder.write_state_experience_tool_trajectory(
                    task_id=task_id,
                    traj_id=traj_id,
                    query=query,
                    state_tool_ablation_mode=str(self.cmt.metadata.get("state_tool_ablation_mode", "standard")),
                    used_state_experience_tool=bool(self.cmt.metadata.get("used_state_experience_tool", False)),
                    state_experience_tool_calls=int(self.cmt.metadata.get("state_experience_tool_calls", 0) or 0),
                    state_experience_tool_call_steps=list(self.cmt.metadata.get("state_experience_tool_call_steps", []) or []),
                    state_experience_tool_injections=int(self.cmt.metadata.get("state_experience_tool_injections", 0) or 0),
                    state_experience_tool_nonempty_retrievals=int(self.cmt.metadata.get("state_experience_tool_nonempty_retrievals", 0) or 0),
                    state_experience_tool_empty_results=int(self.cmt.metadata.get("state_experience_tool_empty_results", 0) or 0),
                    state_experience_tool_overflow_reverts=int(self.cmt.metadata.get("state_experience_tool_overflow_reverts", 0) or 0),
                    state_experience_tool_repeat_calls_after_injection=int(self.cmt.metadata.get("state_experience_tool_repeat_calls_after_injection", 0) or 0),
                    state_experience_tool_retrieved_token_count=int(self.cmt.metadata.get("state_experience_tool_retrieved_token_count", 0) or 0),
                    completed_env_steps=int(self.cmt.metadata.get("completed_env_steps", 0) or 0),
                    stop_reason=str(self.cmt.metadata.get("stop_reason", "")),
                    generation_failure=bool(self.cmt.metadata.get("generation_failure", False)),
                    is_terminated=bool(self.cmt.is_terminated),
                    success=bool(self.cmt.reward is not None and self.cmt.reward.outcome > 0),
                    reward_value=float(self.cmt.reward.outcome) if self.cmt.reward is not None else 0.0,
                    reward_success_rate=float(self.cmt.reward.success_rate) if self.cmt.reward is not None else 0.0,
                    reward_description=self.cmt.reward.description if self.cmt.reward is not None else "",
                    experience_tool_call_issues=list(self.cmt.metadata.get("experience_tool_call_issues", []) or []),
                    tool_traces=list(self.cmt.metadata.get("state_experience_tool_traces", []) or []),
                )
            except Exception as e:
                import logging
                logging.warning(f"Failed to record rollout/reward: {e}")

        with log_generate_lock:
            self.cmt.generate_log(task_id=task_id)  # ⭐ Generate the log for the task


        return self.cmt
