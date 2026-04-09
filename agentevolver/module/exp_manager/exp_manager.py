import json
import random
import re
from loguru import logger
from dataclasses import dataclass, field
from omegaconf import DictConfig
from typing import List, Dict, Any, Optional, Literal, Tuple
from itertools import groupby
from concurrent.futures import as_completed, Future
from concurrent.futures.thread import ThreadPoolExecutor
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
from agentevolver.client.em_client import EMClient
from agentevolver.client.state_em_client import StateEMClient
import time
from concurrent.futures import Future
from typing import Optional, List, Union


EXPERIENCE_GUIDANCE_TOOL = {
    "name": "get_experience_guidance",
    "description": "Call this tool IMMEDIATELY when you encounter an API error, missing information, or feel stuck. It retrieves actionable experience from the memory database to guide your next step.",
    "parameters": {
        "type": "object",
        "properties": {
            "current_intent": {
                "type": "string",
                "description": "What specific subtask are you trying to complete right now? (e.g., 'Retrieve invoice for booked flight')."
            },
            "last_action": {
                "type": "string",
                "description": "The exact tool name and parameters you just used (e.g., 'retrieve_invoice with insurance_id=123')."
            },
            "current_observation": {
                "type": "string",
                "description": "The exact error message, unexpected result, or current environment state."
            },
            "issue_type": {
                "type": "string",
                "enum": [
                    "api_error",
                    "stuck_in_loop",
                    "missing_parameter",
                    "uncertain_next_step",
                    "constraint_violation"
                ],
                "description": "Categorize the type of obstacle you are currently facing."
            }
        },
        "required": ["current_intent", "last_action", "current_observation", "issue_type"]
    }
}

STATE_TOOL_ABLATION_DEFAULT_MODE = "standard"
STATE_TOOL_ABLATION_TOOL_MODES = {"tool_empty", "tool_fallback", "tool_real"}
STATE_TOOL_ABLATION_NO_TOOL_SECOND_CHANCE_MODES = {
    "all_second_chance_no_tool",
    "random_second_chance_no_tool",
}


def get_state_tool_ablation_mode(config: DictConfig) -> str:
    ablation_config = getattr(getattr(config, "exp_manager", None), "state_tool_ablation", None)
    mode = getattr(ablation_config, "mode", STATE_TOOL_ABLATION_DEFAULT_MODE)
    if mode is None:
        return STATE_TOOL_ABLATION_DEFAULT_MODE
    return str(mode).strip().lower() or STATE_TOOL_ABLATION_DEFAULT_MODE


def uses_state_tool_guidance(config: DictConfig) -> bool:
    mode = get_state_tool_ablation_mode(config)
    if mode != STATE_TOOL_ABLATION_DEFAULT_MODE:
        return mode in STATE_TOOL_ABLATION_TOOL_MODES
    reme_config = getattr(getattr(config, "exp_manager", None), "reme", None)
    return bool(getattr(reme_config, "enable_state_tool_retrieval", False))

@dataclass
class TaskExpConfig:
    add_exp: List[bool]  #长度等于 rollout_n
    train_mode: str = "discard"     # "keep" | "discard" 训练时是否丢弃经验

@dataclass
class TrajExpConfig:
    add_exp: bool = True
    train_mode: str = "discard"
    task_id: str = ""
    data_id: str = ""
    rollout_id: str = ""
    query: str = ""
    mode: str = "sample"            # "sample" | "validate"
    experience_list: List[str] = field(default_factory=list)



class ExperienceManager(object):

    def __init__(self, config: DictConfig):
        """
        Initializes the ExperienceManager with the provided configuration.

        Args:
            config (DictConfig): The configuration dictionary containing settings for the experience manager, rollout, and other components.
        """
        self.config: DictConfig = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.exp_manager_config = config.exp_manager
        self.reme_config = config.exp_manager.reme

        self.val_rollout_mode = self.exp_manager_config.val_rollout_mode
        self.train_rollout_mode = self.exp_manager_config.train_rollout_mode
        self.rollout_ratio = self.exp_manager_config.rollout_ratio
        self.train_sample_mode = self.exp_manager_config.train_sample_mode
        self.train_sample_keepratio = self.exp_manager_config.train_sample_keepratio

        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool.max_workers)
        if self._use_state_tool_experience():
            self.em_client = StateEMClient(base_url=self._get_reme_base_url())
        else:
            self.em_client = EMClient(base_url=self._get_reme_base_url())

    def _use_state_tool_experience(self) -> bool:
        return uses_state_tool_guidance(self.config)

    def _get_reme_base_url(self) -> str:
        if self._use_state_tool_experience():
            return getattr(self.reme_config, "state_base_url", "http://127.0.0.1:8002")
        return self.reme_config.base_url

    def get_experience_pool_mode(self) -> str:
        return "state" if self._use_state_tool_experience() else "task"
    
    def summarize_in_batch(
        self,
        trajectories: List[Trajectory],
        request_timeout: Optional[float] = None,
        wait_indefinitely: bool = False,
        max_concurrent_batches: Optional[int] = None,
    ) -> None:
        trajectories_sorted = sorted(trajectories, key=lambda traj: traj.task_id)
        grouped_trajectories = [list(group) for key, group in groupby(trajectories_sorted, key=lambda traj: traj.task_id)]
        batch_size = self.exp_manager_config.summary_batch_size
        all_batches = []
        for group in grouped_trajectories:
            for i in range(0, len(group), batch_size):
                all_batches.append(group[i:i + batch_size])

        if max_concurrent_batches is not None and max_concurrent_batches <= 1:
            for batch_idx, batch in enumerate(all_batches, 1):
                try:
                    self.em_client.call_summarizer(
                        trajectories=batch,
                        workspace_id=self.reme_config.workspace_id,
                        request_timeout=request_timeout,
                        wait_indefinitely=wait_indefinitely,
                    )
                    logger.info(
                        f"[SummaryInit] completed batch {batch_idx}/{len(all_batches)} "
                        f"sequentially with {len(batch)} trajectories"
                    )
                except Exception as e:
                    err_msg = str(e).lower()
                    if "content_filter" in err_msg or "responsibleai" in err_msg or "self_harm" in err_msg:
                        logger.warning(f"Content filter error in summary task (trajectories will be skipped): {e}")
                    else:
                        logger.error(f"Error in summary task: {e}")
            return

        executor = self.thread_pool
        owns_executor = False
        if max_concurrent_batches is not None and max_concurrent_batches > 1:
            executor = ThreadPoolExecutor(max_workers=max_concurrent_batches)
            owns_executor = True

        futures = []
        try:
            for batch in all_batches:
                future = executor.submit(
                    self.em_client.call_summarizer,
                    trajectories=batch,
                    workspace_id=self.reme_config.workspace_id,
                    request_timeout=request_timeout,
                    wait_indefinitely=wait_indefinitely,
                )
                futures.append(future)

            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    err_msg = str(e).lower()
                    # 处理 Azure OpenAI 内容过滤错误
                    if "content_filter" in err_msg or "responsibleai" in err_msg or "self_harm" in err_msg:
                        logger.warning(f"Content filter error in summary task (trajectories will be skipped): {e}")
                        # 继续执行，不中断整个流程
                    else:
                        logger.error(f"Error in summary task: {e}")
        finally:
            if owns_executor:
                executor.shutdown(wait=True)
        
        return

    # def submit_summary_task(self, trajectories: List[Trajectory], global_steps: int) -> Optional[Future]:
    #     """
    #     Submits a summary task to the thread pool for asynchronous processing.

    #     Args:
    #         trajectories (List[Trajectory]): A list of trajectory objects to be summarized.
    #         global_steps (int): The current global step count used to determine task submission timing.

    #     Returns:
    #         Optional[Future]: A Future object representing the submitted task, or None if the task
    #                         should not be submitted or submission fails.
    #     """
    #     if not self._should_submit_summary(global_steps):
    #         return None
        
    #     try:
    #         summary_task = self.thread_pool.submit(
    #             self.em_client.call_summarizer,
    #             trajectories=trajectories,
    #             workspace_id=self.reme_config.workspace_id
    #         )
    #         print(f"[Summary] Async task submitted at step {global_steps}")
    #         return summary_task
    #     except Exception as e:
    #         print(f"[Summary] Failed to submit task: {e}")
    #         return None
    def submit_summary_task(self, trajectories: List[Trajectory], global_steps: int) -> Optional[List[Future]]:
        """
        Submits summary tasks to the thread pool for asynchronous processing in batches.

        Args:
            trajectories (List[Trajectory]): A list of trajectory objects to be summarized.
            global_steps (int): The current global step count used to determine task submission timing.

        Returns:
            Optional[List[Future]]: A list of Future objects representing the submitted batch tasks, 
                                   or None if the task should not be submitted or submission fails.
        """
        if not self._should_submit_summary(global_steps):
            return None
        
        try:
            # Get batch size from config, default to 32 if not specified
            batch_size = 8
            
            # Split trajectories into batches
            task_futures = []
            total_batches = (len(trajectories) + batch_size - 1) // batch_size  # Ceiling division
            
            for i in range(0, len(trajectories), batch_size):
                batch = trajectories[i:i + batch_size]
                batch_num = i // batch_size + 1
                future = self.thread_pool.submit(
                    self.em_client.call_summarizer,
                    trajectories=batch,
                    workspace_id=self.reme_config.workspace_id
                )
                task_futures.append(future)
                print(f"[Summary] Batch {batch_num}/{total_batches} ({len(batch)} trajectories) submitted at step {global_steps}")
            
            print(f"[Summary] Total {total_batches} async batch tasks submitted at step {global_steps}")
            return task_futures
        except Exception as e:
            print(f"[Summary] Failed to submit tasks: {e}")
            return None

    def _should_submit_summary(self, global_steps: int) -> bool:
        """
        Determines whether a summary task should be submitted based on configuration settings.

        Args:
            global_steps (int): The current global step count.

        Returns:
            bool: True if the summary task should be submitted, False otherwise.
        """
        return (
            self.reme_config.enable_summarizer
            and self.reme_config.updated_freq
            and global_steps % self.reme_config.updated_freq == 0
        )
    
    def collect_summary_result(
        self,
        summary_task: Optional[Union[List[Future], Future]],
        timeout: Optional[float] = None,
    ) -> Optional[float]:
        """
        Collects the result from submitted summary tasks, waiting for all batches to complete.

        Args:
            summary_task:
                - Optional[List[Future]]: list of batch futures
                - Optional[Future]: a single future (backward compatibility)
            timeout:
                - Per-batch timeout in seconds for Future.result().
                - If None, uses self.reme_config.summary_timeout if exists, else waits indefinitely.

        Returns:
            Optional[float]:
                Sum of time_cost reported by each completed batch (NOT wall clock time).
                Returns None if all batches fail / timeout or summary_task is None.
        """
        if summary_task is None:
            return None

        # Backward compatibility: handle single Future
        if isinstance(summary_task, Future):
            summary_task = [summary_task]

        if not summary_task:
            return None

        # Default timeout from config if not provided
        if timeout is None:
            timeout = 300.0

        num_tasks = len(summary_task)
        print(f"[Summary] Waiting for {num_tasks} batch tasks to complete... (per-batch timeout={timeout})")

        # Wall time measurement (how long we actually block this step)
        wall_t0 = time.perf_counter()

        total_time_cost = 0.0          # sum of per-batch time_cost
        completed_count = 0
        failed_count = 0
        timeout_count = 0

        for idx, task in enumerate(summary_task, 1):
            try:
                # Wait this batch (blocking). If timeout is None -> wait indefinitely.
                summarizer_response, time_cost = task.result(timeout=timeout)
                total_time_cost += float(time_cost)
                completed_count += 1
                print(f"[Summary] Batch {idx}/{num_tasks} completed in {float(time_cost):.2f}s")
            except Exception as e:
                # Distinguish timeout if it is a concurrent.futures.TimeoutError
                # (avoid importing TimeoutError name clash with built-in)
                if e.__class__.__name__ == "TimeoutError":
                    timeout_count += 1
                    print(f"[Summary] Batch {idx}/{num_tasks} timed out after {timeout}s")
                else:
                    failed_count += 1
                    print(f"[Summary] Batch {idx}/{num_tasks} failed: {e}")

        wall = time.perf_counter() - wall_t0
        print(
            f"[Summary] Done. completed={completed_count}/{num_tasks}, "
            f"timeout={timeout_count}, failed={failed_count}, "
            f"sum_time_cost={total_time_cost:.2f}s, wall={wall:.2f}s"
        )

        # If you want to log wall time somewhere, do it here, e.g.:
        # self._last_summary_wall = wall

        return total_time_cost if completed_count > 0 else None
    

    # def collect_summary_result(self, summary_task: Optional[Future]) -> Optional[float]:
    #     """
    #     Collects the result from a submitted summary task.

    #     Args:
    #         summary_task (Optional[Future]): The Future object representing the summary task to collect.
    #         timeout (Optional[float]): Maximum time in seconds to wait for the task completion.
    #                                 Defaults to None (wait indefinitely).

    #     Returns:
    #         Optional[float]: The time cost of the summary task in seconds, or None if the task
    #                         is None, times out, or encounters an error.
    #     """
    #     if summary_task is None:
    #         return None
    #     try:
    #         print("[Summary] Waiting for task completion...")
    #         summarizer_response, time_cost = summary_task.result()
    #         print(f"[Summary] Task completed in {time_cost:.2f}s")
    #         return time_cost
    #     except Exception as e:
    #         print(f"[Summary] Task failed: {e}")
    #         return None

    def get_complete_exp_configs(self, tasks: List[Task], mode: Literal["sample", "validate"]) -> List[TaskExpConfig]:
        """
        Generates complete experience configurations for the given tasks.

        Args:
            tasks (List[Task]): A list of Task objects for which to generate configurations.
            mode (Literal["sample", "validate"]): The mode of operation, either "sample" or "validate".

        Returns:
            List[TaskExpConfig]: A list of TaskExpConfig objects with allocated training modes and experience addition settings.
        """
        exp_manager_configs = self.allocate_train_mode(tasks)
        exp_manager_configs = self.allocate_add_exp(exp_manager_configs, mode)
        return exp_manager_configs

    def allocate_train_mode(self, tasks: List[Task]) -> List[TaskExpConfig]:
        """
        Allocates training modes for the given tasks based on the configured training sample experience mode.

        Args:
            tasks (List[Task]): A list of Task objects for which to allocate training modes.

        Returns:
            List[TaskExpConfig]: A list of TaskExpConfig objects with allocated training modes.
        """
        mode_to_ratio = {
            "allkeep": 1.0,
            "alldiscard": 0.0,
            "hybrid": self.train_sample_keepratio
        }
        keep_ratio = mode_to_ratio.get(
            self.train_sample_mode, self.train_sample_keepratio
        )
        keep_count = int(len(tasks) * keep_ratio)
        exp_modes = ['keep'] * keep_count + ['discard'] * (len(tasks) - keep_count)
        random.shuffle(exp_modes)
        return [TaskExpConfig(add_exp=[], train_mode=exp_mode) for exp_mode in exp_modes]
    
    def allocate_add_exp(self, exp_configs: List[TaskExpConfig], mode: Literal["sample", "validate"]) -> List[TaskExpConfig]:
        """
        Allocates experience addition settings for the given tasks based on the mode and configured experience modes.

        Args:
            exp_configs (List[TaskExpConfig]): A list of TaskExpConfig objects to be updated.
            mode (Literal["sample", "validate"]): The mode of operation, either "sample" or "validate".

        Returns:
            List[TaskExpConfig]: An updated list of TaskExpConfig objects with allocated experience addition settings.
        """
        is_validate = mode == "validate"
        rollout_n = self.rollout_config.val_kwargs.n if is_validate else self.rollout_config.n
        exp_mode = self.val_rollout_mode if is_validate else self.train_rollout_mode
        for task_exp_config in exp_configs:
            add_exp_choices = {
                "woexp": [False] * rollout_n,
                "mixed": sorted([i < round(rollout_n*self.rollout_ratio) for i in range(rollout_n)], key=lambda _: random.random()),
                "all": [True] * rollout_n
            }[exp_mode]
            task_exp_config.add_exp = add_exp_choices
        
        return exp_configs




class ExperienceWorker(object):
    def __init__(self, config: DictConfig, tokenizer=None):
        """
        Initializes the ExperienceWorker with the provided configuration.

        Args:
            config (DictConfig): Configuration settings for the experience worker.
        """
        self.config: DictConfig = config
        self.tokenizer = tokenizer
        self.experience_template = self.config.exp_manager.experience_template
        # artifact_recorder will be set by ExperienceManager if available
        self.artifact_recorder = None

    def _use_state_tool_experience(self) -> bool:
        return uses_state_tool_guidance(self.config)

    def get_state_tool_ablation_mode(self) -> str:
        return get_state_tool_ablation_mode(self.config)

    def should_force_state_tool_ablation(self) -> bool:
        return self.get_state_tool_ablation_mode() in STATE_TOOL_ABLATION_TOOL_MODES

    def is_no_tool_second_chance_mode(self) -> bool:
        return self.get_state_tool_ablation_mode() in STATE_TOOL_ABLATION_NO_TOOL_SECOND_CHANCE_MODES

    def get_no_tool_second_chance_mode(self) -> str:
        mode = self.get_state_tool_ablation_mode()
        if mode in STATE_TOOL_ABLATION_NO_TOOL_SECOND_CHANCE_MODES:
            return mode
        return ""

    def get_no_tool_second_chance_ratio(self) -> float:
        ablation_config = getattr(self.config.exp_manager, "state_tool_ablation", None)
        ratio = getattr(ablation_config, "random_second_chance_ratio", 0.0)
        try:
            ratio = float(ratio)
        except (TypeError, ValueError):
            ratio = 0.0
        return max(0.0, min(1.0, ratio))

    def get_no_tool_second_chance_seed(self) -> int:
        ablation_config = getattr(self.config.exp_manager, "state_tool_ablation", None)
        seed = getattr(ablation_config, "random_second_chance_seed", 0)
        try:
            return int(seed)
        except (TypeError, ValueError):
            return 0

    def get_no_tool_second_chance_retries(self) -> int:
        ablation_config = getattr(self.config.exp_manager, "state_tool_ablation", None)
        retries = getattr(ablation_config, "no_tool_second_chance_retries", 1)
        try:
            retries = int(retries)
        except (TypeError, ValueError):
            retries = 1
        return max(0, retries)

    def get_guided_generation_retries(self) -> int:
        ablation_config = getattr(self.config.exp_manager, "state_tool_ablation", None)
        retries = getattr(ablation_config, "guided_generation_retries", 2)
        try:
            retries = int(retries)
        except (TypeError, ValueError):
            retries = 2
        return max(0, retries)

    def get_state_tool_fallback_text(self) -> str:
        ablation_config = getattr(self.config.exp_manager, "state_tool_ablation", None)
        fallback_text = getattr(ablation_config, "fallback_text", "No matching state memories found")
        return str(fallback_text)

    def _get_reme_base_url(self) -> str:
        reme_config = self.config.exp_manager.reme
        if self._use_state_tool_experience():
            return getattr(reme_config, "state_base_url", "http://127.0.0.1:8002")
        return reme_config.base_url

    def _insert_tool_before_tools_end(self, system_content: str) -> str:
        tool_str = json.dumps(EXPERIENCE_GUIDANCE_TOOL, ensure_ascii=False)
        if EXPERIENCE_GUIDANCE_TOOL["name"] in system_content:
            return system_content
        if "</tools>" in system_content:
            return system_content.replace("</tools>", f"{tool_str}\n</tools>", 1)
        return system_content + f"\n<tools>\n{tool_str}\n</tools>\n"

    def _inject_experience_guidance_tool(self, init_messages: List[dict]) -> List[dict]:
        patched_messages = [dict(msg) for msg in init_messages]
        for message in patched_messages:
            if message.get("role") == "system" and isinstance(message.get("content"), str):
                message["content"] = self._insert_tool_before_tools_end(message["content"])
                break
        return patched_messages

    def _build_declarative_query(self, intent: str, action: str, obs: str, issue_type: str) -> str:
        if issue_type == "api_error":
            return f"The agent is trying to {intent} and called {action}, but encountered the API error: {obs}."
        if issue_type == "stuck_in_loop":
            return f"The agent is stuck in a loop while trying to {intent}. It repeatedly called {action} and observed {obs}."
        if issue_type == "missing_parameter":
            return f"The agent is trying to {intent} but is missing required parameters. The last action was {action} and observation was {obs}."
        if issue_type == "uncertain_next_step":
            return f"The agent successfully completed {action} with observation '{obs}', but is uncertain about the next step to achieve {intent}."
        if issue_type == "constraint_violation":
            return f"The agent is trying to {intent} using {action}, but the observation '{obs}' violates business constraints."
        return f"The agent is trying to {intent} and observed {obs}."

    def _extract_experience_tool_payload_from_text(self, content: str) -> Optional[Dict[str, Any]]:
        for raw in self._extract_tool_call_blocks_from_text(content):
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if payload.get("name") != EXPERIENCE_GUIDANCE_TOOL["name"]:
                continue
            arguments = payload.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}
            if isinstance(arguments, dict):
                return arguments
        return None

    def _extract_tool_call_blocks_from_text(self, content: str) -> List[str]:
        text = content or ""
        start_tag = "<tool_call>"
        end_tag = "</tool_call>"
        blocks: List[str] = []
        cursor = 0

        while True:
            start_idx = text.find(start_tag, cursor)
            if start_idx == -1:
                break
            start_idx += len(start_tag)
            end_idx = text.find(end_tag, start_idx)
            if end_idx == -1:
                break
            block = text[start_idx:end_idx].strip()
            if block:
                blocks.append(block)
            cursor = end_idx + len(end_tag)

        return blocks

    def _extract_tool_call_names_from_text(self, content: str) -> List[str]:
        tool_names: List[str] = []
        for raw in self._extract_tool_call_blocks_from_text(content):
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            name = payload.get("name")
            if isinstance(name, str) and name:
                tool_names.append(name)
        return tool_names

    def get_called_tool_names(self, llm_output: Dict[str, Any]) -> List[str]:
        tool_names: List[str] = []
        for tool_call in llm_output.get("tool_calls") or []:
            function_info = tool_call.get("function") or {}
            name = function_info.get("name")
            if isinstance(name, str) and name:
                tool_names.append(name)
        if tool_names:
            return tool_names
        return self._extract_tool_call_names_from_text(llm_output.get("content", ""))

    def has_mixed_experience_and_other_tool_calls(self, llm_output: Dict[str, Any]) -> bool:
        tool_names = self.get_called_tool_names(llm_output)
        if not tool_names:
            return False
        has_experience_tool = EXPERIENCE_GUIDANCE_TOOL["name"] in tool_names
        has_other_tool = any(name != EXPERIENCE_GUIDANCE_TOOL["name"] for name in tool_names)
        return has_experience_tool and has_other_tool

    def _extract_experience_tool_payload(self, llm_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for tool_call in llm_output.get("tool_calls") or []:
            function_info = tool_call.get("function") or {}
            if function_info.get("name") != EXPERIENCE_GUIDANCE_TOOL["name"]:
                continue
            arguments = function_info.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}
            if isinstance(arguments, dict):
                return arguments
        return self._extract_experience_tool_payload_from_text(llm_output.get("content", ""))

    def get_experience_tool_payload(self, llm_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self._extract_experience_tool_payload(llm_output)

    def has_experience_guidance_tool_call(self, llm_output: Dict[str, Any]) -> bool:
        return self._extract_experience_tool_payload(llm_output) is not None

    def render_experience_tool_call(self, llm_output: Dict[str, Any]) -> str:
        payload = self._extract_experience_tool_payload(llm_output)
        if payload is None:
            return llm_output.get("content", "")
        rendered = {
            "name": EXPERIENCE_GUIDANCE_TOOL["name"],
            "arguments": payload,
        }
        return f"<tool_call>\n{json.dumps(rendered, ensure_ascii=False)}\n</tool_call>"

    def prepend_experience_to_latest_message(
        self,
        messages: List[Dict[str, Any]],
        formatted_experience: str,
    ) -> List[Dict[str, Any]]:
        patched_messages = [dict(message) for message in messages]
        if not formatted_experience:
            return patched_messages

        for idx in range(len(patched_messages) - 1, -1, -1):
            content = patched_messages[idx].get("content")
            if not isinstance(content, str):
                continue
            if patched_messages[idx].get("role") != "user":
                continue
            patched_messages[idx]["content"] = formatted_experience + content
            return patched_messages

        if patched_messages and isinstance(patched_messages[-1].get("content"), str):
            patched_messages[-1]["content"] = formatted_experience + patched_messages[-1]["content"]
        return patched_messages

    def record_experience_usage(
        self,
        formatted_experience: str,
        traj_exp_config: TrajExpConfig,
        task_id: str = "unknown",
        prompt_with_exp: str = "",
        prompt_without_exp: str = "",
    ) -> None:
        if not formatted_experience:
            return

        traj_exp_config.experience_list.append(formatted_experience)

        if hasattr(self, 'artifact_recorder') and self.artifact_recorder and self.artifact_recorder.enable:
            try:
                self.artifact_recorder.write_experience_injection(
                    task_id=task_id,
                    injection_template_name=self.experience_template,
                    prompt_with_exp=prompt_with_exp,
                    prompt_without_exp=prompt_without_exp,
                )
            except Exception as e:
                logger.warning(f"Failed to record state experience injection: {e}")

    def _build_state_retrieval_topk(self, history_experience: Any) -> List[Dict[str, Any]]:
        topk_list: List[Dict[str, Any]] = []
        if isinstance(history_experience, list):
            for exp in history_experience:
                if isinstance(exp, dict):
                    topk_list.append({
                        "exp_id": exp.get("id", exp.get("exp_id", "unknown")),
                        "score": exp.get("score", exp.get("similarity", 0.0)),
                        "when_to_use": exp.get("when_to_use", ""),
                        "content": exp.get("content", str(exp)),
                        "source_task_id": exp.get("source_task_id", None),
                        "source_traj_id": exp.get("source_traj_id", None),
                    })
                else:
                    topk_list.append({
                        "exp_id": "unknown",
                        "score": 1.0,
                        "when_to_use": "",
                        "content": str(exp),
                    })
            return topk_list

        if isinstance(history_experience, str) and history_experience:
            topk_list.append({
                "exp_id": "unknown",
                "score": 1.0,
                "when_to_use": "",
                "content": history_experience,
            })
        return topk_list

    def retrieve_state_tool_experience_details(
        self,
        llm_output: Dict[str, Any],
        traj_exp_config: TrajExpConfig,
        task_id: str = "unknown",
    ) -> Dict[str, Any]:
        payload = self._extract_experience_tool_payload(llm_output)
        if payload is None:
            return {
                "payload": None,
                "query": "",
                "topk": [],
                "raw_experience": "",
                "formatted_experience": "",
                "retrieval_mode": self.get_state_tool_ablation_mode(),
            }

        intent = str(payload.get("current_intent", "")).strip()
        action = str(payload.get("last_action", "")).strip()
        obs = str(payload.get("current_observation", "")).strip()
        issue_type = str(payload.get("issue_type", "")).strip()
        query = self._build_declarative_query(intent, action, obs, issue_type)
        ablation_mode = self.get_state_tool_ablation_mode()
        retrieval_mode = ablation_mode if ablation_mode in STATE_TOOL_ABLATION_TOOL_MODES else "tool_real"
        reme_config = self.config.exp_manager.reme

        if retrieval_mode == "tool_empty":
            history_experience: Any = ""
            topk_list: List[Dict[str, Any]] = []
        elif retrieval_mode == "tool_fallback":
            history_experience = self.get_state_tool_fallback_text()
            topk_list = [{
                "exp_id": "ablation_fallback",
                "score": 1.0,
                "when_to_use": "Use only for ablation; indicates retrieval was intentionally replaced with a fixed fallback.",
                "content": history_experience,
                "source_task_id": None,
                "source_traj_id": None,
            }]
        else:
            self._ensure_em_client()
            history_experience = self.em_client.call_context_generator(
                state=query,
                retrieve_top_k=reme_config.retrieve_top_k,
                workspace_id=reme_config.workspace_id,
            )
            topk_list = self._build_state_retrieval_topk(history_experience)

        if hasattr(self, 'artifact_recorder') and self.artifact_recorder and self.artifact_recorder.enable:
            try:
                self.artifact_recorder.write_experience_retrieval(
                    task_id=task_id,
                    query=query,
                    topk=topk_list,
                )
            except Exception as e:
                logger.warning(f"Failed to record state experience retrieval: {e}")

        if not history_experience:
            return {
                "payload": payload,
                "query": query,
                "topk": topk_list,
                "raw_experience": "",
                "formatted_experience": "",
                "retrieval_mode": retrieval_mode,
            }

        if retrieval_mode == "tool_real" and self._should_replace_with_dummy_experience():
            history_experience = self._build_same_token_dummy_experience(str(history_experience))

        formatted_experience = self.experience_template.format(history_experience)
        return {
            "payload": payload,
            "query": query,
            "topk": topk_list,
            "raw_experience": history_experience,
            "formatted_experience": formatted_experience,
            "retrieval_mode": retrieval_mode,
        }

    def retrieve_state_tool_experience(
        self,
        llm_output: Dict[str, Any],
        traj_exp_config: TrajExpConfig,
        task_id: str = "unknown",
    ) -> str:
        details = self.retrieve_state_tool_experience_details(
            llm_output=llm_output,
            traj_exp_config=traj_exp_config,
            task_id=task_id,
        )
        return str(details.get("formatted_experience", ""))

    def _should_replace_with_dummy_experience(self) -> bool:
        return bool(
            getattr(
                self.config.exp_manager,
                "replace_experience_with_same_token_dummy",
                False,
            )
        )

    def _select_repeatable_dummy_fragment(self) -> str:
        configured_fragment = getattr(
            self.config.exp_manager, "same_token_dummy_fragment", None
        )
        candidate_fragments = [configured_fragment, "x", "~", ".", "啊", "哈", "废"]

        if self.tokenizer is None:
            for fragment in candidate_fragments:
                if fragment:
                    return fragment
            return "x"

        for fragment in candidate_fragments:
            if not fragment:
                continue
            token_ids = self.tokenizer.encode(fragment, add_special_tokens=False)
            if len(token_ids) != 1:
                continue
            if self.tokenizer.encode(fragment * 8, add_special_tokens=False) == token_ids * 8:
                return fragment

        vocab_size = getattr(self.tokenizer, "vocab_size", 0) or 0
        for token_id in range(vocab_size):
            piece = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if not piece or piece.strip() == "":
                continue
            if any(ord(ch) < 32 for ch in piece):
                continue
            if self.tokenizer.encode(piece, add_special_tokens=False) != [token_id]:
                continue
            if self.tokenizer.encode(piece * 8, add_special_tokens=False) == [token_id] * 8:
                return piece

        return "x"

    def _build_same_token_dummy_experience(self, original_experience: str) -> str:
        if self.tokenizer is None:
            dummy_fragment = self._select_repeatable_dummy_fragment()
            return dummy_fragment * len(original_experience)

        target_token_count = len(
            self.tokenizer.encode(original_experience, add_special_tokens=False)
        )
        if target_token_count == 0:
            return ""

        dummy_fragment = self._select_repeatable_dummy_fragment()
        dummy_token_ids = self.tokenizer.encode(dummy_fragment, add_special_tokens=False)
        if len(dummy_token_ids) != 1:
            raise ValueError(
                "same_token_dummy_fragment must map to exactly one token for exact replacement."
            )

        dummy_text = dummy_fragment * target_token_count
        actual_token_count = len(
            self.tokenizer.encode(dummy_text, add_special_tokens=False)
        )
        if actual_token_count != target_token_count:
            raise ValueError(
                "Failed to build dummy experience with the same token length."
            )
        return dummy_text
    
    def manage_rollout_context(self, init_messages: List[dict], traj_exp_config: TrajExpConfig) -> Tuple[List[dict], TrajExpConfig]:
        """
        Manages the context for the rollout phase, potentially adding historical experience.

        Args:
            init_messages (List[dict]): Initial messages for the rollout.
            traj_exp_config (TrajExpConfig): Configuration for the trajectory experience.

        Returns:
            Tuple[List[dict], TrajExpConfig]: Updated messages and modified trajectory experience config.
        """
        if self._use_state_tool_experience():
            if self.should_force_state_tool_ablation():
                traj_exp_config.add_exp = True
                return self._inject_experience_guidance_tool(init_messages), traj_exp_config
            if not traj_exp_config.add_exp:
                return init_messages, traj_exp_config
            return self._inject_experience_guidance_tool(init_messages), traj_exp_config

        # check experience conditions
        if not self._should_process_experience(traj_exp_config):
            return init_messages, traj_exp_config
        
        # initialize em client
        self._ensure_em_client()
        
        # construct trajectory
        trajectory = Trajectory(
            data_id=traj_exp_config.data_id,
            rollout_id=traj_exp_config.rollout_id,
            steps=init_messages,
            query=traj_exp_config.query
        )

        # retrieve experience
        reme_config = self.config.exp_manager.reme
        history_experience = self.em_client.call_context_generator(
            trajectory=trajectory,
            retrieve_top_k=reme_config.retrieve_top_k,
            workspace_id=reme_config.workspace_id
        )

        # check empty condition
        if not history_experience:
            logger.info("Experience is empty!")
            return init_messages, traj_exp_config

        # Record experience retrieval
        if hasattr(self, 'artifact_recorder') and self.artifact_recorder and self.artifact_recorder.enable:
            try:
                # Convert history_experience to list format for recording
                topk_list = []
                if isinstance(history_experience, list):
                    for exp in history_experience:
                        if isinstance(exp, dict):
                            topk_list.append({
                                "exp_id": exp.get("id", exp.get("exp_id", "unknown")),
                                "score": exp.get("score", exp.get("similarity", 0.0)),
                                "when_to_use": exp.get("when_to_use", ""),
                                "content": exp.get("content", str(exp)),
                                "source_task_id": exp.get("source_task_id", None),
                                "source_traj_id": exp.get("source_traj_id", None),
                            })
                elif isinstance(history_experience, str):
                    # If it's a string, treat as single experience
                    topk_list.append({
                        "exp_id": "unknown",
                        "score": 1.0,
                        "when_to_use": "",
                        "content": history_experience,
                    })
                
                task_id = getattr(trajectory, 'task_id', traj_exp_config.data_id if hasattr(traj_exp_config, 'data_id') else "unknown")
                self.artifact_recorder.write_experience_retrieval(
                    task_id=task_id,
                    query=trajectory.query,
                    topk=topk_list,
                )
            except Exception as e:
                logger.warning(f"Failed to record experience retrieval: {e}")

        # apply experience to trajectory
        # logger.info(f"Retrieved history experience: {history_experience}")
        if self._should_replace_with_dummy_experience():
            history_experience = self._build_same_token_dummy_experience(
                str(history_experience)
            )

        formatted_experience = self.experience_template.format(history_experience)
        new_content = formatted_experience + trajectory.steps[-1]["content"]
        original_content = trajectory.steps[-1]["content"]
        trajectory.steps[-1]["content"] = new_content
        traj_exp_config.experience_list.append(formatted_experience)

        # Record experience injection
        if hasattr(self, 'artifact_recorder') and self.artifact_recorder and self.artifact_recorder.enable:
            try:
                task_id = getattr(trajectory, 'task_id', traj_exp_config.data_id if hasattr(traj_exp_config, 'data_id') else "unknown")
                self.artifact_recorder.write_experience_injection(
                    task_id=task_id,
                    injection_template_name=self.experience_template,
                    prompt_with_exp=new_content,
                    prompt_without_exp=original_content,
                )
            except Exception as e:
                logger.warning(f"Failed to record experience injection: {e}")

        return trajectory.steps, traj_exp_config
    
    def _should_process_experience(self, traj_exp_config: TrajExpConfig) -> bool:
        """
        Checks if experience processing should be performed.

        Args:
            traj_exp_config (TrajExpConfig): Configuration for the trajectory experience.

        Returns:
            bool: True if experience should be processed, False otherwise.
        """
        return (traj_exp_config.add_exp and
                self.config.exp_manager.reme.enable_context_generator)
    
    def _ensure_em_client(self) -> None:
        """
        Initializes the EM client if it doesn't exist.
        """
        if not hasattr(self, 'em_client'):
            client_cls = StateEMClient if self._use_state_tool_experience() else EMClient
            self.em_client = client_cls(base_url=self._get_reme_base_url())



    def manage_training_context(self, message: str, metadata_config: Dict) -> Tuple[str, str]:
        """
        Extracts and removes experience information from the given message.

        Args:
            message (str): Input message potentially containing experience information.
            metadata_config (Dict): Configuration for the trajectory experience.

        Returns:
            Tuple[str, str]: Extracted experience and the message with experience information removed.
        """
        experience = ""
        cleaned_message = message

        if metadata_config.get("task_train_exp_mode", "discard") == "discard": 
            pattern = re.escape(self.experience_template).replace(r'\{\}', '(.*?)')
            match = re.search(pattern, message, re.DOTALL)
            if match:
                experience = match.group(1)
                cleaned_message = re.sub(pattern, '', message, flags=re.DOTALL)
                
                # Record experience stripping
                if hasattr(self, 'artifact_recorder') and self.artifact_recorder and self.artifact_recorder.enable:
                    try:
                        task_id = metadata_config.get('task_id', 'unknown')
                        # Extract token spans if possible (simplified - would need tokenizer)
                        token_spans = []
                        if match:
                            token_spans.append({
                                "start": match.start(),
                                "end": match.end(),
                            })
                        self.artifact_recorder.write_experience_stripping(
                            task_id=task_id,
                            before=message,
                            after=cleaned_message,
                            token_spans=token_spans if token_spans else None,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to record experience stripping: {e}")

        
        return experience, cleaned_message
