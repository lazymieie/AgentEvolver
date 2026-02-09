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
import time
from concurrent.futures import Future
from typing import Optional, List, Union

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
        self.em_client = EMClient(base_url=self.reme_config.base_url)
    
    def summarize_in_batch(self, trajectories: List[Trajectory]) -> None:
        trajectories_sorted = sorted(trajectories, key=lambda traj: traj.task_id)
        grouped_trajectories = [list(group) for key, group in groupby(trajectories_sorted, key=lambda traj: traj.task_id)]
        batch_size = self.exp_manager_config.summary_batch_size
        all_batches = []
        for group in grouped_trajectories:
            for i in range(0, len(group), batch_size):
                all_batches.append(group[i:i + batch_size])
        
        futures = []
        for batch in all_batches:
            future = self.thread_pool.submit(
                self.em_client.call_summarizer,
                trajectories=batch,
                workspace_id=self.reme_config.workspace_id
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
    def __init__(self, config: DictConfig):
        """
        Initializes the ExperienceWorker with the provided configuration.

        Args:
            config (DictConfig): Configuration settings for the experience worker.
        """
        self.config: DictConfig = config
        self.experience_template = self.config.exp_manager.experience_template
        # artifact_recorder will be set by ExperienceManager if available
        self.artifact_recorder = None
    
    def manage_rollout_context(self, init_messages: List[dict], traj_exp_config: TrajExpConfig) -> Tuple[List[dict], TrajExpConfig]:
        """
        Manages the context for the rollout phase, potentially adding historical experience.

        Args:
            init_messages (List[dict]): Initial messages for the rollout.
            traj_exp_config (TrajExpConfig): Configuration for the trajectory experience.

        Returns:
            Tuple[List[dict], TrajExpConfig]: Updated messages and modified trajectory experience config.
        """
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
        formatted_experience = self.experience_template.format(history_experience)
        new_content = formatted_experience + trajectory.steps[-1]["content"]
        original_content = trajectory.steps[-1]["content"]
        trajectory.steps[-1]["content"] = new_content
        traj_exp_config.experience_list = traj_exp_config.experience_list + [formatted_experience]

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
            self.em_client = EMClient(
                base_url=self.config.exp_manager.reme.base_url
            )



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

        if metadata_config.get("task_train_mode", "discard") == "discard": 
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

