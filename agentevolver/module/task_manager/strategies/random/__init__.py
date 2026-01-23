import copy
import os
import time
from typing import Callable, NotRequired, Optional, Sequence, TypedDict, Unpack
import uuid

from loguru import logger

from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from agentevolver.module.exp_manager.exp_manager import TrajExpConfig
from agentevolver.module.task_manager.agent_flow import ModifiedAgentFlow
from agentevolver.module.task_manager.base import LlmClient
from agentevolver.module.env_manager.env_worker import EnvWorker
from agentevolver.module.task_manager.strategies.common.prompts.prompt_explore import get_agent_interaction_system_prompt
from agentevolver.module.task_manager.strategies.common.prompts.prompt_summarize import (
    get_task_summarize_prompt,
    parse_tasks_from_response,
)
from agentevolver.module.task_manager.strategies import TaskExploreStrategy
from agentevolver.module.task_manager.prelude_profiles import bfcl, appworld
from agentevolver.schema.task import Task, TaskObjective
from agentevolver.schema.trajectory import Trajectory



class LlmRandomSamplingExploreStrategyProps(TypedDict):
    max_explore_step: int
    max_llm_retries: int
    env_url: str
    exploration_llm_temperature: NotRequired[float]
    exploration_llm_top_p: NotRequired[float]
    exploration_llm_top_k: NotRequired[int]
    
    

class LlmRandomSamplingExploreStrategy(TaskExploreStrategy):
    def __init__(self, * , tokenizer, config,**kwargs: Unpack[LlmRandomSamplingExploreStrategyProps]):
        self._tokenizer = tokenizer
        # this is used in other classes
        self._config = config
        
        self._max_llm_retries = kwargs.get("max_llm_retries", 3)
        self._max_explore_step = kwargs.get("max_explore_step", 10)
        self._env_service_url = kwargs.get("env_url")
        
        self._exploration_llm_temperature=kwargs.get("exploration_llm_temperature", 1.0)
        self._exploration_llm_top_p=kwargs.get("exploration_llm_top_p", 1.0)
        self._exploration_llm_top_k=kwargs.get("exploration_llm_top_k", 1)
        
    
    def explore(self, task: Task, data_id: str, rollout_id: str) -> list[Trajectory]:
        env_worker = EnvWorker(
            task=task,
            config=self._config, # FIXME must use these parameters, and their default values are incorrect
            thread_index=0,
            tokenizer=self._tokenizer
        )
        llm_chat_fn = self._get_llm_chat_fn(self.llm_client_explore,
            sampling_params={
                "temperature": self._exploration_llm_temperature,
                "top_p": self._exploration_llm_top_p,
                "top_k": self._exploration_llm_top_k,
            }
        )
        agent_flow: BaseAgentFlow = ModifiedAgentFlow(
            enable_context_generator=False,
            llm_chat_fn=llm_chat_fn,
            tokenizer=self._tokenizer,
            config=self._config,
        )
        agent_flow.max_steps = self._max_explore_step  # this is ugly
        agent_flow.max_model_len=102400 # TODO max len
        
        
        traj = env_worker.execute(
            data_id=data_id,
            rollout_id=rollout_id,
            traj_exp_config=TrajExpConfig(add_exp=False),
            agent_flow=agent_flow,
            tmux={
                'step':[0],
                'token':[0],
            },
            stop=[False], # this method could be refactored
            system_prompt=get_agent_interaction_system_prompt(self.env_profile),
        )

        return [traj]
#gjx   
    # def summarize(self, task: Task, trajectory: Trajectory) -> list[TaskObjective]:
    #     llm_fn = self._get_llm_chat_fn(
    #         self.llm_client_summarize,
    #         sampling_params={
    #             "temperature": self._exploration_llm_temperature,
    #             "top_p": self._exploration_llm_top_p,
    #             "top_k": self._exploration_llm_top_k,
    #         }
    #     )
    #     old_objectives = self._old_retrival.retrieve_objectives(task)
    #     # mask information that may include the real query
    #     # [0]: system prompt, [1]: may be user query or system prompt in user role, [2]: user query
    #     trajectory.steps[1]['content'] = '[MASKED]'
    #     trajectory.steps[2]['content'] = "[MASKED]"
        
    #     system_prompt, user_prompt = get_task_summarize_prompt(
    #         [trajectory], old_objectives, self.env_profile
    #     )
    #     messages = [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt},
    #     ]
    #     llm_output = llm_fn(messages=messages)["content"]
        
    #     task=task.copy()
    #     task.evaluator='synthetic'
    #     tasks = parse_tasks_from_response(task, llm_output)
    #     return tasks
    def summarize(self, task: Task, trajectory: Trajectory) -> list[TaskObjective]:
        import os, time
        from loguru import logger

        llm_fn = self._get_llm_chat_fn(
            self.llm_client_summarize,
            sampling_params={
                "temperature": self._exploration_llm_temperature,
                "top_p": self._exploration_llm_top_p,
                "top_k": self._exploration_llm_top_k,
            }
        )

        old_objectives = self._old_retrival.retrieve_objectives(task)

        # ⚠️ 建议 deepcopy，避免污染原 trajectory（不是主因，但好习惯）
        import copy
        trajectory = copy.deepcopy(trajectory)
        trajectory.steps[1]["content"] = "[MASKED]"
        trajectory.steps[2]["content"] = "[MASKED]"

        system_prompt, user_prompt = get_task_summarize_prompt(
            [trajectory], old_objectives, self.env_profile
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # ===== 调模型 =====
        resp = llm_fn(messages=messages)
        llm_output = resp.get("content", "")

        # ===== 🔍 DEBUG：打印 & 落盘 =====
        logger.warning(
            f"[SUM DEBUG] task_id={task.task_id} llm_output_len={len(llm_output)}"
        )

        logger.warning("[SUM DEBUG] llm_output HEAD ↓↓↓")
        logger.warning(llm_output[:800] if llm_output else "<<< EMPTY >>>")

        logger.warning("[SUM DEBUG] llm_output TAIL ↑↑↑")
        logger.warning(llm_output[-800:] if llm_output else "<<< EMPTY >>>")

        dump_path = f"/tmp/summarize_llm_output_{os.getpid()}_{int(time.time())}.txt"
        try:
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write(llm_output or "")
            logger.warning(f"[SUM DEBUG] raw llm_output dumped to {dump_path}")
        except Exception as e:
            logger.warning(f"[SUM DEBUG] dump failed: {e}")

        # ===== 解析 =====
        task = task.copy()
        task.evaluator = "synthetic"

        try:
            tasks = parse_tasks_from_response(task, llm_output)
        except Exception as e:
            logger.exception(f"[SUM DEBUG] parse_tasks_from_response crashed: {e}")
            return []

        logger.warning(
            f"[SUM DEBUG] task_id={task.task_id} parsed_objectives={len(tasks)}"
        )

        return tasks

    
    def _get_llm_chat_fn(self, llm_client:LlmClient, sampling_params: Optional[dict] = None) -> Callable:
        def llm_chat(
            messages: list[dict[str, str]],
            custom_sampling_params: Optional[dict] = None,
            request_id: Optional[str] = None,
        ) -> dict:
            """
            input messages: [{"role": "system", "value": "..."}, {"role": "user", "value": "..."}]
            output messages: [{"role": "assistant", "value": "..."}]
            """
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            # output_messages = []
            input_messages = copy.deepcopy(messages)
            res = None
            for i in range(self._max_llm_retries):
                try:
                    res = llm_client.chat(
                        messages=input_messages, sampling_params=updated_sampling_params
                    )
                    if res is not None and res!="":
                        break

                except Exception as e:
                    logger.exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(2**i)

            # assert res is not None and res!="", f"LLM client failed to chat"
            if not res:
                logger.error("LLM client returned empty. Skip this rollout by returning placeholder.")
                return {
                    "role": "assistant",
                    "content": "[LLM_EMPTY]"   # 或者 "[CONTENT_FILTERED]" / "[HIT_LIMIT]"
                }

        
            return {
                "role": "assistant",
                "content": res,
            }

        return llm_chat