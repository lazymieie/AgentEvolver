import json
import time
from typing import Any, Sequence

from loguru import logger
from pydantic import Field

from agentevolver.schema.trajectory import Trajectory
from agentevolver.utils.http_client import HttpClient


class StateEMClient(HttpClient):
    base_url: str = Field(default="http://localhost:8002")
    timeout: int = Field(default=1200, description="request timeout, second")

    def _serialize_query(self, state: Any) -> str:
        if isinstance(state, Trajectory):
            return json.dumps(state.steps, ensure_ascii=False)
        if isinstance(state, (list, dict)):
            return json.dumps(state, ensure_ascii=False)
        return str(state)

    def _serialize_summary_item(self, state: Any) -> dict:
        if isinstance(state, Trajectory):
            return {
                "messages": state.steps,
                "score": state.reward.outcome if state.reward is not None else 0.0,
            }
        if isinstance(state, dict):
            return state
        if isinstance(state, list):
            return {"messages": state, "score": 0.0}
        return {"content": str(state), "score": 0.0}

    def call_context_generator(
        self,
        state: Trajectory | dict | list | str,
        retrieve_top_k: int = 1,
        workspace_id: str = "default",
        **kwargs,
    ) -> str:
        """
        Retrieve state experience from the remote state-memory service.
        """
        start_time = time.time()
        self.url = self.base_url + "/retrieve_state_memory"
        json_data = {
            "query": self._serialize_query(state),
            "top_k": retrieve_top_k,
            "workspace_id": workspace_id,
        }
        response = self.request(json_data=json_data, headers={"Content-Type": "application/json"})
        if response is None:
            logger.warning("error call_context_generator for state memory")
            return ""

        if isinstance(state, Trajectory):
            state.metadata["context_time_cost"] = time.time() - start_time
        return response.get("answer", "")

    def call_summarizer(
        self,
        states: Sequence[Trajectory | dict | list | str] | None = None,
        workspace_id: str = "default",
        **kwargs,
    ):
        """
        Send state records to the summary_state_memory endpoint.
        """
        start_time = time.time()
        if states is None:
            states = kwargs.pop("trajectories", None)
        if states is None:
            raise ValueError("states or trajectories must be provided")
        self.url = self.base_url + "/summary_state_memory"
        json_data = {
            "trajectories": [self._serialize_summary_item(state) for state in states],
            "workspace_id": workspace_id,
        }
        try:
            response = self.request(json_data=json_data, headers={"Content-Type": "application/json"})
            if response is None:
                logger.warning("error call_summarizer for state memory: response is None")
                return "", time.time() - start_time
            return response, time.time() - start_time
        except Exception as e:
            err_msg = str(e).lower()
            if "content_filter" in err_msg or "responsibleai" in err_msg or "self_harm" in err_msg:
                logger.warning(f"Content filter error in state call_summarizer (states will be skipped): {e}")
            else:
                logger.error(f"Error in state call_summarizer: {e}")
            return "", time.time() - start_time
