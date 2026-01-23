"""
Inspect what "seed tasks" fetched from EnvService look like.

Usage (PowerShell):
  python scripts/inspect_seed_tasks.py --env_url http://127.0.0.1:8080 --env_type appworld --split train --k 5

What you'll see:
  1) Raw task_id list returned by EnvService (/get_env_profile)
"""

from __future__ import annotations

import argparse
import json
# env_client.py
from typing import Dict, List, Any

import requests
from loguru import logger


class EnvClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self.timeout = 300.0

    def _make_request(
        self,
        endpoint: str,
        env_type: str = "default",
        task_id: str = None,
        instance_id: str = None,
        messages: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
        **kwargs,
    ) -> Dict:
        """
        Handles making a POST request to the specified API endpoint.

        Args:
            endpoint (str): The API endpoint to send the request to.
            env_type (str, optional): The type of environment. Defaults to "default".
            task_id (str, optional): The task ID. Defaults to None.
            instance_id (str, optional): The instance ID. Defaults to None.
            messages (Dict[str, Any], optional): Messages to be sent. Defaults to None.
            params (Dict[str, Any], optional): Additional parameters. Defaults to None.

        Returns:
            Dict: The JSON response from the API.
        """
        url = f"{self.base_url}/{endpoint}"  # ⭐ Constructs the full URL for the request
        data = {
            "env_type": env_type,
            "task_id": task_id,
            "instance_id": instance_id,
            "messages": messages or {},
            "params": params or {},
            **kwargs,
        }
        try:
            response = requests.post(url, json=data, timeout=self.timeout)  # ⭐ Sends the POST request
            response.raise_for_status()
            return response.json()  # ⭐ Parses and returns the JSON response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}, data: {data}")
            raise

    def get_env_profile(
        self, env_type: str, split: str = "train", params: dict | None = None
    ) -> List[str]:
        """
        Retrieves a list of task IDs based on the specified environment type, split, and optional parameters.

        Args:
            env_type (str): The type of the environment.
            split (str, optional): The data split to use. Defaults to "train".
            params (dict | None, optional): Additional parameters for the request. Defaults to None.

        Returns:
            List[str]: A list of task IDs.
        """
        payload: dict = {"env_type": env_type}  # ⭐ Initialize the payload with the environment type
        if params:
            payload["params"] = params  # ⭐ Add additional parameters to the payload if provided
        response = self._make_request(
            endpoint="/get_env_profile", env_type=env_type, params={"split": split}
        )  
        logger.debug(f"get_env_profile split: {split}")
        # ⭐ Make the request to the API endpoint
        return response["data"]  # ⭐ Return the list of task IDs from the response

    def get_tools_info(
        self, instance_id: str, messages: Dict = {}, params: Dict = {}
    ) -> float:
        """
        Retrieves information about the tools in a specific environment instance.

        Args:
            instance_id (str): The ID of the environment instance.
            messages (Dict, optional): Additional messages to be sent with the request. Defaults to {}.
            params (Dict, optional): Additional parameters to be sent with the request. Defaults to {}.

        Returns:
            float: The data from the API response.
        """
        response = self._make_request(
            endpoint="get_info",
            instance_id=instance_id,
            messages=messages,
            params=params,
        )  # ⭐ Make the API request to get tools information
        return response["data"]

    def create_instance(
        self, env_type: str, task_id: str, instance_id: str = None, params: Dict = None
    ) -> dict:
        """
        Creates an environment instance by sending a request to the API.

        Args:
            env_type (str): The type of the environment to be created.
            task_id (str): The unique identifier for the task.
            instance_id (str, optional): The unique identifier for the instance. Defaults to None.
            params (Dict, optional): Additional parameters for the environment creation. Defaults to None.

        Returns:
            dict: The data part of the API response containing information about the created instance.
        """
        response = self._make_request(  # ⭐ Sends the request to the API to create the environment instance
            endpoint="create",
            env_type=env_type,
            task_id=task_id,
            instance_id=instance_id,
            params=params,
        )
        return response["data"]

    def step(self, instance_id: str, action: Dict = {}, params: Dict = {}) -> dict:
        """
        Sends a request to the environment API to execute a step in the specified instance.

        Args:
            instance_id (str): The ID of the environment instance.
            action (Dict, optional): The action to be performed. Defaults to {}.
            params (Dict, optional): Additional parameters for the action. Defaults to {}.

        Returns:
            dict: The data returned from the environment API after executing the step.
        """
        response = self._make_request(
            endpoint="step", instance_id=instance_id, messages=action, params=params
        )  # ⭐ Sends the request to the environment API
        return response["data"]

    def evaluate(
        self, instance_id: str, messages: Dict = {}, params: Dict = {}
    ) -> float:
        """
        Sends a request to evaluate the specified environment instance and returns the evaluation result.

        Args:
            instance_id (str): The ID of the environment instance to be evaluated.
            messages (Dict, optional): A dictionary containing messages for the evaluation. Defaults to {}.
            params (Dict, optional): A dictionary containing additional parameters for the evaluation. Defaults to {}.

        Returns:
            float: The evaluation result.
        """
        response = self._make_request(  # ⭐ Sends the evaluation request to the API
            endpoint="evaluate",
            instance_id=instance_id,
            messages=messages,
            params=params,
        )
        return response["data"]

    def release_instance(self, instance_id: str) -> bool:
        """
        Sends a request to release the specified environment instance.

        Args:
            instance_id (str): The ID of the environment instance to be released.

        Returns:
            bool: True if the release operation was successful, False otherwise.
        """
        response = self._make_request(endpoint="release", instance_id=instance_id)  # ⭐ Send the release request
        return response["success"]


def main():
    """
    Demonstrates the use of EnvClient by performing a sequence of operations:
    - Fetching available tasks for a given environment type
    - Creating an instance based on one of the fetched tasks
    - Stepping through the created instance with a specified action
    - Evaluating the instance
    - Releasing the instance

    This function is intended to be run as a standalone script to test the functionality of the EnvClient.
    """
    client = EnvClient()

    env_type = "bfcl"
    # get the task list
    task_ids = client.get_env_profile(env_type)  # ⭐ Retrieve the list of available tasks for the specified environment type
    print(f"Available tasks: {task_ids}")

    # init instance
    task_id = task_ids[0]
    init_response = client.create_instance(env_type, task_id)  # ⭐ Create an instance using the first available task
    print("init state", init_response)
    instance_id = init_response["info"]["instance_id"]
    query = init_response["state"]
    print(f"Created instance {instance_id} with query: {query}")

    # act
    action = {"role": "assistant", "content": "print('hello appworld!!')"}
    result = client.step(instance_id, action)  # ⭐ Execute an action within the created instance
    print(f"Step result: {result}")

    # evaluate
    score = client.evaluate(instance_id)  # ⭐ Evaluate the current state of the instance
    print(f"Evaluation score: {score}")

    # release instance
    success = client.release_instance(instance_id)  # ⭐ Release the instance, freeing up resources
    print(f"Instance released: {success}")






import argparse
import json


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env_url", type=str, required=True, help="EnvService base url, e.g. http://127.0.0.1:8080")
    p.add_argument("--env_type", type=str, required=True, help="Environment type, e.g. appworld/bfcl/openworld")
    p.add_argument("--split", type=str, default="train", help="Split name passed to EnvService, e.g. train/val/dev/test_normal")
    p.add_argument(
        "--k",
        type=int,
        default=5,
        help="How many tasks to inspect. Use --k -1 to inspect all tasks in the split (may be slow).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="seed_tasks.details.jsonl",
        help="Output jsonl path (one line per task detail).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Raw ids from EnvService

    env = EnvClient(args.env_url)
    task_ids = env.get_env_profile(env_type=args.env_type, split=args.split)
    if args.k == -1:
        selected_ids = task_ids
    else:
        selected_ids = task_ids[: max(0, args.k)]

    print(f"/get_env_profile returned {len(task_ids)} task_ids; inspecting {len(selected_ids)} of them.")
    print("first few task_ids:")
    print(json.dumps(selected_ids[: min(5, len(selected_ids))], ensure_ascii=False, indent=2))

    # Fetch "details" by creating an instance for each task_id.
    # Most envs return the initial conversation in `state`, which includes the user query / system messages.
    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for idx, task_id in enumerate(selected_ids):
            instance_id = None
            try:
                init = env.create_instance(env_type=args.env_type, task_id=str(task_id), instance_id=None, params={})
                # common: init["info"]["instance_id"], init["state"] (list[message])
                info = init.get("info", {}) if isinstance(init, dict) else {}
                instance_id = info.get("instance_id")

                record = {
                    "env_type": args.env_type,
                    "split": args.split,
                    "task_id": str(task_id),
                    "init": init,  # raw init payload from env service
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

                # Print a compact, human-readable preview to stdout
                state = init.get("state") if isinstance(init, dict) else None
                if isinstance(state, list) and state:
                    last_msg = state[-1]
                    last_content = last_msg.get("content") if isinstance(last_msg, dict) else None
                    print(f"[{idx}] task_id={task_id} last_message.role={getattr(last_msg, 'get', lambda _ : None)('role')} len(content)={len(last_content) if isinstance(last_content,str) else 'NA'}")
                else:
                    print(f"[{idx}] task_id={task_id} (no state in init response)")

            finally:
                # Always release to avoid leaking env instances
                if instance_id:
                    try:
                        env.release_instance(instance_id)
                    except Exception:
                        pass

    print(f"\nWrote {written} task details to: {args.out}")


if __name__ == "__main__":
    main()




