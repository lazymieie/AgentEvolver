import json
import os
import time
from typing import Any, Optional, Generator, cast

from loguru import logger
import requests

from dotenv import load_dotenv
load_dotenv()  # 默认加载当前目录的 .env
class LlmException(Exception):
    def __init__(self, typ: str):
        self._type = typ

    @property
    def typ(self):
        return self._type


class AzureOpenAICompatClient:
    """
    Azure OpenAI ChatCompletions client using requests (OpenAI-compatible response parsing).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 600,
    ):
        # Read from env (support both AZURE_* and OPENAI_* if you want)
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION")

        if not self.api_key:
            raise ValueError("Missing API key. Set AZURE_OPENAI_API_KEY (or OPENAI_API_KEY).")
        if not self.endpoint:
            raise ValueError("Missing endpoint. Set AZURE_OPENAI_ENDPOINT, e.g. https://xxx.openai.azure.com")
        if not self.deployment_name:
            raise ValueError("Missing deployment. Set AZURE_OPENAI_DEPLOYMENT_NAME, e.g. gpt-4o-2")
        if not self.api_version:
            raise ValueError("Missing api version. Set AZURE_OPENAI_API_VERSION or OPENAI_API_VERSION.")

        # Normalize endpoint (no trailing slash)
        self.endpoint = self.endpoint.rstrip("/")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Azure uses api-key header (NOT Bearer)
        self.headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def set_deployment(self, deployment_name: str):
        self.deployment_name = deployment_name

    def chat(self, messages: list[dict[str, str]], sampling_params: dict[str, Any]) -> str:
        res = ""
        for x in self.chat_stream(messages, sampling_params):
            res += x
        return res

    def chat_stream(self, messages: list[dict[str, str]], sampling_params: dict[str, Any]) -> Generator[str, None, None]:
        return self.chat_stream_with_retry(messages, **sampling_params)

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        **kwargs,
    ) -> str | Generator[str, None, None]:
        # Azure URL includes deployment in path and api-version in query
        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions"
            f"?api-version={self.api_version}"
        )

        params = {
            # "model": self.deployment_name,  # optional; usually omit for Azure
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "stream": stream,
            **kwargs,
        }

        try:
            if stream:
                return self._handle_stream_response(url, params)
            else:
                return self._handle_normal_response(url, params)

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return "" if not stream else (x for x in [])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            return "" if not stream else (x for x in [])
        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            return "" if not stream else (x for x in [])

    def _handle_normal_response(self, url: str, params: dict) -> str:
        response = requests.post(url, headers=self.headers, json=params, timeout=self.timeout)

        if not response.ok:
            # Azure error format typically: {"error": {"code": "...", "message": "..."}}
            try:
                err = response.json().get("error", {})
                msg = err.get("message", "")
                if "inappropriate" in msg:
                    raise LlmException("inappropriate content")
                if "limit" in msg or "Rate limit" in msg or "Too Many Requests" in msg:
                    raise LlmException("hit limit")
            except LlmException:
                raise
            except Exception:
                logger.error(f"API request failed: {response.text}")
                response.raise_for_status()

        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        logger.error(f"Unexpected response format: {result}")
        return ""

    def _handle_stream_response(self, url: str, params: dict) -> Generator[str, None, None]:
        response = requests.post(url, headers=self.headers, json=params, stream=True, timeout=self.timeout)

        if not response.ok:
            try:
                err = response.json().get("error", {})
                msg = err.get("message", "")
                if "inappropriate" in msg:
                    raise LlmException("inappropriate content")
                if "limit" in msg or "Rate limit" in msg or "Too Many Requests" in msg:
                    raise LlmException("hit limit")
            except LlmException:
                raise
            except Exception:
                logger.error(f"API request failed: {response.text}")
                response.raise_for_status()

        # Azure streaming is SSE with lines like: b"data: {...}"
        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
            except json.JSONDecodeError:
                continue

    def chat_with_retry(self, messages: list[dict[str, str]], max_retries: int = 3, retry_delay: float = 1.0, **kwargs) -> str:
        for attempt in range(max_retries):
            try:
                result = cast(str, self.chat_completion(messages, stream=False, **kwargs))
                if result:
                    return result
            except LlmException as e:
                if e.typ == "inappropriate content":
                    logger.warning("llm return inappropriate content, blocked by remote")
                    return "[inappropriate content]"
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))

        logger.error(f"All {max_retries} attempts failed")
        return ""

    def chat_stream_with_retry(
        self,
        messages: list[dict[str, str]],
        max_retries: int = 3,
        retry_delay: float = 10.0,
        **kwargs,
    ) -> Generator[str, None, None]:
        for attempt in range(max_retries):
            try:
                stream_generator = cast(Generator[str, None, None], self.chat_completion(messages, stream=True, **kwargs))
                first_chunk = next(stream_generator, None)
                if first_chunk is not None:
                    yield first_chunk
                    for chunk in stream_generator:
                        yield chunk
                    return
            except LlmException as e:
                if e.typ == "inappropriate content":
                    logger.warning("llm return inappropriate content, blocked by remote")
                    yield "[inappropriate content]"
                    return
            except Exception as e:
                logger.warning(f"Stream attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))

        logger.error(f"All {max_retries} stream attempts failed")
        return


if __name__ == "__main__":
    # 需要你 .env 里有这些变量
    # AZURE_OPENAI_API_KEY=...
    # AZURE_OPENAI_ENDPOINT=https://cv1-gpt4o.openai.azure.com
    # AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-2
    # AZURE_OPENAI_API_VERSION=2024-12-01-preview

    client = AzureOpenAICompatClient()

    messages = [{"role": "user", "content": "Write a poem about Spring."}]

    print("\n=== streaming ===")
    for chunk in client.chat_completion(messages, stream=True):
        print(chunk, end="", flush=True)
