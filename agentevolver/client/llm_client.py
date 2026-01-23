import json
import os
import time
import random
import threading
from typing import Any, Optional, Generator, cast

from loguru import logger
import requests
from dotenv import load_dotenv

load_dotenv()


# ✅ 每个进程限制同时在飞的请求数（强烈建议 1~4）
_AZURE_MAX_CONCURRENCY = int(os.getenv("AZURE_LLM_MAX_CONCURRENCY", "2"))
_LLM_SEM = threading.Semaphore(_AZURE_MAX_CONCURRENCY)


class LlmException(Exception):
    def __init__(self, typ: str, response: Optional[requests.Response] = None):
        super().__init__(typ)
        self._type = typ
        self.response = response  # ✅ 带上 response 便于 Retry-After/backoff

    @property
    def typ(self):
        return self._type


class DashScopeClient:
    """
    Azure OpenAI compatible client
    (keep DashScopeClient name for compatibility with your codebase)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-2",  # Azure deployment name
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 600,
    ):
        self.api_key = (
            api_key
            or os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = (
            os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("OPENAI_API_VERSION")
        )

        if not self.api_key:
            raise ValueError("Missing AZURE_OPENAI_API_KEY / OPENAI_API_KEY")
        if not self.endpoint:
            raise ValueError("Missing AZURE_OPENAI_ENDPOINT")
        if not self.api_version:
            raise ValueError("Missing AZURE_OPENAI_API_VERSION / OPENAI_API_VERSION")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self.endpoint = self.endpoint.rstrip("/")

        self.headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def set_model(self, model_name: str):
        self.model_name = model_name

    # -------------------------
    # public API
    # -------------------------
    def chat(self, messages: list[dict[str, str]], sampling_params: dict[str, Any]) -> str:
        res = ""
        for x in self.chat_stream(messages, sampling_params):
            res += x
        return res

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        sampling_params: dict[str, Any],
    ) -> Generator[str, None, None]:
        return self.chat_stream_with_retry(messages, **sampling_params)

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        **kwargs,
    ) -> str | Generator[str, None, None]:
        url = (
            f"{self.endpoint}/openai/deployments/{self.model_name}/chat/completions"
            f"?api-version={self.api_version}"
        )

        # ✅ Azure chat/completions 不支持的参数过滤
        UNSUPPORTED = {
            "top_k",
            "min_p",
            "typical_p",
            "tfs_z",
            "mirostat",
            "mirostat_tau",
            "mirostat_eta",
        }
        for k in list(kwargs.keys()):
            if k in UNSUPPORTED:
                kwargs.pop(k, None)

        params = {
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "stream": stream,
            **kwargs,
        }

        if stream:
            return self._handle_stream_response(url, params)
        else:
            return self._handle_normal_response(url, params)

    # -------------------------
    # backoff helpers
    # -------------------------
    def _sleep_backoff(self, attempt: int, response: Optional[requests.Response] = None):
        """
        Prefer Retry-After if available, else exponential backoff + jitter.
        """
        cap = 60.0

        if response is not None:
            ra = response.headers.get("Retry-After") or response.headers.get("retry-after")
            if ra:
                try:
                    wait = min(float(ra), cap) + random.uniform(0.0, 0.8)
                    time.sleep(wait)
                    return
                except Exception:
                    pass

        # expo + jitter
        base = 2.0
        wait = min(cap, base * (2 ** attempt)) + random.uniform(0.0, 1.5)
        time.sleep(wait)

    # -------------------------
    # Normal response (non-stream)
    # -------------------------
    def _handle_normal_response(self, url: str, params: dict) -> str:
        with _LLM_SEM:  # ✅ 限制并发
            response = requests.post(
                url,
                headers=self.headers,
                json=params,
                timeout=self.timeout,
            )

        if not response.ok:
            self._raise_typed_error_from_response(response)

        try:
            result = response.json()
        except Exception:
            logger.error(f"Non-JSON response: {response.text[:500]}")
            return ""

        if "choices" in result and result["choices"]:
            return (result["choices"][0]["message"].get("content") or "").strip()

        logger.error(f"Unexpected response format: {result}")
        return ""

    # -------------------------
    # Stream response (SSE)
    # -------------------------
    def _handle_stream_response(self, url: str, params: dict) -> Generator[str, None, None]:
        with _LLM_SEM:  # ✅ 限制并发（stream 连接会占用更久）
            response = requests.post(
                url,
                headers={**self.headers, "Accept": "text/event-stream"},
                json=params,
                stream=True,
                timeout=self.timeout,
            )

        if not response.ok:
            self._raise_typed_error_from_response(response)

        ctype = (response.headers.get("Content-Type") or "").lower()
        if "text/event-stream" not in ctype:
            self._raise_typed_error_from_text_or_json(response)
            raise RuntimeError(f"Non-stream response (Content-Type={ctype}): {response.text[:300]}")

        stream_alive = False
        tool_calls_by_index: dict[int, dict] = {}

        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue

            line = raw_line.strip("\r\n")
            if not line:
                continue

            s = line.lstrip("\ufeff").lstrip()

            if s.startswith(":") or s.startswith("event:") or s.startswith("data:"):
                stream_alive = True

            if not s.startswith("data:"):
                continue

            data = s[5:].lstrip()
            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            if "error" in chunk:
                msg = (chunk["error"].get("message", "") or "")
                low = msg.lower()
                if "inappropriate" in low:
                    raise LlmException("inappropriate content", response=response)
                if "limit" in low or "rate" in low:
                    raise LlmException("hit limit", response=response)
                raise RuntimeError(f"Stream error event: {msg}")

            choices = chunk.get("choices") or []
            if not choices:
                continue

            choice0 = choices[0]
            delta = choice0.get("delta") or {}
            finish_reason = choice0.get("finish_reason")

            content = delta.get("content")
            if content:
                yield content

            for tc in (delta.get("tool_calls") or []):
                idx = tc.get("index", 0)
                cur = tool_calls_by_index.setdefault(
                    idx,
                    {
                        "id": tc.get("id"),
                        "type": tc.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    },
                )

                if tc.get("id"):
                    cur["id"] = tc["id"]
                if tc.get("type"):
                    cur["type"] = tc["type"]

                fn = tc.get("function") or {}
                name_piece = fn.get("name")
                if name_piece:
                    cur["function"]["name"] += name_piece
                args_piece = fn.get("arguments")
                if args_piece:
                    cur["function"]["arguments"] += args_piece

            if finish_reason == "tool_calls":
                payload = {"tool_calls": list(tool_calls_by_index.values())}
                yield "[TOOL_CALLS] " + json.dumps(payload, ensure_ascii=False)
                return

        if not stream_alive:
            raise RuntimeError(
                f"Stream ended without SSE events "
                f"(status={response.status_code}, content-type={ctype})"
            )

    # -------------------------
    # Retry wrappers
    # -------------------------
    def chat_stream_with_retry(
        self,
        messages: list[dict[str, str]],
        max_retries: int = 3,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        Correct retry semantics for Azure tools streaming:
        - Do NOT pre-read first chunk
        - Treat stream as successful once ANY chunk arrives
        - 429 uses Retry-After / jitter backoff
        - 400/401/403 etc do NOT retry
        """
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                stream = cast(Generator[str, None, None], self.chat_completion(messages, stream=True, **kwargs))
                got_any = False

                for chunk in stream:
                    got_any = True
                    yield chunk

                if got_any:
                    return

                raise RuntimeError("Stream finished without yielding any chunk")

            except LlmException as e:
                last_error = e

                if e.typ == "inappropriate content":
                    yield "[inappropriate content]"
                    return

                if e.typ == "hit limit":
                    logger.warning(f"Stream attempt {attempt + 1}/{max_retries} hit limit (429)")
                    if attempt < max_retries - 1:
                        self._sleep_backoff(attempt, response=getattr(e, "response", None))
                        continue

                logger.warning(f"Stream attempt {attempt + 1}/{max_retries} failed: {e}")

            except requests.HTTPError as e:
                # ✅ 400/401/403 这种不要重试
                resp = getattr(e, "response", None)
                code = getattr(resp, "status_code", None)
                last_error = e
                if code in (400, 401, 403, 404):
                    raise
                logger.warning(f"HTTPError attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    self._sleep_backoff(attempt, response=resp)

            except Exception as e:
                last_error = e
                logger.warning(f"Stream attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    self._sleep_backoff(attempt)

        logger.error(f"All {max_retries} stream attempts failed")
        if last_error:
            raise last_error

    # -------------------------
    # Error helpers
    # -------------------------
    def _raise_typed_error_from_response(self, response: requests.Response) -> None:
        """
        Map Azure errors to LlmException, with full debug info for 400s.
        """

        # ✅ 1. 429：优先处理
        if response.status_code == 429:
            raise LlmException("hit limit", response=response)

        error_json = None
        error_msg = ""
        error_code = ""

        # ✅ 2. 尝试解析 Azure 标准 error JSON
        try:
            error_json = response.json()
            err = error_json.get("error", {}) if isinstance(error_json, dict) else {}
            error_msg = err.get("message", "") or ""
            error_code = err.get("code", "") or ""
            low = error_msg.lower()

            if "inappropriate" in low:
                raise LlmException("inappropriate content", response=response)
            if "limit" in low or "rate" in low:
                raise LlmException("hit limit", response=response)

        except LlmException:
            raise
        except Exception:
            # response.json() 失败是正常的（比如 HTML）
            pass

        # ✅ 3. 关键：完整打印 400 调试信息
        logger.error("===== Azure OpenAI HTTP Error =====")
        logger.error(f"Status Code: {response.status_code}")
        logger.error(f"URL: {response.url}")

        # Azure 常用 request id
        req_id = (
            response.headers.get("x-request-id")
            or response.headers.get("apim-request-id")
            or response.headers.get("request-id")
        )
        if req_id:
            logger.error(f"Request ID: {req_id}")

        if error_json is not None:
            logger.error(f"Error JSON: {error_json}")
        else:
            logger.error(f"Raw Response Text: {response.text}")

        logger.error("===================================")

        # ✅ 4. 最终抛出原始 HTTPError（保留 traceback）
        response.raise_for_status()


    def _raise_typed_error_from_text_or_json(self, response: requests.Response) -> None:
        text = response.text or ""
        try:
            j = response.json()
            err = (j.get("error") or {})
            msg = (err.get("message", "") or "")
            low = msg.lower()
            if "inappropriate" in low:
                raise LlmException("inappropriate content", response=response)
            if "limit" in low or "rate" in low:
                raise LlmException("hit limit", response=response)
        except LlmException:
            raise
        except Exception:
            if "limit" in text.lower() or "rate" in text.lower():
                raise LlmException("hit limit", response=response)


# -------------------------
# Demo
# -------------------------
if __name__ == "__main__":
    client = DashScopeClient(model_name="gpt-4o-2")

    messages = [
        {"role": "system", "content": "If calling a tool is appropriate, do NOT answer directly."},
        {"role": "user", "content": "What is the weather in Beijing today?"},
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                },
            },
        }
    ]

    print("=== streaming ===")
    for chunk in client.chat_completion(messages, stream=True, tools=tools):
        print(repr(chunk))
