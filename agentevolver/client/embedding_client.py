import os
import time
import threading
from loguru import logger
import requests
import json
from typing import List, Sequence, Union, Optional, Dict, Any


class RateLimiter:
    """
    A thread-safe rate limiter using the token bucket algorithm.

    Attributes:
        max_calls (int): Maximum number of calls allowed within the time window.
        time_window (int): Time window in seconds for which the call limit applies.
        interval (float): Minimum required interval between two consecutive calls.
        _lock (threading.Lock): Lock to ensure thread safety.
        _last_call_time (float): Timestamp of the last call made.
    """
    
    def __init__(self, max_calls: int, time_window: int = 60):
        """
        Initializes the rate limiter with given maximum calls and time window.

        Args:
            max_calls (int): Maximum number of calls allowed within the time window.
            time_window (int): Time window in seconds for which the call limit applies, default is 60 seconds.
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.interval = time_window / max_calls  # ⭐ Calculate the minimum interval between calls
        
        self._lock = threading.Lock()
        self._last_call_time = 0
        
        logger.info(f"Initializing rate limiter: {max_calls} calls/{time_window} seconds, minimum interval: {self.interval:.2f} seconds")
    
    def acquire(self):
        """
        Acquires permission to execute. If the call limit is exceeded, it blocks until the next available slot.

        This method ensures that the calls are spaced out according to the defined rate limit.
        """
        with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self._last_call_time
            
            if time_since_last_call < self.interval:
                wait_time = self.interval - time_since_last_call
                # logger.debug(f"Rate limit triggered, waiting for {wait_time:.2f} seconds")
                # Wait inside the lock to ensure thread safety
                time.sleep(wait_time)
                current_time = time.time()
            
            self._last_call_time = current_time  # ⭐ Update the last call time
            # logger.debug(f"Execution permission acquired, time: {current_time}")


def _mean_pooling(last_hidden_state, attention_mask):
    # last_hidden_state: [B, T, H], attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class OpenAIEmbeddingClient:
    """
    Embedding client supporting:
    1) Local transformers embedding model (recommended)
    2) Remote OpenAI-compatible /embeddings endpoint (fallback)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "text-embedding-3-small",
        rate_limit_calls: int = 60,
        rate_limit_window: int = 60,
        # local embedding
        local_model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 2048,
        batch_size: int = 16,
    ):
        self.rate_limiter = RateLimiter(rate_limit_calls, rate_limit_window)

        # ---- local mode detection ----
        # 优先：显式传参 local_model_path；其次：读环境变量 LOCAL_EMBEDDING_MODEL_PATH
        self.local_model_path = local_model_path or os.getenv("LOCAL_EMBEDDING_MODEL_PATH")
        self.is_local = bool(self.local_model_path and os.path.isdir(self.local_model_path))

        self.max_length = max_length
        self.batch_size = batch_size

        if self.is_local:
            # lazy import to avoid forcing deps when using remote mode
            import torch
            import torch.nn.functional as F
            from transformers import AutoTokenizer, AutoModel

            self.torch = torch
            self.F = F

            # choose device
            if device:
                self.device = device
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"init LOCAL embedding client: path={self.local_model_path}, device={self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.local_model_path, trust_remote_code=True)
            self.model.eval()
            self.model.to(self.device)

            # model_name is only for compatibility with caller
            self.model_name = os.path.basename(self.local_model_path.rstrip("/"))

            # remote fields kept for API compatibility but unused in local mode
            self.api_key = ""
            self.base_url = ""
            self.headers = {}
            return

        # ---- remote mode (original behavior) ----
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        logger.info(f"init REMOTE embedding client, base_url={self.base_url}, model={self.model_name}")

    # ---------- public APIs (unchanged) ----------

    def get_embeddings(
        self,
        texts: Union[str, Sequence[str]],
        model: Optional[str] = None,
        encoding_format: str = "float",
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns OpenAI-style response:
        {"data":[{"embedding":[...]} , ...]}
        """

        if not self.is_local:
            self.rate_limiter.acquire()


        if not texts:
            raise ValueError("texts cannot be empty")

        # ---- local ----
        if self.is_local:
            embs = self._local_embeddings(texts)
            # build OpenAI-like json
            data = [{"embedding": e} for e in embs]
            return {"data": data}

        # ---- remote ----
        payload: Dict[str, Any] = {
            "input": texts,
            "model": model or self.model_name,
            "encoding_format": encoding_format,
        }
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if user is not None:
            payload["user"] = user

        url = f"{self.base_url}/embeddings"
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            if not response.ok:
                logger.error(f"failed to request embedding: {response.status_code} {response.reason}")
                try:
                    logger.error(f"err json: {response.json()}")
                except Exception:
                    logger.error(f"err text: {response.text}")
                response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise requests.RequestException(f"failed to request embedding: {e}")

    def get_single_embedding(self, text: str, **kwargs) -> List[float]:
        result = self.get_embeddings(text, **kwargs)
        return result["data"][0]["embedding"]

    def get_multiple_embeddings(self, texts: Sequence[str], **kwargs) -> List[List[float]]:
        result = self.get_embeddings(texts, **kwargs)
        return [item["embedding"] for item in result["data"]]

    # ---------- local implementation ----------

    def _local_embeddings(self, texts: Union[str, Sequence[str]]) -> List[List[float]]:
        torch = self.torch
        F = self.F

        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)

        all_vecs: List[List[float]] = []

        with torch.no_grad():
            for i in range(0, len(texts_list), self.batch_size):
                batch_texts = texts_list[i : i + self.batch_size]
                batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                out = self.model(**batch)

                # most embedding models expose last_hidden_state
                vec = _mean_pooling(out.last_hidden_state, batch["attention_mask"])
                vec = F.normalize(vec, p=2, dim=1)  # L2 normalize (recommended for cosine)

                all_vecs.extend(vec.detach().cpu().tolist())

        return all_vecs

if __name__ == "__main__":
    import concurrent.futures

    LOCAL_PATH = "/vepfs-cnbj3fa964354bf4/gjx/AgentEvolver/model/Qwen/Qwen3-Embedding-0___6B"

    client = OpenAIEmbeddingClient(
        local_model_path=LOCAL_PATH,
        rate_limit_calls=10,
        rate_limit_window=60,
        # 可选：如果你想强制用 GPU/CPU
        # device="cuda",
        # device="cpu",
        device="cuda:4",
        batch_size=8,      # 显存不够就调小：4/2/1
        max_length=2048,   # 文本很长再调大（但更慢更占显存）
    )

    def test_embedding(thread_id: int, text: str):
        try:
            start_time = time.time()
            embedding = client.get_single_embedding(f"{text} - Thread {thread_id}")
            end_time = time.time()
            print(f"thread {thread_id}: recv embedding, time {end_time - start_time:.2f}s, #dim: {len(embedding)}")
            return True
        except Exception as e:
            print(f"thread {thread_id} error: {e}")
            return False

    try:
        print("=== single thread test ===")
        for i in range(3):
            start_time = time.time()
            embedding = client.get_single_embedding(f"Test text {i}")
            end_time = time.time()
            print(f"single {i+1}: time {end_time - start_time:.2f}s, #dim: {len(embedding)}")

        print("\n=== multi thread test ===")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(test_embedding, i + 1, "Hello world") for i in range(8)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            print(f"finished multi-thread test: {sum(results)}/{len(results)}")

    except Exception as e:
        print(f"failed test: {e}")
