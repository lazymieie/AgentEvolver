#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReMe/FlowLLM HTTP endpoint smoke tests.

Usage:
  python test_reme_endpoints.py --base-url http://127.0.0.1:8001 --workspace-id debug_ws
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class Resp:
    status_code: int
    elapsed_s: float
    text: str
    json_obj: Optional[Dict[str, Any]]


def post_json(base_url: str, path: str, payload: Dict[str, Any], timeout_s: int = 30) -> Resp:
    url = base_url.rstrip("/") + path
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout_s)
    elapsed = time.time() - t0
    text = r.text

    obj = None
    try:
        obj = r.json()
    except Exception:
        obj = None

    return Resp(status_code=r.status_code, elapsed_s=elapsed, text=text, json_obj=obj)


def pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def assert_ok_json(resp: Resp, *, allow_200_with_error: bool = True) -> None:
    # ReMe sometimes returns 200 even if server had an internal exception.
    if resp.json_obj is None:
        raise AssertionError(f"Response is not JSON. status={resp.status_code}, text={resp.text[:300]}")

    if resp.status_code >= 500:
        raise AssertionError(f"HTTP {resp.status_code} server error:\n{pretty(resp.json_obj)}")

    if not allow_200_with_error:
        # Some frameworks put error info in "detail"/"error"/"message" etc.
        for k in ("error", "errors", "detail", "message", "traceback"):
            if k in resp.json_obj:
                raise AssertionError(f"Response contains error-like field '{k}':\n{pretty(resp.json_obj)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--workspace-id", default="debug_ws")
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    base_url: str = args.base_url
    ws: str = args.workspace_id

    print(f"[INFO] base_url={base_url}, workspace_id={ws}")

    # 1) Repro: call retrieve_task_memory with empty fields (this is what your logs show)
    print("\n=== 1) Repro: /retrieve_task_memory with empty query/messages/workspace_id ===")
    payload_empty = {
        "workspace_id": "",
        "query": "",
        "messages": [],
        "metadata": {},
    }
    resp = post_json(base_url, "/retrieve_task_memory", payload_empty, timeout_s=args.timeout)
    print(f"status={resp.status_code}, elapsed={resp.elapsed_s:.3f}s")
    print("json=" if resp.json_obj is not None else "text=", pretty(resp.json_obj or resp.text[:1000]))

    # We *expect* this to be problematic; don't hard-fail here.
    if resp.json_obj is None:
        print("[WARN] Non-JSON response; check server logs.")
    else:
        # Detect the "KeyError: 'memory_list'" symptom if it's returned in-body (depends on exception handling)
        body_str = json.dumps(resp.json_obj, ensure_ascii=False)
        if "memory_list" in body_str and ("KeyError" in body_str or "keyerror" in body_str.lower()):
            print("[WARN] Looks like KeyError 'memory_list' surfaced in response body.")

    # 2) Seed memory via summary_task_memory_simple (writes one memory record)
    print("\n=== 2) Seed: /summary_task_memory_simple ===")
    payload_seed = {
        "workspace_id": ws,
        # this endpoint schema says trajectories can be omitted, but your service accepts messages in some ops;
        # we include a minimal "messages" style structure commonly used in your earlier curl examples.
        "messages": [
            {"role": "user", "content": "把 report.txt 按行排序，然后发到推特并@Julia"},
            {"role": "assistant", "content": "好的，我会先读取文件并排序，然后发推。"},
        ],
        "query": "把 report.txt 排序后发到推特并@Julia",
        "metadata": {},
    }
    resp = post_json(base_url, "/summary_task_memory_simple", payload_seed, timeout_s=args.timeout)
    print(f"status={resp.status_code}, elapsed={resp.elapsed_s:.3f}s")
    if resp.json_obj is not None:
        print(pretty(resp.json_obj))
    else:
        print(resp.text[:1000])
    assert_ok_json(resp)

    # 3) Retrieve memory (simple endpoint, usually most stable)
    print("\n=== 3) Retrieve: /retrieve_task_memory_simple ===")
    payload_retrieve_simple = {
        "workspace_id": ws,
        "query": "怎么把 report.txt 排序并发推？",
        "messages": [{"role": "user", "content": "怎么把 report.txt 排序并发推？"}],
        "metadata": {},
    }
    resp = post_json(base_url, "/retrieve_task_memory_simple", payload_retrieve_simple, timeout_s=args.timeout)
    print(f"status={resp.status_code}, elapsed={resp.elapsed_s:.3f}s")
    if resp.json_obj is not None:
        print(pretty(resp.json_obj))
    else:
        print(resp.text[:1000])
    assert_ok_json(resp)

    # 4) Retrieve memory (full endpoint, includes Rerank/Rewrite chain; this is the one that KeyError'd)
    print("\n=== 4) Retrieve: /retrieve_task_memory (full flow) ===")
    payload_retrieve_full = {
        "workspace_id": ws,
        "query": "怎么把 report.txt 排序并发推？",
        "messages": [{"role": "user", "content": "怎么把 report.txt 排序并发推？"}],
        "metadata": {},
    }
    resp = post_json(base_url, "/retrieve_task_memory", payload_retrieve_full, timeout_s=args.timeout)
    print(f"status={resp.status_code}, elapsed={resp.elapsed_s:.3f}s")
    if resp.json_obj is not None:
        print(pretty(resp.json_obj))
    else:
        print(resp.text[:1000])

    # Here we *do* fail if the response looks like an internal error even with HTTP 200.
    if resp.json_obj is not None:
        body_str = json.dumps(resp.json_obj, ensure_ascii=False).lower()
        suspicious = any(s in body_str for s in ["traceback", "keyerror", "exception", "internal server error"])
        if suspicious:
            raise AssertionError(f"/retrieve_task_memory seems to have failed (even if HTTP={resp.status_code}).\n{pretty(resp.json_obj)}")

    print("\n[PASS] Basic tests finished.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        print(f"\n[FAIL] {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(1)
