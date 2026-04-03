# -*- coding: utf-8 -*-
"""Quick checks for the linear short-think hint switch."""

from types import SimpleNamespace

from agentevolver.module.context_manager.cmt_base import ExtendedMessage
from agentevolver.module.context_manager.cmt_linear import Linear_CMT


class MockTokenizer:
    eos_token_id = 0

    def encode(self, text):
        return list(range(len(text)))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        prompt = "\n".join(f"{msg['role']}:{msg['content']}" for msg in messages)
        if add_generation_prompt:
            prompt += "\nassistant:"
        return prompt

    def __call__(self, text, return_tensors="pt", padding=False):
        return {"input_ids": [[max(1, len(text))]]}


def build_config(enable_short_hint: bool):
    rollout = SimpleNamespace(
        response_length=256,
        max_model_len=2048,
        max_env_len=512,
        enable_linear_short_think_hint=enable_short_hint,
        linear_short_think_hint_prompt="\n\nKeep your reasoning minimal.",
    )
    return SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=rollout),
        data=SimpleNamespace(max_prompt_length=1024, max_response_length=512),
    )


def build_cmt(enable_short_hint: bool, content: str) -> Linear_CMT:
    tokenizer = MockTokenizer()
    cmt = Linear_CMT(config=build_config(enable_short_hint), tokenizer=tokenizer)
    cmt.full_context = [
        ExtendedMessage(
            author="initialization",
            role="user",
            content=content,
            tokenizer=tokenizer,
            token_generator="manual",
        )
    ]
    return cmt


def test_short_think_hint_is_appended_only_to_model_input():
    cmt = build_cmt(enable_short_hint=True, content="Solve the task.")

    model_input = cmt.prepare_next_llm_context()

    assert model_input[-1]["content"].endswith("\n\nKeep your reasoning minimal.")
    assert cmt.full_context[-1].content == "Solve the task."


def test_short_think_hint_can_be_disabled():
    cmt = build_cmt(enable_short_hint=False, content="Solve the task.")

    model_input = cmt.prepare_next_llm_context()

    assert model_input[-1]["content"] == "Solve the task."


def test_short_think_hint_respects_no_think():
    cmt = build_cmt(enable_short_hint=True, content="Solve the task.\n/no_think")

    model_input = cmt.prepare_next_llm_context()

    assert model_input[-1]["content"] == "Solve the task.\n/no_think"
