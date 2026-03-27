from typing import Any, Dict, List

from loguru import logger
from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from verl import DataProto
from verl.workers.rollout.chat_scheduler import CompletionCallback

class TokenAndProb:
    def __init__(self, t):
        # print(t)
        # ChatCompletionTokenLogprob(token='token_id:73594', bytes=[96, 96, 96], logprob=-1.9073468138230965e-06, top_logprobs=[])
        self.token_id = int(t.token.split('token_id:')[-1])
        self.logprob = t.logprob
        try:
            self.decoded_string = bytes(t.bytes).decode('utf-8')
        except:
            self.decoded_string = '<cannot decode>' + str(t.bytes)

class SimpleCompletionCallback(CompletionCallback):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        super().__init__(config, scheduler)
        logger.info("=" * 10 + "SimpleCompletionCallback is inited~" + "=" * 10)

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        if "content" not in message:
            message["content"] = "vllm failed"

        finish_reason = completions.choices[0].finish_reason
        if message['content'] == '':
            logger.warning(str(finish_reason))
            logger.bind(bad_case=True).error('empty content from completion')
            logger.bind(bad_case=True).error(str(completions.choices[0]))
            message['content'] = 'im_end'  # fill a token when vllm failed
        elif finish_reason != 'stop':
            logger.warning(str(finish_reason))
            if finish_reason == 'length':
                logger.bind(truncated=True).warning('completion truncated (finish_reason=length); keeping partial content')
            else:
                logger.bind(bad_case=True).error('non-stop finish reason')
                logger.bind(bad_case=True).error(str(completions.choices[0]))
                message['content'] = 'im_end'  # fill a token when finish reason is abnormal

        t = {"role": message["role"], "request_id":completions.id, "content": message['content'], "tokens": [TokenAndProb(t) for t in completions.choices[0].logprobs.content]}
        messages.append(t)
# import re

# def clean_qwen_thinking_output(text: str) -> str:
#     if not text:
#         return ""

#     s = text

#     # 1) 如果出现 </think> 而没有 <think>，认为之前全是推理，直接截断掉之前内容
#     if "</think>" in s and "<think>" not in s:
#         s = s.split("</think>", 1)[1]

#     # 2) 去掉完整 think 块（兼容有起止标签的情况）
#     import re
#     s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)

#     # 3) 去掉残留标签
#     s = s.replace("<think>", "").replace("</think>", "")

#     return s.strip()



# class SimpleCompletionCallback(CompletionCallback):
#     def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
#         super().__init__(config, scheduler)
#         logger.info("=" * 10 + "SimpleCompletionCallback is inited~" + "=" * 10)

#     async def __call__(self, messages, completions: ChatCompletion, info):
#         choice = completions.choices[0]
#         message = choice.message.model_dump(exclude_unset=True, exclude_none=True)

#         finish_reason = getattr(choice, "finish_reason", None)
#         content = clean_qwen_thinking_output(message.get("content", "") or "")

#         if content == "":
#             content = "vllm failed"

#         if finish_reason != "stop":
#             logger.warning(str(finish_reason))
#             if finish_reason == "length":
#                 # 被截断：不一定是失败，只是 max_tokens / 上下文预算不够
#                 logger.bind(truncated=True).warning("completion truncated (finish_reason=length)")
#             else:
#                 logger.bind(bad_case=True).error("empty content or non-stop finish reason")
#                 logger.bind(bad_case=True).error(str(choice))
#                 content = "im_end"  # ✅ 注意是 '='

#         # logprobs 容错
#         logprobs_content = []
#         if getattr(choice, "logprobs", None) and getattr(choice.logprobs, "content", None):
#             logprobs_content = [TokenAndProb(t) for t in choice.logprobs.content]

#         t = {
#             "role": message.get("role", "assistant"),
#             "request_id": completions.id,
#             "content": content,
#             "tokens": logprobs_content,
#             # 可选：保留 meta 方便你后续统计/过滤
#             # "meta": {"finish_reason": finish_reason},
#         }
#         messages.append(t)

    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        """Post process batch data.

        Args:
            batch: Batch input messages from RLHFDataset.
            batch_conversations: List of messages including raw prompt, assistant response, tool response.
                Note that `len(batch_conversations) == len(batch) * n`, e.g n=2,
                batch_conversations=[messages_0_0, messages_0_1, messages_1_0, messages_1_1, ...]
            n: How many chat completion choices to generate for each input message.

        Returns:
            Batch data, should include ["prompts", "responses", "response_mask", "input_ids", "attention_mask", "position_ids"].
        """
        raise NotImplementedError
