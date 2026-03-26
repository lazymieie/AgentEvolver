# https://arxiv.org/pdf/2505.10978
# Memory template with alien LLM for information extraction (similar to context_clip)

import torch
import copy
import re
import json
import random
import time
import os
from typing import List, Callable
from agentevolver.schema.trajectory import Sample
from best_logger import print_dict, print_listofdict
from agentevolver.module.context_manager.cmt_linear_think import ExtendedMessage, Linear_CMT, LinearThinkCMT
from agentevolver.module.context_manager.cmt_linear import find_sublist_indices, replace_token_ids
from best_logger import register_logger, print_dict, print_nested, NestedJsonItem, SeqItem
from textwrap import dedent
from openai import AzureOpenAI
from loguru import logger
from agentevolver.utils.markdown_parser import read_markdown_and_extract_sections


def construct_alien_llm_chat_fn(config, rollout_config):
    """Construct alien LLM chat function for memory extraction using Azure OpenAI"""
    def alien_llm_chat_fn(messages, request_id=""):
        max_try = getattr(
            config.actor_rollout_ref.rollout,
            'context_template_alien_llm_max_try',
            2
        )
        retry_sleep_s = getattr(
            config.actor_rollout_ref.rollout,
            'context_template_alien_llm_retry_sleep_s',
            2
        )
        
        # Get Azure OpenAI configuration from config or environment variables
        azure_api_key = getattr(
            config.actor_rollout_ref.rollout, 
            'context_template_alien_llm_api_key', 
            None
        ) or os.getenv("AZURE_OPENAI_API_KEY")
        
        azure_endpoint = getattr(
            config.actor_rollout_ref.rollout,
            'context_template_alien_llm_endpoint',
            None
        ) or os.getenv("AZURE_OPENAI_ENDPOINT")
        
        azure_api_version = getattr(
            config.actor_rollout_ref.rollout,
            'context_template_alien_llm_api_version',
            None
        ) or os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION") or "2024-12-01-preview"
        
        # Deployment name (model name in Azure OpenAI)
        alien_model_name = getattr(
            config.actor_rollout_ref.rollout,
            'context_template_alien_llm_model',
            None
        ) or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or "gpt-4o-2"
        
        alien_model_response_length = getattr(
            config.actor_rollout_ref.rollout,
            'context_template_alien_model_response_length',
            2048
        )
        alien_request_timeout_s = getattr(
            config.actor_rollout_ref.rollout,
            'context_template_alien_llm_timeout_s',
            60
        )
        
        if not azure_api_key:
            raise ValueError("Missing Azure OpenAI API key. Set AZURE_OPENAI_API_KEY or context_template_alien_llm_api_key in config.")
        if not azure_endpoint:
            raise ValueError("Missing Azure OpenAI endpoint. Set AZURE_OPENAI_ENDPOINT or context_template_alien_llm_endpoint in config.")
        
        # Normalize endpoint (remove trailing slash)
        azure_endpoint = azure_endpoint.rstrip("/")
        
        client = AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
        )

        for n_try in range(max_try):
            try:
                request_kwargs = dict(
                    model=alien_model_name,  # This is the deployment name in Azure OpenAI
                    messages=messages,
                    temperature=0,
                    max_tokens=alien_model_response_length,
                )
                if alien_request_timeout_s is not None:
                    request_kwargs["timeout"] = alien_request_timeout_s

                completion = client.chat.completions.create(**request_kwargs)
                
                message = completion.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
                if "content" not in message: 
                    message["content"] = ""
                return {"role": message["role"], "content": message['content']}
            except Exception as e:
                logger.bind(exception=True).exception(f"Error calling Azure OpenAI alien llm: {e}")
                if n_try < max_try - 1:
                    time.sleep(retry_sleep_s)
                    print(f"Error calling Azure OpenAI alien llm: {e}, retrying... ({n_try + 1}/{max_try})")
                else:
                    raise
        raise RuntimeError(f"Failed to get response from Azure OpenAI alien llm after {max_try} attempts")
    return alien_llm_chat_fn


class MemoryNewCMT(LinearThinkCMT):
    """
    Memory context manager template that uses alien LLM to extract and compress information.
    
    Key differences from original MemoryCMT:
    1. Inherits from LinearThinkCMT (not Linear_CMT) - supports grouping and reward
    2. Uses alien LLM to extract 4 sections from normal LLM output (instead of requiring structured output)
    3. Automatically compresses context by storing only extracted memory
    4. Main model can output naturally, alien LLM handles extraction
    """

    def __init__(self, config, tokenizer, llm_chat_fn):
        super().__init__(config, tokenizer)
        self.current_step = 0
        self.llm_chat_fn = llm_chat_fn
        self.alien_llm_chat_fn: Callable = construct_alien_llm_chat_fn(config, config.actor_rollout_ref.rollout)
        self.console_debug_mode = False
        self.train_sp_action = config.actor_rollout_ref.rollout.context_template_train_sp_action
        self.memory_extracted_before = False
        
        # Memory extraction trigger threshold
        self.memory_extract_trigger_token_num = getattr(
            config.actor_rollout_ref.rollout, 
            'context_template_memory_extract_trigger_token_num', 
            8000
        )

    def _get_seq_length(self, messages: List[dict]) -> int:
        """Calculate sequence length for token counting"""
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])

    def prepare_next_llm_context(self):
        """
        Prepares the next context for the LLM.
        
        Structure: [initialization] -> [memory history] -> [previous code] -> [env feedback]
        """
        self.latest_llm_interaction_socket = []
        
        # Filter context: initialization, memory, llm, env
        self.latest_llm_interaction_socket = self.filter_context_via_authors(["initialization", "memory", "llm", "env"])
        
        is_first_interaction = (len(self.filter_context_via_author("llm")) == 0)
        
        if is_first_interaction:
            # First interaction: just initialization
            dict_context = self.to_role_content(self.latest_llm_interaction_socket)
            return dict_context
        
        # For subsequent interactions, structure the context
        processed_context = []
        memory_turn = 1
        llm_turn = 1
        env_turn = 1
        
        for index, ext_msg in enumerate(list(self.latest_llm_interaction_socket)):
            is_last = (index == len(self.latest_llm_interaction_socket) - 1)
            
            if ext_msg.author == "memory":
                # Memory messages: format with turn ID
                turn_id = f"{memory_turn:03d}"
                content = dedent(f"""
                    [Memory, id=M{turn_id}]
                    ---
                """).strip() + '\n' + ext_msg.content_for_future.strip()
                processed_context.append(ExtendedMessage(
                    author=ext_msg.author,
                    role=ext_msg.role,
                    content=content,
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                    build_from_uuid=ext_msg.uuid,
                ))
                memory_turn += 1
                
            elif ext_msg.author == "llm":
                # LLM messages: remove think tags, mark as do_not_train
                new_content = self.strip_think_tags(ext_msg.content)
                turn_id = f"{llm_turn:03d}"
                content = dedent(f"""
                    [Assistant Response, id=AR{turn_id}]
                    ---
                """).strip() + '\n' + new_content.strip()
                processed_context.append(ExtendedMessage(
                    author="llm(do_not_train)",
                    role=ext_msg.role,
                    content=content,
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                    build_from_uuid=ext_msg.uuid,
                ))
                llm_turn += 1
                
            elif ext_msg.author == "env":
                # Environment messages: format with turn ID
                turn_id = f"{env_turn:03d}"
                content = dedent(f"""
                    [Environment Response, id=ER{turn_id}]
                    ---
                """).strip() + '\n' + ext_msg.content_for_future.strip()
                processed_context.append(ExtendedMessage(
                    author=ext_msg.author,
                    role=ext_msg.role,
                    content=content,
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                    build_from_uuid=ext_msg.uuid,
                ))
                env_turn += 1
                
            elif ext_msg.author == "initialization":
                # Initialization messages: keep as is
                processed_context.append(ExtendedMessage(
                    author=ext_msg.author,
                    role=ext_msg.role,
                    content=ext_msg.content_for_future,
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                    build_from_uuid=ext_msg.uuid,
                ))
            else:
                raise RuntimeError(f"Unknown author {ext_msg.author} in latest_llm_interaction_socket")
        
        self.latest_llm_interaction_socket = processed_context

        # 内部 latest_llm_interaction_socket 保留带 Mxxx/ARxxx/ERxxx 标签，供 alien_llm 做提取使用。
        # 但在喂给本地模型时，去掉这些标签头，避免模型学习到无用格式。
        cleaned_messages = []
        tag_pattern_assistant = re.compile(
            r'^\[Assistant Response, id=AR\d+\]\s*---\s*', flags=re.MULTILINE
        )
        tag_pattern_env = re.compile(
            r'^\[Environment Response, id=ER\d+\]\s*---\s*', flags=re.MULTILINE
        )
        tag_pattern_memory = re.compile(
            r'^\[Memory, id=M\d+\]\s*---\s*', flags=re.MULTILINE
        )
        for ext_msg in self.latest_llm_interaction_socket:
            content = ext_msg.content_for_future.strip()
            # 只清理最前面的标签头
            content = tag_pattern_assistant.sub('', content, count=1)
            content = tag_pattern_env.sub('', content, count=1)
            content = tag_pattern_memory.sub('', content, count=1)
            cleaned_messages.append({"role": ext_msg.role, "content": content})

        return cleaned_messages

    def strip_think_tags(self, text: str) -> str:
        """Remove think tags from text (same as context_clip)"""
        new_ext_msg_content = re.sub(r'\<think\>.*?\<\/think\>', '', text, flags=re.DOTALL).strip()
        new_ext_msg_content = new_ext_msg_content.replace("<think>", "")
        new_ext_msg_content = new_ext_msg_content.replace("</think>", "")
        return new_ext_msg_content

    def impl_new_request_from_previous_interaction(self, new_message, this_interaction, strip_think=False):
        """
        Process a new request using alien LLM (same pattern as context_clip).
        
        Args:
            new_message: The new message to add
            this_interaction: Previous interaction list
            strip_think: Whether to strip think tags
            
        Returns:
            tuple: (updated_interaction, llm_output_content)
        """
        latest_llm_interaction_socket_additional = copy.deepcopy(this_interaction)
        if strip_think:
            for index, ext_msg in enumerate(latest_llm_interaction_socket_additional):
                if ext_msg.author == "llm(do_not_train)" or ext_msg.author == "llm":
                    latest_llm_interaction_socket_additional[index] = ExtendedMessage(
                        author=ext_msg.author,
                        role=ext_msg.role,
                        content=self.strip_think_tags(ext_msg.content),
                        token_generator='auto',
                        tokenizer=self.tokenizer,
                        build_from_uuid=ext_msg.build_from_uuid if ext_msg.build_from_uuid else ext_msg.uuid,
                    )
                else:
                    continue
        latest_llm_interaction_socket_additional += [new_message]
        dict_context = self.to_role_content(latest_llm_interaction_socket_additional)
        
        # Use alien LLM for memory extraction tasks
        llm_output = self.alien_llm_chat_fn(dict_context, request_id="")
        
        latest_llm_interaction_socket_additional += [
            self.save_llm_output_do_not_register_full_context(llm_output, dict_context)
        ]
        
        if self.train_sp_action:
            this_interaction = copy.deepcopy(latest_llm_interaction_socket_additional)
            self.grouped_steps += [this_interaction]
            
        if self.console_debug_mode:
            print_listofdict(
                dict_context + [{'role': 'llm_latest', 'content': llm_output['content']}], mod='c'
            )
        else:
            print_listofdict(
                dict_context + [{'role': 'llm_latest', 'content': llm_output['content']}], mod='memory_new'
            )
        output_llm_content = llm_output['content'].strip()
        return latest_llm_interaction_socket_additional, output_llm_content

    def after_save_llm_output(self, this_interaction):
        """
        Extract and compress information into memory after saving LLM output.
        
        This is the core memory extraction logic:
        1. Check if token count exceeds threshold
        2. Use alien LLM to extract 4 sections from recent interactions
        3. Create memory message and add to full_context
        4. Optionally remove or compress old messages
        """
        if self.memory_extracted_before:
            # Only extract once per trajectory to avoid overhead
            return
        
        # Check token threshold
        this_interaction = copy.deepcopy(this_interaction)
        interaction_messages = self.to_role_content(this_interaction)
        interaction_seq_len = self._get_seq_length(interaction_messages)
        if interaction_seq_len < self.memory_extract_trigger_token_num:
            return
        
        self.memory_extracted_before = True
        self.metadata["memory_extraction_triggered"] = True
        self.metadata["memory_extraction_seq_len"] = interaction_seq_len
        self.metadata["memory_extraction_message_count"] = len(interaction_messages)
        
        # Get recent LLM and env messages for extraction
        recent_llm_msgs = [msg for msg in this_interaction if msg.author == "llm"]
        recent_env_msgs = [msg for msg in this_interaction if msg.author == "env"]
        
        if len(recent_llm_msgs) == 0 or len(recent_env_msgs) == 0:
            return
        
        try:
            # Keep rollout alive even if auxiliary memory extraction times out.
            _, extracted_content = self.impl_new_request_from_previous_interaction(
                new_message=ExtendedMessage(
                    author='user',
                    role='user',
                    content=dedent("""
                        Your task is to analyze the recent conversation and extract key information in the following 4 sections:
                        
                        1. **current step**: Summarize what step we are at and what we're trying to accomplish
                        2. **previous instruction code**: Extract the code/action that was executed in the previous step
                        3. **relevant environment feedback**: Extract all useful information from environment responses, including:
                           - Account credentials, keys, access tokens
                           - Important state information
                           - Error messages or warnings
                           - Any data that might be needed in future steps
                        4. **next-step instruction code**: Based on the context, what code/action should be executed next?
                        
                        Format your response as markdown with these 4 sections:
                        ```markdown
                        # current step
                        [your summary here]
                        
                        # previous instruction code
                        ```[language]
                        [code here]
                        ```
                        
                        # relevant environment feedback
                        [extracted feedback here]
                        
                        # next-step instruction code
                        ```[language]
                        [code here]
                        ```
                        ```
                        
                        Important: Extract ALL useful information from environment feedback, as it will be lost otherwise.
                    """),
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                ),
                this_interaction=this_interaction,
                strip_think=True,
            )

            # Parse extracted content
            if extracted_content.count("```") >= 2:
                # Extract markdown content
                if "```markdown" in extracted_content:
                    extracted_markdown = extracted_content.split("```markdown")[1].split("```")[0].strip()
                else:
                    extracted_markdown = extracted_content.split("```")[1].split("```")[0].strip()
            else:
                extracted_markdown = extracted_content
            
            # Parse the 4 sections
            lm_result_decompose, find_everything, find_nothing = read_markdown_and_extract_sections(
                markdown_text=extracted_markdown,
                expected_sections=["current step", "previous instruction code", "relevant environment feedback", "next-step instruction code"],
                default_placeholder="❌ not available."
            )
            
            if find_nothing:
                logger.warning("Failed to extract memory sections from alien LLM output")
                return
            
            # Create memory construct
            memory_construct = dedent("""
                >> step: {current_step}
                >> instruction:
                {next_step_instruction_code}
                >> feedback from environment:
                {relevant_environment_feedback}
                ---
            """).format(
                current_step=lm_result_decompose.get('current step', 'Unknown step'),
                next_step_instruction_code=lm_result_decompose.get('next-step instruction code', 'No code'),
                relevant_environment_feedback=lm_result_decompose.get('relevant environment feedback', 'No feedback')
            )
            
            # Add memory to full_context
            ext_msg_memory = ExtendedMessage(
                author="memory",
                role="assistant",
                content=memory_construct,
                token_generator="auto",
                tokenizer=self.tokenizer,
            )
            self.full_context += [ext_msg_memory]
            
            # Optionally: mark old LLM/env messages as discard to save tokens
            # (similar to context_clip's remove action)
            recent_llm_uuids = {msg.uuid for msg in recent_llm_msgs[:-1]}  # Keep last one
            recent_env_uuids = {msg.uuid for msg in recent_env_msgs[:-1]}  # Keep last one
            
            for index, msg in enumerate(self.full_context):
                if msg.uuid in recent_llm_uuids or msg.uuid in recent_env_uuids:
                    if msg.author not in ["memory"]:  # Don't discard memory
                        self.full_context[index] = ExtendedMessage(
                            author=msg.author + "(discard)",
                            role=msg.role,
                            content=msg.content,
                            token_generator='auto',
                            tokenizer=self.tokenizer,
                        )
            
            logger.info(f"Memory extracted and added. Total memory messages: {len(self.filter_context_via_author('memory'))}")
            
        except Exception as e:
            self.metadata["memory_extraction_failed"] = True
            self.metadata["memory_extraction_error"] = f"{type(e).__name__}: {e}"
            logger.bind(exception=True).exception(f"Error extracting memory: {e}")
            print(f"Error extracting memory: {e}; skip memory extraction for this trajectory.")

    def save_llm_output(self, llm_output, input_msg_ref):
        """
        Save LLM output and trigger memory extraction if needed.
        """
        ext_msg = Linear_CMT.save_llm_output(self, llm_output, input_msg_ref)
        this_interaction = copy.deepcopy(self.latest_llm_interaction_socket + [ext_msg])
        self.grouped_steps += [this_interaction]
        self.after_save_llm_output(this_interaction)  # Trigger memory extraction
        self.latest_llm_interaction_socket = []
        return

    def prepare_world_interaction(self) -> str:
        """
        Extract the action/code to send to environment.
        
        If LLM output contains structured sections, extract 'next-step instruction code'.
        Otherwise, return raw content (with fallback).
        """
        ext_message_arr_memory = self.filter_context_via_author("llm")
        if len(ext_message_arr_memory) == 0:
            return ""

        last_llm_content = ext_message_arr_memory[-1].content
        
        # Try to extract structured sections
        lm_result_decompose, find_everything, find_nothing = read_markdown_and_extract_sections(
            markdown_text=last_llm_content,
            expected_sections=["current step", "previous instruction code", "relevant environment feedback", "next-step instruction code"],
            default_placeholder="❌ not available."
        )

        next_step_code = lm_result_decompose.get('next-step instruction code', '')
        
        # Fallback: if parsing failed, return raw content
        if find_nothing or not next_step_code or next_step_code.strip() == "" or "❌ not available" in next_step_code:
            return last_llm_content
        
        return next_step_code
