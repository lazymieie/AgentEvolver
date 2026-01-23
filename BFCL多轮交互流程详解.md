# BFCL多轮交互流程详解

## 概述
BFCL (Berkeley-Function-Calling-Leaderboard) 是一个多轮对话环境，支持工具调用。本文档详细说明AgentEvolver系统与BFCL环境的多轮交互流程。

---

## 一、整体架构

### 1.1 核心组件
- **ParallelEnvManager**: 并行环境管理器，负责管理多个任务的并行执行
- **EnvWorker**: 环境工作器，负责单个任务的环境交互
- **AgentFlow**: 代理流程，负责LLM与环境的多轮交互循环
- **BfclEnv**: BFCL环境实现，封装了BFCL的具体逻辑
- **EnvHandler**: 环境处理器，处理工具调用和用户轮次

### 1.2 数据流
```
ParallelEnvManager → EnvWorker → AgentFlow → EnvClient → BfclEnv → EnvHandler
```

---

## 二、初始化阶段

### 2.1 任务提交 (ParallelEnvManager.rollout)
**位置**: `env_manager.py:317-412`

1. **任务准备**
   - 接收任务列表 `tasks: List[Task]`
   - 为每个任务创建多个rollout（根据`rollout_n`配置）
   - 初始化线程状态跟踪 `tmux` 和停止标志 `stop`

2. **线程池提交**
   ```python
   # 为每个任务和每个rollout创建线程
   for data_id, (task, task_exp_config) in enumerate(zip(tasks, task_exp_configs)):
       for rollout_id in range(rollout_n):
           thread_index = data_id * rollout_n + rollout_id
           future = executor.submit(self.rollout_env_worker, ...)
   ```

### 2.2 环境工作器初始化 (EnvWorker.execute)
**位置**: `env_worker.py:40-165`

1. **创建环境实例**
   ```python
   init_response = self.env.create_instance(
       env_type=self.env_type,  # "bfcl"
       task_id=self.task_id,
       instance_id=self.instance_id,
       params={'is_open_query': self.is_open_query}
   )
   ```

2. **获取初始状态**
   - `init_response["state"]` 包含初始消息列表
   - 格式: `[{"role": "system", "content": tool_prompt}, {"role": "user", "content": first_query}]`
   - `tool_prompt`: 包含可用工具的XML格式描述（通过`tools_schema_to_qwen_prompt`生成）

3. **初始化上下文管理器**
   - 根据配置选择上下文模板（`linear`, `linear_think`, `context_selfclip`）
   - 创建 `Linear_CMT` 实例管理对话历史

### 2.3 BFCL环境初始化 (BfclEnv.get_init_state)
**位置**: `bfcl_env.py:238-285`

1. **加载测试用例**
   ```python
   self.test_entry = self._load_test_case(self.data_path, self.task_id)
   self.original_test_entry = self.test_entry
   ```

2. **创建EnvHandler**
   ```python
   self.env_handler = EnvHandler(
       model_name=self.model_name,
       answer_path=Path(self.answer_path)
   )
   ```

3. **初始化对话历史**
   ```python
   self.conversation_history = self.test_entry.get("question", [[]])[0].copy()
   self.current_turn = 0
   ```

4. **生成工具提示**
   - 从 `test_entry["function"]` 提取工具列表
   - 使用 `tools_schema_to_qwen_prompt` 转换为Qwen格式的工具描述
   - 返回包含system消息（工具描述）和user消息（第一个查询）的初始状态

---

## 三、多轮交互循环

### 3.1 AgentFlow执行循环 (AgentFlow.execute)
**位置**: `agent_flow.py:49-223`

主循环在 `for act_step in range(self.max_steps)` 中执行，每轮包含以下步骤：

#### 步骤1: 更新线程进度
```python
tmux['step'][thread_index] = act_step
```
- 跟踪当前执行步数，用于监控和进度显示

#### 步骤2: 准备LLM输入上下文
```python
step_input_message_arr = self.cmt.prepare_next_llm_context()
```
**详细过程** (`cmt_linear.py`):
- 从 `full_context` 中提取所有消息
- 使用 `content_for_future` 格式化消息（处理工具调用格式）
- 返回消息列表，格式: `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]`

#### 步骤3: 检查Token溢出
```python
is_safe = self.cmt.check_context_token_num_safe(step_input_message_arr)
```
- 计算当前上下文的token数量
- 如果超过 `max_seq_length`，终止轨迹（标记为未完成）

#### 步骤4: 调用LLM生成响应
```python
llm_output = self.llm_chat_fn(step_input_message_arr, request_id=request_id)
```
**LLM输出格式**:
```python
{
    "role": "assistant",
    "content": "思考内容...\n<tool_call>\n{\"name\": \"mkdir\", \"arguments\": {\"dir_name\": \"temp\"}}\n</tool_call>",
    "request_id": "..."
}
```

#### 步骤5: 保存LLM输出
```python
self.cmt.save_llm_output(llm_output, input_msg_ref=step_input_message_arr)
tmux['token'][thread_index] += self.cmt.generated_token_cnt
```
- 将LLM响应添加到 `full_context`
- 更新token计数

#### 步骤6: 与环境交互
```python
env_output = env.step(
    instance_id,
    {"content": self.cmt.prepare_world_interaction(), "role": "assistant"}
)
```

**prepare_world_interaction()** 处理:
- 提取LLM输出中的工具调用部分
- 转换为环境可理解的格式

**env.step() 调用链**:
1. `EnvClient.step()` → HTTP请求到环境服务
2. `BfclEnv.step()` → 处理action并调用transition
3. `BfclEnv.transition()` → 调用EnvHandler.interact()

#### 步骤7: 处理环境响应
```python
if env_output["state"]["role"] == "tool":
    env_output["state"] = convert_tool_to_user_message(env_output["state"], ...)
```

环境可能返回两种类型的响应：

**类型A: 工具调用结果**
```python
{
    "role": "tool",
    "content": {"travel_cost_list": [1140.0]},  # 或 {"error": "..."}
    "tool_call_id": "chatcmpl-tool-xxx"
}
```

**类型B: 新的用户查询**
```python
{
    "role": "user",
    "content": "What's the total cost?"
}
```

#### 步骤8: 保存环境输出
```python
self.cmt.save_env_output(state, input_msg_ref=step_input_message_arr, add_nothink=add_nothink)
```
- 将环境响应添加到 `full_context`
- 如果是工具结果，转换为Qwen格式的工具消息

#### 步骤9: 检查终止条件
```python
self.cmt.is_terminated = env_output["is_terminated"]
if self.cmt.is_terminated or err_in_env:
    break
```

---

## 四、BFCL环境内部处理

### 4.1 BfclEnv.step() 处理流程
**位置**: `bfcl_env.py:287-308`

1. **解析Assistant消息**
   ```python
   assistant_entry = parse_assistant_content_to_tool_calls(assistant_entry)
   ```
   - 从content中提取 `<tool_call>...</tool_call>` 标签
   - 解析JSON格式的工具调用
   - 转换为标准格式: `{"role": "assistant", "content": "...", "tool_calls": [...]}`

2. **添加到对话历史**
   ```python
   self.conversation_history.append(assistant_entry)
   ```

3. **调用EnvHandler**
   ```python
   env_resp = self.env_handler.interact(
       self.conversation_history,
       self.original_test_entry
   )
   ```

4. **处理EnvHandler响应**
   - 如果返回工具结果: 转换为Qwen格式的工具消息
   - 如果返回用户查询: 更新 `current_turn`，返回用户消息
   - 如果返回完成信号: `[CONVERSATION_COMPLETED]`

### 4.2 EnvHandler.interact() 处理逻辑
**位置**: `env_handler.py:78-128`

#### 4.2.1 判断当前轮次
```python
current_turn = self._get_current_turn(messages, test_entry)
# 统计user消息数量作为当前轮次
```

#### 4.2.2 处理工具调用
如果最后一条消息包含 `tool_calls`:

1. **转换工具调用格式**
   ```python
   decoded_calls = self._convert_tool_calls_to_execution_format(tool_calls)
   # 转换: [{"function": {"name": "mkdir", "arguments": {"dir_name": "temp"}}}]
   # 到: ["mkdir(dir_name='temp')"]
   ```

2. **执行工具调用**
   ```python
   execution_results, _ = execute_multi_turn_func_call(
       func_call_list=decoded_calls,
       initial_config=test_entry["initial_config"],
       involved_classes=test_entry["involved_classes"],
       ...
   )
   ```
   - 在模拟环境中执行Python函数调用
   - 返回执行结果或错误信息

3. **创建工具响应**
   ```python
   return self._create_tool_response(tool_calls, execution_results)
   # 返回: {"messages": [{"role": "tool", "content": result, "tool_call_id": "..."}]}
   ```

#### 4.2.3 处理用户轮次
如果没有工具调用或工具调用为空:

1. **检查是否有新查询**
   ```python
   if current_turn >= len(questions):
       return self._create_completion_response()  # [CONVERSATION_COMPLETED]
   ```

2. **获取当前轮次的查询**
   ```python
   current_turn_message = questions[current_turn]
   ```

3. **编译可用工具**
   ```python
   tools = self._compile_tools(test_entry)
   # 处理holdout_function（某些轮次新增的工具）
   ```

4. **创建用户响应**
   ```python
   return self._create_user_response(current_turn_message, tools)
   # 返回: {"messages": [{"role": "user", "content": "..."}], "tools": [...]}
   ```

### 4.3 工具调用执行
**位置**: `env_handler.py:146-177`

1. **执行函数调用**
   - 使用 `execute_multi_turn_func_call` 在模拟环境中执行
   - 支持多工具并行调用
   - 捕获执行结果或错误

2. **返回结果格式**
   - 成功: `{"travel_cost_list": [1140.0]}`
   - 失败: `{"error": "cd: temporary: No such directory..."}`

---

## 五、消息格式转换

### 5.1 LLM输出 → 环境输入
**位置**: `bfcl_env.py:39-105`

LLM输出格式（Qwen）:
```
思考内容...

<tool_call>
{"name": "mkdir", "arguments": {"dir_name": "temp"}}
</tool_call>
```

转换为环境格式:
```python
{
    "role": "assistant",
    "content": "思考内容...",
    "tool_calls": [
        {
            "id": "mkdir_1",
            "type": "function",
            "function": {
                "name": "mkdir",
                "arguments": {"dir_name": "temp"}
            }
        }
    ]
}
```

### 5.2 环境输出 → LLM输入
**位置**: `bfcl_env.py:153-203`

工具结果格式:
```python
{
    "role": "tool",
    "content": {"travel_cost_list": [1140.0]},
    "tool_call_id": "chatcmpl-tool-xxx"
}
```

转换为Qwen格式:
```
<tool_call>
{"name": "chatcmpl-tool-xxx", "content": {"travel_cost_list": [1140.0]}}
</tool_call>
```

---

## 六、终止和评估

### 6.1 终止条件
**位置**: `bfcl_env.py:414-419`

```python
def _is_terminated(self, env_content) -> bool:
    return env_content == "[CONVERSATION_COMPLETED]"
```

触发终止的情况:
1. 所有轮次的查询都已处理完成
2. 环境返回 `[CONVERSATION_COMPLETED]` 消息
3. 达到最大步数限制 (`max_steps`)
4. Token溢出

### 6.2 评估流程
**位置**: `agent_flow.py:159-170`

1. **计算奖励**
   ```python
   if self._reward_calculator is not None:
       grader_res = self._reward_calculator.calculate_reward(self.cmt, env, instance_id)
       score = grader_res["score"]
   else:
       score = env.evaluate(instance_id, params={"sparse": self.sparse})
   ```

2. **BFCL评估** (`BfclEnv.evaluate`)
   - 调用 `EnvHandler.evaluate()`
   - 根据测试类别选择评估方法:
     - `multi_turn`: 多轮评估
     - `single_turn`: 单轮AST评估
     - `relevance`: 相关性评估
   - 返回准确率 (0.0-1.0)

3. **保存奖励**
   ```python
   self.cmt.reward = Reward(
       outcome=score,
       success_rate=success_rate,
       madness=self.cmt.compute_madness(),
       description=reason
   )
   ```

---

## 七、完整交互示例

### 7.1 示例场景
任务: 创建一个目录并查看其内容

### 7.2 交互序列

**轮次0 (初始化)**
```
System: [工具描述XML]
User: "创建一个名为temp的目录，然后查看其内容"
```

**轮次1 (第1步)**
```
LLM输出:
"我来帮你创建目录。
<tool_call>
{"name": "mkdir", "arguments": {"dir_name": "temp"}}
</tool_call>"

环境响应 (工具结果):
<tool_call>
{"name": "mkdir_1", "content": {"status": "success"}}
</tool_call>
```

**轮次2 (第2步)**
```
LLM输出:
"现在查看目录内容。
<tool_call>
{"name": "ls", "arguments": {}}
</tool_call>"

环境响应 (工具结果):
<tool_call>
{"name": "ls_1", "content": {"files": ["temp"]}}
</tool_call>
```

**轮次3 (第3步)**
```
LLM输出:
"目录创建成功，内容为空目录。"

环境响应 (新查询):
User: "请列出temp目录中的所有文件"

[继续下一轮...]
```

**最终轮次**
```
环境响应:
[CONVERSATION_COMPLETED]

评估: accuracy = 1.0 (成功)
```

---

## 八、关键数据结构

### 8.1 conversation_history
```python
[
    {"role": "user", "content": "第一个查询"},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": {...}, "tool_call_id": "..."},
    {"role": "user", "content": "第二个查询"},
    ...
]
```

### 8.2 test_entry
```python
{
    "id": "multi_turn_base_0",
    "question": [
        [{"role": "user", "content": "第一个查询"}],
        [{"role": "user", "content": "第二个查询"}],
        ...
    ],
    "function": [...],  # 工具定义列表
    "initial_config": {...},  # 环境初始配置
    "involved_classes": {...},  # 涉及的类定义
    "holdout_function": {...}  # 特定轮次新增的工具
}
```

### 8.3 Trajectory (CMT)
```python
{
    "full_context": [ExtendedMessage, ...],  # 完整对话历史
    "steps": [...],  # 步骤列表
    "is_terminated": bool,
    "reward": Reward,
    "metadata": {...}
}
```

---

## 九、错误处理

### 9.1 工具调用错误
- 环境返回 `{"error": "错误信息"}`
- 转换为工具消息格式，LLM可以基于错误信息重试

### 9.2 环境交互错误
- 捕获异常，标记轨迹为未完成
- 设置 `err_in_env = True`，终止循环

### 9.3 Token溢出
- 检查上下文长度
- 如果超过限制，提前终止（标记为未完成）

### 9.4 重试机制
- `max_llm_retries`: LLM调用重试次数（默认3次）
- `max_retry`: 环境工作器重试次数（默认4次）

---

## 十、性能优化

### 10.1 并行执行
- 使用 `ThreadPoolExecutor` 并行执行多个rollout
- 通过 `tmux` 跟踪所有线程的进度

### 10.2 上下文管理
- 使用 `Linear_CMT` 管理对话历史
- 支持上下文裁剪（`SelfContextClipCMT`）

### 10.3 异步LLM调用
- 使用 `BaAsyncLLMServerManager` 管理异步LLM请求
- 支持多个LLM服务器负载均衡

---

## 总结

BFCL多轮交互的核心流程:
1. **初始化**: 创建环境实例，加载测试用例，初始化对话历史
2. **循环交互**: LLM生成响应 → 解析工具调用 → 执行工具 → 获取结果 → 处理新查询
3. **终止评估**: 检查终止条件，计算奖励，保存轨迹

关键特点:
- 支持多轮对话，每轮可能有多个工具调用
- 工具调用结果和新查询交替出现
- 使用Qwen格式的工具调用标记 (`<tool_call>`)
- 通过 `current_turn` 跟踪对话轮次


