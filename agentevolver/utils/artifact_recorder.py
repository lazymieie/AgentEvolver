"""
Artifact Recorder for paper run artifacts.

This module provides a non-intrusive logging system that records key artifacts
during training without affecting the training logic.
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Lazy import logger to avoid serialization issues
def _get_logger():
    import logging
    return logging.getLogger(__name__)

# Use file-based locking instead of threading.Lock for Ray compatibility
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    try:
        import msvcrt
        HAS_MSVCRT = True
    except ImportError:
        HAS_MSVCRT = False


class ArtifactRecorder:
    """
    A thread-safe recorder for writing training artifacts to JSONL files.
    
    Features:
    - Thread-safe file writing (each rank writes to separate file)
    - Automatic truncation of long strings
    - Automatic metadata injection (timestamp, global_step, rank, pid)
    - Configurable output directory and enable/disable flag
    """
    
    def __init__(
        self,
        enable: bool = False,
        out_dir: str = "./artifacts",
        max_str_chars: int = 20000,
        dump_prompts: bool = True,
        dump_steps: bool = True,
        dump_experiences: bool = True,
        dump_adca: bool = True,
        dump_filters: bool = True,
        dump_rewards: bool = True,
        max_examples_per_step: int = 3,
        rank: Optional[int] = None,
    ):
        """
        Initialize the artifact recorder.
        
        Args:
            enable: Whether to enable artifact recording
            out_dir: Output directory for artifacts
            max_str_chars: Maximum characters for string fields (truncate if longer)
            dump_prompts: Whether to dump prompts in records
            dump_steps: Whether to dump step details in rollouts
            dump_experiences: Whether to dump experience-related artifacts
            dump_adca: Whether to dump ADCA-related artifacts
            dump_filters: Whether to dump filter-related artifacts
            dump_rewards: Whether to dump reward-related artifacts
            max_examples_per_step: Maximum examples to record per global step
            rank: Process rank (for multi-process safety)
        """
        self.enable = enable
        # Store as string for Ray serialization compatibility
        self.out_dir_str = str(out_dir)
        self.max_str_chars = max_str_chars
        self.dump_prompts = dump_prompts
        self.dump_steps = dump_steps
        self.dump_experiences = dump_experiences
        self.dump_adca = dump_adca
        self.dump_filters = dump_filters
        self.dump_rewards = dump_rewards
        self.max_examples_per_step = max_examples_per_step
        
        # Get rank and pid for file naming
        self.rank = rank if rank is not None else self._get_rank()
        self.pid = os.getpid()
        
        # Track first write for logging (no file handles needed - open/close each time for Ray compatibility)
        self._first_write_flags: Dict[str, bool] = {}
        
        # Per-step counters (to limit examples per step)
        self._step_counters: Dict[int, Dict[str, int]] = {}
        
        if self.enable:
            out_dir_path = Path(self.out_dir_str)
            out_dir_path.mkdir(parents=True, exist_ok=True)
            _get_logger().info(f"ArtifactRecorder enabled: out_dir={self.out_dir_str}, rank={self.rank}, pid={self.pid}")
    
    def __getstate__(self):
        """Custom serialization for Ray compatibility."""
        # Return only serializable attributes
        return {
            'enable': self.enable,
            'out_dir_str': self.out_dir_str,
            'max_str_chars': self.max_str_chars,
            'dump_prompts': self.dump_prompts,
            'dump_steps': self.dump_steps,
            'dump_experiences': self.dump_experiences,
            'dump_adca': self.dump_adca,
            'dump_filters': self.dump_filters,
            'dump_rewards': self.dump_rewards,
            'max_examples_per_step': self.max_examples_per_step,
            'rank': self.rank,
            'pid': self.pid,
            '_first_write_flags': self._first_write_flags,
            '_step_counters': self._step_counters,
        }
    
    def __setstate__(self, state):
        """Custom deserialization for Ray compatibility."""
        # Restore all attributes
        self.enable = state['enable']
        self.out_dir_str = state['out_dir_str']
        self.max_str_chars = state['max_str_chars']
        self.dump_prompts = state['dump_prompts']
        self.dump_steps = state['dump_steps']
        self.dump_experiences = state['dump_experiences']
        self.dump_adca = state['dump_adca']
        self.dump_filters = state['dump_filters']
        self.dump_rewards = state['dump_rewards']
        self.max_examples_per_step = state['max_examples_per_step']
        self.rank = state['rank']
        self.pid = state['pid']
        self._first_write_flags = state.get('_first_write_flags', {})
        self._step_counters = state.get('_step_counters', {})
    
    def _get_rank(self) -> int:
        """Try to get rank from environment or return 0."""
        # Try from environment first (most reliable)
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        try:
            return int(rank)
        except:
            pass
        
        # Try to get from Ray (only if Ray is available and initialized)
        # Note: This is called during __init__, so Ray may not be initialized yet
        # We avoid calling this during serialization by caching the rank
        try:
            import ray
            if ray.is_initialized():
                try:
                    ctx = ray.get_runtime_context()
                    if hasattr(ctx, 'get_node_id'):
                        # Use node_id as a proxy for rank (not perfect but works)
                        node_id = ctx.get_node_id()
                        return hash(node_id) % 1000  # Simple hash
                except:
                    pass
        except:
            pass
        
        return 0
    
    
    def _get_file_path(self, name: str) -> Path:
        """Get file path for a record name."""
        # Add rank suffix for multi-process safety
        if self.rank is not None and self.rank != 0:
            base_name = f"{name}.rank{self.rank}"
        else:
            base_name = name
        return Path(self.out_dir_str) / f"{base_name}.jsonl"
    
    def _truncate_str(self, s: Union[str, Any], max_chars: Optional[int] = None) -> Union[str, Any]:
        """Truncate string if it exceeds max_chars."""
        if max_chars is None:
            max_chars = self.max_str_chars
        
        if isinstance(s, str) and len(s) > max_chars:
            return s[:max_chars] + f"... [truncated, original length: {len(s)}]"
        return s
    
    def _truncate_dict(self, d: Dict[str, Any], max_chars: Optional[int] = None) -> Dict[str, Any]:
        """Recursively truncate string values in a dict."""
        if max_chars is None:
            max_chars = self.max_str_chars
        
        result = {}
        for k, v in d.items():
            if isinstance(v, str):
                result[k] = self._truncate_str(v, max_chars)
            elif isinstance(v, dict):
                result[k] = self._truncate_dict(v, max_chars)
            elif isinstance(v, list):
                result[k] = [self._truncate_str(item, max_chars) if isinstance(item, str) else item for item in v]
            else:
                result[k] = v
        return result
    
    def _should_record(self, name: str, global_step: Optional[int] = None) -> bool:
        """Check if we should record this entry (respect max_examples_per_step)."""
        if not self.enable:
            return False
        
        if global_step is None:
            return True
        
        if global_step not in self._step_counters:
            self._step_counters[global_step] = {}
        
        counter = self._step_counters[global_step].get(name, 0)
        if counter >= self.max_examples_per_step:
            return False
        
        self._step_counters[global_step][name] = counter + 1
        return True
    
    def write_jsonl(
        self,
        name: str,
        obj: Dict[str, Any],
        global_step: Optional[int] = None,
        force: bool = False,
    ):
        """
        Write an object to a JSONL file.
        
        Args:
            name: Record name (will be used as filename base)
            obj: Dictionary to write
            global_step: Current global step (for limiting examples per step)
            force: Force write even if max_examples_per_step limit reached
        """
        if not self.enable:
            return
        
        if not force and not self._should_record(name, global_step):
            return
        
        # Add metadata
        record = {
            "_ts": time.time(),
            "_pid": self.pid,
            "_rank": self.rank,
            **obj,
        }
        if global_step is not None:
            record["global_step"] = global_step
        
        # Truncate long strings
        record = self._truncate_dict(record)
        
        # Get file path
        file_path = self._get_file_path(name)
        
        # Log first write
        if name not in self._first_write_flags:
            _get_logger().info(f"Writing artifacts to {file_path}")
            self._first_write_flags[name] = True
        
        # Write to file (open/close each time to avoid serialization issues with Ray)
        # Each rank writes to separate file, so no locking needed
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                # Optional: use file locking for extra safety (but not required since each rank has separate file)
                try:
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    elif HAS_MSVCRT:
                        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                except (OSError, IOError, AttributeError):
                    pass  # Lock failed, continue anyway
                
                try:
                    # Write record
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()  # Ensure data is written
                finally:
                    # Unlock file
                    try:
                        if HAS_FCNTL:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        elif HAS_MSVCRT:
                            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                    except (OSError, IOError, AttributeError):
                        pass
                        
        except Exception as e:
            _get_logger().warning(f"Failed to write artifact {name}: {e}")
    
    def write_tasks_sampled(
        self,
        tasks: List[Any],
        global_step: Optional[int] = None,
        mixture_source: Optional[str] = None,
        is_task_objective: bool = False,
    ):
        """Record sampled tasks from TaskManager."""
        if not self.enable:
            return
        
        for i, task in enumerate(tasks[:self.max_examples_per_step]):
            if not self._should_record("tasks_sampled", global_step):
                break
            
            # Handle TaskObjective vs Task
            if is_task_objective or hasattr(task, "task"):
                # This is a TaskObjective, extract the task
                task_obj = task
                task = task_obj.task if hasattr(task_obj, "task") else task
                confidence = getattr(task_obj, "confidence", None)
                reward = getattr(task_obj, "reward", None)
            else:
                confidence = None
                reward = None
            
            # Extract task fields
            query_value = getattr(task, "query", None) or getattr(task, "new_query", None)
            is_seed_task = query_value is None or query_value == ""
            
            task_dict = {
                "task_id": getattr(task, "task_id", f"unknown_{i}"),
                "query": self._truncate_str(query_value or ""),
                "open_query": getattr(task, "open_query", False),
                "env_type": getattr(task, "env_type", "unknown"),
                "evaluator": getattr(task, "evaluator", "unknown"),
                "ground_truth": self._truncate_str(getattr(task, "ground_truth", None) or ""),
                "is_seed_task": is_seed_task,  # Mark if this is a seed task (no query yet)
            }
            
            # Add TaskObjective-specific fields if available
            if confidence is not None:
                task_dict["confidence"] = confidence
            if reward is not None:
                task_dict["reward"] = reward
            
            # Add metadata if available
            if hasattr(task, "metadata") and task.metadata:
                task_dict["metadata"] = task.metadata
            
            if mixture_source:
                task_dict["mixture_source"] = mixture_source
            
            self.write_jsonl("tasks_sampled", task_dict, global_step)
    
    def write_tasks_filtered(
        self,
        task_id: str,
        passed: bool,
        reasons: Optional[List[str]] = None,
        scores: Optional[Dict[str, Any]] = None,
        global_step: Optional[int] = None,
    ):
        """Record task filtering results."""
        if not self.enable or not self.dump_filters:
            return
        
        record = {
            "task_id": task_id,
            "passed": passed,
            "reasons": reasons or [],
            "scores": scores or {},
        }
        self.write_jsonl("tasks_filtered", record, global_step)
    
    def write_rollouts(
        self,
        task_id: str,
        traj_id: Optional[str],
        mode: str,
        n_steps: int,
        steps: Optional[List[Dict[str, Any]]] = None,
        global_step: Optional[int] = None,
    ):
        """Record rollout trajectories."""
        if not self.enable:
            return
        
        record = {
            "task_id": task_id,
            "traj_id": traj_id or f"{task_id}_unknown",
            "mode": mode,
            "n_steps": n_steps,
        }
        
        if self.dump_steps and steps:
            # Truncate step details
            truncated_steps = []
            for step in steps[:50]:  # Limit to 50 steps
                step_dict = {
                    "t": step.get("t", step.get("step", 0)),
                    "obs": self._truncate_str(step.get("obs", step.get("observation", ""))),
                    "action": self._truncate_str(step.get("action", step.get("response", ""))),
                }
                if "reward" in step:
                    step_dict["reward"] = step.get("reward")
                truncated_steps.append(step_dict)
            record["steps"] = truncated_steps
        
        self.write_jsonl("rollouts", record, global_step)
    
    def write_experience_retrieval(
        self,
        task_id: str,
        query: str,
        topk: List[Dict[str, Any]],
        global_step: Optional[int] = None,
    ):
        """Record experience retrieval results."""
        if not self.enable or not self.dump_experiences:
            return
        
        record = {
            "task_id": task_id,
            "query": self._truncate_str(query),
            "topk": [
                {
                    "exp_id": exp.get("exp_id", exp.get("id", "unknown")),
                    "score": exp.get("score", exp.get("similarity", 0.0)),
                    "when_to_use": self._truncate_str(exp.get("when_to_use", "")),
                    "content": self._truncate_str(exp.get("content", "")),
                    "source_task_id": exp.get("source_task_id", None),
                    "source_traj_id": exp.get("source_traj_id", None),
                }
                for exp in topk
            ],
        }
        self.write_jsonl("experience_retrieval", record, global_step)

    def write_generation_failures(
        self,
        task_id: str,
        traj_id: str,
        mode: str,
        step: int,
        reason: str,
        stage: Optional[str] = None,
        prompt_with_exp: Optional[str] = None,
        prompt_without_exp: Optional[str] = None,
        llm_output: Optional[Dict[str, Any]] = None,
        tool_names: Optional[List[str]] = None,
        global_step: Optional[int] = None,
    ):
        """Record generation failures during rollout."""
        if not self.enable:
            return

        record: Dict[str, Any] = {
            "task_id": task_id,
            "traj_id": traj_id,
            "mode": mode,
            "step": step,
            "reason": reason,
        }

        if stage is not None:
            record["stage"] = stage
        if prompt_with_exp is not None:
            record["prompt_with_exp"] = self._truncate_str(prompt_with_exp)
        if prompt_without_exp is not None:
            record["prompt_without_exp"] = self._truncate_str(prompt_without_exp)
        if llm_output is not None:
            record["llm_output"] = self._truncate_dict(llm_output)
        if tool_names is not None:
            record["tool_names"] = tool_names

        self.write_jsonl("generation_failures", record, global_step)
    
    def write_experience_injection(
        self,
        task_id: str,
        injection_template_name: Optional[str],
        prompt_with_exp: Optional[str] = None,
        prompt_without_exp: Optional[str] = None,
        global_step: Optional[int] = None,
    ):
        """Record experience injection."""
        if not self.enable or not self.dump_experiences or not self.dump_prompts:
            return
        
        record = {
            "task_id": task_id,
            "injection_template_name": injection_template_name,
        }
        
        if prompt_with_exp:
            record["prompt_with_exp"] = self._truncate_str(prompt_with_exp)
        if prompt_without_exp:
            record["prompt_without_exp"] = self._truncate_str(prompt_without_exp)
        
        self.write_jsonl("experience_injection", record, global_step)

    def write_state_experience_tool_event(
        self,
        task_id: str,
        traj_id: str,
        step: int,
        event_type: str,
        retry_idx: Optional[int] = None,
        max_retries: Optional[int] = None,
        injection_applied: Optional[bool] = None,
        tool_names: Optional[List[str]] = None,
        prompt_with_exp: Optional[str] = None,
        prompt_without_exp: Optional[str] = None,
        injected_experience: Optional[str] = None,
        llm_output: Optional[Dict[str, Any]] = None,
        global_step: Optional[int] = None,
    ):
        """Record state-level experience-tool events during rollout."""
        if not self.enable or not self.dump_experiences:
            return

        record: Dict[str, Any] = {
            "task_id": task_id,
            "traj_id": traj_id,
            "step": step,
            "event_type": event_type,
        }

        if retry_idx is not None:
            record["retry_idx"] = retry_idx
        if max_retries is not None:
            record["max_retries"] = max_retries
        if injection_applied is not None:
            record["injection_applied"] = injection_applied
        if tool_names is not None:
            record["tool_names"] = tool_names
        if prompt_with_exp is not None and self.dump_prompts:
            record["prompt_with_exp"] = self._truncate_str(prompt_with_exp)
        if prompt_without_exp is not None and self.dump_prompts:
            record["prompt_without_exp"] = self._truncate_str(prompt_without_exp)
        if injected_experience is not None:
            record["injected_experience"] = self._truncate_str(injected_experience)
        if llm_output is not None:
            record["llm_output"] = self._truncate_dict(llm_output)

        self.write_jsonl("state_experience_tool_events", record, global_step)

    def write_experience_stripping(
        self,
        task_id: str,
        before: str,
        after: str,
        token_spans: Optional[List[Dict[str, int]]] = None,
        global_step: Optional[int] = None,
    ):
        """Record experience stripping during training."""
        if not self.enable or not self.dump_experiences:
            return
        
        record = {
            "task_id": task_id,
            "before": self._truncate_str(before),
            "after": self._truncate_str(after),
        }
        
        if token_spans:
            record["token_spans"] = token_spans
        
        self.write_jsonl("experience_stripping", record, global_step)
    
    def write_rewards(
        self,
        task_id: str,
        traj_id: str,
        reward_value: float,
        reward_detail: Optional[Dict[str, Any]] = None,
        grader_name: Optional[str] = None,
        judge_model: Optional[str] = None,
        global_step: Optional[int] = None,
    ):
        """Record reward calculation results."""
        if not self.enable or not self.dump_rewards:
            return
        
        record = {
            "task_id": task_id,
            "traj_id": traj_id,
            "reward_value": reward_value,
            "reward_detail": reward_detail or {},
            "grader_name": grader_name,
            "judge_model": judge_model,
        }
        self.write_jsonl("rewards", record, global_step)
    
    def write_adca_step_scores(
        self,
        task_id: str,
        traj_id: str,
        steps: List[Dict[str, Any]],
        global_step: Optional[int] = None,
    ):
        """Record ADCA step-wise scores."""
        if not self.enable or not self.dump_adca:
            return
        
        record = {
            "task_id": task_id,
            "traj_id": traj_id,
            "steps": steps,
        }
        self.write_jsonl("adca_step_scores", record, global_step)
    
    def write_adca_composite(
        self,
        task_id: str,
        traj_id: str,
        r_out: Optional[float],
        r_attr_norm: Optional[float],
        r_out_norm: Optional[float],
        alpha: Optional[float],
        reward_final: Optional[float],
        advantage_steps: Optional[List[float]] = None,
        step_to_token_map: Optional[List[Dict[str, Any]]] = None,
        global_step: Optional[int] = None,
    ):
        """Record ADCA composite reward and advantage."""
        if not self.enable or not self.dump_adca:
            return
        
        record = {
            "task_id": task_id,
            "traj_id": traj_id,
            "r_out": r_out,
            "r_attr_norm": r_attr_norm,
            "r_out_norm": r_out_norm,
            "alpha": alpha,
            "reward_final": reward_final,
            "advantage_steps": advantage_steps,
            "step_to_token_map": step_to_token_map or [],
        }
        self.write_jsonl("adca_composite", record, global_step)
    
    def generate_readme(self):
        """Generate README.md explaining the artifacts."""
        if not self.enable:
            return
        
        readme_path = Path(self.out_dir_str) / "README.md"
        
        readme_content = """# Training Artifacts

This directory contains artifacts recorded during training for paper reproduction and analysis.

## Files

### tasks_sampled.jsonl
Tasks sampled from TaskManager for training.
- **Source**: `agentevolver.module.task_manager.task_manager.TaskManager.generate_task()`
- **Fields**: task_id, query, open_query, env_type, evaluator, ground_truth, metadata, mixture_source
- **When**: Recorded when tasks are sampled from the dataset

### tasks_filtered.jsonl
Task filtering results (deduplication, feasibility, etc.).
- **Source**: `agentevolver.module.task_manager.filters.*`
- **Fields**: task_id, passed, reasons, scores
- **When**: Recorded when tasks go through filtering stages

### rollouts.jsonl
Rollout trajectories from environment interaction.
- **Source**: `agentevolver.module.agent_flow.agent_flow.AgentFlow.execute()`
- **Fields**: task_id, traj_id, mode, n_steps, steps (if dump_steps=true)
- **When**: Recorded after rollout completion

### experience_retrieval.jsonl
Experience retrieval results from experience pool.
- **Source**: `agentevolver.module.exp_manager.exp_manager.ExperienceWorker.manage_rollout_context()`
- **Fields**: task_id, query, topk (list of experiences with scores)
- **When**: Recorded when experiences are retrieved for a task

### experience_injection.jsonl
Experience injection into prompts.
- **Source**: `agentevolver.module.exp_manager.exp_manager.ExperienceWorker.manage_rollout_context()`
- **Fields**: task_id, injection_template_name, prompt_with_exp, prompt_without_exp
- **When**: Recorded when experiences are injected into rollout prompts

### experience_stripping.jsonl
Experience removal during training (for loss calculation).
- **Source**: `agentevolver.module.exp_manager.exp_manager.ExperienceWorker.manage_training_context()`
- **Fields**: task_id, before, after, token_spans
- **When**: Recorded when experiences are stripped from training prompts

### state_experience_tool_events.jsonl
State-level experience-tool events during rollout, including repeated calls after transient guidance injection.
- **Source**: `agentevolver.module.agent_flow.agent_flow.AgentFlow.execute()`
- **Fields**: task_id, traj_id, step, event_type, retry_idx, max_retries, injection_applied, tool_names, prompt_with_exp, prompt_without_exp, injected_experience, llm_output
- **When**: Recorded immediately when the model calls `get_experience_guidance` again after guidance injection/attempted injection

### generation_failures.jsonl
Generation failures during rollout, including repeated experience-guidance requests.
- **Source**: `agentevolver.module.agent_flow.agent_flow.AgentFlow.execute()`
- **Fields**: task_id, traj_id, mode, step, reason, stage, prompt_with_exp, prompt_without_exp, llm_output, tool_names
- **When**: Recorded when a rollout step is marked as generation failure

### rewards.jsonl
Reward calculation results.
- **Source**: `agentevolver.module.agent_flow.reward_calculator.*`
- **Fields**: task_id, traj_id, reward_value, reward_detail, grader_name, judge_model
- **When**: Recorded when rewards are calculated for trajectories

### adca_step_scores.jsonl
ADCA (Attribution-Driven Credit Assignment) step-wise scores.
- **Source**: `agentevolver.module.adv_processor.semantic_attribution.evaluate_step_flags_parallel_sync()`
- **Fields**: task_id, traj_id, steps (list of {t, label, r_attr, analysis})
- **When**: Recorded when ADCA evaluates steps (if enabled)

### adca_composite.jsonl
ADCA composite reward and advantage calculation.
- **Source**: `agentevolver.module.adv_processor.adca_grpo.compute_prm_grpo_advantages()`
- **Fields**: task_id, traj_id, r_out, r_attr_norm, r_out_norm, alpha, reward_final, advantage_steps, step_to_token_map
- **When**: Recorded when ADCA composite rewards are calculated (if enabled)

## Configuration

Artifacts are controlled by `debug_artifacts` config:
- `enable`: Enable/disable artifact recording (default: false)
- `out_dir`: Output directory (default: `${trainer.rollout_data_dir}/artifacts`)
- `max_examples_per_step`: Maximum examples to record per global step (default: 3)
- `max_str_chars`: Maximum characters for string fields (default: 20000)
- `dump_*`: Flags to control what to dump

## Usage

Enable artifacts by adding to your config:
```yaml
debug_artifacts:
  enable: true
  out_dir: ${trainer.rollout_data_dir}/artifacts
```

Or via Hydra override:
```bash
python3 -m agentevolver.main_ppo ... debug_artifacts.enable=true
```

## Notes

- Files are written per rank (rank suffix in filename) for multi-process safety
- Long strings are automatically truncated to `max_str_chars`
- Only `max_examples_per_step` examples are recorded per global step to avoid performance impact
- All records include metadata: `_ts` (timestamp), `_pid` (process ID), `_rank` (rank), `global_step`
"""
        
        try:
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_content)
        except Exception as e:
            _get_logger().warning(f"Failed to write README: {e}")



