# AgentFlow Training and Reward Design Overview

This note summarizes how AgentFlow trains the planner agent, what data is used, how Flow-GRPO is wired, and how rewards are computed for open-ended answers. It also sketches a training approach for SQL-style tasks.

## Datasets and Task Mix
- **Training set**: `data/train/combined_train.parquet`, a shuffled mix of:
  - **Natural Questions (NQ)** for open-domain search and tool use.
  - **DeepMath-103K** for multi-step mathematical reasoning.
- **Validation set**: `data/val/aime24.parquet` for quick convergence checks.
- **Answer extraction helper**: `train/rollout.py` wraps each question with `<answer></answer>` guidance so the final answer can be extracted reliably from multi-turn traces.

## Training Approach and Highlights
- **What is trained**: The **Planner** is optimized online; Executor/Verifier/Generator typically use fixed or lower-temperature engines.
- **Flow-based GRPO**: Uses a system-in-the-loop variant of Group Refined Policy Optimization to sample, score, and update on real tool-augmented rollouts, reducing train–test mismatch.
- **Tool-rich long-horizon reasoning**: `enabled_tools`/`tool_engine` allow search + code execution. Flow-GRPO optimizes planner decisions on when/which tool to call and when to stop.

## Flow-GRPO Wiring
- **Entry point**: `train/train_agent.py` loads `train/config.yaml`, sets env vars, then runs `python -m agentflow.verl`.
- **Key knobs** (see `train/config.yaml`):
  - `algorithm.adv_estimator: grpo`, `data.train_batch_size: 32`, `actor_rollout_ref.rollout.n: 8`, `total_epochs: 5`.
  - `MODEL_ENGINE: ["trainable","dashscope","dashscope","dashscope"]` → only the planner is trainable.
  - `TOOL_STEPS: 3` caps tool calls per episode to avoid context overflow.

## Reward Design (incl. Open-Ended Answers)
- **Where computed**: `@reward`-decorated `eval` in `train/rollout.py`, which calls `utils.compute_score`.
- **How judged**:
  - After rollout, extract the last `<answer>...</answer>` block; fallback to `direct_output` if missing.
  - `utils.compute_score` uses `gpt-4o` to test semantic/mathematical equivalence (numeric normalization, LaTeX math, MCQ option mapping).
  - Reward is **binary 1.0/0.0**; optional KL term (`algorithm.use_kl_in_reward`) can be enabled.
- **Why this helps open-ended tasks**: LLM judging tolerates format differences while still enforcing correctness, avoiding brittle string equality for long-form reasoning outputs.

## Suggested Path for SQL Tasks
- **Data**:
  - Start with SFT on structured QA corpora (e.g., Spider/BIRD): provide *question + schema + gold SQL + execution result* to teach schema grounding and SQL generation (planner/executor).
  - RL stage: keep the same rollout scaffold; feed database schemas via prompts.
- **Rewards**:
  - Primary: execution correctness (run model SQL, compare result to expected → 1/0).
  - Auxiliary (weighted, optional): syntax passes, safety constraints (no DROP/DELETE; allowed tables/columns only), and plan validity.
  - If a natural-language explanation is required, combine execution reward with a light LLM-consistency score on the explanation.
- **Practice tips**:
  - Include schema + few example rows in the prompt; keep `TOOL_STEPS` ≤ 3–4 to stay within context.
  - For hard multi-join queries, let the planner outline sub-queries first, then have the executor compose the final SQL—mirrors the existing multi-module flow.

## Takeaway
AgentFlow couples mixed search/math data, online Flow-GRPO, and LLM-based rewards to strengthen the planner’s tool decisions and long-horizon reasoning. The same recipe applies to SQL: swap in execution-based rewards and schema-aware data while reusing the current multi-tool, multi-module training pipeline.
