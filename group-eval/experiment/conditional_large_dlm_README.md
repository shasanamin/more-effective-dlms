# Conditional quality criterion check

This workflow checks whether task-specific cGEQ tracks deterministic success on the paper's greedy LLaDA-8B-Instruct and Dream-v0-Base-7B outputs. A group contains one public task condition and one blinded response from each model. The judge receives no answers, hidden tests, canonical solutions, model names, or automatic pass/fail labels. With only one response per model and condition, GED is undefined.

The released [aggregate results](../results/large_dlm_conditional_gpt_oss_120b_20260731/) cover 256 aligned GSM8K conditions and 164 aligned HumanEval conditions. The task-conditioned rubric, schema, and profile are separate from unconditional GroupEval. `tests/test_conditional_large_dlm.py` exercises this workflow offline without the original archives.

## Reproduction from original archives

The preparation utility is specific to the frozen paper cohort. It verifies archived generation files, checkpoint revisions, source hashes, and deterministic success totals before constructing judge inputs. Supply the archived run root and GSM8K Arrow file explicitly. The raw archives are not redistributed in this release.

```bash
python experiment/prepare_large_dlm_conditional.py \
  --run-root /path/to/archived/evaluation/runs \
  --gsm8k-arrow /path/to/gsm8k-test.arrow \
  --output-dir /tmp/conditional-geq-inputs \
  --allow-code-execution
```

This step requires the original task-evaluation environment. It reconstructs HumanEval labels by executing generated programs through the supplied evaluator, hence the explicit `--allow-code-execution` acknowledgement. The default rescoring helper (`code/scripts/research/rescore_fast_dllm_humaneval.py`) and sanitizer (`fast-dllm/llada/sanitize.py`), both relative to the release root, can be overridden with `--rescore-script` and `--sanitizer`. Keep private labels separate from judge input.

Render requests without contacting a service:

```bash
python experiment/judge_runner_template.py \
  --groups /tmp/conditional-geq-inputs/conditional_groups.jsonl \
  --output /tmp/conditional-geq-preview.jsonl \
  --provider inst --model 'gpt-oss:120b' \
  --prompt prompts/group_eval_conditional_task.md \
  --schema schemas/conditional_geq_output.schema.json \
  --target-profile protocol/target_profile_large_dlm_tasks.md \
  --max-output-tokens 3000 --dry-run
```

To collect actual judgments, configure your own supported service as described in the main README, remove `--dry-run`, and choose a new output path. A changed judge is a separate series.

The paper-specific analyzer checks all 420 conditions and the preparation hashes before computing task/model means, paired contrasts, AUROC, and contrasts among conditions where exactly one model passes:

```bash
python experiment/analyze_large_dlm_conditional.py \
  --groups /tmp/conditional-geq-inputs/conditional_groups.jsonl \
  --judgments /path/to/conditional-judgments.jsonl \
  --private-label-map /tmp/conditional-geq-inputs/private_label_map.json \
  --output-dir /tmp/conditional-geq-analysis
```

The 10,000-replicate condition-cluster bootstrap is conditional on the accepted response-pair judgments. Functional execution remains the HumanEval success criterion; cGEQ is a separate task-quality diagnostic. Do not compare its numerical scale with unconditional GEQ.
