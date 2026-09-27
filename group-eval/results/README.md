# GroupEval results used in the paper

These are aggregate outputs from the paper's fixed generation pools, with unrounded estimates and diagnostics. Directory dates identify judge runs and are also what the analysis scripts expect. No raw generations, individual judgments, response bodies, or credentials are included.

| Paper analysis | Directory or pattern | Main files |
|---|---|---|
| MDLM default and judge sensitivity, including the `M=4` floor case | `mdlm_pilot_*` (five directories) | `pilot_results.csv`, `pilot_deltas.csv`, `pilot_summary.json` |
| DiDi boundary case | `didi_pilot_*` (three directories) | Same layout |
| Initial three-judge agreement | `mdlm_judge_comparison_20260727` | `judge_bridge.csv`, `judge_bridge_summary.json` |
| Matched 40-output MDLM/DiDi comparison | `cross_model_exact40_gpt_oss_120b_20260731` | `cell_results.csv`, `matched_cross_model_contrasts.csv`, `summary.json` |
| Common GPT-2-Large and Qwen3-8B-Base rescoring | `common_causal_rescoring_qwen3_8b_20260731` | `cell_metrics.csv`, `matched_mdlm_didi.csv`, `rank_correlations.csv`, four evaluator/span summaries |
| GSM8K and HumanEval criterion checks | `large_dlm_conditional_gpt_oss_120b_20260731` | `task_model_summary.csv`, `construct_validity.csv`, `discordant_comparison.csv`, `paired_model_comparison.csv`, `summary.json` |

## Cohorts and interpretation

- The standalone MDLM studies use 128 outputs per setting, with GPT-OSS, Gemma, Qwen, Luna, and Terra as separate judges. The standalone DiDi studies use 40 outputs per setting, with GPT-OSS, Luna, and Terra. The shared panel supports sensitivity comparisons; no scores are pooled across judges.
- `reference` denotes `(temperature=1, categorical_precision=fp64)` and `selected` denotes `(.9, fp32)`. These files contain no GroupEval judgments for the two intermediate temperature-only or precision-only configurations.
- The cross-generator study is a distinct, globally interleaved GPT-OSS run on the exact 40 outputs per model/setting/budget used for common likelihood rescoring. Use its matched contrasts for cross-model claims, rather than mixing standalone cohorts.
- Common rescoring reports both `full_stream` and `first_document` spans under the declared aggregation rules, together with bits-per-byte, rank associations, and the coverage diagnostics used in the appendix. The four source summaries are the inputs for regenerating the aggregate analysis (command below).
- Conditional cGEQ uses 256 paired GSM8K conditions and 164 paired HumanEval conditions, each with one response per model. The task-specific scales differ from unconditional GEQ. GED is not defined for this design.

GEQ intervals condition on the accepted judgments and the declared bootstrap units. Fields named `ged_ci_*`, and analogous GED ranges, are **weighted sensitivity ranges**, not calibrated confidence intervals. The paper does not infer diversity significance from them. Report point GED together with invalidity, singleton/ceiling rates, and clustering diagnostics.

JSON summaries record the evaluation settings, input hashes, and provider-returned metadata needed to trace provenance and rerun the aggregate combiners. An omitted reasoning parameter means the request left that field out. It does not verify the provider's effective reasoning level. Hosted model aliases and serving fingerprints do not pin immutable model weights.

## Recompute the common causal-rescoring analysis

From the `group-eval` directory, this uses only released aggregates and CPU:

```bash
python experiment/analyze_causal_rescoring.py \
  --gpt2-full-summary results/common_causal_rescoring_qwen3_8b_20260731/gpt2_large_full_stream_summary.json \
  --gpt2-first-summary results/common_causal_rescoring_qwen3_8b_20260731/gpt2_large_first_document_summary.json \
  --qwen-full-summary results/common_causal_rescoring_qwen3_8b_20260731/qwen3_8b_base_full_stream_summary.json \
  --qwen-first-summary results/common_causal_rescoring_qwen3_8b_20260731/qwen3_8b_base_first_document_summary.json \
  --group-eval-csv results/cross_model_exact40_gpt_oss_120b_20260731/cell_results.csv \
  --output-dir /tmp/groupeval-causal-analysis
```

Rebuilding intervals from individual judgments or rerunning evaluator models requires the original archives or a new, separately reported study. The offline example and tests exercise the analysis interfaces with synthetic fixtures.
