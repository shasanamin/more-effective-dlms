# GroupEval

**Judge generations in groups, so repetition across outputs counts.** Perplexity and token entropy grade one output at a time. A generator that keeps producing the same fluent document can score well on both. GroupEval shows a blinded LLM judge a group of outputs from one generator and returns two separate scores on a 0–10 scale:

- **GEQ (GroupEval Quality): is each output good?** The mean of four rubric dimensions (readability, coherence, substance, non-degeneration), with invalid outputs counted as zero.
- **GED (GroupEval Diversity): do the outputs say different things?** The entropy of the group's semantic modes among valid outputs, scaled by the fraction of valid outputs.

Read the two together. High GEQ with low GED means good but similar outputs. High GED with low GEQ means varied but weak ones. Use GroupEval to compare checkpoints, samplers, or step budgets under one fixed rubric and judge.

This directory has the paper's prompts, schemas, grouping and scoring code, and judge-separated aggregate results. Scoring and request preparation run on CPU, and the offline example needs no model weights or API credentials.

## Start here

From this directory, with Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
python examples/run_offline.py --output-dir /tmp/groupeval-example
```

The example validates a supplied synthetic judgment, computes **GEQ = 7.34375** and **GED = 6.2069154**, and renders one request without sending it. Its eight short texts and hand-written judgments show the interface and are not experimental evidence. It uses a single group rather than the paper's two-partition design. One candidate deliberately contains an instruction aimed at the evaluator, to show that candidate text is treated as untrusted data.

Outputs include `scores/system_scores.csv`, `scores/group_scores.csv`, `scores/sample_scores.csv`, and `request_preview.jsonl`. A temporary output directory is used if `--output-dir` is omitted.

## Evaluate your own outputs

### 1. Prepare a manifest

Write a CSV with one row per output, using either `text` or `text_path`. Relative text paths are resolved beside the manifest. Use one `method_code` per generator setting and the same set of sample indices across settings. The default is 128 outputs per setting.

```csv
method_code,sample_index,sample_id,target_profile_id,text_path
setting_a,0,a_000,owt1024-fullstream-v1,samples/a_000.txt
setting_a,1,a_001,owt1024-fullstream-v1,samples/a_001.txt
```

The two rows above illustrate the format; supply all 128 rows for each setting. Keep model and sampler names out of candidate text. Before writing the manifest, follow the [full-stream target profile](protocol/target_profile_owt1024_fullstream.md): remove exactly one leading `<|endoftext|>` when present, replace subsequent sentinels with `[DOCUMENT BOUNDARY]`, and preserve the entire stream. The generic grouping tool does not tokenize or preprocess text. For another domain or length policy, declare a separate profile and use it consistently across compared systems.

### 2. Form blinded groups and inspect requests

```bash
python experiment/grouping.py \
  --manifest runs/manifest.csv --output runs/groups.jsonl

python experiment/judge_runner_template.py \
  --groups runs/groups.jsonl --output runs/request_preview.jsonl \
  --provider inst --model 'gpt-oss:120b' \
  --target-profile protocol/target_profile_owt1024_fullstream.md \
  --dry-run
```

The grouping tool partitions each output pool twice into groups of eight. It uses seeds `271828,314159` and selects the second partition from 2,000 deterministic candidates to reduce repeated pairs. Each output is judged twice, giving **32 accepted group calls per 128-output setting**. Method labels are retained privately for scoring and omitted from the judge prompt.

The commands above are offline. To judge outputs, install `requirements-api.txt`, configure your own endpoint and credentials, and run the second command without `--dry-run`, with `--output runs/judgments.jsonl`. The `inst` adapter speaks Chat Completions with strict structured output. Set `INST_GENAI_URL` to the full HTTPS completion endpoint and `INSTITUTIONAL_GENAI_API_KEY` to its bearer credential. `.env.example` lists these placeholders, but the runner does not load it automatically. The hosted service used in the paper is not included.

The runner also supports `--provider openai --model YOUR_MODEL` through the Responses API and `OPENAI_API_KEY`. Treat a different judge as a separate result series. Use `--resume` to continue a compatible partial run. Responses are validated locally against the schema and its semantic invariants, with bounded retries and atomic checkpoints. Invalid judge content is never repaired into a score.

### 3. Score accepted judgments

```bash
python scoring/validate_outputs.py --input runs/judgments.jsonl
python scoring/score_group_eval.py \
  --judgments runs/judgments.jsonl --out-dir runs/scores
python scoring/bootstrap_group_eval.py \
  --judgments runs/judgments.jsonl --out-dir runs/uncertainty \
  --replicates 10000 --pairing-mode partitions
```

Report GEQ and GED separately. Retain invalidity, severe degeneration, singleton/ceiling rates, and clustering confidence alongside the headline scores. The bootstrap gives conditional GEQ intervals. Its GED percentile columns are **weight-sensitivity ranges, not calibrated confidence intervals**. No GED threshold or non-inferiority margin is used in the paper.

## Paper default and comparable reporting

[paper_default.yaml](protocol/paper_default.yaml) records the paper's unconditional evaluation profile: full-stream web text, groups of eight, two partitions, the v1 rubric, and `gpt-oss:120b`. It documents the configuration and is not a runner option. The commands above pass its settings explicitly.

[protocol_v1.yaml](protocol/protocol_v1.yaml) is the earlier protocol file that archived runs and analyzers identify by hash. Its proposed reference judge and threshold policy are not the paper's settings. Where the two files differ, `paper_default.yaml` describes what the paper used.

For comparisons, hold the target profile, preprocessing, judge configuration, group size, grouping design, and sample count fixed. Record an immutable judge revision when available, otherwise the returned model/fingerprint and access date. The paper's hosted aliases do not identify immutable weights, so the same alias cannot guarantee identical future scores. Include a common baseline when changing the judge or serving setup and report the resulting series separately. Conditional task rubrics define separate scales.

GEQ measures judged quality, and GED measures local semantic diversity. Neither is a claim of human calibration or complete distributional coverage. GED can saturate when a small group contains only distinct modes; the accompanying diagnostics expose this case.

## Paper results and reproduction

See [results/README.md](results/README.md) for the mapping from paper claims to CSVs: five MDLM judge series, three DiDi series, the matched 40-output comparison, common likelihood rescoring, and the conditional criterion check. The matched comparison uses its own cohort. Do not substitute the standalone 128-output MDLM scores into a comparison with 40-output DiDi scores.

The aggregate CSVs are ready for plotting and table inspection, and the JSON summaries record analysis settings, score diagnostics, and artifact hashes. Raw generations, individual judgments, and credentials are not included, so exact reaggregation of every interval needs the original per-output archives. The study-specific preparers and analyzers are included for anyone with those archives, and their input checks reject changed cohorts.

The conditional study has a separate [workflow guide](experiment/conditional_large_dlm_README.md). It assesses one response per model and task condition, so it defines cGEQ but no GED. The general-purpose path for new unconditional generators is the manifest workflow above.

## Files and offline checks

| Directory | Contents |
|---|---|
| `protocol/`, `prompts/`, `schemas/` | Measurement specifications used in the paper, and the paper-default profile |
| `scoring/` | Validation, GEQ/GED aggregation, conditional bootstrap |
| `experiment/` | Group construction, request runner, study-specific preparation and analysis |
| `examples/` | Small synthetic input, judgment, expected scores, offline walkthrough |
| `results/` | Current-paper aggregate results and provenance |
| `tests/` | CPU-only checks using fixtures and mocked providers |

```bash
python tests/smoke_test.py
python -m unittest discover -s tests -p 'test_*.py'
```

These checks do not contact judge services, download models, or train models. `smoke_test.py` expects a POSIX system (Linux, macOS, or WSL). The causal-LM accounting tests use a tiny CPU toy model when `torch` is installed and otherwise skip that module. Actual causal-LM rescoring needs a separate model environment (`torch`, `transformers`, and the declared evaluator weights).

Code is distributed under the included [MIT license](LICENSE). Cite the accompanying paper when using GroupEval; the author citation is omitted from this anonymous review package.
