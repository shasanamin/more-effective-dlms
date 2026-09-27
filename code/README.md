# Sampling and model experiments

The MDLM, DUO, and SEDD implementation behind the paper, with sampling
temperature and categorical precision exposed as sampler options. Use it to
generate new samples under the standard and sharpened samplers, rerun the
paper's sampling grids, or measure posterior loss and sampling cost.

For a fast introduction, use the CPU figure and theory commands in the
[root README](../README.md). Everything here generates new samples and needs
checkpoints and a GPU environment.

## Environment

Use a separate Linux/CUDA environment. From the release root, run
`python -m pip install -r code/requirements.txt`. These are the research
dependency pins, except that Triton is left for PyTorch to choose.
FlashAttention is optional for the MDLM model and should be installed
separately for your PyTorch/CUDA build if you want it. Do not mix this
environment with Fast-dLLM, which needs a different Transformers version.

Obtain the public MDLM checkpoint from the upstream
[MDLM project](https://github.com/kuleshov-group/mdlm) or its linked release.
Checkpoint identities for the cross-model study are recorded in
[the manifest](../paper_data/cross_model_manifest.csv). We do not redistribute
checkpoints or our trained early/late checkpoints. The latter experiments can
be inspected from aggregates; independently repeating their training does not
reconstruct their original weights exactly.

## Compare the standard and sharpened MDLM samplers

Run from the release root with an absolute path to your checkpoint:

```bash
export CHECKPOINT="${PWD}/checkpoints/mdlm.ckpt"
export OUT_DIR="${PWD}/outputs/mdlm_grid"
BUDGETS="16 64 256 1024" TEMPERATURES="0.9 1.0" \
PRECISIONS="fp32 fp64" GPU_ID=0 \
bash code/scripts/research/eval_precision_temperature_grid.sh
```

The driver uses 128 length-1,024 outputs per cell, seed 1, and GPT-2 gPPL by
default. Override `PYTHON_BIN` for another interpreter and `CACHE_ROOT` for the
dataset cache. The command runs all four temperature and precision
combinations, which isolates each knob. The two named settings are:

| Setting | `sampling.x0_temperature` | `sampling.gumbel_precision` | `sampling.use_float64` |
| --- | --- | --- | --- |
| Standard | 1.0 | float64 | True |
| Sharpened | 0.9 | float32 | False |

Temperature is applied to the predicted clean-token distribution. Categorical
precision controls the random-number computation, not network weight precision.
The sampler is `ancestral_cache`, with the independent reveal rule and constant
temperature schedule. Configured updates are not measured network calls.

The paper's wider grid uses temperatures `0.80 0.85 0.90 0.95 1.00 1.05 1.10`
and steps `1 2 4 8 16 32 64 128 256 512 1024 2048`. For MDLM-Tiny, set
`MODEL_SIZE=tiny`, supply the corresponding checkpoint, and use
`EVAL_BATCH_SIZE=16 NUM_SAMPLE_BATCHES=16` for 256 outputs per cell. Sampling
writes generations to your output directory; those new files are not release
aggregates. Evaluate semantic quality and diversity with [GroupEval](../group-eval/README.md).

## Other paper experiments

| Purpose | Entry point and inputs |
| --- | --- |
| Cross-model generation | `scripts/research/eval_cross_model_gp_may05.py`: 40 outputs per point, GPT-2-Large. `--labels` selects individual configurations. Checkpoint filenames in `_default_specs` resolve under `${OUTPUT_ROOT}/dlm_asym/runs/`. Artifact label `duo_distilled` corresponds to DUO-DCD in the paper; the manifest records the mapping. |
| Sampling cost | `scripts/research/benchmark_sampling_time_memory_may06.py --codebase mdlm`: one warm-up and three measured batches of eight. Use `--labels` to select the paper's standard and sharpened public-checkpoint configurations. |
| Held-out posterior loss | `scripts/research/measure_schedule_matched_calibration.py`: accepts `--run-dir`, `--checkpoint`, `--label`, and `--output`; requires a training-run `.hydra/config.yaml`, cached tokenizer, and cached evaluation data. The script defaults to offline Hugging Face access. `summarize_schedule_matched_calibration.py` analyzes fresh measurements, including their per-sequence NPZ files. Only compact published summaries are shipped. |
| DiDi-Instruct grids | `scripts/didi/sweep_didi_full_grid.sh`: `MODE=manifest` prepares jobs without a GPU; `MODE=run GPU=0` runs them. See [DiDi setup](../didi-instruct/README.md). |
| Large diffusion-model stress test | `scripts/research/fast_dllm_temperature_sweep.sh` with `llada` or `dream`, and `gsm8k` or `humaneval`. See [Fast-dLLM setup](../fast-dllm/README.md). |
| Training and dataset caching | `scripts/research/train_mdlm_o2c_generic.sh` and `precache_dataset.py`. The retained general trainer exposes architecture, batch size, checkpoint frequency, and evaluation controls. The paper's trained checkpoints are early/late selections from a run, not newly trained models required for the public-checkpoint headline. |

GPU experiments are separate from the CPU quick start. See each script's
arguments and the paper's experimental setup (Appendix A.1) for exact settings.
Cross-model checkpoint loading needs the original compatible checkpoint
format, and a matching model name alone is not enough. The autoregressive
stress-test control is provided as its published aggregate only.

The implementation derives from [DUO](https://github.com/s-sahoo/duo) and
[MDLM](https://github.com/kuleshov-group/mdlm) and is distributed with
upstream attribution under the [Apache-2.0 license](LICENSE).
