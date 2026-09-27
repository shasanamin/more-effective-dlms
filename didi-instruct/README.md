# DiDi-Instruct evaluation component

The model and evaluation code used for the paper's DiDi-Instruct comparisons,
derived from the [DiDi-Instruct implementation](https://github.com/haoyangzheng-ai/didi-instruct)
under its [MIT license](LICENSE.md) and copyright. Only the model, sampler,
and evaluation code are included here. Training launchers and demos are not.

Use the compatible upstream Linux/CUDA environment in `environment.yml`, and
also install `hydra-core==1.3.2` and `omegaconf==2.3.0`, which that export
omits. The upstream
[checkpoint release](https://huggingface.co/haoyangzheng/didi-instruct-small)
and `models/hf_to_ckpt.py` provide the checkpoint and conversion path. The
converter's `--output_dir` argument expects a full filename such as
`/path/to/didi.ckpt`, despite its name, and does not accept a directory.

From the release root, prepare the full paper grid without inference:

```bash
OUT_DIR="${PWD}/build/didi_jobs" MODE=manifest GPUS="0" \
bash code/scripts/didi/sweep_didi_full_grid.sh
```

To generate samples, set `CKPT` to the local checkpoint, `PY` to the interpreter
in your DiDi environment, and use `MODE=run GPU=0` with the same `OUT_DIR` and
`GPUS`. This runs GPU inference. The grid includes ancestral and reward-guided
sampling, both precision settings, and seven temperatures. It uses the EMA
student, 40 outputs per cell, and GPT-2-Large by default. `MODE=combine` creates
summaries from a completed run. The shipped results are in
[the compact sampling grid](../paper_data/didi_sampling_grid.csv).

The direct MDLM/DiDi GroupEval comparison uses the dedicated matched-40 study
described in the [GroupEval results guide](../group-eval/results/README.md),
not the standalone DiDi scores.
