# Large diffusion-model evaluation component

The evaluation code from [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM)
used for the paper's LLaDA-8B-Instruct and Dream-7B temperature stress tests,
distributed with its [Apache-2.0 license](LICENSE) and upstream code notices.
Chat and web demos are not included. Model weights and benchmark data are not
redistributed.

Create a separate Linux/CUDA environment with PyTorch for your GPU, then
install `requirements.txt`. These dependencies differ from the MDLM
environment.

From the release root, set `FAST_DLLM_ENV` to the environment directory:

```bash
export FAST_DLLM_ENV="/path/to/fast-dllm-environment"
CUDA_VISIBLE_DEVICES=0 \
bash code/scripts/research/fast_dllm_temperature_sweep.sh llada gsm8k
```

Replace `llada` with `dream` and `gsm8k` with `humaneval` for the other paper
settings. The baseline decoding sweep uses 256 GSM8K conditions, all 164
HumanEval tasks, greedy decoding once, and three stochastic seeds at positive
temperatures. HumanEval evaluates generated code; use an isolated evaluation
environment. `MODEL_PATH` can select a compatible local checkpoint.

The paper's sampling controls are in `llada/generate.py` and
`dream/model/generation_utils.py`. Cache-based and confidence-based variants
remain in the shared upstream implementation, but the temperature stress test
uses baseline decoding. Aggregate results are in
[paper_data/large_model](../paper_data/large_model/).

`llada/sanitize.py` and
`../code/scripts/research/rescore_fast_dllm_humaneval.py` are also used by the
conditional GroupEval preparation path.
