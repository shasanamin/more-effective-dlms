# Paper data

These compact aggregates hold the measurements reported in the paper. Use them
to plot, inspect, or reanalyze the results without model weights, generated
text, or judge calls. Values are stored at their original precision.

| File | Contents and use |
| --- | --- |
| `mdlm_sampling_grid.csv` | 577 checkpoint, budget, temperature, and precision cells. The main four-corner comparison, early/late and Small/Tiny comparisons, full MDLM appendix grids, and public-checkpoint temperature frontier. |
| `didi_sampling_grid.csv` | 308 cells for the full ancestral and reward-guided DiDi-Instruct appendix grids. |
| `cross_model_gp_may05.csv` | Consolidated 40-sample cross-model gPPL and entropy results, used in the overview and cross-model entropy plot. |
| `cross_model_manifest.csv` | Checkpoint hashes, evaluator revision, generation seed, and mappings between archived labels and paper names for the cross-model comparison. |
| `large_model/diffusion_temperature.csv` | Temperature-sweep means and seed standard errors for LLaDA and Dream on GSM8K and HumanEval. |
| `large_model/autoregressive_temperature.csv` | Corresponding autoregressive control aggregates for four models. |

The MDLM grid has one row per cell, and the DiDi grid reports both metrics for
each full-grid cell. Large-model aggregates give the reported means, seed
standard errors, and run counts. Their HumanEval scores use the
complete-program extraction described in the paper.

## Column conventions

- `ckpt` is `public`, `early`, or `late` for MDLM-Small, and `tiny` for the late
  MDLM-Tiny checkpoint. `public` names the released MDLM-Small checkpoint.
- `N` (or `steps` in the cross-model file) is the paper's configured reverse
  update budget `M`, not measured network-function evaluations.
- `tau` is the explicit sampling temperature. `prec` is categorical-kernel
  precision, not the precision of the neural network.
- `gPPL` and `gen_ppl` are generative perplexity under the named evaluator.
  `entropy` is mean within-sequence unigram entropy in nats.
- `total_samples` records each cell's sample count. MDLM-Small grids use 128,
  MDLM-Tiny uses 256, and the separate cross-model/DiDi grids use 40. Do not
  pool these populations or treat their evaluator scales as interchangeable.
- Large-model `accuracy_pct` and `seed_sem_pct` are percentages. Positive
  temperatures use three seeds, while greedy decoding has one run and zero
  plotted seed SEM. Zero seed SEM for greedy is not a confidence interval.

The standard operating point is temperature 1 with fp64 categorical draws.
The sharpened point is temperature 0.9 with fp32 draws. The cross-model files
use archived labels such as `MDLM calibrated`, which the archived-output
preparation tool checks by hash. These labels mean the sharpened sampler, not
improved posterior calibration. The `source` column holds anonymized
provenance strings that plotting does not need to resolve. `Autoregressive` is
a single update-independent GPT-2 reference repeated across budgets for
plotting.

## Rebuild the figures

From the release root:

```bash
python analysis/make_iclr_figures.py
```

All nine paper figure files are written to `build/figures/`. Each figure reads
the inputs below:

| Figure filename | Inputs |
| --- | --- |
| `overview.pdf` | Cross-model gPPL, MDLM sampling grid, the primary MDLM GroupEval aggregate in `group-eval/results/`, and posterior-loss temperature contrasts in `results/calibration_certificate/summary/`. |
| `less_is_more.pdf` | MDLM sampling grid. |
| `theory.pdf` | The closed-form binary example and `theory/generic_suboptimality_results.json`. |
| `tiny_calibration_vs_steps.pdf` | MDLM sampling grid, `ckpt=tiny`. |
| `tau_frontier_sweep.pdf` | MDLM sampling grid, `ckpt=public`. |
| `sampling_cost_combined.pdf` | `results/sampling_cost/summary_mdlm.tsv`. |
| `cross_model_entropy_vs_steps.pdf` | Consolidated cross-model results. |
| `large_model_temperature_stress.pdf` | Diffusion temperature aggregate. |
| `autoregressive_temperature_stress.pdf` | Autoregressive temperature aggregate. |

The plotting script also accepts `--output-dir /path/to/figures` or
`--paper-dir /path/to/latex-source` (which writes to its `figs/` directory).
Neither a LaTeX installation nor the paper source is required for the default
build. The shipped `assets/teaser.png` previews the overview figure.
