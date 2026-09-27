# Supporting measurements

Compact numerical results for the posterior-loss and sampling-cost
experiments. They reproduce the displayed values and plots. Recalculating
bootstrap intervals from per-sequence observations, or repeating the GPU
measurements, needs fresh experiment outputs.

## Schedule-matched posterior loss

`calibration_certificate/summary/` contains:

- `temperature_certificate.csv`: the 15 paired contrasts between temperature
  0.9 and 1, with 95% intervals, across three MDLM-Small checkpoints and five
  update budgets.
- `checkpoint_certificate.csv`: five late-minus-early checkpoint contrasts at
  temperature 1, with paired 95% intervals.
- `checkpoint_profile.csv`: the late-minus-early contrast over 32 clean-token
  probabilities.
- `summary.json`: the integrated profile contrast and its paired interval.

The measurements use EMA weights, 512 paired held-out sequences for each
configured-update contrast, and 128 for the separate 32-level profile. For
`M` configured updates, the deployed schedule has `M` masses of
`(1 - 1e-5) / M` and terminal cleanup mass `1e-5`. Temperatures share the
corruption for each sequence. Reported summary intervals use 10,000 paired
sequence-bootstrap replicates.

Cross-entropy differences are per-position nats. Multiplying by sequence length
1,024 gives the summed scale used in the theory. Positive temperature contrasts
mean sharpening worsens the posterior loss. Negative checkpoint contrasts mean
the later checkpoint predicts better. The directory name
`calibration_certificate` refers to this posterior-loss measurement. It does
not claim that sharpening improves calibration.

## Sampling cost

`sampling_cost/summary_mdlm.tsv` contains 22 matching-configuration measurements
of the standard and sharpened public MDLM-Small samplers across
`M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024`.

The protocol uses one NVIDIA H100 80 GB GPU, sequence length 1,024, batch size
8, one unmeasured warm-up batch, and three measured batches (24 sequences).
Timing covers generation including terminal cleanup, with no external
perplexity evaluator. Seconds per generation are amortized over the batch.
Peak allocated memory includes resident model and sampling allocations.
Configured updates are not measured network-call counts.

At `M=1024`, sharpened sampling uses 8,263 MiB (8.07 GiB) peak allocation and
4.065 wall-seconds per sequence, against 13,762 MiB (13.44 GiB) and 4.764
seconds for the standard sampler. That is 39.96% lower peak allocation and
14.67% less time, or equivalently 17.19% higher throughput. The TSV has the
unrounded values and both wall and CUDA time.
