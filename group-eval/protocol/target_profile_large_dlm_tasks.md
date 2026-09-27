# Conditional large-DLM task profile (`large-dlm-tasks-v1`)

This construct-validation profile covers two English conditional-generation
tasks. Each call contains one task condition and two blinded responses produced
by public diffusion language models at their greedy deployed operating point.

## GSM8K

The condition is one grade-school mathematical word problem. A successful
response should interpret the stated quantities, use compatible arithmetic,
and reach the requested answer. Concise correct solutions and longer correct
derivations are both acceptable. Do not reward verbosity or familiar benchmark
formatting. The reference answer is intentionally withheld from the judge.

## HumanEval

The condition is the public function signature and docstring for one Python
programming problem. A successful response should provide a complete candidate
program that appears to satisfy the stated behavior, edge cases, types, and
return contract. Candidates have undergone the deterministic complete-program
sanitization used by the paper's functional evaluator. Judge likely correctness
by static inspection only: no hidden tests, canonical solution, execution tools,
or pass/fail label are supplied.

## Scope and interpretation

Quality is scored separately for each response. The condition is context, not
model output, and must not contribute directly to the score. A wrong but
interpretable response remains valid and receives a low `task_success` score;
validity is reserved for catastrophic usability failures.

The four `0/1/2` dimensions are readability, reasoning coherence, apparent task
success, and non-degeneration/structure. Conditional GEQ is
`1.25 * sum(dimensions)` for a valid response and zero for an invalid response.
This task-specific score is not numerically interchangeable with unconditional
OpenWebText GEQ.

There is only one response per model and condition. This profile therefore does
not define semantic modes or GED. Diversity across different task prompts would
mostly reflect the task set and must not be reported as model diversity.
