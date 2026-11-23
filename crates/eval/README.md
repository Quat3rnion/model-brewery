# Brewery evaluation worker

`brewery-eval` reads one UTF-8 JSON job from stdin. On success it writes
one JSON result to stdout. On failure it writes a diagnostic to stderr,
returns a nonzero exit code, and does not write a result.

## Job protocol

Inputs contain `kind` and `path` on the worker; upstream routing fields
such as `host` do not affect evaluation. Use `options: {}` when there
are no options.

```json
{"kind":"generate","inputs":[{"kind":"hgn","host":"frame","path":"/cache/bundle"}],"options":{"max_tokens":128},"prompt":"Explain fixed-point arithmetic."}
```

- `generate`: one qualified HGN bundle directory and one prompt.
  `max_tokens` is 1–512, default 128. The result contains `text`, a
  prompt and decoding `protocol`, and a load `identity`.
- `baseline`: one `gsm8k-baseline`, `gsm8k-predictions`, or
  `gsm8k-evaluation` file, **or** no inputs and `options.baseline_source`
  with an operator-configured ID.
- `evaluate`: one HGN bundle directory. It scores the candidate without a
  reference by default. Optionally add one `gsm8k-baseline`,
  `gsm8k-predictions`, or `gsm8k-evaluation` file, or set
  `options.baseline_source`, but not both.

Both GSM8K jobs use every row in dataset order. They do not accept prompt,
token-budget, or subset overrides. Results include `dataset`, `protocol_id`,
`score`, `identity`, and `artifact` (`kind`, absolute `path`, `sha256`, and
`bytes`). An `evaluate` result always scores the candidate and writes a
`gsm8k-evaluation` artifact. Its `baseline_identity`, `baseline_score`,
`comparison`, and `reference_scope` fields are null without a reference.
With a reference, the result includes its identity and score, a per-question
comparison, and its scope. Only an endpoint reference also writes
`baseline_artifact` for reuse; otherwise that field is null. `baseline`
writes a scored `gsm8k-baseline` artifact.

An imported `gsm8k-evaluation` artifact contains predictions, not a trusted
score. The evaluator checks and scores its predictions again. References must
match the pinned dataset, protocol, row IDs, and question digests. Their source
and precision do not need to match the candidate. Ordinary comparisons have
scope `diagnostic`; the scope is null without a reference. Precision from an
imported file is descriptive only. The evaluator does not copy an imported
BF16 attestation or endpoint identity claim.

`synthetic_fixture: true` is available only for direct test jobs. Synthetic
outputs are not release evidence.

## Pinned test set and scoring

[`openai/grade-school-math` test.jsonl](https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl)
is vendored verbatim under the MIT license in `data/`. Its revision is
`3101c7d5072418e28b9008a6636bde82a006892c`. Its SHA-256 is
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
The evaluator checks the full digest and the 1,319-row count before scoring.

The evaluator processes all 1,319 rows in dataset order. It rejects prompt,
token-limit, and subset overrides. It uses this prompt for each question:

```text
Solve the following math word problem. Show your reasoning and put the final numeric answer on the last line as #### <number>.

Question: <question>
Answer:
```

It generates one completion per question with temperature zero and a
512-token limit. Only the last nonblank line, beginning with `####`, is
scored. Normalization accepts signed decimal notation and valid thousands
grouping. Missing or invalid answers get zero credit. The rounded accuracy
in basis points is for display only.

With a reference, `comparison` reports `changed_answers`,
`changed_answer_ids`, `delta_correct`, and `within_two_percentage_points`.
Changed answer IDs are one-based. Two unparsed answers count as unchanged.
The flag is true when
`100 * (candidate_correct - reference_correct) >= -2 * total`. It is
diagnostic for ordinary references, not a release verdict.

Prediction artifacts use `brewery-gsm8k-predictions-v1` with
`source_sha256`, `dataset_sha256`, `protocol_id`, `synthetic`, `provenance`,
and `predictions: [{id, question_sha256, text}, ...]`. Each question
digest is SHA-256 of the decoded UTF-8 `question` field, not the JSONL line.
Real imports require object provenance from the operator. The evaluator
records the immutable file digest and does not copy arbitrary metadata. It may
keep a valid `precision` label as descriptive data. An import does not attest
how the model was served.

## Scratch behavior checks

These CPU checks are **not release-qualified**. Run them on private local
copies. Convert the parent and child SafeTensors files to GGUF with the same
converter, tokenizer files, and precision. Do not register a private scratch
file as a Library source.

Build the runner against a CPU build of llama.cpp that has
`llama_get_logits_ith` and `llama_memory_clear`. The test used base commit
`184f6536024668e8908ab8c86033d56708e62b40` with the W0 trace changes.
Run these commands from the Brewery repository root. Set `LLAMA` to the
llama.cpp checkout and `BIN` to its CPU build directory that has `libllama`.
Set `WORK` to a private local directory. Use Python 3.11 or later.

```sh
c++ -std=c++17 -O2 -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  tools/scratch_check_logits.cpp -L"$BIN" -Wl,-rpath,"$BIN" -lllama \
  -o "$WORK/scratch-check-logits"
python3 tools/scratch_checks.py --runner "$WORK/scratch-check-logits" \
  --cache "$WORK/check-data" --fetch \
  --parent-gguf "$WORK/parent.gguf" --child-gguf "$WORK/child.gguf" \
  --parent-safetensors "$WORK/parent.safetensors" \
  --child-safetensors "$WORK/child.safetensors" > "$WORK/behavior.json"
```

For worker jobs, set `BREWERY_SCRATCH_CHECKS_RUNNER_LIB_DIR` in `engine_env`
to the directory with the [four pinned CPU libraries](../../OPERATOR.md#cpu-trace-jobs).
The engine checks private copies of the libraries before it starts the
runner. It does not load them from the build directory.

`--fetch` gets missing files only. Omit it to use only the local cache. The
runner checks the full source hashes before it reads rows. It then checks
the hash of each selected subset and of the held-out corpus.

- [`sources.json`](data/edit-checks/sources.json) pins the source URLs,
  loader commits, full file SHA-256 hashes, and 50 item IDs per dataset.
  CounterFact uses the case IDs from ROME. zsRE uses zero-based row IDs from
  the MEND QA validation adaptation distributed with MEMIT.
- [`manifest.json`](data/edit-checks/manifest.json) pins the SHA-256 of the
  normalized JSON Lines subsets and the held-out text. The held-out text
  contains 16 original sentences supplied by this project. The ROME, MEMIT,
  and MEND code repositories use MIT licenses. The dataset downloads do not
  state a separate data license. This repository does not redistribute the
  downloaded datasets or their subsets.

For each row, the runner reads the probability and rank of each expected
answer token, given the prompt and the previous answer tokens.
It checks that both models use the same token IDs
for the scored text. Efficacy and paraphrase scores are the fraction of
answers whose tokens are all rank one. Rank is one plus the count of logits
above the answer logit. Probability and rank means count each answer token
once. CounterFact uses two paraphrases and two neighborhood prompts per
case; zsRE uses one of each. Locality compares the same control answer before
and after the edit. Perplexity is the exponential of the mean negative log
probability of the held-out suffix tokens. The first word of each sentence
is its prompt. The report includes samples from an eight-token greedy smoke
test on both files.

Send `behavior.json` as `offline_report` in a scratch check request. Results
from this tool are diagnostic. Use the paired scores to review the effect of
the edit. A change in score does not qualify a release. The controller does
not verify the ancestry of the local GGUF files.

## Operator configuration

For HGN inputs use `kind` `hgn`, `brewery-bundle`, `hgn-v2`, or `ht43`.
`manifest.json` in the directory must contain `format`, `source_sha256`
(from the original source bundle), `files` as in `halogen-proof`, and
`tokenizer: [{path, sha256}, ...]` for the exact flat contents of either
`tokenizer/` or `native/tokenizer/`. `tokenizer.json` is required. Each
asset is independently hashed.
The load receipt records the manifest digest, source digest, tokenizer
digests, pinned runtime image ID, and engine mapping evidence.

After admitting the job under the GPU gate, the worker operator can set
`BREWERY_ALLOW_GPU_LOAD=1` and `BREWERY_HALOGEN_IMAGE_ID` to the full ID of
the installed trusted Halogen Docker image. The normal Halogen `all` mode
runs the engine and API in a private Docker network with no published port.
Requests use the isolated frontend only. The engine checkpoint and declared
sidecars are checked before and after inference. The job renews a
read-only lease mounted into its container. If the job is killed, the
in-container watchdog expires the lease after 20 seconds and gives
Halogen up to 30 seconds to exit before it sends `SIGKILL`. An abrupt
cancel can leave a stopped container on disk. Its ID is in stderr.
Do not launch on a busy host; see `halogen-proof` memory requirements.

For live reference endpoints, set `BREWERY_BASELINE_CONFIG` to a private JSON
file on the worker. The endpoint's `source_sha256` identifies its own model,
not the candidate. Example entry:

```json
{
  "int8-reference": {
    "url": "https://reference.example/v1/completions",
    "model": "model-name-as-the-endpoint-reports-it",
    "source_sha256": "<64 lowercase hex characters>",
    "runtime_id": "sha256:<pinned runtime identity>",
    "precision": "int8",
    "bearer_env": "BREWERY_BASELINE_TOKEN_REFERENCE"
  }
}
```

The URL path must be exactly `/v1/completions`. The endpoint must return
`X-Brewery-Source-Sha256` and
`X-Brewery-Runtime-Id` that match its configuration on every response.
The response `model` must also match. These values identify the reference
endpoint, not the candidate. The evaluator checks them on all 1,319 rows.
Use `https`, except for `http` on `127.0.0.1` or `[::1]`. The client follows
no redirects and uses no proxy. Job data cannot set a URL. URLs and
authorization values are not recorded.

Each request asks for one non-streaming raw completion. It sends the
configured `model`, `temperature: 0`, `n: 1`, and `max_tokens: 512`. The
response must contain one choice and a `finish_reason` of `stop` or `length`.
Settings can set optional `precision`, but cannot set `bf16_attested`.
Imported precision labels and endpoint headers do not independently prove
model precision or source.

For the separate release gate, only a trusted direct endpoint configuration
can set `bf16_attested: true`, with `precision: "bf16"` and a non-synthetic
endpoint. The evaluator marks `attested_bf16_same_source` only when that
reference has the candidate's source hash. This flag is an operator assertion.
Release qualification still requires the exact BF16 source, the pinned full
protocol, and independent review of endpoint evidence. Ordinary
precision-neutral comparisons do not satisfy this gate.

The loopback endpoint fixture sets `"synthetic": true` in its config and
uses `options.synthetic_fixture: true`.

The CPU tests exercise all 1,319 rows with synthetic predictions and an
operator-configured loopback test double. No production BF16 endpoint or
full HGN inference was run during implementation.
