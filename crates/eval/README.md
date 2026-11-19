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
- `baseline`: one `gsm8k-baseline` or `gsm8k-predictions` file, **or**
  no inputs and `options.baseline_source` with an operator-configured ID.
- `evaluate`: one HGN bundle directory, optionally followed by one
  `gsm8k-baseline` file. `options.baseline_source` can replace that second
  input to run a live BF16 baseline. Without a baseline, the result has
  no comparison or quality verdict.

Both GSM8K jobs use every row in dataset order. They do not accept prompt,
token-budget, or subset overrides. Results include `dataset`, `protocol_id`,
`score`, `identity`, and `artifact` (`kind`, absolute `path`, `sha256`, and
`bytes`). Evaluation adds `baseline_identity`, `baseline_score`, and
`comparison` when matched. An inline endpoint baseline also returns
`baseline_artifact` for reuse. `synthetic_fixture: true` is available only
for direct test jobs. Synthetic outputs are not release evidence.

## Pinned test set and scoring

[`openai/grade-school-math` test.jsonl](https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl)
is vendored verbatim under the MIT license in `data/`. Its revision is
`3101c7d5072418e28b9008a6636bde82a006892c`. Its SHA-256 is
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
The evaluator checks the full digest and the 1,319-row count before scoring.

Protocol `gsm8k-zero-shot-raw-v1` sends one non-streaming raw completion per
question to `/v1/completions`, at temperature zero with `n: 1` and
`max_tokens: 512`. The exact prompt is in `src/gsm8k.rs`. Only the last
nonblank line, beginning with `####`, is scored. Normalization accepts
signed decimal notation and valid thousands grouping. Missing or invalid
answers get zero credit. The candidate passes the two-percentage-point
gate if `100 * (candidate_correct - baseline_correct) >= -2 * total`.
The rounded accuracy in basis points is display only. Changed answers
are listed by one-based row ID; two unparsed answers count as unchanged.

Prediction artifacts use `brewery-gsm8k-predictions-v1` with
`source_sha256`, `dataset_sha256`, `protocol_id`, `synthetic`, `provenance`,
and `predictions: [{id, question_sha256, text}, ...]`. Each question
digest is SHA-256 of the decoded UTF-8 `question` field, not the JSONL line.
Real imports require object provenance from the operator; import identity
records the immutable file digest and does not re-emit arbitrary metadata.
An import is not an independent attestation of how the model was served.

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

For live BF16 baselines set `BREWERY_BASELINE_CONFIG` to a private JSON
file on the worker. Example entry:

```json
{
  "bf16-source": {
    "url": "https://baseline.example/v1/completions",
    "model": "bf16-checkpoint",
    "source_sha256": "<64 lowercase hex characters>",
    "runtime_id": "sha256:<pinned runtime identity>",
    "bearer_env": "BREWERY_BASELINE_TOKEN_BF16"
  }
}
```

The endpoint must serve the exact BF16 source under that model name and
return `X-Brewery-Source-Sha256` and `X-Brewery-Runtime-Id` on every
response. These are operator assertions checked against the config, not
attestation from an unaffiliated service. HTTPS is required except for
loopback test fixtures. The client follows no redirects, uses no proxy,
and rejects a source mismatch before using the baseline for comparison.
Job data cannot set a URL. URLs and authorization values are not recorded.
The loopback endpoint fixture sets `"synthetic": true` in its config and
uses `options.synthetic_fixture: true`.

The CPU tests exercise all 1,319 rows with synthetic predictions and an
operator-configured loopback test double. No production BF16 endpoint or
full HGN inference was run during implementation.
