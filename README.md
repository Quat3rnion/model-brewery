# Model Brewery

![Model Brewery banner with a tasting flight from BF16 to Q4_K_M](branding/model-brewery-banner.png)

Model Brewery is a Linux workbench for inspecting, editing, converting and evaluating model checkpoints. A Rust controller serves the web interface and keeps a shared model library and durable job queue. Workers can run locally or on explicitly configured SSH hosts.

Model Brewery is a work in progress. [Issues](https://github.com/Quat3rnion/model-brewery/issues) and [pull requests](https://github.com/Quat3rnion/model-brewery/pulls) are welcome.

**Start here:** [Install Model Brewery](INSTALL.md). The guide sets up a loopback user service, installs the agent skills, and tells the installing agent to ask before adding extra worker hosts. You can start the controller without a GPU or a worker.

## What you can do

- Download a pinned checkpoint and inspect its tensor inventory.
- Edit individual weights in supported formats, review the changes, and publish a separate revision without replacing the source.
- Convert supported models to HGN or GGUF through inspect, plan, convert, and independent verification. Available conversions depend on the source architecture and worker runtime.
- Trace models, run benchmarks, and score GSM8K with or without a diagnostic reference.
- Program model weights with AI agents through an MCP server, or with the built-in workbench agent. See [Weight-space programming](#weight-space-programming).

## Weight-space programming

Agents and operators can read, change and test model weights. All changes go to scratch revisions. A scratch revision is a private child copy of a SafeTensors model with a reversible delta and a provenance record. The source model does not change. Only an operator can approve a scratch revision for the Library, in the workbench. Results are not release-qualified.

- **Trace.** Capture activations, model logits and a logit lens per layer for GGUF or Qwen3.8 Flash Next v2 HGN. GGUF also supports attention capture and private scratch revisions. The worker uses a pinned CPU runtime. For GGUF, the worker runs Lanternfish as a headless command-line program that is pinned by SHA-256. It does not start a Lanternfish server or use the Lanternfish UI. See [CPU trace jobs](OPERATOR.md#cpu-trace-jobs).
- **MoE routing atlas.** Read expert load, entropy, co-activation and token clusters for Qwen2MoE and Qwen3MoE. Compare paired expert ablations without changing weights. See [MoE routing atlas](OPERATOR.md#moe-routing-atlas).
- **Measure behavior directions.** Fit per-layer directions for slop, sycophancy, and response length on a Qwen2 GGUF. Read held-out separation scores and versioned vectors without changing weights. See [Behavior directions](OPERATOR.md#behavior-directions).
- **Ablate a direction.** Edit residual-writing weights in a private scratch child with a reversible delta and a layer kernel. See [Native directional ablation](OPERATOR.md#native-directional-ablation).
- **Search ablation settings.** Screen a fitted direction against a matched random control before making scratch children. See [Native ablation search](OPERATOR.md#native-ablation-search).
- **Run pinned Heretic offline.** Use the local GPU queue to make a private SafeTensors scratch child or reproduce a verified community manifest. See [Heretic jobs](OPERATOR.md#heretic-jobs).
- **Review abliteration results.** Inspect direction pins, native screens with matched controls, paired child checks, and Heretic trial journals. See [Abliteration workflow through MCP](OPERATOR.md#abliteration-workflow-through-mcp).
- **Steer and patch.** Add, replace, scale or remove activations during a trace, or copy activations from an earlier trace. Save versioned Library interventions that project out or subtract a verified direction. Export fixed steering vectors, not projections, as llama.cpp control vectors. See [Trace interventions](OPERATOR.md#trace-interventions).
- **Edit weights.** Set or add values in a tensor slice, apply a rank-one update, or orthogonalize a weight against a direction. See [Scratch weight revisions](OPERATOR.md#scratch-weight-revisions).
- **Edit facts.** Use ROME, MEMIT or AlphaEdit on dense Llama and Qwen2 models, on the CPU or with a pinned CUDA or ROCm runtime. See [Locate and edit a fact](OPERATOR.md#locate-and-edit-a-fact).
- **Merge models.** Use linear averages, task arithmetic, TIES, DARE-linear or DARE-TIES. See [Merge revisions](OPERATOR.md#merge-revisions).
- **Prune.** Set weights to zero by magnitude, by MLP neuron, by attention head, or with a Wanda score. Plan REAP expert pruning for MoE models. Applying an expert plan to a scratch revision is not available yet.
- **Measure spectral health.** Compare singular values, ranks, and power-law fits for scratch revisions and supported Library checkpoints. See [Spectral weight health](OPERATOR.md#spectral-weight-health).
- **Check model lineage.** Compare Library weights to find a likely base and see evidence for an edit, fine-tune, pruning, or merge. See [Model lineage](OPERATOR.md#model-lineage).
- **Use SAE features.** Train a TopK sparse autoencoder on the CPU or import a public one, read feature activations in a trace, and steer one feature. See [Sparse autoencoders](OPERATOR.md#sparse-autoencoders-saes).
- **Measure tensor sensitivity.** Use empirical Fisher on the pinned W4 held-out corpus. Read tensor and layer scores in the Jobs heatmap or with MCP. See [Tensor sensitivity](OPERATOR.md#tensor-sensitivity).
- **Find activation circuits.** Rank activation sites on a clean and corrupt prompt pair, then measure the top sites with independent W2 patches. Read the Jobs heatmap or use `find_circuit`. See [Activation attribution](OPERATOR.md#activation-attribution).
- **Compare features across revisions.** Train a crosscoder on paired traces and inspect decoder changes and held-out reconstruction for each revision. See [Crosscoder model diff](OPERATOR.md#crosscoder-model-diff).
- **Map quantization damage.** Rank tensor and layer effects on target logits or mean corpus loss. Check logit estimates with real swaps, then review the recipe proposal. See [Quantization damage map](OPERATOR.md#quantization-damage-map).
- **Compile programs.** Compile RASP programs with Tracr or ALTA into small standalone models, and check their outputs. See [Compiled programs](OPERATOR.md#compiled-programs).
- **Measure interpretability on known circuits.** Read benchmark scores for activation attribution, SAE features, and paired crosscoder features. See [Compiled-program benchmark](OPERATOR.md#compiled-program-benchmark).
- **Check and compare.** Before an agent can request publication, the worker checks tensor integrity, loads and runs the model, and measures target, paraphrase, locality and perplexity scores. Optional [behavior direction checks](OPERATOR.md#behavior-direction-checks) measure held-out behavior, including over-refusal on reviewed benign prompts, and KL divergence from the parent. Compare two revisions by tensors, by traces or by behavior. See [Checks before a publish request](OPERATOR.md#checks-before-a-publish-request) and [Compare revisions and traces](OPERATOR.md#compare-revisions-and-traces).

**MCP for agents.** The controller serves an MCP endpoint (Streamable HTTP) at `/mcp` on the workbench address. It gives agents tools to list and register models, trace, steer, edit, merge, prune, check, compare and request publication. MCP has no approve tool. MCP uses the same access checks as the workbench and has no separate token. See [MCP for weight-space work](OPERATOR.md#mcp-for-weight-space-work) for the client configuration and the access risk.

**Built-in agent.** The **Agent** page runs an agent in the workbench with the same tools as MCP. Set an Anthropic or OpenAI-compatible provider and its API key in Settings. The key is write-only. See [Hosted workbench agent](OPERATOR.md#hosted-workbench-agent).

**Workbench views.** **Trace** shows layer-by-token heatmaps, the logit lens and attention. **Edits** shows the scratch revisions of each source model, their checks and diffs, and the approval action for the operator. **Abliteration** shows direction fits, native ablation searches and Heretic runs. **Findings** shows hypotheses with their evidence. **Agent** runs the built-in agent. See [Inside the workbench](#inside-the-workbench) for screenshots.

- **Token loss diff.** See which heldout sentences and tokens account for a perplexity change. Read the ranked results in Edits or with `read_token_loss_diff`. See [Token loss diff](OPERATOR.md#token-loss-diff).
- **Findings.** Record a hypothesis, link measured evidence, and replay the same experiments on another pinned revision. See [Findings](OPERATOR.md#findings).

Stage acceptance results on Qwen2.5-0.5B-Instruct: [Stage 1](qualification/weight_space_acceptance.md), [ROME, MEMIT and AlphaEdit](qualification/weight_space_stage2.md), [merges](qualification/weight_space_merge_acceptance.md), [SAE features](qualification/weight_space_sae.md), [crosscoder diffs](qualification/weight_space_crosscoder.md), and the [CUDA edit runtime](qualification/lanternfish-w14b-cuda.md).

## Inside the workbench

**Edit a weight.** Select a tensor, then choose a cell in its heatmap to inspect the stored value and propose a change.

![Weight editor with a tensor heatmap, a selected cell and controls for changing its value](docs/images/weight-editor.png)

**Review before applying.** In this example, the original weight is `0.125`. The requested `0.1` rounds to `0.0999755859375` in F16. Applying the proposal creates a separate revision and keeps the original checkpoint.

![Pending weight edit showing the requested and representable values before the change is applied](docs/images/weight-review.png)

**Set a quantization profile.** In this example, layers 2 through 7 use HT43 for eligible tensors; the other layers keep the base profile.

![Quantization profile with six of 48 layers selected and HT43 chosen for all roles](docs/images/quantization-profile.png)

For ROCmFPX formats and worker requirements, see [ROCmFPX GGUF qualification](qualification/rocmfpx.md).

**Choose checkpoint components.** Keep the embedded components or select replacements from the model library.

![The Convert page showing the tokenizer, vision tower, n-gram lookup and MTP head supplied by a Flash-Next checkpoint](docs/images/checkpoint-components.png)

The screenshots below show weight-space work. They use synthetic data from the test controller in `web/tests/flows`, not measured results. Interpretability and abliteration results in the workbench are not release-qualified.

### Trace a prompt

**Activations and logit lens.** The heatmap shows the activation norm for each layer and token. The logit lens shows the top token ID for each layer and position. Select a cell to read its exact value or its top tokens.

![Trace results with a layer-by-token norm heatmap, a logit lens table and the model logits for each position](docs/images/trace-activations.png)

**Attention.** Select a layer and a head to see the attention probabilities of each query position.

![Attention pattern for layer 12, head 0, with most of the weight on the first token](docs/images/trace-attention.png)

**SAE features.** If the trace used a sparse autoencoder, select a token position to read its top features.

![Top SAE features at token position 4 with their activations](docs/images/sae-features.png)

**MoE routing atlas.** For a mixture-of-experts model, read the expert load, router entropy, token clusters and co-activation for one layer. An optional paired ablation trace shows the effect of one expert.

![MoE routing atlas for layer 12 with expert 17 selected and the results of its ablation](docs/images/routing-atlas.png)

**Steer and patch.** Add a saved intervention and activation patches from an earlier trace before you queue a trace.

![New trace form with a saved steering intervention and an activation patch for layer 12](docs/images/interventions.png)

### Review scratch revisions

**Edit sessions.** The lineage tree shows each scratch revision, its method and its checks. The detail shows provenance, the delta, the checks, the publish request and the diffs against the parent.

![Edits page with a ROME revision that waits for publish approval and its behavior diff against the parent](docs/images/edits.png)

**Token loss diff.** See which held-out sentences and tokens cause a perplexity change.

![Token loss diff with the sentence that gives the largest gain and the top token contributors](docs/images/token-loss-diff.png)

**Weight health.** Compare the effective rank, stable rank and alpha of each layer with the parent. The chart marks layers that changed by 1% or more.

![Weight health charts and table with a change at layer 6, and the singular values of the edited tensor](docs/images/weight-health.png)

**Crosscoder diff.** Read the reconstruction metrics and the features that are shared, changed, or unique to the parent or the child.

![Crosscoder report with reconstruction metrics, feature counts per class and a scatter of child share against decoder cosine](docs/images/crosscoder-diff.png)

**Merges and compiled programs.** A merge revision shows its method, for example TIES. A compiled program is a root revision without a parent. Its checks include the exact program outputs.

![Edits page with a TIES merge revision selected in the lineage tree](docs/images/merge-revision.png)

![Edits page with a compiled sort program, its configuration hash and its program checks](docs/images/compiled-program.png)

**Model lineage.** In the Library, compare a model with candidate bases to find a likely base and the type of change.

![Model lineage panel that finds a rank-one edit of Qwen2.5-0.5B-Instruct and shows the evidence for each candidate](docs/images/model-lineage.png)

### Measure model internals

**Tensor sensitivity.** The job result shows the empirical Fisher score for each layer and tensor.

![Empirical Fisher scores in a table with one row for each layer and one column for each tensor type](docs/images/sensitivity.png)

**Activation circuits.** The job result shows the attribution for each layer and position, and the measured activation patches for the top sites.

![Attribution heatmap by layer and position for the attention output, and a table of measured patches with their correlation](docs/images/circuit.png)

**Quantization damage map.** Agents read the ranked tensors and the recipe proposal with MCP. The workbench has no separate view for this result.

![Agent transcript that queues a quantization damage map and reads the two tensors with the largest effect](docs/images/quantization-damage.png)

### Abliteration

**Behavior direction.** The fit shows the held-out separation, the margin and the norms for each layer, and the selected layer.

![Sycophancy direction with charts of held-out Cohen's d, margin and norms for each layer](docs/images/abliteration-direction.png)

**Native ablation search.** The screen compares each setting with a matched random control and marks the Pareto front. Paired checks on the scratch child confirm or reject the effect.

![Native search with a plot of marker reduction against KL, the screen trials and the paired child checks](docs/images/abliteration-search.png)

**Heretic.** A Heretic run shows the pinned invocation, the selected trial, the trial journal and a link to the imported scratch child.

![Heretic run with its profile, seed, hashes and trial scores](docs/images/heretic.png)

**Behavior-direction checks.** In Edits, the check bundle of the child shows held-out behavior, first-token KL and benign over-refusal against the parent.

![Behavior-direction check table with the agreement rate, first-token KL and benign refusal rate for the parent and the child](docs/images/behavior-checks.png)

### Agents and findings

**Built-in agent.** The transcript shows each tool call and result. The agent can request publication, but only an operator can approve it.

![Agent session that finds a circuit, applies a ROME edit, runs the checks and requests publication](docs/images/agent.png)

**Findings.** A finding records a hypothesis, its experiments, a verdict and the provenance. A replay on another revision compares each value with the original.

![Finding that replays an earlier finding on an approved revision and shows three changed values](docs/images/findings.png)

To make all README screenshots again, run `node scripts/readme-shots.mjs` in `web/`. First install the web dependencies with `npm ci` and the Playwright Chromium browser with `npx playwright install chromium`. The script starts the test controller on a loopback port and writes the images to `docs/images`. It does not connect to a real controller.

## Reference

- [Operator guide](OPERATOR.md): worker setup, supported formats, job rules, Settings and troubleshooting.
- [Design guide](DESIGN.md): current web interface tokens, layout, interaction and accessibility patterns for contributors.
- [API contract](api/openapi.json), [command-line client](tools/brewery-agent) and [MCP endpoint](OPERATOR.md#mcp-for-weight-space-work).
- [Project skills](.opencode/skills/): workbench operation, weight-space work through MCP ([`brewery-weight-space`](.opencode/skills/brewery-weight-space/SKILL.md)), calibration and benchmarking, and release qualification.
