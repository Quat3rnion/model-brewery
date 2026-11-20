# Model Brewery

![Model Brewery banner with a tasting flight from BF16 to Q4_K_M](branding/model-brewery-banner.png)

Model Brewery is a Linux workbench for inspecting, editing, converting and evaluating model checkpoints. A Rust controller serves the web interface and keeps a shared model library and durable job queue. Workers can run locally or on explicitly configured SSH hosts.

**Start here:** [Install Model Brewery](INSTALL.md). The guide sets up a loopback user service, installs the three agent skills, and tells the installing agent to ask before adding extra worker hosts. You can start the controller without a GPU or a worker.

## What you can do

- Download a pinned checkpoint and inspect its tensor inventory.
- Edit individual weights in supported formats, review the changes, and publish a separate revision without replacing the source.
- Convert supported models to HGN or GGUF through inspect, plan, convert, and independent verification. Available conversions depend on the source architecture and worker runtime.
- Trace models, run benchmarks, and score GSM8K with or without a diagnostic reference.

![The Convert page showing the tokenizer, vision tower, n-gram lookup and MTP head supplied by a Flash-Next checkpoint](docs/images/checkpoint-components.png)

## Reference

- [Operator guide](OPERATOR.md): worker setup, supported formats, job rules, Settings and troubleshooting.
- [API contract](api/openapi.json) and [command-line client](tools/brewery-agent).
- [Project skills](.opencode/skills/): workbench operation, calibration and benchmarking, and release qualification.
