#!/usr/bin/env python3
"""Run the Qwen Stage 1 workflow through MCP on an isolated CPU workbench."""

import argparse
import asyncio
import base64
import hashlib
import importlib.metadata
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import time


ARB = Path("/arboretum")
REPO = Path(__file__).resolve().parents[1]
HF = ARB / "models/qwen2.5-0.5b-instruct-w4-hf"
SAFE = HF / "model.safetensors"
BASE = ARB / "models/qwen2.5-0.5b-instruct-w0/Qwen2.5-0.5B-Instruct.bf16.gguf"
LANTERN = ARB / "pins/lanternfish-d842085/lanternfish"
TRACE_LIBS = ARB / "llama-a4/build/bin"
W0_LIBS = ARB / "llama-w0/build/bin"
LLAMA = ARB / "llama.cpp"
QUANT = ARB / "brewery-standalone-admit-integration.s4xL9Wu0/stage/bin/llama-quantize"
PYTHON = ARB / "brewery-flash-gguf.tXbsagtV/standalone/runtime/bin/python3"
HELPER = ARB / "weight-runtime-smoke.09h393rr/release-a/brewery-weight-edit"
FIXTURES = ARB / "tmp/w4-data/new-cache"
RUNNER = FIXTURES.parent / "scratch-check-logits-v1"
ASSETS = (
    "config.json", "generation_config.json", "merges.txt", "tokenizer.json",
    "tokenizer_config.json", "vocab.json",
)
SNAPSHOT_PIN = "e702c869285f94912fd8f9bcd717750649f6876070dbaadfbe4e22181852696f"
PINS = {
    SAFE: "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    BASE: "a33250ab73bbef0d876bf04e01517f5f43292dd8ede34e12d3a3a886e49f36f0",
    LANTERN: "3009f4b0c362ddc04d955e40efac9e1d88659b9a0bc393fdc8ea6da287bde010",
    QUANT: "7ceb0e39e4f7e207fe8c55215368163bb5734072dc871a66b45527e1a5c9c672",
    RUNNER: "505759d6749dc5c593e74749689edbfe3d8ff0d833c91db8c11a051781e631a3",
    FIXTURES / "counterfact.json": "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f",
    FIXTURES / "zsre.json": "8a371d512f8a6ab175db4ef672181d5c0e52da23298cb51e5a2eeea2f89aa9cd",
    REPO / "crates/eval/data/gsm8k-test.jsonl": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
}
TARGET = {
    "dataset": "CounterFact", "case_id": 7, "prompt": "Apple A5 was created by",
    "answer": " Google", "token_id": 5085, "prompt_token_ids": [26567, 362, 20, 572, 3465, 553],
}
WIDTH = 896
TENSOR = "model.norm.weight"
EMBEDDING = "model.embed_tokens.weight"
QUALIFICATION = "not release-qualified"
TRACE = {"layers": [1, 23], "roles": ["l_out", "kq_soft_max"],
         "positions": list(range(len(TARGET["prompt_token_ids"]))), "top_k": 100}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safetensors_header(stream):
    header_size, = struct.unpack("<Q", stream.read(8))
    require(header_size <= 1 << 20, "SafeTensors header exceeds 1 MiB")
    return 8 + header_size, json.loads(stream.read(header_size))


def bf16_values(data):
    return [value for value, in struct.iter_unpack("<f", b"".join(
        b"\x00\x00" + data[i:i+2] for i in range(0, len(data), 2)))]


def read_embedding_rows(path, *token_ids):
    with path.open("rb") as stream:
        start, header = safetensors_header(stream)
        require(header[TENSOR]["shape"] == [WIDTH] and header[TENSOR]["dtype"] == "BF16",
                "unexpected final norm")
        embed = header[EMBEDDING]
        require(embed["shape"] == [151936, WIDTH] and embed["dtype"] == "BF16" and "lm_head.weight" not in header,
                "expected tied Qwen token embeddings")
        rows = []
        for token in token_ids:
            require(0 <= token < embed["shape"][0], f"invalid embedding token: {token}")
            stream.seek(start + embed["data_offsets"][0] + token * WIDTH * 2)
            rows.append(bf16_values(stream.read(WIDTH * 2)))
        return rows


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def target_edit(residual, target, competitor):
    require(len(residual) == len(target) == len(competitor) == WIDTH, "unexpected hidden size")
    rms = math.sqrt(math.fsum(x * x for x in residual) / WIDTH + 1e-6)
    target_grad = [x * w / rms for x, w in zip(residual, target)]
    competitor_grad = [x * w / rms for x, w in zip(residual, competitor)]
    # Change the final norm gain. Aim at +4 for the target while leaving
    # the original top token's logit unchanged before BF16 rounding.
    projection = dot(target_grad, competitor_grad) / dot(competitor_grad, competitor_grad)
    direction = [g - projection * c for g, c in zip(target_grad, competitor_grad)]
    factor = 4.0 / dot(target_grad, direction)
    return {"op": "add", "tensor": TENSOR, "origin": [0],
            "shape": [WIDTH], "values": [factor * x for x in direction]}


def residual_row(capture, position):
    row, = capture["rows"]
    require(row["position"] == position and row["shape"] == [WIDTH], "wrong final residual")
    return struct.unpack(f"<{WIDTH}f", base64.b64decode(row["data_base64"], validate=True))


def project_target_probability(revision_path, residual, reference):
    """Apply the known last norm and tied output projection to a captured residual.

    This is client-side math, not a model run. Validate it against the trace
    logits before using the full-vocabulary softmax (trace results are top-k).
    """
    with revision_path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as weights:
        start, header = safetensors_header(stream)
        norm = header[TENSOR]
        require(norm["dtype"] == "BF16" and norm["shape"] == [WIDTH], "unexpected output norm")
        a, b = norm["data_offsets"]
        gains = bf16_values(weights[start + a:start + b])
        rms = math.sqrt(dot(residual, residual) / WIDTH + 1e-6)
        hidden = [x * gain / rms for x, gain in zip(residual, gains, strict=True)]
        embed = header[EMBEDDING]
        rows, width = embed["shape"]
        require(embed["dtype"] == "BF16" and (rows, width) == (151936, WIDTH) and
                "lm_head.weight" not in header, "unexpected output projection")
        offset = start + embed["data_offsets"][0]
        row_format = struct.Struct(f"<{WIDTH}H")
        lookup = bf16_values(struct.pack("<65536H", *range(65536)))
        logits = [
            math.sumprod(hidden, map(lookup.__getitem__, row_format.unpack_from(weights, offset + i * row_format.size)))
            for i in range(rows)
        ]
    error = max(abs(logits[row["token_id"]] - row["logit"]) for row in reference)
    # The GGUF backend rounds activation operands; this FP64 projection does not.
    require(error < 0.05, f"client projection disagrees with trace logits by {error:.4f}")
    best = max(logits)
    target = TARGET["token_id"]
    return {
        "probability": math.exp(logits[target] - best) / math.fsum(math.exp(x - best) for x in logits),
        "rank": 1 + sum(x > logits[target] for x in logits),
        "vocabulary_size": rows,
        "max_abs_error_vs_trace_top_k": error,
    }


def save(root, name, value):
    path = root / f"{name}.json"
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def setup(args, run):
    assets = run / "assets"
    assets.mkdir()
    for name in ASSETS:
        shutil.copyfile(HF / name, assets / name)
    worker = run / "worker"
    worker.mkdir(mode=0o700)
    save(run, "worker", {
        "root": str(worker), "engine": str(args.bins / "brewery-engine"),
        "source_roots": [str(assets), str(HF), str(BASE.parent)], "disk_reserve_bytes": 0,
        "engine_env": {
            "BREWERY_GGUF_SAFETENSORS": "1", "BREWERY_GGUF_QUANTIZER": str(QUANT),
            "BREWERY_GGUF_HF_EXPORTER": str(REPO / "tools/gguf-export-hf"),
            "BREWERY_GGUF_PYTHON": str(PYTHON), "BREWERY_LLAMA_CPP": str(LLAMA),
            "BREWERY_CONVERTER_DISK_RESERVE_BYTES": "0", "BREWERY_LANTERNFISH_TRACE": str(LANTERN),
            "BREWERY_LANTERNFISH_TRACE_LIB_DIR": str(TRACE_LIBS),
            "BREWERY_CPU_THREADS": "4", "BREWERY_SCRATCH_CHECKS_SCRIPT": str(REPO / "tools/scratch_checks.py"),
            "BREWERY_SCRATCH_CHECKS_RUNNER": str(RUNNER),
            "BREWERY_SCRATCH_CHECKS_RUNNER_SHA256": PINS[RUNNER],
            "BREWERY_SCRATCH_CHECKS_RUNNER_LIB_DIR": str(W0_LIBS),
            "BREWERY_SCRATCH_CHECKS_CACHE": str(FIXTURES),
        },
    })
    config = run / "worker.json"
    config.chmod(0o600)
    save(run, "dispatch", {"hosts": {"local": {"local": True,
         "worker": str(args.bins / "brewery-worker"), "config": str(config),
         "capabilities": {"trace": True}}}})
    return {
        "PATH": "/usr/bin:/bin", "HOME": str(run), "TMPDIR": str(ARB / "tmp"),
        "BREWERY_LISTEN": "127.0.0.1:0", "BREWERY_DATA": str(run / "state"),
        "BREWERY_WEB_DIST": str(args.web_dist), "BREWERY_DISPATCH_CONFIG": str(run / "dispatch.json"),
        "BREWERY_DISPATCH": str(args.bins / "brewery-dispatch"), "BREWERY_POLL_MS": "200",
        "BREWERY_LOCAL_EDIT_DISPATCH": str(args.bins / "brewery-dispatch"),
        "BREWERY_LOCAL_EDIT_WORKER": str(args.bins / "brewery-worker"),
        "BREWERY_LOCAL_EDIT_ENGINE": str(args.bins / "brewery-engine"),
        "BREWERY_LOCAL_EDIT_HELPER": str(HELPER),
    }


async def wait_marker(run, name, browser):
    path = run / f"{name}.json"
    for _ in range(300):
        if path.is_file():
            return json.loads(path.read_text())
        require(browser.returncode is None, f"browser exited before {name}; see workbench.log")
        await asyncio.sleep(0.2)
    raise TimeoutError(f"browser did not record {name}")


async def exercise(base, args, run, browser):
    import httpx
    from mcp import ClientSession, McpError
    from mcp.client.streamable_http import streamable_http_client

    evidence = args.evidence
    async with httpx.AsyncClient(trust_env=False, timeout=120) as http, \
            streamable_http_client(base + "/mcp", http_client=http) as (read, write, _), \
            ClientSession(read, write) as client:
        init = await client.initialize()
        catalog = [tool.name for tool in (await client.list_tools()).tools]
        save(evidence, "mcp-session", {"sdk": importlib.metadata.version("mcp"),
             "initialize": init.model_dump(mode="json"), "tools": catalog})

        async def call(tool, label=None, **params):
            result = await client.call_tool(tool, params)
            if label:
                payload = result.model_dump(mode="json") if result.isError else result.structuredContent
                save(evidence, label, payload)
                print(f"{tool}: {label}", flush=True)
            require(not result.isError and result.structuredContent is not None, f"{tool}: {result}")
            return result.structuredContent

        async def submit(tool, label, **params):
            return await call(tool, label, idempotency_key=f"w8-{label}", **params)

        async def wait(job, label):
            deadline = time.monotonic() + 1200
            samples = []
            while time.monotonic() < deadline:
                state = await call("get_job", id=job["id"], wait_ms=5000)
                sample = {k: state.get(k) for k in ("updated_at", "state", "progress", "progress_detail")}
                if not samples or samples[-1] != sample:
                    samples.append(sample)
                if state["state"] not in ("queued", "running"):
                    save(evidence, label, state)
                    save(evidence, label + "-progress", samples)
                    require(state["state"] == "succeeded", f"{label}: {state.get('error')}")
                    print(f"{label}: succeeded ({job['id']})", flush=True)
                    return state["result"]
            raise TimeoutError(f"{label}: {job['id']}")

        async def trace(label, model, model_sha, artifact=None, live=False):
            key = "trace-" + label
            options = TRACE if artifact is None else {**TRACE, "intervention_artifact": artifact}
            job = await submit("start_trace", key + "-queued", input_ids=[model],
                               prompt=TARGET["prompt"], options=options)
            if live:
                save(run, "baseline-job", {"id": job["id"]})
            result = await wait(job, key)
            require(result["model_sha256"] == model_sha and result["executable_sha256"] == PINS[LANTERN],
                    "trace does not match model and Lanternfish pins")
            manifest = await call("read_trace", key + "-manifest", id=job["id"], view="manifest")
            require(manifest["manifest_sha256"] == result["prompts"][0]["manifest_sha256"],
                    "trace manifest did not match its receipt")
            return job["id"], manifest

        async def register(label, kind, path, name):
            return (await submit("register_model", label, host="local", kind=kind, path=str(path), name=name))["id"]

        require((await call("list_models"))["models"] == [], "expected an empty isolated Library")
        base_id = await register("baseline-model", "gguf", BASE, "Qwen2.5-0.5B-Instruct BF16 (W8)")
        parent_id = await register("parent-model", "safetensors", SAFE, "Qwen2.5-0.5B-Instruct SafeTensors (W8)")
        assets_id = await register("assets-model", "assets", run / "assets", "Qwen2.5-0.5B-Instruct assets (W8)")
        baseline, manifest = await trace("baseline", base_id, PINS[BASE], live=True)
        require([t["token_id"] for t in manifest["tokens"]] == TARGET["prompt_token_ids"],
                "unexpected prompt tokens in the pinned GGUF")
        pos = manifest["tokens"][-1]["position"]
        artifact = await submit("create_intervention", "intervention", name="W8 residual steering",
                                model_id=base_id, model_sha256=PINS[BASE],
                                interventions=[{"op": "add", "layer": 1, "role": "l_out",
                                                "positions": [pos], "vector": [4 * math.sin(i) for i in range(WIDTH)]}])
        ref = {"id": artifact["id"], "version": artifact["version"]}
        steered, steered_manifest = await trace("steered", base_id, PINS[BASE], ref)
        require(steered_manifest["intervention_count"] == 1, "steering was not applied")
        steering = await call("diff_traces", "diff-steering", a=baseline, b=steered)
        shift = max(row["linf"] or 0 for row in steering["model_logits"])
        require(shift > 0.1, "expected a measurable change in a shared model logit")

        root = await submit("create_scratch", "scratch-parent", parent_id=parent_id, parent_sha256=PINS[SAFE])
        capture = await call("read_trace", "target-residual", id=baseline, view="tensor",
                             layer=23, role="l_out", start=pos, end=pos + 1)
        residual = residual_row(capture, pos)
        logits = next(row["a"] for row in steering["model_logits"] if row["position"] == pos)
        head = logits[0]["token_id"]
        require(head != TARGET["token_id"], "expected a different token at the top of the baseline trace")
        operation = target_edit(residual, *read_embedding_rows(SAFE, TARGET["token_id"], head))
        child = await submit("apply_edit", "scratch-child", id=root["id"], operations=[operation])
        save(run, "child", {"id": child["id"]})

        private = {}
        for label, revision in (("parent", root), ("child", child)):
            job = await submit("convert_scratch_gguf", f"convert-{label}-queued",
                               id=revision["id"], sha256=revision["sha256"], assets_id=assets_id)
            receipt = (await wait(job, "convert-" + label))["gguf"]
            require(receipt["revision_sha256"] == revision["sha256"], "private GGUF does not match revision")
            private[label] = receipt
        parent_trace, parent_manifest = await trace("private-parent", root["id"], private["parent"]["sha256"])
        child_trace, child_manifest = await trace("private-child", child["id"], private["child"]["sha256"])
        require(parent_manifest["tokens"] == child_manifest["tokens"] == manifest["tokens"], "trace tokens differ")
        tensor = await call("diff_tensors", "diff-tensors", a=root["id"], b=child["id"])
        traces = await call("diff_traces", "diff-traces", a=parent_trace, b=child_trace)
        require(len(tensor["tensors"]) == 1 and tensor["tensors"][0]["name"] == TENSOR,
                "expected only the edited embedding tensor to change")

        def target_logit(side):
            row = next(row for row in traces["model_logits"] if row["position"] == pos)
            return next((token["logit"] for token in row[side] if token["token_id"] == TARGET["token_id"]), None)

        before, after = target_logit("a"), target_logit("b")
        require(before is not None and after is not None and after > before + 1.0, "target logit did not rise")
        checks_job = await submit("run_checks", "checks-queued", id=child["id"], sha256=child["sha256"])
        probabilities = {"method": "client-side cross-check; not a worker check"}
        final = next(row for row in traces["model_logits"] if row["position"] == pos)
        for label, revision, trace_id, side in (
            ("before", root, parent_trace, "a"), ("after", child, child_trace, "b"),
        ):
            capture = await call("read_trace", "target-residual-" + label, id=trace_id, view="tensor",
                                 layer=23, role="l_out", start=pos, end=pos + 1)
            checkpoint = run / "state" / "scratch" / revision["id"] / "model.safetensors"
            require(sha(checkpoint) == revision["sha256"], "output projection uses the wrong SafeTensors hash")
            probabilities[label] = await asyncio.to_thread(
                project_target_probability, checkpoint, residual_row(capture, pos), final[side])
        save(evidence, "target-probability", probabilities)
        await wait(checks_job, "checks-job")
        checks = await call("get_checks", "checks", id=child["id"])
        measured = checks["measured_report"]
        require(all(checks[k]["status"] == "passed" for k in ("integrity", "smoke", "behavior")) and
                len(measured["metrics"]) == 17 and measured["runner_sha256"] == PINS[RUNNER] and
                measured["parent_gguf_sha256"] == private["parent"]["sha256"] and
                measured["revision_gguf_sha256"] == private["child"]["sha256"],
                "checks did not pass on the pinned GGUF pair")
        behavior = await call("diff_behavior", "diff-behavior", a=root["id"], b=child["id"])
        require(len(behavior["metrics"]) == 17, "missing behavior diff metrics")

        pending = await submit("request_publish", "publish-request", id=child["id"], sha256=child["sha256"])
        require(pending["state"] == "publish_requested" and pending["sha256"] == child["sha256"],
                "publication was not requested for the checked revision")
        require("approve" not in catalog, "operator approval must not be an MCP tool")
        try:
            await client.call_tool("approve", {"id": child["id"], "sha256": child["sha256"]})
        except McpError as exc:
            require(exc.error.code == -32602 and exc.error.message == "unknown tool", str(exc))
            save(evidence, "mcp-approval-refused", exc.error.model_dump(mode="json"))
        else:
            raise RuntimeError("MCP accepted operator approval")
        models = (await call("list_models", "library-before-approval"))["models"]
        require({m["id"] for m in models} == {base_id, parent_id, assets_id}, "scratch leaked into Library")
        save(run, "publish", {"id": child["id"], "sha256": child["sha256"]})
        ui = await wait_marker(run, "operator-approved", browser)
        approved = await call("get_scratch", "approved", id=child["id"])
        models = (await call("list_models", "library-after-approval"))["models"]
        new = [m for m in models if m["id"] not in {base_id, parent_id, assets_id}]
        require(approved["state"] == "approved" and len(new) == 1 and
                new[0]["id"] == approved["library_model_id"] == ui["library_model_id"] and
                new[0]["kind"] == "safetensors" and sha(Path(new[0]["path"])) == child["sha256"],
                "Library must contain only the approved hash, not the private GGUF")
        return {
            "qualification": QUALIFICATION, "controller": base, "runtime_dir": str(run),
            "status": "passed (GSM8K not run)", "target": {
                **TARGET, "worker_fixture_contains_case": True,
                "worker_probability_scope": "CounterFact and zsRE group aggregates; no per-case scores are returned",
                "baseline_top_token_id": head, "before_logit": before, "after_logit": after,
                "logit_shift": after - before, "client_side_cross_check": probabilities,
            },
            "steering_max_shared_model_logit_shift": shift, "steering_artifact": ref,
            "private_gguf": private, "delta": child["delta"], "tensor_diff": tensor["tensors"][0],
            "trace_diff": {k: len(traces[k]) for k in ("activations", "logit_lens", "model_logits")},
            "trace_activation_max_l2": max(row["l2"] or 0 for row in traces["activations"]),
            "checks": measured, "check_status": {k: checks[k]["status"] for k in ("integrity", "smoke", "behavior")},
            "approval": {"mcp": "refused", "operator_workbench": "passed"},
            "approved_library_model": new[0], "workbench": ui,
            "gsm8k": {
                "status": "not run",
                "reason": "The Brewery harness requires the full pinned split and an HGN GPU candidate. "
                          "It has no MCP path or CPU subset for scratch BF16 GGUF.",
            },
        }


async def run_acceptance(args, run):
    env = setup(args, run)
    controller = browser = None
    with (args.evidence / "controller.log").open("w") as log, (args.evidence / "workbench.log").open("w") as ui:
        try:
            controller = await asyncio.create_subprocess_exec(
                args.bins / "brewery", env=env, cwd=REPO, stdin=subprocess.DEVNULL,
                stdout=log, stderr=log)
            match = None
            for _ in range(200):
                require(controller.returncode is None, "controller exited; see controller.log")
                match = re.search(r"^brewery listening on (http://127\.0\.0\.1:(\d+))$",
                                  (args.evidence / "controller.log").read_text(), re.M)
                if match:
                    break
                await asyncio.sleep(0.1)
            require(match and int(match[2]) not in {0, 3000, 4095, 4141}, "no safe loopback port in controller.log")
            browser = await asyncio.create_subprocess_exec(
                "node", str(Path(__file__).with_suffix(".mjs")), match[1], str(run), str(args.evidence),
                env={k: v for k, v in os.environ.items() if not k.startswith("BREWERY_")} | {
                    "TMPDIR": env["TMPDIR"], "HOME": env["HOME"], "XDG_CACHE_HOME": str(run / "cache"),
                }, stdout=ui, stderr=ui)
            await wait_marker(run, "operator-ready", browser)
            result = await exercise(match[1], args, run, browser)
            require(await asyncio.wait_for(browser.wait(), timeout=20) == 0, "browser did not exit cleanly")
            return result
        finally:
            for process in (browser, controller):
                if process and process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=10)
                    except TimeoutError:
                        process.kill()
                        await process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the workflow without writing files or starting services")
    parser.add_argument("--root", type=Path, default=ARB / "w8-accept", help="isolated run root under /arboretum")
    parser.add_argument("--bins", type=Path, default=ARB / "brewery-w3-target/debug")
    parser.add_argument("--web-dist", type=Path, default=ARB / "w8-accept/web")
    args = parser.parse_args()
    args.root = args.root.resolve()
    if ARB not in args.root.parents:
        parser.error("use a child of /arboretum for --root")
    if args.dry_run:
        target, competitor = [1] * WIDTH, [1, 0] * (WIDTH // 2)
        op = target_edit([1] * WIDTH, target, competitor)
        require(abs(dot(target, op["values"]) - 4.0) < 1e-4 and abs(dot(competitor, op["values"])) < 1e-6,
                "edit did not aim at a +4 logit shift without changing the top token")
        print(json.dumps({
            "root": str(args.root), "listen": "127.0.0.1:0", "target": TARGET,
            "workflow": ["operator enables local editor", "register via MCP", "baseline trace",
                         "saved intervention", "steered trace", "scratch and one edit",
                         "two private BF16 conversions", "paired scratch traces", "tensor and trace diffs",
                         "worker checks and behavior diff", "publish request", "MCP approval refused",
                         "operator approves in workbench", "verify SafeTensors Library row via MCP"],
        }, indent=2))
        return

    inputs = {str(path): sha(path) for path in PINS}
    require(hasattr(math, "sumprod"), "use Python 3.12 or later")
    require(all(inputs[str(path)] == pin for path, pin in PINS.items()), "input pin mismatch")
    inputs.update({str(HF / name): sha(HF / name) for name in ASSETS})
    manifest = "".join(f"{inputs[str(HF / name)]}  {name}\n" for name in sorted((*ASSETS, SAFE.name)))
    require(hashlib.sha256(manifest.encode()).hexdigest() == SNAPSHOT_PIN, "HF snapshot pin mismatch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    args.evidence = args.root / "evidence"
    args.evidence.mkdir(parents=True, exist_ok=True)
    require(not (args.evidence / "mcp-session.json").exists(), "use a fresh --root for a new run")
    run = Path(tempfile.mkdtemp(prefix="isolated-", dir=args.root))
    (args.evidence / "hf-snapshot.sha256").write_text(manifest)
    save(args.evidence, "input-pins", {"brewery_commit": commit, "hf_upstream_revision": None, "files": inputs})
    report = asyncio.run(run_acceptance(args, run))
    report["input_pins"] = inputs
    report["brewery_commit"] = commit
    report["hf_upstream_revision"] = None
    report["hf_snapshot_manifest_sha256"] = SNAPSHOT_PIN
    save(args.evidence, "report", report)
    print(json.dumps(report["target"], indent=2))


if __name__ == "__main__":
    main()
