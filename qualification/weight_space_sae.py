#!/usr/bin/env python3
"""Qualify W11 against an isolated CPU controller and pinned traces."""

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from weight_space_acceptance import ARB, BASE, LANTERN as W0_PATH, PINS, QUALIFICATION, REPO, W0_LIBS, require, save, sha


LAYER, ROLE = 16, "l_out"
LANTERN = ARB / "pins/lanternfish-0ea30b3/lanternfish"
LANTERN_SHA = PINS[W0_PATH]
CORPUS = [
    ("alice", "lewis-carroll_alices-adventures-in-wonderland", "8da5cac1a4ec2c811322eff79daa5e11e851a813", "a006f6595337f1a3b35e46676352200514f1a50511125edda6bf898f83027486"),
    ("wells", "h-g-wells_the-time-machine", "fbd98543d4195ffdd8dbb90facb4309223bcb147", "271ade0cbef9469a3b6afac85e6b76d49f632398ff35dae08c7cf1f63ee268ee"),
    ("dickens", "charles-dickens_a-christmas-carol", "2b37c1cc1f4db69e7806816893afbadba9fa8da7", "2390fc9d0ced003cf444cf419b0fc5bdf11bdd0d456bb4d57c01b80a9055fb81"),
    ("shelley", "mary-shelley_frankenstein", "52a3e09c21b3cebcb8eec129d274b8801a229147", "3dc161f1f60ad594583567b76040cbb7367106e51d24af213a19b2e893c67508"),
]
PUBLIC_REPO = "https://huggingface.co/HuggingAnalist/sae-qwen2.5-0.5B-res"
PUBLIC_REV = "59535fa7275112a34424fe247028d2b8391c63c9"
PUBLIC_PATH = "layer_16/final_sae/sae_weights.safetensors"
PUBLIC_SHA = "972a8a6529ebbe766d63450455f450cb7fb05726f210e552224153dc91751d55"
PROMPTS = ["The capital of France is", "When the White Rabbit reached the garden, it"]


def fetch(path, url, digest):
    if not path.is_file():
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    require(sha(path) == digest, f"SHA-256 mismatch: {path}")
    return path


class Controller:
    def __init__(self, base):
        self.base = base
        self.client = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path, body=None, key=None):
        headers = {} if key is None else {"Idempotency-Key": key}
        data = None
        if body is not None:
            data = json.dumps(body, allow_nan=False).encode()
            headers["Content-Type"] = "application/json"
        if path == "/mcp":
            headers.update({"Accept":"application/json, text/event-stream", "MCP-Protocol-Version":"2025-06-18"})
        with self.client.open(urllib.request.Request(self.base + path, data=data, headers=headers),
                              timeout=120) as response:
            return json.load(response)

    def mcp(self, tool, **args):
        response = self.request("/mcp", {"jsonrpc":"2.0", "id":1, "method":"tools/call",
                                        "params":{"name":tool, "arguments":args}})
        require("error" not in response and not response["result"]["isError"], f"{tool}: {response}")
        return response["result"]["structuredContent"]

    def trace(self, name, model, prompts, positions, *, sae=None, steering=None):
        options = {"layers":[LAYER], "roles":[ROLE], "positions":list(positions), "prompts":prompts,
                   "top_k":50 if sae else 0, "byte_cap":4 << 20}
        if sae:
            options["sae"] = sae
        if steering:
            options["feature_steering"] = steering
        job = self.mcp("steer_feature" if steering else "start_trace", input_ids=[model],
                       options=options, idempotency_key=name)["id"]
        for _ in range(900):
            status = self.mcp("get_job", id=job, wait_ms=500)
            if status["state"] in ("queued", "running"):
                continue
            require(status["state"] == "succeeded", f"{name}: {status}")
            r = status["result"]
            require(r["model_sha256"] == PINS[BASE] and r["executable_sha256"] == LANTERN_SHA,
                    f"{name}: wrong model or tool revision")
            print(f"{name}: succeeded ({job})", flush=True)
            return job, r
        raise TimeoutError(f"trace timed out: {job}")

    def add_sae(self, metadata, blob, key):
        return json.loads(subprocess.check_output([
            "curl", "--noproxy", "*", "--fail-with-body", "--silent", "--show-error",
            "--max-time", "120", "-H", f"Origin: {self.base}", "-H", f"Idempotency-Key: {key}",
            "-F", f"metadata=@{metadata};type=application/json",
            "-F", f"weights=@{blob};type=application/octet-stream", self.base + "/api/saes",
        ]))


def capture(controller, model, sources):
    rows = []
    for name, paragraphs in sources:
        # Trace 8 disjoint, contiguous 150-word chunks per book. Keep each trace
        # within 4 prompts and 128 positions. Do not capture generated tokens.
        words = " ".join(paragraphs).split()
        require(len(words) >= 1200, f"{name}: not enough source text")
        for block in range(0, 1200, 600):
            prompts = [" ".join(words[i:i + 150]) for i in range(block, block + 600, 150)]
            job, receipt = controller.trace(f"corpus-{name}-{block}", model, prompts, range(128))
            for p in range(len(prompts)):
                manifest = controller.mcp("read_trace", id=job, view="manifest", prompt=p)
                require(manifest["manifest_sha256"] == receipt["prompts"][p]["manifest_sha256"], "manifest hash")
                positions = {x["position"] for x in manifest["tokens"] if x["position"] < 128}
                query = urllib.parse.urlencode(dict(prompt=p, layer=LAYER, role=ROLE, start=0, end=min(max(positions) + 1, 128)))
                trace = controller.request(f"/api/jobs/{job}/trace/tensor?{query}")
                require({x["position"] for x in trace["rows"]} == positions, "incomplete activation capture")
                for row in trace["rows"]:
                    data = base64.b64decode(row["data_base64"], validate=True)
                    require(row["shape"] == [896] and hashlib.sha256(data).hexdigest() == row["sha256"],
                            "wrong activation width or hash")
                    rows.append(np.frombuffer(data, dtype="<f4"))
    return torch.from_numpy(np.stack(rows))


def exercise(c, args, run):
    model = c.mcp("register_model", host="local", kind="gguf", path=str(BASE),
                  name="Qwen2.5-0.5B-Instruct BF16 (W11)", idempotency_key="qwen")["id"]
    corpus, texts = [], []
    for name, repo, commit, digest in CORPUS:
        url = f"https://raw.githubusercontent.com/standardebooks/{repo}/{commit}/src/epub/text/chapter-1.xhtml"
        doc = ET.parse(fetch(args.inputs / f"{name}.xhtml", url, digest))
        texts.append((name, ["".join(p.itertext()) for p in doc.findall(".//{*}section/{*}p")]))
        corpus.append({"source":url, "sha256":digest, "license":"US public domain; Standard Ebooks edits CC0-1.0",
                       "split":"heldout" if name == "shelley" else "train"})
    x = {"train":capture(c, model, texts[:3]), "heldout":capture(c, model, texts[3:])}
    save_file(x, run / "dataset.safetensors")
    binding = {"model_id":model, "model_sha256":PINS[BASE], "layer":LAYER, "role":ROLE}
    save(run, "binding", binding | {"name":"W11 CPU TopK SAE", "provenance":{
        "corpus":corpus, "selection":"first 1200 words per source, split into 150-word chunks; positions <128",
        "lanternfish_sha256":LANTERN_SHA}})
    local = run / "trained"
    subprocess.run([sys.executable, str(REPO / "tools/train_sae.py"), str(run / "dataset.safetensors"),
                    str(run / "binding.json"), str(local)], check=True)
    trained = c.add_sae(local / "metadata.json", local / "sae.safetensors", "trained-sae")
    require(trained["weights_sha256"] == sha(local / "sae.safetensors"), "trained weights hash mismatch")

    public_blob = fetch(args.inputs / "public.safetensors", f"{PUBLIC_REPO}/resolve/{PUBLIC_REV}/{PUBLIC_PATH}", PUBLIC_SHA)
    public = binding | {"schema_version":1, "name":"HuggingAnalist ReLU SAE (layer 16)",
        "d_in":896, "d_sae":14336, "activation":{"kind":"relu"},
        "normalization":{"scale":1, "subtract_decoder_bias":True},
        "provenance":{"method":"public-import", "license":"Apache-2.0", "source":PUBLIC_REPO,
                      "revision":PUBLIC_REV, "weights":PUBLIC_PATH, "sha256":PUBLIC_SHA,
                      "config":f"{PUBLIC_REPO}/blob/{PUBLIC_REV}/layer_16/final_sae/cfg.json",
                      "note":"Source config does not pin model revision and differs from card on corpus."}}
    save(run, "public-metadata", public)
    imported = c.add_sae(run / "public-metadata.json", public_blob, "public-sae")
    require(imported["weights_sha256"] == PUBLIC_SHA, "public weights hash mismatch")

    positions = range(5)

    def feature_run(name, a, blob):
        selection = {"id":a["id"], "version":a["version"], "top_k":5}
        job, _ = c.trace(name, model, PROMPTS, positions, sae=selection)
        reads = []
        w = load_file(blob)
        max_error = 0.
        for p, prompt in enumerate(PROMPTS):
            result = c.mcp("read_feature_trace", id=job, prompt=p, start=0, end=16)
            require(result["rows"] and result["sae_id"] == selection["id"] and result["version"] == selection["version"],
                    "missing captured features or wrong SAE version")
            reference = c.mcp("read_trace", id=job, view="tensor", prompt=p, layer=LAYER, role=ROLE, start=0, end=16)
            inputs = {r["position"]:np.frombuffer(base64.b64decode(r["data_base64"]), dtype="<f4") for r in reference["rows"]}
            require(set(inputs) == {r["position"] for r in result["rows"]}, "wrong feature trace positions")
            for r in result["rows"]:
                x = torch.tensor(inputs[r["position"]]) * a["normalization"]["scale"] - w["b_dec"]
                acts, ids = (x @ w["W_enc"] + w["b_enc"]).relu().topk(5)
                expected = [(int(i), float(value)) for i, value in zip(ids, acts) if value > 0]
                require([f["feature"] for f in r["top"]] == [i for i, _ in expected], "feature ID mismatch vs Torch")
                for f, (_, value) in zip(r["top"], expected, strict=True):
                    err = abs(f["activation"] - value)
                    require(err < 2e-4 * max(1, value), "feature activation mismatch vs Torch")
                    max_error = max(max_error, err)
            reads.append({"prompt_text":prompt, **result})
        save(args.evidence, name, {"job_id":job, "features":reads, "max_activation_error_vs_torch":max_error})
        return job, selection, reads

    torch.set_num_threads(4)
    feature_run("imported-features", imported, public_blob)
    baseline, selection, reads = feature_run("trained-features", trained, local / "sae.safetensors")
    p = 0
    row = max(reads[p]["rows"], key=lambda row: row["position"])
    position, feature = row["position"], row["top"][0]["feature"]
    steering = {"id":trained["id"], "version":trained["version"], "feature":feature,
                "positions":[position], "alpha":8.0}
    steered, _ = c.trace("steered", model, PROMPTS, positions, sae=selection, steering=steering)
    manifest = c.mcp("read_trace", id=steered, view="manifest", prompt=p)
    require(manifest["intervention_count"] == 1, "steering did not run")
    diff = c.mcp("diff_traces", a=baseline, b=steered, prompt=p)
    shift = max(logits["linf"] or 0 for logits in diff["model_logits"])
    require(shift > 0.1, "no measurable shared-logit change")
    weights = load_file(local / "sae.safetensors")
    save(args.evidence, "steering-diff", diff)
    save(args.evidence, "report", {"qualification":QUALIFICATION, "model_sha256":PINS[BASE],
        "lanternfish_sha256":LANTERN_SHA, "trained":trained, "imported":imported, "baseline_job_id":baseline,
        "steered_job_id":steered, "steering":steering, "max_shared_model_logit_shift":shift,
        "decoder_direction_l2":float(weights["W_dec"][feature].norm()),
        "w4":"Not run: steering changes activations, not model weights."})
    print(f"Max shared model logit shift: {shift:.6f}; {QUALIFICATION}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bins", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    args = p.parse_args()
    require(args.bins.is_absolute() and args.inputs.is_absolute() and args.evidence.is_absolute(),
            "Use absolute paths")
    args.inputs.mkdir(parents=True, exist_ok=True)
    args.evidence.mkdir(parents=True, exist_ok=False)
    for path, digest in ((BASE, PINS[BASE]), (LANTERN, LANTERN_SHA)):
        require(path.is_file() and sha(path) == digest, f"Missing file or SHA-256 mismatch: {path}")
    with tempfile.TemporaryDirectory(prefix="w11-run-", dir=ARB / "tmp") as directory, \
            (args.evidence / "controller.log").open("w") as log:
        run = Path(directory)
        worker = run / "worker"
        worker.mkdir(mode=0o700)
        save(run, "worker", {"root":str(worker), "engine":str(args.bins / "brewery-engine"),
            "source_roots":[str(BASE.parent)], "disk_reserve_bytes":0, "engine_env":{
                "BREWERY_LANTERNFISH_TRACE":str(LANTERN), "BREWERY_LANTERNFISH_TRACE_LIB_DIR":str(W0_LIBS),
                "BREWERY_CPU_THREADS":"4"}})
        config = run / "worker.json"
        config.chmod(0o600)
        save(run, "dispatch", {"hosts":{"local":{"local":True, "worker":str(args.bins / "brewery-worker"),
                                                "config":str(config), "capabilities":{"trace":True}}}})
        env = {"PATH":"/usr/bin:/bin", "HOME":str(run), "TMPDIR":str(ARB / "tmp"),
               "BREWERY_LISTEN":"127.0.0.1:0", "BREWERY_DATA":str(run / "state"),
               "BREWERY_WEB_DIST":str(run), "BREWERY_DISPATCH_CONFIG":str(run / "dispatch.json"),
               "BREWERY_DISPATCH":str(args.bins / "brewery-dispatch"), "BREWERY_POLL_MS":"200"}
        with subprocess.Popen([args.bins / "brewery"], env=env, stdin=subprocess.DEVNULL,
                              stdout=log, stderr=log) as server:
            try:
                match = None
                for _ in range(100):
                    require(server.poll() is None, "controller exited; see controller.log")
                    match = re.search(r"^brewery listening on (http://127\.0\.0\.1:(\d+))$",
                                      (args.evidence / "controller.log").read_text(), re.M)
                    if match:
                        break
                    time.sleep(0.1)
                require(match and int(match[2]) not in {0, 3000, 4095, 4141}, "no safe loopback port")
                try:
                    exercise(Controller(match[1]), args, run)
                except Exception:
                    errors = sorted(worker.glob("jobs/*/engine-stderr.log"))
                    if errors:
                        save(args.evidence, "worker-errors", {str(e.relative_to(worker)):e.read_text(errors="replace")[-4000:] for e in errors})
                    raise
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()


if __name__ == "__main__":
    main()
