#!/usr/bin/env python3
"""Offline text-indexer distillation for the dense Qwen3.8-27B checkpoint.

Reference: kuraneko1/qsa-indexer-forge, see REFERENCE_REVISION.
The Transformers decoder retains its original Q/K projections, GQA, output
gating and GDN layers. Only its full-attention backend is substituted so the
trainer can consume the same causal Q/K that the frozen forward pass uses.

The held-out corpus must not share any exact packed token sequence with the
training corpus. This does not detect partial, shifted, or fuzzy matches.
"""

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
from tempfile import TemporaryDirectory

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

REFERENCE_REVISION = "9eb7192828ebc6234a892d718fc775b3df648596"
TEACHER = "dense_causal_qwen3_5"
BLOCK, HEADS, WIDTH, ROPE = 4, 4, 128, 64
ATTENTION = "brewery_qsa_dense"
MAX_HELDOUT_EXAMPLE_HASHES = 250_000
DISTRIBUTED_SHARD = re.compile(r"shard-[0-9]{5}-model-[0-9]{5}-of-[0-9]{5}\.safetensors")
TOKENIZER_CLASSES = (None, "Qwen2Tokenizer", "Qwen2TokenizerFast", "Qwen3_5Tokenizer",
                     "Qwen3_5TokenizerFast", "TokenizersBackend", "PreTrainedTokenizerFast")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def regular_member(path):
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate object key in teacher JSON")
        result[key] = value
    return result


@contextmanager
def open_teacher_member(path):
    # The source is local. Never follow a member symlink or block opening a FIFO.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Teacher files must be regular local members")
        yield stream


@dataclass(frozen=True)
class TeacherSource:
    config: dict
    config_sha256: str
    shards: tuple[str, ...]
    index: dict | None


def teacher_source(source):
    """Select the standard dense Qwen config and the exact SafeTensors members."""
    if not source.is_dir():
        raise ValueError("--source must be a local Hugging Face directory")
    if os.path.lexists(source / "adapter_config.json"):
        raise ValueError("Teacher must not specify an adapter")
    if any(DISTRIBUTED_SHARD.fullmatch(path.name) for path in source.iterdir()):
        raise ValueError("Teacher must not specify a distributed checkpoint")
    with open_teacher_member(source / "config.json") as stream:
        config_bytes = stream.read()
    config = json.loads(config_bytes, object_pairs_hook=unique_json_object)
    if not isinstance(config, dict) or config.get("model_type") != "qwen3_5":
        raise ValueError("Teacher config must describe the full Qwen3.5/3.8 architecture")
    for selector in ("configuration_files", "transformers_weights"):
        if selector in config:
            raise ValueError(f"Teacher config must not specify {selector}")
    if config.get("tokenizer_class") not in TOKENIZER_CLASSES:
        raise ValueError("Teacher config must use a supported Qwen tokenizer class")
    for subconfig, types in ((config, ("qwen3_5",)),
                             (config.get("text_config"), (None, "qwen3_5_text")),
                             (config.get("vision_config"), (None, "qwen3_5", "qwen3_5_vision"))):
        if not isinstance(subconfig, dict) or subconfig.get("quantization_config") is not None:
            raise ValueError("Teacher config must describe an unquantized Qwen text and vision model")
        if subconfig.get("model_type") not in types or subconfig.get("fusion_config") is not None:
            raise ValueError("Teacher config must use the built-in Qwen components without fusions")
    config_hash = hashlib.sha256(config_bytes).hexdigest()

    mono = source / "model.safetensors"
    index = source / "model.safetensors.index.json"
    selected = [path for path in (mono, index) if os.path.lexists(path)]
    if len(selected) != 1 or not regular_member(selected[0]):
        raise ValueError("Teacher must supply one regular model.safetensors or model.safetensors.index.json")
    if selected[0] == mono:
        return TeacherSource(config, config_hash, (mono.name,), None)
    with open_teacher_member(index) as stream:
        manifest = json.load(stream, object_pairs_hook=unique_json_object)
    weight_map = manifest.get("weight_map") if isinstance(manifest, dict) else None
    if not isinstance(weight_map, dict) or not weight_map or not isinstance(manifest.get("metadata"), dict):
        raise ValueError("Teacher SafeTensors index must contain metadata and a nonempty weight_map")
    checked = set()
    for name in weight_map.values():
        if (not isinstance(name, str) or not name.endswith(".safetensors")
                or any(ch in name for ch in ("/", "\\", "\0"))):
            raise ValueError("Teacher SafeTensors shard is not a regular local member")
        if name not in checked:
            if not regular_member(source / name):
                raise ValueError("Teacher SafeTensors shard is not a regular local member")
            checked.add(name)
    return TeacherSource(config, config_hash, tuple(sorted(checked)), manifest)


@contextmanager
def teacher_file_view(source, names, output):
    # Present only validated members and hold their descriptors until the
    # loader finishes. /proc/self/fd works across filesystems without copying.
    with ExitStack() as stack:
        view = Path(stack.enter_context(TemporaryDirectory(prefix=".teacher-", dir=output)))
        for name in names:
            stream = stack.enter_context(open_teacher_member(source / name))
            (view / name).symlink_to(f"/proc/self/fd/{stream.fileno()}")
        yield view


@contextmanager
def teacher_weight_view(source, teacher, output):
    # Transformers has no public checkpoint-file-list API. Snapshot the
    # validated index in the view so its resolver sees the same shard list.
    with teacher_file_view(source, teacher.shards, output) as view:
        if teacher.index is not None:
            index = view / "model.safetensors.index.json"
            index.write_text(json.dumps(teacher.index), encoding="utf-8")
        yield view


def tokenizer_source(source):
    """Read only canonical Qwen tokenizer assets; do not redirect via config."""
    with open_teacher_member(source / "tokenizer_config.json") as stream:
        config = json.load(stream, object_pairs_hook=unique_json_object)
    if not isinstance(config, dict):
        raise ValueError("Teacher tokenizer_config.json must contain an object")
    for selector in ("fast_tokenizer_files", "auto_map", "init_inputs", "tokenizer_object", "vocab", "merges"):
        if selector in config:
            raise ValueError(f"Teacher tokenizer config must not specify {selector}")
    if config.get("tokenizer_class") not in TOKENIZER_CLASSES:
        raise ValueError("Teacher tokenizer config must use a supported Qwen tokenizer class")
    if any(key.endswith(("_file", "_files", "_path")) and key != "name_or_path" for key in config):
        raise ValueError("Teacher tokenizer config must not select files")
    if not regular_member(source / "tokenizer.json"):
        raise ValueError("Teacher must supply a regular tokenizer.json")
    names = ["tokenizer.json"]
    for name in ("added_tokens.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
        if os.path.lexists(source / name):
            names.append(name)
    return config, names


def load_teacher_tokenizer(source, model_config, output):
    from transformers import AutoTokenizer

    config, names = tokenizer_source(source)
    with teacher_file_view(source, names, output) as view:
        (view / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
        return AutoTokenizer.from_pretrained(
            view, config=model_config, local_files_only=True, trust_remote_code=False,
        )


def plan_teacher_devices(model, gpu_capacity):
    from transformers.integrations.accelerate import _get_device_map

    # In Transformers 5.22.1, sequential planning respects each supplied cap.
    # Unlike "auto", it does not clip larger devices to a balanced share.
    caps = {i: capacity for i, capacity in gpu_capacity.items() if capacity > 0}
    plan = _get_device_map(model, "sequential", {**caps, "cpu": 0}, hf_quantizer=None)
    if not plan or any(device not in caps for device in plan.values()):
        raise RuntimeError("The BF16 teacher does not fit the per-GPU budgets; CPU and disk offload are not supported")
    return plan


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="Check GPU/backend readiness without loading a model")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--budget", type=int, default=2048, help="Token budget for held-out top-block capture")
    parser.add_argument("--query-chunk", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto",
                        help="Auto uses local GPUs; CPU is for small fixtures")
    args = parser.parse_args()
    if args.probe:
        return args
    for name in ("source", "train_data", "eval_data", "output_dir", "steps", "seq_len", "seed", "lr"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    for name, low, high in (("steps", 1, 10000), ("seq_len", 8, 16384), ("seed", 0, 2**32 - 1),
                            ("budget", 1, 16384), ("query_chunk", 1, 512)):
        if not low <= getattr(args, name) <= high:
            parser.error(f"--{name.replace('_', '-')} must be between {low} and {high}")
    if not 0 < args.lr <= 1:
        parser.error("--lr must be finite and in (0, 1]")
    return args


def main():
    args = arguments()
    if not args.probe:
        source = args.source.resolve(strict=True)
        teacher = teacher_source(source)
        source_config_hash = teacher.config_sha256
        train_hash, eval_hash = digest(args.train_data), digest(args.eval_data)
        if train_hash == eval_hash:
            raise ValueError("The training and held-out corpora must be different")
        output = args.output_dir
        output.mkdir(mode=0o700, exist_ok=True)
        entry = output.lstat()
        if (not stat.S_ISDIR(entry.st_mode) or entry.st_uid != os.getuid()
                or entry.st_mode & 0o077 or any(output.iterdir())):
            raise ValueError("--output-dir must be an empty private directory owned by this process")

    import accelerate  # noqa: F401 - required by device-map loading
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from transformers import GenerationConfig, Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.modeling_qwen3_5 import ALL_ATTENTION_FUNCTIONS, Qwen3_5Attention

    if args.probe:
        probe(torch, F)
        return

    config = Qwen3_5Config.from_dict(teacher.config)
    tokenizer = load_teacher_tokenizer(source, config, output)

    def chunks(path):
        return corpus(tokenizer, path, args.seq_len)

    reject_packed_overlap(chunks(args.train_data), chunks(args.eval_data))

    torch.manual_seed(args.seed)
    cpu = args.device == "cpu"
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError("QSA training requires a local CUDA or ROCm device")
    text_config = config.text_config
    expected = [i for i, kind in enumerate(text_config.layer_types) if kind == "full_attention"]
    if not expected:
        raise ValueError("The teacher has no full-attention text layers")
    if not cpu and ((text_config.hidden_size, text_config.num_attention_heads,
                     text_config.num_key_value_heads, text_config.head_dim) != (5120, 24, 4, 256)
                    or expected != list(range(3, 64, 4))):
        raise ValueError("GPU training requires the Qwen3.8-27B dense text architecture")
    if args.seq_len > text_config.max_position_embeddings:
        raise ValueError("--seq-len exceeds the teacher context length")
    load = {"device_map": {"": "cpu"}, "dtype": torch.float32}
    if not cpu:
        # Leave room for activations, FP32 indexers and optimizer state on each GPU.
        gpu_capacity = {i: max(0, torch.cuda.mem_get_info(i)[0] - 3 * 1024**3)
                        for i in range(torch.cuda.device_count())}
        if sum(gpu_capacity.values()) < 50 * 1024**3:
            raise RuntimeError("Not enough free GPU memory for the BF16 27B checkpoint and training")
        # The aggregate check is only a lower bound. Plan on a metadata-only
        # copy so placement failures cannot start loading or offloading shards.
        with torch.device("meta"):
            template = Qwen3_5ForConditionalGeneration._from_config(
                deepcopy(config), dtype=torch.bfloat16, attn_implementation="sdpa",
            )
        load = {"device_map": plan_teacher_devices(template, gpu_capacity), "dtype": torch.bfloat16}
        del template
    with teacher_weight_view(source, teacher, output) as weight_view:
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            weight_view, config=config, generation_config=GenerationConfig.from_model_config(config),
            local_files_only=True, trust_remote_code=False, use_safetensors=True, attn_implementation="sdpa",
            output_loading_info=True, use_kernels=False, **load,
        )
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("The teacher checkpoint did not supply every required backbone tensor")
    if not cpu and any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("QSA training needs the whole teacher on GPUs; CPU and disk offload are not supported")
    model.eval().requires_grad_(False)
    teacher_parameters = {name: (parameter, parameter._version) for name, parameter in model.named_parameters()}
    text = model.model.language_model
    if any("_qsa_indexer." in name for name in loading.get("unexpected_keys", ())):
        raise ValueError("Use a dense checkpoint without existing QSA tensors")

    torch.manual_seed(args.seed)
    heads = {}
    for i in expected:
        attention = text.layers[i].self_attn
        if not isinstance(attention, Qwen3_5Attention):
            raise ValueError(f"Unsupported full-attention layer {i}")
        heads[i] = new_indexer(torch, attention.q_proj.in_features, attention.q_proj.weight.device)
    params = list(itertools.chain.from_iterable(head.parameters() for head in heads.values()))
    initial = [p.detach().to("cpu", copy=True) for p in params]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)

    attention_backend = Distiller(torch, F, heads, args.query_chunk, math.ceil(args.budget / BLOCK))
    ALL_ATTENTION_FUNCTIONS.register(ATTENTION, attention_backend.attention)
    text.config._attn_implementation = ATTENTION
    for i in expected:
        text.layers[i].self_attn.register_forward_pre_hook(attention_backend.input, with_kwargs=True)

    def run(ids, training):
        return attention_backend.run(text, ids, training)

    before = evaluate(run, chunks(args.eval_data))
    train_iterator = iter(chunks(args.train_data))
    train_total = Score()
    for step in range(args.steps):
        ids = next(train_iterator, None)
        if ids is None:
            train_iterator = iter(chunks(args.train_data))
            ids = next(train_iterator, None)
        if ids is None:
            raise ValueError("Training corpus has fewer than eight tokens")
        optimizer.zero_grad(set_to_none=True)
        result = run(ids, training=True)
        train_total.add(result)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
            raise RuntimeError("The indexer gradient is missing or non-finite")
        optimizer.step()
        print(json.dumps({"step": step + 1, "steps": args.steps, "train_loss": result.report()["kl"]}),
              flush=True)
    after = evaluate(run, chunks(args.eval_data))
    if any(p.requires_grad or p.grad is not None or p._version != version
           for p, version in teacher_parameters.values()):
        raise RuntimeError("Teacher parameters did not stay frozen")
    if digest(args.train_data) != train_hash or digest(args.eval_data) != eval_hash:
        raise RuntimeError("A corpus changed during training")
    update = math.sqrt(sum((p.detach().cpu() - start).square().sum().item()
                           for p, start in zip(params, initial, strict=True)))
    if not math.isfinite(update) or update <= 0:
        raise RuntimeError("Training did not produce a finite, nonzero indexer update")

    weights = {f"model.language_model.layers.{i}.self_attn._qsa_indexer.{name}":
               tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
               for i, head in heads.items() for name, tensor in head.state_dict().items()}
    if any(not torch.isfinite(weight).all() for weight in weights.values()):
        raise RuntimeError("Non-finite indexer weights cannot be exported")
    metrics = {
        "status": "passed", "steps": args.steps, "seq_len": args.seq_len, "seed": args.seed, "lr": args.lr,
        "train_loss": train_total.report()["kl"], "heldout_loss": after["kl"], "before": before, "after": after,
        "full_attention_layers": expected, "budget": args.budget, "indexer_update_l2": update,
        "source_config_sha256": source_config_hash, "train_data_sha256": train_hash, "eval_data_sha256": eval_hash,
    }
    metadata = {"format": "pt", "teacher": TEACHER, "reference_revision": REFERENCE_REVISION,
                "source_config_sha256": source_config_hash}
    serialized = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    tmp = output / ".indexers-building"
    with tmp.open("xb"):
        pass
    save_file(weights, tmp, metadata=metadata)
    tmp.rename(output / "qsa-indexers.safetensors")
    with (output / "metrics.json").open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def probe(torch, F):
    if not torch.cuda.is_available():
        raise RuntimeError("No local CUDA or ROCm GPU is available to train QSA")
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    config = Qwen3_5TextConfig(
        hidden_size=32, num_hidden_layers=1, layer_types=["linear_attention"],
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2,
    )
    for i in range(torch.cuda.device_count()):
        device = torch.device("cuda", i)
        q = torch.ones((1, 2, 8, 256), device=device, dtype=torch.bfloat16)
        k = torch.ones((1, 1, 8, 256), device=device, dtype=torch.bfloat16)
        dense_attention(torch, F, q, k, k, 256**-0.5, 3)
        target = block_target(torch, q, k, 256**-0.5, 3, 6)
        x = torch.ones((1, 8, 32), device=device)
        with torch.no_grad():
            gdn = Qwen3_5GatedDeltaNet(config, 0).to(device=device, dtype=torch.bfloat16)
            gdn(x.bfloat16())
        indexer = new_indexer(torch, 32, device)
        cos = torch.ones((1, 8, ROPE), device=device)
        scores = indexer.scores(x[:, 3:6], cos[:, 3:6], cos[:, 3:6],
                                indexer.keys(x, cos, cos)[:, :target.shape[-1]])
        F.kl_div(scores.log_softmax(-1), target, reduction="sum").backward()
        torch.optim.AdamW(indexer.parameters(), foreach=False).step()
        torch.cuda.synchronize(i)
    print(json.dumps({"schema": "brewery.qsa-probe.v1", "ready": True}), flush=True)


def corpus(tokenizer, path, length):
    """Pack independent UTF-8 lines with EOS separators; do not pad or cache."""
    ids = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            tokens = tokenizer.encode(line.rstrip("\r\n"), add_special_tokens=False, verbose=False)
            if tokenizer.eos_token_id is not None:
                tokens.append(tokenizer.eos_token_id)
            cursor = 0
            while cursor < len(tokens):
                count = min(length - len(ids), len(tokens) - cursor)
                ids.extend(tokens[cursor:cursor + count])
                cursor += count
                if len(ids) == length:
                    yield ids
                    ids = []
    if len(ids) >= 2 * BLOCK:
        yield ids


def reject_packed_overlap(train, heldout):
    """Keep only fixed-size held-out identities, and stream all training batches."""
    def token_identity(ids):
        return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).digest()

    identities = set()
    for ids in heldout:
        identities.add(token_identity(ids))
        if len(identities) > MAX_HELDOUT_EXAMPLE_HASHES:
            raise ValueError(f"Held-out corpus exceeds {MAX_HELDOUT_EXAMPLE_HASHES} unique packed token sequences")
    if not identities:
        raise ValueError("Held-out corpus has fewer than eight tokens")
    for ids in train:
        if token_identity(ids) in identities:
            raise ValueError("The training and held-out corpora share a packed token sequence")


@dataclass
class Score:
    kl: float = 0.0
    mass: float = 0.0
    queries: int = 0
    capture_queries: int = 0

    def add(self, other):
        self.kl += other.kl
        self.mass += other.mass
        self.queries += other.queries
        self.capture_queries += other.capture_queries

    def report(self):
        if not self.queries:
            raise ValueError("No scoreable queries; the corpus needs at least eight tokens")
        if not math.isfinite(self.kl) or not math.isfinite(self.mass):
            raise RuntimeError("Non-finite QSA distillation metrics")
        result = {"kl": max(0.0, self.kl / self.queries), "capture": None, "queries": self.queries,
                  "capture_queries": self.capture_queries}
        if self.capture_queries:
            result["capture"] = max(0.0, min(1.0, self.mass / self.capture_queries))
        else:
            result["capture_unavailable_reason"] = (
                "No held-out query has more fully observed blocks than the capture budget."
            )
        return result


def evaluate(run, batches):
    score = Score()
    for ids in batches:
        score.add(run(ids, training=False))
    return score.report()


def new_indexer(torch, hidden_size, device):
    from torch import nn

    def norm(x, offset):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + offset)

    def rotate(x, cos, sin):
        left, right = x[..., :ROPE].chunk(2, dim=-1)
        twist = torch.cat((-right, left), dim=-1)
        return torch.cat((x[..., :ROPE] * cos + twist * sin, x[..., ROPE:]), -1)

    class Indexer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden_size, HEADS * WIDTH, bias=False, dtype=torch.float32)
            self.k_proj = nn.Linear(hidden_size, WIDTH, bias=False, dtype=torch.float32)
            self.q_norm = nn.Parameter(torch.zeros(WIDTH, dtype=torch.float32))
            self.k_norm = nn.Parameter(torch.zeros_like(self.q_norm))
            self.k_block_norm = nn.Parameter(torch.zeros_like(self.q_norm))

        def keys(self, hidden, cos, sin):
            end = hidden.shape[1] // BLOCK * BLOCK
            keys = norm(self.k_proj(hidden[:, :end]), self.k_norm)
            pooled = keys.reshape(1, -1, BLOCK, WIDTH).mean(dim=2)
            return rotate(norm(pooled, self.k_block_norm), cos[:, :end:BLOCK], sin[:, :end:BLOCK])

        def scores(self, hidden, cos, sin, keys):
            q = self.q_proj(hidden).reshape(1, -1, HEADS, WIDTH)
            q = rotate(norm(q, self.q_norm), cos.unsqueeze(2), sin.unsqueeze(2))
            return torch.einsum("bqhd,bkd->bhqk", q, keys).relu().sum(1) * WIDTH**-0.5

    return Indexer().to(device)


def expand_kv(tensor, heads):
    return tensor.repeat_interleave(heads // tensor.shape[1], dim=1)


def causal(torch, start, stop, device):
    return torch.arange(stop, device=device)[None, :] <= torch.arange(start, stop, device=device)[:, None]


def dense_attention(torch, F, q, k, v, scale, chunk):
    """Preserve real dense attention outputs without a quadratic fallback."""
    k, v = expand_kv(k, q.shape[1]), expand_kv(v, q.shape[1])
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if q.is_cuda and torch.backends.cuda.can_use_flash_attention(
        torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, True, False)
    ):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    else:
        out = torch.empty_like(q)
        for start in range(0, q.shape[2], chunk):
            stop = min(start + chunk, q.shape[2])
            out[:, :, start:stop] = F.scaled_dot_product_attention(
                q[:, :, start:stop], k[:, :, :stop], v[:, :, :stop],
                attn_mask=causal(torch, start, stop, q.device), scale=scale,
            )
    return out.transpose(1, 2).contiguous()


def block_target(torch, q, expanded_k, scale, start, stop):
    """Dense teacher: causal token softmax, sum heads, max pool each complete block."""
    key = expanded_k if expanded_k.shape[1] == q.shape[1] else expand_kv(expanded_k, q.shape[1])
    logits = (q[:, :, start:stop] @ key[:, :, :stop].transpose(-1, -2)) * scale
    logits.masked_fill_(~causal(torch, start, stop, q.device), -torch.inf)
    mass = logits.softmax(-1, dtype=torch.float32).sum(dim=1)
    blocks = stop // BLOCK
    pooled = mass[:, :, :blocks * BLOCK].reshape(1, stop - start, blocks, BLOCK).amax(-1)
    starts = torch.arange(blocks, device=q.device)[None, :] * BLOCK + BLOCK - 1
    positions = torch.arange(start, stop, device=q.device)[:, None]
    pooled.masked_fill_(starts > positions, 0.0)
    return pooled / pooled.sum(-1, keepdim=True)


class Distiller:
    def __init__(self, torch, functional, indexers, chunk, top_blocks):
        self.torch, self.F = torch, functional
        self.indexers, self.chunk, self.top_blocks = indexers, chunk, top_blocks
        self.pending = {}

    def input(self, module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        rotary = kwargs["position_embeddings"]
        self.pending[module.layer_idx] = hidden, rotary

    def run(self, text, ids, training):
        self.training = training
        self.score = Score()
        self.visited = set()
        with self.torch.no_grad():
            text(
                input_ids=self.torch.tensor([ids], device=text.embed_tokens.weight.device),
                # Single unpadded text sequence, with independent positions and no cache.
                # Causality is applied inside attention, not by a full T×T tensor.
                attention_mask={"full_attention": None, "linear_attention": None},
                use_cache=False,
            )
        if self.visited != set(self.indexers) or self.pending:
            raise RuntimeError("The frozen forward did not observe each full-attention layer exactly once")
        return self.score

    def attention(self, module, q, k, v, mask, *, scaling, dropout, **kwargs):
        if mask is not None or dropout or module.layer_idx in self.visited:
            raise RuntimeError("Expected one unpadded, causal, no-dropout teacher pass per layer")
        hidden, (cos, sin) = self.pending.pop(module.layer_idx)
        if cos.shape[-1] != ROPE or sin.shape != cos.shape:
            raise ValueError("The Qwen text rotary embedding must supply 64 indexer channels")
        self.distill(module.layer_idx, hidden, cos, sin, q, k, scaling)
        self.visited.add(module.layer_idx)
        return dense_attention(self.torch, self.F, q, k, v, scaling, self.chunk), None

    def distill(self, i, hidden, cos, sin, q, k, scale):
        torch = self.torch
        indexer = self.indexers[i]
        # Teacher and hidden states remain detached even while indexer chunks backpropagate.
        h, cos, sin = hidden.detach().float(), cos.detach().float(), sin.detach().float()
        key = expand_kv(k, q.shape[1])
        seq = q.shape[2]
        count = seq - BLOCK + 1
        kl_sum = torch.zeros((), device=q.device)
        mass_sum = torch.zeros_like(kl_sum)
        first_competitive = (self.top_blocks + 1) * BLOCK - 1
        with torch.set_grad_enabled(self.training):
            projected_keys = indexer.keys(h, cos, sin)
            # Accumulate one leaf gradient across chunks, then propagate through K once.
            keys = projected_keys.detach().requires_grad_(self.training)
            for start in range(BLOCK - 1, seq, self.chunk):
                stop = min(start + self.chunk, seq)
                target = block_target(torch, q, key, scale, start, stop)
                rows, columns = stop - start, stop // BLOCK
                scores = indexer.scores(h[:, start:stop], cos[:, start:stop], sin[:, start:stop],
                                        keys[:, :columns])
                ends = torch.arange(columns, device=q.device) * BLOCK + BLOCK - 1
                valid = ends[None, :] <= torch.arange(start, stop, device=q.device)[:, None]
                scores = scores.masked_fill(~valid, -torch.inf)
                logp = scores.log_softmax(-1).masked_fill(~valid, 0.0)
                kl = self.F.kl_div(logp, target, reduction="sum")
                if self.training:
                    (kl / count).backward()
                with torch.no_grad():
                    kl_sum.add_(kl.detach())
                    if not self.training and stop > first_competitive:
                        offset = max(0, first_competitive - start)
                        selected = scores[:, offset:].topk(self.top_blocks, dim=-1).indices
                        mass_sum.add_(target[:, offset:].gather(-1, selected).sum())
            if self.training:
                projected_keys.backward(keys.grad)
        competitive = 0 if self.training else max(0, seq - first_competitive)
        self.score.add(Score(kl_sum.item(), mass_sum.item(), count, competitive))


if __name__ == "__main__":
    try:
        main()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        sys.exit(f"QSA training failed: {error}")
