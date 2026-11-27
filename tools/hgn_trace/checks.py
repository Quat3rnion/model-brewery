"""Compare last-position full-vocabulary logits of two stored HGN files."""
import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from trace import Forward, encode, read_config, setup_threads


def logits(model, sidecar, config, prompts):
    forward = Forward(model, sidecar, config)
    rows = []
    for tokens in prompts:
        for layer, hidden in forward.layers(tokens):
            if layer == config.num_hidden_layers - 1:
                rows.append(forward.mixer(hidden[-1:]))
    if len(rows) != len(prompts):
        raise ValueError("missing final HGN residual")
    scores = torch.cat([chunk.double() for _, chunk in forward.head(torch.cat(rows))], dim=-1)
    if scores.shape != (len(prompts), config.vocab_size) or not torch.isfinite(scores).all():
        raise ValueError("invalid HGN logits")
    return scores


def compare(parent, child, tokens):
    lp, lq = parent.log_softmax(-1), child.log_softmax(-1)
    kl = float((lp.exp() * (lp - lq)).sum())
    delta = child - parent
    top = int(parent.argmax())
    return dict(
        tokens=tokens, kl_parent_child=max(0., kl), top1_agrees=top == int(child.argmax()),
        mean_abs_logit_delta=float(delta.abs().mean()), max_abs_logit_delta=float(delta.abs().max()),
        parent_top1_logit_delta=float(delta[top]),
    )


@torch.inference_mode()
def run(args):
    setup_threads(args.threads)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokens = [tokenizer.encode(prompt, add_special_tokens=False).ids for prompt in args.prompt]
    config = read_config(args.config)
    parent = logits(args.parent, args.sidecar, config, tokens)
    child = logits(args.child, args.sidecar, config, tokens)
    scores = [compare(p, c, len(t)) for p, c, t in zip(parent, child, tokens, strict=True)]
    with args.out.open("xb") as output:
        output.write(encode(scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("parent", "child", "sidecar", "config", "tokenizer", "out"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--threads", type=int, required=True)
    run(parser.parse_args())
