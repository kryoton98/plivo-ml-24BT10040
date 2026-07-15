"""Score a checkpoint: BITS PER BYTE on a text file. Lower is better.

Why per BYTE and not per token: your tokenizer choice changes what a
"token" is, so per-token loss is not comparable across submissions. Bytes
of the original text are the same for everyone.

This exact command is what the hidden evaluation runs inside your
submission folder — keep it working no matter how you change the model or
tokenizer:

    python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt

Prints one JSON line: {"bpb": ..., "n_params": ..., "steps": ...}
Evaluation uses a sliding window with 50% context carry-over, so every
token except the very first is predicted with real left context.
"""
import argparse
import json
import math

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = Config()
    for k, v in ckpt["config"].items():
        setattr(cfg, k, v)
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def bits_per_byte(model, cfg, tok, text):
    """Every token except the very first is scored exactly once, with at
    least block/2 tokens of real left context (sliding window)."""
    n_bytes = len(text.encode("utf-8"))
    assert n_bytes > 0, "eval text is empty"
    id_list = tok.encode(text)
    if tok.decode(id_list) != text:
        raise SystemExit(
            "tokenizer is not lossless: decode(encode(text)) != text. "
            "bpb over the full byte count would be meaningless — and the "
            "graders run this same round-trip on the hidden file.")
    ids = torch.tensor(id_list, dtype=torch.long)
    assert cfg.block_size >= 2, "block_size must be >= 2 to score anything"
    block, stride = cfg.block_size, max(1, cfg.block_size // 2)
    total_nll, n_scored = 0.0, 0
    scored = 1                       # absolute index of next target to score
    while scored < len(ids):
        start = max(0, scored - stride)          # context before fresh region
        end = min(len(ids), start + block)
        window = ids[start:end]
        logits, _ = model(window[None, :])
        logp = torch.log_softmax(logits[0], dim=-1)
        targets = ids[start + 1:end]             # predicted by positions start..end-2
        nll = -logp[torch.arange(len(targets)), targets]
        offset = scored - (start + 1)            # skip already-scored targets
        assert offset >= 0
        total_nll += nll[offset:].sum().item()
        n_scored += len(nll) - offset
        scored = end
    assert n_scored > 0, "nothing scored: tokenizer produced <2 tokens"
    return total_nll / math.log(2) / n_bytes, n_scored, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt.pt")
    ap.add_argument("--text_file", required=True)
    args = ap.parse_args()
    model, cfg, ckpt = load_model(args.checkpoint)
    tok = tokenizer_mod.load()
    text = open(args.text_file, encoding="utf-8").read()
    bpb, n_scored, n_tokens = bits_per_byte(model, cfg, tok, text)
    print(json.dumps({
        "bpb": round(bpb, 4),
        "n_params": model.n_params(),
        "steps": ckpt.get("steps"),
        "tokens_in_eval": n_tokens,
        "tokens_scored": n_scored,
    }))


if __name__ == "__main__":
    main()
