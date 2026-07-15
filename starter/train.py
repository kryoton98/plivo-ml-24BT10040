"""CPU-only speedrun trainer.

Strict rules encoded here:
  * exactly 2,000 optimizer steps (hard stop, checkpoint saved immediately after)
  * pure CPU (no CUDA / AMP)
  * gradient accumulation: micro-batch 8, accumulated over 8 micro-steps
    -> effective batch size 64 per optimizer step
  * AdamW(betas=(0.9, 0.95)), weight decay 0.1 on 2D matrices only
  * cosine LR schedule: linear warmup for the first 200 steps (10%) up to
    4e-3, then cosine decay to 10% of peak (4e-4) by step 2000
  * grad-norm clipping to 1.0 immediately before every optimizer.step()
  * logs step / accumulation-scaled loss / lr / elapsed time every 50 steps

    python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
"""
import argparse
import math
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000

MICRO_BATCH = 8
ACCUM_STEPS = 8          # effective batch = MICRO_BATCH * ACCUM_STEPS = 64
MAX_LR = 4e-3
WARMUP_STEPS = 200       # 10% of 2000
MIN_LR_RATIO = 0.1
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def batch_generator(ids, block_size, micro_batch, device):
    """Yields an endless stream of (X, Y) chunks of length `block_size`,
    each sampled from a random starting index in the tokenized corpus."""
    n = len(ids)
    hi = n - block_size - 1
    while True:
        ix = torch.randint(0, hi, (micro_batch,))
        x = torch.stack([ids[i:i + block_size] for i in ix])
        y = torch.stack([ids[i + 1:i + 1 + block_size] for i in ix])
        yield x.to(device), y.to(device)


# --------------------------------------------------------------------------- #
# LR schedule: linear warmup -> cosine decay to MIN_LR_RATIO * MAX_LR
# --------------------------------------------------------------------------- #
def get_lr(step):
    if step < WARMUP_STEPS:
        return MAX_LR * step / WARMUP_STEPS
    if step >= MAX_STEPS:
        return MAX_LR * MIN_LR_RATIO
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    min_lr = MAX_LR * MIN_LR_RATIO
    return min_lr + coeff * (MAX_LR - min_lr)


# --------------------------------------------------------------------------- #
# Optimizer: weight decay only on 2D matrices, never on norms/biases
# --------------------------------------------------------------------------- #
def build_optimizer(model):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    print(f"optimizer: {len(decay)} decayed tensors, {len(no_decay)} non-decayed tensors")
    return torch.optim.AdamW(groups, lr=MAX_LR, betas=BETAS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=MAX_STEPS)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    opt = build_optimizer(model)
    data_iter = batch_generator(ids, cfg.block_size, MICRO_BATCH, device)

    model.train()
    t0 = time.time()
    losses = []
    step = 0

    while True:
        step += 1
        lr = get_lr(step)
        for group in opt.param_groups:
            group["lr"] = lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(ACCUM_STEPS):
            x, y = next(data_iter)
            _, loss = model(x, y)
            scaled = loss / ACCUM_STEPS
            scaled.backward()
            accum_loss += scaled.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        losses.append(accum_loss)

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"step {step:5d}/{args.steps}  loss {accum_loss:.4f}  "
                  f"lr {lr:.6f}  grad_norm {grad_norm:.3f}  "
                  f"elapsed {elapsed:.1f}s  ({elapsed/step*1000:.0f} ms/step)")

        if step == args.steps:
            break

    # every public config attribute is saved — if you add fields to Config,
    # they ride along automatically and evaluate.py rebuilds the same model
    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "step": step,
                "steps": step,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out} at step {step}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()