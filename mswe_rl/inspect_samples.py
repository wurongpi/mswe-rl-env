"""Show what the trainer will see: tokens colored by loss mask.

    uv run mswe-inspect runs/samples.jsonl --tokenizer Qwen/Qwen3-1.7B
    uv run mswe-inspect runs/samples.jsonl --tokenizer toy        # fake-server runs

green  = sampled by the policy (loss_mask=1, carries rollout logprob)
gray   = prompt / tool output / env text (loss_mask=0)
"""

from __future__ import annotations

import argparse
import json
import sys

GREEN, GRAY, BOLD, END = "\033[32m", "\033[90m", "\033[1m", "\033[0m"


def load_tokenizer(name: str):
    if name == "toy":
        from .toy_tokenizer import ToyTokenizer

        return ToyTokenizer().decode
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    return lambda ids: tok.decode(ids, skip_special_tokens=False)


def runs(mask: list[int]):
    start = 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            yield start, i, mask[start]
            start = i


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("samples")
    p.add_argument("--tokenizer", default="toy")
    p.add_argument("--index", type=int, default=0, help="which sample to render")
    p.add_argument("--max-chars", type=int, default=220, help="truncate each run for display")
    p.add_argument("--color", action="store_true")
    a = p.parse_args(argv)
    color = a.color or sys.stdout.isatty()
    decode = load_tokenizer(a.tokenizer)
    rows = [json.loads(line) for line in open(a.samples)]
    s = rows[a.index]

    n_tok, n_train = len(s["tokens"]), sum(s["loss_mask"])
    trained = [lp for lp, m in zip(s["rollout_logprobs"], s["loss_mask"]) if m]
    mean_lp = sum(trained) / len(trained) if trained else 0.0
    head = (f"episode {s['episode_id']}  segment {s['segment_index'] + 1}/{s['num_segments']}  "
            f"reward={s['reward']}  tokens={n_tok}  trained={n_train} ({n_train / n_tok:.0%})  "
            f"mean logprob={mean_lp:.3f}")
    print(f"{BOLD}{head}{END}" if color else head)
    for st, en, m in runs(s["loss_mask"]):
        text = decode(s["tokens"][st:en])
        if len(text) > a.max_chars:
            text = text[: a.max_chars // 2] + f" …[{len(text) - a.max_chars} chars]… " + text[-a.max_chars // 2:]
        label = f"[mask={m} · {en - st} tok]"
        if color:
            print(f"{GREEN if m else GRAY}{label}{END}")
            print(f"{GREEN if m else GRAY}{text}{END}")
        else:
            print(label)
            print(text)


if __name__ == "__main__":
    main()
