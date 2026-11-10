"""Byte-level tokenizer with ChatML special tokens. Only for the fake server and tests.

It has the subset of the Hugging Face tokenizer interface the recording proxy uses
(``apply_chat_template(tokenize=False)``, ``encode``, ``decode``, ``eos_token_id``).

``think_style=True`` mimics Qwen3 with thinking disabled: the generation prompt ends
with an empty ``<think>`` block, but assistant turns already in the history are
rendered without it. Re-rendering the history therefore changes earlier tokens,
which is exactly what broke the token prefix on every turn with the real model.
"""

from __future__ import annotations

IM_START, IM_END = 256, 257
SPECIAL = {"<|im_start|>": IM_START, "<|im_end|>": IM_END}
SPECIAL_INV = {v: k for k, v in SPECIAL.items()}
EMPTY_THINK = "<think>\n\n</think>\n\n"


class ToyTokenizer:
    eos_token_id = IM_END

    def __init__(self, think_style: bool = False):
        self.think_style = think_style

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            for s, tid in SPECIAL.items():
                if text.startswith(s, i):
                    out.append(tid)
                    i += len(s)
                    break
            else:
                j = i
                while j < len(text) and not any(text.startswith(s, j) for s in SPECIAL):
                    j += 1
                out.extend(text[i:j].encode("utf-8"))
                i = j
        return out

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        buf, parts = bytearray(), []
        for t in ids:
            if t in SPECIAL_INV:
                parts.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
                if not skip_special_tokens:
                    parts.append(SPECIAL_INV[t])
            else:
                buf.append(t)
        parts.append(buf.decode("utf-8", errors="replace"))
        return "".join(parts)

    def apply_chat_template(
        self, messages, tokenize: bool = False, add_generation_prompt: bool = True, **_
    ) -> str | list[int]:
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n" + (EMPTY_THINK if self.think_style else "")
        return self.encode(text) if tokenize else text
