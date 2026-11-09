"""Byte-level tokenizer with ChatML special tokens. Only for the fake server and tests."""

from __future__ import annotations

IM_START, IM_END = 256, 257
SPECIAL = {"<|im_start|>": IM_START, "<|im_end|>": IM_END}
SPECIAL_INV = {v: k for k, v in SPECIAL.items()}


class ToyTokenizer:
    def encode(self, text: str) -> list[int]:
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

    def decode(self, ids: list[int]) -> str:
        buf, parts = bytearray(), []
        for t in ids:
            if t in SPECIAL_INV:
                parts.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
                parts.append(SPECIAL_INV[t])
            else:
                buf.append(t)
        parts.append(buf.decode("utf-8", errors="replace"))
        return "".join(parts)

    def apply_chat_template(self, messages, add_generation_prompt: bool = True) -> list[int]:
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self.encode(text)
