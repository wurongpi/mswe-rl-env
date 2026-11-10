"""Chat template handling for token-in, token-out rollouts.

Re-rendering a whole conversation with the chat template on every call is not
prefix-stable: templates rewrite earlier turns. Qwen3, for example, puts an empty
``<think>`` block into the generation prompt but leaves it out of assistant turns
that are already in the history, so the tokens the model saw on turn k differ from
how turn k is rendered on turn k+1. Measured with Qwen3-1.7B: 14 of 15 calls broke
the token prefix.

The recording proxy therefore keeps each episode's token sequence itself and only
renders the *new* messages of every turn (``continuation``), appending them to the
tokens the model actually consumed and produced.
"""

from __future__ import annotations

from typing import Any

# Placeholder conversation the new messages are rendered after. Its assistant turn
# is followed by the new messages, so the template renders it as a history turn,
# which is the role the real previous reply now plays.
_ANCHOR_USER = "⁣mswe-anchor-user⁣"
_ANCHOR_ASSISTANT = "⁣mswe-anchor-assistant⁣"
_ANCHOR = [
    {"role": "user", "content": _ANCHOR_USER},
    {"role": "assistant", "content": _ANCHOR_ASSISTANT},
]


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Keep role and text only (litellm-style clients may send content as a list of parts)."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        out.append({"role": m["role"], "content": content or ""})
    return out


class ChatTemplate:
    """Wraps a Hugging Face tokenizer (or the toy tokenizer used in tests)."""

    def __init__(self, tokenizer, template_kwargs: dict[str, Any] | None = None):
        self.tok = tokenizer
        self.template_kwargs = template_kwargs or {}
        eos = getattr(tokenizer, "eos_token_id", None)
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}

    def render(self, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, **self.template_kwargs
        )

    def encode(self, text: str) -> list[int]:
        # The rendered text already contains any BOS the template wants.
        return list(self.tok.encode(text, add_special_tokens=False))

    def prompt_ids(self, messages: list[dict[str, str]]) -> list[int]:
        """Token ids of a conversation rendered from scratch (first call of an episode)."""
        return self.encode(self.render(messages, add_generation_prompt=True))

    def continuation(self, new_messages: list[dict[str, str]], last_token: int) -> list[int]:
        """Tokens to append after the previous completion so the model sees ``new_messages``.

        Renders anchor + new messages and keeps the text after the anchor reply. That
        text starts with the template's end-of-turn marker; if the model already
        generated it (the completion ended with EOS) it is dropped, otherwise (the
        reply was cut off by max_tokens) it is kept to close the turn.
        """
        full = self.render(_ANCHOR + new_messages, add_generation_prompt=True)
        start = full.find(_ANCHOR_ASSISTANT)
        if start < 0:
            raise ValueError("chat template dropped the anchor reply; cannot compute the continuation")
        tail = full[start + len(_ANCHOR_ASSISTANT) :]
        if last_token in self.eos_ids:
            eos_text = self.tok.decode([last_token], skip_special_tokens=False)
            if tail.startswith(eos_text):
                tail = tail[len(eos_text) :]
        return self.encode(tail)


def load_chat_template(name: str, enable_thinking: bool = False) -> ChatTemplate:
    """``name`` is a Hugging Face repo id / local directory, or ``toy`` / ``toy-think`` for tests."""
    if name in ("toy", "toy-think"):
        from .testing.toy_tokenizer import ToyTokenizer

        return ChatTemplate(ToyTokenizer(think_style=name == "toy-think"))
    from transformers import AutoTokenizer

    # enable_thinking is a Qwen3 template variable; other templates ignore unknown variables.
    return ChatTemplate(AutoTokenizer.from_pretrained(name), {"enable_thinking": enable_thinking})
