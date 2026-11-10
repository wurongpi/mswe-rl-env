from mswe_rl.chat_template import load_chat_template, normalize_messages
from mswe_rl.testing.toy_tokenizer import IM_END

SYSTEM = {"role": "system", "content": "be brief"}
USER = {"role": "user", "content": "fix calc.py"}
REPLY = "THOUGHT: look first"
OBS = {"role": "user", "content": "<output>def add(a, b): return a - b</output>"}


def _first_turn(template):
    prompt = template.prompt_ids([SYSTEM, USER])
    completion = template.encode(REPLY) + [IM_END]  # what the model sampled
    return prompt + completion


def test_qwen3_style_template_is_not_prefix_stable():
    """The bug this module exists for: re-rendering rewrites the earlier turn."""
    t = load_chat_template("toy-think")
    seen = _first_turn(t)
    rerendered = t.prompt_ids([SYSTEM, USER, {"role": "assistant", "content": REPLY}, OBS])
    assert rerendered[: len(seen)] != seen


def test_continuation_equals_a_fresh_render_when_the_template_is_stable():
    t = load_chat_template("toy")
    seen = _first_turn(t)
    extended = seen + t.continuation([OBS], last_token=seen[-1])
    assert extended == t.prompt_ids([SYSTEM, USER, {"role": "assistant", "content": REPLY}, OBS])


def test_continuation_keeps_what_the_model_saw_with_a_rewriting_template():
    t = load_chat_template("toy-think")
    seen = _first_turn(t)
    extended = seen + t.continuation([OBS], last_token=seen[-1])
    text = t.tok.decode(extended)
    # the empty think block of turn 1 stays, and the new turn is opened the template's way
    assert text.count("<think>") == 2
    assert text.endswith(
        "<|im_end|>\n<|im_start|>user\n"
        + OBS["content"]
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_cut_off_reply_gets_its_turn_closed():
    t = load_chat_template("toy")
    prompt = t.prompt_ids([USER])
    truncated = prompt + t.encode("THOUGHT: I was cut of")  # finish_reason=length, no EOS
    cont = t.continuation([OBS], last_token=truncated[-1])
    assert cont[0] == IM_END


def test_normalize_messages_joins_content_parts():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "x": 1}
    ]
    assert normalize_messages(msgs) == [{"role": "user", "content": "ab"}]
