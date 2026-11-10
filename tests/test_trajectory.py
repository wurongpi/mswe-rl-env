from mswe_rl.schema import LLMCall
from mswe_rl.trajectory import assemble_segments


def call(i, prompt, comp, lps=True):
    return LLMCall(
        session_id="s",
        index=i,
        messages=[],
        prompt_token_ids=prompt,
        completion_token_ids=comp,
        completion_logprobs=[-0.5] * len(comp) if lps else None,
        text="",
    )


def test_consistent_prefix_is_one_segment():
    c0 = call(0, [1, 2, 3], [10, 11])
    c1 = call(1, [1, 2, 3, 10, 11, 20, 21], [12])  # 20, 21 = tool output appended by the harness
    segs, breaks = assemble_segments([c0, c1])
    assert len(segs) == 1 and breaks == []
    s = segs[0]
    assert s.tokens == [1, 2, 3, 10, 11, 20, 21, 12]
    assert s.loss_mask == [0, 0, 0, 1, 1, 0, 0, 1]
    assert s.logprobs == [0, 0, 0, -0.5, -0.5, 0, 0, -0.5]
    assert s.prompt_length == 3 and s.response_length == 5


def test_truncated_history_starts_new_segment():
    c0 = call(0, [1, 2, 3], [10])
    c1 = call(1, [1, 2, 3, 10, 20], [11])
    c2 = call(2, [1, 2, 20, 11, 30], [12])  # turn dropped from the middle
    segs, breaks = assemble_segments([c0, c1, c2])
    assert len(segs) == 2
    assert breaks[0]["at_call"] == 2 and breaks[0]["mismatch_pos"] == 2
    assert segs[1].tokens[:5] == [1, 2, 20, 11, 30] and segs[1].loss_mask[-1] == 1


def test_calls_are_ordered_by_index():
    c0 = call(0, [1], [2])
    c1 = call(1, [1, 2, 3], [4])
    segs, _ = assemble_segments([c1, c0])
    assert segs[0].tokens == [1, 2, 3, 4]


def test_missing_or_misaligned_logprobs_are_flagged():
    c0 = call(0, [1], [2, 3], lps=False)
    bad = call(1, [1, 2, 3], [4, 5])
    bad.completion_logprobs = [-1.0]  # wrong length -> dropped
    segs, _ = assemble_segments([c0, bad])
    assert segs[0].logprobs_complete is False
    assert segs[0].logprobs == [0.0] * 5
