from mswe_rl.report import build_report


def ep(task, status, reward, breaks=0, calls=3):
    return {
        "task": task,
        "status": status,
        "reward": reward,
        "num_calls": calls,
        "prefix_breaks": [{}] * breaks,
        "duration_s": 2.0,
    }


def test_reward_signal_counts_mixed_groups_only():
    episodes = [
        ep("a", "pass", 1.0),
        ep("a", "fail", 0.0),  # mixed: the only group GRPO learns from
        ep("b", "pass", 1.0),
        ep("b", "pass", 1.0),  # all-pass
        ep("c", "fail", 0.0),
        ep("c", "fail", 0.0, breaks=1),  # all-fail
        ep("d", "env_error", None),
        ep("d", "fail", 0.0),  # one valid episode: not a group
    ]
    samples = [{"tokens": [1, 2, 3, 4], "loss_mask": [0, 0, 1, 1], "logprobs_complete": True}]
    r = build_report(episodes, samples, {"a": "operator misuse", "b": "operator misuse"})
    assert r["episodes"] == 8 and r["valid_episodes"] == 7
    assert r["env_error_rate"] == 1 / 8
    assert r["pass_rate"] == 3 / 7
    assert (r["groups"], r["groups_mixed"], r["groups_all_pass"], r["groups_all_fail"]) == (3, 1, 1, 1)
    assert r["tasks_solved_at_least_once"] == 2
    assert r["episodes_with_prefix_break"] == 1 and r["prefix_breaks_per_call"] == 1 / 21
    assert r["trained_token_fraction"] == 0.5
    assert r["pass_rate_by_bug_type"]["operator misuse"] == {"episodes": 4, "pass_rate": 0.75}


def test_empty_rollout_does_not_crash():
    r = build_report([], [])
    assert r["pass_rate"] == 0.0 and r["groups"] == 0
