"""Unit test for the streaming repetition-penalty fix in _sample_next_token.

Pure-logits test (no model weights / GPU): verifies that penalizing an already-generated token demotes
it below a competitor, which is what stops the streaming talker from looping and over-generating audio.
Run on the server env (needs torch): pytest tests/test_streaming_repetition_penalty.py
"""
import torch

from qwen_tts.core.models.modeling_qwen3_tts import _sample_next_token


def test_repetition_penalty_demotes_repeated_token():
    # Token 2 is the argmax (2.0), token 3 is the runner-up (1.9).
    logits = torch.tensor([[1.0, 0.0, 2.0, 1.9]])

    # No penalty -> the model would re-pick token 2 (the looping behaviour).
    assert _sample_next_token(logits, temperature=0.0).item() == 2

    # With token 2 already generated and penalty 2.0: 2.0 / 2.0 = 1.0 < 1.9 -> picks token 3 instead.
    out = _sample_next_token(
        logits, temperature=0.0, repetition_penalty=2.0, prev_tokens=torch.tensor([2])
    )
    assert out.item() == 3

    # penalty == 1.0 (or no history) is a no-op -> still token 2.
    assert _sample_next_token(
        logits, temperature=0.0, repetition_penalty=1.0, prev_tokens=torch.tensor([2])
    ).item() == 2
    assert _sample_next_token(logits, temperature=0.0, repetition_penalty=2.0).item() == 2
