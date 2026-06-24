# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Streaming-text TTS session (approach B) — one continuous synthesis driven by text that
arrives incrementally (e.g. from a streaming LLM), read back as one PCM stream.

This is the server side of the wabi `/v1/audio/stream` WebSocket contract. The mechanism
was validated by two standalone prototypes (examples/prototype*_streaming*.py):

  * append-mid-generation: the talker conditions on trailing_text_hidden[generation_step]
    (one row per decoded frame); revealing rows LATE reproduces a whole-text render, as
    long as each row is in place before generation_step indexes it (slice 1, bit-exact).
  * stable-prefix tokenization: tokenize the CUMULATIVE text each chunk and append only
    newly-stable rows (hold back the last few, which may re-merge) so the talker sees the
    same token granularity as a joint render — independent per-chunk tokenization drifts
    (29 vs 21 tokens) (slice 2).
  * pause-don't-pad: generation_step advances 1/frame (~12.5/s); if the text source lags,
    STALL (don't step) instead of indexing into tts_pad_embed, which would skip text and
    drift to a premature EOS (slice 2).

The loop is a blocking sync generator (torch forwards + a blocking queue.get for the
stall). It runs in a worker thread; the WS handler bridges text in / PCM out via queues.
A stall yields NO chunks (not silence), so the onset/tail trim and WSOLA cadence wrappers
compose unchanged — a stall is just a gap in emission the client's buffer absorbs.
"""

import logging
import queue

import numpy as np
import torch

from .backends.optimized_backend import _trim_silence_stream, _speed_stream

logger = logging.getLogger(__name__)

# Sentinel pushed to the text queue to mark end-of-utterance (-> EOS flush).
DONE = object()

# Anti-loop penalty (this talker loops under plain greedy — see the fix branch) and the
# stable-prefix holdback (tokens that may re-merge with incoming text).
_REP_PENALTY = 1.3
_HOLDBACK = 2
# How often the stall re-checks the stop flag while blocked waiting for text.
_STALL_POLL_S = 0.1


def _greedy_penalized(logits, gen_tokens, suppress, rep=_REP_PENALTY):
    """Deterministic penalized-greedy next-token (matches the server's streaming path)."""
    logits = logits.clone()
    if suppress:
        logits[:, suppress] = float("-inf")
    if gen_tokens:
        idx = torch.unique(torch.cat(gen_tokens))
        s = logits[0, idx]
        logits[0, idx] = torch.where(s < 0, s * rep, s / rep)
    return torch.argmax(logits, dim=-1)


def _project_text(model, cumulative):
    """Cumulative assistant-wrapped text -> projected trailing hiddens [1, T, H] (the same
    projection _build_talker_inputs uses, so ids match the joint render -> same granularity)."""
    inner = model.model
    ids0 = model._tokenize_texts([model._build_assistant_text(cumulative)])[0]  # [1, T]
    text_ids = ids0[:, 4:-5]                                                     # text tokens only
    return inner.talker.text_projection(inner.talker.get_text_embeddings()(text_ids))


def _decode_window(inner, codes_buffer, decode_window_frames, device):
    """Decode the trailing `decode_window_frames` codes to a PCM wav (float). Uses the CUDA-graph
    optimized streaming decoder when available/warmed, else the plain decoder (offline/fresh model)."""
    start = max(0, len(codes_buffer) - decode_window_frames)
    window = torch.stack(codes_buffer[start:], dim=0).to(device)
    if hasattr(inner.speech_tokenizer, "decode_streaming"):
        try:
            wavs, sr = inner.speech_tokenizer.decode_streaming(
                window, use_optimized=True, pad_to_size=decode_window_frames,
            )
            return wavs[0].astype(np.float32), sr
        except Exception as exc:  # not warmed / shape mismatch -> plain decode
            logger.debug("decode_streaming fell back to decode(): %s", exc)
    wavs, sr = inner.speech_tokenizer.decode([{"audio_codes": window}])
    return wavs[0].astype(np.float32), sr


def _get_text(text_q, stop):
    """Blocking queue.get that wakes every _STALL_POLL_S to honor the stop flag.
    Returns the item, or DONE if the session was stopped/aborted."""
    while True:
        if stop is not None and stop():
            return DONE
        try:
            return text_q.get(timeout=_STALL_POLL_S)
        except queue.Empty:
            continue


@torch.no_grad()
def raw_session_pcm(model, text_q, speaker, language, *,
                    decode_window_frames=80, emit_every_frames=3,
                    holdback=_HOLDBACK, max_frames=4000, stop=None):
    """Core streaming-text talker loop. Yields (float_chunk in [-1,1], sample_rate).

    text_q items are str text deltas; DONE marks end-of-utterance. Runs in a worker thread
    (blocking forwards + blocking stall). `stop()` -> True aborts (barge-in / disconnect)."""
    inner = model.model
    device = inner.talker.device
    cfg = inner.config.talker_config
    eos_id = cfg.codec_eos_token_id
    vocab = cfg.vocab_size
    suppress = [i for i in range(vocab - 1024, vocab) if i != eos_id]
    spf = inner.speech_tokenizer.get_decode_upsample_rate()
    step_samples = spf * emit_every_frames

    cumulative = ""
    done = False

    # --- wait for the first text (prefill needs at least chunk 0) ---
    item = _get_text(text_q, stop)
    if item is DONE:
        return
    cumulative += item

    # Prefill: only the first text token enters the prompt (input_id[3:4]), so building it
    # from whatever text has arrived is stable. Its standalone trailing is discarded — we
    # feed trailing via the stable-prefix projection instead.
    ids = model._tokenize_texts([model._build_assistant_text(cumulative)])
    tie, mask, trailing_full, pad = inner._build_talker_inputs(
        input_ids=ids, instruct_ids=None, ref_ids=None, voice_clone_prompt=None,
        speakers=[speaker], languages=[language], non_streaming_mode=False,
    )
    eos_hidden = trailing_full[:, -1:, :]

    hid = _project_text(model, cumulative)
    committed = max(0, hid.shape[1] - holdback)
    trailing = hid[:, :committed, :]

    torch.compiler.cudagraph_mark_step_begin()
    out = inner.talker.forward(
        inputs_embeds=tie, attention_mask=mask, use_cache=True, output_hidden_states=True,
        return_dict=True, trailing_text_hidden=trailing, tts_pad_embed=pad,
        generation_step=None, past_hidden=None, past_key_values=None,
    )
    pkv, past_hidden, gstep = out.past_key_values, out.past_hidden, out.generation_step
    gen_tokens = []
    token = _greedy_penalized(out.logits[:, -1, :], gen_tokens, suppress)
    gen_tokens.append(token.detach().reshape(-1))

    codes_buffer = []
    total_emitted = 0
    eos_appended = False
    frames_since_emit = 0
    frames = 0

    def reproject(is_final):
        nonlocal trailing, committed
        h = _project_text(model, cumulative)
        stable = h.shape[1] if is_final else max(committed, h.shape[1] - holdback)
        if stable > committed:
            trailing = torch.cat([trailing, h[:, committed:stable, :]], dim=1)
            committed = stable

    while frames < max_frames:
        if stop is not None and stop():
            return

        # absorb any text that arrived since the last step
        new = False
        while True:
            try:
                it = text_q.get_nowait()
            except queue.Empty:
                break
            if it is DONE:
                done = True
            else:
                cumulative += it
                new = True
        if new:
            reproject(False)
        if done and not eos_appended:
            reproject(True)                                   # final rows, no holdback
            trailing = torch.cat([trailing, eos_hidden], dim=1)
            eos_appended = True

        # pause-don't-pad: all real text consumed but more is coming -> STALL (no step)
        real_rows = trailing.shape[1] - (1 if eos_appended else 0)
        if gstep >= real_rows and not done:
            it = _get_text(text_q, stop)
            if it is DONE:
                done = True
            else:
                cumulative += it
            continue                                          # re-project + re-check

        torch.compiler.cudagraph_mark_step_begin()
        step_out = inner.talker.forward(
            input_ids=token.unsqueeze(1), use_cache=True, return_dict=True,
            output_hidden_states=False, past_key_values=pkv, past_hidden=past_hidden,
            generation_step=gstep, trailing_text_hidden=trailing, tts_pad_embed=pad,
        )
        pkv, past_hidden, gstep = step_out.past_key_values, step_out.past_hidden, step_out.generation_step
        frames += 1
        codec_ids = step_out.hidden_states[1]
        if codec_ids[0, 0] == eos_id:
            break
        codes_buffer.append(codec_ids[0].detach())
        token = _greedy_penalized(step_out.logits[:, -1, :], gen_tokens, suppress)
        gen_tokens.append(token.detach().reshape(-1))

        frames_since_emit += 1
        if frames_since_emit < emit_every_frames:
            continue
        frames_since_emit = 0
        wav, sr = _decode_window(inner, codes_buffer, decode_window_frames, device)
        chunk = wav[-step_samples:] if step_samples > 0 else wav
        total_emitted = len(codes_buffer)
        yield chunk, sr

    # flush frames generated since the last emit (the tail before EOS)
    remaining = len(codes_buffer) - total_emitted
    if remaining > 0:
        wav, sr = _decode_window(inner, codes_buffer, decode_window_frames, device)
        tail = spf * remaining
        yield (wav[-tail:] if tail > 0 else wav), sr


def stream_text_session(model, text_q, speaker, language, speed, streaming_opts, stop=None):
    """raw_session_pcm wrapped with onset/tail-silence trim + pitch-preserving WSOLA at a
    FIXED `speed` (the streaming path can't know total length for the length-aware target).
    Yields (float_chunk, sr)."""
    raw = raw_session_pcm(
        model, text_q, speaker, language,
        decode_window_frames=streaming_opts.get("decode_window_frames", 80),
        emit_every_frames=streaming_opts.get("emit_every_frames", 3),
        stop=stop,
    )
    trimmed = _trim_silence_stream(
        raw,
        streaming_opts.get("onset_gate_rms", 0.05),
        streaming_opts.get("onset_attack_ms", 10.0),
        streaming_opts.get("onset_max_lead_ms", 2000.0),
        streaming_opts.get("tail_silence_rms", 0.02),
        streaming_opts.get("tail_keep_ms", 120.0),
    )
    for chunk, sr in _speed_stream(trimmed, speed):
        yield chunk, sr
