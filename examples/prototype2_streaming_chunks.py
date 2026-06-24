"""
PROTOTYPE-2 — slice 2 of streaming-tts-websocket (.scratch/streaming-tts-websocket/issues/02).

Slice 1 PROVED the append mechanism is bit-exact when text is fed from the JOINT render's pre-computed
hiddens. Slice 2 must prove the two things the real streaming server needs, which slice 1 deliberately
left out:

  (A) PER-CHUNK INDEPENDENT TOKENIZATION. The server won't have the joint render's hiddens — it gets text
      chunks as the LLM emits them and must tokenize+project EACH chunk on its own, then append. Does
      continuity still hold when chunk K is tokenized independently (different token boundaries than the
      joint tokenization)? Expect: NOT bit-identical to the joint reference (tokenization differs), but
      CONTINUOUS / one voice / no seam. Judge by ear + a small diff.

  (B) PAUSE-DON'T-PAD. The talker consumes one text row per decode frame at ~12.5 frames/s (12Hz model).
      If the LLM lags that rate, the talker would index past the available text into tts_pad_embed and
      emit premature trailing/silence or drift to EOS. The server must PAUSE (stop stepping) when
      text-starved and resume when the next chunk lands. This script demonstrates pause vs pad.

!!! v0 AUTHORED WITHOUT A GPU. The single biggest debug surface is the TEXT-TOKEN SLICE per chunk:
    _build_talker_inputs puts input_id[:,3:4] (the FIRST text token) into the PREFILL and input_id[:,4:-5]
    into trailing (modeling_qwen3_tts.py:2447-2478). So:
      - chunk 0 (drives the prefill): its trailing tokens are [4:-5]  (first token already in prefill).
      - chunk K>0 (no new prefill): ALL its text tokens belong in trailing -> slice [3:-5], NOT [4:-5],
        or you silently DROP the first word of every later chunk.
    This is exactly the per-chunk boundary slice 2 exists to nail. Verify the [3:-5] vs [4:-5] choice and
    the eos handling on the box (diagnostics print the slice lengths).

Run on the GPU box (same model/speaker as slice 1):
    python examples/prototype2_streaming_chunks.py
"""

import re

import numpy as np
import soundfile as sf
import torch

# --- adjust to your deployment (same as slice 1) ---------------------------------------------------
MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
SPEAKER = "Eric"
LANGUAGE = "English"
TEXT = "I was just thinking about you earlier and wanted to check in. How are things feeling for you right now?"
FRAMES_PER_SEC = 12.5   # 12Hz model: one text row consumed per decoded frame
REP_PENALTY = 1.3       # this model loops without it (see slice 1)
# ---------------------------------------------------------------------------------------------------

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel  # noqa: E402


def load():
    m = Qwen3TTSModel.from_pretrained(
        MODEL_PATH, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    m.model.eval()
    return m


def _greedy_penalized(logits, gen_tokens, suppress):
    logits = logits.clone()
    if suppress:
        logits[:, suppress] = float("-inf")
    if gen_tokens:
        idx = torch.unique(torch.cat(gen_tokens))
        s = logits[0, idx]
        logits[0, idx] = torch.where(s < 0, s * REP_PENALTY, s / REP_PENALTY)
    return torch.argmax(logits, dim=-1)


def build_prefill(m, chunk0):
    """Prefill talker inputs from chunk 0 (reuses the model's builder). Returns prefill embeds/mask, the
    chunk-0 trailing hiddens WITHOUT eos, the eos hidden row, and tts_pad_embed."""
    inner = m.model
    ids = m._tokenize_texts([m._build_assistant_text(chunk0)])
    tie, mask, trailing, pad = inner._build_talker_inputs(
        input_ids=ids, instruct_ids=None, ref_ids=None, voice_clone_prompt=None,
        speakers=[SPEAKER], languages=[LANGUAGE], non_streaming_mode=False,
    )
    trailing0 = trailing[:, :-1, :]          # chunk-0 text hiddens (drop the builder's eos row)
    eos_hidden = trailing[:, -1:, :]         # the eos row (tts_eos_embed), appended only at the very end
    return tie, mask, trailing0, eos_hidden, pad


def build_stable_deltas(m, chunks, holdback=2):
    """STABLE-PREFIX incremental tokenization (the slice-2 fix).

    Independent per-chunk tokenization fragmented words at boundaries (29 tokens vs the joint
    render's 21) because BPE merges differ when a word starts a chunk vs. follows context — so the
    talker saw a lumpier token stream and rendered ~30% longer. Instead, tokenize the CUMULATIVE
    text at each chunk and append only the newly-stable rows, holding back the last `holdback`
    tokens (which may still re-merge as more text arrives). This reproduces the JOINT tokenization
    incrementally: same token ids -> same projected hiddens -> same granularity as a whole-text
    render. Held-back rows are delivered with the next chunk (or fully on the final chunk, since no
    more text can change them). Returns one delta tensor [1, dT, H] per chunk."""
    inner = m.model
    deltas, committed, cum = [], 0, ""
    n = len(chunks)
    for i, ch in enumerate(chunks):
        cum += ch
        ids0 = m._tokenize_texts([m._build_assistant_text(cum)])[0]      # [1, T]
        text_ids = ids0[:, 4:-5]                                         # cumulative trailing target
        hid = inner.talker.text_projection(inner.talker.get_text_embeddings()(text_ids))  # [1, T, H]
        T = hid.shape[1]
        stable = T if i == n - 1 else max(committed, T - holdback)       # final chunk: no holdback
        deltas.append(hid[:, committed:stable, :])
        committed = stable
    return deltas


def chunk_text(s):
    """Word/comma-aligned chunks, like the wabi WS client streams."""
    parts = re.findall(r"\S+\s*", s)
    # group into ~3-word chunks to exercise multiple appends
    out, cur = [], ""
    for i, p in enumerate(parts):
        cur += p
        if (i + 1) % 3 == 0:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def _suppress(inner):
    cfg = inner.config.talker_config
    eos_id = cfg.codec_eos_token_id
    vocab = cfg.vocab_size
    return eos_id, [i for i in range(vocab - 1024, vocab) if i != eos_id]


@torch.no_grad()
def generate_reference(m, text, max_frames=600):
    """Whole-text render — the continuity/quality baseline."""
    inner = m.model
    ids = m._tokenize_texts([m._build_assistant_text(text)])
    tie, mask, trailing, pad = inner._build_talker_inputs(
        input_ids=ids, instruct_ids=None, ref_ids=None, voice_clone_prompt=None,
        speakers=[SPEAKER], languages=[LANGUAGE], non_streaming_mode=False,
    )
    return _decode_loop(inner, tie, mask, trailing, pad, eos_hidden=None, pending=None,
                        pause=False, max_frames=max_frames)[0]


@torch.no_grad()
def _decode_loop(inner, tie, mask, trailing, pad, eos_hidden, pending, pause, max_frames):
    """Shared talker loop. If `pending` is None this is the plain reference loop. Otherwise it appends
    later chunks at their wall-clock arrival times (pending = [(arrival_s, hiddens), ...]) and applies
    pause-don't-pad when `pause`. Returns (codes, stats)."""
    eos_id, suppress = _suppress(inner)
    device = inner.talker.device
    DT = 1.0 / FRAMES_PER_SEC

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

    pend = list(pending) if pending else []
    eos_done = pending is None  # reference already has its eos baked into `trailing`
    t = 0.0
    pads = 0
    codes = []

    for _ in range(max_frames):
        # deliver chunks whose arrival time has passed
        while pend and pend[0][0] <= t:
            trailing = torch.cat([trailing, pend.pop(0)[1].to(device)], dim=1)

        real_rows = trailing.shape[1] - (1 if eos_done and pending is not None else 0)
        starved = pending is not None and gstep >= real_rows

        if starved and pend and pause:
            t += DT          # PAUSE: text en route, advance wall time, do NOT decode (no pad emitted)
            continue
        if starved and not pend and not eos_done:
            trailing = torch.cat([trailing, eos_hidden.to(device)], dim=1)  # all text in -> cap with eos
            eos_done = True
        if starved and not pause:
            pads += 1        # PAD: stepping into a pad/eos position while text could still arrive

        torch.compiler.cudagraph_mark_step_begin()
        step_out = inner.talker.forward(
            input_ids=token.unsqueeze(1), use_cache=True, return_dict=True, output_hidden_states=False,
            past_key_values=pkv, past_hidden=past_hidden, generation_step=gstep,
            trailing_text_hidden=trailing, tts_pad_embed=pad,
        )
        pkv, past_hidden, gstep = step_out.past_key_values, step_out.past_hidden, step_out.generation_step
        t += DT
        codec_ids = step_out.hidden_states[1]
        if codec_ids[0, 0] == eos_id:
            break
        codes.append(codec_ids[0].detach())
        token = _greedy_penalized(step_out.logits[:, -1, :], gen_tokens, suppress)
        gen_tokens.append(token.detach().reshape(-1))

    return codes, {"pads": pads, "frames": len(codes), "t": round(t, 2)}


@torch.no_grad()
def decode(inner, codes):
    if not codes:
        return np.zeros(0, np.float32), 24000
    w = torch.stack(codes, dim=0).to(inner.talker.device)
    wavs, sr = inner.speech_tokenizer.decode([{"audio_codes": w}])
    return wavs[0].astype(np.float32), sr


def run_streamed(m, chunks, arrivals, pause, max_frames=600):
    inner = m.model
    # build_prefill gives the prompt (tie/mask), eos row and pad; its standalone trailing0 is
    # discarded — trailing now comes from the cumulative stable-prefix deltas instead.
    tie, mask, _t0, eos_hidden, pad = build_prefill(m, chunks[0])
    deltas = build_stable_deltas(m, chunks)
    trailing0 = deltas[0]                                   # chunk-0 stable rows -> initial trailing
    pending = [(arrivals[i], deltas[i]) for i in range(1, len(chunks)) if deltas[i].shape[1] > 0]
    print(f"[diag] chunks={len(chunks)} stable_deltas={[int(d.shape[1]) for d in deltas]} "
          f"sum={sum(int(d.shape[1]) for d in deltas)} arrivals={[round(a,2) for a in arrivals]} pause={pause}")
    codes, stats = _decode_loop(inner, tie, mask, trailing0, pad, eos_hidden, pending, pause, max_frames)
    return codes, stats


def main():
    m = load()
    inner = m.model
    chunks = chunk_text(TEXT)
    print(f"[diag] {len(chunks)} chunks: {chunks}")

    ref_codes = generate_reference(m, TEXT)
    ref, sr = decode(inner, ref_codes)
    sf.write("ref.wav", ref, sr)
    print(f"reference: {len(ref_codes)} frames {len(ref)/sr:.2f}s")

    # (A) per-chunk tokenization, all chunks available immediately (talker never starves)
    a_codes, a_stats = run_streamed(m, chunks, arrivals=[0.0] * len(chunks), pause=True)
    a_wav, _ = decode(inner, a_codes)
    sf.write("streamed_perchunk.wav", a_wav, sr)

    # (B) LLM lags: each chunk arrives ~0.6s apart — slower than the ~12.5 fps consumption -> talker starves
    lag = [0.0] + [0.6 * i for i in range(1, len(chunks))]
    b_codes, b_stats = run_streamed(m, chunks, arrivals=lag, pause=True)   # pause-don't-pad (correct)
    b_wav, _ = decode(inner, b_codes)
    sf.write("streamed_lag_pause.wav", b_wav, sr)
    c_codes, c_stats = run_streamed(m, chunks, arrivals=lag, pause=False)  # pad (wrong) — contrast
    c_wav, _ = decode(inner, c_codes)
    sf.write("streamed_lag_pad.wav", c_wav, sr)

    def cmp(name, w):
        n = min(len(ref), len(w))
        d = np.abs(ref[:n] - w[:n]) if n else np.array([1.0])
        print(f"{name:24s} frames~{len(w)//(len(ref)//max(1,len(ref_codes)) or 1):>4} "
              f"dur={len(w)/sr:5.2f}s  max|d|vs_ref={d.max():.3f}")

    print("\n=== RESULTS ===")
    print(f"(A) per-chunk on-time : {a_stats}")
    cmp("  streamed_perchunk", a_wav)
    print(f"(B) lag + PAUSE       : {b_stats}   <- pads should be 0")
    cmp("  streamed_lag_pause", b_wav)
    print(f"(B) lag + PAD (wrong) : {c_stats}   <- pads > 0, audio degraded/longer")
    cmp("  streamed_lag_pad", c_wav)
    print("\nGO for slice 2 if: streamed_perchunk sounds continuous (one voice, no seam) AND "
          "streamed_lag_pause has pads==0 and sounds like ref, while streamed_lag_pad is audibly worse.")


if __name__ == "__main__":
    main()
