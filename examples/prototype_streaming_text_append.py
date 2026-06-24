"""
PROTOTYPE — slice 1 of streaming-tts-websocket (.scratch/streaming-tts-websocket/issues/01).

Go/no-go gate: prove that the talker can be driven with text APPENDED mid-generation (preserving the KV
cache) and produce audio that is CONTINUOUS and matches a single whole-text generation — same voice, no
seam. If this holds, approach B (one continuous voice over a streaming-text WebSocket) is viable.

WHY THIS SHOULD WORK (from the model source, verified):
  - Text conditions the talker ADDITIVELY, one hidden per decode step:
        modeling_qwen3_tts.py:1927-1930  ->  inputs_embeds += trailing_text_hidden[:, generation_step]
    When text runs out it pads with tts_pad_embed. So the text is NOT baked into the prefill; it is a flat
    positional stream consumed one-per-step.
  - All talker state is explicit and returned every forward (past_key_values, past_hidden, generation_step),
    and stream_generate_pcm threads it through the loop. The KV cache IS the continuation state.
  Therefore: revealing B's text-hiddens LATE (as if they arrived late from the LLM) must give the same
  output as having them present up front — provided they are appended BEFORE generation_step indexes into
  them. That last clause is the pause-don't-pad concern productionized in slice 2; here we just append
  early enough.

WHAT THIS SCRIPT DOES:
  1. Renders "A B" as ONE normal generation            -> reference.wav
  2. Renders the same text but with B's hiddens appended only after `split_after_frames` decode steps,
     reusing the same KV cache                          -> streamed.wav
  3. Prints max/mean abs sample diff. Near-zero (modulo sampling nondeterminism — use do_sample=False for a
     deterministic A/B) == continuity holds == GO.

!!! THIS IS A v0 AUTHORED WITHOUT A GPU. Expect to debug the marked spots on real hardware. The single
    most likely bug is the split index (how many text-token positions belong to "A") and its alignment
    with generation_step at the append boundary. Each DEBUG marker calls that out.

Run on the GPU box, from the fork root:
    python examples/prototype_streaming_text_append.py
Adjust MODEL_PATH / SPEAKER / the two text halves to taste.
"""

import numpy as np
import soundfile as sf
import torch

# --- adjust these to your deployment ---------------------------------------------------------------
MODEL_PATH = "Qwen/Qwen3-TTS-Flash"  # or your local 1.7B custom-voice checkpoint path
SPEAKER = "alloy"                    # a supported custom-voice speaker
LANGUAGE = "English"
TEXT_A = "I was just thinking about you earlier and wanted to check in."
TEXT_B = " How are things feeling for you right now?"
SPLIT_AFTER_FRAMES = 40             # reveal B after this many decoded frames (must be < len(A) frames)
DETERMINISTIC = True               # do_sample=False so reference vs streamed are bit-comparable
# ---------------------------------------------------------------------------------------------------

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel  # noqa: E402  (the high-level wrapper)
from qwen_tts.core.models.modeling_qwen3_tts import _sample_next_token  # noqa: E402


def load():
    # Correct loader (qwen3_tts_model.py:from_pretrained). Match your deploy's dtype/attn/device.
    m = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    m.model.eval()
    return m


def text_to_trailing(inner, ids_text):
    """Project text token ids -> trailing_text_hidden, the way _build_talker_inputs does (line 2476-2478):
       text_projection(get_text_embeddings()(text_ids)).  Returns [1, T, H]. NO eos appended here — eos is
       added only for the final chunk by the caller."""
    talker = inner.talker
    emb = talker.get_text_embeddings()(ids_text)          # [1, T, H_text]
    return talker.text_projection(emb)                    # [1, T, H]


@torch.no_grad()
def generate_collect_codes(inner, talker_input_embeds, talker_attention_mask, trailing, tts_pad_embed,
                           append_at=None, append_trailing=None, eos_trailing=None,
                           max_frames=4000, do_sample=not DETERMINISTIC):
    """A trimmed copy of stream_generate_pcm's TALKER loop (modeling_qwen3_tts.py:2611-2837) that COLLECTS
    all codec frames (no windowed streaming decode — we decode once at the end, which is enough to prove
    talker continuity). Optionally appends `append_trailing` (and finally `eos_trailing`) into the live
    `trailing` buffer at decode step `append_at`, simulating B arriving late.
    """
    cfg = inner.config.talker_config
    eos_id = cfg.codec_eos_token_id
    vocab = cfg.vocab_size
    suppress = [i for i in range(vocab - 1024, vocab) if i != eos_id]
    device = inner.talker.device

    torch.compiler.cudagraph_mark_step_begin()
    out = inner.talker.forward(
        inputs_embeds=talker_input_embeds, attention_mask=talker_attention_mask,
        use_cache=True, output_hidden_states=True, return_dict=True,
        trailing_text_hidden=trailing, tts_pad_embed=tts_pad_embed,
        generation_step=None, past_hidden=None, past_key_values=None,
    )
    pkv, past_hidden, gstep = out.past_key_values, out.past_hidden, out.generation_step
    last = out.logits[:, -1, :]
    token = _sample_next_token(last, 0.9, 50, 1.0, suppress) if do_sample else torch.argmax(last, dim=-1)

    codes = []
    for step in range(max_frames):
        # DEBUG(1): append boundary. `trailing` must be extended BEFORE gstep reaches len(A). If you see
        # the stream end early / a pad-burst, SPLIT_AFTER_FRAMES is too late vs the A text length, or the
        # split index below is wrong. Compare gstep here to the A trailing length.
        if append_at is not None and step == append_at and append_trailing is not None:
            pieces = [trailing, append_trailing.to(device)]
            if eos_trailing is not None:
                pieces.append(eos_trailing.to(device))
            trailing = torch.cat(pieces, dim=1)

        torch.compiler.cudagraph_mark_step_begin()
        step_out = inner.talker.forward(
            input_ids=token.unsqueeze(1), use_cache=True, return_dict=True, output_hidden_states=False,
            past_key_values=pkv, past_hidden=past_hidden, generation_step=gstep,
            trailing_text_hidden=trailing, tts_pad_embed=tts_pad_embed,
        )
        pkv, past_hidden, gstep = step_out.past_key_values, step_out.past_hidden, step_out.generation_step
        codec_ids = step_out.hidden_states[1]            # [B, num_code_groups]
        if codec_ids[0, 0] == eos_id:
            break
        codes.append(codec_ids[0].detach())
        logits = step_out.logits[:, -1, :]
        token = _sample_next_token(logits, 0.9, 50, 1.0, suppress) if do_sample else torch.argmax(logits, dim=-1)

    return codes


@torch.no_grad()
def decode_codes(inner, codes):
    if not codes:
        return np.zeros(0, np.float32), 24000
    window = torch.stack(codes, dim=0).to(inner.talker.device)
    wavs, sr = inner.speech_tokenizer.decode([{"audio_codes": window}])
    return wavs[0].astype(np.float32), sr


def build(inner, text):
    """Tokenize + build talker inputs for a whole utterance (reuses the model's own builder)."""
    ids = inner._tokenize_texts([inner._build_assistant_text(text)])
    tie, mask, trailing, pad = inner._build_talker_inputs(
        input_ids=ids, speakers=[SPEAKER], languages=[LANGUAGE], non_streaming_mode=False,
    )
    return ids, tie, mask, trailing, pad


def main():
    m = load()
    inner = m.model  # the Qwen3TTSForConditionalGeneration

    # ---- reference: whole "A B" in one go ----
    ids_ab, tie, mask, trailing_full, pad = build(inner, TEXT_A + TEXT_B)
    ref_codes = generate_collect_codes(inner, tie, mask, trailing_full, pad)
    ref_wav, sr = decode_codes(inner, ref_codes)
    sf.write("reference.wav", ref_wav, sr)
    print(f"reference: {len(ref_codes)} frames, {len(ref_wav)/sr:.2f}s")

    # ---- streamed: start with A's trailing only, append B (then eos) at the boundary ----
    # DEBUG(2): split index. trailing_full corresponds to input_id[:, 4:-5] of "A B" (modeling:2476-2478),
    # i.e. the TEXT tokens only, with a trailing eos hidden appended. We need (a) nA = #text-token positions
    # belonging to A, and (b) B's projected hiddens, and (c) the eos hidden (the last row of trailing_full).
    ids_a = inner._tokenize_texts([inner._build_assistant_text(TEXT_A)])
    # text-token slice mirrors _build_talker_inputs line 2477: input_id[:, 4:-5]
    a_text_ids = ids_a[0][:, 4:-5]
    nA = a_text_ids.shape[1]
    ids_b = inner._tokenize_texts([inner._build_assistant_text(TEXT_B)])
    b_text_ids = ids_b[0][:, 4:-5]
    trailing_A = trailing_full[:, :nA, :]                 # A's hiddens, no eos
    trailing_B = text_to_trailing(inner, b_text_ids)      # B's hiddens, no eos
    eos_hidden = trailing_full[:, -1:, :]                 # the eos row appended by the builder

    # Prefill with A-only trailing; append B+eos at SPLIT_AFTER_FRAMES.
    stream_codes = generate_collect_codes(
        inner, tie, mask, trailing_A, pad,
        append_at=SPLIT_AFTER_FRAMES, append_trailing=trailing_B, eos_trailing=eos_hidden,
    )
    stream_wav, sr = decode_codes(inner, stream_codes)
    sf.write("streamed.wav", stream_wav, sr)
    print(f"streamed:  {len(stream_codes)} frames, {len(stream_wav)/sr:.2f}s")

    # ---- compare ----
    n = min(len(ref_wav), len(stream_wav))
    if n:
        d = np.abs(ref_wav[:n] - stream_wav[:n])
        print(f"len ref={len(ref_wav)} stream={len(stream_wav)}  max|d|={d.max():.4f} mean|d|={d.mean():.5f}")
        print("GO if streamed.wav sounds identical/continuous to reference.wav (DETERMINISTIC=True -> ~0 diff).")
    else:
        print("EMPTY output — debug the prefill/append indices (see DEBUG markers).")


if __name__ == "__main__":
    main()
