"""Offline test for the streaming-text session generator (api/streaming_session.py), without the
WebSocket layer. Feeds text into the queue from a timer thread (simulating LLM deltas, optionally
lagged to exercise pause-don't-pad) and renders PCM via stream_text_session. Writes wavs + prints
frame/duration stats. Run in the container:  python examples/test_session_offline.py
"""

import sys, re, threading, time, queue
sys.path.insert(0, "/app")
import numpy as np
import soundfile as sf
import torch
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from api.streaming_session import stream_text_session, DONE

SR = 24000
SPEAKER, LANGUAGE = "Eric", "English"
TEXT = "I was just thinking about you earlier and wanted to check in. How are things feeling for you right now?"
OPTS = {"decode_window_frames": 80, "emit_every_frames": 3, "onset_gate_rms": 0.05,
        "onset_attack_ms": 10.0, "onset_max_lead_ms": 2000.0, "tail_silence_rms": 0.02,
        "tail_keep_ms": 120.0}


def chunk_text(s):
    parts = re.findall(r"\S+\s*", s); out, cur = [], ""
    for i, p in enumerate(parts):
        cur += p
        if (i + 1) % 3 == 0: out.append(cur); cur = ""
    if cur: out.append(cur)
    return out


def feed(q, chunks, gap):
    for c in chunks:
        if gap: time.sleep(gap)
        q.put(c)
    q.put(DONE)


def render(model, chunks, gap, speed=1.0):
    q = queue.Queue()
    t = threading.Thread(target=feed, args=(q, chunks, gap), daemon=True)
    t.start()
    out = [c for c, _ in stream_text_session(model, q, SPEAKER, LANGUAGE, speed, OPTS)]
    t.join(timeout=5)
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def main():
    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    m.model.eval()
    chunks = chunk_text(TEXT)
    print(f"[diag] {len(chunks)} chunks")

    whole = render(m, [TEXT], 0.0)
    sf.write("session_whole.wav", whole, SR)
    print(f"whole-text 1 chunk : {len(whole)/SR:.2f}s ({len(whole)} samp)")

    ontime = render(m, chunks, 0.0)
    sf.write("session_ontime.wav", ontime, SR)
    print(f"chunked on-time    : {len(ontime)/SR:.2f}s ({len(ontime)} samp)")

    lagged = render(m, chunks, 0.6)   # 0.6s/chunk -> slower than ~12.5 fps -> exercises stall
    sf.write("session_lagged.wav", lagged, SR)
    print(f"chunked LAGGED 0.6s: {len(lagged)/SR:.2f}s ({len(lagged)} samp)  <- pause-don't-pad")

    spd = render(m, chunks, 0.0, speed=1.15)
    sf.write("session_speed115.wav", spd, SR)
    print(f"on-time speed=1.15 : {len(spd)/SR:.2f}s ({len(spd)} samp)  <- WSOLA cadence")


if __name__ == "__main__":
    main()
