# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Optimized Qwen3-TTS backend with dynamic model switching, torch.compile,
CUDA graph captures, voice prompt caching, and real-time streaming.

This backend reads its model roster and optimization knobs from a YAML config
file (default: ~/qwen3-tts/config.yaml, overridable via TTS_CONFIG env var).
It auto-switches between the CustomVoice and Base models depending on the
request (voice-library profiles always require the Base model).
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Any

import numpy as np

from .base import TTSBackend

logger = logging.getLogger(__name__)

# Leading noise gate for the START of a stream. The streaming decoder has a
# cold-start that emits low-level junk BEFORE the first phoneme: a ~80ms
# low-frequency thump at t=0, and a ~10ms broadband HF "static" tick right as the
# first speech window is decoded. Both sit well below speech level (RMS <0.01 vs
# speech RMS ~0.13), so we hold the output muted until a short-window RMS crosses
# a threshold (= speech onset), then ramp in over a brief attack. This removes
# both artifacts without clipping speech (a fixed-position fade could not, since
# the tick lands at variable offset). Tunables (optimization.streaming):
#   onset_gate_rms   — RMS threshold that counts as speech (0 disables gating)
#   onset_attack_ms  — raised-cosine ramp length applied at the detected onset
#   onset_max_lead_ms— safety: force the gate open after this much muted lead, so
#                      a too-high threshold can never swallow a whole utterance
_DEFAULT_ONSET_GATE_RMS = 0.05
_DEFAULT_ONSET_ATTACK_MS = 10.0
_DEFAULT_ONSET_MAX_LEAD_MS = 2000.0

# Trailing-silence suppression. The talker emits ~0.6s of trailing silence before
# EOS fires on short utterances, so a 0.5s "Yes." ships as a 1.5s clip that's
# mostly dead air — the "drag" voice agents hear on short backchannels. We buffer
# near-silent chunks once speech has started; an internal pause is preserved
# (flushed when speech resumes), but the FINAL trailing run is dropped, keeping a
# short natural tail. Tunables (optimization.streaming):
#   tail_silence_rms — chunk RMS below this counts as silence (0 disables)
#   tail_keep_ms     — natural tail kept after the last speech
_DEFAULT_TAIL_SILENCE_RMS = 0.02
_DEFAULT_TAIL_KEEP_MS = 120.0


def _chunk_rms(chunk) -> float:
    if chunk is None or len(chunk) == 0:
        return 0.0
    x = chunk.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(x * x)))


def _max_frames_for_text(text: str) -> int:
    """Defensive per-input cap on generated codec frames. EOS normally fires far
    sooner; this only bounds a runaway (no EOS) so it can't reach the 10000 default
    (~800s). Measured legit rate is ~1 frame/char (e.g. 34 chars -> ~32 frames);
    ~4 frames/char is 4x headroom (never truncates real speech) while bounding a
    runaway to ~4x instead of the old 12x (a 34-char runaway capped ~32s -> ~11s).
    Floor 64 (~5s) for very short inputs; ceiling 1200 (~96s) per sentence."""
    n = len(text or "")
    return max(64, min(1200, n * 4))


# Calibrated native streaming duration model: dur(N) ~= overhead + N / asymptote
# (measured 2026-06-23, seed 1234, Vivian: overhead 0.23s, asymptote ~15.3 ch/s).
# Used to pick a per-reply tempo so rendered cadence hits a constant target ch/s
# regardless of length — short replies (natively 8-10 ch/s from the fixed overhead)
# speed up; long replies (already ~15 ch/s) stay near native.
_SPEED_OVERHEAD_S = 0.23
_SPEED_ASYMPTOTE_CHS = 15.3
_SPEED_MIN = 0.85
_SPEED_MAX = 1.5


def _target_speed_for_text(n_chars, target_chs, overhead=_SPEED_OVERHEAD_S,
                           asymptote=_SPEED_ASYMPTOTE_CHS,
                           lo=_SPEED_MIN, hi=_SPEED_MAX):
    """Tempo multiplier so an N-char reply renders at ~target_chs chars/sec.
    Returns None if targeting is disabled. Clamped to [lo, hi]: very short replies
    (e.g. 'Okay.', 5 chars) can't physically reach the target, so they hit hi and
    land just under the band rather than turning into chipmunk speech."""
    if not target_chs or target_chs <= 0 or n_chars <= 0:
        return None
    native_dur = overhead + n_chars / asymptote
    target_dur = n_chars / target_chs
    return float(min(hi, max(lo, native_dur / target_dur)))


def _resolve_speed(text, request_speed, gen_cfg):
    """Effective streaming tempo. An explicit per-request OpenAI speed (!=1.0) wins
    outright; otherwise use the length-aware target cadence; otherwise the fixed
    `generation.speed` fallback."""
    if request_speed is not None and abs(request_speed - 1.0) > 1e-3:
        return request_speed
    ts = _target_speed_for_text(
        len(text or ""),
        gen_cfg.get("target_chars_per_sec", 0),
        overhead=float(gen_cfg.get("speed_overhead_s", _SPEED_OVERHEAD_S)),
        asymptote=float(gen_cfg.get("speed_asymptote_chars_per_sec", _SPEED_ASYMPTOTE_CHS)),
        lo=float(gen_cfg.get("speed_min", _SPEED_MIN)),
        hi=float(gen_cfg.get("speed_max", _SPEED_MAX)),
    )
    if ts is not None:
        return ts
    return float(gen_cfg.get("speed", 1.0) or 1.0)


def _trim_silence_stream(source, gate_rms, attack_ms, max_lead_ms,
                         tail_silence_rms, tail_keep_ms):
    """Wrap a (chunk, sr) generator: drop leading silence (onset gate; fully-muted
    lead chunks are skipped, not emitted) and suppress trailing silence."""
    gate = {"open": gate_rms <= 0, "muted": 0}  # disabled gate => start open
    buf = []  # buffered near-silent chunks (internal pause vs trailing silence)
    for chunk, sr in source:
        if not gate.get("open"):
            gated = _gate_onset(chunk, sr, gate, gate_rms, attack_ms, max_lead_ms)
            if not gate.get("open"):
                continue  # still leading silence -> drop entirely (no dead air)
            yield gated, sr  # gate opened on this chunk
            continue
        if tail_silence_rms > 0 and _chunk_rms(chunk) < tail_silence_rms:
            buf.append((chunk, sr))  # hold: trailing silence or an internal pause?
        else:
            for b in buf:            # speech resumed -> it was a pause; keep it
                yield b
            buf.clear()
            yield chunk, sr
    # stream ended: the buffered run is trailing silence -> drop it, keep a short tail
    if buf:
        sr = buf[0][1]
        keep = int(sr * tail_keep_ms / 1000.0)
        if keep > 0:
            joined = np.concatenate([c for c, _ in buf])
            yield (joined[:keep] if len(joined) > keep else joined), sr


def _speed_stream(source, speed, frame=1024, syn_hop=512, search=400):
    """Pitch-preserving tempo change for a streaming (chunk, sr) float source via
    streaming WSOLA (Waveform-Similarity Overlap-Add).

    Qwen3-TTS has no native rate control: short replies render at the correct
    length but an unhurried cadence (~8-10 chars/s vs ~16 for longer text). We
    speed up delivery without shifting pitch.

    A phase vocoder (librosa.time_stretch) does this but smears phase, adding a
    robotic/echoey 'phasiness' on speech. WSOLA instead aligns each synthesis
    frame to the previous one by waveform similarity (time-domain cross-correlation)
    and overlap-adds with a Hann window, preserving local waveform shape — natural
    voice, no echo. It runs as ONE continuous synthesis with only ~(frame+search)
    samples (~60ms) of lookahead, so there are no per-chunk seams and TTFB is
    barely affected. speed>1 = faster/shorter; speed<1 = slower. Chunks are float32
    in [-1, 1] (encode_audio does the int16 step downstream)."""
    if speed is None or abs(speed - 1.0) < 1e-3:
        yield from source
        return

    w = np.hanning(frame).astype(np.float32)
    ov = syn_hop                                   # similarity / overlap length
    ana_hop = max(1, int(round(syn_hop * speed)))  # input advance per frame

    buf = np.zeros(0, dtype=np.float32)            # input[buf_start:]
    buf_start = 0
    a = 0                                          # next analysis pos (absolute)
    s = 0                                          # next synthesis pos (absolute)
    out_start = 0                                  # absolute index of oacc[0]
    oacc = np.zeros(0, dtype=np.float32)           # output overlap-add accumulator
    owin = np.zeros(0, dtype=np.float32)           # matching window-sum (for COLA norm)
    natural = [None]                               # expected continuation (len ov)

    def _process(final):
        nonlocal buf, buf_start, a, s, out_start, oacc, owin
        out = []
        while True:
            avail = buf_start + buf.size
            if not final and a + search + frame > avail:
                break                              # wait for more lookahead
            if a + frame > avail:
                break                              # nothing left to place
            lo = max(buf_start, a - search)
            hi = min(avail - ov, a + search)
            if natural[0] is None or hi <= lo:
                pos = min(a, avail - frame)
            else:
                region = buf[lo - buf_start: hi + ov - buf_start]
                if region.size < ov:
                    pos = min(a, avail - frame)
                else:
                    sc = np.correlate(region, natural[0], mode="valid")
                    pos = lo + int(np.argmax(sc))
            pos = min(max(pos, buf_start), avail - frame)
            seg = buf[pos - buf_start: pos - buf_start + frame]
            if seg.size < frame:
                seg = np.concatenate([seg, np.zeros(frame - seg.size, np.float32)])
            need = (s + frame) - out_start
            if need > oacc.size:
                g = need - oacc.size
                oacc = np.concatenate([oacc, np.zeros(g, np.float32)])
                owin = np.concatenate([owin, np.zeros(g, np.float32)])
            i0 = s - out_start
            oacc[i0:i0 + frame] += seg * w
            owin[i0:i0 + frame] += w
            nat0 = pos + syn_hop
            if nat0 + ov <= avail:
                natural[0] = buf[nat0 - buf_start: nat0 - buf_start + ov].copy()
            else:
                natural[0] = None
            s += syn_hop
            a += ana_hop
            fin = s - out_start                    # samples < s are final
            if fin > 0:
                wn = owin[:fin].copy()
                wn[wn < 1e-6] = 1.0
                out.append((oacc[:fin] / wn).astype(np.float32))
                oacc = oacc[fin:].copy()
                owin = owin[fin:].copy()
                out_start = s
            keep = max(buf_start, a - search - frame)   # drop consumed input
            if keep > buf_start:
                buf = buf[keep - buf_start:].copy()
                buf_start = keep
        return out

    sr_last = 24000
    for chunk, sr in source:
        sr_last = sr
        x = np.asarray(chunk, dtype=np.float32).reshape(-1)
        buf = np.concatenate([buf, x]) if buf.size else x
        for oc in _process(final=False):
            if oc.size:
                yield oc, sr
    for oc in _process(final=True):
        if oc.size:
            yield oc, sr_last
    if oacc.size:                                  # leftover synthesized tail
        wn = owin.copy()
        wn[wn < 1e-6] = 1.0
        tail = (oacc / wn).astype(np.float32)
        if tail.size:
            yield tail, sr_last


def _gate_onset(chunk, sr, state, threshold, attack_ms, max_lead_ms):
    """Mute leading sub-threshold junk; open with a raised-cosine attack at onset.

    ``state`` is a per-stream dict {'open': bool, 'muted': int}. Returns the
    (possibly muted/ramped) chunk; once open, chunks pass through unchanged.
    """
    if threshold <= 0 or state.get("open"):
        return chunk
    if chunk is None or len(chunk) == 0 or sr <= 0:
        return chunk
    x = chunk.astype(np.float32, copy=False)
    win = max(1, int(sr * 0.008))  # 8ms moving-RMS window
    csq = np.concatenate([[0.0], np.cumsum(x * x, dtype=np.float64)])
    if len(x) >= win:
        mrms = np.sqrt(np.maximum(csq[win:] - csq[:-win], 0.0) / win)
        above = np.where(mrms > threshold)[0]
    else:
        above = np.array([], dtype=int)

    if len(above) == 0:
        # Whole chunk is sub-threshold: stay closed and emit silence — unless we
        # have already muted too much (threshold likely too high for this voice).
        state["muted"] = state.get("muted", 0) + len(x)
        if state["muted"] >= int(sr * max_lead_ms / 1000.0):
            state["open"] = True
            return chunk
        return np.zeros_like(x)

    # Anchor onset at the END of the first above-threshold window: the forward
    # moving-RMS crosses as soon as the window's leading edge touches rising
    # energy, which is ~one window before speech is actually sustained. Using the
    # window end places k inside solid speech so the muted region fully covers the
    # pre-speech HF tick rather than ramping through it.
    k = min(int(above[0]) + win, len(x))
    a = int(sr * attack_ms / 1000.0)
    out = x.copy()
    s = max(0, k - a)
    out[:s] = 0.0
    n = k - s
    if n > 1:
        out[s:k] *= 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32)))
    state["open"] = True
    return out


# Location of the YAML config file (overridable via TTS_CONFIG env var)
_DEFAULT_CONFIG_PATH = Path.home() / "qwen3-tts" / "config.yaml"


def _load_config() -> dict:
    """Load the YAML configuration file, returning an empty dict on failure."""
    config_path = Path(os.environ.get("TTS_CONFIG", str(_DEFAULT_CONFIG_PATH)))
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as fh:
                return yaml.safe_load(fh) or {}
        except Exception as exc:
            logger.warning(f"Could not load config {config_path}: {exc}")
    return {}


class OptimizedQwen3TTSBackend(TTSBackend):
    """
    Optimized backend with dynamic model switching and real-time streaming.

    Key capabilities:
    - torch.compile + CUDA graph captures (configured via config.yaml)
    - Switches between CustomVoice and Base models on demand
    - Voice prompt caching (~0.7 s saved per repeated voice-clone request)
    - Real-time PCM streaming for both CustomVoice and Base (voice-clone) models

    Config file (~/qwen3-tts/config.yaml) example::

        default_model: 0.6B-CustomVoice
        models:
          0.6B-CustomVoice:
            hf_id: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
            type: customvoice          # "customvoice" | "base"
          0.6B-Base:
            hf_id: Qwen/Qwen3-TTS-12Hz-0.6B-Base
            type: base
        optimization:
          attention: flash_attention_2
          use_compile: true
          compile_mode: max-autotune   # "default" | "reduce-overhead" | "max-autotune"
          use_cuda_graphs: true
          use_fast_codebook: true
          compile_codebook_predictor: true
          streaming:
            decode_window_frames: 80   # AMD users: try 72; NVIDIA: 80 is fine
            emit_every_frames: 6       # lower = lower TTFB; higher = better RTF
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = _load_config()
        self.current_model_key: Optional[str] = None
        self._voice_prompt_cache: Dict[str, Any] = {}  # cache_key -> VoiceClonePromptItem list
        self._ready = False
        # Optional fixed RNG seed for reproducible (deterministic) generation.
        # None => random/varied output each call (sampling defaults).
        self.seed: Optional[int] = self.config.get("seed", None)

    def _apply_seed(self) -> None:
        """Reseed torch RNGs before a generation so output is reproducible.

        With a seed set, identical (text, speaker) input yields identical audio
        and the speaker voice stays stable across requests. No-op when seed is None.
        """
        if self.seed is None:
            return
        import torch
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _gen_kwargs(self) -> Dict[str, Any]:
        """Sampling controls from config['generation'].

        Lower randomness here = more stable speaker timbre across different
        sentences. The sub-talker predicts the fine codec codes that carry the
        voice's timbre, so its settings matter most (subtalker_dosample: false
        locks timbre). Empty dict => model defaults (do_sample, temperature 0.9).
        """
        gen = self.config.get("generation", {})
        keys = (
            "do_sample", "temperature", "top_k", "top_p",
            "subtalker_dosample", "subtalker_temperature",
            "subtalker_top_k", "subtalker_top_p",
            # Anti-loop: higher discourages the talker from repeating frames and
            # running away. Flows to stream_generate_pcm (streaming) + generate().
            "repetition_penalty",
        )
        return {k: gen[k] for k in keys if k in gen}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_model_key(self) -> str:
        return self.config.get("default_model", "0.6B-CustomVoice")

    def _base_model_key(self) -> str:
        """Return the first Base model key from config, falling back to '0.6B-Base'."""
        for key, cfg in self.config.get("models", {}).items():
            if isinstance(cfg, dict) and cfg.get("type") == "base":
                return key
        return "0.6B-Base"

    def _model_info(self, model_key: str) -> dict:
        return self.config.get("models", {}).get(model_key, {})

    async def _ensure_model_loaded(self, model_key: str) -> None:
        """Load *model_key* if it is not the currently active model."""
        import torch

        if self.current_model_key == model_key and self.model is not None:
            return

        model_info = self._model_info(model_key)
        if not model_info:
            raise ValueError(
                f"Unknown model key: '{model_key}'. "
                f"Available: {list(self.config.get('models', {}).keys())}"
            )

        hf_id = model_info["hf_id"]

        # Unload the previous model and clear any cached voice prompts
        if self.model is not None:
            logger.info(f"Unloading {self.current_model_key!r}…")
            if self._voice_prompt_cache:
                logger.info(
                    f"Clearing voice prompt cache "
                    f"({len(self._voice_prompt_cache)} entries)"
                )
                self._voice_prompt_cache.clear()
            del self.model
            self.model = None
            torch.cuda.empty_cache()

        logger.info(f"Loading {model_key!r} ({hf_id})…")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device != "cpu" else torch.float32

        from qwen_tts import Qwen3TTSModel
        torch.set_float32_matmul_precision("high")

        opt = self.config.get("optimization", {})
        attn_impl = opt.get("attention", "flash_attention_2")

        # cuDNN autotuning: the vocoder decoder is conv-heavy and streaming runs at
        # a fixed window size, so benchmark mode picks the best conv kernels once and
        # reuses them. Gated so it can be measured/disabled (optimization.cudnn_benchmark).
        if self.device != "cpu" and opt.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("cuDNN benchmark + TF32 enabled")

        try:
            self.model = Qwen3TTSModel.from_pretrained(
                hf_id,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation=attn_impl,
            )
            logger.info(f"Model loaded with {attn_impl} attention")
        except Exception as exc:
            logger.warning(f"Failed with {attn_impl}: {exc}; retrying with sdpa")
            self.model = Qwen3TTSModel.from_pretrained(
                hf_id,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation="sdpa",
            )
            logger.info("Model loaded with sdpa attention")

        # torch.compile + CUDA graph optimizations
        if opt.get("use_compile", True) and self.device != "cpu":
            await self._apply_optimizations(model_key, model_info, opt)

        self.current_model_key = model_key
        self._ready = True
        logger.info(f"Model {model_key!r} ready on {self.device}")

    async def _apply_optimizations(
        self, model_key: str, model_info: dict, opt: dict
    ) -> None:
        """Enable torch.compile and run mandatory warmup passes."""
        streaming_opts = opt.get("streaming", {})
        decode_window = streaming_opts.get("decode_window_frames", 80)
        emit_every = streaming_opts.get("emit_every_frames", 6)

        try:
            self.model.enable_streaming_optimizations(
                decode_window_frames=decode_window,
                use_compile=True,
                use_cuda_graphs=opt.get("use_cuda_graphs", False),
                compile_mode=opt.get("compile_mode", "max-autotune"),
                use_fast_codebook=opt.get("use_fast_codebook", True),
                compile_codebook_predictor=opt.get("compile_codebook_predictor", True),
            )
            logger.info(
                f"torch.compile enabled: mode={opt.get('compile_mode', 'max-autotune')}, "
                f"cuda_graphs={opt.get('use_cuda_graphs', False)}"
            )
        except Exception as exc:
            logger.warning(f"Could not enable streaming optimizations: {exc}")
            return

        # Warmup — triggers actual kernel compilation (torch.compile is lazy)
        model_type = model_info.get("type", "customvoice")
        import numpy as _np

        dummy_audio = _np.sin(
            2 * _np.pi * 440 * _np.arange(24000) / 24000
        ).astype(_np.float32)

        try:
            if model_type == "base":
                await self._warmup_base_model(dummy_audio, emit_every, decode_window)
            else:
                await self._warmup_customvoice_model(emit_every, decode_window)
        except Exception as exc:
            logger.warning(f"Warmup failed (non-critical): {exc}")

    async def _warmup_base_model(
        self,
        dummy_audio: "np.ndarray",
        emit_every: int,
        decode_window: int,
    ) -> None:
        """Three-pass warmup for the Base model (x-vector + ICL + stabilisation)."""
        logger.info("Warmup 1/3: Base — x_vector_only non-streaming…")
        self.model.generate_voice_clone(
            text="Warmup sentence.",
            language="English",
            ref_audio=(dummy_audio, 24000),
            x_vector_only_mode=True,
        )
        logger.info("Warmup 1/3: Base — x_vector_only streaming…")
        for _ in self.model.stream_generate_voice_clone(
            text="Streaming warmup for voice clone.",
            language="English",
            ref_audio=(dummy_audio, 24000),
            x_vector_only_mode=True,
            emit_every_frames=emit_every,
            decode_window_frames=decode_window,
        ):
            pass

        logger.info("Warmup 2/3: Base — ICL mode streaming…")
        for _ in self.model.stream_generate_voice_clone(
            text="Second warmup to compile ICL reference-code path.",
            language="English",
            ref_audio=(dummy_audio, 24000),
            ref_text="Warmup reference text.",
            x_vector_only_mode=False,
            emit_every_frames=emit_every,
            decode_window_frames=decode_window,
        ):
            pass

        logger.info("Warmup 3/3: Base — GPU power stabilisation…")
        for _ in self.model.stream_generate_voice_clone(
            text="Third pass to stabilise GPU power state.",
            language="English",
            ref_audio=(dummy_audio, 24000),
            x_vector_only_mode=True,
            emit_every_frames=emit_every,
            decode_window_frames=decode_window,
        ):
            pass
        logger.info("Warmup complete (base)")

    async def _warmup_customvoice_model(
        self, emit_every: int, decode_window: int
    ) -> None:
        """Three-pass warmup for the CustomVoice model."""
        logger.info("Warmup 1/3: CustomVoice — non-streaming…")
        self.model.generate_custom_voice(
            text="This is a warmup sentence to trigger torch.compile kernel compilation.",
            language="English",
            speaker="Eric",
        )
        logger.info("Warmup 1/3: CustomVoice — streaming…")
        for _ in self.model.stream_generate_custom_voice(
            text="Streaming warmup sentence for compiling the streaming code path.",
            speaker="Eric",
            language="English",
            emit_every_frames=emit_every,
            decode_window_frames=decode_window,
        ):
            pass

        logger.info("Warmup 2/3: CustomVoice — medium text…")
        for _ in self.model.stream_generate_custom_voice(
            text="A medium-length sentence to warm up different tensor shapes.",
            speaker="Eric",
            language="English",
            emit_every_frames=emit_every,
            decode_window_frames=decode_window,
        ):
            pass

        logger.info("Warmup 3/3: CustomVoice — short text…")
        for _ in self.model.stream_generate_custom_voice(
            text="Short test.",
            speaker="Eric",
            language="English",
            emit_every_frames=emit_every,
            decode_window_frames=decode_window,
        ):
            pass
        logger.info("Warmup complete (customvoice)")

    # ------------------------------------------------------------------
    # TTSBackend interface — initialisation
    # ------------------------------------------------------------------

    async def initialize(
        self, model_key: Optional[str] = None, warmup: bool = False
    ) -> None:
        """Load the default (or specified) model."""
        if model_key is None:
            model_key = self._default_model_key()
        await self._ensure_model_loaded(model_key)

    async def switch_model(self, model_key: str) -> None:
        """Hot-swap to a different model."""
        await self._ensure_model_loaded(model_key)

    # ------------------------------------------------------------------
    # TTSBackend interface — generation
    # ------------------------------------------------------------------

    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
        model: str = "tts-1",
    ) -> Tuple[np.ndarray, int]:
        """Non-streaming CustomVoice generation."""
        model_key = self._default_model_key()
        await self._ensure_model_loaded(model_key)

        self._apply_seed()
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=voice,
            instruct=instruct,
            **self._gen_kwargs(),
        )

        audio = wavs[0]
        if speed != 1.0:
            try:
                import librosa
                audio = librosa.effects.time_stretch(
                    audio.astype(np.float32), rate=speed
                )
            except ImportError:
                pass
        return audio, sr

    async def generate_speech_streaming(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
        model: str = "tts-1",
    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """
        Real token-by-token streaming via stream_generate_custom_voice.

        Yields (pcm_chunk, sample_rate) tuples as the model generates audio.
        """
        model_key = self._default_model_key()
        await self._ensure_model_loaded(model_key)

        streaming_opts = self.config.get("optimization", {}).get("streaming", {})
        decode_window_frames = streaming_opts.get("decode_window_frames", 80)
        emit_every_frames = streaming_opts.get("emit_every_frames", 6)
        gate_rms = streaming_opts.get("onset_gate_rms", _DEFAULT_ONSET_GATE_RMS)
        attack_ms = streaming_opts.get("onset_attack_ms", _DEFAULT_ONSET_ATTACK_MS)
        max_lead_ms = streaming_opts.get("onset_max_lead_ms", _DEFAULT_ONSET_MAX_LEAD_MS)
        tail_rms = streaming_opts.get("tail_silence_rms", _DEFAULT_TAIL_SILENCE_RMS)
        tail_keep_ms = streaming_opts.get("tail_keep_ms", _DEFAULT_TAIL_KEEP_MS)

        # Length-aware tempo so every reply renders near a constant target cadence
        # (config generation.target_chars_per_sec); explicit per-request speed wins.
        eff_speed = _resolve_speed(text, speed, self.config.get("generation", {}))

        self._apply_seed()
        source = self.model.stream_generate_custom_voice(
            text=text,
            speaker=voice,
            language=language,
            instruct=instruct,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
            max_frames=_max_frames_for_text(text),
            **self._gen_kwargs(),
        )
        trimmed = _trim_silence_stream(
            source, gate_rms, attack_ms, max_lead_ms, tail_rms, tail_keep_ms
        )
        for chunk, sr in _speed_stream(trimmed, eff_speed):
            yield chunk, sr

    async def generate_voice_clone(
        self,
        text: str,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str] = None,
        language: str = "Auto",
        x_vector_only_mode: bool = False,
        speed: float = 1.0,
        cache_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Non-streaming voice cloning (uses Base model)."""
        await self._ensure_model_loaded(self._base_model_key())

        t0 = time.time()

        if cache_key and cache_key in self._voice_prompt_cache:
            prompt_items = self._voice_prompt_cache[cache_key]
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt_items,
            )
            logger.info(
                f"Voice clone (cached prompt '{cache_key}'): "
                f"generate={time.time()-t0:.3f}s"
            )
        else:
            t_prompt_start = time.time()
            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=(ref_audio, ref_audio_sr),
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            t_prompt = time.time() - t_prompt_start
            if cache_key:
                self._voice_prompt_cache[cache_key] = prompt_items
                logger.info(
                    f"Voice prompt cached for '{cache_key}' "
                    f"(build={t_prompt:.3f}s)"
                )
            t_gen_start = time.time()
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt_items,
            )
            logger.info(
                f"Voice clone: prompt={t_prompt:.3f}s "
                f"generate={time.time()-t_gen_start:.3f}s"
            )

        audio = wavs[0]
        if speed != 1.0:
            try:
                import librosa
                audio = librosa.effects.time_stretch(
                    audio.astype(np.float32), rate=speed
                )
            except ImportError:
                pass
        return audio, sr

    async def generate_voice_clone_streaming(
        self,
        text: str,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str] = None,
        language: str = "Auto",
        x_vector_only_mode: bool = False,
        speed: float = 1.0,
        cache_key: Optional[str] = None,
    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """
        Real token-by-token streaming voice cloning (uses Base model).

        Yields (pcm_chunk, sample_rate) tuples as the model generates audio.
        """
        await self._ensure_model_loaded(self._base_model_key())

        streaming_opts = self.config.get("optimization", {}).get("streaming", {})
        decode_window_frames = streaming_opts.get("decode_window_frames", 80)
        emit_every_frames = streaming_opts.get("emit_every_frames", 6)

        # Build or retrieve cached voice clone prompt
        t0 = time.time()
        if cache_key and cache_key in self._voice_prompt_cache:
            prompt_items = self._voice_prompt_cache[cache_key]
        else:
            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=(ref_audio, ref_audio_sr),
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            t_prompt = time.time() - t0
            if cache_key:
                self._voice_prompt_cache[cache_key] = prompt_items
                logger.info(
                    f"Voice prompt cached for '{cache_key}' "
                    f"(build={t_prompt:.3f}s)"
                )
            else:
                logger.info(f"Voice prompt built (no cache): {time.time()-t0:.3f}s")

        gate_rms = streaming_opts.get("onset_gate_rms", _DEFAULT_ONSET_GATE_RMS)
        attack_ms = streaming_opts.get("onset_attack_ms", _DEFAULT_ONSET_ATTACK_MS)
        max_lead_ms = streaming_opts.get("onset_max_lead_ms", _DEFAULT_ONSET_MAX_LEAD_MS)
        tail_rms = streaming_opts.get("tail_silence_rms", _DEFAULT_TAIL_SILENCE_RMS)
        tail_keep_ms = streaming_opts.get("tail_keep_ms", _DEFAULT_TAIL_KEEP_MS)
        source = self.model.stream_generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt_items,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
            max_frames=_max_frames_for_text(text),
        )
        eff_speed = _resolve_speed(text, speed, self.config.get("generation", {}))
        trimmed = _trim_silence_stream(
            source, gate_rms, attack_ms, max_lead_ms, tail_rms, tail_keep_ms
        )
        for chunk, sr in _speed_stream(trimmed, eff_speed):
            yield chunk, sr

    # ------------------------------------------------------------------
    # TTSBackend interface — metadata / introspection
    # ------------------------------------------------------------------

    def get_backend_name(self) -> str:
        return "optimized"

    def get_model_id(self) -> str:
        if self.current_model_key:
            info = self._model_info(self.current_model_key)
            return info.get("hf_id", "unknown")
        return "not-loaded"

    def get_supported_voices(self) -> List[str]:
        """Return voice names listed in config.yaml (voices section)."""
        return [v["name"] for v in self.config.get("voices", [])]

    def get_supported_languages(self) -> List[str]:
        return [
            "English", "Chinese", "Japanese", "Korean", "German",
            "French", "Spanish", "Russian", "Portuguese", "Italian",
        ]

    def is_ready(self) -> bool:
        return self._ready

    def supports_voice_cloning(self) -> bool:
        return True

    def get_model_type(self) -> str:
        if not self.current_model_key:
            return "unknown"
        return self._model_info(self.current_model_key).get("type", "unknown")

    def get_device_info(self) -> Dict[str, Any]:
        try:
            import torch
        except ImportError:
            return {"device": "unknown", "gpu_available": False}

        info: Dict[str, Any] = {
            "device": str(self.device) if self.device else "unknown",
            "gpu_available": False,
        }
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_total"] = f"{props.total_memory / 1024**3:.1f} GB"
        return info

    def get_available_models(self) -> List[str]:
        return list(self.config.get("models", {}).keys())

    def get_current_model_key(self) -> Optional[str]:
        return self.current_model_key

    def get_config(self) -> dict:
        return self.config
