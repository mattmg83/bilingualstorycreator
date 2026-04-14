from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
import wave
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List

import streamlit as st
from openai import OpenAI

try:
    from pydub import AudioSegment
except ImportError:  # optional dependency for MP3-safe merge
    AudioSegment = None

# Pricing constants based on official OpenAI docs checked on 2026-04-01.
# See README for sources.
TRANSLATION_MODELS: Dict[str, Dict[str, float | str]] = {
    "gpt-5-mini": {
        "label": "GPT-5 mini",
        "input_per_mtok": 0.25,
        "output_per_mtok": 2.00,
        "notes": "Strong quality/cost balance.",
    },
    "gpt-5-nano": {
        "label": "GPT-5 nano",
        "input_per_mtok": 0.05,
        "output_per_mtok": 0.40,
        "notes": "Cheapest translation option.",
    },
    "gpt-4o-mini": {
        "label": "GPT-4o mini",
        "input_per_mtok": 0.15,
        "output_per_mtok": 0.60,
        "notes": "Very low cost, good general-purpose choice.",
    },
    "gpt-4.1-mini": {
        "label": "GPT-4.1 mini",
        "input_per_mtok": 0.40,
        "output_per_mtok": 1.60,
        "notes": "Higher-cost mid-tier translation model.",
    },
    "gpt-5": {
        "label": "GPT-5",
        "input_per_mtok": 1.25,
        "output_per_mtok": 10.00,
        "notes": "Most expensive option here; likely unnecessary for routine translation.",
    },
}

TTS_MODELS: Dict[str, Dict[str, float | str | bool]] = {
    "gpt-4o-mini-tts": {
        "label": "GPT-4o mini TTS",
        "text_input_per_mtok": 0.60,
        "audio_output_per_mtok": 12.00,
        "supports_instructions": True,
        "notes": "Best flexibility; supports speaking instructions.",
    },
    "tts-1": {
        "label": "TTS-1",
        "per_mchar": 15.00,
        "supports_instructions": False,
        "notes": "Fast traditional TTS model.",
    },
    "tts-1-hd": {
        "label": "TTS-1 HD",
        "per_mchar": 30.00,
        "supports_instructions": False,
        "notes": "Higher quality, higher cost.",
    },
}

VOICE_OPTIONS = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
]

LANGUAGE_OPTIONS = [
    "English",
    "French",
    "Spanish",
    "German",
    "Italian",
    "Portuguese",
    "Dutch",
    "Japanese",
    "Korean",
    "Chinese",
    "LinkedIn (satirical)",
]

DEFAULT_SETTINGS = {
    "source_text": "",
    "source_language": "English",
    "target_language": "French",
    "translation_model": "gpt-5-mini",
    "tts_model": "gpt-4o-mini-tts",
    "source_voice": "alloy",
    "target_voice": "fable",
    "speed": 1.0,
    "output_format": "wav",
    "output_basename": "bilingual_audio",
    "source_first": True,
    "target_duration_seconds": 30,
    "segment_length": "medium",
    "min_segment_chars": 120,
    "max_segment_chars": 800,
    "source_instructions": "Use a warm, cheerful, expressive storytelling tone for young children. Keep pacing clear and friendly.",
    "target_instructions": "Use a warm, cheerful, expressive storytelling tone for young children. Keep pacing clear and friendly.",
    "terminology_map": {},
    "translation_max_workers": 2,
}

SEGMENT_LENGTH_OPTIONS = {
    "short": 25,
    "medium": 50,
    "long": 75,
}

SOUNDFX_DIR = Path("soundfx")


@dataclass
class Segment:
    idx: int
    source_text: str
    translated_text: str = ""
    source_audio_filename: str = ""
    translated_audio_filename: str = ""
    source_chars: int = 0
    translated_chars: int = 0


@dataclass
class CostLine:
    model: str
    estimated_translation_cost: float
    estimated_tts_cost: float
    estimated_total_cost: float
    notes: str


def estimate_tokens_from_chars(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def estimate_minutes_from_chars(text: str, chars_per_minute: int = 750) -> float:
    return max(0.05, len(text) / chars_per_minute)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sentence_split(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?;:])\s+(?=[A-ZÀ-ÖØ-ÝÄËÏÖÜÂÊÎÔÛÀÂÇÉÈÊËÎÏÔÙÛÜŸÆŒ0-9\"'])", text.strip())
    chunks = [c.strip() for c in chunks if c.strip()]
    if not chunks:
        return [text.strip()] if text.strip() else []
    return chunks


def split_long_unit(unit: str, max_chars: int) -> List[str]:
    if len(unit) <= max_chars:
        return [unit]

    parts = re.split(r"(?<=[,—-])\s+", unit)
    if len(parts) == 1:
        words = unit.split()
        out: List[str] = []
        current: List[str] = []
        for word in words:
            candidate = " ".join(current + [word]).strip()
            if len(candidate) <= max_chars or not current:
                current.append(word)
            else:
                out.append(" ".join(current).strip())
                current = [word]
        if current:
            out.append(" ".join(current).strip())
        return out

    out: List[str] = []
    current = ""
    for part in parts:
        candidate = (current + " " + part).strip() if current else part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                out.append(current)
            current = part
    if current:
        out.append(current)
    return out


def segment_text(text: str, target_chars: int, min_chars: int = 120, max_chars: int = 800) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    if "##" in text:
        manual_segments = [normalize_whitespace(part) for part in text.split("##")]
        return [part for part in manual_segments if part]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    units: List[str] = []
    for para in paragraphs:
        for sentence in sentence_split(para):
            units.extend(split_long_unit(sentence, max_chars=max_chars))

    segments: List[str] = []
    current = ""
    for unit in units:
        candidate = (current + " " + unit).strip() if current else unit
        if len(candidate) <= target_chars or len(current) < min_chars:
            current = candidate
        else:
            segments.append(current)
            current = unit

    if current:
        if segments and len(current) < min_chars:
            merged = (segments[-1] + " " + current).strip()
            if len(merged) <= max_chars * 1.35:
                segments[-1] = merged
            else:
                segments.append(current)
        else:
            segments.append(current)

    return segments


@st.cache_data(ttl=3600, max_entries=64)
def cached_segment_text(text: str, target_chars: int, min_chars: int, max_chars: int) -> List[str]:
    return segment_text(text=text, target_chars=target_chars, min_chars=min_chars, max_chars=max_chars)


def segment_text_with_openai(client: OpenAI, text: str, target_words: int) -> List[str]:
    prompt = (
        "Segment the text into a JSON array of strings.\n"
        "Rules:\n"
        f"- Aim for about {target_words} words per segment.\n"
        "- Keep the original meaning and order exactly.\n"
        "- Do not omit or add content.\n"
        "- Return JSON array only, no markdown, no commentary.\n\n"
        "Text:\n"
        f"{text}"
    )
    response = client.responses.create(model="gpt-5-mini", input=prompt)
    raw_text = response.output_text.strip()
    try:
        maybe_array = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", raw_text)
        if not match:
            raise
        maybe_array = json.loads(match.group(0))
    if not isinstance(maybe_array, list):
        raise ValueError("Segmentation response is not a JSON array.")
    segments = [normalize_whitespace(str(item)) for item in maybe_array if isinstance(item, str) and str(item).strip()]
    if not segments:
        raise ValueError("Segmentation response returned an empty segment list.")
    return segments


def translate_segments(
    client: OpenAI,
    segments: List[Segment],
    source_language: str,
    target_language: str,
    model: str,
    terminology_map: Dict[str, str] | None = None,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_error: Callable[[int, int, int, str], None] | None = None,
    request_timeout_seconds: float = 45.0,
    max_retries: int = 2,
    max_workers: int = 2,
    continue_on_error: bool = False,
) -> List[Segment]:
    total = len(segments)
    if total == 0:
        return []
    safe_max_workers = max(1, min(max_workers, total))
    glossary_section = ""
    if terminology_map:
        glossary_lines = [f"- {source} = {target}" for source, target in terminology_map.items()]
        glossary_section = (
            "\n\nUse these preferred translations exactly when applicable:\n"
            + "\n".join(glossary_lines)
        )

    def translate_one_segment(seg: Segment) -> Segment:
        if target_language == "LinkedIn (satirical)":
            prompt = (
                "Rewrite the following text as a funny, satirical LinkedIn thought-leadership post. "
                "Keep it readable and concise, sprinkle in classic LinkedIn clichés and humble-brag energy, "
                f"and preserve the original meaning as much as possible. Source language: {source_language}. "
                "Return only the rewritten text.\n\n"
                f"Text:\n{seg.source_text}"
                f"{glossary_section}"
            )
        elif source_language == "LinkedIn (satirical)":
            prompt = (
                f"Translate the following text from satirical LinkedIn thought-leadership style into {target_language}. "
                "Remove cliché corporate tone while preserving the core meaning. Return only the translated text.\n\n"
                f"Text:\n{seg.source_text}"
                f"{glossary_section}"
            )
        else:
            prompt = (
                f"Translate the following text from {source_language} to {target_language}. "
                "Use the same translation for recurring terms throughout all segments. "
                "If a source term repeats, keep the exact same target term unless grammar requires inflection. "
                "Prefer child-friendly, simple vocabulary. "
                "Preserve named entities exactly unless there is a standard localized form. "
                "Return only translated text with no commentary.\n\n"
                f"Text:\n{seg.source_text}"
                f"{glossary_section}"
            )
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = create_response_with_timeout(
                    client=client,
                    model=model,
                    prompt=prompt,
                    request_timeout_seconds=request_timeout_seconds,
                )
                break
            except Exception as exc:
                if attempt >= max_retries:
                    message = (
                        f"Translation request failed for segment {seg.idx} after {max_retries + 1} attempts: {exc}"
                    )
                    raise RuntimeError(message) from exc
        if response is None:
            raise RuntimeError(
                f"Translation request failed for segment {seg.idx} after {max_retries + 1} attempts."
            )
        text = response.output_text.strip()
        return Segment(
            idx=seg.idx,
            source_text=seg.source_text,
            translated_text=text,
            source_chars=len(seg.source_text),
            translated_chars=len(text),
        )

    translated_by_idx: Dict[int, Segment] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=safe_max_workers) as executor:
        future_map = {executor.submit(translate_one_segment, seg): seg for seg in segments}
        for future in as_completed(future_map):
            seg = future_map[future]
            completed += 1
            try:
                translated_seg = future.result()
                translated_by_idx[translated_seg.idx] = translated_seg
                if on_progress:
                    on_progress(completed, total, seg.idx)
            except Exception as exc:
                message = str(exc)
                if on_error:
                    on_error(completed, total, seg.idx, message)
                if not continue_on_error:
                    raise

    return [translated_by_idx[seg.idx] for seg in segments if seg.idx in translated_by_idx]


def create_response_with_timeout(client: OpenAI, model: str, prompt: str, request_timeout_seconds: float):
    def run_request():
        try:
            return client.responses.create(model=model, input=prompt, timeout=request_timeout_seconds)
        except TypeError:
            return client.responses.create(model=model, input=prompt)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_request)
        try:
            return future.result(timeout=request_timeout_seconds + 2)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Request timed out after {request_timeout_seconds:.0f}s. "
                "This segment can be retried without rerunning everything."
            ) from exc


def write_wav_bytes(response) -> bytes:
    if hasattr(response, "read"):
        return response.read()
    if hasattr(response, "content"):
        return response.content
    if isinstance(response, bytes):
        return response
    raise TypeError("Unsupported audio response type from OpenAI SDK")


def tts_segment(
    client: OpenAI,
    text: str,
    model: str,
    voice: str,
    instructions: str,
    speed: float,
    response_format: str,
) -> bytes:
    kwargs = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": response_format,
        "speed": speed,
    }
    if TTS_MODELS[model].get("supports_instructions") and instructions.strip():
        kwargs["instructions"] = instructions.strip()
    response = client.audio.speech.create(**kwargs)
    return write_wav_bytes(response)


def concat_wav_bytes(parts: List[bytes]) -> bytes:
    if not parts:
        return b""

    params = None
    frames = []
    for part in parts:
        with wave.open(io.BytesIO(part), "rb") as w:
            if params is None:
                params = w.getparams()
            else:
                if (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getcomptype()) != (
                    params.nchannels,
                    params.sampwidth,
                    params.framerate,
                    params.comptype,
                ):
                    raise ValueError("WAV segments are not compatible for concatenation")
            frames.append(w.readframes(w.getnframes()))

    out_io = io.BytesIO()
    with wave.open(out_io, "wb") as out:
        out.setnchannels(params.nchannels)
        out.setsampwidth(params.sampwidth)
        out.setframerate(params.framerate)
        out.setcomptype(params.comptype, params.compname)
        for frame in frames:
            out.writeframes(frame)
    return out_io.getvalue()


def build_alternating_wav(source_parts: List[bytes], translated_parts: List[bytes], source_first: bool = True) -> bytes:
    ordered: List[bytes] = []
    for s, t in zip(source_parts, translated_parts):
        ordered.extend([s, t] if source_first else [t, s])
    return concat_wav_bytes(ordered)


def concat_mp3_bytes(parts: List[bytes]) -> bytes:
    if not parts:
        return b""
    if AudioSegment is None:
        raise RuntimeError(
            "MP3-safe merge requires optional dependencies. Install pydub and ffmpeg, or switch output format to WAV."
        )

    combined = AudioSegment.empty()
    for idx, part in enumerate(parts, start=1):
        try:
            combined += AudioSegment.from_file(io.BytesIO(part), format="mp3")
        except Exception as exc:
            raise RuntimeError(
                f"MP3-safe merge failed while decoding segment {idx}. Install/configure ffmpeg, or switch output format to WAV."
            ) from exc

    out_io = io.BytesIO()
    try:
        combined.export(out_io, format="mp3")
    except Exception as exc:
        raise RuntimeError("MP3-safe merge failed during MP3 export. Check ffmpeg, or switch output format to WAV.") from exc
    return out_io.getvalue()


def load_soundfx_files(extension: str) -> List[Path]:
    if not SOUNDFX_DIR.exists():
        return []
    pattern = f"*.{extension.lower()}"
    return sorted([p for p in SOUNDFX_DIR.glob(pattern) if p.is_file()])


def interleave_with_random_soundfx(parts: List[bytes], soundfx_files: List[Path]) -> tuple[List[bytes], List[str]]:
    if not parts:
        return [], []
    if len(parts) == 1 or not soundfx_files:
        return parts, []

    ordered_with_fx: List[bytes] = [parts[0]]
    used_fx: List[str] = []
    for part in parts[1:]:
        chosen_fx = random.choice(soundfx_files)
        ordered_with_fx.append(chosen_fx.read_bytes())
        ordered_with_fx.append(part)
        used_fx.append(chosen_fx.name)
    return ordered_with_fx, used_fx


def filter_compatible_wav_soundfx(soundfx_files: List[Path], reference_wav: bytes) -> tuple[List[Path], List[str]]:
    if not soundfx_files:
        return [], []

    skipped: List[str] = []
    compatible: List[Path] = []
    with wave.open(io.BytesIO(reference_wav), "rb") as ref:
        ref_sig = (ref.getnchannels(), ref.getsampwidth(), ref.getframerate(), ref.getcomptype())

    for path in soundfx_files:
        try:
            with wave.open(str(path), "rb") as fx:
                fx_sig = (fx.getnchannels(), fx.getsampwidth(), fx.getframerate(), fx.getcomptype())
            if fx_sig == ref_sig:
                compatible.append(path)
            else:
                skipped.append(path.name)
        except Exception:
            skipped.append(path.name)
    return compatible, skipped


def estimate_tts_cost(model: str, text_a: str, text_b: str) -> float:
    total_text = text_a + text_b
    if model in {"tts-1", "tts-1-hd"}:
        per_mchar = float(TTS_MODELS[model]["per_mchar"])
        return (len(total_text) / 1_000_000) * per_mchar

    text_tokens = estimate_tokens_from_chars(total_text)
    est_minutes = estimate_minutes_from_chars(total_text)
    audio_tokens = est_minutes * 1200
    text_input = (text_tokens / 1_000_000) * float(TTS_MODELS[model]["text_input_per_mtok"])
    audio_output = (audio_tokens / 1_000_000) * float(TTS_MODELS[model]["audio_output_per_mtok"])
    return text_input + audio_output


def estimate_translation_cost(model: str, source_text: str, target_language_multiplier: float = 1.1) -> float:
    input_tokens = estimate_tokens_from_chars(source_text)
    output_tokens = max(1, math.ceil(input_tokens * target_language_multiplier))
    return (
        (input_tokens / 1_000_000) * float(TRANSLATION_MODELS[model]["input_per_mtok"])
        + (output_tokens / 1_000_000) * float(TRANSLATION_MODELS[model]["output_per_mtok"])
    )


@st.cache_data(ttl=3600, max_entries=128)
def cached_comparison_table(source_text: str, translated_multiplier: float = 1.1) -> List[dict]:
    rows: List[CostLine] = []
    for t_model, t_meta in TRANSLATION_MODELS.items():
        for v_model, v_meta in TTS_MODELS.items():
            translation_cost = estimate_translation_cost(t_model, source_text, translated_multiplier)
            approx_translated_chars = math.ceil(len(source_text) * translated_multiplier)
            proxy_translated_text = "x" * approx_translated_chars
            tts_cost = estimate_tts_cost(v_model, source_text, proxy_translated_text)
            rows.append(
                CostLine(
                    model=f"{t_meta['label']} + {v_meta['label']}",
                    estimated_translation_cost=translation_cost,
                    estimated_tts_cost=tts_cost,
                    estimated_total_cost=translation_cost + tts_cost,
                    notes=f"{t_meta['notes']} / {v_meta['notes']}",
                )
            )
    rows.sort(key=lambda r: r.estimated_total_cost)
    return [asdict(r) for r in rows]


def build_zip(artifacts: Dict[str, bytes], manifest: dict) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in artifacts.items():
            zf.writestr(name, blob)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
    return out.getvalue()


def make_fingerprint(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def ensure_state() -> None:
    if "settings" not in st.session_state:
        st.session_state["settings"] = DEFAULT_SETTINGS.copy()
    else:
        for key, value in DEFAULT_SETTINGS.items():
            st.session_state["settings"].setdefault(key, value)
    if "draft_source_text" not in st.session_state:
        st.session_state["draft_source_text"] = st.session_state["settings"]["source_text"]
    if "base_segments" not in st.session_state:
        st.session_state["base_segments"] = []
    if "prepared_fingerprint" not in st.session_state:
        st.session_state["prepared_fingerprint"] = ""
    if "translated_segments" not in st.session_state:
        st.session_state["translated_segments"] = []
    if "translation_status" not in st.session_state:
        st.session_state["translation_status"] = {}
    if "translation_fingerprint" not in st.session_state:
        st.session_state["translation_fingerprint"] = ""
    if "audio_generation_fingerprint" not in st.session_state:
        st.session_state["audio_generation_fingerprint"] = ""
    if "audio_status" not in st.session_state:
        st.session_state["audio_status"] = {}
    if "segment_audio_files" not in st.session_state:
        st.session_state["segment_audio_files"] = {}
    if "artifacts" not in st.session_state:
        st.session_state["artifacts"] = {}
    if "manifest" not in st.session_state:
        st.session_state["manifest"] = {}
    if "temp_audio_dir" not in st.session_state:
        st.session_state["temp_audio_dir"] = ""
    if "api_client" not in st.session_state:
        st.session_state["api_client"] = None
    if "api_key_fingerprint" not in st.session_state:
        st.session_state["api_key_fingerprint"] = ""


def clear_audio_tempdir() -> None:
    old_dir = st.session_state.get("temp_audio_dir", "")
    if old_dir and Path(old_dir).exists():
        shutil.rmtree(old_dir, ignore_errors=True)
    st.session_state["temp_audio_dir"] = ""
    st.session_state["segment_audio_files"] = {}
    st.session_state["audio_status"] = {}
    st.session_state["artifacts"] = {}
    st.session_state["manifest"] = {}
    st.session_state["audio_generation_fingerprint"] = ""


def get_api_client(api_key: str) -> OpenAI:
    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    if st.session_state.get("api_client") is None or st.session_state.get("api_key_fingerprint") != key_hash:
        st.session_state["api_client"] = OpenAI(api_key=api_key)
        st.session_state["api_key_fingerprint"] = key_hash
    return st.session_state["api_client"]


def render_cost_panel(source_text: str, selected_translation_model: str, selected_tts_model: str) -> None:
    if not source_text.strip():
        st.info("Enter text to see estimated cost comparisons.")
        return

    rows = cached_comparison_table(source_text)
    selected_total = estimate_translation_cost(selected_translation_model, source_text) + estimate_tts_cost(
        selected_tts_model,
        source_text,
        "x" * math.ceil(len(source_text) * 1.1),
    )

    st.subheader("Estimated cost")
    st.metric("Estimated cost for current selection", f"${selected_total:.4f}")
    st.caption(
        "This is an estimate, not a bill. Translation token usage is approximated from character count, and "
        "GPT-4o mini TTS audio output is estimated from speaking duration."
    )

    table_data = [
        {
            "Model pair": r["model"],
            "Translation": round(r["estimated_translation_cost"], 4),
            "TTS": round(r["estimated_tts_cost"], 4),
            "Total": round(r["estimated_total_cost"], 4),
            "Notes": r["notes"],
        }
        for r in rows[:12]
    ]
    st.dataframe(table_data, use_container_width=True)


def parse_terminology_map(raw_value: str | dict | None) -> Dict[str, str]:
    if isinstance(raw_value, dict):
        parsed_from_dict: Dict[str, str] = {}
        for key, value in raw_value.items():
            src = str(key).strip()
            tgt = str(value).strip()
            if src and tgt:
                parsed_from_dict[src] = tgt
        return parsed_from_dict

    if not raw_value:
        return {}

    raw_text = str(raw_value).strip()
    if not raw_text:
        return {}

    try:
        maybe_json = json.loads(raw_text)
        if isinstance(maybe_json, dict):
            return parse_terminology_map(maybe_json)
    except json.JSONDecodeError:
        pass

    parsed: Dict[str, str] = {}
    for line in raw_text.splitlines():
        cleaned = line.strip()
        if not cleaned or "=" not in cleaned:
            continue
        source, target = cleaned.split("=", 1)
        source = source.strip()
        target = target.strip()
        if source and target:
            parsed[source] = target
    return parsed


def terminology_map_to_text(terminology_map: str | dict | None) -> str:
    parsed = parse_terminology_map(terminology_map)
    if not parsed:
        return ""
    return "\n".join(f"{source}={target}" for source, target in parsed.items())


def get_prepare_fingerprint(settings: dict, target_chars: int) -> str:
    payload = {
        "source_text": settings["source_text"],
        "source_language": settings["source_language"],
        "target_language": settings["target_language"],
        "translation_model": settings["translation_model"],
        "terminology_map": parse_terminology_map(settings.get("terminology_map")),
        "segment": {
            "target_chars": target_chars,
            "segment_length": settings.get("segment_length", "medium"),
            "min_segment_chars": settings["min_segment_chars"],
            "max_segment_chars": settings["max_segment_chars"],
            "target_duration_seconds": settings["target_duration_seconds"],
        },
    }
    return make_fingerprint(payload)


def get_audio_fingerprint(settings: dict, prepare_fingerprint: str) -> str:
    payload = {
        "prepare_fingerprint": prepare_fingerprint,
        "tts_model": settings["tts_model"],
        "source_voice": settings["source_voice"],
        "target_voice": settings["target_voice"],
        "speed": settings["speed"],
        "source_first": settings["source_first"],
        "output_format": settings["output_format"],
        "source_instructions": settings["source_instructions"],
        "target_instructions": settings["target_instructions"],
    }
    return make_fingerprint(payload)


def create_segments_from_texts(segment_texts: List[str]) -> List[Segment]:
    return [Segment(idx=i + 1, source_text=s, source_chars=len(s)) for i, s in enumerate(segment_texts)]


def load_segment_bytes(segment_audio_files: dict, segments: List[Segment], side: str) -> List[bytes]:
    blobs: List[bytes] = []
    for seg in segments:
        file_path = segment_audio_files.get(seg.idx, {}).get(side)
        if not file_path:
            raise ValueError(f"Missing {side} audio file for segment {seg.idx}")
        blobs.append(Path(file_path).read_bytes())
    return blobs


def render_prepare_tab(active_api_key: str) -> None:
    settings = st.session_state["settings"]
    left, right = st.columns([1.2, 1])

    with left:
        st.text_area(
            "Source text",
            key="draft_source_text",
            height=300,
            placeholder="Paste text here...",
        )
        st.caption(
            "Segmentation mode: add `##` between chunks for manual segmentation. "
            "If `##` is not present, AI segmentation is used (with automatic fallback if needed)."
        )

        with st.form("main_controls_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                source_language = st.selectbox(
                    "Source language", LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index(settings["source_language"])
                )
            with c2:
                target_language = st.selectbox(
                    "Target language", LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index(settings["target_language"])
                )
            with c3:
                target_duration_seconds = st.slider(
                    "Target segment duration (seconds)", min_value=3, max_value=60, value=settings["target_duration_seconds"], step=1
                )

            min_segment_chars = st.slider(
                "Minimum segment characters", min_value=50, max_value=500, value=settings["min_segment_chars"], step=10
            )
            max_segment_chars = st.slider(
                "Maximum segment characters", min_value=50, max_value=1200, value=settings["max_segment_chars"], step=20
            )
            segment_length = st.selectbox(
                "Segment length",
                options=list(SEGMENT_LENGTH_OPTIONS.keys()),
                index=list(SEGMENT_LENGTH_OPTIONS.keys()).index(settings.get("segment_length", "medium")),
                help="Used by AI segmentation when no manual `##` markers are present.",
            )

            st.divider()
            with st.expander("Generation controls", expanded=False):
                gc1, gc2 = st.columns(2)
                with gc1:
                    translation_model = st.selectbox(
                        "Translation model",
                        options=list(TRANSLATION_MODELS.keys()),
                        index=list(TRANSLATION_MODELS.keys()).index(settings["translation_model"]),
                        format_func=lambda k: TRANSLATION_MODELS[k]["label"],
                    )
                with gc2:
                    tts_model = st.selectbox(
                        "TTS model",
                        options=list(TTS_MODELS.keys()),
                        index=list(TTS_MODELS.keys()).index(settings["tts_model"]),
                        format_func=lambda k: TTS_MODELS[k]["label"],
                    )

                gc3, gc4 = st.columns(2)
                with gc3:
                    source_voice = st.selectbox("Source voice", VOICE_OPTIONS, index=VOICE_OPTIONS.index(settings["source_voice"]))
                with gc4:
                    target_voice = st.selectbox("Target voice", VOICE_OPTIONS, index=VOICE_OPTIONS.index(settings["target_voice"]))

                gc5, gc6 = st.columns(2)
                with gc5:
                    speed = st.slider("Speech speed", min_value=0.75, max_value=1.25, value=settings["speed"], step=0.05)
                with gc6:
                    output_format = st.selectbox(
                        "Audio output format", options=["wav", "mp3"], index=["wav", "mp3"].index(settings["output_format"])
                    )

                output_basename = st.text_input(
                    "Output audio base name",
                    value=settings["output_basename"],
                    help="Used for exported full audio downloads (example: lesson_01).",
                )
                source_first = st.toggle("Source language first in alternating file", value=settings["source_first"])

                source_instructions = st.text_input(
                    "Source voice instructions (only for GPT-4o mini TTS)", value=settings["source_instructions"]
                )
                target_instructions = st.text_input(
                    "Target voice instructions (only for GPT-4o mini TTS)", value=settings["target_instructions"]
                )
                terminology_map_text = st.text_area(
                    "Preferred terms (source=target per line)",
                    value=terminology_map_to_text(settings.get("terminology_map")),
                    height=120,
                    help="Optional glossary. Malformed lines are ignored.",
                )
            submitted = st.form_submit_button("Apply settings & prepare", type="primary", use_container_width=True)

        if submitted:
            if min_segment_chars > max_segment_chars:
                min_segment_chars = max_segment_chars
                st.warning("Minimum segment characters cannot exceed maximum. Using maximum value for both.")

            updated = {
                "source_text": st.session_state.get("draft_source_text", ""),
                "source_language": source_language,
                "target_language": target_language,
                "translation_model": translation_model,
                "tts_model": tts_model,
                "source_voice": source_voice,
                "target_voice": target_voice,
                "speed": speed,
                "output_format": output_format,
                "output_basename": output_basename.strip() or "bilingual_audio",
                "source_first": source_first,
                "target_duration_seconds": target_duration_seconds,
                "segment_length": segment_length,
                "min_segment_chars": min_segment_chars,
                "max_segment_chars": max_segment_chars,
                "source_instructions": source_instructions,
                "target_instructions": target_instructions,
                "terminology_map": parse_terminology_map(terminology_map_text),
                "translation_max_workers": settings.get("translation_max_workers", 2),
            }
            chars_per_minute = 760
            target_chars = round(updated["target_duration_seconds"] * chars_per_minute / 60)
            target_chars = max(updated["min_segment_chars"], min(target_chars, updated["max_segment_chars"]))
            prepare_fingerprint = get_prepare_fingerprint(updated, target_chars)

            if prepare_fingerprint != st.session_state.get("prepared_fingerprint"):
                clear_audio_tempdir()
                st.session_state["translated_segments"] = []
                st.session_state["translation_status"] = {}
                st.session_state["translation_fingerprint"] = ""
                source_text = updated["source_text"]
                uses_manual_markers = "##" in source_text
                segment_texts: List[str]
                if uses_manual_markers:
                    segment_texts = cached_segment_text(
                        text=source_text,
                        target_chars=target_chars,
                        min_chars=updated["min_segment_chars"],
                        max_chars=updated["max_segment_chars"],
                    )
                else:
                    target_words = SEGMENT_LENGTH_OPTIONS.get(updated.get("segment_length", "medium"), 50)
                    try:
                        client = get_api_client(active_api_key)
                        segment_texts = segment_text_with_openai(client=client, text=source_text, target_words=target_words)
                    except Exception as exc:
                        st.error(
                            "AI segmentation failed. Falling back to heuristic segmentation. "
                            f"Details: {exc}"
                        )
                        segment_texts = cached_segment_text(
                            text=source_text,
                            target_chars=target_chars,
                            min_chars=updated["min_segment_chars"],
                            max_chars=updated["max_segment_chars"],
                        )
                st.session_state["base_segments"] = create_segments_from_texts(segment_texts)
                st.session_state["prepared_fingerprint"] = prepare_fingerprint
                st.success("Prepared new segments from current settings.")
            else:
                st.info("Inputs unchanged. Reusing existing prepared segmentation.")
            st.session_state["settings"] = updated

        chars_per_minute = 760
        target_chars = round(settings["target_duration_seconds"] * chars_per_minute / 60)
        target_chars = max(settings["min_segment_chars"], min(target_chars, settings["max_segment_chars"]))
        st.caption(
            f"Heuristic fallback target: ~{target_chars} chars "
            f"({settings['target_duration_seconds']}s at ~{chars_per_minute} chars/min). "
            f"AI target words: ~{SEGMENT_LENGTH_OPTIONS.get(settings.get('segment_length', 'medium'), 50)}."
        )

        if active_api_key:
            st.success("OpenAI API key is configured for this session.")

        if settings["source_text"].strip() and st.session_state["base_segments"]:
            segments_preview = st.session_state["base_segments"]
            if "##" in settings["source_text"]:
                st.info("Manual segmentation mode is active (using `##` markers).")
            else:
                st.info("AI segmentation mode is active (target length applied; heuristic fallback on errors).")
            with st.expander(f"Segment preview ({len(segments_preview)} segments)", expanded=False):
                for seg in segments_preview:
                    st.markdown(f"**Segment {seg.idx}** ({seg.source_chars} chars)")
                    st.write(seg.source_text)

    with right:
        st.caption("Updates live as you type; final pipeline still uses prepared settings.")
        live_source_text = st.session_state.get("draft_source_text", settings["source_text"])
        with st.expander("Cost estimation", expanded=False):
            render_cost_panel(live_source_text, settings["translation_model"], settings["tts_model"])


def render_translate_tab(api_key: str) -> None:
    settings = st.session_state["settings"]
    st.subheader("Translate")

    if not settings["source_text"].strip():
        st.info("Go to Prepare and enter source text first.")
        return
    if not st.session_state["base_segments"]:
        st.info("Go to Prepare and click 'Apply settings & prepare' to segment text.")
        return
    if settings["source_language"] == settings["target_language"]:
        st.error("Choose two different languages in Prepare.")
        return
    if not api_key.strip():
        st.error("Set OPENAI_API_KEY in Streamlit secrets/environment or enter it in the sidebar.")
        return

    base_segments = st.session_state["base_segments"]
    translation_fp = st.session_state["prepared_fingerprint"]
    translated_map = {seg.idx: seg for seg in st.session_state.get("translated_segments", [])}
    status_map = st.session_state.get("translation_status", {})

    for seg in base_segments:
        status_map.setdefault(seg.idx, {"done": seg.idx in translated_map, "error": ""})
    st.session_state["translation_status"] = status_map

    failed_indices = [seg.idx for seg in base_segments if status_map.get(seg.idx, {}).get("error")]
    c1, c2 = st.columns(2)
    run_all = c1.button("Run translation", type="primary", use_container_width=True, key="translation_run_all")
    retry_failed = c2.button(
        "Retry failed segments",
        use_container_width=True,
        disabled=not failed_indices,
        key="translation_retry_failed",
    )

    if run_all or retry_failed:
        if run_all and st.session_state.get("translation_fingerprint") == translation_fp and not failed_indices and translated_map:
            st.info("Translation already exists for these inputs. Reusing previous result.")
        else:
            segments_to_run = (
                [seg for seg in base_segments if seg.idx in failed_indices]
                if retry_failed
                else base_segments
            )
            if not segments_to_run:
                st.info("No failed segments to retry.")
            else:
                if retry_failed:
                    for idx in failed_indices:
                        status_map[idx] = {"done": False, "error": ""}
                else:
                    translated_map = {}
                    status_map = {seg.idx: {"done": False, "error": ""} for seg in base_segments}

                progress_bar = st.progress(0.0)
                status_text = st.empty()
                try:
                    client = get_api_client(api_key.strip())
                    total = len(segments_to_run)

                    def on_progress(i: int, run_total: int, seg_idx: int) -> None:
                        progress_bar.progress(i / run_total)
                        status_text.info(f"Translated segment {seg_idx} ({i}/{run_total})")

                    def on_error(i: int, run_total: int, seg_idx: int, message: str) -> None:
                        status_map[seg_idx] = {"done": False, "error": f"Segment {seg_idx}: {message}"}
                        progress_bar.progress(i / run_total)
                        status_text.warning(f"Failed segment {seg_idx} ({i}/{run_total})")

                    with st.status("Translating segments", expanded=True) as status:
                        translated_subset = translate_segments(
                            client=client,
                            segments=segments_to_run,
                            source_language=settings["source_language"],
                            target_language=settings["target_language"],
                            model=settings["translation_model"],
                            terminology_map=parse_terminology_map(settings.get("terminology_map")),
                            on_progress=on_progress,
                            on_error=on_error,
                            max_workers=int(settings.get("translation_max_workers", 2)),
                            continue_on_error=True,
                        )
                        for seg in translated_subset:
                            translated_map[seg.idx] = seg
                            status_map[seg.idx] = {"done": True, "error": ""}
                        status.update(label="Translation run complete", state="complete")
                        status_text.success(f"Completed {len(translated_subset)}/{total} requested segments.")
                except Exception as exc:
                    message = f"Translation run stopped: {exc}"
                    status_text.error(message)
                    st.error(message)

                missing = [seg.idx for seg in segments_to_run if seg.idx not in translated_map]
                if missing:
                    for idx in missing:
                        status_map[idx] = {"done": False, "error": f"Segment {idx} failed: translation missing after run."}

                st.session_state["translated_segments"] = [translated_map[seg.idx] for seg in base_segments if seg.idx in translated_map]
                st.session_state["translation_status"] = status_map

                all_done = len(st.session_state["translated_segments"]) == len(base_segments) and not any(
                    s.get("error") for s in status_map.values()
                )
                if all_done:
                    st.session_state["translation_fingerprint"] = translation_fp
                    clear_audio_tempdir()
                else:
                    st.session_state["translation_fingerprint"] = ""

    translated_segments = st.session_state.get("translated_segments", [])
    status_rows = []
    for seg in base_segments:
        translated = translated_map.get(seg.idx)
        seg_status = st.session_state.get("translation_status", {}).get(seg.idx, {})
        status_rows.append(
            {
                "#": seg.idx,
                "Status": "✅ Done" if seg_status.get("done") else "❌ Failed" if seg_status.get("error") else "— Pending",
                "Error": seg_status.get("error", ""),
                "Source chars": seg.source_chars,
                "Target chars": translated.translated_chars if translated else 0,
                "Source text": seg.source_text,
                "Translated text": translated.translated_text if translated else "",
            }
        )
    st.dataframe(status_rows, use_container_width=True)


def generate_audio_for_indices(api_key: str, indices: List[int]) -> None:
    settings = st.session_state["settings"]
    translated_segments = st.session_state.get("translated_segments", [])
    if not translated_segments:
        st.error("Run translation first.")
        return

    audio_fp = get_audio_fingerprint(settings, st.session_state["prepared_fingerprint"])
    if not st.session_state.get("temp_audio_dir"):
        st.session_state["temp_audio_dir"] = tempfile.mkdtemp(prefix="bilingual_audio_")

    segment_map = {seg.idx: seg for seg in translated_segments}
    temp_audio_dir = Path(st.session_state["temp_audio_dir"])
    output_format = settings["output_format"]
    client = get_api_client(api_key.strip())
    progress = st.progress(0.0)

    def generate_one_side(seg_idx: int, side: str, text: str, voice: str, instructions: str) -> dict:
        blob = tts_segment(
            client=client,
            text=text,
            model=settings["tts_model"],
            voice=voice,
            instructions=instructions,
            speed=settings["speed"],
            response_format=output_format,
        )
        file_path = temp_audio_dir / f"{side}_{seg_idx:03d}.{output_format}"
        file_path.write_bytes(blob)
        rel_name = f"segments/{side}_{seg_idx:03d}.{output_format}"
        return {"seg_idx": seg_idx, "side": side, "path": str(file_path), "filename": rel_name}

    tasks: List[tuple[int, str, str, str, str]] = []
    for seg_idx in indices:
        seg = segment_map[seg_idx]
        seg_files = st.session_state["segment_audio_files"].setdefault(seg_idx, {})
        seg_status = st.session_state["audio_status"].setdefault(seg_idx, {"source": False, "target": False, "error": ""})

        source_ok = bool(seg_status.get("source")) and bool(seg_files.get("source")) and Path(seg_files["source"]).exists()
        target_ok = bool(seg_status.get("target")) and bool(seg_files.get("target")) and Path(seg_files["target"]).exists()

        if not source_ok:
            tasks.append((seg_idx, "source", seg.source_text, settings["source_voice"], settings["source_instructions"]))
        if not target_ok:
            tasks.append((seg_idx, "target", seg.translated_text, settings["target_voice"], settings["target_instructions"]))

    with st.status("Generating audio", expanded=True) as status:
        total_steps = max(1, len(tasks))
        completed = 0
        had_failure = False

        if tasks:
            max_workers = min(3, len(tasks))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(generate_one_side, seg_idx, side, text, voice, instructions): (seg_idx, side)
                    for seg_idx, side, text, voice, instructions in tasks
                }

                for future in as_completed(future_map):
                    seg_idx, side = future_map[future]
                    seg_status = st.session_state["audio_status"].setdefault(
                        seg_idx, {"source": False, "target": False, "error": ""}
                    )
                    try:
                        result = future.result()
                        seg_files = st.session_state["segment_audio_files"].setdefault(seg_idx, {})
                        seg_files[side] = result["path"]
                        if side == "source":
                            segment_map[seg_idx].source_audio_filename = result["filename"]
                            seg_status["source"] = True
                        else:
                            segment_map[seg_idx].translated_audio_filename = result["filename"]
                            seg_status["target"] = True
                        if seg_status.get("source") and seg_status.get("target"):
                            seg_status["error"] = ""
                    except Exception as exc:
                        had_failure = True
                        seg_status["error"] = str(exc)
                        st.error(f"Segment {seg_idx} {side} failed: {exc}")

                    completed += 1
                    progress.progress(completed / total_steps)
        else:
            progress.progress(1.0)

        status.update(label="Audio generation run complete", state="complete")

    st.session_state["translated_segments"] = translated_segments

    if tasks and had_failure:
        st.warning("Some segments failed. Review failures below and retry failed segments.")
        return

    all_done = True
    for seg in translated_segments:
        seg_status = st.session_state["audio_status"].get(seg.idx, {})
        if not (seg_status.get("source") and seg_status.get("target")):
            all_done = False
            break

    if not all_done:
        st.warning("Some segments failed. Review failures below and retry failed segments.")
        return

    segment_audio_files = st.session_state["segment_audio_files"]
    source_parts = load_segment_bytes(segment_audio_files, translated_segments, "source")
    target_parts = load_segment_bytes(segment_audio_files, translated_segments, "target")
    used_soundfx: List[str] = []
    soundfx_files = load_soundfx_files(settings["output_format"])

    if settings["output_format"] == "wav":
        full_source = concat_wav_bytes(source_parts)
        full_target = concat_wav_bytes(target_parts)
        ordered_wav: List[bytes] = []
        for s, t in zip(source_parts, target_parts):
            ordered_wav.extend([s, t] if settings["source_first"] else [t, s])
        compatible_soundfx, skipped_soundfx = filter_compatible_wav_soundfx(soundfx_files, ordered_wav[0])
        if skipped_soundfx:
            st.info(
                "Skipped incompatible WAV sound effects: "
                + ", ".join(skipped_soundfx[:5])
                + ("..." if len(skipped_soundfx) > 5 else "")
            )
        alternating_parts, used_soundfx = interleave_with_random_soundfx(ordered_wav, compatible_soundfx)
        alternating = concat_wav_bytes(alternating_parts)
    else:
        try:
            full_source = concat_mp3_bytes(source_parts)
            full_target = concat_mp3_bytes(target_parts)
            ordered_mp3: List[bytes] = []
            for s, t in zip(source_parts, target_parts):
                ordered_mp3.extend([s, t] if settings["source_first"] else [t, s])
            alternating_parts, used_soundfx = interleave_with_random_soundfx(ordered_mp3, soundfx_files)
            alternating = concat_mp3_bytes(alternating_parts)
        except RuntimeError as exc:
            st.error(str(exc))
            st.error("Tip: set audio output format to WAV in Prepare (WAV concatenation is built in).")
            return

    artifacts: Dict[str, bytes] = {
        f"full_source.{settings['output_format']}": full_source,
        f"full_target.{settings['output_format']}": full_target,
        f"alternating_bilingual.{settings['output_format']}": alternating,
    }

    for seg in translated_segments:
        source_path = st.session_state["segment_audio_files"][seg.idx]["source"]
        target_path = st.session_state["segment_audio_files"][seg.idx]["target"]
        artifacts[seg.source_audio_filename] = Path(source_path).read_bytes()
        artifacts[seg.translated_audio_filename] = Path(target_path).read_bytes()

    manifest = {
        "pipeline": "text_only_bilingual_recomposer",
        "source_language": settings["source_language"],
        "target_language": settings["target_language"],
        "translation_model": settings["translation_model"],
        "tts_model": settings["tts_model"],
        "source_voice": settings["source_voice"],
        "target_voice": settings["target_voice"],
        "speed": settings["speed"],
        "source_first": settings["source_first"],
        "output_format": settings["output_format"],
        "output_basename": settings["output_basename"],
        "alternating_track_transition_soundfx": {
            "enabled": len(soundfx_files) > 0,
            "available_files": [p.name for p in soundfx_files],
            "used_files_in_order": used_soundfx,
            "notes": (
                f"Transition sound effects are inserted between alternating segments for {settings['output_format'].upper()} output when matching soundfx files exist."
            ),
        },
        "segmentation_settings": {
            "target_duration_seconds": settings["target_duration_seconds"],
            "chars_per_minute": 760,
            "target_chars": round(settings["target_duration_seconds"] * 760 / 60),
            "min_segment_chars": settings["min_segment_chars"],
            "max_segment_chars": settings["max_segment_chars"],
        },
        "segments": [asdict(s) for s in translated_segments],
        "estimated_cost_usd": {
            "translation": round(estimate_translation_cost(settings["translation_model"], settings["source_text"]), 6),
            "tts": round(
                estimate_tts_cost(
                    settings["tts_model"],
                    settings["source_text"],
                    " ".join(s.translated_text for s in translated_segments),
                ),
                6,
            ),
        },
    }
    manifest["estimated_cost_usd"]["total"] = round(
        manifest["estimated_cost_usd"]["translation"] + manifest["estimated_cost_usd"]["tts"], 6
    )

    st.session_state["artifacts"] = artifacts
    st.session_state["manifest"] = manifest
    st.session_state["audio_generation_fingerprint"] = audio_fp
    st.success("Audio artifacts ready for export.")


def render_audio_tab(api_key: str) -> None:
    settings = st.session_state["settings"]
    st.subheader("Generate Audio")

    if not st.session_state.get("translated_segments"):
        st.info("Run translation first.")
        return
    if len(st.session_state["translated_segments"]) < len(st.session_state.get("base_segments", [])):
        st.warning("Some segments are not translated yet. Retry failed translation segments before generating audio.")
        return
    if any(v.get("error") for v in st.session_state.get("translation_status", {}).values()):
        st.warning("There are translation errors. Retry failed translation segments before generating audio.")
        return

    if not api_key.strip():
        st.error("Set OPENAI_API_KEY in Streamlit secrets/environment or enter it in the sidebar.")
        return

    current_audio_fp = get_audio_fingerprint(settings, st.session_state["prepared_fingerprint"])
    if st.session_state.get("audio_generation_fingerprint") == current_audio_fp and st.session_state.get("artifacts"):
        st.info("Audio already matches current inputs. Reusing existing artifacts.")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(
            "Generate / refresh all audio",
            type="primary",
            use_container_width=True,
            key="audio_generate_refresh_all",
        ):
            clear_audio_tempdir()
            indices = [seg.idx for seg in st.session_state["translated_segments"]]
            generate_audio_for_indices(api_key=api_key, indices=indices)
    with col_b:
        failed_indices = [
            idx for idx, status in st.session_state.get("audio_status", {}).items() if status.get("error") or not (status.get("source") and status.get("target"))
        ]
        if st.button(
            "Retry failed segments",
            use_container_width=True,
            disabled=not failed_indices,
            key="audio_retry_failed",
        ):
            generate_audio_for_indices(api_key=api_key, indices=sorted(failed_indices))

    translated_segments = st.session_state["translated_segments"]
    status_rows = []
    for seg in translated_segments:
        status = st.session_state.get("audio_status", {}).get(seg.idx, {})
        status_rows.append(
            {
                "#": seg.idx,
                "Source audio": "✅" if status.get("source") else "—",
                "Target audio": "✅" if status.get("target") else "—",
                "Error": status.get("error", ""),
            }
        )
    st.dataframe(status_rows, use_container_width=True)


def render_export_tab() -> None:
    artifacts = st.session_state.get("artifacts", {})
    manifest = st.session_state.get("manifest", {})
    settings = st.session_state.get("settings", {})
    st.subheader("Export")

    if not artifacts:
        st.info("Generate audio first, then downloads will appear here.")
        return

    full_keys = [
        f"alternating_bilingual.{manifest.get('output_format', 'wav')}",
        f"full_source.{manifest.get('output_format', 'wav')}",
        f"full_target.{manifest.get('output_format', 'wav')}",
    ]
    mime_type = "audio/wav" if manifest.get("output_format", "wav") == "wav" else "audio/mpeg"
    audio_format = "audio/wav" if manifest.get("output_format", "wav") == "wav" else "audio/mp3"

    output_basename = settings.get("output_basename", "bilingual_audio").strip() or "bilingual_audio"
    custom_names = {
        full_keys[0]: f"{output_basename}.{manifest.get('output_format', 'wav')}",
        full_keys[1]: f"{output_basename}_source.{manifest.get('output_format', 'wav')}",
        full_keys[2]: f"{output_basename}_target.{manifest.get('output_format', 'wav')}",
    }

    for key in full_keys:
        if key in artifacts:
            st.download_button(
                label=f"Download {custom_names[key]}",
                data=artifacts[key],
                file_name=custom_names[key],
                mime=mime_type,
            )
            st.audio(artifacts[key], format=audio_format)

    zip_blob = build_zip(artifacts, manifest)
    st.download_button(
        label="Download full ZIP package",
        data=zip_blob,
        file_name="bilingual_audio_package.zip",
        mime="application/zip",
    )

    st.subheader("Manifest")
    st.json(manifest)


def main() -> None:
    st.set_page_config(page_title="Bilingual story creator", layout="wide")
    st.title("Bilingual story creator")
    st.write(
        "Generate a bilingual story from just text. One story generates three spoken audio files: "
        "two single language versions and a combined alternating bilingual version"
    )

    ensure_state()

    configured_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    with st.sidebar:
        st.header("OpenAI settings")
        manual_api_key = st.text_input(
            "OpenAI API key (optional override)",
            type="password",
            value=st.session_state.get("manual_api_key", ""),
            help="If provided, this key is used for API calls. If blank, app falls back to secrets/environment key.",
        )
        st.session_state["manual_api_key"] = manual_api_key
        api_key = manual_api_key.strip() or configured_api_key
        if manual_api_key.strip():
            st.caption("Using API key from sidebar input.")
        elif configured_api_key:
            st.caption("Using OPENAI_API_KEY from secrets/environment.")
        else:
            st.caption("No API key set yet.")
        worker_limit = st.slider(
            "Translation parallel workers",
            min_value=1,
            max_value=4,
            value=int(st.session_state["settings"].get("translation_max_workers", 2)),
            step=1,
            help="Lower this if you hit rate limits. 2-3 is usually safest on Streamlit Cloud.",
        )
        st.session_state["settings"]["translation_max_workers"] = worker_limit
        st.markdown(
            "Try and compare available voices on the official OpenAI voice page: "
            "[openai.fm](https://www.openai.fm/)."
        )

    tab_prepare, tab_translate, tab_audio, tab_export = st.tabs(["Prepare", "Translate", "Generate Audio", "Export"])

    with tab_prepare:
        render_prepare_tab(active_api_key=api_key)
    with tab_translate:
        render_translate_tab(api_key=api_key)
    with tab_audio:
        render_audio_tab(api_key=api_key)
    with tab_export:
        render_export_tab()


if __name__ == "__main__":
    main()
