from __future__ import annotations

import io
import json
import math
import re
import wave
import zipfile
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from openai import OpenAI

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
]


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
    # Deliberately conservative rough estimate for planning UI.
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


def translate_segments(
    client: OpenAI,
    segments: List[Segment],
    source_language: str,
    target_language: str,
    model: str,
) -> List[Segment]:
    translated: List[Segment] = []
    for seg in segments:
        prompt = (
            f"Translate the following text from {source_language} to {target_language}. "
            "Preserve meaning and tone, keep it natural, and return only the translated text.\n\n"
            f"Text:\n{seg.source_text}"
        )
        response = client.responses.create(model=model, input=prompt)
        text = response.output_text.strip()
        translated.append(
            Segment(
                idx=seg.idx,
                source_text=seg.source_text,
                translated_text=text,
                source_chars=len(seg.source_text),
                translated_chars=len(text),
            )
        )
    return translated


def write_wav_bytes(response) -> bytes:
    # SDK returns binary content helper on audio endpoints.
    # Try common helper methods first, then raw bytes fallback.
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
) -> bytes:
    kwargs = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": "wav",
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
        if source_first:
            ordered.extend([s, t])
        else:
            ordered.extend([t, s])
    return concat_wav_bytes(ordered)


def estimate_tts_cost(model: str, text_a: str, text_b: str) -> float:
    total_text = text_a + text_b
    if model in {"tts-1", "tts-1-hd"}:
        per_mchar = float(TTS_MODELS[model]["per_mchar"])
        return (len(total_text) / 1_000_000) * per_mchar

    text_tokens = estimate_tokens_from_chars(total_text)
    est_minutes = estimate_minutes_from_chars(total_text)
    # Official realtime cost guide states assistant audio messages use 1 audio token per 50ms.
    # 60 sec / 0.05 sec = 1200 audio tokens per minute.
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


def comparison_table(source_text: str, translated_multiplier: float = 1.1) -> List[CostLine]:
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
    return rows


def build_zip(artifacts: Dict[str, bytes], manifest: dict) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in artifacts.items():
            zf.writestr(name, blob)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
    return out.getvalue()


def render_cost_panel(source_text: str, selected_translation_model: str, selected_tts_model: str) -> None:
    if not source_text.strip():
        st.info("Enter text to see estimated cost comparisons.")
        return

    rows = comparison_table(source_text)
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
            "Model pair": r.model,
            "Translation": round(r.estimated_translation_cost, 4),
            "TTS": round(r.estimated_tts_cost, 4),
            "Total": round(r.estimated_total_cost, 4),
            "Notes": r.notes,
        }
        for r in rows[:12]
    ]
    st.dataframe(table_data, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Bilingual Text-to-Audio Composer", layout="wide")
    st.title("Bilingual Text-to-Audio Composer")
    st.write(
        "Generate source-language audio, translated-language audio, and an alternating bilingual WAV file from raw text."
    )

    if "segments" not in st.session_state:
        st.session_state["segments"] = []
    if "artifacts" not in st.session_state:
        st.session_state["artifacts"] = {}
    if "manifest" not in st.session_state:
        st.session_state["manifest"] = {}

    configured_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")

    with st.sidebar:
        st.header("OpenAI settings")
        if configured_api_key:
            st.success("Using OPENAI_API_KEY from secrets/environment.")
            api_key = configured_api_key
        else:
            api_key = st.text_input("OpenAI API key", type="password", help="Used only for this session.")
        translation_model = st.selectbox(
            "Translation model",
            options=list(TRANSLATION_MODELS.keys()),
            format_func=lambda k: TRANSLATION_MODELS[k]["label"],
        )
        tts_model = st.selectbox(
            "TTS model",
            options=list(TTS_MODELS.keys()),
            format_func=lambda k: TTS_MODELS[k]["label"],
        )
        source_voice = st.selectbox("Source voice", VOICE_OPTIONS, index=0)
        target_voice = st.selectbox("Target voice", VOICE_OPTIONS, index=6)
        speed = st.slider("Speech speed", min_value=0.75, max_value=1.25, value=1.0, step=0.05)
        source_first = st.toggle("Source language first in alternating file", value=True)

    left, right = st.columns([1.2, 1])

    with left:
        source_text = st.text_area("Source text", height=340, placeholder="Paste text here...")
        c1, c2, c3 = st.columns(3)
        with c1:
            source_language = st.selectbox("Source language", LANGUAGE_OPTIONS, index=0)
        with c2:
            target_language = st.selectbox("Target language", LANGUAGE_OPTIONS, index=1)
        with c3:
            segment_style = st.selectbox("Target segment size", ["Short", "Medium", "Long"], index=1)

        target_chars_map = {"Short": 220, "Medium": 380, "Long": 650}
        target_chars = target_chars_map[segment_style]

        source_instructions = st.text_input(
            "Source voice instructions (only for GPT-4o mini TTS)",
            value="Speak clearly and naturally.",
        )
        target_instructions = st.text_input(
            "Target voice instructions (only for GPT-4o mini TTS)",
            value="Speak clearly and naturally.",
        )

        preview_only = st.checkbox("Preview segmentation only", value=False)
        generate = st.button("Generate bilingual audio", type="primary", use_container_width=True)

    with right:
        render_cost_panel(source_text, translation_model, tts_model)

    if source_text.strip():
        segments_preview = [
            Segment(idx=i + 1, source_text=s, source_chars=len(s))
            for i, s in enumerate(segment_text(source_text, target_chars=target_chars))
        ]
        with st.expander(f"Segment preview ({len(segments_preview)} segments)", expanded=False):
            for seg in segments_preview:
                st.markdown(f"**Segment {seg.idx}** ({seg.source_chars} chars)")
                st.write(seg.source_text)

    if generate:
        if not api_key.strip():
            st.error("Set OPENAI_API_KEY in Streamlit secrets (recommended) or enter it in the sidebar.")
            return
        if not source_text.strip():
            st.error("Enter source text.")
            return
        if source_language == target_language:
            st.error("Choose two different languages.")
            return

        client = OpenAI(api_key=api_key.strip())
        base_segments = [
            Segment(idx=i + 1, source_text=s, source_chars=len(s))
            for i, s in enumerate(segment_text(source_text, target_chars=target_chars))
        ]

        if not base_segments:
            st.error("No segments were produced from the input text.")
            return

        st.session_state["segments"] = []
        st.session_state["artifacts"] = {}
        st.session_state["manifest"] = {}

        try:
            with st.status("Running pipeline", expanded=True) as status:
                st.write(f"Segmented input into {len(base_segments)} chunks.")
                translated_segments = translate_segments(
                    client=client,
                    segments=base_segments,
                    source_language=source_language,
                    target_language=target_language,
                    model=translation_model,
                )
                st.write("Translation complete.")

                if preview_only:
                    st.session_state["segments"] = translated_segments
                    status.update(label="Preview complete", state="complete")
                    st.success("Preview ready. Audio generation was skipped.")
                    return

                source_audio_parts: List[bytes] = []
                target_audio_parts: List[bytes] = []
                artifacts: Dict[str, bytes] = {}

                progress = st.progress(0.0)
                total_steps = len(translated_segments) * 2
                completed = 0

                for seg in translated_segments:
                    source_wav = tts_segment(
                        client=client,
                        text=seg.source_text,
                        model=tts_model,
                        voice=source_voice,
                        instructions=source_instructions,
                        speed=speed,
                    )
                    source_name = f"segments/source_{seg.idx:03d}.wav"
                    artifacts[source_name] = source_wav
                    source_audio_parts.append(source_wav)
                    seg.source_audio_filename = source_name
                    completed += 1
                    progress.progress(completed / total_steps)

                    target_wav = tts_segment(
                        client=client,
                        text=seg.translated_text,
                        model=tts_model,
                        voice=target_voice,
                        instructions=target_instructions,
                        speed=speed,
                    )
                    target_name = f"segments/target_{seg.idx:03d}.wav"
                    artifacts[target_name] = target_wav
                    target_audio_parts.append(target_wav)
                    seg.translated_audio_filename = target_name
                    completed += 1
                    progress.progress(completed / total_steps)

                full_source = concat_wav_bytes(source_audio_parts)
                full_target = concat_wav_bytes(target_audio_parts)
                alternating = build_alternating_wav(source_audio_parts, target_audio_parts, source_first=source_first)

                artifacts["full_source.wav"] = full_source
                artifacts["full_target.wav"] = full_target
                artifacts["alternating_bilingual.wav"] = alternating

                manifest = {
                    "pipeline": "text_only_bilingual_recomposer",
                    "source_language": source_language,
                    "target_language": target_language,
                    "translation_model": translation_model,
                    "tts_model": tts_model,
                    "source_voice": source_voice,
                    "target_voice": target_voice,
                    "speed": speed,
                    "source_first": source_first,
                    "segments": [asdict(s) for s in translated_segments],
                    "estimated_cost_usd": {
                        "translation": round(estimate_translation_cost(translation_model, source_text), 6),
                        "tts": round(estimate_tts_cost(tts_model, source_text, " ".join(s.translated_text for s in translated_segments)), 6),
                    },
                }
                manifest["estimated_cost_usd"]["total"] = round(
                    manifest["estimated_cost_usd"]["translation"] + manifest["estimated_cost_usd"]["tts"], 6
                )

                st.session_state["segments"] = translated_segments
                st.session_state["artifacts"] = artifacts
                st.session_state["manifest"] = manifest
                status.update(label="Generation complete", state="complete")
        except Exception as exc:
            st.exception(exc)
            return

    if st.session_state.get("segments"):
        st.subheader("Generated segment map")
        st.dataframe(
            [
                {
                    "#": seg.idx,
                    "Source chars": seg.source_chars,
                    "Target chars": seg.translated_chars,
                    "Source text": seg.source_text,
                    "Translated text": seg.translated_text,
                }
                for seg in st.session_state["segments"]
            ],
            use_container_width=True,
        )

    artifacts = st.session_state.get("artifacts", {})
    manifest = st.session_state.get("manifest", {})
    if artifacts:
        st.subheader("Downloads")
        for key in ["alternating_bilingual.wav", "full_source.wav", "full_target.wav"]:
            if key in artifacts:
                st.download_button(
                    label=f"Download {key}",
                    data=artifacts[key],
                    file_name=key,
                    mime="audio/wav",
                )
                st.audio(artifacts[key], format="audio/wav")

        zip_blob = build_zip(artifacts, manifest)
        st.download_button(
            label="Download full ZIP package",
            data=zip_blob,
            file_name="bilingual_audio_package.zip",
            mime="application/zip",
        )

        st.subheader("Manifest")
        st.json(manifest)


if __name__ == "__main__":
    main()
