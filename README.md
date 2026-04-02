# Bilingual Text-to-Audio Composer

A small Streamlit app that takes raw text, translates it with the OpenAI API, segments it into natural chunks, generates two sets of WAV files using OpenAI text-to-speech, and exports:

- `full_source.wav`
- `full_target.wav`
- `alternating_bilingual.wav`
- a ZIP with all segment files plus `manifest.json`

## Features

- text-only workflow
- OpenAI API key entered directly in the UI
- translation model choice
- TTS model choice
- cost estimate panel with model-pair comparison
- source/target voice selection
- WAV concatenation without ffmpeg
- segment preview before generation

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The app keeps the API key in session memory only.
- Cost numbers are estimates, not billing records.
- `gpt-4o-mini-tts` cost is estimated from both text tokens and estimated audio duration.
- `tts-1` and `tts-1-hd` are estimated from character count.
- The speech endpoint currently limits each input to 4096 characters, so the app segments input before TTS.

## Pricing/model references used in the code

Checked on 2026-04-01 from official OpenAI docs:

- TTS endpoint and supported models/formats
- `gpt-4o-mini-tts` pricing
- `tts-1` pricing
- `gpt-4o-mini`, `gpt-4.1-mini`, `gpt-5-mini`, `gpt-5-nano`, and `gpt-5` pricing
- audio token guidance used for `gpt-4o-mini-tts` estimation

## Limitations

- Translation is done segment-by-segment for simplicity.
- Token estimation uses a rough chars-to-tokens heuristic.
- Audio generation retries are not implemented yet.
- Segment alignment is semantic-by-order rather than duration-balanced.
