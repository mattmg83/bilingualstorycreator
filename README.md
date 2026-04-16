# Bilingual story creator

Streamlit app that translates text and generates bilingual audio outputs:

- `full_source.wav` / `full_source.mp3`
- `full_target.wav` / `full_target.mp3`
- `alternating_bilingual.wav` / `alternating_bilingual.mp3`
- ZIP with per-segment files + `manifest.json`

## Recommended stack (simplest)

**Use Streamlit Community Cloud** for hosting.

Why this is simplest:
- no container setup
- no custom web server
- built for Streamlit apps
- easy secret management

Dependency risk: **Low** (Streamlit + OpenAI SDK + pydub; ffmpeg system package).  
Hosting complexity: **Low** (single hosted app).  
Maintenance burden: **Low** (mostly dependency updates).

## Python version

This repo targets **Python 3.13.2** (also pinned in `runtime.txt`).

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## MP3 merge requirements (important)

WAV output is the safe default and works with the base dependencies.

If you choose **MP3 output**, the app uses a safe decode→PCM→re-encode merge flow and needs:

- `pydub` Python package
- `ffmpeg` installed and available on PATH

Example local setup:

```bash
pip install pydub
# macOS (Homebrew)
brew install ffmpeg
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## Transition sound effects by format

The app inserts transition SFX in the **alternating** output when matching files exist in `soundfx/`:

- WAV workflow uses `soundfx/*.wav`
- MP3 workflow uses `soundfx/*.mp3`

## Deploy (recommended): Streamlit Community Cloud

### Required secrets

Add this in Streamlit app settings → **Secrets**:

```toml
OPENAI_API_KEY = "your_openai_api_key"
ELEVENLABS_API_KEY = "your_elevenlabs_api_key" # required only when TTS provider is ElevenLabs
```

> The app supports manual API key entry in the sidebar for local testing, but hosted deploys should use secrets.

### Exact deployment steps

1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud and click **Create app**.
3. Select your repo + branch.
4. Set **Main file path** to `app.py`.
5. In **Advanced settings → Secrets**, paste:

   ```toml
   OPENAI_API_KEY = "your_openai_api_key"
   ELEVENLABS_API_KEY = "your_elevenlabs_api_key" # only needed for ElevenLabs TTS
   ```
6. Deploy.

### Expected startup command

Streamlit Cloud runs the app file directly (`app.py`).
Equivalent local startup command:

```bash
streamlit run app.py
```

## Local dev vs hosted

- **API key source**
  - Local: usually typed into sidebar (or set `OPENAI_API_KEY` / `ELEVENLABS_API_KEY` env vars).
  - Hosted: should come from Streamlit Secrets (`OPENAI_API_KEY`; add `ELEVENLABS_API_KEY` if using ElevenLabs TTS).

- **Networking**
  - Local: uses your machine defaults.
  - Hosted: uses `.streamlit/config.toml` (`headless`, `0.0.0.0`, relaxed CORS/XSRF for managed proxy setups).

- **Port**
  - Local: default `8501` unless overridden.
  - Hosted: platform controls port routing; Streamlit Cloud handles this automatically.

## Notes

- Cost numbers are estimates, not billing records.
- OpenAI TTS has built-in cost estimates in the app.
- ElevenLabs TTS intentionally shows **not estimated** in the cost panel until pricing constants are added.
- Speech input is segmented before TTS to stay within endpoint limits.
- Translation is segment-by-segment for simplicity.
- Audio reuse safety: cache/artifact fingerprints include the TTS provider, so switching OpenAI ↔ ElevenLabs forces fresh audio generation.

## Manual segmentation with `##`

If your source text includes `##`, the app switches to manual segmentation mode:

- each `##` acts as a segment cue point
- automatic sentence/character-based segmentation is skipped
- `##` markers are removed from segment text
- empty chunks are ignored

Example:

```text
Hello and welcome. ## This is section two. ## Final section.
```
