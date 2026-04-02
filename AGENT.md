# AGENT.md

## 1) Purpose + current architecture (today)

This repo ships a **single-file Streamlit app** (`app.py`) that does:
- text segmentation
- translation with OpenAI Responses API
- TTS generation with OpenAI audio models
- WAV assembly + ZIP export (including `manifest.json`)

Current shape is intentionally a **monolith** for speed and low ops.

## 2) Near-term extension points

When growth is needed, extract focused modules without changing behavior first:
- `segmentation.py` → whitespace cleanup, sentence splitting, segment sizing
- `audio.py` → TTS calls, WAV merge helpers, alternating track builder
- `costs.py` → token/character estimates + pricing table calculations

Rule of thumb: move pure helpers first, keep Streamlit UI wiring in `app.py` until the second pass.

## 3) Coding conventions (keep it lightweight)

- Prefer small, explicit functions over abstractions.
- Keep dependencies minimal (default: stdlib + Streamlit + OpenAI SDK).
- Add comments only when intent is not obvious (short, practical comments).
- Avoid clever patterns; optimize for readability and quick edits.
- Keep file structure shallow and easy to scan.

## 4) Release checklist (Streamlit hosted app)

Before deploy:
1. Update `requirements.txt` if package versions changed.
2. Confirm README deploy steps still match reality.
3. Run a smoke test locally:
   - app starts
   - translation works
   - TTS works
   - ZIP download includes expected WAVs + `manifest.json`
4. Deploy to Streamlit Community Cloud and re-run a quick live smoke test.

## 5) Safe-change guidance (models + pricing)

For OpenAI model/pricing updates:
- Update model IDs and pricing constants together in one PR.
- Add the **date checked** in comments near pricing constants.
- Keep one conservative default model choice.
- If uncertain, prefer backward-compatible changes (add new model options before removing old ones).
- Validate cost estimator output with a short known sample before release.

Goal: keep changes boring, reversible, and easy for the next contributor to verify.
