# Multilingual Document Intelligence for Government Records

Pipeline for extracting structured data from scanned Indian government documents
(ration cards, land records, birth certificates) written in regional scripts —
Bangla, Tamil, Hindi, Meitei Mayek, etc.

## What's inside

- **Sarvam Vision** as the VLM extractor (uses the OpenAI-compatible Sarvam endpoint).
- **Tesseract** baseline with Indic language packs.
- **Schema-based JSON extraction** — pick a doc type, get typed fields.
- **NL query layer** — ask "what is the expiry date?" against the extracted record.
- **Accuracy harness** — field precision/recall and CER vs ground-truth JSON.
- **Gradio demo** — upload, see both pipelines side-by-side, query the result.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your SARVAM_API_KEY

# Tesseract + Indic lang packs (macOS)
brew install tesseract tesseract-lang
```

## Run

```bash
python app.py
```

## Eval

Drop ground-truth JSONs into `data/eval/` next to the source images
(`foo.jpg` ↔ `foo.json`), then:

```bash
python -m pipeline.compare data/eval
```

## Project layout

```
app.py                 Gradio entrypoint
pipeline/
  schemas.py           Pydantic schemas per document type
  sarvam_vision.py     Sarvam Vision API client
  tesseract_ocr.py     Tesseract baseline + naive field extraction
  extractor.py         Provider-agnostic structured extraction
  query.py             Natural-language Q&A over extracted records
  compare.py           Field-level + character-level accuracy
data/
  eval/                Image + ground-truth JSON pairs
```

## Open dataset goal

`data/eval/` is the seed for an open Meitei Mayek + multi-script benchmark.
Add real samples here, commit ground-truth JSONs, and the harness produces
reproducible numbers.
# Multi-lingual-document-intelligence-
