"""Sarvam two-stage extraction client.

Stage 1 — Sarvam Document Intelligence (`/doc-digitization/job/v1/*`):
  Async job pipeline that OCRs scanned images / PDFs into layout-aware
  Markdown. Purpose-trained on 23 Indian languages including the long-tail
  scripts (Manipuri, Sanskrit, Santali) where Tesseract under-performs.

Stage 2 — Sarvam chat completions (`/v1/chat/completions`):
  Text-only LLM (default `sarvam-105b`) that consumes the OCR markdown
  plus the schema prompt and emits structured JSON. The chat endpoint is
  OpenAI-compatible, so we use the `openai` SDK as transport.

`call_vlm(image_bytes, prompt)` orchestrates both stages and returns a
single `VLMResult`. The intermediate OCR markdown is also exposed on
`VLMResult.ocr_text` so downstream Q&A can reuse it without re-OCR'ing.
"""
from __future__ import annotations

import io
import json
import os
import time
import zipfile
from dataclasses import dataclass, field
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class VLMResult:
    provider: str
    raw_text: str
    json_obj: dict
    latency_ms: int
    ocr_text: str = ""
    ocr_latency_ms: int = 0
    chat_latency_ms: int = 0


def _coerce_json(text: str) -> dict:
    """Pull the first JSON object out of a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _dd_base() -> str:
    """Resolve the doc-digitization base URL.

    `SARVAM_BASE_URL` historically pointed at `.../v1` for the chat endpoint.
    DD lives at `.../doc-digitization/job/v1`, so we strip the trailing `/v1`
    if present and re-attach the DD path.
    """
    base = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai/v1").rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return f"{base}/doc-digitization/job/v1"


def _sarvam_dd_ocr(image_bytes: bytes) -> tuple[str, int]:
    """Run Sarvam Document Intelligence on a single image.

    Returns (markdown_text, latency_ms). DD requires uploads to be PDF or a
    flat ZIP of JPEG/PNG images, so we wrap the raw bytes in a ZIP first.
    """
    key = os.getenv("SARVAM_API_KEY")
    if not key:
        raise RuntimeError("SARVAM_API_KEY is not set")
    base = _dd_base()
    headers = {"api-subscription-key": key}
    json_headers = {**headers, "Content-Type": "application/json"}

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("page1.jpg", image_bytes)
    zip_bytes = zip_buf.getvalue()

    timeout_s = float(os.getenv("SARVAM_DD_TIMEOUT_S", "180"))
    poll_max_s = float(os.getenv("SARVAM_DD_POLL_MAX_S", "5"))

    t0 = time.time()

    job_params: dict = {"output_format": "md"}
    lang = os.getenv("SARVAM_DD_LANGUAGE")
    if lang:
        job_params["language"] = lang
    r = requests.post(base, headers=json_headers, json={"job_parameters": job_params}, timeout=30)
    r.raise_for_status()
    job_id = r.json()["job_id"]

    r = requests.post(
        f"{base}/upload-files",
        headers=json_headers,
        json={"job_id": job_id, "files": ["doc.zip"]},
        timeout=30,
    )
    r.raise_for_status()
    upload_url = r.json()["upload_urls"]["doc.zip"]["file_url"]

    r = requests.put(
        upload_url,
        data=zip_bytes,
        headers={"Content-Type": "application/zip", "x-ms-blob-type": "BlockBlob"},
        timeout=60,
    )
    r.raise_for_status()

    r = requests.post(f"{base}/{job_id}/start", headers=headers, timeout=30)
    r.raise_for_status()

    deadline = time.time() + timeout_s
    backoff = 1.0
    state = ""
    last_status: dict = {}
    while time.time() < deadline:
        r = requests.get(f"{base}/{job_id}/status", headers=headers, timeout=30)
        r.raise_for_status()
        last_status = r.json()
        state = last_status["job_state"]
        if state in ("Completed", "PartiallyCompleted"):
            break
        if state == "Failed":
            raise RuntimeError(
                f"Sarvam DD job failed: {last_status.get('error_message', '')}"
            )
        time.sleep(backoff)
        backoff = min(backoff * 1.5, poll_max_s)
    else:
        raise RuntimeError(f"Sarvam DD job timed out after {timeout_s}s (last state: {state})")

    r = requests.post(f"{base}/{job_id}/download-files", headers=headers, timeout=30)
    r.raise_for_status()
    download_url = r.json()["download_urls"]["document.zip"]["file_url"]

    zr = requests.get(download_url, timeout=120)
    zr.raise_for_status()

    md_chunks: list[str] = []
    with zipfile.ZipFile(io.BytesIO(zr.content)) as zf:
        for name in sorted(zf.namelist()):
            if name.lower().endswith(".md"):
                md_chunks.append(zf.read(name).decode("utf-8", errors="replace"))
    md_text = "\n\n".join(c.strip() for c in md_chunks if c.strip())

    latency = int((time.time() - t0) * 1000)
    return md_text, latency


def _sarvam_chat_text(prompt: str) -> tuple[str, int]:
    """Send a text-only prompt to Sarvam's chat endpoint. Returns (raw_text, latency_ms)."""
    key = os.getenv("SARVAM_API_KEY")
    if not key:
        raise RuntimeError("SARVAM_API_KEY is not set")
    from openai import OpenAI

    base_url = os.getenv("SARVAM_CHAT_BASE_URL") or os.getenv(
        "SARVAM_BASE_URL", "https://api.sarvam.ai/v1"
    )
    model = os.getenv("SARVAM_CHAT_MODEL", "sarvam-105b")
    client = OpenAI(api_key=key, base_url=base_url)

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = int((time.time() - t0) * 1000)
    text = resp.choices[0].message.content or ""
    return text, latency


_OCR_INJECTION = (
    "\n\nThe text below was OCR'd from the document by Sarvam Document Intelligence. "
    "Use it as the ground truth for field values:\n\n"
    "----- BEGIN OCR -----\n{ocr}\n----- END OCR -----\n"
)


def call_vlm(image_bytes: bytes, prompt: str, prefer: Optional[str] = None) -> VLMResult:
    """Two-stage Sarvam pipeline: DD OCR, then chat structuring.

    `prefer` is accepted for backward compatibility and ignored — Sarvam is
    the only supported provider.
    """
    del prefer
    if not os.getenv("SARVAM_API_KEY"):
        raise RuntimeError("No VLM provider configured. Set SARVAM_API_KEY in your .env.")

    try:
        ocr_text, ocr_latency = _sarvam_dd_ocr(image_bytes)
    except (requests.RequestException, RuntimeError) as exc:
        raise RuntimeError(f"Sarvam DD OCR failed: {type(exc).__name__}: {exc}") from exc

    full_prompt = prompt + _OCR_INJECTION.format(ocr=ocr_text or "(no text detected)")
    try:
        chat_text, chat_latency = _sarvam_chat_text(full_prompt)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Sarvam chat structuring failed: {type(exc).__name__}: {exc}"
        ) from exc

    return VLMResult(
        provider="sarvam",
        raw_text=chat_text,
        json_obj=_coerce_json(chat_text),
        latency_ms=ocr_latency + chat_latency,
        ocr_text=ocr_text,
        ocr_latency_ms=ocr_latency,
        chat_latency_ms=chat_latency,
    )


def call_vlm_text(prompt: str) -> VLMResult:
    """Text-only Sarvam chat call — no OCR stage. For Q&A over already-OCR'd text."""
    if not os.getenv("SARVAM_API_KEY"):
        raise RuntimeError("No VLM provider configured. Set SARVAM_API_KEY in your .env.")
    try:
        chat_text, chat_latency = _sarvam_chat_text(prompt)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Sarvam chat call failed: {type(exc).__name__}: {exc}") from exc
    return VLMResult(
        provider="sarvam",
        raw_text=chat_text,
        json_obj=_coerce_json(chat_text),
        latency_ms=chat_latency,
        chat_latency_ms=chat_latency,
    )
