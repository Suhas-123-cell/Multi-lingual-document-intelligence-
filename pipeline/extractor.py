"""Schema-driven structured extraction over the Sarvam two-stage pipeline.

Each `extract()` call performs a single Sarvam Document Intelligence OCR
pass, then reuses the resulting markdown for (optional) document-type
classification and for the schema-driven structuring chat call.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError

from .sarvam_vision import (
    VLMResult,
    _sarvam_dd_ocr,
    call_vlm_text,
    _coerce_json,
)
from .schemas import DocType, SCHEMAS, schema_for


_DOC_TYPE_LIST = (
    "aadhar, pan_card, voter_id, driving_license, passport, ration_card, "
    "land_record, birth_certificate, death_certificate, marriage_certificate, "
    "caste_certificate, income_certificate, domicile_certificate, unknown"
)


def _classify_prompt(ocr_text: str) -> str:
    return (
        "Identify the Indian government document type given the OCR'd text below. "
        f"Reply with ONLY one of: {_DOC_TYPE_LIST}.\n\n"
        "----- BEGIN OCR -----\n"
        f"{ocr_text}\n"
        "----- END OCR -----\n"
    )


def _extract_prompt(doc_type: DocType, ocr_text: str) -> str:
    schema = schema_for(doc_type)
    schema_json = json.dumps(schema.model_json_schema(), indent=2, ensure_ascii=False)
    return f"""You are extracting fields from a transcribed Indian government document.

Document type: {doc_type.value}

The OCR text below may include English, Hindi, Bangla, Tamil, Telugu, Kannada,
Malayalam, Gujarati, Punjabi, Oriya, Manipuri (Meitei Mayek), or other Indian
scripts. Return field VALUES TRANSLITERATED OR TRANSLATED INTO ENGLISH where it
makes sense (names stay in original transliteration; numbers and dates in ISO
format).

Return ONLY a JSON object matching this Pydantic schema. Use null for missing
fields. Do not invent values.

Schema:
{schema_json}

----- BEGIN OCR -----
{ocr_text}
----- END OCR -----
"""


@dataclass
class ExtractionResult:
    doc_type: DocType
    record: dict
    raw: dict
    raw_text: str
    ocr_text: str
    provider: str
    latency_ms: int
    ocr_latency_ms: int
    chat_latency_ms: int
    validation_errors: list[str]


def _classify_from_text(ocr_text: str) -> tuple[DocType, int]:
    result: VLMResult = call_vlm_text(_classify_prompt(ocr_text))
    text = result.raw_text.strip().lower()
    candidates = (
        DocType.AADHAR,
        DocType.PAN_CARD,
        DocType.VOTER_ID,
        DocType.DRIVING_LICENSE,
        DocType.PASSPORT,
        DocType.RATION_CARD,
        DocType.LAND_RECORD,
        DocType.BIRTH_CERTIFICATE,
        DocType.DEATH_CERTIFICATE,
        DocType.MARRIAGE_CERTIFICATE,
        DocType.CASTE_CERTIFICATE,
        DocType.INCOME_CERTIFICATE,
        DocType.DOMICILE_CERTIFICATE,
    )
    for dt in candidates:
        if dt.value in text:
            return dt, result.chat_latency_ms
    return DocType.AADHAR, result.chat_latency_ms


def extract(
    image_bytes: bytes,
    doc_type: DocType = DocType.AUTO,
    prefer: Optional[str] = None,
) -> ExtractionResult:
    """Run Sarvam DD OCR once, then classify (if needed) and structure via Sarvam chat.

    `prefer` is accepted for backward compatibility and ignored.
    """
    del prefer

    ocr_text, ocr_latency = _sarvam_dd_ocr(image_bytes)

    chat_latency_total = 0
    if doc_type == DocType.AUTO:
        doc_type, classify_latency = _classify_from_text(ocr_text)
        chat_latency_total += classify_latency

    chat_result: VLMResult = call_vlm_text(_extract_prompt(doc_type, ocr_text))
    chat_latency_total += chat_result.chat_latency_ms

    raw = chat_result.json_obj or {}
    schema = schema_for(doc_type)

    errors: list[str] = []
    try:
        record = schema.model_validate(raw).model_dump()
    except ValidationError as exc:
        errors = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        cleaned: dict = {}
        for field_name in schema.model_fields:
            if field_name in raw:
                try:
                    schema.model_validate({field_name: raw[field_name]})
                    cleaned[field_name] = raw[field_name]
                except ValidationError:
                    continue
        record = cleaned

    return ExtractionResult(
        doc_type=doc_type,
        record=record,
        raw=raw,
        raw_text=chat_result.raw_text,
        ocr_text=ocr_text,
        provider="sarvam",
        latency_ms=ocr_latency + chat_latency_total,
        ocr_latency_ms=ocr_latency,
        chat_latency_ms=chat_latency_total,
        validation_errors=errors,
    )
