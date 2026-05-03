"""Natural-language Q&A over an extracted record.

Two-step strategy:
  1. Try answering from the structured JSON alone — fastest, no API call.
  2. Fall back to a Sarvam chat call against the cached OCR markdown that the
     extraction step already produced. This avoids re-running Sarvam Document
     Intelligence on every question.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from .sarvam_vision import call_vlm_text


@dataclass
class Answer:
    text: str
    source: str  # "record" or "ocr" or "vlm"


_RECORD_ANSWER_PROMPT = """You will be given a JSON record extracted from a scanned
government document, and a question about it.

If the answer is present in the JSON, respond with the exact value and nothing else.
If the JSON does not contain the answer, respond with the single token: UNKNOWN.

JSON:
{record}

Question: {question}
"""


_OCR_ANSWER_PROMPT = """You are answering a question about an Indian government
document. The OCR'd text of the document is below. Answer concisely. If the
answer is genuinely not visible, say "not present in the document".

Question: {question}

----- BEGIN OCR -----
{ocr}
----- END OCR -----
"""


_DIRECT_KEYS: dict[str, list[str]] = {
    "expiry": ["expiry_date", "date_of_expiry", "valid_until"],
    "issue": ["issue_date", "date_of_issue"],
    "birth": ["date_of_birth"],
    "death": ["date_of_death"],
    "marriage": ["date_of_marriage"],
    "card number": ["card_number"],
    "registration": ["registration_number"],
    "owner": ["owner_name"],
    "head": ["head_of_family"],
    "father": ["fathers_name", "fathers_or_husbands_name", "fathers_or_spouses_name"],
    "mother": ["mothers_name"],
    "address": ["address"],
    "village": ["village"],
    "district": ["district"],
    "state": ["state"],
    "area": ["area"],
    "aadhar": ["aadhar_number"],
    "aadhaar": ["aadhar_number"],
    "pan": ["pan_number"],
    "passport": ["passport_number"],
    "license": ["license_number"],
    "epic": ["epic_number"],
    "voter": ["epic_number"],
    "name": ["name", "child_name", "deceased_name", "owner_name", "head_of_family"],
}


def _direct_lookup(record: dict, question: str) -> Optional[str]:
    q = question.lower()
    for needle, fields in _DIRECT_KEYS.items():
        if needle in q:
            for f in fields:
                v = record.get(f)
                if v:
                    return str(v)
    return None


def _answer_from_record(record: dict, question: str) -> Optional[str]:
    direct = _direct_lookup(record, question)
    if direct:
        return direct

    try:
        prompt = _RECORD_ANSWER_PROMPT.format(
            record=json.dumps(record, ensure_ascii=False, indent=2),
            question=question,
        )
        result = call_vlm_text(prompt)
    except Exception:
        return None
    ans = result.raw_text.strip()
    if ans and ans.upper() != "UNKNOWN":
        return ans
    return None


def answer_question(
    record: dict,
    question: str,
    image_bytes: Optional[bytes] = None,
    ocr_text: Optional[str] = None,
    prefer: Optional[str] = None,
) -> Answer:
    """Answer a question using the structured record first, falling back to OCR text.

    `image_bytes` and `prefer` are accepted for backward compatibility and ignored —
    the OCR pass already happened during extraction; we reuse `ocr_text`.
    """
    del image_bytes, prefer

    fast = _direct_lookup(record, question)
    if fast:
        return Answer(text=fast, source="record")

    record_answer = _answer_from_record(record, question)
    if record_answer:
        return Answer(text=record_answer, source="record")

    if not ocr_text:
        return Answer(text="not present in the extracted record", source="record")

    try:
        result = call_vlm_text(_OCR_ANSWER_PROMPT.format(question=question, ocr=ocr_text))
    except Exception as exc:  # noqa: BLE001
        return Answer(text=f"(query failed: {exc})", source="ocr")
    return Answer(text=result.raw_text.strip(), source="ocr")
