"""Tesseract baseline.

Tesseract is the standard OSS OCR. It supports ~100 languages, including most
Indic scripts (ben, tam, hin, tel, kan, mal, guj, pan, ori, asm). It does NOT
ship a Meitei Mayek model — extracting that script will produce mostly garbage,
which is precisely the point of the comparison.
"""
from __future__ import annotations

import io
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from PIL import Image
import pytesseract
from dotenv import load_dotenv

from .schemas import DocType, schema_for

load_dotenv()


@dataclass
class TesseractResult:
    raw_text: str
    record: dict
    latency_ms: int
    langs: str


_DATE_RE = re.compile(r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
_NUM_RE = re.compile(r"\b[A-Z0-9]{6,}\b")


_FIELD_HINTS: dict[DocType, dict[str, list[str]]] = {
    DocType.AADHAR: {
        "aadhar_number": ["aadhaar no", "aadhar no", "uid"],
        "name": ["name"],
        "date_of_birth": ["date of birth", "dob", "year of birth", "yob"],
        "gender": ["gender", "male", "female"],
        "address": ["address"],
        "fathers_name": ["s/o", "d/o", "w/o", "father"],
        "vid": ["vid"],
    },
    DocType.PAN_CARD: {
        "pan_number": ["permanent account number", "pan"],
        "name": ["name"],
        "fathers_name": ["father", "father's name"],
        "date_of_birth": ["date of birth", "dob"],
    },
    DocType.VOTER_ID: {
        "epic_number": ["epic no", "elector", "identity card no"],
        "name": ["name"],
        "fathers_or_husbands_name": ["father", "husband", "s/o", "w/o"],
        "date_of_birth": ["date of birth", "dob"],
        "age": ["age"],
        "gender": ["sex", "gender"],
        "address": ["address"],
        "assembly_constituency": ["assembly constituency", "constituency"],
    },
    DocType.DRIVING_LICENSE: {
        "license_number": ["dl no", "licence no", "license no"],
        "name": ["name"],
        "fathers_name": ["father", "s/o"],
        "date_of_birth": ["date of birth", "dob"],
        "address": ["address"],
        "blood_group": ["blood group", "bg"],
        "issue_date": ["date of issue", "doi"],
        "expiry_date": ["valid till", "valid upto", "doe"],
        "issuing_authority": ["issuing authority", "rto"],
    },
    DocType.PASSPORT: {
        "passport_number": ["passport no"],
        "surname": ["surname"],
        "given_names": ["given name", "given names"],
        "nationality": ["nationality"],
        "date_of_birth": ["date of birth", "dob"],
        "place_of_birth": ["place of birth"],
        "gender": ["sex"],
        "date_of_issue": ["date of issue"],
        "date_of_expiry": ["date of expiry"],
        "place_of_issue": ["place of issue"],
    },
    DocType.RATION_CARD: {
        "card_number": ["card no", "ration card", "rc no"],
        "head_of_family": ["head of family", "name of head", "hof"],
        "address": ["address", "residence"],
        "district": ["district"],
        "state": ["state"],
        "issue_date": ["issue date", "date of issue"],
        "expiry_date": ["expiry", "valid till", "valid upto"],
    },
    DocType.LAND_RECORD: {
        "survey_number": ["survey no", "survey number"],
        "khasra_number": ["khasra"],
        "owner_name": ["owner", "name of owner"],
        "fathers_name": ["father", "s/o", "d/o"],
        "village": ["village"],
        "tehsil": ["tehsil", "taluka"],
        "district": ["district"],
        "area": ["area", "extent"],
    },
    DocType.BIRTH_CERTIFICATE: {
        "registration_number": ["registration no", "reg no"],
        "child_name": ["name of child", "child"],
        "date_of_birth": ["date of birth", "dob"],
        "place_of_birth": ["place of birth"],
        "fathers_name": ["father"],
        "mothers_name": ["mother"],
        "issue_date": ["date of issue", "issued on"],
    },
    DocType.DEATH_CERTIFICATE: {
        "registration_number": ["registration no", "reg no"],
        "deceased_name": ["name of deceased", "deceased"],
        "date_of_death": ["date of death", "dod"],
        "place_of_death": ["place of death"],
        "gender": ["sex", "gender"],
        "age_at_death": ["age at death", "age"],
        "fathers_or_spouses_name": ["father", "spouse", "husband", "wife"],
        "address": ["address"],
        "cause_of_death": ["cause of death"],
        "issue_date": ["date of issue", "issued on"],
    },
    DocType.MARRIAGE_CERTIFICATE: {
        "registration_number": ["registration no", "reg no"],
        "husband_name": ["husband", "groom", "name of husband"],
        "wife_name": ["wife", "bride", "name of wife"],
        "date_of_marriage": ["date of marriage", "solemnized on"],
        "place_of_marriage": ["place of marriage"],
        "issue_date": ["date of issue"],
    },
    DocType.CASTE_CERTIFICATE: {
        "certificate_number": ["certificate no", "serial no"],
        "name": ["name"],
        "fathers_name": ["father", "s/o", "d/o"],
        "caste": ["caste"],
        "sub_caste": ["sub caste", "sub-caste"],
        "category": ["category"],
        "address": ["address"],
        "state": ["state"],
        "issue_date": ["date of issue"],
    },
    DocType.INCOME_CERTIFICATE: {
        "certificate_number": ["certificate no", "serial no"],
        "name": ["name"],
        "fathers_or_husbands_name": ["father", "husband", "s/o", "w/o"],
        "annual_income": ["annual income", "income"],
        "income_source": ["source of income", "source"],
        "address": ["address"],
        "issue_date": ["date of issue"],
        "valid_until": ["valid till", "valid upto"],
    },
    DocType.DOMICILE_CERTIFICATE: {
        "certificate_number": ["certificate no", "serial no"],
        "name": ["name"],
        "fathers_name": ["father", "s/o"],
        "date_of_birth": ["date of birth", "dob"],
        "state": ["state"],
        "district": ["district"],
        "address": ["address"],
        "issue_date": ["date of issue"],
    },
}


def _value_after(line: str, hint: str) -> Optional[str]:
    idx = line.lower().find(hint)
    if idx == -1:
        return None
    rest = line[idx + len(hint) :].lstrip(" :|-—.\t")
    return rest.strip() or None


def _naive_extract(text: str, doc_type: DocType) -> dict:
    """Heuristic field extraction over Tesseract output — deliberately weak."""
    schema = schema_for(doc_type)
    out: dict = {k: None for k in schema.model_fields}
    if "members" in out:
        out["members"] = []

    for line in text.splitlines():
        for field, hints in _FIELD_HINTS.get(doc_type, {}).items():
            if out.get(field):
                continue
            for hint in hints:
                val = _value_after(line, hint)
                if val:
                    out[field] = val
                    break

    # Backfill date-shaped and number-shaped fields if still empty.
    for field in ("issue_date", "expiry_date", "date_of_birth", "mutation_date"):
        if field in out and not out[field]:
            m = _DATE_RE.search(text)
            if m:
                out[field] = m.group(0)
                break
    for field in ("card_number", "registration_number", "survey_number"):
        if field in out and not out[field]:
            m = _NUM_RE.search(text)
            if m:
                out[field] = m.group(0)
                break

    return out


def tesseract_extract(image_bytes: bytes, doc_type: DocType) -> TesseractResult:
    if doc_type == DocType.AUTO:
        doc_type = DocType.RATION_CARD  # baseline can't classify

    langs = os.getenv("TESSERACT_LANGS", "eng")
    img = Image.open(io.BytesIO(image_bytes))
    t0 = time.time()
    try:
        text = pytesseract.image_to_string(img, lang=langs)
    except pytesseract.TesseractError:
        # Lang pack missing — fall back to English only.
        text = pytesseract.image_to_string(img, lang="eng")
        langs = "eng"
    latency = int((time.time() - t0) * 1000)
    return TesseractResult(
        raw_text=text,
        record=_naive_extract(text, doc_type),
        latency_ms=latency,
        langs=langs,
    )
