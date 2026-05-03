"""Field-level + character-level accuracy vs ground-truth JSON.

Usage:
    python -m pipeline.compare data/eval

Each `<name>.{jpg,png,pdf}` in the directory must have a sibling `<name>.json`
ground-truth file that matches one of the schemas in `pipeline.schemas`.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz
from jiwer import cer

from .extractor import extract
from .schemas import DocType, SCHEMAS
from .tesseract_ocr import tesseract_extract


def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    out.update(_flatten(item, f"{key}[{i}]"))
                else:
                    out[f"{key}[{i}]"] = "" if item is None else str(item)
        else:
            out[key] = "" if v is None else str(v)
    return out


@dataclass
class FieldScore:
    field: str
    truth: str
    pred: str
    exact: bool
    fuzzy: float  # 0..100
    cer: float


@dataclass
class DocScore:
    name: str
    pipeline: str
    fields: list[FieldScore] = field(default_factory=list)

    @property
    def exact_match_rate(self) -> float:
        if not self.fields:
            return 0.0
        return sum(1 for f in self.fields if f.exact) / len(self.fields)

    @property
    def mean_fuzzy(self) -> float:
        if not self.fields:
            return 0.0
        return sum(f.fuzzy for f in self.fields) / len(self.fields)

    @property
    def mean_cer(self) -> float:
        if not self.fields:
            return 0.0
        return sum(f.cer for f in self.fields) / len(self.fields)


def score_extraction(truth: dict, pred: dict, name: str, pipeline: str) -> DocScore:
    flat_truth = _flatten(truth)
    flat_pred = _flatten(pred)
    score = DocScore(name=name, pipeline=pipeline)
    for field_name, t in flat_truth.items():
        p = flat_pred.get(field_name, "")
        exact = t.strip() == p.strip()
        fuzzy = float(fuzz.ratio(t, p)) if t or p else 100.0
        c = cer(t, p) if t or p else 0.0
        score.fields.append(FieldScore(field_name, t, p, exact, fuzzy, c))
    return score


def _resolve_doc_type(truth: dict) -> Optional[DocType]:
    """Pick the schema whose required-ish keys match best."""
    best, best_overlap = None, -1
    truth_keys = set(truth.keys())
    for dt, schema in SCHEMAS.items():
        overlap = len(truth_keys & set(schema.model_fields.keys()))
        if overlap > best_overlap:
            best_overlap = overlap
            best = dt
    return best


def _print_score(score: DocScore) -> None:
    print(f"\n=== {score.name} | {score.pipeline} ===")
    print(f"  exact match: {score.exact_match_rate:.1%}")
    print(f"  mean fuzzy:  {score.mean_fuzzy:.1f} / 100")
    print(f"  mean CER:    {score.mean_cer:.3f}")
    for f in score.fields:
        marker = "OK" if f.exact else "  "
        print(f"  [{marker}] {f.field}: truth={f.truth!r} pred={f.pred!r} (fuzzy={f.fuzzy:.0f})")


def main(eval_dir: str) -> None:
    root = Path(eval_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in root.iterdir() if p.suffix.lower() in image_exts]
    if not images:
        print(f"No images found in {root.resolve()}")
        return

    sarvam_scores: list[DocScore] = []
    tess_scores: list[DocScore] = []

    for img_path in sorted(images):
        truth_path = img_path.with_suffix(".json")
        if not truth_path.exists():
            print(f"skip {img_path.name}: no ground-truth JSON")
            continue
        truth = json.loads(truth_path.read_text())
        doc_type = _resolve_doc_type(truth) or DocType.RATION_CARD
        image_bytes = img_path.read_bytes()

        vlm = extract(image_bytes, doc_type=doc_type)
        sarvam_scores.append(score_extraction(truth, vlm.record, img_path.name, vlm.provider))

        tess = tesseract_extract(image_bytes, doc_type=doc_type)
        tess_scores.append(score_extraction(truth, tess.record, img_path.name, "tesseract"))

    for s in sarvam_scores + tess_scores:
        _print_score(s)

    def agg(scores: list[DocScore], label: str) -> None:
        if not scores:
            return
        em = sum(s.exact_match_rate for s in scores) / len(scores)
        fz = sum(s.mean_fuzzy for s in scores) / len(scores)
        ce = sum(s.mean_cer for s in scores) / len(scores)
        print(f"\n{label:>12}  exact={em:.1%}  fuzzy={fz:.1f}  CER={ce:.3f}  n={len(scores)}")

    print("\n" + "=" * 50)
    print("AGGREGATE")
    print("=" * 50)
    agg(sarvam_scores, "vlm")
    agg(tess_scores, "tesseract")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/eval")
