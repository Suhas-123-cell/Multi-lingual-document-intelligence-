from .schemas import DocType, SCHEMAS, schema_for
from .extractor import extract
from .tesseract_ocr import tesseract_extract
from .query import answer_question
from .compare import score_extraction

__all__ = [
    "DocType",
    "SCHEMAS",
    "schema_for",
    "extract",
    "tesseract_extract",
    "answer_question",
    "score_extraction",
]
