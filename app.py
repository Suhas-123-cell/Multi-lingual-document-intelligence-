"""Gradio demo for the multilingual document intelligence pipeline."""
from __future__ import annotations

import io
import json

import gradio as gr
from PIL import Image

from pipeline import (
    DocType,
    answer_question,
    extract,
    score_extraction,
    tesseract_extract,
)


def _to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def run_pipeline(image: Image.Image, doc_type_label: str):
    if image is None:
        return "Upload an image first.", "{}", "{}", "", "", "", "{}"

    doc_type = DocType(doc_type_label)
    img_bytes = _to_bytes(image)

    vlm = extract(img_bytes, doc_type=doc_type)
    tess = tesseract_extract(img_bytes, doc_type=vlm.doc_type)

    summary = (
        f"Document type: **{vlm.doc_type.value}**  \n"
        f"Sarvam pipeline: DD OCR `{vlm.ocr_latency_ms} ms` + LLM `{vlm.chat_latency_ms} ms` "
        f"= **{vlm.latency_ms} ms**  \n"
        f"Tesseract langs: `{tess.langs}` ({tess.latency_ms} ms)"
    )
    if vlm.validation_errors:
        summary += "  \n_validation warnings: " + "; ".join(vlm.validation_errors[:3]) + "_"

    vlm_json = json.dumps(vlm.record, ensure_ascii=False, indent=2)
    tess_json = json.dumps(tess.record, ensure_ascii=False, indent=2)

    record_blob = json.dumps(
        {"record": vlm.record, "ocr_text": vlm.ocr_text}, ensure_ascii=False
    )

    return (
        summary,
        vlm_json,
        tess_json,
        vlm.ocr_text,
        tess.raw_text,
        vlm.doc_type.value,
        record_blob,
    )


def run_query(question: str, record_blob: str):
    if not question:
        return "Type a question."
    if not record_blob:
        return "Run extraction first."
    try:
        blob = json.loads(record_blob)
    except json.JSONDecodeError:
        return "Run extraction first."
    record = blob.get("record", {})
    ocr_text = blob.get("ocr_text", "")
    ans = answer_question(record, question, ocr_text=ocr_text)
    return f"**{ans.text}**  \n_source: {ans.source}_"


def run_compare(image: Image.Image, vlm_json: str, tess_json: str, truth_json: str):
    if not truth_json.strip():
        return "Paste ground-truth JSON to score."
    try:
        truth = json.loads(truth_json)
        vlm = json.loads(vlm_json) if vlm_json else {}
        tess = json.loads(tess_json) if tess_json else {}
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    vlm_score = score_extraction(truth, vlm, "current", "vlm")
    tess_score = score_extraction(truth, tess, "current", "tesseract")

    def fmt(s):
        return (
            f"exact: {s.exact_match_rate:.1%} | "
            f"fuzzy: {s.mean_fuzzy:.1f}/100 | "
            f"CER: {s.mean_cer:.3f}"
        )

    return f"**VLM**  \n{fmt(vlm_score)}  \n\n**Tesseract**  \n{fmt(tess_score)}"


with gr.Blocks(title="Indic Document Intelligence") as demo:
    gr.Markdown(
        "# Multilingual Document Intelligence\n"
        "Upload a scanned Indian government document — Aadhar, PAN, Voter ID, "
        "driving license, passport, ration card, land record, or a birth / "
        "death / marriage / caste / income / domicile certificate. The pipeline "
        "OCRs the page with **Sarvam Document Intelligence**, then structures "
        "the text with **Sarvam chat (sarvam-105b)**, and shows a Tesseract "
        "baseline alongside for comparison."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Scanned document")
            doc_type_dd = gr.Dropdown(
                choices=[d.value for d in DocType],
                value=DocType.AUTO.value,
                label="Document type",
            )
            run_btn = gr.Button("Extract", variant="primary")
        with gr.Column(scale=2):
            summary_md = gr.Markdown()
            with gr.Tabs():
                with gr.Tab("VLM JSON"):
                    vlm_json_out = gr.Code(language="json", label="Extracted record")
                with gr.Tab("Sarvam OCR"):
                    ocr_text_out = gr.Textbox(lines=20, label="Sarvam Document Intelligence (markdown)")
                with gr.Tab("Tesseract JSON"):
                    tess_json_out = gr.Code(language="json", label="Baseline record")
                with gr.Tab("Tesseract raw text"):
                    tess_text_out = gr.Textbox(lines=20, label="Raw OCR")

    doc_type_state = gr.State()
    record_state = gr.State()

    gr.Markdown("## Ask a question")
    with gr.Row():
        question_in = gr.Textbox(
            label="Question",
            placeholder='e.g. "What is the expiry date on this ration card?"',
            scale=4,
        )
        ask_btn = gr.Button("Ask", scale=1)
    answer_md = gr.Markdown()

    gr.Markdown("## Score against ground truth")
    truth_in = gr.Code(language="json", label="Ground-truth JSON (paste)")
    score_btn = gr.Button("Score")
    score_md = gr.Markdown()

    run_btn.click(
        run_pipeline,
        inputs=[image_in, doc_type_dd],
        outputs=[
            summary_md,
            vlm_json_out,
            tess_json_out,
            ocr_text_out,
            tess_text_out,
            doc_type_state,
            record_state,
        ],
    )
    ask_btn.click(
        run_query,
        inputs=[question_in, record_state],
        outputs=[answer_md],
    )
    score_btn.click(
        run_compare,
        inputs=[image_in, vlm_json_out, tess_json_out, truth_in],
        outputs=[score_md],
    )


if __name__ == "__main__":
    demo.launch()
