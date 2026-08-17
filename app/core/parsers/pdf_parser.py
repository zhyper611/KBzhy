from __future__ import annotations

import io
import math
import re
import statistics
from collections import Counter
from pathlib import Path

from KBzhy.app.core.document_models import DocumentElement
from KBzhy.app.core.parser import DocumentParseError
from KBzhy.app.core.parsers.section_path import section_titles, update_section_stack


_LIST_LINE = re.compile(r"^\s*(?:[-+*•‣▪]|\d+[.)])\s+\S")


def parse_pdf(source: str | Path | bytes, *, document_id: str) -> tuple[DocumentElement, ...]:
    import fitz

    try:
        pdf = (
            fitz.open(stream=source, filetype="pdf")
            if isinstance(source, bytes)
            else fitz.open(str(source))
        )
        try:
            return _parse_open_pdf(pdf, document_id=document_id)
        finally:
            pdf.close()
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError("PDF 文档解析失败") from exc


def _parse_open_pdf(pdf, *, document_id: str) -> tuple[DocumentElement, ...]:
    pages = [_extract_page(page, page_index + 1) for page_index, page in enumerate(pdf)]
    repeated_margins = _repeated_margin_keys(pages)
    blocks = [
        block
        for page in pages
        for block in page["blocks"]
        if not (block["is_margin"] and block["normalized"] in repeated_margins)
    ]
    if not blocks:
        raise DocumentParseError("未提取到可索引文字")

    font_sizes = [block["font_size"] for block in blocks if block["font_size"] > 0]
    body_size = statistics.median(font_sizes) if font_sizes else 0.0
    heading_sizes = sorted(
        {
            block["font_size"]
            for block in blocks
            if _is_heading(block, body_size)
        },
        reverse=True,
    )
    section_stack: list[tuple[int, str]] = []
    elements: list[DocumentElement] = []
    for block in blocks:
        is_heading = _is_heading(block, body_size)
        metadata = {
            "source_format": "pdf",
            "page_block_index": block["block_index"],
            "font_size": block["font_size"],
            "font_names": block["font_names"],
        }
        if is_heading:
            level = heading_sizes.index(block["font_size"]) + 1
            update_section_stack(section_stack, level, block["text"])
            metadata["level"] = level
        order = len(elements)
        x0, y0, x1, y1 = block["bbox"]
        elements.append(
            DocumentElement(
                element_id=f"{document_id}:pdf:{order}",
                element_type="heading" if is_heading else "paragraph",
                text=block["text"],
                order=order,
                page=block["page"],
                section_path=section_titles(section_stack),
                bounding_box={"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                metadata=metadata,
            )
        )
    if not elements:
        raise DocumentParseError("未提取到可索引文字")
    return tuple(elements)


def _extract_page(page, page_number: int) -> dict:
    page_dict = page.get_text("dict")
    height = _finite_number(page.rect.height) or 0.0
    blocks = []
    raw_blocks = page_dict.get("blocks", []) if isinstance(page_dict, dict) else []
    for block_index, block in enumerate(raw_blocks if isinstance(raw_blocks, list) else []):
        if not isinstance(block, dict):
            continue
        if block.get("type") != 0:
            continue
        lines: list[dict] = []
        sizes: list[float] = []
        font_names: set[str] = set()
        flags: list[int] = []
        raw_lines = block.get("lines", [])
        for line in raw_lines if isinstance(raw_lines, list) else []:
            if not isinstance(line, dict):
                continue
            line_bbox = _finite_bbox(line.get("bbox"))
            raw_spans = line.get("spans", [])
            if line_bbox is None or not isinstance(raw_spans, list):
                continue
            valid_spans: list[tuple[str, float, str, int]] = []
            for span in raw_spans:
                if not isinstance(span, dict):
                    continue
                text = span.get("text", "")
                size = _finite_number(span.get("size", 0.0))
                try:
                    span_flags = int(span.get("flags", 0))
                except (TypeError, ValueError, OverflowError):
                    continue
                if not isinstance(text, str) or not text.strip() or size is None:
                    continue
                font = span.get("font", "")
                valid_spans.append((text, round(size, 2), font if isinstance(font, str) else "", span_flags))
            line_text = "".join(item[0] for item in valid_spans).strip()
            if not line_text:
                continue
            line_sizes = [item[1] for item in valid_spans]
            lines.append(
                {
                    "text": line_text,
                    "bbox": line_bbox,
                    "font_size": max(line_sizes, default=0.0),
                }
            )
            sizes.extend(line_sizes)
            font_names.update(item[2] for item in valid_spans if item[2])
            flags.extend(item[3] for item in valid_spans)
        text = _join_block_lines(lines)
        if not text:
            continue
        bbox = _finite_bbox(block.get("bbox")) or _union_bboxes(
            [line["bbox"] for line in lines]
        )
        if bbox is None:
            continue
        blocks.append(
            {
                "text": text,
                "normalized": _normalize_margin_text(text),
                "page": page_number,
                "block_index": block_index,
                "bbox": bbox,
                "font_size": max(sizes, default=0.0),
                "font_names": sorted(font_names),
                "bold": any(flag & 16 for flag in flags)
                or any("bold" in name.lower() for name in font_names),
                "is_margin": bbox[1] <= height * 0.1 or bbox[3] >= height * 0.9,
            }
        )
    return {"blocks": blocks}


def _finite_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _finite_bbox(value) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    numbers = tuple(_finite_number(item) for item in value)
    if any(number is None for number in numbers):
        return None
    return numbers  # type: ignore[return-value]


def _union_bboxes(
    bboxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if not bboxes:
        return None
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _join_block_lines(lines: list[dict]) -> str:
    if not lines:
        return ""
    text = lines[0]["text"]
    previous = lines[0]
    for current in lines[1:]:
        separator, trim_hyphen = _line_separator(previous, current)
        if trim_hyphen:
            text = text[:-1]
        text += separator + current["text"]
        previous = current
    return text.strip()


def _line_separator(previous: dict, current: dict) -> tuple[str, bool]:
    previous_text = previous["text"]
    current_text = current["text"]
    previous_bbox = previous["bbox"]
    current_bbox = current["bbox"]
    previous_height = max(previous_bbox[3] - previous_bbox[1], 0.0)
    current_height = max(current_bbox[3] - current_bbox[1], 0.0)
    vertical_gap = current_bbox[1] - previous_bbox[3]
    visibly_separate = vertical_gap > max(previous_height, current_height, 1.0) * 0.75
    font_size_changed = (
        previous["font_size"] > 0
        and current["font_size"] > 0
        and abs(previous["font_size"] - current["font_size"])
        >= max(previous["font_size"], current["font_size"]) * 0.2
    )
    if visibly_separate or font_size_changed or _LIST_LINE.match(previous_text) or _LIST_LINE.match(current_text):
        return "\n", False
    if re.search(r"[A-Za-z]-$", previous_text) and re.match(r"^[A-Za-z]", current_text):
        return "", True
    if _has_cjk_boundary(previous_text, current_text):
        return "", False
    return " ", False


def _has_cjk_boundary(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    return bool(
        re.match(r"[\u3400-\u9fff]", current)
        or re.search(r"[\u3400-\u9fff]$", previous)
    )


def _normalize_margin_text(text: str) -> str:
    normalized = re.sub(r"\d+", "#", text.casefold())
    return re.sub(r"\s+", " ", normalized).strip()


def _repeated_margin_keys(pages: list[dict]) -> set[str]:
    page_count = len(pages)
    if page_count < 2:
        return set()
    occurrences = Counter()
    for page in pages:
        occurrences.update(
            {
                block["normalized"]
                for block in page["blocks"]
                if block["is_margin"] and block["normalized"]
            }
        )
    threshold = max(2, math.ceil(page_count * 0.5))
    return {text for text, count in occurrences.items() if count >= threshold}


def _is_heading(block: dict, body_size: float) -> bool:
    text = block["text"].replace("\n", " ").strip()
    if not text or len(text) > 120 or text.endswith(("。", ".", "！", "!", "？", "?", ":", "：")):
        return False
    larger_than_body = body_size > 0 and block["font_size"] >= body_size * 1.2
    emphasized = block["bold"] and block["font_size"] >= body_size
    return larger_than_body or emphasized
