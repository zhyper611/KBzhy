from __future__ import annotations

import io
import re
from pathlib import Path

from KBzhy.app.core.document_models import DocumentElement
from KBzhy.app.core.parser import DocumentParseError
from KBzhy.app.core.parsers.section_path import section_titles, update_section_stack


_HEADING_STYLE = re.compile(r"^Heading\s+([1-9]\d*)$", re.IGNORECASE)


def parse_word(source: str | Path | bytes, *, document_id: str) -> tuple[DocumentElement, ...]:
    from docx import Document as WordDocument
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = WordDocument(io.BytesIO(source) if isinstance(source, bytes) else str(source))
        return _parse_word_document(
            document,
            document_id=document_id,
            table_class=Table,
            paragraph_class=Paragraph,
        )
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError("Word 文档解析失败") from exc


def _parse_word_document(
    document,
    *,
    document_id: str,
    table_class,
    paragraph_class,
) -> tuple[DocumentElement, ...]:
    elements: list[DocumentElement] = []
    section_stack: list[tuple[int, str]] = []

    def append_element(element_type: str, text: str, metadata: dict | None = None) -> None:
        normalized = text.strip()
        if not normalized:
            return
        order = len(elements)
        elements.append(
            DocumentElement(
                element_id=f"{document_id}:word:{order}",
                element_type=element_type,
                text=normalized,
                order=order,
                section_path=section_titles(section_stack),
                metadata={"source_format": "word", **(metadata or {})},
            )
        )

    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = paragraph_class(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style_name = paragraph.style.name if paragraph.style is not None else ""
            heading_match = _HEADING_STYLE.match(style_name)
            if heading_match:
                level = int(heading_match.group(1))
                update_section_stack(section_stack, level, text)
                append_element("heading", text, {"level": level, "style": style_name})
            elif _is_list_paragraph(paragraph, style_name):
                append_element("list", text, {"style": style_name})
            else:
                append_element("paragraph", text, {"style": style_name})
        elif child.tag.endswith("}tbl"):
            table = table_class(child, document)
            rows = _normalized_table_rows(table)
            if not rows or not any(any(cell for cell in row) for row in rows):
                continue
            header = rows[0]
            data_rows = rows[1:] or [header]
            for row_index, row in enumerate(data_rows, start=1 if len(rows) > 1 else 0):
                append_element(
                    "table",
                    _markdown_table(header, row if len(rows) > 1 else None),
                    {"row_index": row_index, "header": header},
                )

    if not elements:
        raise DocumentParseError("未提取到可索引文字")
    return tuple(elements)


def _is_list_paragraph(paragraph, style_name: str) -> bool:
    properties = paragraph._p.pPr
    return style_name.lower().startswith("list") or (
        properties is not None and properties.numPr is not None
    )


def _cell_text(value: str) -> str:
    return " ".join(value.split()).replace("|", "\\|")


def _normalized_table_rows(table) -> list[list[str]]:
    seen_vertical_cells: set[object] = set()
    rows: list[list[str]] = []
    for row in table.rows:
        normalized: list[str] = []
        seen_in_row: set[object] = set()
        for cell in row.cells:
            cell_identity = cell._tc
            if cell_identity in seen_in_row or cell_identity in seen_vertical_cells:
                normalized.append("")
            else:
                normalized.append(_cell_text(cell.text))
                seen_in_row.add(cell_identity)
        seen_vertical_cells.update(seen_in_row)
        rows.append(normalized)
    return rows


def _markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _markdown_table(header: list[str], row: list[str] | None) -> str:
    lines = [_markdown_row(header), _markdown_row(["---"] * len(header))]
    if row is not None:
        padded = (row + [""] * len(header))[: len(header)]
        lines.append(_markdown_row(padded))
    return "\n".join(lines)
