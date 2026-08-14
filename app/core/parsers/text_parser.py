from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from KBzhy.app.core.document_models import DocumentElement
from KBzhy.app.core.parser import DocumentParseError
from KBzhy.app.core.parsers.section_path import section_titles, update_section_stack


_MARKDOWN_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+\S")
_TABLE_DIVIDER = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_CHINESE_HEADING = re.compile(r"^第[一二三四五六七八九十百千万零〇0-9]+[编章节篇部]\s*\S*")
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,5})[、.)]?\s+\S+")


def _table_cells(line: str) -> list[str] | None:
    stripped = line.strip()
    if "|" not in stripped:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]
    cells = re.split(r"(?<!\\)\|", stripped)
    return cells if len(cells) >= 2 else None


def _starts_markdown_block(line: str) -> bool:
    stripped = line.lstrip()
    return bool(
        _MARKDOWN_HEADING.match(line)
        or _LIST_ITEM.match(line)
        or _FENCE.match(line)
        or stripped.startswith(">")
        or re.match(r"^(?:-{3,}|\*{3,}|_{3,})\s*$", stripped)
    )


def read_text(source: str | Path | bytes) -> str:
    if isinstance(source, bytes):
        raw = source
    else:
        raw = Path(source).read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise DocumentParseError("文本编码无法识别")
    _validate_text_quality(text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _validate_text_quality(text: str) -> None:
    visible_count = 0
    unreadable_count = 0
    for char in text:
        category = unicodedata.category(char)
        if char == "\ufffd" or (category == "Cc" and char not in "\n\r\t"):
            raise DocumentParseError("文本质量过低")
        if char.isspace():
            continue
        visible_count += 1
        if category.startswith("C"):
            unreadable_count += 1
    if visible_count and unreadable_count / visible_count >= 0.2:
        raise DocumentParseError("文本质量过低")


def parse_markdown(source: str | Path | bytes, *, document_id: str) -> tuple[DocumentElement, ...]:
    lines = read_text(source).split("\n")
    elements: list[DocumentElement] = []
    section_stack: list[tuple[int, str]] = []
    paragraph: list[str] = []

    def append_element(element_type: str, text: str, metadata: dict | None = None) -> None:
        normalized = text.strip("\n")
        if not normalized.strip():
            return
        order = len(elements)
        elements.append(
            DocumentElement(
                element_id=f"{document_id}:markdown:{order}",
                element_type=element_type,
                text=normalized,
                order=order,
                section_path=section_titles(section_stack),
                metadata={"source_format": "markdown", **(metadata or {})},
            )
        )

    def flush_paragraph() -> None:
        if paragraph:
            append_element("paragraph", "\n".join(paragraph))
            paragraph.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        fence_match = _FENCE.match(line)
        if fence_match:
            flush_paragraph()
            marker = fence_match.group(1)
            fenced = [line]
            index += 1
            while index < len(lines):
                fenced.append(lines[index])
                if re.match(rf"^\s*{re.escape(marker[0])}{{{len(marker)},}}\s*$", lines[index]):
                    index += 1
                    break
                index += 1
            append_element("code", "\n".join(fenced), {"fence": marker[0]})
            continue

        heading_match = _MARKDOWN_HEADING.match(line)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            title = re.sub(r"[ \t]+#+[ \t]*$", "", heading_match.group(2)).strip()
            update_section_stack(section_stack, level, title)
            append_element("heading", title, {"level": level})
            index += 1
            continue

        header_cells = _table_cells(line)
        if header_cells is not None and index + 1 < len(lines) and _TABLE_DIVIDER.match(lines[index + 1]):
            divider_cells = _table_cells(lines[index + 1])
            if divider_cells is None or len(divider_cells) != len(header_cells):
                paragraph.append(line)
                index += 1
                continue
            flush_paragraph()
            table = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].strip():
                if _starts_markdown_block(lines[index]):
                    break
                row_cells = _table_cells(lines[index])
                if row_cells is None or len(row_cells) != len(header_cells):
                    break
                table.append(lines[index])
                index += 1
            append_element("table", "\n".join(table))
            continue

        if _LIST_ITEM.match(line):
            flush_paragraph()
            items = [line]
            index += 1
            while index < len(lines) and lines[index].strip():
                candidate = lines[index]
                if _LIST_ITEM.match(candidate):
                    items.append(candidate)
                    index += 1
                    continue
                if candidate[:1].isspace() and not _starts_markdown_block(candidate):
                    items.append(candidate)
                    index += 1
                    continue
                break
            append_element("list", "\n".join(items))
            continue

        if not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line)
        index += 1

    flush_paragraph()
    if not elements:
        raise DocumentParseError("未提取到可索引文字")
    return tuple(elements)


def parse_text(source: str | Path | bytes, *, document_id: str) -> tuple[DocumentElement, ...]:
    text = read_text(source)
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    elements: list[DocumentElement] = []
    section_stack: list[tuple[int, str]] = []

    for block in blocks:
        heading_level = _text_heading_level(block)
        element_type = "heading" if heading_level is not None else "paragraph"
        if heading_level is not None:
            update_section_stack(section_stack, heading_level, block)
        order = len(elements)
        elements.append(
            DocumentElement(
                element_id=f"{document_id}:text:{order}",
                element_type=element_type,
                text=block,
                order=order,
                section_path=section_titles(section_stack),
                metadata={
                    "source_format": "text",
                    **({"level": heading_level} if heading_level is not None else {}),
                },
            )
        )

    if not elements:
        raise DocumentParseError("未提取到可索引文字")
    return tuple(elements)


def _text_heading_level(block: str) -> int | None:
    if "\n" in block or len(block) > 100:
        return None
    chinese = _CHINESE_HEADING.match(block)
    if chinese:
        return 2 if "节" in chinese.group(0) else 1
    numbered = _NUMBERED_HEADING.match(block)
    if numbered:
        return min(numbered.group(1).count(".") + 1, 6)
    return None
