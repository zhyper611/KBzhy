"""多格式文档解析器

支持 PDF / Word / Excel / PPT / txt / md / csv / 图片(OCR via VLM)
"""

from __future__ import annotations

import io
import logging
import os
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Document:
    """解析后的文档"""

    def __init__(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.content = content
        self.metadata = metadata or {}
        self.doc_id: str = str(uuid.uuid4())

    def __repr__(self) -> str:
        src = self.metadata.get("source", "unknown")
        return f"Document(id={self.doc_id[:8]}, source={src}, len={len(self.content)})"


class DocumentParser:
    """多格式文档解析器"""

    SUPPORTED_TYPES: dict[str, list[str]] = {
        "pdf": [".pdf"],
        "word": [".docx"],
        "excel": [".xlsx", ".xls"],
        "ppt": [".pptx", ".ppt"],
        "text": [".txt", ".md"],
        "csv": [".csv"],
        "image": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    }

    def __init__(self, vlm_model: str | None = None):
        self._vlm_model = vlm_model
        self._ext_map = self._build_ext_map()

    def _build_ext_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for category, exts in self.SUPPORTED_TYPES.items():
            for ext in exts:
                mapping[ext] = category
        return mapping

    def parse(self, file_path: str | Path) -> list[Document]:
        """解析文件，返回文档列表"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = file_path.suffix.lower()
        category = self._ext_map.get(ext)
        if category is None:
            raise ValueError(f"不支持的文件格式: {ext}")

        logger.info("解析文档: %s (类型: %s)", file_path.name, category)

        parser = getattr(self, f"_parse_{category}", None)
        if parser is None:
            raise NotImplementedError(f"解析器未实现: {category}")

        docs = parser(str(file_path))
        for doc in docs:
            doc.metadata.setdefault("source", file_path.name)
            doc.metadata.setdefault("file_type", category)
            doc.metadata.setdefault("file_path", str(file_path))
        return docs

    def parse_bytes(self, content: bytes, filename: str) -> list[Document]:
        """解析字节流（上传文件用）"""
        ext = Path(filename).suffix.lower()
        category = self._ext_map.get(ext)
        if category is None:
            raise ValueError(f"不支持的文件格式: {ext}")

        parser = getattr(self, f"_parse_{category}")
        docs = parser(io.BytesIO(content)) if category in ("pdf", "image") else parser(content)
        for doc in docs:
            doc.metadata.setdefault("source", filename)
            doc.metadata.setdefault("file_type", category)
        return docs

    # ── PDF ────────────────────────────────────

    def _parse_pdf(self, source: str | io.BytesIO) -> list[Document]:
        import fitz

        docs: list[Document] = []
        if isinstance(source, str):
            pdf = fitz.open(source)
        else:
            pdf = fitz.open(stream=source.read(), filetype="pdf")

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text().strip()
            if text:
                docs.append(Document(
                    content=text,
                    metadata={"page": page_num + 1},
                ))
        pdf.close()
        return docs if docs else [Document(content="[PDF 无法提取文字，可能是扫描件]", metadata={"page": 0})]

    # ── Word ───────────────────────────────────

    def _parse_word(self, source: str | bytes) -> list[Document]:
        from docx import Document as DocxDocument

        if isinstance(source, str):
            docx = DocxDocument(source)
        else:
            docx = DocxDocument(io.BytesIO(source))

        full_text: list[str] = []
        for para in docx.paragraphs:
            full_text.append(para.text)

        for table in docx.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                full_text.append(" | ".join(cells))

        return [Document(content="\n".join(full_text))]

    # ── Excel ──────────────────────────────────

    def _parse_excel(self, source: str | bytes) -> list[Document]:
        import openpyxl

        if isinstance(source, str):
            wb = openpyxl.load_workbook(source, data_only=True)
        else:
            wb = openpyxl.load_workbook(io.BytesIO(source), data_only=True)

        docs: list[Document] = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows: list[str] = []
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(c.strip() for c in cells):
                    rows.append(" | ".join(cells))
            if rows:
                docs.append(Document(
                    content="\n".join(rows),
                    metadata={"sheet": sheet_name},
                ))
        wb.close()
        return docs if docs else [Document(content="[Excel 文件为空]")]

    # ── PPT ────────────────────────────────────

    def _parse_ppt(self, source: str | bytes) -> list[Document]:
        from pptx import Presentation

        if isinstance(source, str):
            prs = Presentation(source)
        else:
            prs = Presentation(io.BytesIO(source))

        docs: list[Document] = []
        for slide_num, slide in enumerate(prs.slides, 1):
            texts: list[str] = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        t = para.text.strip()
                        if t:
                            texts.append(t)
                if shape.has_table:
                    table = shape.table
                    for row in table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        texts.append(" | ".join(cells))
            if texts:
                docs.append(Document(
                    content="\n".join(texts),
                    metadata={"slide": slide_num},
                ))
        return docs if docs else [Document(content="[PPT 无文字内容]")]

    # ── 纯文本 ─────────────────────────────────

    @staticmethod
    def _read_with_fallback(file_path: str) -> str:
        """读取文本文件，UTF-8 → GBK → GB18030 → latin-1 依次尝试"""
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "gb18030", "latin-1"]
        for enc in encodings:
            try:
                return Path(file_path).read_text(encoding=enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return Path(file_path).read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _decode_with_fallback(data: bytes) -> str:
        """解码字节，UTF-8 → GBK → GB18030 → latin-1 依次尝试"""
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "gb18030", "latin-1"]
        for enc in encodings:
            try:
                return data.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return data.decode("utf-8", errors="replace")

    def _parse_text(self, source: str | bytes) -> list[Document]:
        if isinstance(source, str):
            content = self._read_with_fallback(source)
        else:
            content = self._decode_with_fallback(source)
        return [Document(content=content)]

    # ── CSV ────────────────────────────────────

    def _parse_csv(self, source: str | bytes) -> list[Document]:
        import csv

        if isinstance(source, str):
            text = self._read_with_fallback(source)
        else:
            text = self._decode_with_fallback(source)

        reader = csv.reader(io.StringIO(text))
        rows = [" | ".join(row) for row in reader if any(c.strip() for c in row)]
        return [Document(content="\n".join(rows))]

    # ── 图片 (OCR via VLM) ─────────────────────

    def _parse_image(self, source: str | io.BytesIO) -> list[Document]:
        import base64

        if isinstance(source, str):
            with open(source, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            source_name = os.path.basename(source)
        else:
            img_data = base64.b64encode(source.read()).decode()
            source_name = "uploaded_image"

        from KBzhy.config import API_KEY, API_BASE, LLM_MODEL

        if not API_KEY or API_KEY == "your-api-key":
            return [Document(content="[图片 OCR 需要配置 API Key]")]

        model = self._vlm_model or LLM_MODEL

        try:
            import httpx
            resp = httpx.post(
                f"{API_BASE.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
                                },
                                {
                                    "type": "text",
                                    "text": "请提取并输出这张图片中的所有文字内容，保持原文格式（表格、列表等结构）。只输出文字，不要添加额外说明。",
                                },
                            ],
                        }
                    ],
                    "max_tokens": 4096,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return [Document(content=text, metadata={"ocr_method": "vlm", "source": source_name})]
        except Exception as exc:
            logger.error("VLM OCR 失败: %s", exc)
            return [Document(content=f"[图片 OCR 失败: {exc}]", metadata={"source": source_name})]
