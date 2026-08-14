from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from KBzhy.app.core.document_models import DocumentElement, ParsedDocument
from KBzhy.app.core.parser import Document, DocumentParser, ParsedArtifactError


def _markdown_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\nInstall the package.", encoding="utf-8")
    return path


def test_parse_structured_returns_parsed_document_and_keeps_legacy_parse(tmp_path):
    source = _markdown_fixture(tmp_path)
    parser = DocumentParser()

    legacy = parser.parse(source)
    parsed = parser.parse_structured(
        source,
        document_id="doc1",
        version=2,
        kb_id="kb1",
    )

    assert isinstance(legacy, list)
    assert all(isinstance(document, Document) for document in legacy)
    assert parsed.document_id == "doc1"
    assert parsed.version == 2
    assert parsed.title == "guide"
    assert parsed.language == "und"
    assert parsed.metadata["kb_id"] == "kb1"
    assert parsed.elements
    assert parsed.elements[0].text == legacy[0].content


def test_legacy_documents_become_ordered_paragraph_and_table_elements(monkeypatch):
    parser = DocumentParser()
    legacy_documents = [
        Document("first paragraph", {"page": 3, "source": "guide.pdf"}),
        Document("name | value", {"sheet": "Sheet1", "source": "guide.xlsx"}),
    ]
    monkeypatch.setattr(parser, "parse", lambda _source: legacy_documents)

    parsed = parser.parse_structured(
        "ignored.txt",
        document_id="doc1",
        version=1,
        kb_id="kb1",
    )

    assert [element.element_type for element in parsed.elements] == ["paragraph", "table"]
    assert [element.order for element in parsed.elements] == [0, 1]
    assert parsed.elements[0].page == 3
    assert parsed.elements[1].metadata["sheet"] == "Sheet1"


def test_parsed_artifact_round_trip_is_lossless(tmp_path):
    parser = DocumentParser(artifact_dir=tmp_path)
    parsed = ParsedDocument(
        document_id="doc1",
        version=4,
        title="指南",
        language="zh-CN",
        elements=(
            DocumentElement(
                element_id="doc1:0",
                element_type="paragraph",
                text="正文",
                order=0,
                page=2,
                section_path=("第一章", "概述"),
                bounding_box={"x0": 1.25, "y0": 2.5},
                metadata={"bold": True, "nested": {"values": [1, 2]}},
            ),
        ),
        metadata={"kb_id": "kb1", "source": "指南.md", "tags": ["a", "b"]},
    )

    path = parser.save_artifact(parsed)

    assert path == tmp_path / "kb1" / "doc1" / "v4.json"
    assert parser.load_artifact(path) == parsed


def test_load_artifact_rejects_path_outside_artifact_directory(tmp_path):
    parser = DocumentParser(artifact_dir=tmp_path / "artifacts")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_dir"):
        parser.load_artifact(outside)


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        "[]",
        '{"document_id":"doc1","version":1,"elements":[],"metadata":{}}',
        '{"document_id":1,"version":1,"elements":[],"metadata":{}}',
        '{"document_id":"doc1","version":"1","elements":[],"metadata":{}}',
        '{"document_id":"doc1","version":1,"elements":[],"metadata":[]}',
        '{"document_id":"doc1","version":1,"elements":{},"metadata":{}}',
        (
            '{"document_id":"doc1","version":1,"title":"guide",'
            '"language":"und","elements":[{"element_id":"e1",'
            '"element_type":"image","text":"body","order":0}],"metadata":{}}'
        ),
        (
            '{"document_id":"doc1","version":1,"title":"guide",'
            '"language":"und","elements":["paragraph"],"metadata":{}}'
        ),
        (
            '{"document_id":"doc1","version":1,"title":"guide",'
            '"language":"und","elements":[{"element_id":"e1",'
            '"element_type":"paragraph","text":"body","order":"0"}],"metadata":{}}'
        ),
    ],
)
def test_load_artifact_rejects_malformed_artifacts_with_stable_error(tmp_path, payload):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    artifact = artifact_dir / "bad.json"
    artifact.write_text(payload, encoding="utf-8")
    parser = DocumentParser(artifact_dir=artifact_dir)

    with pytest.raises(ParsedArtifactError, match="invalid parsed artifact"):
        parser.load_artifact(artifact)


def test_save_artifact_uses_atomic_replace_and_cleans_temp_on_failure(tmp_path, monkeypatch):
    parser = DocumentParser(artifact_dir=tmp_path)
    parsed = ParsedDocument(
        document_id="doc1",
        version=1,
        title="guide",
        language="und",
        metadata={"kb_id": "kb1"},
    )

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr("KBzhy.app.core.parser.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        parser.save_artifact(parsed)

    target_dir = tmp_path / "kb1" / "doc1"
    assert not (target_dir / "v1.json").exists()
    assert list(target_dir.glob("*.tmp")) == []


def test_save_artifact_cleans_temp_when_serialization_fails(tmp_path):
    parser = DocumentParser(artifact_dir=tmp_path)
    parsed = ParsedDocument(
        document_id="doc1",
        version=1,
        title="guide",
        language="und",
        metadata={"kb_id": "kb1", "unsupported": object()},
    )

    with pytest.raises(TypeError):
        parser.save_artifact(parsed)

    target_dir = tmp_path / "kb1" / "doc1"
    assert list(target_dir.glob("*.tmp")) == []


def test_save_artifact_rejects_symlink_escape(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    artifact_dir.mkdir()
    outside.mkdir()
    link = artifact_dir / "kb1"
    junction_created = False
    try:
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            if os.name != "nt":
                pytest.skip(f"directory symlinks unavailable: {exc}")
            result = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(outside)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                pytest.skip(f"directory links unavailable: {result.stderr}")
            junction_created = True

        parser = DocumentParser(artifact_dir=artifact_dir)
        parsed = ParsedDocument(
            document_id="doc1",
            version=1,
            title="guide",
            language="und",
            metadata={"kb_id": "kb1"},
        )

        with pytest.raises(ValueError, match="artifact_dir"):
            parser.save_artifact(parsed)

        assert not (outside / "doc1" / "v1.json").exists()
    finally:
        if junction_created and link.exists():
            os.rmdir(link)


@pytest.mark.parametrize("field,value", [("kb_id", "../kb"), ("document_id", "doc/child")])
def test_artifact_path_rejects_unsafe_identifiers(tmp_path, field, value):
    parser = DocumentParser(artifact_dir=tmp_path)
    values = {"kb_id": "kb1", "document_id": "doc1"}
    values[field] = value
    parsed = ParsedDocument(
        document_id=values["document_id"],
        version=1,
        title="guide",
        language="und",
        metadata={"kb_id": values["kb_id"]},
    )

    with pytest.raises(ValueError, match=field):
        parser.save_artifact(parsed)


@pytest.mark.parametrize("version", [0, -1, True])
def test_parse_structured_rejects_invalid_version(tmp_path, version):
    parser = DocumentParser()

    with pytest.raises(ValueError, match="version"):
        parser.parse_structured(
            _markdown_fixture(tmp_path),
            document_id="doc1",
            version=version,
            kb_id="kb1",
        )


@pytest.mark.parametrize(
    "raw_section_path,expected",
    [(None, ()), ("Overview", ("Overview",)), (["A", "B"], ("A", "B")), (("A",), ("A",))],
)
def test_legacy_section_path_is_normalized(monkeypatch, raw_section_path, expected):
    parser = DocumentParser()
    monkeypatch.setattr(
        parser,
        "parse",
        lambda _source: [Document("body", {"section_path": raw_section_path})],
    )

    parsed = parser.parse_structured("ignored.txt", document_id="doc1", version=1)

    assert parsed.elements[0].section_path == expected


def test_legacy_section_path_rejects_unsupported_type(monkeypatch):
    parser = DocumentParser()
    monkeypatch.setattr(
        parser,
        "parse",
        lambda _source: [Document("body", {"section_path": {"heading": "A"}})],
    )

    with pytest.raises(ParsedArtifactError, match="section_path"):
        parser.parse_structured("ignored.txt", document_id="doc1", version=1)
