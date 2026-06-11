from __future__ import annotations

from KBzhy.app.core.splitter import SmartSplitter


def test_clause_split_keeps_clause_numbers():
    content = "\n".join(
        [
            "第一条 总则内容",
            "第二条 用户应遵守平台规则",
            "第三条 管理员负责审核",
            "第四条 数据应定期备份",
            "第五条 违规将被处理",
        ]
    )
    splitter = SmartSplitter(chunk_size=20, chunk_overlap=0)

    chunks = splitter.split(content, doc_type="text")
    joined = "\n".join(chunk.content for chunk in chunks)

    assert "第一条" in joined
    assert "第五条" in joined
