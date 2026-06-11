from __future__ import annotations

from KBzhy.app.api.chat import _sse_data


def test_sse_data_preserves_embedded_newlines():
    assert _sse_data("第一行\n第二行") == "data: 第一行\ndata: 第二行\n\n"
    assert _sse_data("第一行\n") == "data: 第一行\ndata: \n\n"
    assert _sse_data("\n") == "data: \ndata: \n\n"
