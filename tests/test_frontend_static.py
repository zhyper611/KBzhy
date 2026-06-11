from __future__ import annotations

from pathlib import Path


def test_chat_textarea_keeps_shift_enter_as_newline():
    source = Path("frontend/src/pages/ChatPage.jsx").read_text(encoding="utf-8")

    assert "e.shiftKey" in source
    assert "e.preventDefault()" in source


def test_chat_defaults_disable_expansion_and_rewrite():
    chat_page = Path("frontend/src/pages/ChatPage.jsx").read_text(encoding="utf-8")
    chat_api = Path("frontend/src/api/chat.js").read_text(encoding="utf-8")
    config_panel = Path("frontend/src/components/ConfigPanel.jsx").read_text(encoding="utf-8")

    assert "enable_expansion: false" in chat_page
    assert "enable_rewrite: false" in chat_page
    assert "enable_expansion = false" in chat_api
    assert "enable_rewrite = false" in chat_api
    assert 'name="enable_rewrite"' not in config_panel
    assert "多轮改写" not in config_panel


def test_frontend_removes_prompt_scene_selection():
    chat_page = Path("frontend/src/pages/ChatPage.jsx").read_text(encoding="utf-8")
    chat_api = Path("frontend/src/api/chat.js").read_text(encoding="utf-8")
    config_panel = Path("frontend/src/components/ConfigPanel.jsx").read_text(encoding="utf-8")

    assert "prompt_scene" not in chat_page
    assert "prompt_scene" not in chat_api
    assert "prompt_scene" not in config_panel
    assert "Prompt 场景" not in config_panel
    assert "SCENE_OPTIONS" not in config_panel


def test_frontend_default_similarity_threshold_is_balanced():
    chat_page = Path("frontend/src/pages/ChatPage.jsx").read_text(encoding="utf-8")
    chat_api = Path("frontend/src/api/chat.js").read_text(encoding="utf-8")
    config_panel = Path("frontend/src/components/ConfigPanel.jsx").read_text(encoding="utf-8")

    assert "similarity_threshold: 0.35" in chat_page
    assert "similarity_threshold = 0.35" in chat_api
    assert "similarity_threshold: 0.35" in config_panel
    assert "0.35: '默认'" in config_panel


def test_vite_splits_large_vendor_chunks():
    config = Path("frontend/vite.config.js").read_text(encoding="utf-8")

    assert "manualChunks" in config
    assert "react: ['react', 'react-dom', 'react-router-dom']" in config
    assert "antd: ['antd', '@ant-design/icons']" in config
    assert "markdown: ['react-markdown']" in config
    assert "chunkSizeWarningLimit: 1000" in config


def test_app_lazy_loads_pages():
    source = Path("frontend/src/App.jsx").read_text(encoding="utf-8")

    assert "lazy" in source
    assert "Suspense" in source
    assert "const ChatPage = lazy(() => import('./pages/ChatPage'))" in source
    assert "const DocumentsPage = lazy(() => import('./pages/DocumentsPage'))" in source


def test_chat_refreshes_session_list_after_first_question():
    source = Path("frontend/src/pages/ChatPage.jsx").read_text(encoding="utf-8")

    assert "const isFirstQuestion = messages.length === 0;" in source
    assert "if (isFirstQuestion) setRefreshKey((k) => k + 1);" in source


def test_stream_chat_parses_sse_events_with_multiline_data():
    source = Path("frontend/src/api/chat.js").read_text(encoding="utf-8")

    assert "split(/\\r?\\n/)" in source
    assert "buffer.split(/\\r?\\n\\r?\\n/)" in source
    assert ".filter((line) => line.startsWith('data:'))" in source
    assert ".join('\\n')" in source


def test_chat_markdown_supports_gfm_and_soft_breaks():
    source = Path("frontend/src/components/ChatBubble.jsx").read_text(encoding="utf-8")

    assert "import remarkGfm from 'remark-gfm';" in source
    assert "import remarkBreaks from 'remark-breaks';" in source
    assert "remarkPlugins={[remarkGfm, remarkBreaks]}" in source


def test_document_table_can_open_document_chunks_drawer():
    table = Path("frontend/src/components/DocumentTable.jsx").read_text(encoding="utf-8")
    api = Path("frontend/src/api/documents.js").read_text(encoding="utf-8")

    assert "getDocumentChunks" in api
    assert "Drawer" in table
    assert "getDocumentChunks" in table
    assert "onRow" in table
    assert "selectedDoc" in table
    assert "chunks" in table
