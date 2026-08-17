from __future__ import annotations

import importlib
import os
from pathlib import Path

from KBzhy import config


def test_shared_env_file_precedes_project_dotenv(monkeypatch, tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    shared_env = tmp_path / "shared.env"
    shared_env.write_text("MYSQL_USER=shared-user\n", encoding="utf-8")
    (project_dir / ".env").write_text(
        f"KBZHY_ENV_FILE={shared_env.as_posix()}\nMYSQL_USER=project-user\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("KBZHY_ENV_FILE", raising=False)
    monkeypatch.delenv("MYSQL_USER", raising=False)

    config._load_environment(str(project_dir))

    assert os.getenv("MYSQL_USER") == "shared-user"


def test_parent_dotenv_remains_the_lowest_priority_fallback(monkeypatch, tmp_path):
    project_dir = tmp_path / "nested" / "project"
    project_dir.mkdir(parents=True)
    (tmp_path / ".env").write_text("MYSQL_USER=parent-user\n", encoding="utf-8")
    monkeypatch.delenv("KBZHY_ENV_FILE", raising=False)
    monkeypatch.delenv("KBZHY_STORAGE_ROOT", raising=False)
    monkeypatch.delenv("MYSQL_USER", raising=False)

    config._load_environment(str(project_dir))

    assert os.getenv("MYSQL_USER") == "parent-user"


def test_storage_paths_follow_configured_root(monkeypatch, tmp_path):
    monkeypatch.setenv("KBZHY_STORAGE_ROOT", str(tmp_path))
    monkeypatch.delenv("KBZHY_CHROMA_PERSIST_DIR", raising=False)
    monkeypatch.delenv("KBZHY_DATA_DIR", raising=False)
    try:
        loaded = importlib.reload(config)

        assert Path(loaded.CHROMA_PERSIST_DIR) == tmp_path / "chroma_db"
        assert Path(loaded.DATA_DIR) == tmp_path / "data"
        assert Path(loaded.FILE_STORAGE_DIR) == tmp_path / "data" / "conversations"
        assert Path(loaded.UPLOAD_STORAGE_DIR) == tmp_path / "data" / "uploads"
        assert Path(loaded.PARSED_ARTIFACT_DIR) == tmp_path / "data" / "parsed"
    finally:
        monkeypatch.delenv("KBZHY_STORAGE_ROOT", raising=False)
        importlib.reload(config)


def test_individual_storage_path_overrides_root(monkeypatch, tmp_path):
    root = tmp_path / "root"
    chroma = tmp_path / "external-chroma"
    data = tmp_path / "external-data"
    monkeypatch.setenv("KBZHY_STORAGE_ROOT", str(root))
    monkeypatch.setenv("KBZHY_CHROMA_PERSIST_DIR", str(chroma))
    monkeypatch.setenv("KBZHY_DATA_DIR", str(data))
    try:
        loaded = importlib.reload(config)

        assert Path(loaded.CHROMA_PERSIST_DIR) == chroma
        assert Path(loaded.DATA_DIR) == data
        assert Path(loaded.UPLOAD_STORAGE_DIR) == data / "uploads"
    finally:
        monkeypatch.delenv("KBZHY_STORAGE_ROOT", raising=False)
        monkeypatch.delenv("KBZHY_CHROMA_PERSIST_DIR", raising=False)
        monkeypatch.delenv("KBZHY_DATA_DIR", raising=False)
        importlib.reload(config)


def test_storage_root_dotenv_supplies_shared_runtime_configuration(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "MYSQL_USER=shared-user\nDASHSCOPE_API_KEY=shared-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KBZHY_STORAGE_ROOT", str(tmp_path))
    monkeypatch.delenv("MYSQL_USER", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    try:
        loaded = importlib.reload(config)

        assert loaded.MYSQL_USER == "shared-user"
        assert loaded.API_KEY == "shared-key"
    finally:
        monkeypatch.delenv("KBZHY_STORAGE_ROOT", raising=False)
        monkeypatch.delenv("MYSQL_USER", raising=False)
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        importlib.reload(config)


def test_explicit_shared_env_file_supplies_runtime_configuration(monkeypatch, tmp_path):
    shared_env = tmp_path / "shared.env"
    shared_env.write_text(
        "MYSQL_USER=explicit-user\nDASHSCOPE_API_KEY=explicit-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KBZHY_ENV_FILE", str(shared_env))
    monkeypatch.delenv("MYSQL_USER", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    try:
        loaded = importlib.reload(config)

        assert loaded.MYSQL_USER == "explicit-user"
        assert loaded.API_KEY == "explicit-key"
    finally:
        monkeypatch.delenv("KBZHY_ENV_FILE", raising=False)
        monkeypatch.delenv("MYSQL_USER", raising=False)
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        importlib.reload(config)
