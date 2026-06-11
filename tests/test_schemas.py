from __future__ import annotations

from KBzhy.app.models.schemas import ChatRequest


def test_chat_request_defaults_disable_expensive_preprocessing():
    request = ChatRequest(question="孟子的主要思想")

    assert request.enable_expansion is False
    assert request.enable_rewrite is False


def test_chat_request_default_similarity_threshold_is_balanced():
    request = ChatRequest(question="孟子的主要思想")

    assert request.similarity_threshold == 0.35


def test_chat_request_has_no_prompt_scene_field():
    assert "prompt_scene" not in ChatRequest.model_fields
