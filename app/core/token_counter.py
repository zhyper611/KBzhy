from __future__ import annotations

import tiktoken

from KBzhy.config import TOKEN_ENCODING


class TokenCounter:
    def __init__(self, encoding_name: str = TOKEN_ENCODING):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if max_tokens == 0:
            return ""

        tokens = self.encoding.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text

        truncated_bytes = b"".join(
            self.encoding.decode_single_token_bytes(token)
            for token in tokens[:max_tokens]
        )
        truncated = truncated_bytes.decode("utf-8", errors="ignore")
        while self.count(truncated) > max_tokens:
            truncated = truncated[:-1]
        return truncated
