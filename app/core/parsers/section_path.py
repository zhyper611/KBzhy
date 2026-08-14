from __future__ import annotations


def update_section_stack(
    stack: list[tuple[int, str]], level: int, title: str
) -> tuple[str, ...]:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))
    return tuple(item_title for _, item_title in stack)


def section_titles(stack: list[tuple[int, str]]) -> tuple[str, ...]:
    return tuple(title for _, title in stack)
