from __future__ import annotations


def normalize_symbol(value: str | int) -> str:
    text = str(value).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    return text.zfill(6)


def normalize_symbol_series(value: object) -> str:
    return normalize_symbol(value)
