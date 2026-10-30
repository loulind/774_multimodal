"""Text preprocessing: Unicode and whitespace normalisation.

Truncation to the planned token budget is applied by each encoder's own
tokenizer, so the cut falls on token boundaries the encoder understands.
"""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")


def clean(text: str | None, mode: str = "basic") -> str:
    if text is None:
        return ""
    if mode == "none":
        return str(text)
    s = unicodedata.normalize("NFKC", str(text))
    return _WS.sub(" ", s).strip()
