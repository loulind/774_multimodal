"""Report prose: the LLM writer when available, deterministic templates otherwise.

Either way, every number in the prose is checked against the run's computed record. A
number that cannot be matched, allowing for rounding, thousands separators, sign and
proportions written as percentages, is listed in the report under "numbers not found in
the outputs", so an invented or mis-copied value is visible rather than silent. Small
integers (folds, k in recall@k), figure or table references and numbers that are part of a name
(ResNet-50) are not checked.
"""

from __future__ import annotations

import json
import re
from typing import Callable

import numpy as np

from .decision_log import DecisionLog
from .llm import LLMClient, LLMUnavailable
from .planner.llm import load_prompt
from .schema import WriterReply

SECTIONS = list(WriterReply.model_fields)
TRIVIAL_INT = 10                      # integers up to this are not checked
CONSTANTS = {95.0, 100.0}             # "95% interval", "100%"
_NUM = re.compile(r"(?<![\w.@/])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(\s?%)?(?![\w@])")
_REF = re.compile(r"(?:\bfig(?:ure)?s?\.?|\btables?|\bsection|\bappendix|\btop|r@|@|\bk\s?=|\bv)\s*-?\s*$", re.I)


def _parse(m: re.Match) -> tuple[float, int, bool]:
    whole, frac, pct = m.group(1), m.group(2) or "", m.group(3)
    return float(whole.replace(",", "") + frac), max(0, len(frac) - 1), bool(pct)


def record_numbers(record) -> np.ndarray:
    """Absolute values of every number in the record, including numbers quoted in its text."""
    vals: list[float] = []

    def walk(x):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float, np.integer, np.floating)):
            if np.isfinite(x):
                vals.append(abs(float(x)))
        elif isinstance(x, str):
            vals.extend(_parse(m)[0] for m in _NUM.finditer(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple, np.ndarray)):
            for v in x:
                walk(v)

    walk(record)
    return np.unique(np.array(vals, dtype=float))


def _matches(x: float, decimals: int, pct: bool, values: np.ndarray) -> bool:
    tol = 0.5 * 10.0 ** (-decimals) + 1e-9
    if values.size and np.any(np.abs(values - x) <= tol):
        return True
    return bool(pct and values.size and np.any(np.abs(100 * values - x) <= tol))


def check_numbers(sections: dict[str, str], record: dict) -> dict:
    values = record_numbers(record)
    unmatched, n = [], 0
    for name, text in sections.items():
        text = text or ""
        for m in _NUM.finditer(text):
            if _REF.search(text[max(0, m.start() - 12):m.start()]):
                continue
            before = text[max(0, m.start() - 2):m.start()]
            if len(before) == 2 and before[1] == "-" and before[0].isalpha():
                continue                                       # part of a name: ResNet-50, PET-CT-2
            x, decimals, pct = _parse(m)
            if (decimals == 0 and not pct and x <= TRIVIAL_INT) or x in CONSTANTS:
                continue
            n += 1
            if not _matches(x, decimals, pct, values):
                ctx = text[max(0, m.start() - 60):m.end() + 30].replace("\n", " ")
                unmatched.append({"section": name, "number": m.group(0).strip(), "context": f"...{ctx}..."})
    return {"n_checked": n, "n_unmatched": len(unmatched), "unmatched": unmatched}


def write_sections(record: dict, client: LLMClient | None, log: DecisionLog,
                   template: Callable[[dict], dict[str, str]]) -> dict:
    """Sections of prose for the report, their source, and the number check."""
    sections, source = None, "template"
    if client is not None and client.available:
        try:
            resp = client.complete("writer", load_prompt("writer"),
                                   "Write the report sections for this run.\n\n" +
                                   json.dumps(record, indent=1, default=str), WriterReply.model_json_schema())
            raw = resp.json()
            sections = {k: str(raw.get(k) or "").strip() for k in SECTIONS}
            empty = [k for k, v in sections.items() if not v]
            if len(empty) > len(SECTIONS) // 2:
                raise ValueError(f"the writer left {empty} empty")
            if empty:
                filler = template(record)
                sections.update({k: filler.get(k, "") for k in empty})
            source = f"LLM ({resp.model_used})" + (f"; template for {', '.join(empty)}" if empty else "")
        except (LLMUnavailable, ValueError) as err:
            log.add("writer", "report prose", "template", f"LLM writer failed: {err}", source="system")
            sections = None
    if sections is None:
        sections = template(record)
    check = check_numbers(sections, record)
    log.add("writer", "report prose", source,
            f"{check['n_checked']} numbers in the prose checked against the computed outputs; "
            f"{check['n_unmatched']} not found" + (": " + ", ".join(u["number"] for u in check["unmatched"][:10])
                                                   if check["unmatched"] else "."),
            [], {"n_checked": check["n_checked"], "n_unmatched": check["n_unmatched"]}, source="system")
    return {"sections": sections, "source": source, "number_check": check}
