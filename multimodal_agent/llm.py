"""The one place the agent talks to a language model (adapted from Projects 1-2).

Planner, critic and report writer call ``LLMClient.complete(role, system, user, schema)``.
Which model answers is decided here:

    claude_cli   the local Claude Code CLI, driven through claude-agent-sdk with the CLI's
                 existing login (no API key). Built-in tools, MCP servers and the user's
                 own settings and CLAUDE.md files are switched off, so each call is a plain
                 completion: prompts in, text (or JSON) out.
    offline      always unavailable; callers fall back to the rule-based planner, critic
                 and writer. ``--no-llm`` selects it.

Model. The configured name is passed to the CLI and the model that actually answered is
read back from every response, so a substitution is visible in the cache and the run
manifest. A name the CLI does not know, a missing login or a missing CLI fails the first
call; the client then stops calling (one failure, not three) and every later step uses
its rule-based path, with the CLI's message in the decision log.

Structured output. When the caller passes a JSON schema, the CLI's structured-output mode
enforces it. If that mode fails, the call is retried once as plain text and the JSON is
extracted from the reply. Either way the content is validated afterwards in Python.

Limits. Output tokens per call (CLAUDE_CODE_MAX_OUTPUT_TOKENS), reasoning effort per
role, a wall-clock timeout per call, and a spend cap per run: the CLI is given what is
left of ``max_budget_usd``, and once it is used up later calls fall back to rules.

Reproducibility. Sampling cannot be fixed, so every response is written to
``<run>/llm_cache/<role>_<hash>.json`` with the exact prompts, the model requested and
used, token usage, cost and timing; the run's outputs also keep the same records as
``llm_transcript.json``. ``--replay`` takes either (a cache directory, a transcript file,
or a run directory holding one) and never calls a model, so a run can be repeated
decision for decision.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import LLMConfig

# Failures that will not go away by retrying: stop calling for the rest of the run.
FATAL = ("authenticate", "logged in", "login", "oauth", "not found", "unrecognized", "invalid model",
         "budget", "no such file")


class LLMUnavailable(RuntimeError):
    """No model could be reached (or the replay cache lacks this prompt)."""


@dataclass
class LLMResponse:
    text: str
    backend: str
    model_requested: str
    model_used: str
    seconds: float
    cost_usd: float | None = None
    data: dict | None = None             # JSON from the CLI's structured-output mode
    usage: dict = field(default_factory=dict)
    structured: bool = False
    cached: bool = False
    note: str = ""

    def json(self) -> dict:
        """The reply as a JSON object (structured output if present, else parsed from the text)."""
        return self.data if isinstance(self.data, dict) else extract_json(self.text)

    def to_record(self) -> dict:
        d = asdict(self)
        d.pop("text"), d.pop("data")
        d["response_chars"] = len(self.text)
        return d


def _run_sync(make_coro, timeout: float):
    """Run an async SDK call from synchronous code, also from inside a running event loop."""
    async def bounded():
        return await asyncio.wait_for(make_coro(), timeout)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(bounded())
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(lambda: asyncio.run(bounded())).result()


def is_fatal(err: Exception) -> bool:
    msg = str(err).lower()
    return any(k in msg for k in FATAL)


class ClaudeCLIBackend:
    name = "claude_cli"

    def __init__(self, cfg: LLMConfig):
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as err:
            raise LLMUnavailable("claude-agent-sdk is not installed (uv sync)") from err
        self.cfg = cfg
        self.model = cfg.model
        self.max_output_tokens = cfg.max_output_tokens

    def complete(self, system: str, user: str, schema: dict | None = None, effort: str | None = None,
                 budget_usd: float | None = None) -> LLMResponse:
        note = ""
        if schema is not None and self.cfg.structured_output:
            try:
                return self._once(system, user, schema, effort, budget_usd)
            except LLMUnavailable as err:
                if is_fatal(err) or "timed out" in str(err):
                    raise
                note = f"structured output failed ({str(err)[:200]}); retried as plain text"
        resp = self._with_retries(system, user, None, effort, budget_usd)
        resp.note = note
        return resp

    def _with_retries(self, system, user, schema, effort, budget_usd) -> LLMResponse:
        for attempt in range(self.cfg.max_retries + 1):
            try:
                return self._once(system, user, schema, effort, budget_usd)
            except LLMUnavailable as err:
                if is_fatal(err) or "timed out" in str(err) or attempt == self.cfg.max_retries:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def _once(self, system, user, schema, effort, budget_usd) -> LLMResponse:
        from claude_agent_sdk import ClaudeAgentOptions, query

        options = ClaudeAgentOptions(
            system_prompt=system, model=self.model,
            tools=[], strict_mcp_config=True,        # no tools, no MCP servers: our Python is the tool
            setting_sources=[],                      # ignore the user's settings and CLAUDE.md files
            max_turns=3, effort=effort, max_budget_usd=budget_usd,
            output_format={"type": "json_schema", "schema": schema} if schema else None,
            env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(self.max_output_tokens)},
        )

        async def go():
            chunks, used, result = [], None, None
            async for message in query(prompt=user, options=options):
                kind = type(message).__name__
                if kind == "AssistantMessage":
                    used = getattr(message, "model", None) or used
                    for block in getattr(message, "content", []) or []:
                        if type(block).__name__ == "TextBlock":
                            chunks.append(block.text)
                elif kind == "ResultMessage":
                    result = message
            return "".join(chunks), used, result

        start = time.perf_counter()
        try:
            text, used, result = _run_sync(go, self.cfg.timeout_seconds)
        except TimeoutError as err:
            raise LLMUnavailable(f"CLI call timed out after {self.cfg.timeout_seconds:.0f}s") from err
        except Exception as err:                     # the SDK raises on CLI error results
            raise LLMUnavailable(f"claude CLI failed: {str(err)[:300]}") from err
        if result is None:
            raise LLMUnavailable("claude CLI returned no result")
        if result.is_error:
            raise LLMUnavailable(f"claude CLI error ({result.subtype}): {str(result.result or '')[:300]}")
        data = result.structured_output if schema else None
        if not text.strip() and data is None:
            raise LLMUnavailable("claude CLI returned an empty reply")
        u = result.usage or {}
        usage = {k: u.get(k) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                       "cache_creation_input_tokens")}
        usage["stop_reason"] = result.stop_reason
        usage["models"] = sorted((result.model_usage or {}).keys())
        return LLMResponse(text=text if data is None else json.dumps(data), backend=self.name,
                           model_requested=self.model, model_used=used or (usage["models"] or ["unknown"])[0],
                           seconds=time.perf_counter() - start, cost_usd=result.total_cost_usd,
                           data=data if isinstance(data, dict) else None, usage=usage,
                           structured=isinstance(data, dict))


class OfflineBackend:
    name = "offline"

    def __init__(self, model: str = "none"):
        self.model = model                          # replay keys include the model name

    def complete(self, system, user, schema=None, effort=None, budget_usd=None) -> LLMResponse:
        raise LLMUnavailable("offline: --no-llm selected or no backend available")


class LLMClient:
    """Backend + per-run disk cache + replay + spend cap."""

    def __init__(self, backend, cache_dir: str | Path, replay_from: str | Path | None = None,
                 max_budget_usd: float | None = None, effort: dict[str, str] | None = None):
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.replay_from = Path(replay_from) if replay_from else None
        self.max_budget_usd = max_budget_usd
        self.effort = effort or {}
        self.calls: list[dict] = []
        self.records: list[dict] = []               # full records (prompts + replies) of this run's calls
        self.notes: list[str] = []
        self.disabled: str | None = None            # set after a failure retrying cannot fix
        self._sequence: dict[str, int] = {}
        self._replay = load_records(self.replay_from) if self.replay_from is not None else None

    @property
    def available(self) -> bool:
        if self.replay_from is not None:
            return True
        return self.disabled is None and not isinstance(self.backend, OfflineBackend)

    @property
    def model(self) -> str:
        return getattr(self.backend, "model", "none")

    @property
    def spent_usd(self) -> float:
        return sum(c.get("cost_usd") or 0.0 for c in self.calls if not c.get("cached"))

    def _key(self, role: str, system: str, user: str, schema: dict | None) -> str:
        blob = json.dumps({"model": self.model, "system": system, "user": user, "schema": schema},
                          sort_keys=True, default=str)
        return f"{role}_{hashlib.sha256(blob.encode()).hexdigest()[:16]}.json"

    def complete(self, role: str, system: str, user: str, schema: dict | None = None) -> LLMResponse:
        seq = self._sequence.get(role, 0)
        self._sequence[role] = seq + 1
        key = self._key(role, system, user, schema)
        if self._replay is not None:
            hit = next((r for r in self._replay if r.get("key") == key), None)
            if hit is not None:
                return self._from_record(role, key, hit)
            # Prompts embed run results, which can differ in the last digits between machines
            # (GPU feature extraction) or in measured times. The sequence of calls is what a
            # replay reproduces, so fall back to the response this role gave at the same
            # position in the original run.
            hit = next((r for r in self._replay if r.get("role") == role and r.get("sequence") == seq), None)
            if hit is not None:
                return self._from_record(role, key, hit, matched="role and position")
            raise LLMUnavailable(f"replay: no cached response for {role} call {seq} ({key})")
        if (self.cache_dir / key).exists():
            return self._from_record(role, key, json.loads((self.cache_dir / key).read_text()))
        if self.disabled:
            raise LLMUnavailable(f"LLM disabled for this run: {self.disabled}")
        budget = None
        if self.max_budget_usd is not None:
            budget = round(self.max_budget_usd - self.spent_usd, 4)
            if budget <= 0.01:
                raise LLMUnavailable(f"LLM spend cap reached (${self.spent_usd:.2f} of "
                                     f"${self.max_budget_usd:.2f}); rule-based path used")
        effort = self.effort.get(role.split("_")[0])
        try:
            resp = self.backend.complete(system, user, schema=schema, effort=effort, budget_usd=budget)
        except LLMUnavailable as err:
            if is_fatal(err):
                self.disabled = str(err)
                self.notes.append(f"LLM calls stopped after '{role}': {err}")
            raise
        if resp.model_used not in ("unknown", resp.model_requested):
            self.notes.append(f"{role}: requested {resp.model_requested}, answered by {resp.model_used}")
        if resp.note:
            self.notes.append(f"{role}: {resp.note}")
        rec = {"key": key, "role": role, "sequence": seq, **resp.to_record(), "effort": effort,
               "max_output_tokens": getattr(self.backend, "max_output_tokens", None),
               "budget_usd_given": budget, "system": system, "user": user, "schema": schema,
               "text": resp.text, "data": resp.data}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / key).write_text(json.dumps(rec, indent=1, default=str))
        self.records.append(rec)
        self._record(role, key, resp)
        return resp

    def _from_record(self, role: str, key: str, rec: dict, matched: str = "exact prompt") -> LLMResponse:
        resp = LLMResponse(text=rec["text"], backend=rec["backend"], model_requested=rec["model_requested"],
                           model_used=rec["model_used"], seconds=0.0, cost_usd=rec.get("cost_usd"),
                           data=rec.get("data"), usage=rec.get("usage") or {},
                           structured=rec.get("structured", False), cached=True,
                           note=f"from cache ({matched})")
        rec = {**rec, "key": key, "role": role, "sequence": self._sequence[role] - 1}
        self.cache_dir.mkdir(parents=True, exist_ok=True)   # keep this run's cache complete
        (self.cache_dir / key).write_text(json.dumps(rec, indent=1, default=str))
        self.records.append(rec)
        self._record(role, key, resp)
        return resp

    def _record(self, role: str, key: str, resp: LLMResponse) -> None:
        self.calls.append({"role": role, "cache_file": key, **resp.to_record()})

    def summary(self) -> dict:
        """Provenance for the run manifest."""
        return {"backend": getattr(self.backend, "name", "offline"), "model_requested": self.model,
                "models_used": sorted({c["model_used"] for c in self.calls}),
                "replay_from": str(self.replay_from) if self.replay_from else None,
                "n_calls": len(self.calls), "n_from_cache": sum(c["cached"] for c in self.calls),
                "total_cost_usd": round(self.spent_usd, 4), "max_budget_usd": self.max_budget_usd,
                "disabled": self.disabled, "notes": self.notes, "calls": self.calls}


def load_records(path: str | Path) -> list[dict]:
    """LLM call records from a cache directory, a transcript file, or a run directory holding one."""
    path = Path(path)
    if path.is_dir() and (path / "llm_cache").is_dir():
        path = path / "llm_cache"
    elif path.is_dir() and (path / "llm_transcript.json").is_file():
        # A run directory without its cache (llm_cache/ is not committed, so a clone has
        # only the transcript); cache directories never hold a transcript.
        path = path / "llm_transcript.json"
    if path.is_file():
        data = json.loads(path.read_text())
        return data.get("calls", []) if isinstance(data, dict) else data
    if not path.is_dir():
        raise LLMUnavailable(f"replay source {path} not found")
    out = []
    for f in sorted(path.glob("*.json")):
        rec = json.loads(f.read_text())
        rec.setdefault("key", f.name)
        out.append(rec)
    return out


def make_client(cfg: LLMConfig, cache_dir: str | Path, replay_from: str | Path | None = None,
                enabled: bool | None = None) -> LLMClient:
    enabled = cfg.enabled if enabled is None else enabled
    if replay_from:
        backend = OfflineBackend(cfg.model)
    elif not enabled:
        backend = OfflineBackend()
    else:
        try:
            backend = ClaudeCLIBackend(cfg)
        except LLMUnavailable as err:
            client = LLMClient(OfflineBackend(), cache_dir)
            client.notes.append(f"LLM unavailable ({err}); rule-based planner, critic and writer used")
            return client
    return LLMClient(backend, cache_dir, replay_from, cfg.max_budget_usd, cfg.effort)


# ---------------------------------------------------------------------------
# Response parsing (from Projects 1-2)
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response: fenced blocks first, then the
    outermost brace span; several partial objects are merged (later keys win)."""
    candidates = [m.strip() for m in _FENCE.findall(text)]
    if text.count("```") == 1:
        opening = re.search(r"```(?:json)?\s*", text)
        if opening:
            candidates.append(text[opening.end():].strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    parsed, first_error = [], None
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError as err:
            first_error = first_error or f"{err.msg} at line {err.lineno}"
            continue
        if isinstance(obj, dict):
            parsed.append(obj)
    if len(parsed) == 1:
        return parsed[0]
    if parsed:
        merged: dict = {}
        for obj in parsed:
            merged.update(obj)
        return merged
    raise ValueError(f"no parseable JSON object in response"
                     f"{' (' + first_error + ')' if first_error else ''}; starts: {text[:200]!r}")
