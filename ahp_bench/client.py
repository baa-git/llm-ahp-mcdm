"""Minimal OpenRouter chat-completions client (urllib only) with parameter fallback and retries."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from .config import API_URL, MODELS, SHARED_KEY_ENV, TIMEOUT_S

REASONING_LADDER = [{"effort": "none"}, {"effort": "minimal"}, {"effort": "low"}]  # then: omit
RETRY_DELAYS = (2, 5, 10)
UNSUPPORTED_STATUS = (400, 404, 422)
UNSUPPORTED_HINTS = ("mandatory", "cannot be disabled", "unsupported", "not supported", "does not support", "not allowed", "invalid",
                     "unknown", "unrecognized", "not available", "no endpoints")
_sleep = time.sleep  # patched in tests


def _decode(raw: bytes) -> dict:
    text = raw.decode("utf-8", "replace")
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {"raw": obj}
    except json.JSONDecodeError:
        return {"raw": text[:2000]}


def urllib_transport(url: str, headers: dict, body: dict):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return r.status, _decode(r.read())
    except urllib.error.HTTPError as e:
        return e.code, _decode(e.read())


def _error_text(resp: dict) -> str:
    return json.dumps(resp.get("error", resp), ensure_ascii=False).lower()


def _offending_param(status, resp: dict, params: dict):
    """Name of the request parameter an HTTP 400/404/422 complains about, if any."""
    if status not in UNSUPPORTED_STATUS:
        return None
    text = _error_text(resp)
    if not any(h in text for h in UNSUPPORTED_HINTS):
        return None
    if "reasoning" in params and any(w in text for w in ("reasoning", "effort", "thinking")):
        return "reasoning"
    if "temperature" in params and "temperature" in text:
        return "temperature"
    return None


def _step_down(params: dict, name: str) -> dict:
    p = dict(params)
    if name == "temperature":
        p.pop("temperature")
        return p
    cur = p["reasoning"]
    idx = REASONING_LADDER.index(cur) if cur in REASONING_LADDER else len(REASONING_LADDER) - 1
    if idx + 1 < len(REASONING_LADDER):
        p["reasoning"] = dict(REASONING_LADDER[idx + 1])
    else:
        p.pop("reasoning")
    return p


def _scrub(obj, secret: str):
    if not secret:
        return obj
    return json.loads(json.dumps(obj, ensure_ascii=False).replace(secret, "***"))


def api_key_for(model: dict, panel: dict | None = None):
    """API key for a model dict: its own env var, then the shared OPENROUTER_API_KEY; for a model outside
    the panel (the judge) the first panel-model key that is set is the last fallback. `panel` is the
    model dict of the panel (default: the study panel). Returns None when nothing is set. The value is
    never logged."""
    key = os.environ.get(model.get("env") or SHARED_KEY_ENV)
    if key:
        return key
    if os.environ.get(SHARED_KEY_ENV):
        return os.environ[SHARED_KEY_ENV]
    panel = MODELS if panel is None else panel
    if model.get("key") and model["key"] not in panel:
        for m in panel.values():
            if m.get("env") and os.environ.get(m["env"]):
                return os.environ[m["env"]]
    return None


def call_model(model_key, messages, *, temperature, reasoning, max_tokens, transport=None, model=None,
               panel=None) -> dict:
    """`model` (optional) is a model dict with at least "id" and "env" (e.g. config.JUDGE); by default
    the panel entry MODELS[model_key] is used."""
    model = model if model is not None else MODELS[model_key]
    key = api_key_for(model, panel)
    if not key:
        raise RuntimeError(f"environment variable {model.get('env') or SHARED_KEY_ENV} (or {SHARED_KEY_ENV}) "
                           "is not set")
    transport = transport or urllib_transport
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    requested = {"temperature": temperature, "reasoning": reasoning, "max_tokens": max_tokens}
    params = {k: v for k, v in (("temperature", temperature), ("reasoning", reasoning)) if v is not None}
    attempts, transient_used, resp, status, error = [], 0, {}, None, None
    started_at = datetime.now(timezone.utc).isoformat()
    t_start = time.monotonic()
    while True:
        body = {"model": model["id"], "messages": messages, "max_tokens": max_tokens, **params,
                "usage": {"include": True}}
        t0 = time.monotonic()
        try:
            status, resp = transport(API_URL, headers, body)
        except OSError as e:  # URLError, timeouts, connection resets
            status, resp = None, {"error": {"message": f"{type(e).__name__}: {e}"}}
        ok = status == 200 and bool(resp.get("choices"))
        attempts.append({"status": status, "params": dict(params), "seconds": round(time.monotonic() - t0, 3),
                         "error": None if ok else _error_text(resp)[:500]})
        if ok:
            error = None
            break
        error = f"HTTP {status}: {_error_text(resp)[:300]}"
        name = _offending_param(status, resp, params)
        if name:
            params = _step_down(params, name)
            continue
        transient = status is None or status == 429 or status >= 500 or status == 200
        if transient and transient_used < len(RETRY_DELAYS):
            _sleep(RETRY_DELAYS[transient_used])
            transient_used += 1
            continue
        break

    choice = (resp.get("choices") or [{}])[0] if error is None else {}
    message = choice.get("message") or {}
    usage = resp.get("usage") or {}
    record = {
        "model_key": model_key,
        "model_id": model["id"],
        "ok": error is None,
        "error": error,
        "content": message.get("content"),
        "model_reasoning_text": message.get("reasoning"),
        "finish_reason": choice.get("finish_reason"),
        "native_finish_reason": choice.get("native_finish_reason"),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        "cost": usage.get("cost"),
        "provider": resp.get("provider"),
        "generation_id": resp.get("id"),
        "http_attempts": attempts,
        "requested_params": requested,
        "applied_params": {"temperature": params.get("temperature"), "reasoning": params.get("reasoning"),
                           "max_tokens": max_tokens,
                           "omitted": [k for k in ("temperature", "reasoning") if k not in params]},
        "latency_s": round(time.monotonic() - t_start, 3),
        "timestamp_utc": started_at,
    }
    return _scrub(record, key)
