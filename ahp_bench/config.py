"""Static experiment configuration and .env loading."""
from __future__ import annotations

import json
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent


def resource(name: str) -> Path:
    """Bundled folder or file (problems, skills, models.json): inside the installed package when it was
    installed with pip, else at the repository root of a clone."""
    packaged = PACKAGE_DIR / name
    return packaged if packaged.exists() else PROJECT_ROOT / name


API_URL = "https://openrouter.ai/api/v1/chat/completions"
TEMPERATURE = 0.6
MAX_TOKENS = 8000
TIMEOUT_S = 180
DEFAULT_REASONING = {"effort": "none"}

# key -> OpenRouter id, env var holding the API key, developer.
# Optional "reasoning": request value; None = omit the parameter (gpt-4o).
MODELS = {
    "claude-haiku-4.5": {"id": "anthropic/claude-haiku-4.5", "env": "OPENROUTER_KEY_CLAUDE_HAIKU_4_5", "developer": "Anthropic"},
    "claude-sonnet-4.6": {"id": "anthropic/claude-sonnet-4.6", "env": "OPENROUTER_KEY_CLAUDE_SONNET_4_6", "developer": "Anthropic"},
    "deepseek-v3.2": {"id": "deepseek/deepseek-v3.2", "env": "OPENROUTER_KEY_DEEPSEEK_V3_2", "developer": "DeepSeek"},
    "gemini-3.1-flash-lite": {"id": "google/gemini-3.1-flash-lite", "env": "OPENROUTER_KEY_GEMINI_FLASH", "developer": "Google"},
    "glm-5": {"id": "z-ai/glm-5", "env": "OPENROUTER_KEY_GLM_5", "developer": "Zhipu AI"},
    "gpt-4o": {"id": "openai/gpt-4o", "env": "OPENROUTER_KEY_GPT_4O", "developer": "OpenAI", "reasoning": None},
    "gpt-5.2": {"id": "openai/gpt-5.2", "env": "OPENROUTER_KEY_GPT_5_2", "developer": "OpenAI"},
    "minimax-m2.5": {"id": "minimax/minimax-m2.5", "env": "OPENROUTER_KEY_MINIMAX_M2_5", "developer": "MiniMax", "reasoning": {"effort": "minimal"}},
    "qwen3.5-397b": {"id": "qwen/qwen3.5-397b-a17b", "env": "OPENROUTER_KEY_QWEN3_5_397B", "developer": "Alibaba Cloud"},
}

# Revision-phase judge/orchestrator (python -m ahp_bench.revise --judge). Its API key is
# OPENROUTER_KEY_JUDGE if set, otherwise OPENROUTER_API_KEY, otherwise the first panel-model key that
# is set (OpenRouter keys are not model-bound); see client.api_key_for.
# MODELS and JUDGE are the study's panel; models.json (the default --models-file) holds the same values.
JUDGE = {"key": "grok-4.20", "id": "x-ai/grok-4.20", "developer": "xAI", "env": "OPENROUTER_KEY_JUDGE",
         "temperature": 0.0, "reasoning": None}


# One OpenRouter key for every model (fallback when a model's own variable is not set).
SHARED_KEY_ENV = "OPENROUTER_API_KEY"
# Panel file read by the CLI (--models-file); the default file equals MODELS / JUDGE above.
MODELS_FILE = resource("models.json")


def reasoning_for(model_key: str, models: dict | None = None, default=DEFAULT_REASONING):
    """Reasoning request value for a model; None means the parameter is omitted."""
    value = (models or MODELS)[model_key].get("reasoning", default)
    return dict(value) if value is not None else None


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def builtin_panel() -> dict:
    return {"models": {k: dict(v) for k, v in MODELS.items()}, "judge": dict(JUDGE), "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS, "default_reasoning": dict(DEFAULT_REASONING), "path": None}


def load_panel(path=None) -> dict:
    """Model panel from a JSON file (default models.json; the built-in study panel if that file is absent).

    {"defaults": {"temperature": 0.6, "max_tokens": 8000, "reasoning": {"effort": "none"}},
     "models": {"<key>": {"id": "<openrouter id>", "env": "<KEY VARIABLE>", "developer": "...",
                          "reasoning": null | {...}}},     # "reasoning" absent = default; null = omitted
     "judge": {"key": "...", "id": "...", "env": "...", "developer": "...", "temperature": 0.0, "reasoning": null}}
    "env" is optional; without it only the shared OPENROUTER_API_KEY is used."""
    if path is None:
        if not MODELS_FILE.exists():
            return builtin_panel()
        path = MODELS_FILE
    path = Path(path)
    d = _strip(json.loads(path.read_text(encoding="utf-8-sig")))
    models = d.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"{path}: 'models' must be a non-empty object")
    out_models = {}
    for k, m in models.items():
        if not isinstance(m, dict) or not isinstance(m.get("id"), str) or not m["id"].strip():
            raise ValueError(f"{path}: model {k!r} needs an OpenRouter 'id'")
        m = dict(m)
        m.setdefault("env", SHARED_KEY_ENV)
        m.setdefault("developer", m["id"].split("/")[0])
        out_models[k] = m
    defaults = d.get("defaults") or {}
    judge = dict(d.get("judge") or JUDGE)
    if not isinstance(judge.get("id"), str):
        raise ValueError(f"{path}: judge needs an 'id'")
    judge.setdefault("key", judge["id"].split("/")[-1])
    judge.setdefault("env", "OPENROUTER_KEY_JUDGE")
    judge.setdefault("developer", judge["id"].split("/")[0])
    judge.setdefault("temperature", 0.0)
    judge.setdefault("reasoning", None)
    return {"models": out_models, "judge": judge,
            "temperature": float(defaults.get("temperature", TEMPERATURE)),
            "max_tokens": int(defaults.get("max_tokens", MAX_TOKENS)),
            "default_reasoning": defaults.get("reasoning", DEFAULT_REASONING), "path": path}


def load_env(path=".env") -> list[str]:
    """Parse KEY=VALUE lines into os.environ (existing vars win). Returns loaded names only."""
    path = Path(path)
    if not path.exists():
        return []
    names = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value
            names.append(name)
    return names
