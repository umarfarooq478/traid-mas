"""Minimal Gemini REST client (stdlib only) with disk cache, retries and usage logging.

Every call is cached by a SHA-256 of (model, system, prompt, generation config), so
re-running an experiment never re-spends quota and results are exactly reproducible
from the cache. Token usage and wall-clock latency are recorded per call.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:  # use the OS certificate store (fixes SSL errors behind TLS-inspecting AV/proxies)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache" / "llm"
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


def _load_key(name: str = "GEMINI_API_KEY") -> str:
    key = os.environ.get(name)
    if key:
        return key.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(name):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"{name} not found in environment or .env")


@dataclass
class LLMResult:
    text: str
    in_tokens: int
    out_tokens: int
    latency_s: float
    cached: bool


class Gemini:
    def __init__(self, model: str, max_tokens: int = 1024, temperature: float = 0.0,
                 min_interval_s: float = 0.0, thinking_budget: int | None = 0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.min_interval_s = min_interval_s
        self.thinking_budget = thinking_budget
        self._last = 0.0
        self._key = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cfg(self) -> dict:
        cfg = {"temperature": self.temperature, "maxOutputTokens": self.max_tokens}
        if self.thinking_budget is not None:
            cfg["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        return cfg

    def __call__(self, prompt: str, system: str = "") -> LLMResult:
        cfg = self._cfg()
        h = hashlib.sha256(json.dumps([self.model, system, prompt, cfg], sort_keys=True).encode()).hexdigest()
        path = CACHE_DIR / f"{h}.json"
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return LLMResult(d["text"], d["in_tokens"], d["out_tokens"], d["latency_s"], True)

        if self._key is None:
            self._key = _load_key()
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": cfg}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        data = json.dumps(body).encode()
        url = API.format(model=self.model, key=self._key)

        delay = 5.0
        for attempt in range(8):
            wait = self.min_interval_s - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            t0 = time.time()
            self._last = t0
            try:
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    resp = json.loads(r.read().decode())
                lat = time.time() - t0
                cands = resp.get("candidates") or []
                parts = (cands[0].get("content", {}).get("parts") if cands else None) or []
                text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
                um = resp.get("usageMetadata", {})
                res = {"text": text, "in_tokens": um.get("promptTokenCount", 0),
                       "out_tokens": um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0),
                       "latency_s": lat, "finish": cands[0].get("finishReason") if cands else "NONE"}
                path.write_text(json.dumps(res), encoding="utf-8")
                return LLMResult(text, res["in_tokens"], res["out_tokens"], lat, False)
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="ignore")[:300]
                if e.code in (429, 500, 502, 503, 504):
                    if "PerDay" in msg or "per day" in msg.lower():
                        raise RuntimeError(f"Daily quota exhausted: {msg}")
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {msg}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                time.sleep(delay)
                delay = min(delay * 2, 120)
        raise RuntimeError("LLM call failed after retries")


PROVIDERS = {  # OpenAI-compatible chat-completions endpoints
    "mistral": ("https://api.mistral.ai/v1/chat/completions", "MISTRAL_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
}


class OpenAICompat(Gemini):
    """Same caching/retry/usage contract as Gemini, for OpenAI-compatible APIs (Mistral, Groq, OpenRouter)."""

    def __init__(self, model: str, provider: str, **kw):
        kw.pop("thinking_budget", None)
        super().__init__(model, thinking_budget=None, **kw)
        self.url, self.key_name = PROVIDERS[provider]

    def __call__(self, prompt: str, system: str = "") -> LLMResult:
        cfg = {"temperature": self.temperature, "max_tokens": self.max_tokens}
        h = hashlib.sha256(json.dumps([self.model, system, prompt, cfg], sort_keys=True).encode()).hexdigest()
        path = CACHE_DIR / f"{h}.json"
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return LLMResult(d["text"], d["in_tokens"], d["out_tokens"], d["latency_s"], True)
        if self._key is None:
            self._key = _load_key(self.key_name)
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        data = json.dumps({"model": self.model, "messages": msgs, **cfg}).encode()
        hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"}
        delay = 3.0
        for attempt in range(7):
            wait = self.min_interval_s - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            t0 = time.time(); self._last = t0
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url, data=data, headers=hdr), timeout=600) as r:
                    resp = json.loads(r.read().decode())
                lat = time.time() - t0
                ch = resp["choices"][0]
                text = ch["message"].get("content") or ""
                if isinstance(text, list):
                    text = "".join(c.get("text", "") for c in text if isinstance(c, dict))
                um = resp.get("usage", {})
                res = {"text": text.strip(), "in_tokens": um.get("prompt_tokens", 0), "out_tokens": um.get("completion_tokens", 0),
                       "latency_s": lat, "finish": ch.get("finish_reason")}
                path.write_text(json.dumps(res), encoding="utf-8")
                return LLMResult(res["text"], res["in_tokens"], res["out_tokens"], lat, False)
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="ignore")[:300]
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(delay); delay = min(delay * 2, 90); continue
                raise RuntimeError(f"HTTP {e.code}: {msg}")
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(delay); delay = min(delay * 2, 90)
        raise RuntimeError("LLM call failed after retries")


def make_llm(model: str, **kw):
    """'mistral-medium-latest' -> Mistral; 'groq:llama-3.3-70b-versatile'; 'openrouter:<id>'; else Gemini."""
    if ":" in model and model.split(":", 1)[0] in PROVIDERS:
        prov, name = model.split(":", 1)
        return OpenAICompat(name, prov, **kw)
    if any(model.startswith(p) for p in ("mistral", "magistral", "ministral", "open-mistral", "codestral")):
        return OpenAICompat(model, "mistral", **kw)
    return Gemini(model, **kw)
