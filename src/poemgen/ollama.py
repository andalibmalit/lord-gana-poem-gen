"""A thin client for Ollama's native HTTP API (https://docs.ollama.com/api).

The tool builds every request body itself and records it, so this client sends
JSON as given and returns JSON as received.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

PROVIDER = "ollama"


class OllamaError(Exception):
    """Ollama is unreachable or answered with an error."""


class Ollama:
    def __init__(self, host: str, timeout_s: float = 3600):
        # Ollama is usually local: never route it through a proxy from the environment, and
        # only set up certificate checks when the host actually uses TLS.
        self.http = httpx.Client(
            base_url=host,
            timeout=httpx.Timeout(timeout_s, connect=10),
            trust_env=False,
            verify=host.startswith("https://"),
        )

    def close(self) -> None:
        self.http.close()

    def _request(self, method: str, path: str, **kw) -> dict:
        try:
            r = self.http.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise OllamaError(f"{method} {path}: {e}") from e
        if r.is_error:
            raise OllamaError(f"{method} {path}: HTTP {r.status_code}: {r.text.strip()[:500]}")
        return r.json()

    def version(self) -> str:
        return self._request("GET", "/api/version")["version"]

    def tags(self) -> dict[str, str]:
        """Local models: name -> digest."""
        return {m["name"]: m["digest"] for m in self._request("GET", "/api/tags")["models"]}

    def show(self, model: str) -> dict:
        return self._request("POST", "/api/show", json={"model": model})

    def loaded(self, model: str) -> dict | None:
        """The /api/ps entry for a loaded model (size, size_vram, context_length...)."""
        for m in self._request("GET", "/api/ps").get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return m
        return None

    def chat(self, body: dict) -> dict:
        return self._request("POST", "/api/chat", json=body)

    def render(self, body: dict) -> str | None:
        """The exact prompt text the model would receive, without generating anything.

        Uses Ollama's undocumented `_debug_render_only` flag, so None means "unknown".
        """
        try:
            response = self.chat({**body, "_debug_render_only": True})
        except OllamaError:
            return None
        return response.get("_debug_info", {}).get("rendered_template")

    def unload(self, model: str) -> None:
        self._request("POST", "/api/chat", json={"model": model, "messages": [], "keep_alive": 0})

    def pull(self, model: str, progress: Callable[[dict], None] | None = None) -> None:
        try:
            with self.http.stream(
                "POST", "/api/pull", json={"model": model, "stream": True}, timeout=None
            ) as r:
                if r.is_error:
                    r.read()
                    raise OllamaError(f"pull {model}: HTTP {r.status_code}: {r.text[:500]}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if "error" in event:
                        raise OllamaError(f"pull {model}: {event['error']}")
                    if progress:
                        progress(event)
        except httpx.HTTPError as e:
            raise OllamaError(f"pull {model}: {e}") from e
