"""Look up a model build in its registry without pulling the weights.

Works for Ollama library tags (`gemma4:e4b-it-q8_0`, `user/model:tag`) and Hugging Face
GGUF tags (`hf.co/<user>/<repo>:<quant>`). The manifest lists every layer with its size
and digest; only the small template and params layers are downloaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"
MODEL_LAYER = "application/vnd.ollama.image.model"
TEMPLATE_LAYER = "application/vnd.ollama.image.template"
PARAMS_LAYER = "application/vnd.ollama.image.params"
SMALL_LAYER_LIMIT = 1_000_000  # bytes; never download anything bigger than this


@dataclass
class Build:
    tag: str
    manifest_url: str
    total_bytes: int
    weights_bytes: int
    weights_sha256: str | None
    layers: list[tuple[str, int]]  # (media type, size)
    template: str | None
    params: dict | None

    @property
    def stop(self) -> list[str]:
        stop = (self.params or {}).get("stop", [])
        return [stop] if isinstance(stop, str) else list(stop)


def registry_base(tag: str) -> tuple[str, str]:
    """Return (repository base URL, reference) for an Ollama-style tag."""
    path, ref = tag, "latest"
    last = tag.rsplit("/", 1)[-1]
    if ":" in last:
        path, ref = tag.rsplit(":", 1)
    for prefix in ("hf.co/", "huggingface.co/"):
        if path.startswith(prefix):
            return f"https://huggingface.co/v2/{path.removeprefix(prefix)}", ref
    if "/" not in path:
        path = f"library/{path}"
    return f"https://registry.ollama.ai/v2/{path}", ref


def fetch_build(tag: str, client: httpx.Client | None = None) -> Build:
    """Fetch a tag's manifest plus its template and params. Raises LookupError if missing."""
    own = client is None
    client = client or httpx.Client(timeout=30, follow_redirects=True)
    try:
        base, ref = registry_base(tag)
        url = f"{base}/manifests/{ref}"
        r = client.get(url, headers={"Accept": MANIFEST_ACCEPT})
        if r.status_code in (401, 404):
            raise LookupError(f"{tag}: not found in the registry ({r.status_code}).")
        r.raise_for_status()
        manifest = r.json()
        layers = manifest.get("layers", [])
        weights = [layer for layer in layers if layer.get("mediaType") == MODEL_LAYER]

        def small_blob(media_type: str) -> str | None:
            for layer in layers:
                if layer.get("mediaType") == media_type and layer["size"] < SMALL_LAYER_LIMIT:
                    blob = client.get(f"{base}/blobs/{layer['digest']}")
                    blob.raise_for_status()
                    return blob.content.decode("utf-8")
            return None

        params_text = small_blob(PARAMS_LAYER)
        return Build(
            tag=tag,
            manifest_url=url,
            total_bytes=sum(layer.get("size", 0) for layer in layers)
            + manifest.get("config", {}).get("size", 0),
            weights_bytes=sum(layer["size"] for layer in weights),
            weights_sha256=weights[0]["digest"] if len(weights) == 1 else None,
            layers=[(layer.get("mediaType", "?"), layer.get("size", 0)) for layer in layers],
            template=small_blob(TEMPLATE_LAYER),
            params=json.loads(params_text) if params_text else None,
        )
    finally:
        if own:
            client.close()
