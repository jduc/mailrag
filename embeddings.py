"""Text embeddings with Ollama, cached on disk by hash of (model, text)."""

import hashlib

import numpy as np
import ollama

import config


def embed(texts: list[str], batch_size: int = config.EMBED_BATCH_SIZE, progress: bool = False) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(ollama.embed(model=config.EMBED_MODEL, input=batch).embeddings)
        if progress:
            print(f"\rembedded {start + len(batch)}/{len(texts)}", end="", flush=True)
    if progress:
        print()
    return np.array(vectors, dtype=np.float32)


def text_key(text: str) -> str:
    return hashlib.sha1(f"{config.EMBED_MODEL}\x00{text}".encode()).hexdigest()


def load_cache() -> dict[str, np.ndarray]:
    if not config.EMBEDDINGS_PATH.exists():
        return {}
    data = np.load(config.EMBEDDINGS_PATH)
    return dict(zip(data["keys"].tolist(), data["vectors"]))


def save_cache(cache: dict[str, np.ndarray]) -> None:
    config.EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(config.EMBEDDINGS_PATH, keys=np.array(list(cache)), vectors=np.stack(list(cache.values())))


def embed_cached(texts: list[str], progress: bool = False) -> tuple[np.ndarray, int]:
    """Embed only the texts missing from the cache. Returns (vectors, number newly embedded)."""
    cache = load_cache()
    missing = [t for t in dict.fromkeys(texts) if text_key(t) not in cache]
    if missing:
        cache.update(zip(map(text_key, missing), embed(missing, progress=progress)))
        save_cache(cache)
    return np.stack([cache[text_key(t)] for t in texts]), len(missing)
