import hashlib
import json
import os
import re

import numpy as np

from PIL import Image, ImageFilter, ImageOps


def normalize_text_for_cache(text):
    return ' '.join((text or '').split()).strip().lower()


def capture_signature(img, text=''):
    normalized = ' '.join((text or '').split())
    if normalized:
        return f"text:{hashlib.sha1(normalized.encode('utf-8')).hexdigest()}"
    small = img.resize((64, 64)).convert('L')
    return f"image:{hashlib.sha1(small.tobytes()).hexdigest()}"


def history_path(base_dir):
    return os.path.join(base_dir, 'ai_solver_history.json')


def settings_path(base_dir):
    return os.path.join(base_dir, 'ai_solver_settings.json')


def load_history_cache(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save_history_cache(path, history):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_json_dict(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save_json_dict(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def preprocess_for_ocr(img):
    base = img.convert('L')
    large = ImageOps.autocontrast(base).resize(
        (max(1, base.width * 2), max(1, base.height * 2)),
        Image.Resampling.LANCZOS,
    )
    sharp = large.filter(ImageFilter.SHARPEN)
    binary = sharp.point(lambda p: 255 if p > 170 else 0)
    return [
        ('base', base),
        ('autocontrast', ImageOps.autocontrast(base)),
        ('sharp_large', sharp),
        ('binary', binary),
    ]


def score_ocr_text(text):
    cleaned = normalize_text_for_cache(text)
    if not cleaned:
        return 0.0
    useful = sum(ch.isalnum() for ch in cleaned)
    return len(cleaned) + useful * 0.5


def chunk_text(text, max_chars=1200, min_chars=300):
    cleaned = ' '.join((text or '').split())
    if not cleaned:
        return []

    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        if end < len(cleaned):
            split = cleaned.rfind(' ', start + min_chars, end)
            if split > start:
                end = split
        part = cleaned[start:end].strip()
        if part:
            chunks.append(part)
        start = end
    return chunks


def embed_text(text, dims=256):
    tokens = re.findall(r'\w+', normalize_text_for_cache(text))
    vec = np.zeros(dims, dtype=np.float32)
    if not tokens:
        return vec

    for token in tokens:
        digest = hashlib.sha1(token.encode('utf-8')).digest()
        idx = int.from_bytes(digest[:4], 'big') % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def cosine_similarity(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return 0.0
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)
