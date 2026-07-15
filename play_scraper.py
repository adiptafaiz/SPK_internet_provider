"""
play_scraper.py
---------------
Scraping ulasan Google Play Store per provider internet,
lalu hitung skor sentimen dari ulasan terbaru.

Dependencies:
    pip install google-play-scraper transformers torch

Cara pakai:
    from play_scraper import scrape_and_score
    result = scrape_and_score(count_per_provider=100)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# APP ID GOOGLE PLAY
# Verifikasi manual dari: play.google.com/store/apps/details?id=...

PROVIDER_APP_IDS = {
    "IndiHome":    "com.telkomsel.sobi",
    "MyRepublic":  "id.net.myrepublic",
    "Biznet":      "co.id.biznet.android",
    "CBN":         "com.cbn.dicbn",
    "First Media": "com.firstmedia.idconnect",
    "Iconnet":     "id.co.iconpln.icrm.customer",
}

_POS = {
    "bagus","baik","cepat","memuaskan","puas","mantap","kencang","lancar",
    "stabil","murah","terjangkau","recommended","rekomen","oke","keren",
    "terbaik","top","sukses","berhasil","ramah","responsif","solutif",
    "membantu","nyaman","senang","gercep","ngebut","hebat","profesional",
    "luar biasa","paling","suka","senang","worth","worth it",
}

_NEG = {
    "lambat","lemot","gangguan","down","putus","buruk","jelek","mahal",
    "kecewa","mengecewakan","tidak","ga","gak","kagak","susah","ribet",
    "lama","lelet","payah","parah","rusak","boros","tipu","bohong","cabut",
    "komplain","error","gagal","disconnect","mati","trouble","masalah",
    "problem","zonk","nyesel","kacau","jelek","benci","ampas","sampah",
    "refund","minta uang kembali","penipuan",
}

def _lexicon_label(text: str) -> str:
    tokens = set(text.lower().split())
    pos = len(tokens & _POS)
    neg = len(tokens & _NEG)
    if pos > neg:   return "positif"
    if neg > pos:   return "negatif"
    return "netral"


def _star_to_label(score: int) -> str:
    """
    Konversi bintang Play Store → label sentimen.
    Strategi hybrid: bintang sebagai sinyal utama,
    karena lebih reliable dari teks pendek.
    """
    if score >= 4:   return "positif"
    if score == 3:   return "netral"
    return "negatif"


def _combined_label(text: str, star: int) -> str:
    """
    Gabungkan sinyal bintang + lexicon.
    Bintang diberi bobot lebih besar (2:1).
    """
    star_label   = _star_to_label(star)
    text_label   = _lexicon_label(text)
    weight = {"positif": 1, "netral": 0, "negatif": -1}
    score  = weight[star_label] * 2 + weight[text_label]
    if score > 0:  return "positif"
    if score < 0:  return "negatif"
    return "netral"


# MODEL INDOBERT

import os

DEFAULT_MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"
MODEL_NAME: str = (
    os.environ.get("SENTIMENT_MODEL_PATH")
    or os.environ.get("SENTIMENT_MODEL_NAME")
    or DEFAULT_MODEL_NAME
)
MODEL_LABEL_MAPS = {
    DEFAULT_MODEL_NAME: {
        "label_0": "positif",
        "label_1": "netral",
        "label_2": "negatif",
    },
}

_model_cache: dict = {}

def _load_model():
    if "pipeline" in _model_cache:
        return _model_cache.get("pipeline")
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        clf   = pipeline("text-classification", model=model, tokenizer=tok,
                         truncation=True, max_length=128, device=-1)
        _model_cache["pipeline"] = clf
        _model_cache["type"]     = "indobert"
        logger.info(f"[PlayScraper] IndoBERT loaded: {MODEL_NAME}")
        return clf
    except Exception as e:
        _model_cache["pipeline"] = None
        _model_cache["type"]     = "star+lexicon"
        logger.warning(f"[PlayScraper] Model load failed: {e}. Using star+lexicon.")
        return None

def _predict(text: str, star: int, model) -> str:
    if model is None:
        return _combined_label(text, star)
    try:
        pred      = model([text])[0]
        raw_label = pred["label"].lower()
        model_label_map = MODEL_LABEL_MAPS.get(MODEL_NAME, {})
        if raw_label in model_label_map:
            return model_label_map[raw_label]
        if "positif" in raw_label or raw_label in ("positive", "pos"):
            return "positif"
        if "negatif" in raw_label or raw_label in ("negative", "neg"):
            return "negatif"
        return "netral"
    except Exception:
        return _combined_label(text, star)


def scrape_and_score(
    count_per_provider: int = 100,
    providers: Optional[list[str]] = None,
) -> dict:
    """
    Scrape ulasan terbaru Google Play per provider, lalu hitung skor sentimen.

    Args:
        count_per_provider: jumlah ulasan terbaru yang diambil per provider (max ~200)
        providers: list nama provider yang ingin di-scrape (None = semua)

    Returns:
        {
            "scores":   {"IndiHome": 74.5, ...},
            "details":  {"IndiHome": {"positif":60,"netral":25,"negatif":15,"total":100}, ...},
            "reviews":  {"IndiHome": [{"text":"...", "star":5, "label":"positif"}, ...]},
            "method":   "indobert" | "star+lexicon",
            "model":    "nama model IndoBERT yang dipakai",
            "duration": 8.2,
            "errors":   {"CBN": "App not found"},
        }
    """
    from google_play_scraper import reviews as gp_reviews, Sort

    t_start  = time.time()
    model    = _load_model()
    method   = _model_cache.get("type", "star+lexicon")

    target = providers or list(PROVIDER_APP_IDS.keys())

    scores:  dict[str, float] = {}
    details: dict[str, dict]  = {}
    reviews_out: dict[str, list] = {}
    errors:  dict[str, str]   = {}

    for provider_name in target:
        app_id = PROVIDER_APP_IDS.get(provider_name)
        if not app_id:
            errors[provider_name] = "App ID tidak diketahui"
            continue

        try:
            raw_reviews, _ = gp_reviews(
                app_id,
                lang="id",
                country="id",
                sort=Sort.NEWEST,
                count=count_per_provider,
            )

            # Debug output: tampilkan ringkasan hasil scrape di terminal
            print("=" * 60)
            print("Provider :", provider_name)
            print("App ID   :", app_id)
            print("Review   :", len(raw_reviews))
            if len(raw_reviews) > 0:
                print("Contoh   :", (raw_reviews[0].get("content") or "")[:100])
            print("=" * 60)

            # Jika review kosong, catat sebagai error dan lanjut ke provider berikutnya
            if len(raw_reviews) == 0:
                logger.warning(f"[PlayScraper] {provider_name} tidak memiliki review")
                errors[provider_name] = "Review kosong"
                continue
        except Exception as e:
            logger.warning(f"[PlayScraper] Gagal scrape {provider_name}: {e}")
            errors[provider_name] = str(e)
            continue

        n_pos = n_neu = n_neg = 0
        star_counts = {star: 0 for star in range(1, 6)}
        review_list = []

        for r in raw_reviews:
            text  = r.get("content") or ""
            star  = int(r.get("score", 3))
            label = _predict(text, star, model)
            star_counts[star] = star_counts.get(star, 0) + 1

            # Debug distribusi bintang dan hasil klasifikasi.
            print(f"{star} -> {label} -> {text[:100]}")

            if label == "positif": n_pos += 1
            elif label == "negatif": n_neg += 1
            else: n_neu += 1

            review_list.append({
                "text":  text[:200],   # potong biar JSON tidak terlalu besar
                "star":  star,
                "label": label,
                "date":  str(r.get("at", "")),
            })

        total = n_pos + n_neu + n_neg or 1
        # Skor 0–100
        score = round(((n_pos * 1.0 + n_neu * 0.5) / total) * 100, 2)

        scores[provider_name]      = score
        details[provider_name]     = {
            "positif": n_pos, "netral": n_neu,
            "negatif": n_neg, "total": total,
            "stars": star_counts,
        }
        reviews_out[provider_name] = review_list

        logger.info(f"[PlayScraper] {provider_name}: {total} ulasan → skor {score}")

    return {
        "scores":   scores,
        "details":  details,
        "reviews":  reviews_out,
        "method":   method,
        "model":    MODEL_NAME if method == "indobert" else None,
        "duration": round(time.time() - t_start, 2),
        "errors":   errors,
    }
