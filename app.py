"""
Aplikasi SPK Pemilihan Provider Internet

Arsitektur:
- Flask sebagai backend dan router multi-halaman.
- SQLite sebagai database base.
- MPE untuk pembobotan kriteria.
- SAW untuk normalisasi dan ranking provider.
- KMKK untuk mengelola kompensasi antar kriteria agar hasil tetap realistis.
"""
from __future__ import annotations

from flask import jsonify
from play_scraper import scrape_and_score

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from flask import Flask, flash, g, redirect, render_template, request, url_for


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.path.join(BASE_DIR, "provider_spk.db")

app = Flask(__name__)
app.config["SECRET_KEY"] = "spk-provider-dev-key"

LAST_SENTIMENT_UPDATED: str | None = None
JAKARTA_TZ = ZoneInfo("Asia/Jakarta")
MONTHS_ID = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember",
}


@dataclass(frozen=True)
class Criterion:
    id: int
    code: str
    name: str
    ctype: str
    description: str


@dataclass(frozen=True)
class Provider:
    id: int
    name: str
    price_idr: float
    speed_mbps: float
    stability: float
    service: float
    coverage: float
    sentiment: float


DEFAULT_CRITERIA = [
    ("price_idr", "Harga", "cost", "Semakin rendah biaya bulanan, semakin baik."),
    ("speed_mbps", "Kecepatan", "benefit", "Semakin tinggi kecepatan Mbps, semakin baik."),
    ("stability", "Stabilitas", "benefit", "Konsistensi jaringan, uptime, dan minim gangguan."),
    ("service", "Layanan", "benefit", "Kualitas layanan pelanggan dan respons teknisi."),
    ("coverage", "Jangkauan", "benefit", "Ketersediaan jaringan di area pengguna."),
    ("sentiment", "Sentimen", "benefit", "Hasil analisis sentimen pelanggan menggunakan IndoBERT."),
]

DEFAULT_PROVIDERS = [
    ("IndiHome", 375000, 50, 85, 80, 98, 0),
    ("Biznet", 375000, 100, 90, 85, 75, 0),
    ("MyRepublic", 389000, 100, 88, 82, 70, 0),
    ("CBN", 400000, 100, 92, 88, 65, 0),
    ("First Media", 369000, 75, 82, 78, 80, 0),
    ("Iconnet", 299000, 50, 80, 76, 85, 0),
]

DEFAULT_IMPORTANCE = 5
PROVIDER_FIELDS = {
    "price_idr": "Harga",
    "speed_mbps": "Kecepatan",
    "stability": "Stabilitas",
    "service": "Layanan",
    "coverage": "Jangkauan",
    "sentiment": "Sentimen",
}

DEFAULT_SENTIMENT_BY_PROVIDER = {
    name: sentiment for name, *_values, sentiment in DEFAULT_PROVIDERS
}


def get_db() -> sqlite3.Connection:
    """Membuka koneksi database per request Flask."""
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_: Exception | None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    """Menyiapkan tabel dan data awal tanpa menghapus perubahan pengguna."""
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS criteria (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL CHECK(type IN ('benefit', 'cost')),
                description TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS providers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                price_idr REAL NOT NULL,
                speed_mbps REAL NOT NULL,
                stability REAL NOT NULL,
                service REAL NOT NULL,
                coverage REAL NOT NULL,
                sentiment REAL NOT NULL
            )
            """
        )
        provider_columns = {
            row[1] for row in cur.execute("PRAGMA table_info(providers)").fetchall()
        }
        if "sentiment" not in provider_columns:
            cur.execute("ALTER TABLE providers ADD COLUMN sentiment REAL NOT NULL DEFAULT 0")
            cur.executemany(
                "UPDATE providers SET sentiment = ? WHERE name = ?",
                [
                    (sentiment, name)
                    for name, sentiment in DEFAULT_SENTIMENT_BY_PROVIDER.items()
                ],
            )
        cur.executemany(
            """
            INSERT INTO criteria (code, name, type, description)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                name = excluded.name,
                type = excluded.type,
                description = excluded.description
            """,
            DEFAULT_CRITERIA,
        )
        provider_count = cur.execute("SELECT COUNT(*) FROM providers").fetchone()[0]
        if provider_count == 0:
            cur.executemany(
                """
                INSERT INTO providers (name, price_idr, speed_mbps, stability, service, coverage, sentiment)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                DEFAULT_PROVIDERS,
            )
        conn.commit()


def fetch_criteria() -> list[Criterion]:
    rows = get_db().execute("SELECT * FROM criteria ORDER BY id").fetchall()
    return [
        Criterion(
            id=row["id"],
            code=row["code"],
            name=row["name"],
            ctype=row["type"],
            description=row["description"],
        )
        for row in rows
    ]


def row_to_provider(row: sqlite3.Row) -> Provider:
    return Provider(
        id=row["id"],
        name=row["name"],
        **{field: row[field] for field in PROVIDER_FIELDS},
    )


def fetch_providers() -> list[Provider]:
    rows = get_db().execute("SELECT * FROM providers ORDER BY id").fetchall()
    return [row_to_provider(row) for row in rows]


def fetch_provider(provider_id: int) -> Provider | None:
    row = get_db().execute("SELECT * FROM providers WHERE id = ?", (provider_id,)).fetchone()
    return row_to_provider(row) if row else None


def format_wib_timestamp(value: datetime) -> str:
    return (
        f"{value.day:02d} {MONTHS_ID[value.month]} {value.year} "
        f"{value.hour:02d}:{value.minute:02d}:{value.second:02d} WIB"
    )


def read_provider_form() -> dict[str, float | str]:
    name = request.form.get("name", "").strip()
    if not name:
        raise ValueError("Nama provider wajib diisi.")

    data: dict[str, float | str] = {"name": name}
    for field, label in PROVIDER_FIELDS.items():
        try:
            value = float(request.form.get(field, ""))
        except ValueError as exc:
            raise ValueError(f"{label} harus berupa angka.") from exc
        if value < 0:
            raise ValueError(f"{label} tidak boleh bernilai negatif.")
        if field in {"stability", "service", "coverage", "sentiment"} and value > 100:
            raise ValueError(f"{label} maksimal bernilai 100.")
        data[field] = value
    return data


def read_importance(criteria: list[Criterion]) -> dict[str, int]:
    """Mengambil nilai slider kepentingan dari form dashboard."""
    importance: dict[str, int] = {}
    for criterion in criteria:
        raw_value = request.form.get(criterion.code, DEFAULT_IMPORTANCE)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = DEFAULT_IMPORTANCE
        importance[criterion.code] = max(1, min(value, 5))
    return importance


def calculate_mpe_weights(importance: dict[str, int], exponent: int = 2) -> dict[str, float]:
    """
    MPE menonjolkan perbedaan prioritas dengan pemangkatan.
    Formula: w_j = p_j^k / sum(p_j^k)
    """
    powered = {code: value**exponent for code, value in importance.items()}
    total = sum(powered.values()) or 1
    return {code: score / total for code, score in powered.items()}


def normalize_saw(
    providers: list[Provider], criteria: list[Criterion]
) -> dict[str, dict[str, float]]:
    """
    Normalisasi SAW:
    Benefit = x_ij / max(x_j)
    Cost = min(x_j) / x_ij
    """
    normalized: dict[str, dict[str, float]] = {}
    for criterion in criteria:
        values = [float(getattr(provider, criterion.code)) for provider in providers]
        max_value = max(values) or 1
        min_value = min(values) or 1

        for provider in providers:
            raw = float(getattr(provider, criterion.code))
            normalized.setdefault(provider.name, {})
            if criterion.ctype == "benefit":
                normalized[provider.name][criterion.code] = raw / max_value
            else:
                normalized[provider.name][criterion.code] = min_value / raw if raw else 0
    return normalized


def kmkk_factor(values: dict[str, float], weights: dict[str, float]) -> tuple[float, str]:
    """
    KMKK mengontrol sifat kompensatori SAW.
    Provider tetap boleh unggul karena kekuatan tertentu, tetapi kelemahan ekstrem
    pada kriteria penting akan diberi penalti.
    """
    weighted_gap = sum(weights[code] * (1 - value) for code, value in values.items())
    weakest_code = min(values, key=values.get)
    weakest_value = values[weakest_code]

    factor = 1 - (0.18 * weighted_gap)
    note = "Kompensasi seimbang"

    if weakest_value < 0.55:
        factor -= 0.08
        note = "Ada kriteria lemah, penalti KMKK aktif"
    elif weakest_value < 0.70:
        factor -= 0.04
        note = "Kompensasi dibatasi karena ada kekurangan kecil"

    return max(0.70, min(1.0, factor)), note


def rank_providers(
    providers: list[Provider], criteria: list[Criterion], weights: dict[str, float]
) -> list[dict[str, Any]]:
    normalized = normalize_saw(providers, criteria)
    results = []

    for provider in providers:
        n_values = normalized[provider.name]
        saw_score = sum(weights[code] * n_values[code] for code in weights)
        factor, kmkk_note = kmkk_factor(n_values, weights)
        final_score = saw_score * factor
        results.append(
            {
                "name": provider.name,
                "price_idr": provider.price_idr,
                "speed_mbps": provider.speed_mbps,
                "stability": provider.stability,
                "service": provider.service,
                "coverage": provider.coverage,
                "saw_score": saw_score,
                "kmkk_factor": factor,
                "final_score": final_score,
                "kmkk_note": kmkk_note,
                "normalized": n_values,
            }
        )

    return sorted(results, key=lambda item: item["final_score"], reverse=True)


def filter_providers(providers: list[Provider]) -> list[Provider]:
    """Filter sederhana untuk halaman katalog memakai query string."""
    max_price = request.args.get("max_price", type=float)
    min_speed = request.args.get("min_speed", type=float)
    sort_by = request.args.get("sort_by", "name")

    filtered = providers
    if max_price:
        filtered = [provider for provider in filtered if provider.price_idr <= max_price]
    if min_speed:
        filtered = [provider for provider in filtered if provider.speed_mbps >= min_speed]

    sort_map = {
        "name": lambda provider: provider.name,
        "price": lambda provider: provider.price_idr,
        "speed": lambda provider: -provider.speed_mbps,
        "stability": lambda provider: -provider.stability,
    }
    return sorted(filtered, key=sort_map.get(sort_by, sort_map["name"]))


@app.template_filter("idr")
def format_idr(value: float) -> str:
    return f"Rp{value:,.0f}".replace(",", ".")


@app.route("/")
def home():
    return render_template("home.html", title="Portal SPK Provider")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    criteria = fetch_criteria()
    providers = fetch_providers()
    importance = read_importance(criteria)
    weights = calculate_mpe_weights(importance)
    rankings = rank_providers(providers, criteria, weights)

    chart_payload = {
        "labels": [item["name"] for item in rankings],
        "scores": [round(item["final_score"] * 100, 2) for item in rankings],
        "sawScores": [round(item["saw_score"] * 100, 2) for item in rankings],
        "weights": [round(weights[criterion.code] * 100, 2) for criterion in criteria],
        "criteria": [criterion.name for criterion in criteria],
    }

    return render_template(
        "dashboard.html",
        title="Dashboard Rekomendasi",
        criteria=criteria,
        providers=providers,
        provider_lookup={provider.name: provider for provider in providers},
        importance=importance,
        weights=weights,
        rankings=rankings,
        best=rankings[0],
        chart_payload=json.dumps(chart_payload),
    )


@app.route("/katalog")
def katalog():
    providers = fetch_providers()
    filtered_providers = filter_providers(providers)
    edit_id = request.args.get("edit", type=int)
    edit_provider = fetch_provider(edit_id) if edit_id else None
    return render_template(
        "katalog.html",
        title="Katalog Provider",
        providers=filtered_providers,
        total_providers=len(providers),
        edit_provider=edit_provider,
    )


@app.route("/provider/simpan", methods=["POST"])
def save_provider():
    provider_id = request.form.get("provider_id", type=int)
    try:
        data = read_provider_form()
        db = get_db()
        field_names = list(PROVIDER_FIELDS)
        values = [data["name"], *(data[field] for field in field_names)]
        if provider_id:
            set_clause = ", ".join(f"{field} = ?" for field in ["name", *field_names])
            db.execute(
                f"UPDATE providers SET {set_clause} WHERE id = ?",
                (*values, provider_id),
            )
            flash("Data provider berhasil diperbarui.", "success")
        else:
            columns = ", ".join(["name", *field_names])
            placeholders = ", ".join("?" for _ in values)
            db.execute(
                f"INSERT INTO providers ({columns}) VALUES ({placeholders})",
                values,
            )
            flash("Provider baru berhasil ditambahkan.", "success")
        db.commit()
    except sqlite3.IntegrityError:
        flash("Nama provider sudah digunakan.", "error")
    except ValueError as exc:
        flash(str(exc), "error")
    return redirect(url_for("katalog"))


@app.route("/provider/<int:provider_id>/hapus", methods=["POST"])
def delete_provider(provider_id: int):
    db = get_db()
    provider_count = db.execute("SELECT COUNT(*) FROM providers").fetchone()[0]
    if provider_count <= 1:
        flash("Minimal harus ada satu provider untuk perhitungan ranking.", "error")
        return redirect(url_for("katalog"))
    db.execute("DELETE FROM providers WHERE id = ?", (provider_id,))
    db.commit()
    flash("Data provider berhasil dihapus.", "success")
    return redirect(url_for("katalog"))

@app.route("/api/refresh-sentimen", methods=["POST"])
def api_refresh_sentimen():
    """
    Scrape ulasan terbaru Google Play → inference sentimen → update DB.
    Dipanggil via AJAX dari tombol di halaman /sentimen atau /dashboard.
    """
    global LAST_SENTIMENT_UPDATED
    data = request.get_json(silent=True) or {}
    try:
        count = int(data.get("count", 20))
    except (TypeError, ValueError):
        count = 20
    count = max(1, min(count, 200))
 
    try:
        result = scrape_and_score(count_per_provider=count)
 
        db      = get_db()
        updated = []
 
        for provider_name, score in result["scores"].items():
            rows = db.execute(
                "UPDATE providers SET sentiment = ? WHERE name = ?",
                (score, provider_name),
            ).rowcount
            if rows > 0:
                updated.append(provider_name)
 
        db.commit()

        LAST_SENTIMENT_UPDATED = format_wib_timestamp(datetime.now(JAKARTA_TZ))
        total_reviews = sum(detail.get("total", 0) for detail in result["details"].values())
        reviewed_providers = len(result["details"])
 
        return jsonify({
            "status":   "success",
            "updated":  updated,
            "scores":   result["scores"],
            "details":  result["details"],
            "reviews":  result["reviews"],
            "method":   result["method"],
            "model":    result.get("model"),
            "duration": result["duration"],
            "last_updated": LAST_SENTIMENT_UPDATED,
            "review_count": total_reviews,
            "review_count_label": f"{total_reviews} Review ({count} x {reviewed_providers} provider)",
            "errors":   result["errors"],
        })
 
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
 
 
# ── 3. ROUTE HALAMAN SENTIMEN (opsional) ─────────────────────────────────────
 
@app.route("/sentimen")
def sentimen():
    providers = fetch_providers()
    return render_template(
        "sentimen_playstore.html",
        title="Analisis Sentimen",
        providers=providers,
        last_sentiment_updated=LAST_SENTIMENT_UPDATED,
    )

if __name__ == "__main__":
    init_db()
    app.run(debug=True, use_reloader=False)
