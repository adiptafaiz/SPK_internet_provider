# SPK Pemilihan Provider Internet

Aplikasi Decision Support System berbasis Model-Driven untuk memilih provider internet menggunakan metode hibrid MPE, SAW, dan KMKK.

## Struktur Halaman

- `/` - Portal utama dengan 2 kartu menu.
- `/dashboard` - Dashboard rekomendasi provider.
- `/katalog` - Katalog data provider dengan filter harga, kecepatan, tambah, edit, dan hapus provider.

## Kriteria

- Harga
- Speed
- Stabilitas
- Service
- Coverage

## Cara Menjalankan

```bash
pip install -r requirements.txt
python app.py
```

Buka browser ke:

```text
http://127.0.0.1:5000
```

Database `provider_spk.db` akan dibuat otomatis saat aplikasi dijalankan.

## Susunan Folder Flask

```text
ETS_SPK/
├── app.py
├── requirements.txt
├── README.md
└── templates/
    ├── layout.html
    ├── home.html
    ├── dashboard.html
    └── katalog.html
```
