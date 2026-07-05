# debug_ask.py — proje kökünde, bir kerelik test scripti
import json
import requests

API_KEY = "12345"
BASE_URL = "http://127.0.0.1:8000"

questions = [
    "kıdem tazminatı nasıl hesaplanır?",
    "madde 63 ne diyor",
    "boşanma davası nasıl açılır",
]

headers = {"Authorization": f"Bearer {API_KEY}"}
results = []

for q in questions:
    r = requests.post(f"{BASE_URL}/ask", json={"question": q}, headers=headers)
    results.append({
        "question": q,
        "status_code": r.status_code,
        "response": r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text,
    })

with open("debug_ask_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Yazıldı: debug_ask_results.json — VS Code'da aç")