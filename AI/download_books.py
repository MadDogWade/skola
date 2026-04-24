"""
Laddar ner 5 klassiker från Project Gutenberg till ./books/
Körs en gång innan rag_chatbot.py.
"""

import os
import requests

BOOKS_DIR = "books"

BOOKS = {
    "alice_in_wonderland.txt":      "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "pride_and_prejudice.txt":      "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "frankenstein.txt":             "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "sherlock_holmes.txt":          "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "dracula.txt":                  "https://www.gutenberg.org/cache/epub/345/pg345.txt",
}


def download_all():
    os.makedirs(BOOKS_DIR, exist_ok=True)
    for filename, url in BOOKS.items():
        path = os.path.join(BOOKS_DIR, filename)
        if os.path.exists(path):
            print(f"[SKIP] {filename} finns redan")
            continue
        print(f"[HÄMTAR] {filename} ...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        size_kb = len(resp.text) // 1024
        print(f"   OK ({size_kb} KB)")
    print(f"\n[KLAR] Alla böcker finns i ./{BOOKS_DIR}/")


if __name__ == "__main__":
    download_all()
