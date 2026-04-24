"""
Evalueringssystem för RAG-chattboten
=====================================
Mäter två saker för varje testfråga:

  1. RETRIEVAL ACCURACY — hämtade sökningen rätt bok i top-K?
     (Räknas som korrekt om förväntad boktitel finns bland de K hämtade chunks.)

  2. ANSWER ACCURACY — innehåller svaret rätt nyckelord?
     (Räknas som korrekt om minst ett av de förväntade nyckelorden finns
     i det genererade svaret — case-insensitive.)

Dessutom mäts "abstention correctness" — för frågor vars facit är `None`
(information som INTE finns i böckerna) ska modellen vägra svara.

Kör:
    python evaluate.py

Kräver att chroma_db/ redan är byggd (kör rag_chatbot.py en gång först).
"""

import time
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import chromadb

from rag_chatbot import (
    LLM_PATH, EMBED_MODEL_DIR, CHROMA_DIR, COLLECTION_NAME,
    TOP_K, N_CTX, N_GPU_LAYERS,
    retrieve, build_prompt, generate_answer,
)


# ─────────────────────────────────────────────
# TESTSET
# ─────────────────────────────────────────────
# Varje testfall: (fråga, förväntad bok, lista med godkända nyckelord i svar)
# expected_book = None   → frågan saknar svar i böckerna, modellen ska vägra
# expected_keywords = [] → endast retrieval räknas, inte svaret
TESTS = [
    # ── Enkla faktafrågor ──
    ("Who does Alice follow down the rabbit hole?",
     "Alice's Adventures in Wonderland",
     ["white rabbit", "rabbit"]),

    ("What is the name of Alice's cat?",
     "Alice's Adventures in Wonderland",
     ["dinah"]),

    ("What is Mr. Darcy's first name?",
     "Pride and Prejudice",
     ["fitzwilliam"]),

    ("Who is Elizabeth Bennet's closest sister?",
     "Pride and Prejudice",
     ["jane"]),

    ("Where does Jonathan Harker travel at the start of Dracula?",
     "Dracula",
     ["transylvania", "carpathian", "castle"]),

    ("Who is Sherlock Holmes' friend and chronicler?",
     "The Adventures of Sherlock Holmes",
     ["watson"]),

    ("On what ship does Victor Frankenstein meet Walton?",
     "Frankenstein",
     ["walton", "arctic", "ice", "ship"]),

    ("Who created the creature in Frankenstein?",
     "Frankenstein",
     ["victor", "frankenstein"]),

    # ── Svårare — kräver bra retrieval/syntes ──
    ("What happens to Lucy Westenra in Dracula?",
     "Dracula",
     ["die", "dead", "vampire", "undead", "bitten", "attack"]),

    ("What is the Red-Headed League?",
     "The Adventures of Sherlock Holmes",
     ["red-headed", "red headed", "wilson", "clay", "bank"]),

    ("Who does Elizabeth Bennet eventually marry?",
     "Pride and Prejudice",
     ["darcy"]),

    ("What does the creature demand from Victor Frankenstein?",
     "Frankenstein",
     ["companion", "mate", "female", "wife", "bride"]),

    # ── Hallucinationstest — modellen SKA vägra ──
    ("Who wins the presidential election in Pride and Prejudice?",
     None,
     ["cannot find", "not contain", "no mention", "does not", "no information"]),

    ("What kind of car does Dracula drive?",
     None,
     ["cannot find", "not contain", "no mention", "does not", "no car"]),

    ("What is Sherlock Holmes' favorite smartphone app?",
     None,
     ["cannot find", "not contain", "no mention", "does not", "no information"]),
]


# ─────────────────────────────────────────────
# Utvärdering
# ─────────────────────────────────────────────
def evaluate_case(query, expected_book, expected_keywords,
                  embedder, collection, llm):
    retrieved = retrieve(query, embedder, collection, top_k=TOP_K)
    retrieved_books = [meta["title"] for _, meta in retrieved]

    # Retrieval accuracy
    if expected_book is None:
        # För frågor utan svar i korpus mäter vi inte retrieval —
        # modellen ska vägra oavsett vad som hämtas.
        retrieval_ok = None
    else:
        retrieval_ok = expected_book in retrieved_books

    # Generera svar
    prompt = build_prompt(query, retrieved)
    answer = generate_answer(llm, prompt)
    answer_lower = answer.lower()

    # Answer accuracy — minst ett förväntat nyckelord måste finnas
    if expected_keywords:
        answer_ok = any(kw.lower() in answer_lower for kw in expected_keywords)
    else:
        answer_ok = None

    return {
        "query": query,
        "expected_book": expected_book,
        "retrieved_books": retrieved_books,
        "retrieval_ok": retrieval_ok,
        "answer": answer,
        "answer_ok": answer_ok,
    }


def print_result(i, r):
    print(f"\n─── Fråga {i}: {r['query']}")
    if r["expected_book"] is not None:
        tag = "✓" if r["retrieval_ok"] else "✗"
        print(f"   Retrieval {tag}  förväntat: {r['expected_book']}")
        print(f"                hämtade: {r['retrieved_books'][:3]}...")
    else:
        print(f"   Retrieval —  (hallucinationstest, ingen förväntad bok)")

    tag = "✓" if r["answer_ok"] else "✗"
    short = r["answer"][:140].replace("\n", " ")
    print(f"   Answer    {tag}  {short}{'...' if len(r['answer']) > 140 else ''}")


def main():
    print("=" * 70)
    print("  RAG EVALUATION")
    print("=" * 70)

    print(f"[INFO] Laddar embedding-modell...")
    embedder = SentenceTransformer(EMBED_MODEL_DIR, device="cuda")

    print(f"[INFO] Öppnar ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    print(f"[INFO] Laddar LLM...")
    llm = Llama(
        model_path=LLM_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )

    print(f"[INFO] Kör {len(TESTS)} testfall (TOP_K={TOP_K})...")
    t0 = time.time()
    results = []
    for i, (q, book, kws) in enumerate(TESTS, 1):
        r = evaluate_case(q, book, kws, embedder, collection, llm)
        results.append(r)
        print_result(i, r)

    elapsed = time.time() - t0

    # ── Sammanställning ──
    retrieval_cases   = [r for r in results if r["retrieval_ok"] is not None]
    retrieval_correct = sum(1 for r in retrieval_cases if r["retrieval_ok"])

    factual   = [r for r in results if r["expected_book"] is not None]
    fact_ok   = sum(1 for r in factual if r["answer_ok"])

    abstain   = [r for r in results if r["expected_book"] is None]
    abst_ok   = sum(1 for r in abstain if r["answer_ok"])

    total_ans_ok = sum(1 for r in results if r["answer_ok"])

    print("\n" + "=" * 70)
    print("  SAMMANSTÄLLNING")
    print("=" * 70)
    print(f"  Retrieval accuracy (rätt bok i top-{TOP_K}):  "
          f"{retrieval_correct}/{len(retrieval_cases)} "
          f"= {100*retrieval_correct/len(retrieval_cases):.0f}%")
    print(f"  Faktafrågor — korrekt svar:                 "
          f"{fact_ok}/{len(factual)} "
          f"= {100*fact_ok/len(factual):.0f}%")
    print(f"  Hallucinationstest — vägrade korrekt:        "
          f"{abst_ok}/{len(abstain)} "
          f"= {100*abst_ok/len(abstain):.0f}%")
    print(f"  Totalt korrekt:                              "
          f"{total_ans_ok}/{len(results)} "
          f"= {100*total_ans_ok/len(results):.0f}%")
    print(f"  Total tid: {elapsed:.1f}s  "
          f"({elapsed/len(results):.1f}s per fråga)")
    print("=" * 70)


if __name__ == "__main__":
    main()
