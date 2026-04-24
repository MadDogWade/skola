"""
RAG-chattbot med lokal LLM (Phi-4 mini) + EmbeddingGemma + ChromaDB
===================================================================
Kunskapsbas: 5 klassiker från Project Gutenberg (ligger i ./books/).

Krav:
    pip install llama-cpp-python sentence-transformers chromadb

Användning:
    python download_books.py   # en gång, laddar ner böckerna
    python rag_chatbot.py      # startar chatten
"""

import os
import glob
import textwrap

import chromadb
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────
LLM_PATH        = "Phi-4-mini-instruct-Q6_K.gguf"
EMBED_MODEL_DIR = "embeddinggemma-transformers-embeddinggemma-300m-v1"
BOOKS_DIR       = "books"
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "classics"

CHUNK_TARGET    = 1100   # mål-storlek per chunk (tecken)
CHUNK_MIN       = 400    # slå ihop stycken tills vi når minst så här
TOP_K           = 8
N_CTX           = 4096
N_GPU_LAYERS    = -1   # -1 = allt på GPU

# Snygga visningsnamn för böckerna
BOOK_TITLES = {
    "alice_in_wonderland":  "Alice's Adventures in Wonderland",
    "pride_and_prejudice":  "Pride and Prejudice",
    "frankenstein":         "Frankenstein",
    "sherlock_holmes":      "The Adventures of Sherlock Holmes",
    "dracula":              "Dracula",
}


# ─────────────────────────────────────────────
# Läs in + chunka en bok
# ─────────────────────────────────────────────
def load_and_chunk(filepath):
    """
    Styckesbaserad chunking:
      1. Ta bort Gutenberg header/footer.
      2. Dela på tomma rader (\\n\\n) — bevarar stycken hela.
      3. Slå ihop korta stycken tills vi passerar CHUNK_MIN.
      4. Om ett stycke ensamt är större än CHUNK_TARGET, dela det
         på meningsgränser istället för mitt i ord.
    Inga meningar klipps någonsin mitt i.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Ta bort Project Gutenberg header/footer
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker   = "*** END OF THE PROJECT GUTENBERG EBOOK"
    s = text.find(start_marker)
    e = text.find(end_marker)
    if s != -1:
        text = text[text.index("\n", s) + 1:]
    if e != -1:
        text = text[:e]
    text = text.strip()

    # Dela på tomma rader → stycken
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Normalisera whitespace inom varje stycke (en rad istället för brutna rader)
    paragraphs = [" ".join(p.split()) for p in raw_paragraphs]

    # Om ett stycke är gigantiskt, dela det på meningsgränser
    def split_long_paragraph(par):
        if len(par) <= CHUNK_TARGET:
            return [par]
        import re
        sentences = re.split(r"(?<=[.!?])\s+", par)
        out, buf = [], ""
        for sent in sentences:
            if len(buf) + len(sent) + 1 > CHUNK_TARGET and buf:
                out.append(buf.strip())
                buf = sent
            else:
                buf = (buf + " " + sent).strip()
        if buf:
            out.append(buf.strip())
        return out

    units = []
    for p in paragraphs:
        units.extend(split_long_paragraph(p))

    # Slå ihop korta stycken till chunks runt CHUNK_TARGET
    chunks = []
    buf = ""
    for u in units:
        if not buf:
            buf = u
            continue
        if len(buf) < CHUNK_MIN or len(buf) + len(u) + 1 <= CHUNK_TARGET:
            buf = buf + " " + u
        else:
            chunks.append(buf)
            buf = u
    if buf:
        chunks.append(buf)

    return chunks


def load_all_books():
    """Läser alla .txt i books/, returnerar (chunks, metadatas)."""
    all_chunks, all_meta = [], []
    files = sorted(glob.glob(os.path.join(BOOKS_DIR, "*.txt")))
    if not files:
        raise RuntimeError(f"Inga .txt-filer i {BOOKS_DIR}/ — kör download_books.py först")

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        title = BOOK_TITLES.get(stem, stem)
        chunks = load_and_chunk(path)
        print(f"  {title}: {len(chunks)} chunks")
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            all_meta.append({"title": title, "source": stem, "chunk_index": i})
    print(f"[INFO] Totalt {len(all_chunks)} chunks från {len(files)} böcker")
    return all_chunks, all_meta


# ─────────────────────────────────────────────
# Bygg / öppna vektordatabasen
# ─────────────────────────────────────────────
def build_vector_store(embedder):
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print("[INFO] Vektordatabas finns redan — återanvänder.")
        return client.get_collection(COLLECTION_NAME)

    print("[INFO] Läser och chunkar böcker...")
    chunks, metas = load_all_books()

    print("[INFO] Genererar embeddings (EmbeddingGemma)...")
    # EmbeddingGemma rekommenderar prompt-prefix för dokument
    embeddings = embedder.encode(
        [f"title: none | text: {c}" for c in chunks],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    # Lägg till i batchar för att undvika för stora requests
    B = 500
    for i in range(0, len(chunks), B):
        collection.add(
            ids=ids[i:i+B],
            documents=chunks[i:i+B],
            embeddings=embeddings[i:i+B],
            metadatas=metas[i:i+B],
        )
    print(f"[INFO] Indexerade {len(chunks)} chunks i ChromaDB")
    return collection


# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────
def retrieve(query, embedder, collection, top_k=TOP_K):
    q_emb = embedder.encode(
        [f"task: search result | query: {query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    docs  = results["documents"][0]
    metas = results["metadatas"][0]
    return list(zip(docs, metas))


# ─────────────────────────────────────────────
# Prompt + generering
# ─────────────────────────────────────────────
def build_prompt(query, retrieved):
    context_blocks = []
    for doc, meta in retrieved:
        context_blocks.append(f"[From: {meta['title']}]\n{doc}")
    context = "\n\n---\n\n".join(context_blocks)

    return (
        "<|system|>\n"
        "You are a helpful assistant that answers questions about classic literature. "
        "Base your answer on the provided context passages. You may synthesize and "
        "reason across multiple passages to form a complete answer. Always mention "
        "which book the information comes from. Only say 'I cannot find that "
        "information in the provided books' if the context contains nothing relevant "
        "to the question at all.\n"
        "<|end|>\n"
        "<|user|>\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "<|end|>\n"
        "<|assistant|>\n"
    )


def generate_answer(llm, prompt):
    out = llm(
        prompt,
        max_tokens=512,
        stop=["<|end|>", "<|user|>"],
        temperature=0.3,
        top_p=0.9,
        echo=False,
    )
    return out["choices"][0]["text"].strip()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  RAG-CHATTBOT: 5 klassiker från Project Gutenberg")
    print("  Phi-4 mini + EmbeddingGemma + ChromaDB")
    print("=" * 60)

    print(f"[INFO] Laddar embedding-modell från {EMBED_MODEL_DIR}...")
    embedder = SentenceTransformer(EMBED_MODEL_DIR, device="cuda")

    collection = build_vector_store(embedder)

    print(f"[INFO] Laddar LLM från {LLM_PATH}...")
    llm = Llama(
        model_path=LLM_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
    print("[INFO] Modell laddad! Skriv dina frågor nedan.\n")

    while True:
        query = input("\nDin fråga (eller 'quit'): ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Hej då!")
            break
        if not query:
            continue

        retrieved = retrieve(query, embedder, collection)

        print("\n[RETRIEVAL] Hämtade chunks:")
        for i, (doc, meta) in enumerate(retrieved, 1):
            preview = doc[:100].replace("\n", " ")
            print(f"  {i}. [{meta['title']}] {preview}...")

        prompt = build_prompt(query, retrieved)
        print("\n[SVAR]")
        print(textwrap.fill(generate_answer(llm, prompt), width=80))


if __name__ == "__main__":
    main()
