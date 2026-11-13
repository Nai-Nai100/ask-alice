import os
import json
import time
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options={"timeout": 60000})

CHAT_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
BOOK = "data/alice.txt"
CHUNK_SIZE = 800
OVERLAP = 150
#CHUNK_SIZE = 400
#OVERLAP = 80

def load_book(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start != -1 and end != -1:
        text = text[text.find("\n", start):end]
    return text.strip()


def make_chunks(text, size, overlap):
    chunks = []
    step = size - overlap
    for i in range(0, len(text), step):
        piece = text[i:i + size].strip()
        if piece:
            chunks.append({"id": len(chunks), "source": "alice.txt", "text": piece})
    return chunks

def embed_texts(texts, task):
    vectors = []
    for i in range(0, len(texts), 90):
        batch = texts[i:i + 90]
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task),
        )
        if i + 90 < len(texts):
            print(f"Embedded {i + 90}/{len(texts)}, waiting 61s for rate limit...")
            time.sleep(61)
        vectors.extend(e.values for e in result.embeddings)
    return np.array(vectors)


def build_index(chunks):
    cache = f"index_{CHUNK_SIZE}_{OVERLAP}.npy"
    if os.path.exists(cache):
        return np.load(cache)
    vectors = embed_texts([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    np.save(cache, vectors)
    return vectors


def search(query, chunks, vectors, k=3):
    q = embed_texts([query], "RETRIEVAL_QUERY")[0]
    scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    top = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top]

TEST_QUESTIONS = [
    ("What words were written on the little bottle?", "drink me"),
    ("What was written on the little cake?", "eat me"),
    ("What is the name of Alice's cat?", "dinah"),
    ("What did they use as mallets in the Queen's croquet game?", "flamingo"),
    ("What riddle does the Hatter ask Alice?", "raven"),
]


def evaluate(chunks, vectors, k=3):
    hits = 0
    for question, answer in TEST_QUESTIONS:
        results = search(question, chunks, vectors, k)
        found = any(answer in c["text"].lower() for c, s in results)
        hits += found
        ids = [c["id"] for c, s in results]
        print(f"\nQ: {question}")
        print(f"   Expected word: '{answer}'  |  Top-{k} chunks: {ids}  |  {'HIT' if found else 'MISS'}")
        for c, s in results:
            print(f"   - chunk {c['id']} ({s:.3f}): {c['text'][:90].replace(chr(10), ' ')}...")
    print(f"\nHit rate: {hits}/{len(TEST_QUESTIONS)}")

ANSWER_RULES = (
    "You answer questions about the book Alice's Adventures in Wonderland. "
    "Use ONLY the context chunks given to you. Do not use outside knowledge. "
    "After each fact, cite the chunk it came from like [chunk 12]. "
    "If the context does not contain the answer, reply exactly: "
    "\"I don't know - the provided text does not cover that.\""
)

def answer(question, chunks, vectors, k=3):
    results = search(question, chunks, vectors, k)
    context = "\n\n".join(f"[chunk {c['id']}]\n{c['text']}" for c, s in results)

    response = client.interactions.create(
        model=CHAT_MODEL,
        store=False,
        system_instruction=ANSWER_RULES,
        input=f"Context:\n{context}\n\nQuestion: {question}",
        generation_config={"max_output_tokens": 1024},
    )

    print(f"\nAnswer: {response.output_text}")
    print("\nSources:")
    for c, s in results:
        snippet = c["text"][:120].replace("\n", " ")
        print(f"  - {c['source']} chunk {c['id']} (score {s:.3f}): \"{snippet}...\"")

if __name__ == "__main__":
    text = load_book(BOOK)
    chunks = make_chunks(text, CHUNK_SIZE, OVERLAP)
    print(f"Chunk size {CHUNK_SIZE}, overlap {OVERLAP} -> {len(chunks)} chunks")

    vectors = build_index(chunks)
    print(f"Embedded {len(vectors)} chunks, each vector has {vectors.shape[1]} numbers")

    evaluate(chunks, vectors)

    print("\n===== Ask the book (type 'quit' to stop) =====")
    while True:
        question = input("\nyou > ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if question:
            answer(question, chunks, vectors)