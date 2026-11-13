"""Lab 3 - evaluation harness for the Alice RAG system from Lab 2.

Run every config:      python eval.py
Run one config:        python eval.py A_baseline
"""
import os
import sys
import json
import time
import statistics
import numpy as np
from google.genai import types

# reuse the pieces I already built in Lab 2
from rag import client, CHAT_MODEL, BOOK, ANSWER_RULES, load_book, make_chunks, embed_texts

# Gemini 3.5 Flash-Lite paid-tier prices (USD per token)
PRICE_IN = 0.30 / 1_000_000
PRICE_OUT = 2.50 / 1_000_000

JUDGE_MODEL = CHAT_MODEL
REFUSAL = "don't know"

# each config changes ONE thing compared to the baseline
CONFIGS = {
    "A_baseline":  {"chunk": 800, "overlap": 150, "k": 3},
    "B_k5":        {"chunk": 800, "overlap": 150, "k": 5},   # variable 1: top-k up
    "C_k2":        {"chunk": 800, "overlap": 150, "k": 2},   # variable 1: top-k down
    "D_chunk400":  {"chunk": 400, "overlap": 80,  "k": 3},   # variable 2: chunk size
}


def clean(text):
    # the book uses curly apostrophes, so normalize before matching
    return text.lower().replace("’", "'").replace("‘", "'")


def with_retry(fn, tries=5):
    # free tier hits rate limits, so wait and try again instead of crashing
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if attempt == tries - 1:
                raise
            wait = 15 * (attempt + 1)
            print(f"   API error ({type(e).__name__}), retrying in {wait}s...")
            time.sleep(wait)


def ask(question, chunks, vectors, k):
    """One full RAG call: embed question -> search -> generate. Timed end to end."""
    start = time.perf_counter()

    q = embed_texts([question], "RETRIEVAL_QUERY")[0]
    scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    top = np.argsort(scores)[::-1][:k]
    retrieved = [chunks[i] for i in top]
    context = "\n\n".join(f"[chunk {c['id']}]\n{c['text']}" for c in retrieved)

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=f"Context:\n{context}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(system_instruction=ANSWER_RULES, max_output_tokens=1024),
    )
    latency = time.perf_counter() - start

    usage = response.usage_metadata
    tokens_in = usage.prompt_token_count or 0
    tokens_out = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)

    return {
        "answer": response.text or "",
        "chunk_ids": [c["id"] for c in retrieved],
        "retrieved_text": clean(" ".join(c["text"] for c in retrieved)),
        "latency": latency,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost": tokens_in * PRICE_IN + tokens_out * PRICE_OUT,
    }


def judge(question, expected, actual):
    """LLM-as-a-judge: returns (0 or 1, one-sentence reason)."""
    prompt = (
        "You are grading an answer about Alice's Adventures in Wonderland.\n"
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Actual answer: {actual}\n\n"
        "Give score 1 if the actual answer gets the main point of the expected answer right "
        "(wording can differ, extra correct detail is fine). Give 0 if it is wrong, missing "
        "the main point, or says it doesn't know.\n"
        'Reply as JSON only: {"score": 0 or 1, "reason": "one sentence"}'
    )
    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
    )
    result = json.loads(response.text)
    return int(result["score"]), result["reason"]


def score_item(item, out):
    """Returns (passed, reason)."""
    answer = clean(out["answer"])
    if item["type"] == "keyword":
        passed = any(word in answer for word in item["expected"])
        return passed, f"looked for {item['expected']}"
    if item["type"] == "refusal":
        passed = REFUSAL in answer
        return passed, "should refuse (not in the book)"
    score, reason = with_retry(lambda: judge(item["question"], item["expected"], out["answer"]))
    return bool(score), reason


def failure_bucket(item, out):
    """Puts every failure in one of 4 named buckets."""
    if item["type"] == "refusal":
        return "hallucinated (answered an out-of-book question)"
    if item["evidence"] not in out["retrieved_text"]:
        return "retrieval missed the right chunk"
    if REFUSAL in clean(out["answer"]):
        return "right chunk retrieved but model refused"
    return "right chunk retrieved but answer wrong/incomplete"


def run_config(name, cfg, testset, text):
    print(f"\n========== {name}: chunk {cfg['chunk']}/{cfg['overlap']}, k={cfg['k']} ==========")
    chunks = make_chunks(text, cfg["chunk"], cfg["overlap"])
    vectors = np.load(f"index_{cfg['chunk']}_{cfg['overlap']}.npy")  # saved in Lab 2, no re-embedding

    results = []
    for item in testset:
        out = with_retry(lambda: ask(item["question"], chunks, vectors, cfg["k"]))
        passed, reason = score_item(item, out)
        bucket = None if passed else failure_bucket(item, out)

        tag = "HOLDOUT " if item["holdout"] else ""
        print(f"[{'PASS' if passed else 'FAIL'}] {tag}#{item['id']} {item['question']}")
        if not passed:
            print(f"       -> {bucket} | {reason}")

        out.pop("retrieved_text")
        results.append({**item, **out, "passed": passed, "reason": reason, "bucket": bucket})

    dev = [r for r in results if not r["holdout"]]
    hold = [r for r in results if r["holdout"]]
    summary = {
        "config": name,
        **cfg,
        "score": round(100 * sum(r["passed"] for r in results) / len(results), 1),
        "dev_score": round(100 * sum(r["passed"] for r in dev) / len(dev), 1),
        "holdout_score": round(100 * sum(r["passed"] for r in hold) / len(hold), 1),
        "median_latency_s": round(statistics.median(r["latency"] for r in results), 3),
        "cost_per_query": sum(r["cost"] for r in results) / len(results),
        "buckets": {},
    }
    for r in results:
        if r["bucket"]:
            summary["buckets"][r["bucket"]] = summary["buckets"].get(r["bucket"], 0) + 1

    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "items": results}, f, indent=2)

    print(f"\nSCORE: {summary['score']}%  (dev {summary['dev_score']}%, holdout {summary['holdout_score']}%)")
    print(f"Median latency: {summary['median_latency_s']}s | Cost/query: ${summary['cost_per_query']:.6f}")
    print("Failure buckets:")
    for bucket, count in summary["buckets"].items():
        print(f"   {count}  {bucket}")
    return summary


def comparison_table(summaries):
    base = summaries[0]
    lines = [
        "| Config | Chunk | k | Quality | Dev | Holdout | Median latency | Cost/query | Cost vs baseline |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        change = 100 * (s["cost_per_query"] - base["cost_per_query"]) / base["cost_per_query"]
        lines.append(
            f"| {s['config']} | {s['chunk']}/{s['overlap']} | {s['k']} | {s['score']}% | {s['dev_score']}% | "
            f"{s['holdout_score']}% | {s['median_latency_s']}s | ${s['cost_per_query']:.6f} | {change:+.0f}% |"
        )
    table = "\n".join(lines)
    with open("results/comparison.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    with open("testset.json", encoding="utf-8") as f:
        testset = json.load(f)
    text = load_book(BOOK)

    names = sys.argv[1:] or list(CONFIGS)
    summaries = [run_config(n, CONFIGS[n], testset, text) for n in names]
    if len(summaries) > 1:
        comparison_table(summaries)
