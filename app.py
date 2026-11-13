"""HW 4 - Alice RAG as a public web app with logging, rate limiting and a usage page.

Run locally:  streamlit run app.py
"""
import os
import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag import load_book, make_chunks, ANSWER_RULES, BOOK

load_dotenv()  # locally the key comes from .env, on Streamlit Cloud it comes from Secrets

CHAT_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
CHUNK_SIZE, OVERLAP, TOP_K = 800, 150, 5   # k=5 was the best config in Lab 3

PRICE_IN = 0.30 / 1_000_000    # Flash-Lite, USD per input token
PRICE_OUT = 2.50 / 1_000_000   # USD per output token

RATE_LIMIT = 5                 # max requests per user...
RATE_WINDOW = 60 * 60          # ...per hour
TIMEOUT_MS = 15000             # give up on the API after 15 seconds

LOG_FILE = "logs/requests.jsonl"


# ---------- setup (cached so it only runs once) ----------

@st.cache_resource
def load_index():
    chunks = make_chunks(load_book(BOOK), CHUNK_SIZE, OVERLAP)
    vectors = np.load(f"index_{CHUNK_SIZE}_{OVERLAP}.npy")
    return chunks, vectors


@st.cache_resource
def rate_limit_store():
    # one dictionary shared by every visitor: {user: [timestamps]}
    return {}


def make_client(failure):
    key = os.environ.get("GEMINI_API_KEY", "")
    timeout = TIMEOUT_MS
    if failure == "Revoked API key":
        key = "this-key-was-revoked"
    elif failure == "Timeout":
        timeout = 1   # 1 ms, so the request can't possibly finish
    return genai.Client(api_key=key, http_options={"timeout": timeout})


# ---------- logging ----------

def write_log(entry):
    os.makedirs("logs", exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_logs():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- guardrail: rate limit ----------

def allowed(user):
    store = rate_limit_store()
    now = time.time()
    recent = [t for t in store.get(user, []) if now - t < RATE_WINDOW]
    if len(recent) >= RATE_LIMIT:
        store[user] = recent
        wait_min = math.ceil((RATE_WINDOW - (now - recent[0])) / 60)
        return False, wait_min
    recent.append(now)
    store[user] = recent
    return True, 0


# ---------- the RAG call ----------

def ask(client, question):
    chunks, vectors = load_index()
    q = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[question],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    ).embeddings[0].values
    q = np.array(q)
    scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    top = np.argsort(scores)[::-1][:TOP_K]
    context = "\n\n".join(f"[chunk {chunks[i]['id']}]\n{chunks[i]['text']}" for i in top)

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=f"Context:\n{context}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(system_instruction=ANSWER_RULES, max_output_tokens=1024),
    )
    usage = response.usage_metadata
    tokens_in = usage.prompt_token_count or 0
    tokens_out = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    return response.text or "", tokens_in, tokens_out


def friendly_error(e):
    text = str(e).lower()
    if "timeout" in type(e).__name__.lower() or "timeout" in text or "timed out" in text:
        return "timeout", "The AI service took too long to answer. Please try again in a minute."
    if "api key" in text or "api_key" in text or "permission" in text or "401" in text or "403" in text:
        return "auth_error", "The app can't reach the AI service right now (key problem). Please try again later."
    if "429" in text or "quota" in text or "exhausted" in text:
        return "upstream_rate_limit", "The AI service is busy right now. Please wait a minute and try again."
    return "error", "Something went wrong on our side. Please try again later."


def handle_question(user, question, failure):
    ok, wait_min = allowed(user)
    if not ok:
        write_log({"user": user, "question": question[:100], "latency_ms": 0,
                   "tokens_in": 0, "tokens_out": 0, "cost_usd": 0, "outcome": "rate_limited"})
        return f"⛔ You've hit the limit of {RATE_LIMIT} questions per hour. Try again in about {wait_min} min."

    start = time.perf_counter()
    try:
        answer, tokens_in, tokens_out = ask(make_client(failure), question)
        outcome, error = "ok", None
    except Exception as e:
        answer, tokens_in, tokens_out = None, 0, 0
        outcome, message = friendly_error(e)
        error = f"{type(e).__name__}: {str(e)[:200]}"
    latency_ms = round((time.perf_counter() - start) * 1000)

    cost = tokens_in * PRICE_IN + tokens_out * PRICE_OUT
    write_log({"user": user, "question": question[:100], "latency_ms": latency_ms,
               "tokens_in": tokens_in, "tokens_out": tokens_out, "cost_usd": round(cost, 8),
               "outcome": outcome, "error": error})
    return answer if outcome == "ok" else f"⚠️ {message}"


# ---------- usage summary ----------

def percentile(values, p):
    values = sorted(values)
    index = max(0, math.ceil(p / 100 * len(values)) - 1)
    return values[index]


def usage_page():
    logs = read_logs()
    if not logs:
        st.info("No requests logged yet.")
        return
    api_calls = [l for l in logs if l["outcome"] != "rate_limited"]
    latencies = [l["latency_ms"] for l in api_calls] or [0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Requests", len(logs))
    c2.metric("p50 latency", f"{percentile(latencies, 50)} ms")
    c3.metric("p95 latency", f"{percentile(latencies, 95)} ms")
    c4.metric("Total est. cost", f"${sum(l['cost_usd'] for l in logs):.5f}")

    st.caption(f"Period: {logs[0]['timestamp']} → {logs[-1]['timestamp']} (UTC)")

    outcomes = {}
    for l in logs:
        outcomes[l["outcome"]] = outcomes.get(l["outcome"], 0) + 1
    st.subheader("Outcomes")
    st.bar_chart(outcomes)

    st.subheader("Latency per request (ms)")
    st.line_chart([l["latency_ms"] for l in api_calls])

    st.subheader("Raw log (newest first)")
    st.dataframe(list(reversed(logs)), width="stretch")
    st.download_button("Download requests.jsonl", open(LOG_FILE, "rb").read(), "requests.jsonl")


# ---------- page ----------

st.set_page_config(page_title="Ask Alice", page_icon="🐇")
st.title("🐇 Ask Alice in Wonderland")
st.caption("A RAG app that answers questions using only the text of the book.")

with st.sidebar:
    user = st.text_input("Your name", value="guest").strip().lower() or "guest"
    st.caption(f"Limit: {RATE_LIMIT} questions per user per hour.")
    st.divider()
    st.write("**Failure simulation (for testing)**")
    failure = st.radio("Mode", ["None", "Revoked API key", "Timeout"], label_visibility="collapsed")
    st.divider()
    page = st.radio("Page", ["Ask", "Usage"])

if page == "Ask":
    if "history" not in st.session_state:
        st.session_state.history = []
    for role, text in st.session_state.history:
        st.chat_message(role).write(text)

    # chat_input at the top level (not inside tabs) stays pinned to the bottom
    question = st.chat_input("Ask something about the book...")
    if question:
        st.chat_message("user").write(question)
        with st.spinner("Looking it up..."):
            reply = handle_question(user, question, failure)
        st.chat_message("assistant").write(reply)
        st.session_state.history += [("user", question), ("assistant", reply)]
else:
    usage_page()
