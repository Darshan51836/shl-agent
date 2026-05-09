import json, os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
from groq import Groq

GROQ_API_KEY   = "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"
MODEL_NAME     = "llama-3.1-8b-instant"
EMBED_MODEL    = "all-MiniLM-L6-v2"
FAISS_INDEX    = "data/catalog_index.faiss"
META_PATH      = "data/catalog_index_meta.json"
CATALOG_PATH   = "data/catalog_clean.json"
TOP_K          = 10

print("[startup] Loading embedding model...")
_embedder = SentenceTransformer(EMBED_MODEL)
print("[startup] Loading FAISS index...")
_index = faiss.read_index(FAISS_INDEX)
print("[startup] Loading metadata...")
with open(META_PATH, encoding="utf-8") as f:
    _meta = json.load(f)
with open(CATALOG_PATH, encoding="utf-8") as f:
    _catalog = {item["entity_id"]: item for item in json.load(f)}
print("[startup] Configuring Groq...")
_client = Groq(api_key=GROQ_API_KEY)
print(f"[startup] Ready — {_index.ntotal} assessments indexed.")

app = FastAPI(title="SHL Assessment Intelligence", description="Conversational SHL assessment recommender using RAG + LLaMA 3.1", version="1.0.0")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: Optional[List[Recommendation]]
    end_of_conversation: bool

def retrieve(query: str, k: int = TOP_K) -> List[dict]:
    vec = _embedder.encode([query], normalize_embeddings=True)
    _, indices = _index.search(np.array(vec, dtype="float32"), k)
    return [_meta[i] for i in indices[0] if 0 <= i < len(_meta)]

SYSTEM_PROMPT = """You are an SHL assessment recommendation assistant.
Return a JSON object ONLY — no extra text, no markdown, no explanation.
Response format:
{"reply": "your message", "recommendations": [{"name": "...", "url": "https://...", "test_type": "K"}], "end_of_conversation": false}
Rules:
- If you need more info, set recommendations to null and ask ONE clarifying question
- When you have enough info, include 1-10 recommendations
- Set end_of_conversation to true only when user says goodbye
- test_type codes: A=Ability, K=Knowledge, P=Personality, S=Simulation, C=Competency, B=Biodata, E=Exercise, D=Development
- Always return valid JSON only"""

@app.get("/", response_class=HTMLResponse)
async def root():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>SHL Assessment API</h1><a href='/docs'>API Docs</a>")

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    last_user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    candidates = retrieve(last_user_msg)
    catalog_context = "\n".join(
        f"- {c['name']} | type:{c.get('test_type','')} | url:{c.get('url','')} | {c.get('description','')[:120]}"
        for c in candidates
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT + f"\n\nRelevant catalog items:\n{catalog_context}"}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages]
    response = _client.chat.completions.create(model=MODEL_NAME, messages=messages, max_tokens=1024, temperature=0.3)
    raw = response.choices[0].message.content.strip()
    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        data = json.loads(raw)
        recs = [Recommendation(**r) for r in data["recommendations"]] if data.get("recommendations") else None
        return ChatResponse(reply=data.get("reply",""), recommendations=recs, end_of_conversation=data.get("end_of_conversation", False))
    except Exception:
        return ChatResponse(reply=raw, recommendations=None, end_of_conversation=False)

@app.get("/health")
async def health():
    return {"status": "ok"}










# import json, os
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import Optional, List
# from groq import Groq

# # ── Config ────────────────────────────────────────────────────────────────────
# GROQ_API_KEY = "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"   # ← replace this
# MODEL_NAME = "llama-3.1-8b-instant"             # free on Groq
# EMBED_MODEL    = "all-MiniLM-L6-v2"
# FAISS_INDEX    = "data/catalog_index.faiss"
# META_PATH      = "data/catalog_index_meta.json"
# CATALOG_PATH   = "data/catalog_clean.json"
# TOP_K          = 10


# # ── Startup: load everything once ────────────────────────────────────────────
# print("[startup] Loading embedding model...")
# _embedder = SentenceTransformer(EMBED_MODEL)

# print("[startup] Loading FAISS index...")
# _index = faiss.read_index(FAISS_INDEX)

# print("[startup] Loading metadata...")
# with open(META_PATH, encoding="utf-8") as f:
#     _meta = json.load(f)          # list of dicts with name, url, test_type, etc.

# with open(CATALOG_PATH, encoding="utf-8") as f:
#     _catalog = {item["entity_id"]: item for item in json.load(f)}

# print("[startup] Configuring Groq...")
# _client = Groq(api_key=GROQ_API_KEY)

# print(f"[startup] Ready — {_index.ntotal} assessments indexed.")

# # ── FastAPI app ───────────────────────────────────────────────────────────────
# app = FastAPI(title="SHL Assessment Recommender")

# # ── Pydantic models ───────────────────────────────────────────────────────────
# class Message(BaseModel):
#     role: str
#     content: str

# class ChatRequest(BaseModel):
#     messages: List[Message]

# class Recommendation(BaseModel):
#     name: str
#     url: str
#     test_type: str

# class ChatResponse(BaseModel):
#     reply: str
#     recommendations: Optional[List[Recommendation]]
#     end_of_conversation: bool

# # ── Retriever ─────────────────────────────────────────────────────────────────
# def retrieve(query: str, k: int = TOP_K) -> List[dict]:
#     vec = _embedder.encode([query], normalize_embeddings=True)
#     _, indices = _index.search(np.array(vec, dtype="float32"), k)
#     results = []
#     for idx in indices[0]:
#         if 0 <= idx < len(_meta):
#             results.append(_meta[idx])
#     return results

# # ── System prompt ─────────────────────────────────────────────────────────────
# SYSTEM_PROMPT = """You are an SHL assessment recommendation assistant.

# Your job:
# 1. Clarify the user's hiring/development need (role, level, what they want to measure)
# 2. Search the catalog and recommend the best assessments
# 3. Return a JSON object ONLY — no extra text, no markdown, no explanation.

# Response format (always return valid JSON):
# {
#   "reply": "your conversational message to the user",
#   "recommendations": [
#     {
#       "name": "assessment name",
#       "url": "https://...",
#       "test_type": "K"
#     }
#   ],
#   "end_of_conversation": false
# }

# Rules:
# - If you need more info before recommending, set recommendations to null and ask ONE clarifying question in reply
# - When you have enough info, include 1–10 recommendations
# - Set end_of_conversation to true only when the user says goodbye or thanks and is done
# - test_type must use short codes: A=Ability, K=Knowledge, P=Personality, S=Simulation, C=Competency, B=Biodata, E=Exercise, D=Development
# - Always return valid JSON — never return plain text
# """

# # ── Chat endpoint ─────────────────────────────────────────────────────────────
# @app.post("/chat", response_model=ChatResponse)
# async def chat(req: ChatRequest):
#     # Build context from last user message
#     last_user_msg = ""
#     for m in reversed(req.messages):
#         if m.role == "user":
#             last_user_msg = m.content
#             break

#     # Retrieve relevant assessments
#     candidates = retrieve(last_user_msg)
#     catalog_context = "\n".join(
#         f"- {c['name']} | type:{c.get('test_type','')} | url:{c.get('url','')} | {c.get('description','')[:120]}"
#         for c in candidates
#     )

#     # Build messages for Groq
#     messages = [{"role": "system", "content": SYSTEM_PROMPT + f"\n\nRelevant catalog items:\n{catalog_context}"}]
#     for m in req.messages:
#         messages.append({"role": m.role, "content": m.content})

#     # Call Groq
#     response = _client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=messages,
#         max_tokens=1024,
#         temperature=0.3,
#     )

#     raw = response.choices[0].message.content.strip()

#     # Parse JSON response
#     try:
#         # Strip markdown fences if present
#         if raw.startswith("```"):
#             raw = raw.split("```")[1]
#             if raw.startswith("json"):
#                 raw = raw[4:]
#         data = json.loads(raw)
#         recs = None
#         if data.get("recommendations"):
#             recs = [Recommendation(**r) for r in data["recommendations"]]
#         return ChatResponse(
#             reply=data.get("reply", ""),
#             recommendations=recs,
#             end_of_conversation=data.get("end_of_conversation", False),
#         )
#     except Exception:
#         # Fallback: return raw as reply
#         return ChatResponse(
#             reply=raw,
#             recommendations=None,
#             end_of_conversation=False,
#         )

# # ── Health check ──────────────────────────────────────────────────────────────
# @app.get("/health")
# async def health():
#     return {"status": "ok"}