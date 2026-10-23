#!/usr/bin/env python3
"""Arabic Fatwa Q&A — semantic search over Sheikh Ibn Baz's fatwas (Gradio app).

Retrieval: `Qwen/Qwen3-Embedding-0.6B` (CPU by default) over precomputed corpus
embeddings. Reranking: `ALJIACHI/Mizan-Rerank-V2` (GPU by default).

LLM steps are optional. From the UI you can configure:
  * the provider (Gemini or OpenRouter) and the model for query rewriting;
  * whether to judge, and the provider/model for judging (Gemini, or JEV via
    OpenRouter's Decisions API, or any OpenRouter chat model with JSON output);
  * retrieval tuning: how many to retrieve, how many to rerank, how many to show;
  * which verdicts to display (e.g. only answers-it, or answers-it + related-only);
  * per-step timings and token usage.

Errors are surfaced in a persistent modal (copy it and hand it to your AI) and in
a copyable error box at the bottom.

Quick start
-----------
    python app.py

Then open the printed local URL (default http://127.0.0.1:7860).

The three data files are resolved from `./data` next to this script and are
downloaded from the Hugging Face dataset on first run (see `--no-download`):
    binbaz_clean.csv, qwen3_06b_title_answer_embeddings.pt, embedding_metadata.json

Keys (optional, auto-loaded from `.env`, see `.env.example`):
    GEMINI_API_KEY       for Gemini rewrite/judge
    OPENROUTER_API_KEY   for OpenRouter rewrite/judge (incl. JEV)
"""

from __future__ import annotations

import argparse
import ast
import functools
import gc
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

APP_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def load_env_files() -> None:
    """Populate os.environ from local .env files (existing variables win)."""
    candidates = []
    custom = os.environ.get("FATWA_ENV_FILE")
    if custom:
        candidates.append(Path(custom).expanduser())
    candidates.append(APP_DIR / ".env")
    candidates.append(Path.home() / ".config" / "fatwa_qa" / ".env")

    for path in candidates:
        try:
            if not path.is_file():
                continue
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
        except OSError:
            continue


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_REPO_ID = os.environ.get("FATWA_DATA_REPO", "ZenoAbderrahim/binbaz_fatwas_rag")
DATA_FILES = [
    "binbaz_clean.csv",
    "qwen3_06b_title_answer_embeddings.pt",
    "embedding_metadata.json",
]

EMBEDDER_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL_NAME = "ALJIACHI/Mizan-Rerank-V2"

EMBEDDER_MAX_SEQ_LENGTH = 1024
RERANKER_MAX_LENGTH = 1024
RERANKER_BATCH_SIZE = 16

DEFAULT_RETRIEVE_K = 20
MAX_RETRIEVE_K = 200
DEFAULT_RERANK_POOL = 20
DEFAULT_SHOW_N = 5
DEFAULT_MAX_CANDIDATE_CHARS = 3000

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

PROVIDER_GEMINI = "Gemini"
PROVIDER_OPENROUTER = "OpenRouter"
PROVIDERS = [PROVIDER_GEMINI, PROVIDER_OPENROUTER]

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]
OPENROUTER_CHAT_MODELS = [
    "google/gemini-3.1-flash-lite",
    "google/gemini-3.5-flash",
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
    "qwen/qwen3.5-flash-02-23",
    "qwen/qwen3-30b-a3b-instruct-2507",
]
JEV_MODELS = ["typesafe/jev-1.13", "~typesafe/jev-latest"]

VERDICTS = ["answers-it", "related-only", "unrelated"]

JEV_CRITERIA = {
    "answers-it": "The fatwa directly answers the user's question or gives the requested ruling/information.",
    "related-only": "The fatwa is on a related topic but does not answer the specific question.",
    "unrelated": "The fatwa does not meaningfully address the user's question.",
}


def models_for(task: str, provider: str) -> list[str]:
    if task == "judge":
        return GEMINI_MODELS if provider == PROVIDER_GEMINI else JEV_MODELS
    return GEMINI_MODELS if provider == PROVIDER_GEMINI else OPENROUTER_CHAT_MODELS


def is_jev_model(model: str) -> bool:
    m = (model or "").lower()
    return "jev" in m and ("typesafe" in m or m.startswith("~"))


# ---------------------------------------------------------------------------
# Schemas / small data structures
# ---------------------------------------------------------------------------


class QueryRewrite(BaseModel):
    language: str = Field(description="Detected input language, e.g. Arabic, English, French")
    improved_query: str = Field(description="A clear Arabic search query preserving the user's intent")
    changed: bool = Field(description="Whether translation or reformulation materially changed the query")


class FatwaJudgementModel(BaseModel):
    candidate_id: int
    verdict: str = Field(description="Exactly one of: answers-it, related-only, unrelated")
    confidence: str = Field(description="low, medium, or high")
    answer: str = Field(description="A concise Arabic answer grounded strictly in this fatwa")
    explanation: str = Field(description="Brief Arabic explanation of the classification")


class JudgeResponse(BaseModel):
    judgements: list[FatwaJudgementModel]


@dataclass
class FatwaJudgement:
    verdict: str
    confidence: str | float | None = None
    answer: str = ""
    explanation: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class Stat:
    label: str
    seconds: float = 0.0
    prompt: int = 0
    completion: int = 0
    total: int = 0
    cost: float = 0.0
    provider: str = ""
    model: str = ""
    note: str = ""


def usage_dict(prompt: Any = 0, completion: Any = 0, total: Any = None, cost: Any = 0.0) -> dict[str, Any]:
    prompt = int(prompt or 0)
    completion = int(completion or 0)
    return {
        "prompt": prompt,
        "completion": completion,
        "total": int(total) if total is not None else prompt + completion,
        "cost": float(cost or 0.0),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def parse_categories(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    return [x.strip() for x in re.split(r"[,;]", text) if x.strip()]


def load_project_assets(data_dir: Path):
    clean_csv = data_dir / "binbaz_clean.csv"
    embeddings_pt = data_dir / "qwen3_06b_title_answer_embeddings.pt"
    metadata_json = data_dir / "embedding_metadata.json"

    missing = [p for p in (clean_csv, embeddings_pt, metadata_json) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(f"- {p}" for p in missing))

    df = pd.read_csv(clean_csv)
    with open(metadata_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    expected_meta = {
        "model": EMBEDDER_MODEL_NAME,
        "embed_dim": 1024,
        "max_seq_length": EMBEDDER_MAX_SEQ_LENGTH,
        "n_rows": len(df),
    }
    mismatches = [
        (k, e, meta.get(k)) for k, e in expected_meta.items() if meta.get(k) != e
    ]
    if mismatches:
        details = "\n".join(f"- {k}: expected {e!r}, got {a!r}" for k, e, a in mismatches)
        raise ValueError("Embedding metadata mismatch:\n" + details)

    embeddings = torch.load(embeddings_pt, map_location="cpu", weights_only=True)
    if isinstance(embeddings, dict):
        for key in ("embeddings", "tensor", "data"):
            if key in embeddings:
                embeddings = embeddings[key]
                break
    if not torch.is_tensor(embeddings):
        embeddings = torch.as_tensor(embeddings)

    embeddings = embeddings.float().contiguous()
    if embeddings.ndim != 2:
        raise ValueError(f"Expected [N, D] embeddings, got shape {tuple(embeddings.shape)}")
    if embeddings.shape[0] != len(df):
        raise ValueError(f"Embedding row count {embeddings.shape[0]} != CSV row count {len(df)}")
    if embeddings.shape[1] != 1024:
        raise ValueError(f"Expected embedding dimension 1024, got {embeddings.shape[1]}")

    embeddings = F.normalize(embeddings, p=2, dim=1)

    required_cols = {"fatwa_id", "source", "title", "question", "answer"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_cols)}")

    if "categories" not in df.columns:
        df["categories"] = ""
    if "links" not in df.columns:
        df["links"] = ""
    if "embed_text" not in df.columns:
        df["embed_text"] = df["title"].fillna("") + " — " + df["answer"].fillna("")

    df["categories_list"] = df["categories"].apply(parse_categories)
    return df.reset_index(drop=True), embeddings, meta


# ---------------------------------------------------------------------------
# LLM helpers — OpenRouter
# ---------------------------------------------------------------------------


def get_openrouter_key() -> str:
    api_key = os.getenv(OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"{OPENROUTER_API_KEY_ENV} is not set. Put it in .env or export it, "
            f"e.g. export {OPENROUTER_API_KEY_ENV}='sk-or-v1-...'"
        )
    return api_key


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def openrouter_chat(api_key: str, model: str, system_prompt: str, user_prompt: str, timeout: int = 90):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(OPENROUTER_CHAT_URL, headers=_headers(api_key), json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenRouter chat HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    u = data.get("usage", {}) or {}
    usage = usage_dict(u.get("prompt_tokens"), u.get("completion_tokens"), u.get("total_tokens"), u.get("cost"))
    return _extract_json(content), usage


def call_jev(api_key: str, model: str, state: dict[str, Any], questions: dict[str, Any], timeout: int = 150):
    payload = {"model": model, "state": state, "questions": questions}
    resp = requests.post(OPENROUTER_DECISIONS_URL, headers=_headers(api_key), json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"JEV HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    u = data.get("usage", {}) or {}
    usage = usage_dict(u.get("input_tokens"), u.get("output_tokens"), None, u.get("cost"))
    return data.get("answers", {}) or {}, usage


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

REWRITE_SYSTEM = (
    "You preprocess a user's question for an Arabic Islamic-fatwa retrieval system. "
    "Detect the input language; if it is not Arabic, translate the intent into natural Modern "
    "Standard Arabic; if it is Arabic, reformulate it into a clear, concise search query while "
    "preserving the exact intent. Do NOT answer the question and do NOT introduce facts, names, "
    "rulings, or constraints that were not present. Respond with a single JSON object using keys "
    '"language", "improved_query", and "changed" (a boolean).'
)


def rewrite_user_prompt(query: str) -> str:
    return f"User input:\n{query}"


def judge_system_prompt() -> str:
    return (
        "You are a careful relevance judge for an Arabic Islamic-fatwa search system. For each "
        "candidate, independently decide whether its fatwa actually answers the user's question. "
        "Allowed verdicts: answers-it (directly answers), related-only (related topic but does not "
        "answer), unrelated (does not meaningfully address). Ground everything in the candidate text "
        "only; do not fabricate citations, rulings, or facts and do not merge candidates. Respond "
        'with a single JSON object: {"judgements": [{"candidate_id": int, "verdict": str, '
        '"confidence": "low|medium|high", "answer": str (Arabic, grounded), "explanation": str (Arabic)}]}.'
    )


def judge_user_prompt(user_query: str, candidates: pd.DataFrame, max_chars: int) -> str:
    blocks = []
    for row in candidates.itertuples(index=False):
        blocks.append(
            f"\nCANDIDATE_ID: {int(row.fatwa_id)}\n"
            f"TITLE: {str(row.title)[:max_chars]}\n"
            f"QUESTION: {str(row.question)[:max_chars]}\n"
            f"ANSWER:\n{str(row.answer)[:max_chars]}\n"
            f"SOURCE: {row.source}\n"
        )
    return f"User query:\n{user_query}\n\nCandidates:{''.join(blocks)}"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class FatwaApp:
    def __init__(self, settings: "Settings"):
        self.s = settings
        self._gemini_client = None
        self._openrouter_key: str | None = None

        print(f"Data dir: {settings.data_dir}")
        print(f"Embedder device: {settings.device_embedder}")
        print(f"Reranker device: {settings.device_reranker}")

        self.df, self.corpus_embeddings, self.embedding_meta = load_project_assets(settings.data_dir)
        print(f"Loaded {len(self.df):,} fatwas · embeddings {tuple(self.corpus_embeddings.shape)}")

        print(f"Loading embedder ({EMBEDDER_MODEL_NAME}) on {settings.device_embedder}...")
        self.embedder = SentenceTransformer(EMBEDDER_MODEL_NAME, device=settings.device_embedder)
        self.embedder.max_seq_length = EMBEDDER_MAX_SEQ_LENGTH

        print(f"Loading reranker ({RERANKER_MODEL_NAME}) on {settings.device_reranker}...")
        self.reranker = CrossEncoder(
            RERANKER_MODEL_NAME,
            max_length=RERANKER_MAX_LENGTH,
            device=settings.device_reranker,
            trust_remote_code=True,
        )
        print("Models loaded.")

        if settings.warmup:
            self._warmup()

    # -- clients -----------------------------------------------------------

    def ensure_gemini(self):
        if self._gemini_client is None:
            from google import genai

            api_key = os.getenv(GEMINI_API_KEY_ENV)
            if not api_key:
                raise RuntimeError(
                    f"{GEMINI_API_KEY_ENV} is not set. Put it in .env or export it."
                )
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

    def ensure_openrouter(self) -> str:
        if self._openrouter_key is None:
            self._openrouter_key = get_openrouter_key()
        return self._openrouter_key

    # -- rewrite -----------------------------------------------------------

    def rewrite_query(self, provider: str, model: str, query: str) -> tuple[QueryRewrite, dict[str, Any]]:
        if provider == PROVIDER_GEMINI:
            client = self.ensure_gemini()
            response = client.models.generate_content(
                model=model,
                contents=f"{REWRITE_SYSTEM}\n\n{rewrite_user_prompt(query)}",
                config={
                    "response_mime_type": "application/json",
                    "response_schema": QueryRewrite.model_json_schema(),
                },
            )
            um = getattr(response, "usage_metadata", None)
            usage = usage_dict(
                getattr(um, "prompt_token_count", 0),
                getattr(um, "candidates_token_count", 0),
                getattr(um, "total_token_count", None),
            )
            return QueryRewrite.model_validate_json(response.text), usage

        key = self.ensure_openrouter()
        data, usage = openrouter_chat(key, model, REWRITE_SYSTEM, rewrite_user_prompt(query))
        return QueryRewrite.model_validate(data), usage

    # -- judging -----------------------------------------------------------

    def judge_candidates(
        self, provider: str, model: str, user_query: str, candidates: pd.DataFrame, max_chars: int
    ) -> tuple[dict[int, FatwaJudgement], dict[str, Any]]:
        if provider == PROVIDER_GEMINI:
            return self._judge_gemini(model, user_query, candidates, max_chars)
        key = self.ensure_openrouter()
        if is_jev_model(model):
            return self._judge_jev(key, model, user_query, candidates, max_chars)
        return self._judge_openrouter_chat(key, model, user_query, candidates, max_chars)

    def _judge_gemini(self, model, user_query, candidates, max_chars):
        client = self.ensure_gemini()
        response = client.models.generate_content(
            model=model,
            contents=f"{judge_system_prompt()}\n\n{judge_user_prompt(user_query, candidates, max_chars)}",
            config={
                "response_mime_type": "application/json",
                "response_schema": JudgeResponse.model_json_schema(),
            },
        )
        um = getattr(response, "usage_metadata", None)
        usage = usage_dict(
            getattr(um, "prompt_token_count", 0),
            getattr(um, "candidates_token_count", 0),
            getattr(um, "total_token_count", None),
        )
        parsed = JudgeResponse.model_validate_json(response.text)
        out = {
            int(j.candidate_id): FatwaJudgement(
                verdict=j.verdict, confidence=j.confidence, answer=j.answer, explanation=j.explanation
            )
            for j in parsed.judgements
        }
        return out, usage

    def _judge_openrouter_chat(self, key, model, user_query, candidates, max_chars):
        data, usage = openrouter_chat(
            key, model, judge_system_prompt(), judge_user_prompt(user_query, candidates, max_chars)
        )
        parsed = JudgeResponse.model_validate(data)
        out = {
            int(j.candidate_id): FatwaJudgement(
                verdict=j.verdict, confidence=j.confidence, answer=j.answer, explanation=j.explanation
            )
            for j in parsed.judgements
        }
        return out, usage

    def _judge_jev(self, key, model, user_query, candidates, max_chars):
        state: dict[str, Any] = {"user_query": (user_query or "").strip(), "candidates": {}}
        questions: dict[str, Any] = {}
        ids: list[int] = []
        for idx, row in enumerate(candidates.itertuples(index=False)):
            qid = f"cand_{idx}"
            ids.append(int(row.fatwa_id))
            state["candidates"][qid] = {
                "title": str(row.title)[:max_chars],
                "question": str(row.question)[:max_chars],
                "answer": str(row.answer)[:max_chars],
            }
            questions[qid] = {
                "type": "choice",
                "instructions": (
                    f"Judge candidate `candidates.{qid}`. Decide whether its fatwa answers the "
                    f"user's question in `user_query`. Use only the candidate's own title, question, "
                    f"and answer; do not rely on outside knowledge. Pick exactly one option."
                ),
                "criteria": dict(JEV_CRITERIA),
            }

        answers, usage = call_jev(key, model, state, questions)
        out: dict[int, FatwaJudgement] = {}
        for idx, fatwa_id in enumerate(ids):
            answer = answers.get(f"cand_{idx}", {}) or {}
            choice = answer.get("choice")
            confidence = answer.get("confidence")
            probabilities = answer.get("probabilities") or {}
            if not isinstance(choice, str) or choice not in VERDICTS:
                choice = choice or "unknown"
            try:
                confidence = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence = None
            probabilities = {
                str(k): float(v) for k, v in probabilities.items() if isinstance(v, (int, float))
            }
            out[fatwa_id] = FatwaJudgement(
                verdict=choice, confidence=confidence, probabilities=probabilities
            )
        return out, usage

    # -- retrieval / reranking --------------------------------------------

    @functools.lru_cache(maxsize=256)
    def _encode_query_cached(self, query: str) -> torch.Tensor:
        vec = self.embedder.encode(
            query,
            convert_to_tensor=True,
            device=self.s.device_embedder,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.detach().cpu().float()

    def retrieve_top_k(self, query: str, top_k: int) -> pd.DataFrame:
        top_k = max(1, min(int(top_k), MAX_RETRIEVE_K, len(self.df)))
        scores = torch.mv(self.corpus_embeddings, self._encode_query_cached(query))
        values, indices = torch.topk(scores, k=top_k)
        out = self.df.iloc[indices.numpy()].copy().reset_index(drop=True)
        out.insert(0, "retrieval_rank", np.arange(1, len(out) + 1))
        out["embedding_score"] = values.numpy()
        return out

    def rerank_candidates(self, query: str, candidates: pd.DataFrame) -> pd.DataFrame:
        if not len(candidates):
            return candidates
        pairs = [[query, text] for text in candidates["embed_text"].astype(str).tolist()]
        scores = np.asarray(
            self.reranker.predict(pairs, batch_size=RERANKER_BATCH_SIZE, show_progress_bar=False)
        ).reshape(-1)
        reranked = candidates.copy()
        reranked["reranker_score"] = scores
        reranked = reranked.sort_values("reranker_score", ascending=False).reset_index(drop=True)
        reranked.insert(0, "rerank_rank", np.arange(1, len(reranked) + 1))
        return reranked

    def _warmup(self):
        try:
            t0 = time.perf_counter()
            dummy = "ما هي كفارة اليمين؟"
            self.rerank_candidates(dummy, self.retrieve_top_k(dummy, DEFAULT_SHOW_N))
            print(f"Warmup done in {time.perf_counter() - t0:.2f}s")
        except Exception as e:  # pragma: no cover
            print(f"Warmup skipped: {type(e).__name__}: {e}")

    # -- main search -------------------------------------------------------

    def run_search(
        self,
        user_query: str,
        rewrite_enabled: bool,
        rewrite_provider: str,
        rewrite_model: str,
        judge_enabled: bool,
        judge_provider: str,
        judge_model: str,
        retrieve_k: int,
        rerank_pool: int,
        show_n: int,
        max_candidate_chars: int,
        min_rerank_score: float | None,
        filter_verdicts: list[str],
    ):
        steps: list[Stat] = []
        errors: list[str] = []
        t_start = time.perf_counter()

        user_query = (user_query or "").strip()
        if not user_query:
            return "Please enter a question.", "", pd.DataFrame(), pd.DataFrame(), "", ""

        # -- rewrite
        rewrite = None
        search_query = user_query
        if rewrite_enabled:
            t0 = time.perf_counter()
            try:
                rewrite, usage = self.rewrite_query(rewrite_provider, rewrite_model, user_query)
                search_query = rewrite.improved_query.strip() or user_query
                steps.append(
                    Stat("Query rewrite", time.perf_counter() - t0, **usage,
                         provider=rewrite_provider, model=rewrite_model)
                )
            except Exception as e:
                dt = time.perf_counter() - t0
                msg = f"Query rewrite failed ({rewrite_provider} / {rewrite_model}): {type(e).__name__}: {e}"
                errors.append(msg)
                steps.append(Stat("Query rewrite", dt, provider=rewrite_provider, model=rewrite_model, note="failed"))
        else:
            steps.append(Stat("Query rewrite", 0.0, note="disabled"))

        # -- retrieval
        t0 = time.perf_counter()
        try:
            retrieved = self.retrieve_top_k(search_query, int(retrieve_k))
            steps.append(Stat("Embedding retrieval", time.perf_counter() - t0, note=f"{len(retrieved)} candidates"))
        except Exception as e:
            err = positional_error(e)
            return self._fatal(user_query, steps, errors + [f"Retrieval failed: {e}"], err)

        # -- rerank
        t0 = time.perf_counter()
        try:
            pool = min(int(rerank_pool), len(retrieved))
            reranked = self.rerank_candidates(search_query, retrieved.head(pool))
            if min_rerank_score is not None:
                reranked = reranked[reranked["reranker_score"] >= float(min_rerank_score)].reset_index(drop=True)
                reranked["rerank_rank"] = np.arange(1, len(reranked) + 1)
            steps.append(Stat("Mizan rerank", time.perf_counter() - t0, note=f"over {pool}"))
        except Exception as e:
            err = positional_error(e)
            return self._fatal(user_query, steps, errors + [f"Rerank failed: {e}"], err)

        top_n = reranked.head(max(1, int(show_n))).copy()

        # -- judge
        judge_map: dict[int, FatwaJudgement] = {}
        if judge_enabled and len(top_n):
            t0 = time.perf_counter()
            try:
                judge_map, usage = self.judge_candidates(
                    judge_provider, judge_model, user_query, top_n, int(max_candidate_chars)
                )
                steps.append(
                    Stat("Judge", time.perf_counter() - t0, **usage,
                         provider=judge_provider, model=judge_model, note=f"{len(judge_map)} judged")
                )
            except Exception as e:
                dt = time.perf_counter() - t0
                msg = f"Judge failed ({judge_provider} / {judge_model}): {type(e).__name__}: {e}"
                errors.append(msg)
                steps.append(Stat("Judge", dt, provider=judge_provider, model=judge_model, note="failed"))
        else:
            steps.append(Stat("Judge", 0.0, note="disabled" if not judge_enabled else "no candidates"))

        total_seconds = time.perf_counter() - t_start
        steps.append(Stat("Total", total_seconds, note=""))

        # -- verdict filtering
        if judge_enabled and judge_map and filter_verdicts:
            allowed = set(filter_verdicts)
            keep = top_n["fatwa_id"].map(
                lambda fid: (judge_map[int(fid)].verdict in allowed) if int(fid) in judge_map else True
            )
            display_rows = top_n[keep].reset_index(drop=True)
        else:
            display_rows = top_n

        query_info = self._query_panel(user_query, search_query, rewrite, steps, errors, judge_map, judge_enabled)
        stats_html = self._stats_html(steps)
        retrieval_view = dataframe_for_display(retrieved)
        rerank_view = dataframe_for_display(top_n, judge_map=judge_map if judge_enabled else None)

        if len(display_rows):
            cards = "\n".join(
                fatwa_card(row, judge_map.get(int(row.fatwa_id)) if judge_enabled else None, int(row["rerank_rank"]))
                for _, row in display_rows.iterrows()
            )
        else:
            cards = '<p class="empty">No candidates match the selected verdict filter.</p>'

        error_text = "\n\n".join(errors)
        if errors:
            gr.Warning(error_modal_html(errors, user_query, steps), duration=None, title="Error — copy and send to the AI")

        return query_info, stats_html, retrieval_view, rerank_view, cards, error_text

    def _fatal(self, user_query, steps, errors, err):
        steps.append(Stat("Total", sum(s.seconds for s in steps if s.label != "Total"), note="aborted"))
        tb = traceback.format_exc() if err is None else err
        gr.Warning(error_modal_html(errors, user_query, steps, tb), duration=None, title="Error — copy and send to the AI")
        query_info = self._query_panel(user_query, user_query, None, steps, errors, {}, False)
        return query_info, self._stats_html(steps), pd.DataFrame(), pd.DataFrame(), "", "\n\n".join(errors) + "\n\n" + (tb or "")

    # -- panels ------------------------------------------------------------

    def _query_panel(self, user_query, search_query, rewrite, steps, errors, judge_map, judge_enabled):
        return render_query_panel(user_query, search_query, rewrite, errors, judge_map, judge_enabled)

    def _stats_html(self, steps: list[Stat]) -> str:
        return render_pipeline(steps)


@dataclass
class Settings:
    data_dir: Path
    device_embedder: str
    device_reranker: str
    warmup: bool = True


def positional_error(e: Exception) -> str:
    return traceback.format_exc()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def esc(value: Any) -> str:
    value = "" if value is None else str(value)
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


# ---------------------------------------------------------------------------
# Presentation: small HTML renderers
# ---------------------------------------------------------------------------

VERDICT_LABELS = {"answers-it": "Answers it", "related-only": "Related only", "unrelated": "Unrelated"}
VERDICT_ICONS = {"answers-it": "✓", "related-only": "≈", "unrelated": "✕"}
VERDICT_CLASSES = {"answers-it": "good", "related-only": "warn", "unrelated": "bad"}

COPY_JS = (
    "(function(b){var t=b.closest('.fq-card').querySelector('.fq-answer-text').innerText;"
    "navigator.clipboard.writeText(t).then(function(){var o=b.textContent;b.textContent='Copied ✓';"
    "setTimeout(function(){b.textContent=o},1400)})})(this)"
)

EMPTY_QUERY = ""
EMPTY_RESULTS = """
<div class="fq-empty">
  <svg viewBox="0 0 48 48" class="fq-empty-mark" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="1.6">
    <rect x="11" y="11" width="26" height="26" rx="1.5"/><rect x="11" y="11" width="26" height="26" rx="1.5" transform="rotate(45 24 24)"/>
    <circle cx="24" cy="24" r="4.5"/></g></svg>
  <div class="fq-empty-title">Ask a question to begin</div>
  <div class="fq-empty-text">Write in Arabic or any language. Matching fatwas appear here, ranked by relevance.</div>
</div>
"""
EMPTY_FAILED = """
<div class="fq-empty">
  <div class="fq-empty-title">The search did not complete</div>
  <div class="fq-empty-text">Open the <b>Debug</b> tab for the error details and traceback.</div>
</div>
"""
EMPTY_PIPELINE = '<div class="fq-empty-sm">Run a search to see step timings and token usage.</div>'
EMPTY_TABLE = '<div class="fq-empty-sm">Nothing here yet.</div>'
LOADING_QUERY = '<div class="fq-loading"><span class="fq-spin"></span>Searching the corpus…</div>'
LOADING_RESULTS = (
    '<div class="fq-skel">' + '<div class="fq-skel-card"><i></i><i></i><i></i><i></i></div>' * 3 + "</div>"
)


def verdict_badge_class(verdict: str) -> str:
    return VERDICT_CLASSES.get(verdict, "neutral")


def verdict_badge(verdict: str) -> str:
    label = VERDICT_LABELS.get(verdict, verdict or "unknown")
    icon = VERDICT_ICONS.get(verdict, "?")
    return (
        f'<span class="fq-badge {verdict_badge_class(verdict)}" title="{esc(verdict)}">'
        f"<b>{icon}</b>{esc(label)}</span>"
    )


def _safe_url(url: Any) -> str:
    url = str(url or "").strip()
    return url if re.match(r"^https?://", url, re.IGNORECASE) else ""


def _fmt_conf(conf: Any) -> str:
    if isinstance(conf, (int, float)):
        return f"{conf:.2f}"
    return esc(conf) if conf else "—"


def render_query_panel(user_query, search_query, rewrite, errors, judge_map, judge_enabled) -> str:
    rows = [
        '<div class="fq-qrow"><span class="fq-qlabel">You asked</span>'
        f'<span class="fq-qtext" dir="auto">{esc(user_query)}</span></div>'
    ]
    if (search_query or "").strip() != (user_query or "").strip():
        rows.append(
            '<div class="fq-qrow"><span class="fq-qlabel">Searched as</span>'
            f'<span class="fq-qtext fq-ar" dir="auto">{esc(search_query)}</span></div>'
        )

    chips = []
    if rewrite is not None:
        chips.append(f'<span class="fq-chip">{esc(rewrite.language)}</span>')
        chips.append(f'<span class="fq-chip">{"rewritten" if rewrite.changed else "kept as is"}</span>')
    if judge_enabled and judge_map:
        counts: dict[str, int] = {}
        for j in judge_map.values():
            counts[j.verdict] = counts.get(j.verdict, 0) + 1
        for v in VERDICTS:
            if counts.get(v):
                chips.append(
                    f'<span class="fq-pill {verdict_badge_class(v)}">'
                    f"{VERDICT_ICONS[v]} {counts[v]} · {VERDICT_LABELS[v]}</span>"
                )
    if chips:
        rows.append('<div class="fq-qchips">' + "".join(chips) + "</div>")

    if errors:
        items = "".join(f"<li>{esc(e)}</li>" for e in errors)
        rows.append(
            '<div class="fq-alert" role="alert">'
            f'<div class="fq-alert-title">⚠ {len(errors)} issue{"s" if len(errors) != 1 else ""}</div>'
            f"<ul>{items}</ul>"
            '<div class="fq-alert-hint">Full traceback is in the <b>Debug</b> tab.</div></div>'
        )
    return '<section class="fq-query-panel">' + "".join(rows) + "</section>"


def render_pipeline(steps: list[Stat]) -> str:
    real = [s for s in steps if s.label != "Total"]
    total = next((s for s in steps if s.label == "Total"), None)
    total_s = total.seconds if total else sum(s.seconds for s in real)
    denom = total_s if total_s > 0 else 1.0
    tok_in = sum(s.prompt for s in real)
    tok_out = sum(s.completion for s in real)
    cost = sum(s.cost for s in real)
    aborted = bool(total and total.note == "aborted")

    kpis = (
        f'<div class="fq-kpi"><div class="fq-kpi-label">Total time</div>'
        f'<div class="fq-kpi-value">{total_s:.2f}<small>s</small></div>'
        f'<div class="fq-kpi-sub">{"aborted" if aborted else "end to end"}</div></div>'
        f'<div class="fq-kpi"><div class="fq-kpi-label">Tokens</div>'
        f'<div class="fq-kpi-value">{tok_in + tok_out:,}</div>'
        f'<div class="fq-kpi-sub">{tok_in:,} in · {tok_out:,} out</div></div>'
        f'<div class="fq-kpi"><div class="fq-kpi-label">Est. cost</div>'
        f'<div class="fq-kpi-value">{f"${cost:.6f}" if cost > 0 else "—"}</div>'
        f'<div class="fq-kpi-sub">LLM calls only</div></div>'
    )

    items = []
    cursor = 0.0
    for i, s in enumerate(real, 1):
        state = "fail" if s.note == "failed" else "off" if s.note in ("disabled", "no candidates") else "ok"
        left = cursor / denom * 100
        width = max(s.seconds / denom * 100, 0.8) if s.seconds > 0 else 0
        cursor += s.seconds
        meta = []
        if s.model:
            meta.append(f'<span class="fq-chip mono">{esc(s.provider)} · {esc(s.model)}</span>')
        if s.prompt or s.completion:
            meta.append(f'<span class="fq-chip mono">{s.prompt:,} in · {s.completion:,} out</span>')
        if s.cost > 0:
            meta.append(f'<span class="fq-chip mono">${s.cost:.6f}</span>')
        if s.note:
            meta.append(f'<span class="fq-chip note {state}">{esc(s.note)}</span>')
        items.append(
            f'<li class="fq-step {state}">'
            f'<div class="fq-step-top"><span class="fq-step-idx">{i}</span>'
            f'<span class="fq-step-name">{esc(s.label)}</span>'
            f'<span class="fq-step-time">{s.seconds:.2f}s</span></div>'
            f'<div class="fq-bar"><span style="left:{left:.2f}%;width:{width:.2f}%"></span></div>'
            f'<div class="fq-step-meta">{"".join(meta)}</div></li>'
        )
    return (
        '<section class="fq-pipeline">'
        f'<div class="fq-kpis">{kpis}</div><ol class="fq-steps">{"".join(items)}</ol></section>'
    )


def error_modal_html(errors: list[str], user_query: str, steps: list[Stat], tb: str | None = None) -> str:
    err_items = "".join(f"<li>{esc(e)}</li>" for e in errors)
    stat_lines = "".join(f"<li>{esc(s.label)} — {s.seconds:.2f}s {esc(s.note)}</li>" for s in steps)
    tb_html = f"<div class='fq-toast-sub'>Traceback</div><pre class='tb'>{esc(tb)}</pre>" if tb else ""
    return (
        "<div class='fq-toast'>"
        "<div class='fq-toast-title'>Something went wrong</div>"
        f"<div class='fq-toast-q'>{esc(user_query)}</div>"
        f"<div class='fq-toast-sub'>Errors</div><ul>{err_items}</ul>"
        f"<div class='fq-toast-sub'>Timings</div><ul>{stat_lines}</ul>{tb_html}"
        "</div>"
    )


def judge_box(judge: FatwaJudgement) -> str:
    cls = verdict_badge_class(judge.verdict)
    body = ""
    if judge.answer:
        body += (
            '<div class="fq-label">Grounded answer</div>'
            f'<div class="fq-ar fq-judge-answer" dir="rtl">{esc(judge.answer)}</div>'
        )
    if judge.explanation:
        body += (
            '<div class="fq-label">Why</div>'
            f'<div class="fq-ar fq-judge-why" dir="rtl">{esc(judge.explanation)}</div>'
        )
    if judge.probabilities:
        bars = "".join(
            f'<div class="fq-prob"><span class="fq-prob-l">{esc(k)}</span>'
            f'<span class="fq-prob-bar"><span style="width:{max(0.0, min(1.0, v)) * 100:.0f}%"></span></span>'
            f'<span class="fq-prob-v">{v:.2f}</span></div>'
            for k, v in sorted(judge.probabilities.items(), key=lambda kv: -kv[1])
        )
        body += f'<details class="fq-mini"><summary>Probabilities</summary>{bars}</details>'
    return (
        f'<section class="fq-judge {cls}">'
        f'<div class="fq-judge-head">{verdict_badge(judge.verdict)}'
        f'<span class="fq-conf">confidence {_fmt_conf(judge.confidence)}</span></div>{body}</section>'
    )


def fatwa_card(row: pd.Series, judge: FatwaJudgement | None = None, rank: int = 0) -> str:
    categories = row.get("categories_list", []) or []
    tags = "".join(f'<span class="fq-cat fq-ar">{esc(cat)}</span>' for cat in categories)

    link = _safe_url(row.get("links", ""))
    link_html = (
        f'<a class="fq-btn primary" href="{esc(link)}" target="_blank" rel="noopener noreferrer">Original ↗</a>'
        if link
        else ""
    )

    r_rank, e_rank = int(row["rerank_rank"]), int(row["retrieval_rank"])
    delta = e_rank - r_rank
    move = ""
    if delta:
        move = f'<span class="fq-m move {"up" if delta > 0 else "down"}">{"▲" if delta > 0 else "▼"} {abs(delta)}</span>'

    edge = f" v-{verdict_badge_class(judge.verdict)}" if judge is not None else ""
    top = " top" if rank == 1 else ""
    best = '<span class="fq-best">★ Best match</span>' if rank == 1 else ""
    judge_html = judge_box(judge) if judge is not None else ""
    open_attr = " open" if rank == 1 else ""

    return f"""
    <article class="fq-card{top}{edge}">
      <header class="fq-card-head">
        <div class="fq-rank">{r_rank}</div>
        <div class="fq-card-main">
          <h3 class="fq-card-title fq-ar" dir="rtl">{esc(row['title'])}</h3>
          <div class="fq-card-meta">{best}<span class="fq-source">{esc(row['source'])}</span>{tags}</div>
        </div>
      </header>
      {judge_html}
      <div class="fq-qblock"><div class="fq-label">Question</div>
        <div class="fq-ar fq-question" dir="rtl">{esc(row['question'])}</div></div>
      <details class="fq-det"{open_attr}>
        <summary><span>Full answer</span></summary>
        <div class="fq-ar fq-answer-text" dir="rtl">{esc(row['answer'])}</div>
      </details>
      <div class="fq-card-foot">
        <div class="fq-metrics">
          <span class="fq-m" title="Fatwa id">id {int(row['fatwa_id'])}</span>
          <span class="fq-m" title="Embedding cosine score and retrieval rank">embed {float(row['embedding_score']):.3f} <i>#{e_rank}</i></span>
          <span class="fq-m" title="Reranker score">rerank {float(row['reranker_score']):.3f}</span>{move}
        </div>
        <div class="fq-actions">
          <button type="button" class="fq-btn" onclick="{COPY_JS}">Copy answer</button>{link_html}
        </div>
      </div>
    </article>
    """


def dataframe_for_display(rows: pd.DataFrame, judge_map: dict[int, FatwaJudgement] | None = None) -> pd.DataFrame:
    if not len(rows):
        return pd.DataFrame()
    cols = ["retrieval_rank", "embedding_score"]
    rename = {"retrieval_rank": "embed rank", "embedding_score": "embedding score"}
    if "rerank_rank" in rows.columns:
        cols += ["rerank_rank", "reranker_score"]
        rename.update({"rerank_rank": "rerank rank", "reranker_score": "reranker score"})
    cols += ["fatwa_id", "source", "title"]
    view = rows[cols].copy().rename(columns=rename)
    if judge_map:
        view["verdict"] = view["fatwa_id"].map(
            lambda x: judge_map[int(x)].verdict if int(x) in judge_map else ""
        )
    return view


def df_to_html(df: Any) -> str:
    """Debug tables: rank movement, relative score bars, verdict badges, RTL titles."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return EMPTY_TABLE

    ranges = {}
    for c in ("embedding score", "reranker score"):
        if c in df.columns:
            lo, hi = float(df[c].min()), float(df[c].max())
            ranges[c] = (lo, hi)

    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            if c == "embed rank":
                cell = f'<span class="fq-rk">#{int(v)}</span>'
            elif c == "rerank rank":
                mv = ""
                if "embed rank" in df.columns:
                    d = int(r["embed rank"]) - int(v)
                    if d:
                        mv = f'<span class="fq-m move {"up" if d > 0 else "down"}">{"▲" if d > 0 else "▼"}{abs(d)}</span>'
                cell = f'<span class="fq-rk strong">#{int(v)}</span>{mv}'
            elif c in ranges:
                lo, hi = ranges[c]
                pct = 100.0 if hi == lo else 8 + 92 * (float(v) - lo) / (hi - lo)
                cell = f'<span class="fq-sc"><b>{float(v):.4f}</b><i><span style="width:{pct:.0f}%"></span></i></span>'
            elif c == "fatwa_id":
                cell = f'<span class="fq-mono-dim">{int(v)}</span>'
            elif c == "title":
                cell = f'<span class="fq-ar fq-ttl" dir="rtl" title="{esc(v)}">{esc(v)}</span>'
            elif c == "verdict":
                cell = verdict_badge(str(v)) if v else '<span class="fq-mono-dim">—</span>'
            else:
                cell = esc(v)
            tds.append(f"<td>{cell}</td>")
        body.append(f"<tr>{''.join(tds)}</tr>")
    return (
        '<div class="fq-table-wrap"><table class="fq-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


# ---------------------------------------------------------------------------
# Theme + CSS
# ---------------------------------------------------------------------------

_GREEN = gr.themes.Color(
    name="fq_green",
    c50="#eef7f3", c100="#d8ece4", c200="#b2d9c9", c300="#82c0aa", c400="#51a28a",
    c500="#2e8670", c600="#1d6c5a", c700="#17594a", c800="#13483c", c900="#0f3a31", c950="#08231d",
)

THEME = gr.themes.Base(
    primary_hue=_GREEN,
    secondary_hue=_GREEN,
    neutral_hue=gr.themes.colors.stone,
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.Font("Manrope"), gr.themes.Font("IBM Plex Sans Arabic"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.Font("JetBrains Mono"), "ui-monospace", "Consolas", "monospace"],
).set(
    body_background_fill="var(--fq-bg)",
    body_background_fill_dark="var(--fq-bg)",
    button_primary_background_fill="var(--fq-accent)",
    button_primary_background_fill_hover="var(--fq-accent-ink)",
    button_primary_background_fill_dark="var(--fq-accent)",
    button_primary_background_fill_hover_dark="var(--fq-accent-ink)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#06130f",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=IBM+Plex+Sans+Arabic:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --fq-bg:#f4f1e8; --fq-surface:#fffdf8; --fq-surface-2:#f1ede1; --fq-ink:#1b2724; --fq-muted:#6b7570;
  --fq-line:#e3dccb; --fq-accent:#17715c; --fq-accent-ink:#0f5444; --fq-accent-soft:#e1efe9;
  --fq-gold:#b3832a; --fq-gold-soft:#f6ebcc;
  --fq-good:#17794f; --fq-good-soft:#dcf1e6; --fq-warn:#94650a; --fq-warn-soft:#faedc9;
  --fq-bad:#b03a3a; --fq-bad-soft:#f7dfdb; --fq-neutral:#5f6a66; --fq-neutral-soft:#e9e6dc;
  --fq-shadow:0 1px 2px rgba(50,38,10,.05), 0 10px 28px -14px rgba(50,38,10,.16);
  --fq-ar:'Amiri','Noto Naskh Arabic','Traditional Arabic','Geeza Pro',serif;
  --fq-sans:'Manrope','IBM Plex Sans Arabic',system-ui,sans-serif;
  --fq-mono:'JetBrains Mono',ui-monospace,Consolas,monospace;
}
.dark {
  --fq-bg:#0d1412; --fq-surface:#151f1c; --fq-surface-2:#1b2824; --fq-ink:#e7eeeb; --fq-muted:#8fa09a;
  --fq-line:#26352f; --fq-accent:#3dc3a0; --fq-accent-ink:#6fdbbf; --fq-accent-soft:#143229;
  --fq-gold:#e2b658; --fq-gold-soft:#34290f;
  --fq-good:#58d69a; --fq-good-soft:#12301f; --fq-warn:#e8bb55; --fq-warn-soft:#352a0c;
  --fq-bad:#f08a80; --fq-bad-soft:#3a1816; --fq-neutral:#a2b0aa; --fq-neutral-soft:#222e2a;
  --fq-shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 28px -14px rgba(0,0,0,.6);
}

body, .gradio-container { background:var(--fq-bg) !important; font-family:var(--fq-sans) !important; color:var(--fq-ink); }
.gradio-container { max-width:1040px !important; margin:0 auto !important; padding-top:10px !important; }
footer { display:none !important; }
.gradio-container .block, .gradio-container .form { box-shadow:none; }
.fq-ar { font-family:var(--fq-ar) !important; }

/* ---- top bar ---- */
#fq-top { align-items:center !important; gap:12px; margin:6px 0 4px; flex-wrap:nowrap !important; }
#fq-top > div { min-width:0 !important; }
.fq-brand { display:flex; align-items:center; gap:14px; }
.fq-mark { width:44px; height:44px; color:var(--fq-accent); flex:none; }
.fq-mark .gold { fill:var(--fq-gold); stroke:none; }
.fq-name { font-weight:800; font-size:1.45rem; letter-spacing:-.02em; line-height:1.1; color:var(--fq-ink); }
.fq-tagline { font-size:.82rem; color:var(--fq-muted); margin-top:3px; }
#fq-settings-btn { flex:none !important; width:auto !important; min-width:0 !important; }
#fq-settings-btn { border-radius:999px; background:var(--fq-surface); border:1px solid var(--fq-line);
  color:var(--fq-ink) !important; font-weight:600; padding:9px 18px !important; font-size:.9rem !important; box-shadow:var(--fq-shadow); }
#fq-settings-btn:hover { border-color:var(--fq-accent); color:var(--fq-accent); }

/* ---- search ---- */
#fq-search-wrap { background:var(--fq-surface); border:1px solid var(--fq-line); border-radius:20px;
  padding:6px; box-shadow:var(--fq-shadow); margin-top:10px; }
#fq-query, #fq-query *:not(button):not(svg):not(path) { background:transparent !important; border:none !important; box-shadow:none !important; }
#fq-search-wrap, #fq-search-wrap .styler { --block-background-fill:transparent; }
#fq-search-wrap .styler { background:transparent !important; }
#fq-query textarea { font-family:var(--fq-ar) !important; font-size:1.35rem !important; line-height:1.7 !important;
  padding:12px 16px !important; background:transparent !important; border:none !important; box-shadow:none !important; }
#fq-query textarea::placeholder { font-family:var(--fq-sans); font-size:1rem; color:var(--fq-muted); }
#fq-query button { background:var(--fq-accent) !important; color:#fff !important; border-radius:14px !important; }
#fq-examples { gap:8px !important; flex-wrap:wrap !important; margin:12px 2px 0; align-items:center; }
#fq-examples > * { min-width:0 !important; flex:0 0 auto !important; width:auto !important; }
.fq-ex-label { font-size:.78rem; color:var(--fq-muted); text-transform:uppercase; letter-spacing:.08em; font-weight:700; padding-right:4px; }
.fq-ex { border-radius:999px !important; background:var(--fq-surface) !important; border:1px solid var(--fq-line) !important;
  color:var(--fq-ink) !important; font-weight:500 !important; padding:6px 14px !important; font-size:.88rem !important; box-shadow:none !important; }
.fq-ex:hover { border-color:var(--fq-accent) !important; background:var(--fq-accent-soft) !important; }

/* ---- generic ---- */
.fq-flat, .fq-flat.block { border:none !important; background:transparent !important; box-shadow:none !important; padding:0 !important; }
.fq-chip { display:inline-block; padding:2px 10px; border-radius:999px; background:var(--fq-surface-2); color:var(--fq-muted);
  font-size:.76rem; font-weight:600; border:1px solid var(--fq-line); }
.fq-chip.mono { font-family:var(--fq-mono); font-weight:400; font-size:.72rem; }
.fq-chip.note.fail { background:var(--fq-bad-soft); color:var(--fq-bad); border-color:transparent; }
.fq-chip.note.off { opacity:.8; }
.fq-pill { display:inline-block; padding:3px 11px; border-radius:999px; font-size:.78rem; font-weight:700; }
.fq-pill.good,.fq-badge.good { background:var(--fq-good-soft); color:var(--fq-good); }
.fq-pill.warn,.fq-badge.warn { background:var(--fq-warn-soft); color:var(--fq-warn); }
.fq-pill.bad,.fq-badge.bad { background:var(--fq-bad-soft); color:var(--fq-bad); }
.fq-pill.neutral,.fq-badge.neutral { background:var(--fq-neutral-soft); color:var(--fq-neutral); }
.fq-badge { display:inline-flex; align-items:center; gap:6px; padding:3px 12px 3px 9px; border-radius:999px; font-size:.82rem; font-weight:700; }
.fq-badge b { font-size:.9rem; }
.fq-label { font-size:.7rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; color:var(--fq-muted); margin:0 0 5px; }
.fq-empty { text-align:center; padding:56px 20px; color:var(--fq-muted); }
.fq-empty-mark { display:block; width:58px; height:58px; color:var(--fq-line); margin:0 auto 10px; }
.fq-empty-title { font-weight:700; font-size:1.05rem; color:var(--fq-ink); margin-bottom:4px; }
.fq-empty-text { font-size:.92rem; max-width:420px; margin:0 auto; }
.fq-empty-sm { padding:22px; text-align:center; color:var(--fq-muted); font-size:.9rem; }
.fq-note { padding:12px 16px; border-radius:14px; background:var(--fq-gold-soft); color:var(--fq-ink); font-size:.92rem; margin-top:12px; }

/* ---- query panel ---- */
.fq-query-panel { margin:16px 2px 6px; padding:14px 18px; border-radius:16px; background:var(--fq-surface-2); border:1px solid var(--fq-line); }
.fq-qrow { display:flex; gap:12px; align-items:baseline; padding:3px 0; }
.fq-qlabel { flex:none; width:92px; font-size:.7rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; color:var(--fq-muted); }
.fq-qtext { font-size:1rem; font-weight:500; }
.fq-qtext.fq-ar { font-size:1.3rem; }
.fq-qchips { display:flex; gap:8px; flex-wrap:wrap; margin:9px 0 0 104px; }
.fq-alert { margin-top:12px; padding:12px 16px; border-radius:12px; background:var(--fq-bad-soft); color:var(--fq-bad); border:1px solid color-mix(in srgb, var(--fq-bad) 30%, transparent); }
.fq-alert-title { font-weight:800; }
.fq-alert ul { margin:6px 0 4px 18px; font-size:.88rem; word-break:break-word; }
.fq-alert-hint { font-size:.8rem; opacity:.85; }
.fq-loading { display:flex; align-items:center; gap:12px; margin:16px 2px 6px; padding:14px 18px; border-radius:16px;
  background:var(--fq-surface-2); border:1px solid var(--fq-line); color:var(--fq-muted); font-weight:600; }
.fq-spin { width:16px; height:16px; border-radius:50%; border:2.5px solid var(--fq-line); border-top-color:var(--fq-accent); animation:fq-rot .8s linear infinite; }
@keyframes fq-rot { to { transform:rotate(360deg); } }

/* ---- tabs ---- */
#fq-tabs { margin-top:14px; }
#fq-tabs > .tab-wrapper, #fq-tabs .tab-container { border-color:var(--fq-line) !important; }
#fq-tabs button[role=tab] { font-weight:700; font-size:.95rem; }
#fq-tabs button[role=tab].selected { color:var(--fq-accent) !important; }
#fq-tabs button[role=tab].selected::after { background:var(--fq-accent) !important; }

/* ---- result cards ---- */
.fq-card { position:relative; background:var(--fq-surface); border:1px solid var(--fq-line); border-radius:20px;
  padding:20px 24px 16px; margin:14px 0; box-shadow:var(--fq-shadow); --edge:var(--fq-accent); }
.fq-card::before { content:""; position:absolute; left:0; top:22px; bottom:22px; width:4px; border-radius:0 4px 4px 0; background:var(--edge); }
.fq-card.v-good { --edge:var(--fq-good); } .fq-card.v-warn { --edge:var(--fq-warn); } .fq-card.v-bad { --edge:var(--fq-bad); opacity:.92; }
.fq-card.top { border-color:color-mix(in srgb, var(--fq-gold) 55%, var(--fq-line)); }
.fq-card-head { display:flex; gap:16px; align-items:flex-start; }
.fq-rank { flex:none; width:38px; height:38px; border-radius:12px; background:var(--fq-accent-soft); color:var(--fq-accent-ink);
  display:grid; place-items:center; font-weight:800; font-size:1.05rem; font-variant-numeric:tabular-nums; }
.fq-card.top .fq-rank { background:var(--fq-gold-soft); color:var(--fq-gold); }
.fq-card-main { flex:1; min-width:0; }
.fq-card-title { margin:0 0 8px !important; font-size:1.55rem !important; line-height:1.55 !important; font-weight:700; color:var(--fq-ink); text-align:right; }
.fq-card-meta { display:flex; flex-wrap:wrap; gap:6px 8px; align-items:center; }
.fq-best { font-size:.74rem; font-weight:800; color:var(--fq-gold); background:var(--fq-gold-soft); padding:2px 10px; border-radius:999px; }
.fq-source { font-size:.78rem; font-weight:700; color:var(--fq-muted); margin-right:4px; }
.fq-cat { display:inline-block; padding:0 11px; border-radius:999px; background:var(--fq-accent-soft); color:var(--fq-accent-ink); font-size:1rem; line-height:1.7; }

.fq-judge { margin:16px 0 4px; padding:14px 18px 12px; border-radius:14px; background:var(--fq-surface-2); border:1px solid var(--fq-line); }
.fq-judge.good { background:var(--fq-good-soft); border-color:transparent; }
.fq-judge.warn { background:var(--fq-warn-soft); border-color:transparent; }
.fq-judge-head { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:8px; }
.fq-judge-head .fq-badge { background:var(--fq-surface); }
.fq-conf { font-size:.78rem; color:var(--fq-muted); font-weight:600; font-variant-numeric:tabular-nums; }
.fq-judge .fq-label { margin-top:8px; }
.fq-judge-answer { font-size:1.28rem; line-height:2; color:var(--fq-ink); white-space:pre-wrap; }
.fq-judge-why { font-size:1.08rem; line-height:1.9; color:var(--fq-muted); white-space:pre-wrap; }
.fq-mini { margin-top:10px; font-size:.82rem; } .fq-mini summary { cursor:pointer; color:var(--fq-muted); font-weight:600; }
.fq-prob { display:flex; align-items:center; gap:10px; margin:5px 0; font-size:.8rem; }
.fq-prob-l { width:100px; color:var(--fq-muted); } .fq-prob-v { width:40px; text-align:right; font-family:var(--fq-mono); font-size:.76rem; }
.fq-prob-bar { flex:1; height:7px; border-radius:99px; background:var(--fq-surface); overflow:hidden; }
.fq-prob-bar > span { display:block; height:100%; background:var(--fq-accent); border-radius:99px; }

.fq-qblock { margin-top:16px; }
.fq-question { font-size:1.2rem; line-height:1.95; font-weight:700; white-space:pre-wrap; }
.fq-det { margin-top:12px; border-top:1px solid var(--fq-line); padding-top:4px; }
.fq-det > summary { cursor:pointer; list-style:none; display:flex; align-items:center; gap:8px; padding:8px 0; font-size:.7rem;
  font-weight:800; letter-spacing:.09em; text-transform:uppercase; color:var(--fq-muted); user-select:none; }
.fq-det > summary::-webkit-details-marker { display:none; }
.fq-det > summary::before { content:"▸"; font-size:.85rem; transition:transform .15s; color:var(--fq-accent); }
.fq-det[open] > summary::before { transform:rotate(90deg); }
.fq-det > summary:hover { color:var(--fq-accent); }
.fq-answer-text { font-size:1.22rem; line-height:2.1; white-space:pre-wrap; max-height:430px; overflow-y:auto; padding:6px 4px 6px 12px; }
.fq-answer-text::-webkit-scrollbar { width:8px; } .fq-answer-text::-webkit-scrollbar-thumb { background:var(--fq-line); border-radius:8px; }

.fq-card-foot { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; margin-top:10px; padding-top:12px; border-top:1px dashed var(--fq-line); }
.fq-metrics { display:flex; gap:6px; flex-wrap:wrap; }
.fq-m { font-family:var(--fq-mono); font-size:.72rem; color:var(--fq-muted); background:var(--fq-surface-2); padding:3px 9px; border-radius:8px; font-style:normal; }
.fq-m i { font-style:normal; opacity:.7; }
.fq-m.move.up { color:var(--fq-good); background:var(--fq-good-soft); } .fq-m.move.down { color:var(--fq-bad); background:var(--fq-bad-soft); }
.fq-actions { display:flex; gap:8px; }
.fq-btn { font-family:var(--fq-sans); font-size:.8rem; font-weight:700; padding:6px 14px; border-radius:10px; border:1px solid var(--fq-line);
  background:var(--fq-surface) !important; color:var(--fq-ink) !important; cursor:pointer; text-decoration:none !important; }
.fq-btn:hover { border-color:var(--fq-accent); color:var(--fq-accent) !important; }
.fq-btn.primary { background:var(--fq-accent) !important; border-color:var(--fq-accent); color:#fff !important; }
.fq-btn.primary:hover { background:var(--fq-accent-ink) !important; color:#fff !important; }
.dark .fq-btn.primary, .dark .fq-btn.primary:hover { color:#06130f !important; }

/* ---- skeleton ---- */
.fq-skel-card { background:var(--fq-surface); border:1px solid var(--fq-line); border-radius:20px; padding:22px 24px; margin:14px 0; }
.fq-skel-card i { display:block; height:14px; border-radius:8px; margin:12px 0; background:linear-gradient(90deg,var(--fq-surface-2),var(--fq-line),var(--fq-surface-2)); background-size:200% 100%; animation:fq-shim 1.3s infinite; }
.fq-skel-card i:nth-child(1) { width:55%; height:22px; margin-left:auto; } .fq-skel-card i:nth-child(2) { width:90%; } .fq-skel-card i:nth-child(3) { width:80%; } .fq-skel-card i:nth-child(4) { width:40%; }
@keyframes fq-shim { to { background-position:-200% 0; } }

/* ---- pipeline ---- */
.fq-kpis { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:12px 0 16px; }
.fq-kpi { background:var(--fq-surface); border:1px solid var(--fq-line); border-radius:16px; padding:14px 18px; }
.fq-kpi-label { font-size:.7rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; color:var(--fq-muted); }
.fq-kpi-value { font-size:1.7rem; font-weight:800; letter-spacing:-.02em; font-variant-numeric:tabular-nums; margin:2px 0; }
.fq-kpi-value small { font-size:1rem; color:var(--fq-muted); margin-left:2px; }
.fq-kpi-sub { font-size:.78rem; color:var(--fq-muted); }
.fq-steps { list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:10px; }
.fq-step { background:var(--fq-surface); border:1px solid var(--fq-line); border-radius:14px; padding:12px 16px; }
.fq-step.off { opacity:.62; } .fq-step.fail { border-color:var(--fq-bad); }
.fq-step-top { display:flex; align-items:center; gap:10px; }
.fq-step-idx { width:22px; height:22px; border-radius:50%; background:var(--fq-accent-soft); color:var(--fq-accent-ink); font-size:.72rem; font-weight:800; display:grid; place-items:center; }
.fq-step.fail .fq-step-idx { background:var(--fq-bad-soft); color:var(--fq-bad); }
.fq-step-name { font-weight:700; flex:1; }
.fq-step-time { font-family:var(--fq-mono); font-size:.86rem; font-variant-numeric:tabular-nums; }
.fq-bar { position:relative; height:8px; border-radius:99px; background:var(--fq-surface-2); margin:10px 0 8px; overflow:hidden; }
.fq-bar > span { position:absolute; top:0; bottom:0; border-radius:99px; background:var(--fq-accent); }
.fq-step.fail .fq-bar > span { background:var(--fq-bad); }
.fq-step-meta { display:flex; gap:6px; flex-wrap:wrap; min-height:4px; }

/* ---- debug tables ---- */
.fq-sect-title { font-weight:800; margin:18px 2px 2px; } .fq-sect-sub { color:var(--fq-muted); font-size:.85rem; margin:0 2px 8px; }
.fq-table-wrap { overflow:auto; max-height:520px; border:1px solid var(--fq-line); border-radius:14px; background:var(--fq-surface); }
.fq-table { width:100%; border-collapse:separate; border-spacing:0; font-size:.84rem; }
.fq-table th { position:sticky; top:0; z-index:1; background:var(--fq-surface-2); text-align:left; padding:9px 12px; font-size:.68rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--fq-muted); font-weight:800; white-space:nowrap; border-bottom:1px solid var(--fq-line); }
.fq-table th, .fq-table td { border-left:none !important; border-right:none !important; border-top:none !important; }
.fq-table td { padding:8px 12px; border-bottom:1px solid var(--fq-line); vertical-align:middle; white-space:nowrap; }
.fq-table tbody tr:last-child td { border-bottom:none; } .fq-table tbody tr:hover td { background:var(--fq-accent-soft); }
.fq-rk { font-family:var(--fq-mono); font-size:.8rem; color:var(--fq-muted); } .fq-rk.strong { color:var(--fq-ink); font-weight:700; margin-right:6px; }
.fq-sc { display:inline-flex; flex-direction:column; gap:3px; min-width:84px; } .fq-sc b { font-family:var(--fq-mono); font-weight:500; font-size:.8rem; }
.fq-sc i { display:block; height:4px; border-radius:9px; background:var(--fq-surface-2); overflow:hidden; } .fq-sc i > span { display:block; height:100%; background:var(--fq-accent); }
.fq-mono-dim { font-family:var(--fq-mono); font-size:.78rem; color:var(--fq-muted); }
.fq-ttl { display:block; max-width:420px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:1.05rem; }
.fq-table td:last-child { text-align:right; }

/* ---- error log / toast ---- */
#fq-errlog textarea { font-family:var(--fq-mono) !important; font-size:.78rem !important; }
.fq-toast { text-align:left; } .fq-toast-title { font-weight:800; color:#dc2626; font-size:1.05rem; margin-bottom:6px; }
.fq-toast-q { padding:6px 10px; border-radius:8px; background:rgba(127,127,127,.12); margin-bottom:6px; }
.fq-toast-sub { font-size:.7rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; opacity:.7; margin:10px 0 2px; }
.fq-toast ul { margin:2px 0 0 18px; font-size:.85rem; }
pre.tb { white-space:pre-wrap; font-size:.74rem; background:#0b1220; color:#e5e7eb; padding:10px; border-radius:8px; max-height:300px; overflow:auto; }

/* ---- settings drawer ---- */
.fq-side-title { font-size:1.25rem; font-weight:800; letter-spacing:-.01em; }
.fq-side-sub { font-size:.84rem; color:var(--fq-muted); margin:2px 0 6px; }
.fq-keys { display:flex; gap:8px; flex-wrap:wrap; }
.fq-key { font-size:.78rem; font-weight:700; padding:4px 11px; border-radius:999px; }
.fq-key.ok { background:var(--fq-good-soft); color:var(--fq-good); } .fq-key.no { background:var(--fq-bad-soft); color:var(--fq-bad); }
.fq-sys { display:grid; grid-template-columns:auto 1fr; gap:6px 14px; font-size:.82rem; } .fq-sys dt { color:var(--fq-muted); font-weight:600; }
.fq-sys dd { margin:0; font-family:var(--fq-mono); font-size:.76rem; word-break:break-word; }

@media (max-width:720px) {
  .fq-kpis { grid-template-columns:1fr; } .fq-qchips { margin-left:0; } .fq-qrow { flex-direction:column; gap:2px; }
  .fq-card { padding:16px 16px 12px; } .fq-card-title { font-size:1.35rem !important; }
}
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "كيف أكفر عن يمينٍ حلفتها ثم حنثت؟",
    "ما حكم صلاة الجماعة في المسجد؟",
    "What is the ruling on paying zakat on gold?",
]

BRAND_MARK = """
<svg viewBox="0 0 48 48" class="fq-mark" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="2.4">
<rect x="11" y="11" width="26" height="26" rx="1.5"/><rect x="11" y="11" width="26" height="26" rx="1.5" transform="rotate(45 24 24)"/></g>
<circle class="gold" cx="24" cy="24" r="4"/></svg>
"""


def build_ui(app: FatwaApp) -> gr.Blocks:
    n_fatwas = len(app.df) if hasattr(app, "df") else 0
    emb_dev = getattr(getattr(app, "s", None), "device_embedder", "?")
    rer_dev = getattr(getattr(app, "s", None), "device_reranker", "?")

    def key_chip(env: str, name: str) -> str:
        ok = bool(os.getenv(env))
        return f'<span class="fq-key {"ok" if ok else "no"}">{"✓" if ok else "✕"} {name}</span>'

    with gr.Blocks(title="Fatwa Search") as demo:
        sidebar_open = gr.State(False)

        # ---- top bar -------------------------------------------------------
        with gr.Row(elem_id="fq-top"):
            gr.HTML(
                f"<div class='fq-brand'>{BRAND_MARK}<div><div class='fq-name'>Fatwa Search</div>"
                f"<div class='fq-tagline'>{n_fatwas:,} fatwas indexed · semantic retrieval + reranking</div></div></div>",
                elem_classes="fq-flat",
            )
            settings_btn = gr.Button("⚙  Settings", elem_id="fq-settings-btn", size="sm", scale=0)

        # ---- search --------------------------------------------------------
        with gr.Group(elem_id="fq-search-wrap"):
            query = gr.Textbox(
                placeholder="اكتب سؤالك هنا…  ·  or ask in any language",
                show_label=False,
                container=False,
                lines=1,
                max_lines=5,
                rtl=True,
                autofocus=True,
                submit_btn=True,
                elem_id="fq-query",
            )
        with gr.Row(elem_id="fq-examples"):
            gr.HTML("<span class='fq-ex-label'>Try</span>", elem_classes="fq-flat")
            example_btns = [
                gr.Button(q if len(q) < 46 else q[:44] + "…", size="sm", elem_classes="fq-ex", scale=0)
                for q in EXAMPLE_QUERIES
            ]
            clear_btn = gr.Button("Clear", size="sm", elem_classes="fq-ex", scale=0)

        query_info = gr.HTML(EMPTY_QUERY, elem_classes="fq-flat")

        # ---- results -------------------------------------------------------
        with gr.Tabs(elem_id="fq-tabs"):
            with gr.Tab("Results"):
                results_html = gr.HTML(EMPTY_RESULTS, elem_classes="fq-flat")
            with gr.Tab("Pipeline"):
                stats_html = gr.HTML(EMPTY_PIPELINE, elem_classes="fq-flat")
            with gr.Tab("Debug"):
                gr.HTML(
                    "<div class='fq-sect-title'>Reranked candidates</div>"
                    "<div class='fq-sect-sub'>The fatwas that were shown/judged. Arrows show how far the reranker "
                    "moved each one versus embedding order. Bars are relative within the column.</div>",
                    elem_classes="fq-flat",
                )
                rerank_html = gr.HTML(EMPTY_TABLE, elem_classes="fq-flat")
                gr.HTML(
                    "<div class='fq-sect-title'>Embedding retrieval</div>"
                    "<div class='fq-sect-sub'>Raw top-K by cosine similarity, before reranking.</div>",
                    elem_classes="fq-flat",
                )
                retrieval_html = gr.HTML(EMPTY_TABLE, elem_classes="fq-flat")
                gr.HTML("<div class='fq-sect-title'>Error log</div>", elem_classes="fq-flat")
                error_text = gr.Textbox(
                    show_label=False, lines=8, max_lines=20, interactive=False, rtl=False,
                    buttons=["copy"], placeholder="No errors.", elem_id="fq-errlog",
                )

        # ---- settings drawer ----------------------------------------------
        with gr.Sidebar(label="Settings", open=False, position="right", width=400) as sidebar:
            gr.HTML(
                "<div class='fq-side-title'>Settings</div>"
                "<div class='fq-side-sub'>Defaults work well. Change these only to experiment.</div>",
                elem_classes="fq-flat",
            )
            with gr.Accordion("Query rewrite", open=True):
                rewrite_enabled = gr.Checkbox(
                    value=bool(os.getenv(GEMINI_API_KEY_ENV)), label="Rewrite / translate the question",
                    info="Turns any language into a clean Arabic search query.",
                )
                rewrite_provider = gr.Radio(PROVIDERS, value=PROVIDER_GEMINI, label="Provider")
                rewrite_model = gr.Dropdown(
                    models_for("rewrite", PROVIDER_GEMINI), value=GEMINI_MODELS[0],
                    label="Model", allow_custom_value=True,
                )
            with gr.Accordion("Judging", open=True):
                judge_enabled = gr.Checkbox(
                    value=bool(os.getenv(GEMINI_API_KEY_ENV)), label="Judge each result",
                    info="An LLM checks whether each fatwa really answers the question.",
                )
                judge_provider = gr.Radio(PROVIDERS, value=PROVIDER_GEMINI, label="Provider")
                judge_model = gr.Dropdown(
                    models_for("judge", PROVIDER_GEMINI), value=GEMINI_MODELS[0],
                    label="Model", allow_custom_value=True,
                    info="OpenRouter + typesafe/jev-* uses the JEV Decisions API.",
                )
            with gr.Accordion("Retrieval", open=False):
                retrieve_k = gr.Slider(5, MAX_RETRIEVE_K, value=DEFAULT_RETRIEVE_K, step=1,
                                       label="Candidates to retrieve", info="Embedding top-K")
                rerank_pool = gr.Slider(1, DEFAULT_RETRIEVE_K, value=DEFAULT_RERANK_POOL, step=1,
                                        label="Candidates to rerank")
                show_n = gr.Slider(1, DEFAULT_RERANK_POOL, value=DEFAULT_SHOW_N, step=1,
                                   label="Results to show", info="These are also the ones that get judged.")
                max_candidate_chars = gr.Slider(500, 8000, value=DEFAULT_MAX_CANDIDATE_CHARS, step=250,
                                                label="Judge text limit", info="Max characters per field sent to the judge.")
                min_rerank_score = gr.Number(value=None, label="Minimum rerank score", info="Leave blank for no filter.")
            with gr.Accordion("Show verdicts", open=False):
                filter_verdicts = gr.CheckboxGroup(
                    [(VERDICT_LABELS[v], v) for v in VERDICTS], value=list(VERDICTS),
                    show_label=False, info="Only applies when judging is on. Untick all to disable filtering.",
                )
            with gr.Accordion("System", open=False):
                gr.HTML(
                    f"<div class='fq-keys'>{key_chip(GEMINI_API_KEY_ENV, 'Gemini key')}"
                    f"{key_chip(OPENROUTER_API_KEY_ENV, 'OpenRouter key')}</div>"
                    "<dl class='fq-sys' style='margin-top:12px'>"
                    f"<dt>Embedder</dt><dd>{esc(EMBEDDER_MODEL_NAME)} · {esc(emb_dev)}</dd>"
                    f"<dt>Reranker</dt><dd>{esc(RERANKER_MODEL_NAME)} · {esc(rer_dev)}</dd>"
                    f"<dt>Corpus</dt><dd>{n_fatwas:,} fatwas</dd></dl>",
                    elem_classes="fq-flat",
                )
            reset_btn = gr.Button("Reset to defaults", size="sm")

        # ---- wiring --------------------------------------------------------
        inputs = [
            query,
            rewrite_enabled, rewrite_provider, rewrite_model,
            judge_enabled, judge_provider, judge_model,
            retrieve_k, rerank_pool, show_n,
            max_candidate_chars, min_rerank_score, filter_verdicts,
        ]
        outputs = [query_info, stats_html, retrieval_html, rerank_html, results_html, error_text]

        def search(*args):
            if not (args[0] or "").strip():
                yield ('<div class="fq-note">Please enter a question.</div>',
                       EMPTY_PIPELINE, EMPTY_TABLE, EMPTY_TABLE, EMPTY_RESULTS, "")
                return
            yield LOADING_QUERY, EMPTY_PIPELINE, EMPTY_TABLE, EMPTY_TABLE, LOADING_RESULTS, ""
            q_info, stats, retrieved, reranked, cards, err = app.run_search(*args)
            yield (
                q_info if str(q_info).lstrip().startswith("<") else f'<div class="fq-note">{esc(q_info)}</div>',
                stats,
                df_to_html(retrieved),
                df_to_html(reranked),
                cards or EMPTY_FAILED,
                err,
            )

        run_kw = dict(fn=search, inputs=inputs, outputs=outputs, show_progress="hidden")
        query.submit(**run_kw)
        for btn, text in zip(example_btns, EXAMPLE_QUERIES):
            btn.click(lambda t=text: t, outputs=query).then(**run_kw)

        clear_btn.click(
            lambda: ("", EMPTY_QUERY, EMPTY_PIPELINE, EMPTY_TABLE, EMPTY_TABLE, EMPTY_RESULTS, ""),
            outputs=[query] + outputs, show_progress="hidden",
        )

        # settings drawer toggle (button <-> native sidebar toggle stay in sync)
        settings_btn.click(
            lambda is_open: (not is_open, gr.Sidebar(open=not is_open)),
            inputs=sidebar_open, outputs=[sidebar_open, sidebar], show_progress="hidden",
        )
        sidebar.expand(lambda: True, outputs=sidebar_open, show_progress="hidden")
        sidebar.collapse(lambda: False, outputs=sidebar_open, show_progress="hidden")

        # dependent controls
        rewrite_enabled.change(
            lambda e: (gr.update(interactive=e), gr.update(interactive=e)),
            inputs=rewrite_enabled, outputs=[rewrite_provider, rewrite_model], show_progress="hidden",
        )
        judge_enabled.change(
            lambda e: (gr.update(interactive=e), gr.update(interactive=e)),
            inputs=judge_enabled, outputs=[judge_provider, judge_model], show_progress="hidden",
        )
        rewrite_provider.change(
            lambda p: gr.update(choices=models_for("rewrite", p), value=models_for("rewrite", p)[0]),
            inputs=rewrite_provider, outputs=rewrite_model,
        )
        judge_provider.change(
            lambda p: gr.update(choices=models_for("judge", p), value=models_for("judge", p)[0]),
            inputs=judge_provider, outputs=judge_model,
        )
        retrieve_k.change(
            lambda k, rp: gr.update(maximum=int(k), value=min(int(rp), int(k))),
            inputs=[retrieve_k, rerank_pool], outputs=rerank_pool,
        )
        rerank_pool.change(
            lambda rp, sn: gr.update(maximum=int(rp), value=min(int(sn), int(rp))),
            inputs=[rerank_pool, show_n], outputs=show_n,
        )

        def reset_settings():
            return (
                True, PROVIDER_GEMINI, GEMINI_MODELS[0],
                True, PROVIDER_GEMINI, GEMINI_MODELS[0],
                DEFAULT_RETRIEVE_K, DEFAULT_RERANK_POOL, DEFAULT_SHOW_N,
                DEFAULT_MAX_CANDIDATE_CHARS, None, list(VERDICTS),
            )

        reset_btn.click(
            reset_settings,
            outputs=[
                rewrite_enabled, rewrite_provider, rewrite_model,
                judge_enabled, judge_provider, judge_model,
                retrieve_k, rerank_pool, show_n,
                max_candidate_chars, min_rerank_score, filter_verdicts,
            ],
        )

    return demo



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def ensure_data(data_dir: Path) -> None:
    """Download any missing data file from the Hugging Face dataset `DATA_REPO_ID`."""
    from huggingface_hub import hf_hub_download

    data_dir.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        if (data_dir / name).exists():
            continue
        print(f"Downloading {name} from {DATA_REPO_ID} ...")
        hf_hub_download(repo_id=DATA_REPO_ID, repo_type="dataset", filename=name, local_dir=data_dir)


def default_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arabic Fatwa Q&A: semantic search and reranking over Ibn Baz fatwas.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("FATWA_DATA_DIR", APP_DIR / "data")),
        help="Folder with binbaz_clean.csv + qwen3_06b_title_answer_embeddings.pt + embedding_metadata.json.",
    )
    parser.add_argument("--no-download", action="store_true", help="Do not fetch missing data files from Hugging Face.")
    parser.add_argument("--embedder-device", default="cpu", help="cpu (default), cuda, auto")
    parser.add_argument("--reranker-device", default="auto", help="auto (default), cuda, cpu")
    parser.add_argument("--no-warmup", action="store_true", help="Skip the startup warmup query.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=7860, help="Port (default 7860).")
    parser.add_argument("--share", action="store_true", help="Create a temporary public *.gradio.live link.")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser tab.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_env_files()
    args = parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not args.no_download:
        ensure_data(data_dir)
    if not data_dir.exists():
        print(f"Data dir does not exist: {data_dir}", file=sys.stderr)
        return 2

    if not os.getenv(GEMINI_API_KEY_ENV):
        print(f"Note: {GEMINI_API_KEY_ENV} is not set — Gemini tasks will report an error in the UI.")
    if not os.getenv(OPENROUTER_API_KEY_ENV):
        print(f"Note: {OPENROUTER_API_KEY_ENV} is not set — OpenRouter tasks will report an error in the UI.")

    settings = Settings(
        data_dir=data_dir,
        device_embedder=default_device(args.embedder_device),
        device_reranker=default_device(args.reranker_device),
        warmup=not args.no_warmup,
    )

    app = FatwaApp(settings)
    demo = build_ui(app)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    demo.queue().launch(
        theme=THEME,
        css=CSS,
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
