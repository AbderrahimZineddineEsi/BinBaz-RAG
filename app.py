#!/usr/bin/env python3
"""Fatwa Search — Multi-scholar semantic retrieval and AI analysis over trusted Islamic fatwas.

Upgraded to support multiple scholars:
- Sheikh Abdulaziz ibn Baz
- Sheikh Muhammad ibn Saalih al-Uthaymeen
- Sheikh Saleh Al-Fawzan
- Al-Lajnah Ad-Da'imah lil-Ifta / Council of Senior Scholars
and other trusted Saudi Sunni/Salafi authorities.

Two Search Modes:
1. X answers per scholar (independent search per dataset)
2. X answers ranked overall (combined search across selected scholars)

AI Features:
- Individual grounded verdict per fatwa (Gemini, OpenRouter, or JEV).
- Consensus Classification: do the scholars agree or disagree? (Gemini and JEV).
- Comparative Summary: synthesis of scholarly viewpoints (Gemini).
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import functools
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any
import webbrowser

import gradio as gr
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent


def load_env_files() -> None:
    for env_path in [APP_DIR / ".env", Path(".env")]:
        if not env_path.exists():
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except Exception:
            pass


load_env_files()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_REPO_ID = os.environ.get("FATWA_DATA_REPO", "ZenoAbderrahim/binbaz_fatwas_rag")
MULTI_SCHOLAR_DATA_FILES = [
    "multi_scholar_clean.csv",
    "qwen3_06b_multi_scholar_embeddings.pt",
    "multi_scholar_metadata.json",
]
BINBAZ_DATA_FILES = [
    "binbaz_clean.csv",
    "qwen3_06b_title_answer_embeddings.pt",
    "embedding_metadata.json",
]
DATA_FILES = MULTI_SCHOLAR_DATA_FILES

EMBEDDER_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL_NAME = "ALJIACHI/Mizan-Rerank-V2"

EMBEDDER_MAX_SEQ_LENGTH = 1024
RERANKER_MAX_LENGTH = 1024
RERANKER_BATCH_SIZE = 16

DEFAULT_RETRIEVE_K = 20
MAX_RETRIEVE_K = 200
DEFAULT_RERANK_POOL = 20
DEFAULT_SHOW_N = 5
DEFAULT_X_PER_SCHOLAR = 2
DEFAULT_MAX_CANDIDATE_CHARS = 3000

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

PROVIDER_GEMINI = "Gemini"
PROVIDER_OPENROUTER = "OpenRouter"
PROVIDERS = [PROVIDER_GEMINI, PROVIDER_OPENROUTER]

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]
OPENROUTER_CHAT_MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-3.1-flash-lite",
    "google/gemini-3.5-flash",
    "openai/gpt-4o-mini",
    "qwen/qwen3.5-flash-02-23",
    "qwen/qwen3-30b-a3b-instruct-2507",
]
JEV_MODELS = ["typesafe/jev-1.13", "~typesafe/jev-latest"]

# Modes
SEARCH_MODE_PER_SCHOLAR = "per_scholar"
SEARCH_MODE_OVERALL = "overall"
SEARCH_MODES = [
    ("X إجابة لكل شيخ (منفصلة)", SEARCH_MODE_PER_SCHOLAR),
    ("أفضل X إجابات مجمعة", SEARCH_MODE_OVERALL),
]

# Verdicts for individual fatwas
VERDICTS = ["answers-it", "related-only", "unrelated"]
VERDICT_LABELS = {
    "answers-it": "Answers it",
    "related-only": "Related only",
    "unrelated": "Unrelated",
    "unknown": "Unknown",
}
VERDICT_ICONS = {
    "answers-it": "✓",
    "related-only": "≈",
    "unrelated": "✕",
    "unknown": "?",
}

JEV_CRITERIA = {
    "answers-it": "The fatwa directly answers the user's question or gives the requested ruling/information.",
    "related-only": "The fatwa is on a related topic but does not answer the specific question.",
    "unrelated": "The fatwa does not meaningfully address the user's question.",
}

# Verdicts for consensus classification
CONSENSUS_AGREE = "agree"
CONSENSUS_PARTIAL = "partially-agree"
CONSENSUS_DISAGREE = "disagree"
CONSENSUS_VERDICTS = [CONSENSUS_AGREE, CONSENSUS_PARTIAL, CONSENSUS_DISAGREE]

CONSENSUS_LABELS = {
    CONSENSUS_AGREE: "متفقون (Agreement)",
    CONSENSUS_PARTIAL: "اتفاق مع تفصيل (Partial Agreement)",
    CONSENSUS_DISAGREE: "مختلفون (Divergence / Disagreement)",
    "unknown": "غير محدد (Indeterminate)",
}

CONSENSUS_ICONS = {
    CONSENSUS_AGREE: "✓",
    CONSENSUS_PARTIAL: "≈",
    CONSENSUS_DISAGREE: "✕",
    "unknown": "?",
}

JEV_CONSENSUS_CRITERIA = {
    CONSENSUS_AGREE: "All scholars agree or harmonize on the ruling.",
    CONSENSUS_PARTIAL: "Scholars agree generally but diverge on conditions, details, or specific exceptions.",
    CONSENSUS_DISAGREE: "Scholars hold opposing or conflicting verdicts (e.g., halal vs haram).",
}


def models_for(task: str, provider: str) -> list[str]:
    if task in ("judge", "consensus"):
        return GEMINI_MODELS if provider == PROVIDER_GEMINI else JEV_MODELS
    return GEMINI_MODELS if provider == PROVIDER_GEMINI else OPENROUTER_CHAT_MODELS


def is_jev_model(model: str) -> bool:
    m = (model or "").lower()
    return "jev" in m and ("typesafe" in m or m.startswith("~"))


# ---------------------------------------------------------------------------
# Schemas & Data Structures
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


class ConsensusClassificationModel(BaseModel):
    consensus: str = Field(description="One of: 'agree', 'partially-agree', 'disagree'")
    confidence: float = Field(default=1.0, description="Confidence score between 0.0 and 1.0")
    explanation: str = Field(description="Concise Arabic explanation analyzing consensus or divergence")


@dataclass
class FatwaJudgement:
    verdict: str
    confidence: str | float | None = None
    answer: str = ""
    explanation: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class ConsensusVerdict:
    consensus: str
    confidence: float | None = None
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
    """Loads either multi-scholar corpus assets, or falls back to Bin Baz base corpus."""
    multi_csv = data_dir / "multi_scholar_clean.csv"
    if not multi_csv.exists() and (data_dir / "cleaned" / "multi_scholar_clean.csv").exists():
        multi_csv = data_dir / "cleaned" / "multi_scholar_clean.csv"

    multi_emb = data_dir / "qwen3_06b_multi_scholar_embeddings.pt"
    if not multi_emb.exists() and (data_dir / "cleaned" / "qwen3_06b_multi_scholar_embeddings.pt").exists():
        multi_emb = data_dir / "cleaned" / "qwen3_06b_multi_scholar_embeddings.pt"

    multi_meta = data_dir / "multi_scholar_metadata.json"

    binbaz_csv = data_dir / "binbaz_clean.csv"
    binbaz_emb = data_dir / "qwen3_06b_title_answer_embeddings.pt"
    binbaz_meta = data_dir / "embedding_metadata.json"

    if multi_csv.exists() and multi_emb.exists():
        clean_csv = multi_csv
        embeddings_pt = multi_emb
        metadata_json = multi_meta if multi_meta.exists() else None
        print(f"Loading Multi-Scholar corpus from {clean_csv} and embeddings from {embeddings_pt}...")
    elif binbaz_csv.exists() and binbaz_emb.exists():
        clean_csv = binbaz_csv
        embeddings_pt = binbaz_emb
        metadata_json = binbaz_meta if binbaz_meta.exists() else None
        print(f"Loading Bin Baz corpus from {clean_csv} and embeddings from {embeddings_pt}...")
    else:
        missing = [p for p in (binbaz_csv, binbaz_emb) if not p.exists()]
        raise FileNotFoundError(
            f"Missing required data assets in {data_dir}. Missing:\n" + "\n".join(f"- {p}" for p in missing)
        )

    df = pd.read_csv(clean_csv)
    meta = {}
    if metadata_json and metadata_json.exists():
        with open(metadata_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

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
        if embeddings.shape[0] < len(df):
            print(f"Notice: embeddings count ({embeddings.shape[0]:,}) < CSV rows ({len(df):,}). Slicing DataFrame.")
            df = df.iloc[: embeddings.shape[0]].copy()
        else:
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
    key = os.getenv(OPENROUTER_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{OPENROUTER_API_KEY_ENV} is not set. Put it in .env or export it."
        )
    return key


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON object found in response: {text[:200]}")


def openrouter_chat(api_key: str, model: str, system_prompt: str, user_prompt: str, timeout: int = 150):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(OPENROUTER_CHAT_URL, headers=_headers(api_key), json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:500]}")
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
    return f'User question: "{query}"\nRespond ONLY with the JSON object.'


def judge_system_prompt() -> str:
    return (
        "You are a strict, objective Islamic fatwa evaluation judge. You are given a user question "
        "and a list of candidate fatwas with their title, question, and full answer. "
        "For EACH candidate decide: does it actually ANSWER the user's question? "
        "Classify each with verdict: 'answers-it' (directly answers), 'related-only' (on topic but does not answer), "
        "or 'unrelated'. Quote or ground a concise Arabic answer from the fatwa text."
    )


def judge_user_prompt(user_query: str, candidates: pd.DataFrame, max_chars: int) -> str:
    lines = [f"User question: {user_query}\n\nCandidate fatwas:"]
    for idx, row in enumerate(candidates.itertuples(index=False)):
        lines.append(f"\n--- Candidate {idx} (Scholar: {row.source}, fatwa_id: {row.fatwa_id}) ---")
        lines.append(f"Title: {str(row.title)[:max_chars]}")
        lines.append(f"Question: {str(row.question)[:max_chars]}")
        lines.append(f"Answer: {str(row.answer)[:max_chars]}")
    lines.append("\nReturn JSON matching the schema for all candidates.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FatwaApp
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
        self.available_scholars = sorted(self.df["source"].dropna().unique().tolist())
        print(
            f"Loaded {len(self.df):,} fatwas · embeddings {tuple(self.corpus_embeddings.shape)} · "
            f"Scholars: {self.available_scholars}"
        )

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

    # -- candidate judging -------------------------------------------------

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
                "scholar": str(row.source),
                "title": str(row.title)[:max_chars],
                "question": str(row.question)[:max_chars],
                "answer": str(row.answer)[:max_chars],
            }
            questions[qid] = {
                "type": "choice",
                "instructions": (
                    f"Judge candidate `candidates.{qid}`. Decide whether its fatwa answers the "
                    f"user's question in `user_query`. Pick exactly one option."
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

    # -- AI consensus classification & summary -----------------------------

    def _format_fatwas_for_synthesis(self, candidates: pd.DataFrame, max_chars: int = 1500) -> str:
        blocks = []
        for i, row in enumerate(candidates.itertuples(index=False), 1):
            blocks.append(
                f"[{i}] Scholar: {row.source}\n"
                f"Title: {row.title}\n"
                f"Question: {str(row.question)[:400]}\n"
                f"Answer: {str(row.answer)[:max_chars]}\n"
            )
        return "\n".join(blocks)

    def classify_consensus(
        self, provider: str, model: str, user_query: str, candidates: pd.DataFrame
    ) -> tuple[ConsensusVerdict, dict[str, Any]]:
        if provider == PROVIDER_GEMINI:
            return self._consensus_gemini(model, user_query, candidates)
        key = self.ensure_openrouter()
        if is_jev_model(model):
            return self._consensus_jev(key, model, user_query, candidates)
        return self._consensus_openrouter_chat(key, model, user_query, candidates)

    def _consensus_gemini(self, model: str, user_query: str, candidates: pd.DataFrame):
        client = self.ensure_gemini()
        prompt = (
            f"User Question: {user_query}\n\n"
            f"Scholarly Fatwas:\n{self._format_fatwas_for_synthesis(candidates)}\n\n"
            "Task: Compare the rulings and answers given by the scholars to this question. "
            "Classify their degree of consensus into one of three values:\n"
            "- 'agree': All scholars agree on the core ruling.\n"
            "- 'partially-agree': Scholars agree on the main principle but differ in conditions, details, or exceptions.\n"
            "- 'disagree': Scholars hold conflicting rulings (e.g. permissible vs prohibited, obligatory vs recommended).\n"
            "Provide a clear, respectful Arabic explanation of the agreement or divergence."
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": ConsensusClassificationModel.model_json_schema(),
            },
        )
        um = getattr(response, "usage_metadata", None)
        usage = usage_dict(
            getattr(um, "prompt_token_count", 0),
            getattr(um, "candidates_token_count", 0),
            getattr(um, "total_token_count", None),
        )
        parsed = ConsensusClassificationModel.model_validate_json(response.text)
        choice = parsed.consensus if parsed.consensus in CONSENSUS_VERDICTS else "unknown"
        return ConsensusVerdict(
            consensus=choice,
            confidence=parsed.confidence,
            explanation=parsed.explanation,
        ), usage

    def _consensus_jev(self, key: str, model: str, user_query: str, candidates: pd.DataFrame):
        state = {
            "user_query": (user_query or "").strip(),
            "scholars": list(candidates["source"].unique()),
            "fatwa_answers": [
                {"scholar": str(row.source), "title": str(row.title), "answer": str(row.answer)[:1000]}
                for row in candidates.itertuples(index=False)
            ],
        }
        questions = {
            "scholarly_consensus": {
                "type": "choice",
                "instructions": (
                    "Based on the provided scholarly answers, analyze whether the scholars agree on the ruling for the user's question. "
                    "Select 'agree' if their conclusions and rulings are identical or compatible. "
                    "Select 'partially-agree' if they agree on the general principle but differ in conditions, details, or branches. "
                    "Select 'disagree' if they express conflicting rulings (e.g. permissible vs prohibited)."
                ),
                "criteria": dict(JEV_CONSENSUS_CRITERIA),
            }
        }
        answers, usage = call_jev(key, model, state, questions)
        ans = answers.get("scholarly_consensus", {}) or {}
        choice = ans.get("choice") or "unknown"
        confidence = ans.get("confidence")
        probabilities = ans.get("probabilities") or {}
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None

        labels_ar = {
            CONSENSUS_AGREE: "اتفاق بين العلماء في أصل الحكم",
            CONSENSUS_PARTIAL: "اتفاق على أصل المسألة مع تفصيل وتنوع في الفروع أو الشروط",
            CONSENSUS_DISAGREE: "اختلاف في الحكم بين العلماء المذكورين",
        }
        explanation = labels_ar.get(choice, f"تصنيف التوافق بواسطة JEV: {choice}")
        return ConsensusVerdict(
            consensus=choice if choice in CONSENSUS_VERDICTS else "unknown",
            confidence=confidence,
            explanation=explanation,
            probabilities={str(k): float(v) for k, v in probabilities.items() if isinstance(v, (int, float))},
        ), usage

    def _consensus_openrouter_chat(self, key: str, model: str, user_query: str, candidates: pd.DataFrame):
        sys_prompt = (
            "You are an Islamic jurisprudence scholar analysis assistant. "
            "Analyze the retrieved scholarly fatwas and determine their consensus level on the user question. "
            "Return a JSON object with keys: 'consensus' ('agree'|'partially-agree'|'disagree'), 'confidence' (float), and 'explanation' (Arabic text)."
        )
        user_prompt = f"Question: {user_query}\n\nFatwas:\n{self._format_fatwas_for_synthesis(candidates)}"
        data, usage = openrouter_chat(key, model, sys_prompt, user_prompt)
        parsed = ConsensusClassificationModel.model_validate(data)
        choice = parsed.consensus if parsed.consensus in CONSENSUS_VERDICTS else "unknown"
        return ConsensusVerdict(
            consensus=choice,
            confidence=parsed.confidence,
            explanation=parsed.explanation,
        ), usage

    def summarize_opinions_gemini(
        self, model: str, user_query: str, candidates: pd.DataFrame
    ) -> tuple[str, dict[str, Any]]:
        client = self.ensure_gemini()
        prompt = (
            f"السؤال: {user_query}\n\n"
            f"فتاوى العلماء المعثور عليها:\n{self._format_fatwas_for_synthesis(candidates)}\n\n"
            "المطلوب: قدم خلاصة فقهية جامعة وموجزة ودقيقة باللغة العربية تبين موقف كل شيخ أو هيئة إفتاء من المسألة. "
            "اذكر أوجه الاتفاق والاختلاف (إن وجدت)، وانسب كل قول إلى قائله بأمانة ودقة واختصار، وبأسلوب علمي رصين."
        )
        response = client.models.generate_content(model=model, contents=prompt)
        um = getattr(response, "usage_metadata", None)
        usage = usage_dict(
            getattr(um, "prompt_token_count", 0),
            getattr(um, "candidates_token_count", 0),
            getattr(um, "total_token_count", None),
        )
        return response.text.strip(), usage

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

    def retrieve_top_k(
        self, query: str, top_k: int, allowed_scholars: list[str] | None = None
    ) -> pd.DataFrame:
        top_k = max(1, min(int(top_k), MAX_RETRIEVE_K, len(self.df)))
        q_vec = self._encode_query_cached(query)

        if allowed_scholars and len(allowed_scholars) < len(self.available_scholars):
            mask = self.df["source"].isin(allowed_scholars).values
            indices = np.where(mask)[0]
            if len(indices) == 0:
                return pd.DataFrame()
            sub_emb = self.corpus_embeddings[indices]
            scores = torch.mv(sub_emb, q_vec)
            k = min(top_k, len(indices))
            values, top_rel_idx = torch.topk(scores, k=k)
            top_abs_idx = indices[top_rel_idx.numpy()]
            out = self.df.iloc[top_abs_idx].copy().reset_index(drop=True)
            out.insert(0, "retrieval_rank", np.arange(1, len(out) + 1))
            out["embedding_score"] = values.numpy()
            return out
        else:
            scores = torch.mv(self.corpus_embeddings, q_vec)
            values, indices = torch.topk(scores, k=top_k)
            out = self.df.iloc[indices.numpy()].copy().reset_index(drop=True)
            out.insert(0, "retrieval_rank", np.arange(1, len(out) + 1))
            out["embedding_score"] = values.numpy()
            return out

    def retrieve_top_k_per_scholar(
        self, query: str, scholars: list[str], pool_per_scholar: int
    ) -> dict[str, pd.DataFrame]:
        q_vec = self._encode_query_cached(query)
        out = {}
        for s in scholars:
            mask = (self.df["source"] == s).values
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            sub_emb = self.corpus_embeddings[indices]
            scores = torch.mv(sub_emb, q_vec)
            k = max(1, min(int(pool_per_scholar), len(indices)))
            values, top_rel_idx = torch.topk(scores, k=k)
            top_abs_idx = indices[top_rel_idx.numpy()]
            sub_df = self.df.iloc[top_abs_idx].copy().reset_index(drop=True)
            sub_df.insert(0, "retrieval_rank", np.arange(1, len(sub_df) + 1))
            sub_df["embedding_score"] = values.numpy()
            out[s] = sub_df
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
        except Exception as e:
            print(f"Warmup skipped: {type(e).__name__}: {e}")

    # -- main search -------------------------------------------------------

    def run_search(
        self,
        user_query: str,
        search_mode: str,
        selected_scholars: list[str],
        x_per_scholar: int,
        rewrite_enabled: bool,
        rewrite_provider: str,
        rewrite_model: str,
        judge_enabled: bool,
        judge_provider: str,
        judge_model: str,
        consensus_enabled: bool,
        consensus_provider: str,
        consensus_model: str,
        summary_enabled: bool,
        summary_model: str,
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

        # Default scholars if none selected
        if not selected_scholars:
            selected_scholars = list(self.available_scholars)

        # -- rewrite
        rewrite = None
        search_query = user_query
        if rewrite_enabled:
            t0 = time.perf_counter()
            try:
                rewrite, usage = self.rewrite_query(rewrite_provider, rewrite_model, user_query)
                search_query = rewrite.improved_query.strip() or user_query
                steps.append(
                    Stat("Query rewrite", time.perf_counter() - t0, **usage, provider=rewrite_provider, model=rewrite_model)
                )
            except Exception as e:
                dt = time.perf_counter() - t0
                msg = f"Query rewrite failed ({rewrite_provider} / {rewrite_model}): {type(e).__name__}: {e}"
                errors.append(msg)
                steps.append(Stat("Query rewrite", dt, provider=rewrite_provider, model=rewrite_model, note="failed"))
        else:
            steps.append(Stat("Query rewrite", 0.0, note="disabled"))

        # -- retrieval & rerank based on search mode
        retrieved = pd.DataFrame()
        top_n = pd.DataFrame()

        t0_retrieval = time.perf_counter()
        try:
            if search_mode == SEARCH_MODE_PER_SCHOLAR:
                # MODE 1: X answers per scholar (independent search per dataset)
                pool_s = max(int(x_per_scholar), min(int(rerank_pool), 20))
                per_scholar_cands = self.retrieve_top_k_per_scholar(search_query, selected_scholars, pool_s)
                retrieval_time = time.perf_counter() - t0_retrieval

                t0_rerank = time.perf_counter()
                all_retrieved_list = []
                reranked_parts = []
                for s in selected_scholars:
                    cand_s = per_scholar_cands.get(s)
                    if cand_s is not None and len(cand_s) > 0:
                        all_retrieved_list.append(cand_s)
                        reranked_s = self.rerank_candidates(search_query, cand_s)
                        if min_rerank_score is not None:
                            reranked_s = reranked_s[reranked_s["reranker_score"] >= float(min_rerank_score)].reset_index(drop=True)
                            reranked_s["rerank_rank"] = np.arange(1, len(reranked_s) + 1)
                        reranked_parts.append(reranked_s.head(int(x_per_scholar)))

                rerank_time = time.perf_counter() - t0_rerank
                retrieved = pd.concat(all_retrieved_list, ignore_index=True) if all_retrieved_list else pd.DataFrame()
                if reranked_parts:
                    top_n = pd.concat(reranked_parts, ignore_index=True)
                    top_n["rerank_rank"] = np.arange(1, len(top_n) + 1)
                else:
                    top_n = pd.DataFrame()

                steps.append(Stat("Embedding retrieval", retrieval_time, note=f"{len(retrieved)} candidates across {len(selected_scholars)} scholars"))
                steps.append(Stat("Mizan rerank", rerank_time, note=f"{len(top_n)} selected ({x_per_scholar}/scholar)"))

            else:
                # MODE 2: X answers ranked overall
                retrieved = self.retrieve_top_k(search_query, int(retrieve_k), allowed_scholars=selected_scholars)
                retrieval_time = time.perf_counter() - t0_retrieval
                steps.append(Stat("Embedding retrieval", retrieval_time, note=f"{len(retrieved)} candidates"))

                t0_rerank = time.perf_counter()
                pool = min(int(rerank_pool), len(retrieved))
                reranked = self.rerank_candidates(search_query, retrieved.head(pool))
                if min_rerank_score is not None:
                    reranked = reranked[reranked["reranker_score"] >= float(min_rerank_score)].reset_index(drop=True)
                    reranked["rerank_rank"] = np.arange(1, len(reranked) + 1)
                rerank_time = time.perf_counter() - t0_rerank
                steps.append(Stat("Mizan rerank", rerank_time, note=f"over {pool}"))
                top_n = reranked.head(max(1, int(show_n))).copy()

        except Exception as e:
            err = positional_error(e)
            return self._fatal(user_query, steps, errors + [f"Retrieval/Rerank failed: {e}"], err)

        # -- individual candidate judging
        judge_map: dict[int, FatwaJudgement] = {}
        if judge_enabled and len(top_n):
            t0 = time.perf_counter()
            try:
                judge_map, usage = self.judge_candidates(
                    judge_provider, judge_model, user_query, top_n, int(max_candidate_chars)
                )
                steps.append(
                    Stat("Judge", time.perf_counter() - t0, **usage, provider=judge_provider, model=judge_model, note=f"{len(judge_map)} judged")
                )
            except Exception as e:
                dt = time.perf_counter() - t0
                msg = f"Judge failed ({judge_provider} / {judge_model}): {type(e).__name__}: {e}"
                errors.append(msg)
                steps.append(Stat("Judge", dt, provider=judge_provider, model=judge_model, note="failed"))
        else:
            steps.append(Stat("Judge", 0.0, note="disabled" if not judge_enabled else "no candidates"))

        # -- consensus classification (Gemini or JEV)
        consensus_verdict: ConsensusVerdict | None = None
        if consensus_enabled and len(top_n) > 1:
            t0 = time.perf_counter()
            try:
                consensus_verdict, usage = self.classify_consensus(
                    consensus_provider, consensus_model, user_query, top_n
                )
                steps.append(
                    Stat("Consensus classification", time.perf_counter() - t0, **usage,
                         provider=consensus_provider, model=consensus_model, note=consensus_verdict.consensus)
                )
            except Exception as e:
                dt = time.perf_counter() - t0
                msg = f"Consensus classification failed ({consensus_provider} / {consensus_model}): {type(e).__name__}: {e}"
                errors.append(msg)
                steps.append(Stat("Consensus classification", dt, provider=consensus_provider, model=consensus_model, note="failed"))
        else:
            steps.append(Stat("Consensus classification", 0.0, note="disabled" if not consensus_enabled else "insufficient results"))

        # -- comparative summary (Gemini only)
        summary_text: str | None = None
        if summary_enabled and len(top_n) > 0:
            t0 = time.perf_counter()
            try:
                summary_text, usage = self.summarize_opinions_gemini(summary_model, user_query, top_n)
                steps.append(
                    Stat("Comparative summary", time.perf_counter() - t0, **usage,
                         provider=PROVIDER_GEMINI, model=summary_model, note=f"{len(summary_text)} chars")
                )
            except Exception as e:
                dt = time.perf_counter() - t0
                msg = f"Comparative summary failed ({summary_model}): {type(e).__name__}: {e}"
                errors.append(msg)
                steps.append(Stat("Comparative summary", dt, provider=PROVIDER_GEMINI, model=summary_model, note="failed"))
        else:
            steps.append(Stat("Comparative summary", 0.0, note="disabled" if not summary_enabled else "no candidates"))

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

        query_info = self._query_panel(user_query, search_query, rewrite, steps, errors, judge_map, judge_enabled, consensus_verdict)
        stats_html = self._stats_html(steps)
        retrieval_view = dataframe_for_display(retrieved)
        rerank_view = dataframe_for_display(top_n, judge_map=judge_map if judge_enabled else None)

        # Consensus Box
        consensus_box_html = render_consensus_box(consensus_verdict, summary_text)

        if len(display_rows):
            cards = (consensus_box_html + "\n" if consensus_box_html else "") + "\n".join(
                fatwa_card(row, judge_map.get(int(row.fatwa_id)) if judge_enabled else None, int(row.get("rerank_rank", idx + 1)))
                for idx, (_, row) in enumerate(display_rows.iterrows())
            )
        else:
            cards = (consensus_box_html + "\n" if consensus_box_html else "") + '<p class="empty">No candidates match the selected filter.</p>'

        error_text = "\n\n".join(errors)
        if errors:
            gr.Warning(error_modal_html(errors, user_query, steps), duration=None, title="Error — copy and send to the AI")

        return query_info, stats_html, retrieval_view, rerank_view, cards, error_text

    def _fatal(self, user_query, steps, errors, err):
        steps.append(Stat("Total", sum(s.seconds for s in steps if s.label != "Total"), note="aborted"))
        tb = traceback.format_exc() if err is None else err
        gr.Warning(error_modal_html(errors, user_query, steps, tb), duration=None, title="Error — copy and send to the AI")
        query_info = self._query_panel(user_query, user_query, None, steps, errors, {}, False, None)
        return query_info, self._stats_html(steps), pd.DataFrame(), pd.DataFrame(), "", "\n\n".join(errors) + "\n\n" + (tb or "")

    # -- panels ------------------------------------------------------------

    def _query_panel(self, user_query, search_query, rewrite, steps, errors, judge_map, judge_enabled, consensus_verdict):
        return render_query_panel(user_query, search_query, rewrite, errors, judge_map, judge_enabled, consensus_verdict)

    def _stats_html(self, steps: list[Stat]) -> str:
        return render_pipeline(steps)


@dataclass
class Settings:
    data_dir: Path
    device_embedder: str
    device_reranker: str
    warmup: bool = True


def positional_error(e: Exception) -> str:
    tb = traceback.format_exc()
    lines = [line.strip() for line in tb.splitlines() if line.strip()]
    return lines[-1] if lines else str(e)


# ---------------------------------------------------------------------------
# HTML Rendering Helpers
# ---------------------------------------------------------------------------


def esc(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def verdict_badge_class(verdict: str) -> str:
    return {"answers-it": "good", "related-only": "warn", "unrelated": "bad"}.get(verdict, "neutral")


def verdict_badge(verdict: str) -> str:
    label = VERDICT_LABELS.get(verdict, verdict or "unknown")
    icon = VERDICT_ICONS.get(verdict, "?")
    return (
        f'<span class="fq-badge {verdict_badge_class(verdict)}" title="{esc(verdict)}">'
        f"<b>{icon}</b>{esc(label)}</span>"
    )


def consensus_badge_class(consensus: str) -> str:
    return {
        CONSENSUS_AGREE: "good",
        CONSENSUS_PARTIAL: "warn",
        CONSENSUS_DISAGREE: "bad",
    }.get(consensus, "neutral")


def _safe_url(url: Any) -> str:
    url = str(url or "").strip()
    return url if re.match(r"^https?://", url, re.IGNORECASE) else ""


def _fmt_conf(conf: Any) -> str:
    if isinstance(conf, (int, float)):
        return f"{conf:.2f}"
    return esc(conf) if conf else "—"


def render_consensus_box(consensus: ConsensusVerdict | None, summary: str | None) -> str:
    if not consensus and not summary:
        return ""

    parts = []
    if consensus and consensus.consensus != "unknown":
        badge_cls = consensus_badge_class(consensus.consensus)
        badge_lbl = CONSENSUS_LABELS.get(consensus.consensus, consensus.consensus)
        icon = CONSENSUS_ICONS.get(consensus.consensus, "⚖")
        conf_str = f" · confidence {_fmt_conf(consensus.confidence)}" if consensus.confidence is not None else ""
        parts.append(
            f'<div class="fq-consensus-head">'
            f'<span class="fq-badge {badge_cls}"><b>{icon}</b> {esc(badge_lbl)}</span>'
            f'<span class="fq-conf">{conf_str}</span>'
            f'</div>'
        )
        if consensus.explanation:
            parts.append(
                f'<div class="fq-consensus-exp fq-ar" dir="rtl">{esc(consensus.explanation)}</div>'
            )

    if summary and summary.strip():
        parts.append(
            f'<div class="fq-summary-block">'
            f'<div class="fq-label">ملخص ومقارنة أقوال العلماء (Comparative Summary)</div>'
            f'<div class="fq-summary-text fq-ar" dir="rtl">{esc(summary)}</div>'
            f'</div>'
        )

    if not parts:
        return ""
    return '<section class="fq-consensus-box">' + "".join(parts) + "</section>"


def render_query_panel(user_query, search_query, rewrite, errors, judge_map, judge_enabled, consensus_verdict=None) -> str:
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

    if consensus_verdict and consensus_verdict.consensus != "unknown":
        cls_name = consensus_badge_class(consensus_verdict.consensus)
        icon = CONSENSUS_ICONS.get(consensus_verdict.consensus, "⚖")
        chips.append(f'<span class="fq-pill {cls_name}">{icon} {CONSENSUS_LABELS.get(consensus_verdict.consensus, consensus_verdict.consensus)}</span>')

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
        state = "fail" if s.note == "failed" else "off" if s.note in ("disabled", "no candidates", "insufficient results") else "ok"
        left = cursor / denom * 100
        width = max(s.seconds / denom * 100, 0.8) if s.seconds > 0 else 0
        cursor += s.seconds
        meta = []
        if s.model:
            meta.append(f'<span class="fq-chip mono">{esc(s.model)}</span>')
        if s.note:
            meta.append(f'<span class="fq-chip note {state}">{esc(s.note)}</span>')
        meta_html = "".join(meta)

        items.append(
            f'<li class="fq-pstep {state}">'
            f'<div class="fq-step-num">{i}</div>'
            f'<div class="fq-step-main">'
            f'<div class="fq-step-head"><div class="fq-step-name">{esc(s.label)}{meta_html}</div>'
            f'<div class="fq-step-time">{s.seconds:.2f}s</div></div>'
            f'<div class="fq-track"><span class="fq-fill" style="left:{left:.2f}%;width:{width:.2f}%"></span></div>'
            f"</div></li>"
        )

    return (
        f'<section class="fq-pipe">'
        f'<div class="fq-kpis">{kpis}</div>'
        f'<ol class="fq-psteps">{"".join(items)}</ol>'
        f"</section>"
    )


def error_modal_html(errors: list[str], user_query: str, steps: list[Stat], tb: str | None = None) -> str:
    body = [f"User query:\n  {user_query}\n\nErrors encountered:\n"]
    for e in errors:
        body.append(f"- {e}\n")
    if tb:
        body.append(f"\nTraceback:\n{tb}\n")
    return "\n".join(body)


def judge_box(judge: FatwaJudgement) -> str:
    body = []
    if judge.answer:
        body.append(
            f'<div class="fq-judge-answer fq-ar" dir="rtl">{esc(judge.answer)}</div>'
        )
    if judge.explanation:
        body.append(
            f'<div class="fq-judge-exp fq-ar" dir="rtl">{esc(judge.explanation)}</div>'
        )
    if not body:
        return ""
    return (
        f'<section class="fq-judge">'
        f'<div class="fq-judge-head">{verdict_badge(judge.verdict)}'
        f'<span class="fq-conf">confidence {_fmt_conf(judge.confidence)}</span></div>'
        f'{"".join(body)}</section>'
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

    r_rank = int(row.get("rerank_rank", rank or 1))
    e_rank = int(row.get("retrieval_rank", 1))
    delta = e_rank - r_rank
    move = ""
    if delta:
        move = f'<span class="fq-m move {"up" if delta > 0 else "down"}">{"▲" if delta > 0 else "▼"} {abs(delta)}</span>'

    edge = f" v-{verdict_badge_class(judge.verdict)}" if judge is not None else ""
    top = " top" if rank == 1 else ""
    best = '<span class="fq-best">★ Best match</span>' if rank == 1 else ""
    judge_html = judge_box(judge) if judge is not None else ""
    open_attr = " open" if rank == 1 else ""

    scholar_label = esc(row.get("source", "Scholar"))

    return f"""
    <article class="fq-card{top}{edge}">
      <header class="fq-card-head">
        <div class="fq-rank">{r_rank}</div>
        <div class="fq-card-main">
          <h3 class="fq-card-title fq-ar" dir="rtl">{esc(row['title'])}</h3>
          <div class="fq-card-meta">
            {best}
            <span class="fq-scholar-badge">👤 {scholar_label}</span>
            {tags}
          </div>
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
          <span class="fq-m" title="Reranker score">rerank {float(row.get('reranker_score', 0.0)):.3f}</span>{move}
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
    out = pd.DataFrame()
    if "rerank_rank" in rows.columns:
        out["Rerank #"] = rows["rerank_rank"]
    if "retrieval_rank" in rows.columns:
        out["Retrieval #"] = rows["retrieval_rank"]
    if "source" in rows.columns:
        out["Scholar"] = rows["source"]
    out["Title"] = rows["title"]
    if "reranker_score" in rows.columns:
        out["Rerank score"] = rows["reranker_score"].map(lambda v: f"{float(v):.3f}")
    if "embedding_score" in rows.columns:
        out["Cosine"] = rows["embedding_score"].map(lambda v: f"{float(v):.3f}")
    if judge_map is not None:
        out["Verdict"] = rows["fatwa_id"].map(
            lambda fid: (judge_map[int(fid)].verdict if int(fid) in judge_map else "—")
        )
    return out


def df_to_html(df: Any) -> str:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return EMPTY_TABLE
    headers = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            val = row[c]
            cls = " fq-ar" if c in ("Title", "title") else ""
            cells.append(f"<td class='{cls}'>{esc(val)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f"<div class='fq-tbl-wrap'><table class='fq-tbl'>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
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
.gradio-container { max-width:1080px !important; margin:0 auto !important; padding-top:10px !important; }
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

/* ---- search controls & mode selector ---- */
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

#fq-mode-panel { background:var(--fq-surface); border:1px solid var(--fq-line); border-radius:18px;
  padding:14px 18px; margin-top:12px; box-shadow:var(--fq-shadow); }

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
.fq-badge { display:inline-flex; align-items:center; gap:6px; padding:4px 14px 4px 10px; border-radius:999px; font-size:.85rem; font-weight:700; }
.fq-badge b { font-size:.95rem; }
.fq-label { font-size:.72rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; color:var(--fq-muted); margin:0 0 6px; }

/* ---- consensus & summary box ---- */
.fq-consensus-box { margin:16px 0 20px; padding:20px 24px; border-radius:20px; background:var(--fq-surface);
  border:1.5px solid var(--fq-line); box-shadow:var(--fq-shadow); }
.fq-consensus-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
.fq-consensus-exp { font-size:1.15rem; line-height:1.7; color:var(--fq-ink); margin-bottom:14px; }
.fq-summary-block { margin-top:14px; padding-top:14px; border-top:1px dashed var(--fq-line); }
.fq-summary-text { font-size:1.2rem; line-height:1.8; color:var(--fq-ink); white-space:pre-line; }

.fq-scholar-badge { display:inline-flex; align-items:center; gap:5px; padding:3px 12px; border-radius:999px;
  background:var(--fq-accent-soft); color:var(--fq-accent-ink); font-size:.88rem; font-weight:800; border:1px solid color-mix(in srgb, var(--fq-accent) 30%, transparent); }

/* ---- query panel ---- */
.fq-query-panel { margin:16px 2px 6px; padding:14px 18px; border-radius:16px; background:var(--fq-surface-2); border:1px solid var(--fq-line); }
.fq-qrow { display:flex; gap:12px; align-items:baseline; padding:3px 0; }
.fq-qlabel { flex:none; width:92px; font-size:.7rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; color:var(--fq-muted); }
.fq-qtext { font-size:1rem; font-weight:500; }
.fq-qtext.fq-ar { font-size:1.3rem; }
.fq-qchips { display:flex; gap:8px; flex-wrap:wrap; margin:9px 0 0 104px; }
.fq-alert { margin-top:12px; padding:12px 16px; border-radius:12px; background:var(--fq-bad-soft); color:var(--fq-bad); }

/* ---- tabs ---- */
#fq-tabs { margin-top:14px; }
#fq-tabs button[role=tab] { font-weight:700; font-size:.95rem; }
#fq-tabs button[role=tab].selected { color:var(--fq-accent) !important; }

/* ---- result cards ---- */
.fq-card { position:relative; background:var(--fq-surface); border:1px solid var(--fq-line); border-radius:20px;
  padding:20px 24px 16px; margin:14px 0; box-shadow:var(--fq-shadow); --edge:var(--fq-accent); }
.fq-card::before { content:""; position:absolute; left:0; top:22px; bottom:22px; width:4px; border-radius:0 4px 4px 0; background:var(--edge); }
.fq-card.v-good { --edge:var(--fq-good); } .fq-card.v-warn { --edge:var(--fq-warn); } .fq-card.v-bad { --edge:var(--fq-bad); opacity:.92; }
.fq-card.top { border-color:color-mix(in srgb, var(--fq-gold) 55%, var(--fq-line)); }
.fq-card-head { display:flex; gap:16px; align-items:flex-start; }
.fq-rank { flex:none; width:38px; height:38px; border-radius:12px; background:var(--fq-accent-soft); color:var(--fq-accent-ink);
  display:grid; place-items:center; font-weight:800; font-size:1.05rem; }
.fq-card-main { flex:1; min-width:0; }
.fq-card-title { margin:0 0 8px !important; font-size:1.55rem !important; line-height:1.55 !important; font-weight:700; color:var(--fq-ink); text-align:right; }
.fq-card-meta { display:flex; flex-wrap:wrap; gap:6px 8px; align-items:center; }
.fq-best { font-size:.74rem; font-weight:800; color:var(--fq-gold); background:var(--fq-gold-soft); padding:2px 10px; border-radius:999px; }
.fq-cat { display:inline-block; padding:0 11px; border-radius:999px; background:var(--fq-accent-soft); color:var(--fq-accent-ink); font-size:1rem; line-height:1.7; }

.fq-judge { margin:16px 0 4px; padding:14px 18px 12px; border-radius:14px; background:var(--fq-surface-2); border:1px solid var(--fq-line); }
.fq-judge-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
.fq-conf { font-size:.78rem; font-weight:600; color:var(--fq-muted); text-transform:uppercase; letter-spacing:.05em; }
.fq-judge-answer { font-size:1.25rem; line-height:1.7; color:var(--fq-ink); font-weight:700; margin-bottom:4px; text-align:right; }
.fq-judge-exp { font-size:1.05rem; line-height:1.6; color:var(--fq-muted); text-align:right; }

.fq-qblock { margin:14px 0 10px; }
.fq-question { font-size:1.2rem; line-height:1.65; color:var(--fq-ink); text-align:right; }
.fq-det summary { cursor:pointer; font-weight:700; font-size:.86rem; color:var(--fq-muted); padding:6px 0; }
.fq-det summary:hover { color:var(--fq-accent); }
.fq-answer-text { padding:12px 14px; border-radius:12px; background:var(--fq-surface-2); font-size:1.2rem; line-height:1.8; color:var(--fq-ink); text-align:right; }

.fq-card-foot { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-top:14px; padding-top:12px; border-top:1px solid var(--fq-line); flex-wrap:wrap; }
.fq-metrics { display:flex; gap:10px; align-items:center; flex-wrap:wrap; font-size:.8rem; }
.fq-m { color:var(--fq-muted); font-variant-numeric:tabular-nums; }
.fq-m.move.up { color:var(--fq-good); font-weight:700; }
.fq-m.move.down { color:var(--fq-bad); font-weight:700; }
.fq-actions { display:flex; gap:8px; align-items:center; }
.fq-btn { border-radius:999px; padding:5px 14px; font-size:.82rem; font-weight:600; border:1px solid var(--fq-line); background:var(--fq-surface); color:var(--fq-ink); cursor:pointer; }
.fq-btn:hover { border-color:var(--fq-accent); color:var(--fq-accent); }
.fq-btn.primary { background:var(--fq-accent); color:#fff; border-color:var(--fq-accent); text-decoration:none; display:inline-block; }
.fq-btn.primary:hover { background:var(--fq-accent-ink); border-color:var(--fq-accent-ink); }

/* pipeline */
.fq-pipe { padding:8px 0; }
.fq-kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin-bottom:20px; }
.fq-kpi { padding:14px 18px; border-radius:16px; background:var(--fq-surface); border:1px solid var(--fq-line); }
.fq-kpi-label { font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; color:var(--fq-muted); }
.fq-kpi-value { font-size:1.8rem; font-weight:800; color:var(--fq-ink); margin:2px 0; }
.fq-kpi-sub { font-size:.8rem; color:var(--fq-muted); }
.fq-psteps { list-style:none; padding:0; margin:0; }
.fq-pstep { display:flex; gap:14px; align-items:center; padding:10px 0; border-bottom:1px solid var(--fq-line); }
.fq-step-num { width:28px; height:28px; border-radius:50%; background:var(--fq-surface-2); display:grid; place-items:center; font-weight:700; font-size:.85rem; }
.fq-step-main { flex:1; min-width:0; }
.fq-step-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px; }
.fq-step-name { font-weight:700; font-size:.92rem; display:flex; gap:8px; align-items:center; }
.fq-step-time { font-family:var(--fq-mono); font-size:.85rem; font-weight:600; color:var(--fq-muted); }
.fq-track { height:5px; border-radius:999px; background:var(--fq-surface-2); position:relative; overflow:hidden; }
.fq-fill { position:absolute; height:100%; border-radius:999px; background:var(--fq-accent); }
.fq-pstep.fail .fq-fill { background:var(--fq-bad); }

/* debug table */
.fq-tbl-wrap { overflow-x:auto; border-radius:14px; border:1px solid var(--fq-line); background:var(--fq-surface); }
.fq-tbl { width:100%; border-collapse:collapse; font-size:.86rem; text-align:left; }
.fq-tbl th, .fq-tbl td { padding:10px 14px; border-bottom:1px solid var(--fq-line); }
.fq-tbl th { background:var(--fq-surface-2); font-weight:800; text-transform:uppercase; font-size:.72rem; letter-spacing:.06em; color:var(--fq-muted); }
.fq-tbl td.fq-ar { text-align:right; font-family:var(--fq-ar); font-size:1.15rem; }
"""

BRAND_MARK = """
<svg class="fq-mark" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="22" cy="22" r="21" stroke="currentColor" stroke-width="2"/>
  <path d="M22 6 L25 18 L37 22 L25 26 L22 38 L19 26 L7 22 L19 18 Z" class="gold"/>
  <circle cx="22" cy="22" r="3" fill="currentColor"/>
</svg>
"""

COPY_JS = """
(function(btn){
  var card = btn.closest('.fq-card');
  var txt = card ? card.querySelector('.fq-answer-text') : null;
  if(!txt) return;
  navigator.clipboard.writeText(txt.innerText.trim()).then(function(){
    var old = btn.innerText; btn.innerText = 'Copied!';
    setTimeout(function(){ btn.innerText = old; }, 1800);
  });
})(this);
"""

EMPTY_QUERY = '<div class="fq-empty-sm">Ask a question above to start searching.</div>'
EMPTY_RESULTS = (
    '<div class="fq-empty"><div class="fq-empty-title">Ready to search</div>'
    '<div class="fq-empty-text">Enter a question in Arabic or any language. Multiple scholar datasets are indexed and ready to compare.</div></div>'
)
EMPTY_PIPELINE = '<div class="fq-empty-sm">Pipeline statistics appear here after your first search.</div>'
EMPTY_TABLE = '<div class="fq-empty-sm">No data yet.</div>'
EMPTY_FAILED = '<div class="fq-empty-sm">Search produced no results.</div>'

LOADING_QUERY = (
    '<div class="fq-loading"><div class="fq-spin"></div>'
    '<span>Searching and synthesizing across scholarly fatwas…</span></div>'
)
LOADING_RESULTS = '<div class="fq-empty"><div class="fq-spin" style="margin:0 auto 12px"></div><div class="fq-empty-title">Retrieving & analyzing…</div></div>'

EXAMPLE_QUERIES = [
    "كيف أكفر عن يمينٍ حلفتها ثم حنثت؟",
    "ما حكم صلاة الجماعة في المسجد؟",
    "هل يجوز إخراج زكاة الفطر نقداً؟",
    "What is the ruling on paying zakat on gold?",
]


# ---------------------------------------------------------------------------
# UI Construction
# ---------------------------------------------------------------------------


def build_ui(app: FatwaApp) -> gr.Blocks:
    n_fatwas = len(app.df) if hasattr(app, "df") else 0
    emb_dev = getattr(getattr(app, "s", None), "device_embedder", "?")
    rer_dev = getattr(getattr(app, "s", None), "device_reranker", "?")
    available_scholars = getattr(app, "available_scholars", ["Ibn Baz"])

    def key_chip(env: str, name: str) -> str:
        ok = bool(os.getenv(env))
        return f'<span class="fq-key {"ok" if ok else "no"}">{"✓" if ok else "✕"} {name}</span>'

    with gr.Blocks(title="Fatwa Search — Multi-Scholar RAG", theme=THEME, css=CSS) as demo:
        sidebar_open = gr.State(False)

        # ---- top bar -------------------------------------------------------
        with gr.Row(elem_id="fq-top"):
            gr.HTML(
                f"<div class='fq-brand'>{BRAND_MARK}<div><div class='fq-name'>Fatwa Search — Multi-Scholar RAG</div>"
                f"<div class='fq-tagline'>{n_fatwas:,} fatwas indexed across {len(available_scholars)} scholarly sources · semantic retrieval + reranking + AI synthesis</div></div></div>",
                elem_classes="fq-flat",
            )
            settings_btn = gr.Button("⚙  Settings", elem_id="fq-settings-btn", size="sm", scale=0)

        # ---- search box ----------------------------------------------------
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

        # ---- mode & scholar selectors --------------------------------------
        with gr.Group(elem_id="fq-mode-panel"):
            with gr.Row():
                search_mode = gr.Radio(
                    choices=SEARCH_MODES,
                    value=SEARCH_MODE_PER_SCHOLAR,
                    label="وضع البحث (Search Mode)",
                    info="اختر بين بحث مخصص لكل شيخ على حدة أو ترتيب مجمع لكافة النتائج",
                    scale=1,
                )
                with gr.Column(scale=1):
                    x_per_scholar = gr.Slider(
                        1, 5, value=DEFAULT_X_PER_SCHOLAR, step=1,
                        label="عدد الإجابات لكل شيخ (Answers per scholar)",
                        info="في وضع 'X إجابة لكل شيخ'",
                        visible=True,
                    )
                    show_n = gr.Slider(
                        1, 20, value=DEFAULT_SHOW_N, step=1,
                        label="إجمالي النتائج المعروضة (Total results to show)",
                        info="في وضع 'أفضل X إجابات مجمعة'",
                        visible=False,
                    )
            with gr.Row():
                selected_scholars = gr.CheckboxGroup(
                    choices=available_scholars,
                    value=available_scholars,
                    label="العلماء المشمولون في البحث (Selected Scholars / Sources)",
                    info="اختر أي عدد من العلماء للبحث في فتاواهم",
                )

        # ---- example queries -----------------------------------------------
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
                    "<div class='fq-sect-sub'>The fatwas that were shown/judged.</div>",
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
        with gr.Sidebar(label="Settings", open=False, position="right", width=420) as sidebar:
            gr.HTML(
                "<div class='fq-side-title'>Settings</div>"
                "<div class='fq-side-sub'>Configure AI features, models, retrieval pools and filters.</div>",
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
            with gr.Accordion("Fatwa Verdict Judge", open=True):
                judge_enabled = gr.Checkbox(
                    value=bool(os.getenv(GEMINI_API_KEY_ENV)), label="Judge each result",
                    info="An LLM checks whether each fatwa answers the question.",
                )
                judge_provider = gr.Radio(PROVIDERS, value=PROVIDER_GEMINI, label="Provider")
                judge_model = gr.Dropdown(
                    models_for("judge", PROVIDER_GEMINI), value=GEMINI_MODELS[0],
                    label="Model", allow_custom_value=True,
                )
            with gr.Accordion("AI Consensus & Summary", open=True):
                consensus_enabled = gr.Checkbox(
                    value=bool(os.getenv(GEMINI_API_KEY_ENV)), label="Consensus Classification",
                    info="Classify whether scholars agree, partially agree, or disagree (Gemini or JEV).",
                )
                consensus_provider = gr.Radio(PROVIDERS, value=PROVIDER_GEMINI, label="Consensus Provider")
                consensus_model = gr.Dropdown(
                    models_for("consensus", PROVIDER_GEMINI), value=GEMINI_MODELS[0],
                    label="Consensus Model", allow_custom_value=True,
                )
                summary_enabled = gr.Checkbox(
                    value=bool(os.getenv(GEMINI_API_KEY_ENV)), label="Comparative Summary (Gemini)",
                    info="Generates an Arabic synthesis comparing what the scholars say.",
                )
                summary_model = gr.Dropdown(
                    models_for("rewrite", PROVIDER_GEMINI), value=GEMINI_MODELS[0],
                    label="Summary Model (Gemini)", allow_custom_value=True,
                )
            with gr.Accordion("Retrieval & Thresholds", open=False):
                retrieve_k = gr.Slider(5, MAX_RETRIEVE_K, value=DEFAULT_RETRIEVE_K, step=1,
                                       label="Candidates to retrieve", info="Raw embedding top-K")
                rerank_pool = gr.Slider(1, DEFAULT_RETRIEVE_K, value=DEFAULT_RERANK_POOL, step=1,
                                        label="Candidates to rerank")
                max_candidate_chars = gr.Slider(500, 8000, value=DEFAULT_MAX_CANDIDATE_CHARS, step=250,
                                                label="Judge text limit")
                min_rerank_score = gr.Number(value=None, label="Minimum rerank score", info="Leave blank for no filter.")
            with gr.Accordion("Filter verdicts", open=False):
                filter_verdicts = gr.CheckboxGroup(
                    [(VERDICT_LABELS[v], v) for v in VERDICTS], value=list(VERDICTS),
                    show_label=False, info="Applies when judging is on.",
                )
            with gr.Accordion("System", open=False):
                gr.HTML(
                    f"<div class='fq-keys'>{key_chip(GEMINI_API_KEY_ENV, 'Gemini key')}"
                    f"{key_chip(OPENROUTER_API_KEY_ENV, 'OpenRouter key')}</div>"
                    "<dl class='fq-sys' style='margin-top:12px'>"
                    f"<dt>Embedder</dt><dd>{esc(EMBEDDER_MODEL_NAME)} · {esc(emb_dev)}</dd>"
                    f"<dt>Reranker</dt><dd>{esc(RERANKER_MODEL_NAME)} · {esc(rer_dev)}</dd>"
                    f"<dt>Corpus</dt><dd>{n_fatwas:,} fatwas · {len(available_scholars)} scholars</dd></dl>",
                    elem_classes="fq-flat",
                )
            reset_btn = gr.Button("Reset to defaults", size="sm")

        # ---- wiring --------------------------------------------------------
        inputs = [
            query,
            search_mode, selected_scholars, x_per_scholar,
            rewrite_enabled, rewrite_provider, rewrite_model,
            judge_enabled, judge_provider, judge_model,
            consensus_enabled, consensus_provider, consensus_model,
            summary_enabled, summary_model,
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

        settings_btn.click(
            lambda is_open: (not is_open, gr.Sidebar(open=not is_open)),
            inputs=sidebar_open, outputs=[sidebar_open, sidebar], show_progress="hidden",
        )
        sidebar.expand(lambda: True, outputs=sidebar_open, show_progress="hidden")
        sidebar.collapse(lambda: False, outputs=sidebar_open, show_progress="hidden")

        # Toggle dynamic controls based on search mode
        search_mode.change(
            lambda mode: (
                gr.update(visible=(mode == SEARCH_MODE_PER_SCHOLAR)),
                gr.update(visible=(mode == SEARCH_MODE_OVERALL)),
            ),
            inputs=search_mode,
            outputs=[x_per_scholar, show_n],
            show_progress="hidden",
        )

        # Dependent controls
        rewrite_enabled.change(
            lambda e: (gr.update(interactive=e), gr.update(interactive=e)),
            inputs=rewrite_enabled, outputs=[rewrite_provider, rewrite_model], show_progress="hidden",
        )
        judge_enabled.change(
            lambda e: (gr.update(interactive=e), gr.update(interactive=e)),
            inputs=judge_enabled, outputs=[judge_provider, judge_model], show_progress="hidden",
        )
        consensus_enabled.change(
            lambda e: (gr.update(interactive=e), gr.update(interactive=e)),
            inputs=consensus_enabled, outputs=[consensus_provider, consensus_model], show_progress="hidden",
        )
        summary_enabled.change(
            lambda e: gr.update(interactive=e),
            inputs=summary_enabled, outputs=summary_model, show_progress="hidden",
        )

        rewrite_provider.change(
            lambda p: gr.update(choices=models_for("rewrite", p), value=models_for("rewrite", p)[0]),
            inputs=rewrite_provider, outputs=rewrite_model,
        )
        judge_provider.change(
            lambda p: gr.update(choices=models_for("judge", p), value=models_for("judge", p)[0]),
            inputs=judge_provider, outputs=judge_model,
        )
        consensus_provider.change(
            lambda p: gr.update(choices=models_for("consensus", p), value=models_for("consensus", p)[0]),
            inputs=consensus_provider, outputs=consensus_model,
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
                True, PROVIDER_GEMINI, GEMINI_MODELS[0],
                True, GEMINI_MODELS[0],
                DEFAULT_RETRIEVE_K, DEFAULT_RERANK_POOL, DEFAULT_SHOW_N,
                DEFAULT_MAX_CANDIDATE_CHARS, None, list(VERDICTS),
            )

        reset_btn.click(
            reset_settings,
            outputs=[
                rewrite_enabled, rewrite_provider, rewrite_model,
                judge_enabled, judge_provider, judge_model,
                consensus_enabled, consensus_provider, consensus_model,
                summary_enabled, summary_model,
                retrieve_k, rerank_pool, show_n,
                max_candidate_chars, min_rerank_score, filter_verdicts,
            ],
        )

    return demo


# ---------------------------------------------------------------------------
# CLI & Entrypoint
# ---------------------------------------------------------------------------


def ensure_data(data_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if multi-scholar assets are already in place
    has_multi = (data_dir / "multi_scholar_clean.csv").exists() and (
        data_dir / "qwen3_06b_multi_scholar_embeddings.pt"
    ).exists()
    if has_multi:
        return

    # Check if Bin Baz assets are already in place
    has_binbaz = (data_dir / "binbaz_clean.csv").exists() and (
        data_dir / "qwen3_06b_title_answer_embeddings.pt"
    ).exists()

    # Try downloading multi-scholar files from DATA_REPO_ID first
    print(f"Checking for multi-scholar assets in Hugging Face dataset '{DATA_REPO_ID}'...")
    multi_success = True
    for name in MULTI_SCHOLAR_DATA_FILES:
        target = data_dir / name
        if target.exists():
            continue
        try:
            print(f"Downloading {name} from {DATA_REPO_ID} ...")
            hf_hub_download(repo_id=DATA_REPO_ID, repo_type="dataset", filename=name, local_dir=data_dir)
        except Exception as e:
            print(f"Notice: Could not download {name} from {DATA_REPO_ID} ({type(e).__name__}).")
            multi_success = False
            break

    if multi_success and (data_dir / "multi_scholar_clean.csv").exists():
        print("Successfully fetched multi-scholar assets from Hugging Face.")
        return

    # If multi-scholar files were not found on HF repo and not already on disk, fall back to Bin Baz
    if not has_binbaz:
        print(f"Falling back to Bin Baz base dataset in '{DATA_REPO_ID}'...")
        for name in BINBAZ_DATA_FILES:
            if (data_dir / name).exists():
                continue
            try:
                print(f"Downloading {name} from {DATA_REPO_ID} ...")
                hf_hub_download(repo_id=DATA_REPO_ID, repo_type="dataset", filename=name, local_dir=data_dir)
            except Exception as e:
                print(f"Warning: Failed to download {name}: {e}")


def default_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arabic Fatwa Q&A: semantic search and reranking over trusted Islamic fatwas."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("FATWA_DATA_DIR", APP_DIR / "data")),
        help="Folder with cleaned CSV + embeddings.pt + metadata.json.",
    )
    parser.add_argument("--no-download", action="store_true", help="Do not fetch missing data files from Hugging Face.")
    parser.add_argument("--embedder-device", choices=["cpu", "cuda", "auto"], default="cpu", help="Embedder device.")
    parser.add_argument("--reranker-device", choices=["auto", "cuda", "cpu"], default="auto", help="Reranker device.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip the startup warmup query.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=7860, help="Port (default 7860).")
    parser.add_argument("--share", action="store_true", help="Create a temporary public *.gradio.live link.")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser tab.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.no_download:
        ensure_data(args.data_dir)

    settings = Settings(
        data_dir=args.data_dir,
        device_embedder=default_device(args.embedder_device),
        device_reranker=default_device(args.reranker_device),
        warmup=not args.no_warmup,
    )

    app = FatwaApp(settings)
    demo = build_ui(app)

    url = f"http://{args.host}:{args.port}"
    print(f"\nFatwa Search running on {url}")
    if not args.no_browser and not args.share:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
