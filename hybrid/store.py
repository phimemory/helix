"""
Fact stores — where the *readable* payload lives.

  FactStore     : local SQLite (always works, zero deps beyond stdlib)
  HelixDBStore  : optional HTTP adapter to HelixDB graph-vector DB
                  (https://helix-db.com) when HELIXDB_URL is set

Helix phase φ is the trajectory key.
Facts (text, metadata, embeddings) live here so an LLM can read them.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class Fact:
    id: str
    session_id: str
    text: str
    label: str
    step: int
    embedding: List[float]
    phi: List[float]
    created_at: float
    meta: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


class BaseStore(ABC):
    @abstractmethod
    def add(self, fact: Fact) -> str:
        ...

    @abstractmethod
    def get(self, fact_id: str) -> Optional[Fact]:
        ...

    @abstractmethod
    def list_session(self, session_id: str) -> List[Fact]:
        ...

    @abstractmethod
    def all_facts(self) -> List[Fact]:
        ...

    @abstractmethod
    def search_by_phi(
        self, query_phi: Sequence[float], top_k: int = 5, session_id: Optional[str] = None
    ) -> List[Tuple[Fact, float]]:
        ...

    @abstractmethod
    def search_by_embedding(
        self, query_emb: Sequence[float], top_k: int = 5, session_id: Optional[str] = None
    ) -> List[Tuple[Fact, float]]:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _phi_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Phase similarity via mean cos of angle differences.
    Higher = more similar trajectories.
    """
    if a.shape != b.shape:
        n = min(a.size, b.size)
        a, b = a.ravel()[:n], b.ravel()[:n]
    return float(np.cos(a - b).mean())


class FactStore(BaseStore):
    """SQLite-backed fact store. Default backend."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                text TEXT NOT NULL,
                label TEXT NOT NULL,
                step INTEGER NOT NULL,
                embedding TEXT NOT NULL,
                phi TEXT NOT NULL,
                created_at REAL NOT NULL,
                meta TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id)"
        )
        self._conn.commit()

    def add(self, fact: Fact) -> str:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO facts
            (id, session_id, text, label, step, embedding, phi, created_at, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact.id,
                fact.session_id,
                fact.text,
                fact.label,
                fact.step,
                json.dumps(fact.embedding),
                json.dumps(fact.phi),
                fact.created_at,
                json.dumps(fact.meta),
            ),
        )
        self._conn.commit()
        return fact.id

    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        return Fact(
            id=row["id"],
            session_id=row["session_id"],
            text=row["text"],
            label=row["label"],
            step=row["step"],
            embedding=json.loads(row["embedding"]),
            phi=json.loads(row["phi"]),
            created_at=row["created_at"],
            meta=json.loads(row["meta"]),
        )

    def get(self, fact_id: str) -> Optional[Fact]:
        cur = self._conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,))
        row = cur.fetchone()
        return self._row_to_fact(row) if row else None

    def list_session(self, session_id: str) -> List[Fact]:
        cur = self._conn.execute(
            "SELECT * FROM facts WHERE session_id = ? ORDER BY step ASC",
            (session_id,),
        )
        return [self._row_to_fact(r) for r in cur.fetchall()]

    def all_facts(self) -> List[Fact]:
        cur = self._conn.execute("SELECT * FROM facts ORDER BY created_at ASC")
        return [self._row_to_fact(r) for r in cur.fetchall()]

    def search_by_phi(
        self, query_phi: Sequence[float], top_k: int = 5, session_id: Optional[str] = None
    ) -> List[Tuple[Fact, float]]:
        q = np.asarray(query_phi, dtype=np.float64)
        facts = self.list_session(session_id) if session_id else self.all_facts()
        scored = []
        for f in facts:
            p = np.asarray(f.phi, dtype=np.float64)
            scored.append((f, _phi_similarity(q, p)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_by_embedding(
        self, query_emb: Sequence[float], top_k: int = 5, session_id: Optional[str] = None
    ) -> List[Tuple[Fact, float]]:
        q = np.asarray(query_emb, dtype=np.float64)
        facts = self.list_session(session_id) if session_id else self.all_facts()
        scored = []
        for f in facts:
            e = np.asarray(f.embedding, dtype=np.float64)
            scored.append((f, _cosine(q, e)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        self._conn.execute("DELETE FROM facts")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class HelixDBStore(BaseStore):
    """
    Optional adapter for HelixDB (https://helix-db.com).

    Requires a running instance, e.g.:
      HELIXDB_URL=http://localhost:6969

    Uses the REST /v1/query dynamic endpoint when available.
    Falls back to an in-process FactStore mirror so demos never break
    if HelixDB is offline — writes go to both; reads prefer local cache
    unless force_remote=True.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        local_cache_path: str | Path = ":memory:",
        timeout: float = 5.0,
    ):
        self.url = (url or os.environ.get("HELIXDB_URL") or "").rstrip("/")
        self.timeout = timeout
        self.local = FactStore(local_cache_path)
        self.remote_ok = False
        if self.url:
            self.remote_ok = self._ping()

    def _ping(self) -> bool:
        try:
            import urllib.request

            req = urllib.request.Request(self.url + "/health", method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return 200 <= resp.status < 300
        except Exception:
            # try root
            try:
                import urllib.request

                req = urllib.request.Request(self.url + "/", method="GET")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return 200 <= resp.status < 500
            except Exception:
                return False

    def _post_query(self, payload: dict) -> Optional[dict]:
        if not self.url or not self.remote_ok:
            return None
        try:
            import urllib.request

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.url + "/v1/query",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            self.remote_ok = False
            return None

    def add(self, fact: Fact) -> str:
        # always write local (source of truth for this hybrid layer)
        self.local.add(fact)
        # best-effort remote: store as a document/node payload
        self._post_query(
            {
                "action": "upsert_helix_fact",
                "fact": fact.to_dict(),
                "note": (
                    "Helix hybrid memory fact — trajectory key is phi; "
                    "text is the LLM-readable payload"
                ),
            }
        )
        return fact.id

    def get(self, fact_id: str) -> Optional[Fact]:
        return self.local.get(fact_id)

    def list_session(self, session_id: str) -> List[Fact]:
        return self.local.list_session(session_id)

    def all_facts(self) -> List[Fact]:
        return self.local.all_facts()

    def search_by_phi(
        self, query_phi: Sequence[float], top_k: int = 5, session_id: Optional[str] = None
    ) -> List[Tuple[Fact, float]]:
        return self.local.search_by_phi(query_phi, top_k, session_id)

    def search_by_embedding(
        self, query_emb: Sequence[float], top_k: int = 5, session_id: Optional[str] = None
    ) -> List[Tuple[Fact, float]]:
        return self.local.search_by_embedding(query_emb, top_k, session_id)

    def clear(self) -> None:
        self.local.clear()

    def status(self) -> dict:
        return {
            "backend": "helixdb+sqlite",
            "url": self.url or None,
            "remote_ok": self.remote_ok,
            "facts": len(self.local.all_facts()),
        }


def make_store(
    backend: str = "sqlite",
    path: str | Path = ":memory:",
    helixdb_url: Optional[str] = None,
) -> BaseStore:
    """
    backend: 'sqlite' | 'helixdb'
    """
    if backend == "helixdb":
        return HelixDBStore(url=helixdb_url, local_cache_path=path)
    return FactStore(path)


def new_fact(
    session_id: str,
    text: str,
    label: str,
    step: int,
    embedding: torch.Tensor,
    phi: torch.Tensor,
    meta: Optional[dict] = None,
) -> Fact:
    emb = embedding.detach().cpu().float().view(-1).tolist()
    p = phi.detach().cpu().float().view(-1).tolist()
    return Fact(
        id=str(uuid.uuid4()),
        session_id=session_id,
        text=text,
        label=label,
        step=step,
        embedding=emb,
        phi=p,
        created_at=time.time(),
        meta=meta or {},
    )
