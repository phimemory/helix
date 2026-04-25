"""
Helix Memory Server
A REST + MCP-compatible API server exposing Helix crystal operations.

Endpoints:
  POST   /v1/crystal/{id}/absorb      - feed text/embedding into crystal
  GET    /v1/crystal/{id}/recall      - get feature vector from crystal
  GET    /v1/crystal/{id}/at/{step}   - recall at specific timestep (TPI)
  GET    /v1/crystal/{id}/info        - crystal metadata and stats
  DELETE /v1/crystal/{id}             - delete a crystal
  POST   /v1/crystals/merge           - merge multiple crystals (FPA)
  POST   /v1/crystal/{id}/collapse    - trigger phase collapse event (PCE)
  GET    /v1/crystal/{id}/diff        - diff two crystal states (PDP)
  GET    /v1/crystal/{id}/export      - download .hx file
  POST   /v1/crystal/{id}/import      - upload .hx file

MCP Tool Definitions (Claude/OpenAI compatible):
  GET    /mcp/tools                   - list all tools in MCP format
  POST   /mcp/call                    - invoke a tool by name
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import uuid
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from crystal.substrate import MemoryCrystal
from crystal.phase_collapse import PhaseCollapseRegister
from crystal.federation import PhaseFederation
from crystal.temporal_index import TemporalPhaseIndex
from crystal.phase_diff import PhaseDiff
from crystal.phicrypt import PhiCrypt


# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Helix Memory Server",
    description="Model-agnostic persistent memory API backed by Helix phase geometry",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-Memory Crystal Store ──────────────────────────────────────────────────

# Production would use Redis or a persistent store
# For now: in-memory dict of {crystal_id: CrystalSession}
crystal_store: Dict[str, Dict] = {}
temporal_index_store: Dict[str, TemporalPhaseIndex] = {}
collapse_store: Dict[str, PhaseCollapseRegister] = {}

CRYSTAL_DIR = Path(__file__).parent / "crystals"
CRYSTAL_DIR.mkdir(exist_ok=True)

DEFAULT_INPUT_SIZE = 768   # text embedding size (sentence-transformers default)
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_HARMONICS = [1, 2, 4, 8]


def get_crystal(crystal_id: str) -> MemoryCrystal:
    if crystal_id not in crystal_store:
        raise HTTPException(status_code=404, detail=f"Crystal '{crystal_id}' not found")
    return crystal_store[crystal_id]["crystal"]


def get_or_create_crystal(crystal_id: str, input_size: int = DEFAULT_INPUT_SIZE) -> MemoryCrystal:
    if crystal_id not in crystal_store:
        crystal = MemoryCrystal(
            input_size=input_size,
            hidden_size=DEFAULT_HIDDEN_SIZE,
            harmonics=DEFAULT_HARMONICS
        )
        crystal_store[crystal_id] = {
            "crystal": crystal,
            "created_at": str(torch.tensor(0)),
            "metadata": {}
        }
        temporal_index_store[crystal_id] = TemporalPhaseIndex(
            hidden_size=DEFAULT_HIDDEN_SIZE,
            snapshot_interval=10
        )
        collapse_store[crystal_id] = PhaseCollapseRegister(num_flags=32)
    return crystal_store[crystal_id]["crystal"]


# ─── Request/Response Models ──────────────────────────────────────────────────

class AbsorbTextRequest(BaseModel):
    text: Optional[str] = None
    embedding: Optional[List[float]] = None
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    metadata: Optional[Dict[str, Any]] = None

class AbsorbResponse(BaseModel):
    crystal_id: str
    absorb_count: int
    confidence: float
    crystal_size_bytes: int

class RecallResponse(BaseModel):
    crystal_id: str
    features: List[float]
    feature_dim: int
    absorb_count: int
    crystal_size_bytes: int

class CrystalInfoResponse(BaseModel):
    crystal_id: str
    absorb_count: int
    hidden_size: int
    harmonics: List[int]
    crystal_size_bytes: int
    num_snapshots: int
    num_collapsed_flags: int

class MergeRequest(BaseModel):
    crystal_ids: List[str]
    weights: Optional[List[float]] = None
    output_id: Optional[str] = None

class CollapseRequest(BaseModel):
    flag_index: int
    flag_name: Optional[str] = None

class MCPToolCallRequest(BaseModel):
    tool: str
    parameters: Dict[str, Any]


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "crystals_in_memory": len(crystal_store),
        "version": "1.0.0"
    }


# ─── Core Crystal Endpoints ───────────────────────────────────────────────────

@app.post("/v1/crystal/{crystal_id}/absorb", response_model=AbsorbResponse)
def absorb(crystal_id: str, body: AbsorbTextRequest):
    """
    Feed text or a raw embedding vector into a crystal.
    If text is provided, a simple mean-pool encoding is used.
    For production, pass pre-computed embeddings directly.
    """
    crystal = get_or_create_crystal(crystal_id)
    tpi = temporal_index_store[crystal_id]

    if body.embedding is not None:
        embedding = torch.tensor(body.embedding, dtype=torch.float32)
        if embedding.shape[0] != crystal.input_size:
            raise HTTPException(
                status_code=422,
                detail=f"Embedding dim {embedding.shape[0]} != crystal input_size {crystal.input_size}"
            )
    elif body.text is not None:
        # Simple character-level encoding when no embedding model available
        # In production this would call a real embedding model
        embedding = _text_to_embedding(body.text, crystal.input_size)
    else:
        raise HTTPException(status_code=422, detail="Provide either 'text' or 'embedding'")

    # Apply weight
    if body.weight != 1.0:
        embedding = embedding * body.weight

    confidence = crystal.absorb(embedding)

    # Record in temporal index
    step = crystal.absorb_count - 1
    tpi.record(step, crystal.recall_compact())

    return AbsorbResponse(
        crystal_id=crystal_id,
        absorb_count=crystal.absorb_count,
        confidence=float(confidence) if confidence is not None else 0.0,
        crystal_size_bytes=crystal.size_bytes()
    )


@app.get("/v1/crystal/{crystal_id}/recall", response_model=RecallResponse)
def recall(crystal_id: str):
    """
    Extract the full harmonic feature vector from a crystal.
    This is what you inject into an LLM's context.
    """
    crystal = get_crystal(crystal_id)
    features = crystal.recall()

    return RecallResponse(
        crystal_id=crystal_id,
        features=features.tolist(),
        feature_dim=features.shape[0],
        absorb_count=crystal.absorb_count,
        crystal_size_bytes=crystal.size_bytes()
    )


@app.get("/v1/crystal/{crystal_id}/at/{step}")
def recall_at_step(crystal_id: str, step: int):
    """
    Retrieve the phase state at a specific timestep (Temporal Phase Indexing).
    Interpolates if the exact step wasn't recorded.
    """
    if crystal_id not in temporal_index_store:
        raise HTTPException(status_code=404, detail=f"Crystal '{crystal_id}' not found")

    tpi = temporal_index_store[crystal_id]
    if tpi.num_snapshots() == 0:
        raise HTTPException(status_code=400, detail="No snapshots recorded yet")

    phi = tpi.recall_at(step)
    features = tpi.recall_features_at(step, harmonics=DEFAULT_HARMONICS)

    return {
        "crystal_id": crystal_id,
        "step": step,
        "phi": phi.tolist(),
        "features": features.tolist(),
        "num_snapshots": tpi.num_snapshots()
    }


@app.get("/v1/crystal/{crystal_id}/info", response_model=CrystalInfoResponse)
def crystal_info(crystal_id: str):
    """Get metadata and stats about a crystal."""
    crystal = get_crystal(crystal_id)
    tpi = temporal_index_store.get(crystal_id)
    pcr = collapse_store.get(crystal_id)

    return CrystalInfoResponse(
        crystal_id=crystal_id,
        absorb_count=crystal.absorb_count,
        hidden_size=crystal.hidden_size,
        harmonics=[int(h) for h in crystal.harmonics],
        crystal_size_bytes=crystal.size_bytes(),
        num_snapshots=tpi.num_snapshots() if tpi else 0,
        num_collapsed_flags=pcr.num_collapsed() if pcr else 0
    )


@app.delete("/v1/crystal/{crystal_id}")
def delete_crystal(crystal_id: str):
    """Delete a crystal from the store."""
    if crystal_id not in crystal_store:
        raise HTTPException(status_code=404, detail=f"Crystal '{crystal_id}' not found")
    del crystal_store[crystal_id]
    temporal_index_store.pop(crystal_id, None)
    collapse_store.pop(crystal_id, None)
    return {"deleted": crystal_id}


@app.get("/v1/crystals")
def list_crystals():
    """List all crystals in the store."""
    return {
        "crystals": [
            {
                "id": cid,
                "absorb_count": data["crystal"].absorb_count,
                "size_bytes": data["crystal"].size_bytes()
            }
            for cid, data in crystal_store.items()
        ],
        "total": len(crystal_store)
    }


# ─── Merge (Federated Phase Alignment) ───────────────────────────────────────

@app.post("/v1/crystals/merge")
def merge_crystals(body: MergeRequest):
    """
    Merge multiple crystals into one using Federated Phase Alignment.
    Privacy-preserving: the merge is a circular mean of phase angles.
    """
    if len(body.crystal_ids) < 2:
        raise HTTPException(status_code=422, detail="Provide at least 2 crystal IDs to merge")

    phase_states = []
    for cid in body.crystal_ids:
        crystal = get_crystal(cid)
        phase_states.append(crystal.recall_compact())

    fed = PhaseFederation()
    weights = torch.tensor(body.weights, dtype=torch.float32) if body.weights else None
    merged_phi = fed.merge(phase_states, weights)

    # Create output crystal
    output_id = body.output_id or f"merged_{uuid.uuid4().hex[:8]}"
    source_crystal = crystal_store[body.crystal_ids[0]]["crystal"]

    merged_crystal = MemoryCrystal(
        input_size=source_crystal.input_size,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        harmonics=DEFAULT_HARMONICS
    )
    merged_crystal.phi_state = merged_phi.unsqueeze(0)
    merged_crystal.absorb_count = sum(
        crystal_store[cid]["crystal"].absorb_count for cid in body.crystal_ids
    )

    crystal_store[output_id] = {"crystal": merged_crystal, "metadata": {}}
    temporal_index_store[output_id] = TemporalPhaseIndex(
        hidden_size=DEFAULT_HIDDEN_SIZE, snapshot_interval=10
    )
    collapse_store[output_id] = PhaseCollapseRegister(num_flags=32)

    # Alignment scores between source crystals
    alignments = {}
    for i, cid_a in enumerate(body.crystal_ids):
        for cid_b in body.crystal_ids[i+1:]:
            score = fed.alignment_score(
                crystal_store[cid_a]["crystal"].recall_compact(),
                crystal_store[cid_b]["crystal"].recall_compact()
            )
            alignments[f"{cid_a}_vs_{cid_b}"] = round(score, 3)

    return {
        "output_crystal_id": output_id,
        "merged_from": body.crystal_ids,
        "alignment_scores": alignments,
        "output_size_bytes": merged_crystal.size_bytes()
    }


# ─── Phase Collapse Events ────────────────────────────────────────────────────

@app.post("/v1/crystal/{crystal_id}/collapse")
def collapse_flag(crystal_id: str, body: CollapseRequest):
    """
    Trigger an irreversible Phase Collapse Event on a flag neuron.
    Once collapsed, this flag can never be unset.
    Use for permanent binary facts.
    """
    if crystal_id not in collapse_store:
        raise HTTPException(status_code=404, detail=f"Crystal '{crystal_id}' not found")

    pcr = collapse_store[crystal_id]

    if body.flag_name:
        try:
            pcr.register_flag(body.flag_index, body.flag_name)
            result = pcr.collapse_named(body.flag_name)
        except AssertionError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        result = pcr.collapse(body.flag_index)

    return {
        "crystal_id": crystal_id,
        "flag_index": body.flag_index,
        "flag_name": body.flag_name,
        "newly_collapsed": result,
        "total_collapsed": pcr.num_collapsed()
    }


@app.get("/v1/crystal/{crystal_id}/flags")
def query_flags(crystal_id: str):
    """Query all collapse flags for a crystal."""
    if crystal_id not in collapse_store:
        raise HTTPException(status_code=404, detail=f"Crystal '{crystal_id}' not found")
    pcr = collapse_store[crystal_id]
    return {
        "crystal_id": crystal_id,
        "flags": {
            str(i): bool(pcr.query(i)) for i in range(pcr.num_flags)
        },
        "total_collapsed": pcr.num_collapsed(),
        "state_vector": pcr.get_state_vector().tolist()
    }


# ─── Phase Diff ───────────────────────────────────────────────────────────────

@app.get("/v1/crystals/diff")
def diff_crystals(crystal_id_a: str, crystal_id_b: str):
    """
    Compute the phase diff between two crystals (Phase Diff Protocol).
    Returns what changed between them and which neurons changed most.
    """
    crystal_a = get_crystal(crystal_id_a)
    crystal_b = get_crystal(crystal_id_b)

    differ = PhaseDiff()
    changeset = differ.diff(crystal_a.recall_compact(), crystal_b.recall_compact())

    return {
        "crystal_a": crystal_id_a,
        "crystal_b": crystal_id_b,
        "unchanged_neurons": changeset.num_unchanged(),
        "minor_changes": changeset.num_minor_changes(),
        "major_changes": changeset.num_major_changes(),
        "total_rotation_rad": round(changeset.total_rotation(), 4),
        "mean_rotation_rad": round(changeset.mean_rotation(), 4),
        "max_rotation_rad": round(changeset.max_rotation(), 4),
        "most_changed_neurons": changeset.most_changed_neurons(5),
        "delta": changeset.delta.tolist()
    }


# ─── Import / Export ─────────────────────────────────────────────────────────

@app.get("/v1/crystal/{crystal_id}/export")
def export_crystal(crystal_id: str, encrypt: bool = False, passphrase: Optional[str] = None):
    """Download a crystal as a .hx binary file."""
    crystal = get_crystal(crystal_id)
    hx_path = CRYSTAL_DIR / f"{crystal_id}.hx"
    crystal.export(str(hx_path))

    if encrypt and passphrase:
        crypt = PhiCrypt()
        hxe_path = CRYSTAL_DIR / f"{crystal_id}.hxe"
        crypt.encrypt_file(str(hx_path), str(hxe_path), passphrase)
        return FileResponse(str(hxe_path), filename=f"{crystal_id}.hxe")

    return FileResponse(str(hx_path), filename=f"{crystal_id}.hx")


@app.post("/v1/crystal/{crystal_id}/import")
async def import_crystal(crystal_id: str, file: UploadFile = File(...)):
    """Upload a .hx crystal file to restore a crystal state."""
    hx_path = CRYSTAL_DIR / f"{crystal_id}_upload.hx"
    content = await file.read()
    with open(hx_path, "wb") as f:
        f.write(content)

    crystal = get_or_create_crystal(crystal_id)
    try:
        crystal.load(str(hx_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid .hx file: {e}")

    return {
        "crystal_id": crystal_id,
        "absorb_count": crystal.absorb_count,
        "size_bytes": crystal.size_bytes(),
        "status": "loaded"
    }


# ─── MCP Tool Definitions ─────────────────────────────────────────────────────

MCP_TOOLS = [
    {
        "name": "helix_absorb",
        "description": "Feed text or an embedding into a Helix memory crystal. Creates the crystal if it doesn't exist. Returns the crystal's current state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "crystal_id": {"type": "string", "description": "Unique ID for this memory crystal (e.g. user ID, session ID)"},
                "text": {"type": "string", "description": "Text to absorb into memory"},
                "embedding": {"type": "array", "items": {"type": "number"}, "description": "Pre-computed embedding vector (768 dims)"},
                "weight": {"type": "number", "description": "Importance weight 0-10, default 1.0"}
            },
            "required": ["crystal_id"]
        }
    },
    {
        "name": "helix_recall",
        "description": "Retrieve the full memory feature vector from a Helix crystal. Returns a compact representation of everything absorbed so far.",
        "input_schema": {
            "type": "object",
            "properties": {
                "crystal_id": {"type": "string", "description": "ID of the crystal to recall from"}
            },
            "required": ["crystal_id"]
        }
    },
    {
        "name": "helix_recall_at",
        "description": "Retrieve memory state at a specific timestep. Enables random access into the memory timeline.",
        "input_schema": {
            "type": "object",
            "properties": {
                "crystal_id": {"type": "string"},
                "step": {"type": "integer", "description": "Timestep to recall (0 = first absorbed item)"}
            },
            "required": ["crystal_id", "step"]
        }
    },
    {
        "name": "helix_merge",
        "description": "Merge multiple memory crystals using Federated Phase Alignment. Privacy-preserving: uses circular mean of phase angles.",
        "input_schema": {
            "type": "object",
            "properties": {
                "crystal_ids": {"type": "array", "items": {"type": "string"}, "description": "IDs of crystals to merge"},
                "weights": {"type": "array", "items": {"type": "number"}, "description": "Optional importance weights per crystal"},
                "output_id": {"type": "string", "description": "ID for the merged crystal"}
            },
            "required": ["crystal_ids"]
        }
    },
    {
        "name": "helix_collapse",
        "description": "Set an irreversible binary fact in the crystal. Once set, this flag can never be changed. Use for permanent truths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "crystal_id": {"type": "string"},
                "flag_index": {"type": "integer", "description": "Index of the flag (0-31)"},
                "flag_name": {"type": "string", "description": "Optional name for the flag"}
            },
            "required": ["crystal_id", "flag_index"]
        }
    },
    {
        "name": "helix_diff",
        "description": "Compute the phase diff between two memory crystals. Shows what changed and which memories are most different.",
        "input_schema": {
            "type": "object",
            "properties": {
                "crystal_id_a": {"type": "string"},
                "crystal_id_b": {"type": "string"}
            },
            "required": ["crystal_id_a", "crystal_id_b"]
        }
    },
    {
        "name": "helix_info",
        "description": "Get metadata about a crystal: how much it has absorbed, its size, how many facts have been permanently collapsed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "crystal_id": {"type": "string"}
            },
            "required": ["crystal_id"]
        }
    }
]


@app.get("/mcp/tools")
def list_mcp_tools():
    """MCP-compatible tool listing. Claude and OpenAI can use this to discover available tools."""
    return {"tools": MCP_TOOLS}


@app.post("/mcp/call")
def call_mcp_tool(body: MCPToolCallRequest):
    """
    MCP-compatible tool invocation.
    Claude and OpenAI agents can call Helix tools through this endpoint.
    """
    tool = body.tool
    params = body.parameters

    try:
        if tool == "helix_absorb":
            cid = params["crystal_id"]
            req = AbsorbTextRequest(
                text=params.get("text"),
                embedding=params.get("embedding"),
                weight=params.get("weight", 1.0)
            )
            return absorb(cid, req)

        elif tool == "helix_recall":
            return recall(params["crystal_id"])

        elif tool == "helix_recall_at":
            return recall_at_step(params["crystal_id"], params["step"])

        elif tool == "helix_merge":
            req = MergeRequest(
                crystal_ids=params["crystal_ids"],
                weights=params.get("weights"),
                output_id=params.get("output_id")
            )
            return merge_crystals(req)

        elif tool == "helix_collapse":
            req = CollapseRequest(
                flag_index=params["flag_index"],
                flag_name=params.get("flag_name")
            )
            return collapse_flag(params["crystal_id"], req)

        elif tool == "helix_diff":
            return diff_crystals(params["crystal_id_a"], params["crystal_id_b"])

        elif tool == "helix_info":
            return crystal_info(params["crystal_id"])

        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _text_to_embedding(text: str, target_dim: int) -> torch.Tensor:
    """
    Simple deterministic text encoding when no embedding model is available.
    For production, replace this with a real sentence-transformer call.
    """
    chars = [ord(c) for c in text[:target_dim]]
    while len(chars) < target_dim:
        chars.append(0)
    raw = torch.tensor(chars[:target_dim], dtype=torch.float32)
    # Normalize to unit sphere
    norm = raw.norm()
    if norm > 0:
        raw = raw / norm
    return raw


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8765, reload=True)
