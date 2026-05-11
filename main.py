"""
Agent 2: Diagnosis Agent
RAG-based differential diagnosis over medical knowledge base.
Port: 8002

Improvements over v1:
  - Graceful fallback to LLM-only mode when ChromaDB unavailable
  - Request ID generation + correlation logging
  - Input validation with meaningful error messages
  - /diagnose-batch endpoint for parallel orchestrator use
  - /cache-clear admin endpoint
  - Richer health check (ChromaDB chunk count, cache stats)
  - rag_mode field in response: "rag" | "fallback" | "unavailable"
"""
import os
import sys
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, field_validator

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rag import diagnosis

from stream_endpoint import router as stream_router

from db import init as db_init, close as db_close, save_diagnosis, DiagnosisRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s [%(request_id)s] — %(message)s"
    if False else "%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("meditwin.diagnosis")

# ── Request / Response Models ──────────────────────────────────────────────────

class DiagnoseRequest(BaseModel):
    patient_state: dict
    chief_complaint: str
    include_fhir_resources: bool = True
    request_id: Optional[str] = None  # correlation ID, generated if not provided

    @field_validator("chief_complaint")
    @classmethod
    def chief_complaint_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("chief_complaint cannot be empty")
        if len(v) < 5:
            raise ValueError("chief_complaint too short — provide a meaningful clinical complaint")
        return v

    @field_validator("patient_state")
    @classmethod
    def patient_state_has_required_fields(cls, v: dict) -> dict:
        if not v.get("patient_id"):
            raise ValueError("patient_state.patient_id is required")
        if not v.get("demographics"):
            raise ValueError("patient_state.demographics is required")
        return v


class DiagnoseResponse(BaseModel):
    request_id: str
    differential_diagnosis: list
    top_diagnosis: str
    top_icd10_code: str
    confidence_level: str
    reasoning_summary: str
    recommended_next_steps: list
    fhir_conditions: Optional[list] = None
    rag_mode: str               # "rag" | "fallback"
    penicillin_allergy_flagged: bool = False
    high_suspicion_sepsis: bool = False
    requires_isolation: bool = False


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: try to initialize full RAG chain.
    If ChromaDB is unreachable or empty, fall back to LLM-only mode.
    Agent NEVER refuses to start — graceful degradation is non-negotiable.
    """
    # ── Run ingest pipeline before initializing RAG ──────────────────────────
    try:
        from knowledge_base import ingest
        logger.info("Running knowledge base ingest pipeline...")
        ingest.main()
        logger.info("✓ Ingest pipeline complete")
    except Exception as e:
        logger.warning(f"Ingest pipeline failed (non-fatal): {e} — continuing with existing ChromaDB data")

    try:
        diagnosis.initialize()
        logger.info("✓ Diagnosis Agent started — RAG mode (ChromaDB + Gemini)")
    except Exception as e:
        logger.warning(f"ChromaDB unavailable ({e}) — starting in LLM-only fallback mode")
        try:
            diagnosis.initialize_fallback()
            logger.info("    ✔   Diagnosis Agent started — FALLBACK mode (LLM-only)")
        except Exception as e2:
            logger.error(f"    ✘   FATAL: Could not initialize even in fallback mode: {e2}")
            # Don't sys.exit — let FastAPI start anyway so /health returns degraded status

    await db_init()          # ← ADD THIS
 
    yield
 
    await db_close()         # ← ADD THIS
    logger.info("    ✔   Diagnosis Agent shutdown")


app = FastAPI(
    title="MediTwin Diagnosis Agent",
    description=(
        "RAG-based differential diagnosis from FHIR patient data. "
        "Falls back to LLM-only mode if ChromaDB is unavailable."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(stream_router)
from history_router import router as history_router
app.include_router(history_router, prefix="/history", tags=["history"])

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    """
    Run differential diagnosis for a patient.

    Requires: patient_state (from Patient Context Agent) + chief_complaint.
    Returns: ranked differential, structured next steps, optional FHIR Conditions.
    """
    if not diagnosis._initialized:
        raise HTTPException(
            status_code=503,
            detail="Diagnosis Agent not initialized. Check logs — ChromaDB may be unreachable.",
        )

    request_id = request.request_id or str(uuid.uuid4())[:8]
    patient_id = request.patient_state.get("patient_id", "unknown")
    logger.info(f"[{request_id}] /diagnose patient={patient_id} complaint='{request.chief_complaint[:60]}'")

    try:
        result = diagnosis.run(
            patient_state=request.patient_state,
            chief_complaint=request.chief_complaint,
            request_id=request_id,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Diagnosis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(e)}")

    fhir_conditions = None
    if request.include_fhir_resources:
        try:
            fhir_conditions = diagnosis.build_fhir_conditions(result, patient_id)
        except Exception as e:
            logger.warning(f"[{request_id}] FHIR condition build failed (non-fatal): {e}")

    db_record = DiagnosisRecord(
        request_id=request_id,
        patient_id=patient_id,
        chief_complaint=request.chief_complaint,
        top_diagnosis=result.top_diagnosis,
        top_icd10_code=result.top_icd10_code,
        confidence_level=result.confidence_level,
        rag_mode="rag" if diagnosis.rag_available else "fallback",
        differential_diagnosis=[d.model_dump() for d in result.differential_diagnosis],
        recommended_next_steps=[s.model_dump() for s in result.recommended_next_steps],
        fhir_conditions=fhir_conditions,
        penicillin_alert=result.penicillin_allergy_flagged,
        sepsis_alert=result.high_suspicion_sepsis,
        requires_isolation=result.requires_isolation,
        cache_hit=False,
        source="diagnose",
    )
    await save_diagnosis(db_record)   # non-fatal — won't break response

    return DiagnoseResponse(
        request_id=request_id,
        differential_diagnosis=[d.model_dump() for d in result.differential_diagnosis],
        top_diagnosis=result.top_diagnosis,
        top_icd10_code=result.top_icd10_code,
        confidence_level=result.confidence_level,
        reasoning_summary=result.reasoning_summary,
        recommended_next_steps=[s.model_dump() for s in result.recommended_next_steps],
        fhir_conditions=fhir_conditions,
        rag_mode="rag" if diagnosis.rag_available else "fallback",
        penicillin_allergy_flagged=result.penicillin_allergy_flagged,
        high_suspicion_sepsis=result.high_suspicion_sepsis,
        requires_isolation=result.requires_isolation,
    )


@app.post("/diagnose-batch")
async def diagnose_batch(requests: list[DiagnoseRequest]) -> list[DiagnoseResponse]:
    """
    Batch endpoint — run multiple diagnoses in sequence.
    Useful for testing multiple patient profiles without spinning up
    parallel connections.
    Max 10 requests per batch.
    """
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Batch limited to 10 requests")

    results = []
    for req in requests:
        try:
            result = await diagnose(req)
            results.append(result)
        except HTTPException as e:
            # Don't abort whole batch for one failure — return error entry
            results.append({"error": e.detail, "patient_id": req.patient_state.get("patient_id")})
    return results


@app.post("/cache-clear")
async def cache_clear(
    x_internal_token: Optional[str] = Header(None),
) -> dict:
    """
    Admin endpoint — clear in-memory diagnosis cache.
    Requires internal token header.
    """
    expected_token = os.getenv("INTERNAL_TOKEN", "meditwin-internal")
    if x_internal_token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid internal token")

    from rag import _cache
    cleared = len(_cache._store)
    _cache._store.clear()
    logger.info(f"    ✔   Cache cleared: {cleared} entries removed")
    return {"cleared_entries": cleared, "status": "ok"}


@app.get("/health")
async def health() -> dict:
    """
    Health check with detailed status.
    Returns chromadb chunk count, cache stats, and rag_mode.
    """
    cache_from_module = None
    try:
        from rag import _cache
        cache_from_module = {
            "cached_entries": len(_cache._store),
            "ttl_seconds": _cache._ttl,
        }
    except Exception:
        pass

    chromadb_chunk_count = None
    if diagnosis._initialized and diagnosis.rag_available and diagnosis._vectorstore:
        try:
            chromadb_chunk_count = diagnosis._vectorstore._collection.count()
        except Exception:
            chromadb_chunk_count = "unreachable"

    return {
        "status": "healthy" if diagnosis._initialized else "degraded",
        "agent": "diagnosis",
        "version": "2.0.0",
        "rag_mode": "rag" if (diagnosis._initialized and diagnosis.rag_available) else (
            "fallback" if diagnosis._initialized else "not_initialized"
        ),
        "chromadb_collection": COLLECTION_NAME if diagnosis.rag_available else None,
        "chromadb_chunks": chromadb_chunk_count,
        "cache": cache_from_module,
    }


COLLECTION_NAME = "medical_knowledge"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)