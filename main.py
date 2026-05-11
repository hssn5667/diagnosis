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


from fastapi import Request
from fastapi.responses import JSONResponse


@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request):
    base_url = str(request.base_url).rstrip("/").replace("http://", "https://")
    return JSONResponse({
        "name": "MediTwin Diagnosis Agent",
        "description": (
            "RAG-based differential diagnosis agent for MediTwin AI. Accepts structured "
            "PatientState (from Patient Context Agent) and a chief complaint, retrieves "
            "relevant clinical guidelines from ChromaDB (medical_knowledge collection), "
            "and returns a ranked differential diagnosis with ICD-10 codes, confidence "
            "scores, FHIR Condition resources, and clinical safety flags (penicillin allergy, "
            "sepsis, isolation). Includes in-memory TTL cache (5-min), LLM-only fallback "
            "mode when ChromaDB is unavailable, and PostgreSQL persistence for all results."
        ),
        "version": "2.0.0",
        "url": base_url,
        "provider": {
            "organization": "MediTwin AI",
            "name": "Tayyab Hussain",
            "url": "https://github.com/hssn5667/diagnosis-agent"
        },
        "supportedInterfaces": [
            {
                "url": f"{base_url}/diagnose",
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "2.0",
                "description": "Blocking differential diagnosis — returns full DiagnosisOutput with ranked differentials, FHIR Conditions, and clinical safety flags."
            },
            {
                "url": f"{base_url}/stream",
                "protocolBinding": "HTTP+SSE",
                "protocolVersion": "2.0",
                "description": "SSE streaming diagnosis — emits status/progress/token/complete events in real time, including per-token LLM streaming."
            },
            {
                "url": f"{base_url}/diagnose-batch",
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "2.0",
                "description": "Batch diagnosis for up to 10 patients in a single request. Non-fatal per-item errors — failed items return error entries without aborting the batch."
            }
        ],
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": True,
            "ragMode": True,
            "fallbackMode": True,
            "fhirOutput": True,
            "inMemoryCache": True,
            "batchProcessing": True
        },
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json", "text/event-stream"],
        "skills": [
            {
                "id": "differential_diagnosis",
                "name": "Differential Diagnosis (RAG)",
                "description": (
                    "Runs a full RAG-based differential diagnosis pipeline: builds a structured "
                    "patient query from PatientState, retrieves up to 6 relevant chunks from "
                    "ChromaDB (medical_knowledge), and invokes Gemini via LangChain "
                    "with_structured_output() to produce exactly 3–4 ranked diagnoses. Each "
                    "diagnosis includes ICD-10 code, confidence score, clinical reasoning, and "
                    "supporting evidence. Applies post-inference rule adjustments: penicillin "
                    "allergy filter (removes contraindicated steps), sepsis flag, isolation flag, "
                    "and confidence boost. Results are TTL-cached (5 min, 64 entries max) and "
                    "persisted to PostgreSQL."
                ),
                "tags": ["rag", "chromadb", "gemini", "icd10", "differential-diagnosis", "fhir", "cache"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"]
            },
            {
                "id": "stream_differential_diagnosis",
                "name": "Stream Differential Diagnosis (SSE)",
                "description": (
                    "SSE streaming variant of differential_diagnosis. Emits ordered events: "
                    "status (5 labelled steps), progress (RAG retrieval %, token count), "
                    "token (per-token LLM output for live UI feedback), and a final complete "
                    "event containing the full DiagnosisOutput with FHIR Conditions and a "
                    "summary block. Includes cache-hit fast-path and graceful fallback to "
                    "non-streaming LLM invocation if astream_events fails."
                ),
                "tags": ["sse", "streaming", "rag", "gemini", "real-time", "token-streaming"],
                "inputModes": ["application/json"],
                "outputModes": ["text/event-stream"]
            },
            {
                "id": "batch_diagnosis",
                "name": "Batch Differential Diagnosis",
                "description": (
                    "Runs differential diagnosis sequentially for up to 10 patients in a single "
                    "HTTP request. Each item follows the same pipeline as differential_diagnosis. "
                    "Per-item failures return an error entry without aborting the batch, making "
                    "it safe for orchestrator use across heterogeneous patient profiles."
                ),
                "tags": ["batch", "orchestrator", "multi-patient"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"]
            },
            {
                "id": "get_diagnosis_history",
                "name": "Get Patient Diagnosis History",
                "description": (
                    "Returns paginated diagnosis records for a patient from PostgreSQL, newest "
                    "first. Each record includes top diagnosis, ICD-10 code, confidence, RAG mode "
                    "(rag/fallback), differential list, recommended next steps, FHIR Conditions, "
                    "clinical safety flags, cache hit status, elapsed time, and source endpoint "
                    "(diagnose/stream). Supports limit/offset pagination."
                ),
                "tags": ["history", "audit", "patient", "postgresql", "pagination"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"]
            },
            {
                "id": "get_diagnosis_stats",
                "name": "Get Patient Diagnosis Stats",
                "description": (
                    "Returns aggregate statistics across all diagnosis sessions for a patient: "
                    "total diagnoses, unique chief complaints, top 5 conditions by frequency "
                    "(ICD-10 + display + count), sepsis/penicillin/isolation alert counts, "
                    "RAG vs fallback mode breakdown, diagnose vs stream source breakdown, "
                    "and first/latest session timestamps."
                ),
                "tags": ["stats", "analytics", "patient", "postgresql", "alerts"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"]
            },
            {
                "id": "clear_diagnosis_cache",
                "name": "Clear Diagnosis Cache",
                "description": (
                    "Admin endpoint that clears the in-memory TTL cache for all patients, "
                    "forcing subsequent requests to re-run the full RAG pipeline. "
                    "Requires X-Internal-Token header. Returns count of evicted entries."
                ),
                "tags": ["cache", "admin", "invalidation"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"]
            }
        ],
        "ragConfig": {
            "vectorStore": "ChromaDB Cloud",
            "collection": "medical_knowledge",
            "embeddingModel": "models/gemini-embedding-001",
            "chunkSize": 512,
            "chunkOverlap": 64,
            "retrievalK": 6,
            "fallbackMode": "LLM-only (no retrieval)"
        },
        "llmConfig": {
            "provider": "Google Gemini",
            "invocationPattern": "LangChain with_structured_output()",
            "streamingPattern": "astream_events v2 (on_chat_model_stream)",
            "outputSchema": "DiagnosisOutput (Pydantic v2)",
            "icd10Repair": True
        }
    })

COLLECTION_NAME = "medical_knowledge"
