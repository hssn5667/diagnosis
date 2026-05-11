"""
agents/diagnosis/history_router.py
------------------------------------
History endpoints for diagnosis results.

Add to main.py:
    from history_router import router as history_router
    app.include_router(history_router, prefix="/history", tags=["history"])

Endpoints:
    GET /history/{patient_id}                  — all diagnoses for a patient
    GET /history/{patient_id}/latest           — most recent diagnosis only
    GET /history/request/{request_id}          — single record by request ID
    GET /history/stats/{patient_id}            — aggregate stats for a patient
    DELETE /history/{patient_id}               — delete all records for a patient (admin)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Query
from pydantic import BaseModel

import db
from db import get_by_patient, get_by_request_id, is_available

logger = logging.getLogger("meditwin.diagnosis.history")
router = APIRouter()


# ── Response Models ────────────────────────────────────────────────────────────

class DiagnosisHistoryItem(BaseModel):
    id: int
    request_id: str
    patient_id: str
    chief_complaint: str
    top_diagnosis: str
    top_icd10_code: str
    confidence_level: str
    rag_mode: str
    differential_diagnosis: list
    recommended_next_steps: list
    fhir_conditions: Optional[list]
    flags: dict
    cache_hit: bool
    elapsed_ms: Optional[int]
    source: str
    created_at: str


class PatientHistoryResponse(BaseModel):
    patient_id: str
    total_records: int
    records: list[DiagnosisHistoryItem]


class PatientStatsResponse(BaseModel):
    patient_id: str
    total_diagnoses: int
    unique_complaints: int
    top_conditions: list[dict]       # [{icd10, display, count}]
    sepsis_alerts: int
    penicillin_alerts: int
    isolation_flags: int
    rag_mode_breakdown: dict         # {"rag": n, "fallback": n}
    sources_breakdown: dict          # {"diagnose": n, "stream": n}
    latest_diagnosis: Optional[str]
    first_diagnosis: Optional[str]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_row(row: dict) -> DiagnosisHistoryItem:
    """Normalize asyncpg row → DiagnosisHistoryItem."""
    def _load(val):
        if val is None:
            return None
        if isinstance(val, str):
            return json.loads(val)
        return val  # already dict/list from asyncpg JSONB

    return DiagnosisHistoryItem(
        id=row["id"],
        request_id=row["request_id"],
        patient_id=row["patient_id"],
        chief_complaint=row["chief_complaint"],
        top_diagnosis=row["top_diagnosis"] or "",
        top_icd10_code=row["top_icd10_code"] or "",
        confidence_level=row["confidence_level"] or "",
        rag_mode=row["rag_mode"] or "",
        differential_diagnosis=_load(row["differential_diagnosis"]) or [],
        recommended_next_steps=_load(row["recommended_next_steps"]) or [],
        fhir_conditions=_load(row["fhir_conditions"]),
        flags=_load(row["flags"]) or {},
        cache_hit=row["cache_hit"] or False,
        elapsed_ms=row["elapsed_ms"],
        source=row["source"] or "diagnose",
        created_at=str(row["created_at"]),
    )


def _db_check():
    if not is_available():
        raise HTTPException(status_code=503, detail="Database unavailable")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/{patient_id}", response_model=PatientHistoryResponse)
async def get_patient_history(
    patient_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    Get all diagnosis records for a patient, newest first.
    Supports pagination via limit/offset.
    """
    _db_check()

    async with db._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM diagnosis_results
            WHERE patient_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            patient_id, limit, offset,
        )
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM diagnosis_results WHERE patient_id = $1",
            patient_id,
        )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No diagnosis records found for patient '{patient_id}'",
        )

    return PatientHistoryResponse(
        patient_id=patient_id,
        total_records=total,
        records=[_parse_row(dict(r)) for r in rows],
    )


@router.get("/{patient_id}/latest", response_model=DiagnosisHistoryItem)
async def get_latest_diagnosis(patient_id: str):
    """Get the most recent diagnosis for a patient."""
    _db_check()

    async with db._pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM diagnosis_results
            WHERE patient_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            patient_id,
        )

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No diagnosis records found for patient '{patient_id}'",
        )

    return _parse_row(dict(row))


@router.get("/request/{request_id}", response_model=DiagnosisHistoryItem)
async def get_by_request(request_id: str):
    """Get a single diagnosis record by request ID."""
    _db_check()

    row = await get_by_request_id(request_id)
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No record found for request_id '{request_id}'",
        )

    return _parse_row(row)


@router.get("/stats/{patient_id}", response_model=PatientStatsResponse)
async def get_patient_stats(patient_id: str):
    """
    Aggregate stats for a patient across all diagnosis sessions.
    Useful for frontend patient summary cards.
    """
    _db_check()

    async with db._pool.acquire() as conn:
        # Basic counts
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM diagnosis_results WHERE patient_id = $1",
            patient_id,
        )

        if not total:
            raise HTTPException(
                status_code=404,
                detail=f"No records found for patient '{patient_id}'",
            )

        unique_complaints = await conn.fetchval(
            "SELECT COUNT(DISTINCT chief_complaint) FROM diagnosis_results WHERE patient_id = $1",
            patient_id,
        )

        # Top conditions by frequency
        top_rows = await conn.fetch(
            """
            SELECT top_icd10_code, top_diagnosis, COUNT(*) as count
            FROM diagnosis_results
            WHERE patient_id = $1
            GROUP BY top_icd10_code, top_diagnosis
            ORDER BY count DESC
            LIMIT 5
            """,
            patient_id,
        )

        # Alert counts from JSONB flags
        sepsis_alerts = await conn.fetchval(
            """
            SELECT COUNT(*) FROM diagnosis_results
            WHERE patient_id = $1 AND (flags->>'sepsis_alert')::boolean = true
            """,
            patient_id,
        )
        penicillin_alerts = await conn.fetchval(
            """
            SELECT COUNT(*) FROM diagnosis_results
            WHERE patient_id = $1 AND (flags->>'penicillin_alert')::boolean = true
            """,
            patient_id,
        )
        isolation_flags = await conn.fetchval(
            """
            SELECT COUNT(*) FROM diagnosis_results
            WHERE patient_id = $1 AND (flags->>'requires_isolation')::boolean = true
            """,
            patient_id,
        )

        # RAG mode breakdown
        rag_rows = await conn.fetch(
            """
            SELECT rag_mode, COUNT(*) as count
            FROM diagnosis_results WHERE patient_id = $1
            GROUP BY rag_mode
            """,
            patient_id,
        )

        # Source breakdown
        source_rows = await conn.fetch(
            """
            SELECT source, COUNT(*) as count
            FROM diagnosis_results WHERE patient_id = $1
            GROUP BY source
            """,
            patient_id,
        )

        # Timeline
        latest = await conn.fetchval(
            "SELECT MAX(created_at) FROM diagnosis_results WHERE patient_id = $1",
            patient_id,
        )
        first = await conn.fetchval(
            "SELECT MIN(created_at) FROM diagnosis_results WHERE patient_id = $1",
            patient_id,
        )

    return PatientStatsResponse(
        patient_id=patient_id,
        total_diagnoses=total,
        unique_complaints=unique_complaints,
        top_conditions=[
            {"icd10": r["top_icd10_code"], "display": r["top_diagnosis"], "count": r["count"]}
            for r in top_rows
        ],
        sepsis_alerts=sepsis_alerts or 0,
        penicillin_alerts=penicillin_alerts or 0,
        isolation_flags=isolation_flags or 0,
        rag_mode_breakdown={r["rag_mode"]: r["count"] for r in rag_rows},
        sources_breakdown={r["source"]: r["count"] for r in source_rows},
        latest_diagnosis=str(latest) if latest else None,
        first_diagnosis=str(first) if first else None,
    )


@router.delete("/{patient_id}")
async def delete_patient_history(
    patient_id: str,
    x_internal_token: Optional[str] = Header(None),
):
    """
    Delete all diagnosis records for a patient.
    Requires internal token — admin only.
    """
    _db_check()

    expected = os.getenv("INTERNAL_TOKEN", "meditwin-internal")
    if x_internal_token != expected:
        raise HTTPException(status_code=403, detail="Invalid internal token")

    async with db._pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM diagnosis_results WHERE patient_id = $1",
            patient_id,
        )

    deleted = int(result.split()[-1])
    logger.info(f"    ✔   Deleted {deleted} records for patient={patient_id}")
    return {"patient_id": patient_id, "deleted_records": deleted, "status": "ok"}