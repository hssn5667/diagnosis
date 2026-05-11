"""
agents/diagnosis/db.py
-----------------------
PostgreSQL persistence for diagnosis results.
Uses asyncpg connection pool — zero blocking in FastAPI.

Setup:
    Set env var:  POSTGRES_DSN=postgresql://user:pass@localhost:5432/meditwin
    On startup:   await db.init()
    On shutdown:  await db.close()

Both /diagnose and /stream call:
    await db.save_diagnosis(record: DiagnosisRecord)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import asyncpg

logger = logging.getLogger("meditwin.diagnosis.db")

from dotenv import load_dotenv
load_dotenv()

# ── Pool singleton ─────────────────────────────────────────────────────────────

_pool: Optional[asyncpg.Pool] = None


async def init() -> None:
    """Create connection pool and ensure table exists."""
    global _pool
    dsn = os.getenv("POSTGRES_CHECKPOINT_URI", "postgresql://postgres:postgres@localhost:5432/meditwin")
    try:
        _pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
        await _ensure_table()
        logger.info("     ✔    PostgreSQL pool ready")
    except Exception as e:
        logger.warning(f"    ⚠   PostgreSQL unavailable ({e}) — diagnosis results will NOT be persisted")
        _pool = None


async def close() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def is_available() -> bool:
    return _pool is not None


# ── Table DDL ──────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS diagnosis_results (
    id                      SERIAL PRIMARY KEY,
    request_id              VARCHAR(16) UNIQUE NOT NULL,
    patient_id              TEXT NOT NULL,
    chief_complaint         TEXT NOT NULL,
    top_diagnosis           TEXT,
    top_icd10_code          TEXT,
    confidence_level        TEXT,
    rag_mode                TEXT,
    differential_diagnosis  JSONB,
    recommended_next_steps  JSONB,
    fhir_conditions         JSONB,
    flags                   JSONB,
    cache_hit               BOOLEAN DEFAULT FALSE,
    elapsed_ms              INTEGER,
    source                  TEXT DEFAULT 'diagnose',  -- 'diagnose' | 'stream'
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_diagnosis_patient_id  ON diagnosis_results(patient_id);
CREATE INDEX IF NOT EXISTS idx_diagnosis_created_at  ON diagnosis_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_diagnosis_top_icd10   ON diagnosis_results(top_icd10_code);
"""


async def _ensure_table() -> None:
    async with _pool.acquire() as conn:
        await conn.execute(_CREATE_TABLE)


# ── Record dataclass ───────────────────────────────────────────────────────────

@dataclass
class DiagnosisRecord:
    request_id:             str
    patient_id:             str
    chief_complaint:        str
    top_diagnosis:          str
    top_icd10_code:         str
    confidence_level:       str
    rag_mode:               str
    differential_diagnosis: list
    recommended_next_steps: list
    fhir_conditions:        Optional[list]
    penicillin_alert:       bool
    sepsis_alert:           bool
    requires_isolation:     bool
    cache_hit:              bool = False
    elapsed_ms:             Optional[int] = None
    source:                 str = "diagnose"   # "diagnose" | "stream"


# ── Save ───────────────────────────────────────────────────────────────────────

_INSERT = """
INSERT INTO diagnosis_results (
    request_id, patient_id, chief_complaint,
    top_diagnosis, top_icd10_code, confidence_level, rag_mode,
    differential_diagnosis, recommended_next_steps, fhir_conditions,
    flags, cache_hit, elapsed_ms, source
) VALUES (
    $1, $2, $3,
    $4, $5, $6, $7,
    $8, $9, $10,
    $11, $12, $13, $14
)
ON CONFLICT (request_id) DO NOTHING
RETURNING id;
"""


async def save_diagnosis(record: DiagnosisRecord) -> Optional[int]:
    """
    Persist a diagnosis result. Returns inserted row id, or None if skipped/unavailable.
    Never raises — DB failures are logged but don't break the API response.
    """
    if not _pool:
        return None

    try:
        flags = json.dumps({
            "penicillin_alert":  record.penicillin_alert,
            "sepsis_alert":      record.sepsis_alert,
            "requires_isolation": record.requires_isolation,
        })

        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                _INSERT,
                record.request_id,
                record.patient_id,
                record.chief_complaint,
                record.top_diagnosis,
                record.top_icd10_code,
                record.confidence_level,
                record.rag_mode,
                json.dumps(record.differential_diagnosis),
                json.dumps(record.recommended_next_steps),
                json.dumps(record.fhir_conditions) if record.fhir_conditions else None,
                flags,
                record.cache_hit,
                record.elapsed_ms,
                record.source,
            )

        if row:
            logger.info(f"    ✔   [{record.request_id}] Diagnosis saved → row id={row['id']} patient={record.patient_id}")
            return row["id"]
        else:
            logger.debug(f"    ℹ   [{record.request_id}] Duplicate request_id — skipped insert")
            return None

    except Exception as e:
        logger.error(f"    ✘   [{record.request_id}] DB save failed (non-fatal): {e}")
        return None


# ── Query helpers (optional — for future /history endpoint) ───────────────────

async def get_by_patient(patient_id: str, limit: int = 20) -> list[dict]:
    """Fetch recent diagnosis records for a patient."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT * FROM diagnosis_results
               WHERE patient_id = $1
               ORDER BY created_at DESC LIMIT $2""",
            patient_id, limit,
        )
    return [dict(r) for r in rows]


async def get_by_request_id(request_id: str) -> Optional[dict]:
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM diagnosis_results WHERE request_id = $1",
            request_id,
        )
    return dict(row) if row else None