"""
Shared Pydantic models for MediTwin AI
These models define the data contracts between all agents
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import re


# ── Diagnosis Agent models ────────────────────────────────────────────────────
 
class NextStep(BaseModel):
    """
    Structured next step — consumed by Drug Safety Agent caller to infer
    proposed medications and by Digital Twin for treatment scenarios.
    """
    category: str           # MEDICATION | INVESTIGATION | MONITORING | REFERRAL | PROCEDURE
    description: str        # Human-readable step description
    drug_name: Optional[str] = None     # Populated for MEDICATION steps
    drug_dose: Optional[str] = None
    drug_route: Optional[str] = None    # oral | IV | IM | inhaled
    urgency: str = "routine"            # stat | urgent | routine
    rationale: Optional[str] = None
 
 
class DiagnosisItem(BaseModel):
    rank: int
    display: str
    icd10_code: str
    confidence: float = Field(ge=0.0, le=1.0)
    clinical_reasoning: str
    supporting_evidence: List[str] = Field(default_factory=list)
    against_evidence: List[str] = Field(default_factory=list)
 
    @field_validator("icd10_code")
    @classmethod
    def validate_icd10(cls, v: str) -> str:
        """
        Enforce ICD-10 format: letter + 2 digits, optionally dot + 1-4 chars.
        Rejects LLM hallucinations like 'J.18', 'UNKNOWN', empty strings.
        """
        v = v.strip().upper()
        pattern = r'^[A-Z]\d{2}(\.\d{1,4})?$'
        if not re.match(pattern, v):
            raise ValueError(f"Invalid ICD-10 code format: '{v}'. Expected e.g. J18.9, E11, I48.0")
        return v
 
 
class DiagnosisOutput(BaseModel):
    """Output schema for Diagnosis Agent — returned by /diagnose endpoint."""
    differential_diagnosis: List[DiagnosisItem]
    top_diagnosis: str = ""
    top_icd10_code: str = ""
    confidence_level: str           # HIGH | MODERATE | LOW
    reasoning_summary: str
    recommended_next_steps: List[NextStep] = Field(default_factory=list)
 
    # Flags for downstream consumers
    penicillin_allergy_flagged: bool = False
    high_suspicion_sepsis: bool = False
    requires_isolation: bool = False
