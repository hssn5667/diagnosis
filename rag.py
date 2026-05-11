# agents/diagnosis/rag_chain.py
import os
import re
import json
import hashlib
import logging
import time
from typing import Optional
 
import chromadb
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
 
from shared.models import DiagnosisOutput, DiagnosisItem, NextStep
from dotenv import load_dotenv
 
load_dotenv()
 
logger = logging.getLogger("meditwin.diagnosis.rag")


COLLECTION_NAME = "medical_knowledge"

SYSTEM_PROMPT = """
You are a clinical decision support system performing differential diagnosis.

You will receive structured PATIENT DATA and optional clinical context.

STRICT RULES (MUST FOLLOW):

1. DATA GROUNDING
- Use ONLY information explicitly present in PATIENT DATA
- DO NOT assume or add symptoms
- If data is missing → explicitly acknowledge it

2. DIFFERENTIAL REQUIREMENTS
- Return EXACTLY 3 or 4 diagnoses
- Rank by likelihood
- Each diagnosis must include:
  - ICD-10 code
  - Confidence (0.0–1.0)
  - Reasoning tied directly to patient data

3. CONTEXTUAL REASONING (CRITICAL)
- You MUST consider relationships between:
  - symptoms
  - medications
  - conditions
- If a patient is already receiving treatment (e.g., antibiotics):
  - consider:
    - treatment failure
    - resistant infection
    - alternative (non-infectious) diagnoses

4. DIFFERENTIAL BALANCE (CRITICAL)
- DO NOT return only one category (e.g., all infections)
- If data is limited:
  - include at least ONE non-infectious or alternative diagnosis

5. EVIDENCE QUALITY
- Supporting evidence must be SPECIFIC and meaningful
- Avoid repeating the same generic symptom (e.g., “fever”) for all diagnoses
- Use combinations of data (e.g., fever + medication context)

6. UNCERTAINTY HANDLING
- If labs/vitals are missing:
  - reduce confidence
  - explicitly mention uncertainty
- Avoid overconfident conclusions

7. PRIORITY OF INFORMATION
- PATIENT DATA is primary
- CONTEXT is secondary (support only)
- If conflict → trust PATIENT DATA

8. NO HALLUCINATION
- DO NOT invent:
  - symptoms
  - labs
  - vitals
  - history

9. SELF-CHECK (MANDATORY)
Before final answer:
- Are all diagnoses supported by DIFFERENT reasoning?
- Did I use medication context if present?
- Did I avoid single-cause bias (e.g., all infections)?
- If not → revise

Return structured output only.
"""

HUMAN_PROMPT = """
PATIENT_DATA_JSON:
{patient_json}

RETRIEVED_CONTEXT:
{context}

TASK:
Generate a differential diagnosis strictly based on PATIENT_DATA_JSON.

CHECKLIST BEFORE RETURNING:
- Exactly 3–4 diagnoses
- Not all diagnoses from same category
- Medication context is used if present
- No hallucinated symptoms
- Confidence reflects missing data
"""

# LLM-only prompt used when ChromaDB is unavailable
FALLBACK_SYSTEM_PROMPT = """You are a clinical decision support system performing differential diagnosis WITHOUT external knowledge retrieval.
Apply your internal clinical knowledge carefully.

""" + SYSTEM_PROMPT  # just reuse the full SYSTEM_PROMPT directly


# ── Simple TTL cache ──────────────────────────────────────────────────────────
 
class _SimpleCache:
    """
    In-memory LRU-style cache with TTL.
    Keyed on SHA-256 of (patient_id + chief_complaint).
    Prevents re-hitting ChromaDB + Gemini for identical requests during a demo run.
    """
    def __init__(self, ttl_seconds: int = 300, max_size: int = 64):
        self._store: dict[str, tuple[float, DiagnosisOutput]] = {}
        self._ttl = ttl_seconds
        self._max = max_size
 
    def _key(self, patient_id: str, chief_complaint: str) -> str:
        raw = f"{patient_id}|{chief_complaint.strip().lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
 
    def get(self, patient_id: str, chief_complaint: str) -> Optional[DiagnosisOutput]:
        key = self._key(patient_id, chief_complaint)
        if key in self._store:
            ts, value = self._store[key]
            if time.time() - ts < self._ttl:
                logger.debug(f"    ℹ   Cache HIT for patient={patient_id}")
                return value
            del self._store[key]
        return None
 
    def set(self, patient_id: str, chief_complaint: str, value: DiagnosisOutput):
        if len(self._store) >= self._max:
            # evict oldest
            oldest = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest]
        key = self._key(patient_id, chief_complaint)
        self._store[key] = (time.time(), value)
 
 
_cache = _SimpleCache(ttl_seconds=300)
 
 
# ── ICD-10 repair helpers ─────────────────────────────────────────────────────
 
_ICD10_PATTERN = re.compile(r'^[A-Z]\d{2}(\.\d{1,4})?$')
 
def _repair_icd10(code: str) -> str:
    """
    Attempt to fix common LLM ICD-10 mistakes before Pydantic validation.
      - Remove spaces: 'J 18.9' → 'J18.9'
      - Fix dot placement: 'J.18.9' → 'J18.9' (rare but happens)
      - Uppercase: 'j18.9' → 'J18.9'
    Returns original if repair fails (Pydantic will then raise).
    """
    code = code.strip().upper().replace(" ", "")
    # 'J.18.9' or 'J.18' → remove the first dot
    if re.match(r'^[A-Z]\.\d', code):
        code = code[0] + code[2:]
    return code
 
 
def _validate_and_repair_items(items: list[dict]) -> list[dict]:
    """Repair ICD-10 codes in raw LLM output before Pydantic parsing."""
    for item in items:
        if "icd10_code" in item:
            item["icd10_code"] = _repair_icd10(item["icd10_code"])
    return items

def _validate_no_hallucination(output: DiagnosisOutput, patient_state: dict):
    """
    Ensure model did not use symptoms not present in input.
    """
    allowed_terms = set()

    # Collect allowed evidence from patient data
    for cond in patient_state.get("active_conditions", []):
        if cond.get("display"):
            allowed_terms.update(word.strip(".,;:()") for word in cond.get("display").lower().split())

    for lab in patient_state.get("lab_results", []):
        if lab.get("display"):
            allowed_terms.update(word.strip(".,;:()") for word in lab.get("display").lower().split())

    chief = patient_state.get("chief_complaint", "")
    if chief:
        allowed_terms.update(word.strip(".,;:()") for word in chief.lower().split())

    # Remove short words and common stop words
    stop_words = {"and", "the", "with", "of", "in", "a", "an", "is", "are", "to", "for", "patient"}
    allowed_terms = {t for t in allowed_terms if len(t) > 2 and t not in stop_words}

    if not allowed_terms:
        return

    # Check each diagnosis
    for diag in output.differential_diagnosis:
        for evidence in diag.supporting_evidence:
            evidence_words = {w.strip(".,;:()") for w in evidence.lower().split()}
            
            # Check for any meaningful keyword overlap
            if not allowed_terms.intersection(evidence_words):
                logger.warning(f"    ⚠   Potentially hallucinated evidence detected (no keyword overlap): {evidence}")
 

class DiagnosisRAG:

    def __init__(self):
        self._vectorstore: Optional[Chroma] = None
        self._llm = None
        self._structured_llm = None
        self._fallback_llm = None   # LLM-only, no RAG
        self._retrieve_tool = None
        self._rag_available = False
        self._initialized = False

    def initialize(self):
        chromadb_host = os.getenv("CHROMADB_HOST", "localhost")
        chromadb_port = int(os.getenv("CHROMADB_PORT", "8008"))

        client = chromadb.HttpClient(host=chromadb_host, port=chromadb_port)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        self._vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )

        count = self._vectorstore._collection.count()
        if count == 0:
            raise RuntimeError(f"Collection '{COLLECTION_NAME}' is empty. Run ingest.py first.")

        logger.info(f"    ✔   ChromaDB connected — {count} chunks in '{COLLECTION_NAME}'")

        # ── key change: use with_structured_output instead of JsonOutputParser ──
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0.2)
        self._structured_llm = llm.with_structured_output(DiagnosisOutput)

        # ── new docs pattern: @tool with response_format="content_and_artifact" ──
        vectorstore = self._vectorstore  # capture for closure

        @tool(response_format="content_and_artifact")
        def retrieve_clinical_context(query: str):
            """Retrieve clinical guidelines relevant to a patient presentation."""
            docs = vectorstore.similarity_search(query, k=4)
            serialized = "\n\n---\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}"
                for doc in docs
            )
            return serialized, docs  # (content for LLM, raw docs as artifact)

        self._retrieve_tool = retrieve_clinical_context

        logger.info(f"    ✔   ChromaDB connected — {count} chunks in '{COLLECTION_NAME}'")
        self._rag_available = True
        self._init_llms()
        self._initialized = True
        logger.info("    ✔   Diagnosis RAG chain initialized (RAG mode)")
    
    def initialize_fallback(self):
        """
        Initialize LLM-only mode — no ChromaDB dependency.
        Used when ChromaDB is unavailable. Graceful degradation.
        """
        self._rag_available = False
        self._init_llms()
        self._initialized = True
        logger.warning("    ⚠   Diagnosis RAG chain initialized in FALLBACK mode (LLM-only, no RAG context)")
 
    def _init_llms(self):
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0.1)
        self._llm = llm
        # structured_output parses JSON and validates against DiagnosisOutput
        self._structured_llm = llm.with_structured_output(DiagnosisOutput)

    @property
    def rag_available(self) -> bool:
        return self._rag_available

    def _build_patient_json(self, patient_state: dict, chief_complaint: str) -> str:
        demographics = patient_state.get("demographics", {})

        labs = patient_state.get("lab_results", [])
        vitals_loincs = {"8310-5", "8867-4", "9279-1", "55284-4", "59408-5"}

        structured = {
            "demographics": {
                "age": demographics.get("age"),
                "gender": demographics.get("gender"),
            },
            "chief_complaint": chief_complaint,

            "active_conditions": [
                c.get("display") for c in patient_state.get("active_conditions", [])
            ],

            "medications": [
                {
                    "drug": m.get("drug"),
                    "dose": m.get("dose"),
                    "frequency": m.get("frequency"),
                }
                for m in patient_state.get("medications", [])
            ],

            "allergies": [
                {
                    "substance": a.get("substance"),
                    "severity": a.get("severity"),
                }
                for a in patient_state.get("allergies", [])
            ],

            "labs": [
                {
                    "name": l.get("display"),
                    "value": l.get("value"),
                    "unit": l.get("unit"),
                    "flag": l.get("flag"),
                }
                for l in labs
            ],

            "vitals": [
                {
                    "name": l.get("display"),
                    "value": l.get("value"),
                    "unit": l.get("unit"),
                }
                for l in labs if l.get("loinc") in vitals_loincs
            ],

            "recent_encounter": (
                patient_state.get("recent_encounters", [None])[0]
                if patient_state.get("recent_encounters")
                else None
            ),

            # Explicit missing data signals
            "data_availability": {
                "has_labs": bool(labs),
                "has_vitals": any(l.get("loinc") in vitals_loincs for l in labs),
                "has_history": bool(patient_state.get("recent_encounters")),
            }
        }

        return json.dumps(structured, indent=2)

    # ── Rule-based adjustments ────────────────────────────────────────────────
 
    def _apply_rule_adjustments(
        self, output: DiagnosisOutput, patient_state: dict
    ) -> DiagnosisOutput:
        """
        Deterministic post-processing on top of LLM output.
        Rules engine runs AFTER LLM — adds/modifies deterministically.
        """
        allergies_lower = {
            a.get("substance", "").lower()
            for a in patient_state.get("allergies", [])
        }
        labs = {l.get("loinc"): l for l in patient_state.get("lab_results", [])}
 
        wbc_flag = labs.get("26464-8", {}).get("flag", "")
        wbc_val = labs.get("26464-8", {}).get("value", 0.0)
        crp_flag = labs.get("1988-5", {}).get("flag", "")
        procalcitonin_flag = labs.get("75241-0", {}).get("flag", "")  # PCT
 
        beta_lactam_substances = {"penicillin", "amoxicillin", "ampicillin", "cephalosporin"}
        has_betalactam_allergy = bool(allergies_lower & beta_lactam_substances)
 
        # Flag sepsis suspicion: WBC > 15 in any form
        try:
            if float(wbc_val) >= 15.0:
                output.high_suspicion_sepsis = True
        except (TypeError, ValueError):
            pass
 
        # Flag penicillin allergy
        if has_betalactam_allergy:
            output.penicillin_allergy_flagged = True
 
        # Confidence boosts + allergy notes per diagnosis
        for diag in output.differential_diagnosis:
            code = diag.icd10_code
 
            # Respiratory infections (J10–J22): boost if labs confirm bacterial
            if code[:2] in ("J1", "J2"):
                if wbc_flag in ("HIGH", "CRITICAL") and crp_flag in ("HIGH", "CRITICAL"):
                    diag.confidence = round(min(diag.confidence + 0.08, 0.97), 2)
                if procalcitonin_flag in ("HIGH", "CRITICAL"):
                    diag.confidence = round(min(diag.confidence + 0.04, 0.97), 2)
                if has_betalactam_allergy:
                    note = "Beta-lactam allergy — standard penicillin-class therapy contraindicated"
                    if note not in diag.against_evidence:
                        diag.against_evidence.append(note)
 
            # Isolation flag: TB (A15–A19), active flu (J09–J11)
            if code[:3] in ("A15", "A16", "A17", "A18", "A19") or code[:3] in ("J09", "J10", "J11"):
                output.requires_isolation = True
 
        # Recompute confidence_level from top diagnosis
        if output.differential_diagnosis:
            top_conf = output.differential_diagnosis[0].confidence
            if top_conf >= 0.80:
                output.confidence_level = "HIGH"
            elif top_conf >= 0.60:
                output.confidence_level = "MODERATE"
            else:
                output.confidence_level = "LOW"
 
        # Strip MEDICATION next steps that propose a beta-lactam for allergic patient
        if has_betalactam_allergy:
            cleaned_steps = []
            removed = []
            for step in output.recommended_next_steps:
                drug = (step.drug_name or "").lower()
                if step.category == "MEDICATION" and any(
                    sub in drug for sub in ("amoxicillin", "ampicillin", "penicillin",
                                            "piperacillin", "oxacillin", "cephalexin",
                                            "cefazolin", "ceftriaxone", "cefuroxime")
                ):
                    removed.append(step.drug_name)
                    logger.warning(
                        f"Removed contraindicated medication '{step.drug_name}' "
                        f"(beta-lactam allergy present)"
                    )
                else:
                    cleaned_steps.append(step)
            output.recommended_next_steps = cleaned_steps
            if removed:
                # Add a flag step so downstream knows what was removed
                output.recommended_next_steps.insert(0, NextStep(
                    category="MEDICATION",
                    description=(
                        f"    ⚠   Allergy alert: {', '.join(removed)} removed. "
                        f"Use alternative such as azithromycin or levofloxacin."
                    ),
                    urgency="stat",
                    rationale="Automated allergy safety filter",
                ))
 
        return output
    
    # ── RAG retrieval ─────────────────────────────────────────────────────────
 
    def _retrieve_context(self, query: str, k: int = 6) -> str:
        """Retrieve and format clinical context from ChromaDB."""
        try:
            docs = self._vectorstore.similarity_search(query, k=k)
            if not docs:
                return "No relevant clinical guidelines retrieved."
            return "\n\n---\n\n".join(
                f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            )
        except Exception as e:
            logger.error(f"ChromaDB retrieval failed: {e}")
            return "Clinical guideline retrieval failed — using clinical knowledge only."
 
    # ── LLM invocation with JSON fallback ────────────────────────────────────
 
    def _invoke_llm(
        self, patient_query: str, context: str, system_prompt: str
    ) -> DiagnosisOutput:
        """
        Invoke LLM with structured output. Falls back to raw JSON parse if
        with_structured_output fails (handles Gemini occasional schema refusal).
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", HUMAN_PROMPT),
        ])
 
        try:
            chain = prompt | self._structured_llm
            output: DiagnosisOutput = chain.invoke({
                "context": context,
                "patient_json": patient_query,
            })
            return output
        except Exception as e:
            logger.warning(f"Structured LLM output failed ({e}), attempting raw JSON parse")
            return self._invoke_llm_raw_json(patient_query, context, system_prompt)
 
    def _invoke_llm_raw_json(
        self, patient_query: str, context: str, system_prompt: str
    ) -> DiagnosisOutput:
        """
        Fallback: call LLM without structured output, parse JSON manually.
        Handles cases where Gemini refuses the schema for complex inputs.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", HUMAN_PROMPT),
        ])
        chain = prompt | self._llm
        raw = chain.invoke({"context": context, "patient_json": patient_query})
        text = raw.content if hasattr(raw, "content") else str(raw)
 
        # Strip markdown fences
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
 
        try:
            data = json.loads(text)
        except json.JSONDecodeError as je:
            raise RuntimeError(f"LLM returned non-JSON output: {je}\n---\n{text[:500]}")
 
        # Repair ICD-10 codes before Pydantic validation
        if "differential_diagnosis" in data:
            data["differential_diagnosis"] = _validate_and_repair_items(
                data["differential_diagnosis"]
            )
 
        # Convert recommended_next_steps to NextStep objects if they're dicts
        if "recommended_next_steps" in data:
            steps = []
            for s in data["recommended_next_steps"]:
                if isinstance(s, str):
                    # Old format: plain string → wrap as MEDICATION or INVESTIGATION
                    steps.append(NextStep(
                        category="INVESTIGATION",
                        description=s,
                        urgency="routine",
                    ))
                elif isinstance(s, dict):
                    steps.append(NextStep(**s))
            data["recommended_next_steps"] = steps
 
        return DiagnosisOutput(**data)
    

    # ── Public interface ──────────────────────────────────────────────────────
 
    def run(
        self,
        patient_state: dict,
        chief_complaint: str,
        request_id: Optional[str] = None,
    ) -> DiagnosisOutput:
        """
        Main entry point. Checks cache first, then runs RAG or fallback.
        """
        if not self._initialized:
            raise RuntimeError("DiagnosisRAG not initialized. Call initialize() or initialize_fallback() first.")
 
        patient_id = patient_state.get("patient_id", "unknown")
        log_prefix = f"[{request_id}]" if request_id else f"[{patient_id}]"
 
        # Cache check
        cached = _cache.get(patient_id, chief_complaint)
        if cached:
            logger.info(f"{log_prefix} Returning cached diagnosis result")
            return cached
 
        t0 = time.time()
        patient_query = self._build_patient_json(patient_state, chief_complaint)
 
        if self._rag_available:
            logger.info(f"{log_prefix} Running RAG retrieval + Gemini inference")
            context = self._retrieve_context(patient_query, k=6)
            system_prompt = SYSTEM_PROMPT
        else:
            logger.warning(f"{log_prefix} Running LLM-only fallback (no RAG context)")
            context = "No clinical guidelines available — apply clinical judgment."
            system_prompt = FALLBACK_SYSTEM_PROMPT
 
        output = self._invoke_llm(patient_query, context, system_prompt)
        output = self._apply_rule_adjustments(output, patient_state)
 
        # Sync top_diagnosis / top_icd10_code from rank-1 item
        if output.differential_diagnosis:
            top = output.differential_diagnosis[0]
            output.top_diagnosis = top.display
            output.top_icd10_code = top.icd10_code
 
        elapsed = round(time.time() - t0, 2)
        logger.info(
            f"{log_prefix} Diagnosis complete in {elapsed}s — "
            f"{output.top_diagnosis} ({output.top_icd10_code}) {output.confidence_level}"
            f"{' [FALLBACK MODE]' if not self._rag_available else ''}"
        )
 
        _cache.set(patient_id, chief_complaint, output)
        return output
 
    def build_fhir_conditions(self, output: DiagnosisOutput, patient_id: str) -> list[dict]:
        """
        Build FHIR Condition resources from diagnosis output.
 
        Verification status is confidence-dependent:
          ≥ 0.75 → 'provisional'   (high enough confidence to act on)
          ≥ 0.50 → 'differential'  (working diagnosis, needs confirmation)
          < 0.50 → 'refuted'       (low confidence, listed for completeness)
        """
        conditions = []
        for diag in output.differential_diagnosis[:3]:
            if diag.confidence >= 0.75:
                verification_code = "provisional"
            elif diag.confidence >= 0.50:
                verification_code = "differential"
            else:
                verification_code = "refuted"
 
            conditions.append({
                "resourceType": "Condition",
                "subject": {"reference": f"Patient/{patient_id}"},
                "code": {
                    "coding": [{
                        "system": "http://hl7.org/fhir/sid/icd-10",
                        "code": diag.icd10_code,
                        "display": diag.display,
                    }]
                },
                "verificationStatus": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": verification_code,
                        "display": verification_code.capitalize(),
                    }]
                },
                "clinicalStatus": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                    }]
                },
                "note": [{
                    "text": (
                        f"AI differential rank {diag.rank}. "
                        f"Confidence: {diag.confidence:.0%}. "
                        f"{diag.clinical_reasoning}. "
                        f"Supporting: {'; '.join(diag.supporting_evidence[:3])}. "
                        "REQUIRES PHYSICIAN VERIFICATION — not for direct clinical use."
                    )
                }],
            })
        return conditions
 
 
# Singleton
diagnosis = DiagnosisRAG()