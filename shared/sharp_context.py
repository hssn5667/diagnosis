"""
SHARP context header parsing
Handles SHARP (Standardized Healthcare API Request Protocol) headers from Prompt Opinion platform
"""
from typing import Optional
from fastapi import Header


class SHARPContext:
    """SHARP context from Prompt Opinion platform"""
    
    def __init__(
        self,
        patient_id: Optional[str] = None,
        fhir_token: Optional[str] = None,
        fhir_base_url: Optional[str] = None
    ):
        self.patient_id = patient_id
        self.fhir_token = fhir_token
        self.fhir_base_url = fhir_base_url or "https://hapi.fhir.org/baseR4"
    
    @classmethod
    def from_headers(
        cls,
        x_sharp_patient_id: Optional[str] = Header(None),
        x_sharp_fhir_token: Optional[str] = Header(None),
        x_sharp_fhir_base_url: Optional[str] = Header(None)
    ):
        """Parse SHARP context from FastAPI headers"""
        return cls(
            patient_id=x_sharp_patient_id,
            fhir_token=x_sharp_fhir_token,
            fhir_base_url=x_sharp_fhir_base_url
        )
    
    def get_auth_header(self) -> dict:
        """Get authorization header for FHIR requests"""
        if self.fhir_token:
            return {"Authorization": self.fhir_token}
        return {"none": "none"}
    
    @property
    def is_valid(self) -> bool:
        """Check if SHARP context has minimum required fields"""
        return self.patient_id is not None