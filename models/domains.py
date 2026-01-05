"""
Domain-Specific Models

Custom models for healthcare, legal, and other verticals.
"""

import logging
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class DomainType(str, Enum):
    """Supported domain verticals."""
    GENERAL = "general"
    HEALTHCARE = "healthcare"
    LEGAL = "legal"
    FINANCE = "finance"
    CUSTOMER_SERVICE = "customer_service"
    EDUCATION = "education"
    MEDIA = "media"


@dataclass
class DomainConfig:
    """Configuration for a domain model."""
    domain: DomainType
    vocabulary: List[str]
    prompt_template: str
    post_processing: Optional[Dict[str, Any]] = None


# Domain-specific vocabularies for better transcription
DOMAIN_VOCABULARIES = {
    DomainType.HEALTHCARE: [
        "HIPAA", "diagnosis", "prognosis", "contraindication", "etiology",
        "symptom", "syndrome", "pathology", "cardiology", "neurology",
        "oncology", "radiology", "hematology", "immunology", "endocrinology",
        "patient", "physician", "nurse", "prescription", "medication",
        "dosage", "mg", "ml", "IV", "IM", "PO", "PRN", "BID", "TID", "QID",
        "hypertension", "diabetes", "cholesterol", "anemia", "arrhythmia",
        "MRI", "CT scan", "X-ray", "ultrasound", "EKG", "ECG", "CBC",
    ],
    DomainType.LEGAL: [
        "plaintiff", "defendant", "appellant", "appellee", "jurisdiction",
        "liability", "precedent", "statute", "deposition", "affidavit",
        "testimony", "cross-examination", "objection", "sustained", "overruled",
        "habeas corpus", "pro bono", "amicus curiae", "voir dire", "subpoena",
        "tort", "negligence", "breach", "contract", "damages", "injunction",
        "motion", "ruling", "verdict", "settlement", "litigation", "arbitration",
    ],
    DomainType.FINANCE: [
        "asset", "liability", "equity", "revenue", "expense", "EBITDA",
        "ROI", "ROE", "P/E ratio", "dividend", "portfolio", "hedge",
        "derivative", "option", "futures", "bond", "stock", "ETF", "mutual fund",
        "IPO", "M&A", "due diligence", "valuation", "DCF", "WACC", "NPV",
        "compliance", "SEC", "FINRA", "AML", "KYC", "fiduciary",
    ],
    DomainType.CUSTOMER_SERVICE: [
        "ticket", "escalation", "SLA", "resolution", "refund", "exchange",
        "warranty", "return", "complaint", "feedback", "satisfaction",
        "NPS", "CSAT", "first call resolution", "average handle time",
        "queue", "routing", "IVR", "callback", "hold time", "transfer",
    ],
}

# Domain-specific prompts for better RAG responses
DOMAIN_PROMPTS = {
    DomainType.HEALTHCARE: """You are a medical documentation assistant analyzing clinical conversations.
Important guidelines:
- Maintain patient confidentiality
- Use precise medical terminology
- Note any drug names, dosages, and frequencies mentioned
- Identify symptoms, diagnoses, and treatment plans
- Flag any potential adverse reactions or contraindications

Context information:
{context}

Question: {query}
Answer (using clinical precision):""",

    DomainType.LEGAL: """You are a legal research assistant analyzing legal proceedings.
Important guidelines:
- Reference specific statements with timestamps and speakers
- Identify key legal arguments and precedents mentioned
- Note any objections and rulings
- Highlight testimony that may be relevant to the case
- Maintain objectivity and avoid legal advice

Context information:
{context}

Question: {query}
Answer (with legal precision):""",

    DomainType.FINANCE: """You are a financial analyst assistant reviewing financial discussions.
Important guidelines:
- Extract specific numbers, metrics, and financial terms
- Identify key financial decisions and rationales
- Note any risk factors or compliance concerns mentioned
- Highlight action items and commitments
- Use precise financial terminology

Context information:
{context}

Question: {query}
Answer (with financial precision):""",

    DomainType.CUSTOMER_SERVICE: """You are a customer service quality analyst reviewing support interactions.
Important guidelines:
- Identify the customer's primary concern
- Track how issues were addressed and resolved
- Note any escalations or transfers
- Identify opportunities for service improvement
- Highlight positive and negative customer sentiments

Context information:
{context}

Question: {query}
Answer:""",

    DomainType.GENERAL: """You are a helpful assistant analyzing audio transcripts.
Answer questions based on the context provided.

Context information:
{context}

Question: {query}
Answer:""",
}


class DomainModel:
    """
    Domain-specific model configuration and processing.
    
    Enhances transcription and RAG for specific verticals.
    """
    
    def __init__(self, domain: DomainType = DomainType.GENERAL):
        self.domain = domain
        self.vocabulary = DOMAIN_VOCABULARIES.get(domain, [])
        self.prompt_template = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS[DomainType.GENERAL])
        
        logger.info(f"Initialized domain model: {domain.value}")
    
    def get_vocabulary(self) -> List[str]:
        """Get domain vocabulary for transcription."""
        return self.vocabulary
    
    def get_prompt(self, context: str, query: str) -> str:
        """
        Get domain-specific RAG prompt.
        
        Args:
            context: Retrieved context
            query: User query
            
        Returns:
            Formatted prompt
        """
        return self.prompt_template.format(context=context, query=query)
    
    def post_process_transcript(self, transcript: str) -> str:
        """
        Post-process transcript with domain-specific corrections.
        
        Args:
            transcript: Raw transcript
            
        Returns:
            Corrected transcript
        """
        # Apply domain-specific corrections
        result = transcript
        
        if self.domain == DomainType.HEALTHCARE:
            # Fix common medical transcription errors
            corrections = {
                "be I d": "BID",
                "t I d": "TID",
                "q I d": "QID",
                "p r n": "PRN",
                "milligrams": "mg",
                "milliliters": "ml",
            }
            for wrong, right in corrections.items():
                result = result.replace(wrong, right)
        
        return result
    
    def extract_entities(self, transcript: str) -> Dict[str, List[str]]:
        """
        Extract domain-specific entities from transcript.
        
        Args:
            transcript: Transcript text
            
        Returns:
            Dict of entity types to values
        """
        entities = {}
        
        if self.domain == DomainType.HEALTHCARE:
            # Extract medications, dosages, etc.
            import re
            
            # Simple medication pattern
            med_pattern = r'\b(\d+)\s*(?:mg|ml)\b'
            entities["dosages"] = re.findall(med_pattern, transcript)
            
        elif self.domain == DomainType.LEGAL:
            # Extract case references
            import re
            case_pattern = r'\b([A-Z][a-z]+)\s+v\.?\s+([A-Z][a-z]+)\b'
            entities["cases"] = re.findall(case_pattern, transcript)
        
        return entities


def get_domain_model(domain: str) -> DomainModel:
    """Get domain model by name."""
    try:
        domain_type = DomainType(domain.lower())
    except ValueError:
        domain_type = DomainType.GENERAL
    
    return DomainModel(domain_type)
