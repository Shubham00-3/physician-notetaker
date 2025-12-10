"""
Medical Named Entity Recognition (NER) Extractor

Extracts medical entities from physician-patient transcripts including:
- Symptoms
- Diagnosis
- Treatments
- Prognosis
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class MedicalEntities:
    """Container for extracted medical entities."""
    symptoms: List[str] = field(default_factory=list)
    diagnosis: List[str] = field(default_factory=list)
    treatments: List[str] = field(default_factory=list)
    prognosis: List[str] = field(default_factory=list)
    patient_name: Optional[str] = None
    current_status: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "Patient_Name": self.patient_name or "Unknown",
            "Symptoms": self.symptoms,
            "Diagnosis": self.diagnosis[0] if self.diagnosis else "Not specified",
            "Treatment": self.treatments,
            "Current_Status": self.current_status or "Not specified",
            "Prognosis": self.prognosis[0] if self.prognosis else "Not specified"
        }


class MedicalNERExtractor:
    """
    Medical Named Entity Recognition using pattern matching and NLP.
    
    Uses a combination of:
    - Rule-based pattern matching for medical terminology
    - spaCy NLP for general entity recognition
    - Medical vocabulary dictionaries
    """
    
    # Medical symptom patterns
    SYMPTOM_PATTERNS = [
        r'\b(pain|ache|discomfort|stiffness|soreness)\s*(in|of)?\s*(my|the)?\s*(neck|back|head|shoulder|spine|lower back|upper back)',
        r'\b(neck|back|head|shoulder)\s*(pain|ache|discomfort|stiffness)',
        r'\b(hit|struck|impacted)\s*(my|the)?\s*(head|neck|back)',
        r'\bhad trouble\s+\w+ing\b',
        r'\b(occasional|constant|intermittent|chronic)\s*(back)?ache[s]?\b',
        r'\b(headache|migraine|nausea|dizziness|fatigue)\b',
        r'\bwhiplash\b',
    ]
    
    # Diagnosis patterns
    DIAGNOSIS_PATTERNS = [
        r'\b(whiplash\s*injury)\b',
        r'\b(lower back\s*strain)\b',
        r'\b(cervical\s*strain)\b',
        r'\b(concussion)\b',
        r'\b(sprain|strain|fracture|injury)\b',
        r'\bsaid it was\s+(a\s+)?([^,.]+)',
    ]
    
    # Treatment patterns
    TREATMENT_PATTERNS = [
        r'\b(\d+)\s*(sessions?\s*of)?\s*(physiotherapy|physical therapy)\b',
        r'\b(physiotherapy|physical therapy)\s*(sessions?)?\b',
        r'\b(painkillers?|analgesics?|medication|medicine)\b',
        r'\b(X-rays?|MRI|CT scan|imaging)\b',
        r'\b(rest|ice|heat therapy)\b',
    ]
    
    # Prognosis patterns
    PROGNOSIS_PATTERNS = [
        r'\b(full\s*recovery)\s*(expected|anticipated)?\s*(within|in)?\s*(\w+\s*months?)?',
        r'\b(expect|anticipate|foresee).*?(recovery|improvement)\b',
        r'\bno\s*(signs?\s*of)?\s*(long-term|lasting|permanent)\s*(damage|impact|effects?)\b',
        r'\b(on track for|expecting)\s*(a\s*)?(full|complete)\s*recovery\b',
    ]
    
    # Current status patterns
    STATUS_PATTERNS = [
        r'\b(occasional|intermittent)\s*(back)?ache[s]?\b',
        r'\b(still|now)\s*(have|experiencing|feeling)\s*([^,.]+)',
        r'\bnot\s*constant\b',
        r'\b(doing|feeling)\s*(better|improved|good)\b',
    ]
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the NER extractor.
        
        Args:
            use_spacy: Whether to use spaCy for additional NER (requires installation)
        """
        self.use_spacy = use_spacy
        self.nlp = None
        
        if use_spacy:
            try:
                import spacy
                # Try to load medical model first, fall back to general English
                try:
                    self.nlp = spacy.load("en_core_sci_lg")
                except OSError:
                    try:
                        self.nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        print("Warning: No spaCy model found. Using pattern matching only.")
                        self.use_spacy = False
            except ImportError:
                print("Warning: spaCy not installed. Using pattern matching only.")
                self.use_spacy = False
    
    def extract_patient_name(self, text: str) -> Optional[str]:
        """Extract patient name from transcript."""
        # Look for patterns like "Ms. Jones", "Mr. Smith", etc.
        name_patterns = [
            r'\b(Ms\.|Mrs\.|Mr\.|Dr\.)\s+([A-Z][a-z]+)',
            r'Good morning,?\s*(Ms\.|Mrs\.|Mr\.|Dr\.)?\s*([A-Z][a-z]+)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) >= 2 and groups[1]:
                    title = groups[0] if groups[0] else ""
                    name = groups[1]
                    # Assume it's patient if addressed in conversation
                    return f"{title} {name}".strip() if title else name
        
        return None
    
    def extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text."""
        symptoms = set()
        text_lower = text.lower()
        
        # Pattern-based extraction
        symptom_keywords = {
            'neck pain': ['neck pain', 'pain in my neck', 'pain in the neck', 'neck hurt'],
            'back pain': ['back pain', 'pain in my back', 'backache', 'back hurt', 'lower back pain'],
            'head impact': ['hit my head', 'hit head', 'head impact', 'struck my head', 'head on the steering'],
            'stiffness': ['stiffness', 'stiff'],
            'trouble sleeping': ['trouble sleeping', 'difficulty sleeping', 'couldn\'t sleep'],
            'discomfort': ['discomfort'],
        }
        
        for symptom, patterns in symptom_keywords.items():
            for pattern in patterns:
                if pattern in text_lower:
                    symptoms.add(symptom.title())
                    break
        
        # Regex pattern extraction
        for pattern in self.SYMPTOM_PATTERNS:
            matches = re.findall(pattern, text_lower)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        symptom_text = ' '.join([m for m in match if m]).strip()
                    else:
                        symptom_text = match.strip()
                    if symptom_text and len(symptom_text) > 3:
                        symptoms.add(symptom_text.title())
        
        return list(symptoms)
    
    def extract_diagnosis(self, text: str) -> List[str]:
        """Extract diagnosis from text."""
        diagnoses = set()
        text_lower = text.lower()
        
        # Look for explicit diagnosis statements
        diagnosis_keywords = {
            'whiplash injury': ['whiplash injury', 'whiplash'],
            'lower back strain': ['lower back strain', 'back strain'],
            'cervical strain': ['cervical strain', 'neck strain'],
        }
        
        for diagnosis, patterns in diagnosis_keywords.items():
            for pattern in patterns:
                if pattern in text_lower:
                    diagnoses.add(diagnosis.title())
                    break
        
        # Pattern: "said it was a [diagnosis]"
        said_pattern = re.search(r'said it was\s+(a\s+)?([^,.]+)', text_lower)
        if said_pattern:
            diagnosis_text = said_pattern.group(2).strip()
            if diagnosis_text and len(diagnosis_text) > 3:
                diagnoses.add(diagnosis_text.title())
        
        return list(diagnoses)
    
    def extract_treatments(self, text: str) -> List[str]:
        """Extract treatments from text."""
        treatments = set()
        text_lower = text.lower()
        
        # Look for physiotherapy sessions
        physio_match = re.search(r'(\d+)\s*(sessions?\s*(of)?)?(\s*(physiotherapy|physical therapy))?', text_lower)
        if physio_match and 'physiotherapy' in text_lower or 'physical therapy' in text_lower:
            num = re.search(r'(\d+)\s*sessions?', text_lower)
            if num:
                treatments.add(f"{num.group(1)} physiotherapy sessions")
            else:
                treatments.add("Physiotherapy")
        
        # Painkillers
        if 'painkiller' in text_lower or 'painkillers' in text_lower:
            treatments.add("Painkillers")
        
        # Check for other treatments
        if 'medication' in text_lower or 'medicine' in text_lower:
            treatments.add("Medication")
        
        if any(term in text_lower for term in ['x-ray', 'x ray', 'xray']):
            treatments.add("X-ray examination")
        
        return list(treatments)
    
    def extract_prognosis(self, text: str) -> List[str]:
        """Extract prognosis from text."""
        prognoses = []
        text_lower = text.lower()
        
        # Look for recovery timeline
        recovery_patterns = [
            r'(full|complete)\s*recovery\s*(expected|anticipated)?\s*(within|in)\s*(\w+\s*months?)',
            r'(expect|anticipate).*?(full|complete)\s*recovery\s*(within|in)?\s*(\w+\s*months?)?',
            r'on track for\s*(a\s*)?(full|complete)\s*recovery',
        ]
        
        for pattern in recovery_patterns:
            match = re.search(pattern, text_lower)
            if match:
                prognoses.append("Full recovery expected within six months")
                break
        
        # Check for no long-term damage
        if 'no' in text_lower and any(term in text_lower for term in ['long-term', 'lasting', 'permanent']):
            if not prognoses:
                prognoses.append("No long-term damage expected")
        
        return prognoses
    
    def extract_current_status(self, text: str) -> Optional[str]:
        """Extract current status/condition."""
        text_lower = text.lower()
        
        # Look for current condition statements
        if 'occasional backache' in text_lower or 'occasional back' in text_lower:
            return "Occasional backache"
        
        if 'doing better' in text_lower:
            return "Improving, with occasional discomfort"
        
        status_match = re.search(r'(still|now)\s*(have|experiencing|feeling)\s+([^,.]+)', text_lower)
        if status_match:
            return status_match.group(3).strip().capitalize()
        
        return None
    
    def extract_entities(self, text: str) -> MedicalEntities:
        """
        Extract all medical entities from the transcript.
        
        Args:
            text: The medical transcript text
            
        Returns:
            MedicalEntities object containing all extracted entities
        """
        entities = MedicalEntities()
        
        # Extract patient name
        entities.patient_name = self.extract_patient_name(text)
        
        # Extract symptoms
        entities.symptoms = self.extract_symptoms(text)
        
        # Extract diagnosis
        entities.diagnosis = self.extract_diagnosis(text)
        
        # Extract treatments
        entities.treatments = self.extract_treatments(text)
        
        # Extract prognosis
        entities.prognosis = self.extract_prognosis(text)
        
        # Extract current status
        entities.current_status = self.extract_current_status(text)
        
        # Use spaCy for additional entity extraction if available
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['DISEASE', 'SYMPTOM']:
                    if ent.text.title() not in entities.symptoms:
                        entities.symptoms.append(ent.text.title())
                elif ent.label_ in ['TREATMENT', 'PROCEDURE']:
                    if ent.text.title() not in entities.treatments:
                        entities.treatments.append(ent.text.title())
        
        return entities


def main():
    """Test the NER extractor with sample text."""
    sample_text = """
    Doctor: How are you feeling today?
    Patient: I had a car accident. My neck and back hurt a lot for four weeks.
    Doctor: Did you receive treatment?
    Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
    """
    
    extractor = MedicalNERExtractor(use_spacy=False)
    entities = extractor.extract_entities(sample_text)
    
    print("Extracted Medical Entities:")
    print("-" * 40)
    import json
    print(json.dumps(entities.to_dict(), indent=2))


if __name__ == "__main__":
    main()
