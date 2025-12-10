"""
Medical Text Summarizer

Generates structured medical reports from physician-patient transcripts.
Uses template-based structuring combined with NER extraction.
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MedicalSummary:
    """Structured medical summary."""
    patient_name: str
    symptoms: List[str]
    diagnosis: str
    treatment: List[str]
    current_status: str
    prognosis: str
    
    def to_dict(self) -> Dict:
        return {
            "Patient_Name": self.patient_name,
            "Symptoms": self.symptoms,
            "Diagnosis": self.diagnosis,
            "Treatment": self.treatment,
            "Current_Status": self.current_status,
            "Prognosis": self.prognosis
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class MedicalSummarizer:
    """
    Medical transcript summarizer that generates structured reports.
    
    Can use either:
    - Template-based summarization with NER extraction
    - Transformer-based abstractive summarization (if available)
    """
    
    def __init__(self, use_transformers: bool = True):
        """
        Initialize the summarizer.
        
        Args:
            use_transformers: Whether to use transformer models for summarization
        """
        self.use_transformers = use_transformers
        self.summarizer = None
        
        if use_transformers:
            try:
                from transformers import pipeline
                # Load summarization pipeline
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1  # CPU
                )
            except Exception as e:
                print(f"Warning: Could not load transformer model: {e}")
                print("Falling back to template-based summarization.")
                self.use_transformers = False
    
    def _parse_transcript(self, text: str) -> Dict[str, List[str]]:
        """Parse transcript into speaker turns."""
        lines = text.strip().split('\n')
        parsed = {
            'physician': [],
            'patient': []
        }
        
        current_speaker = None
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.lower().startswith(('physician:', 'doctor:')):
                if current_speaker and current_text:
                    parsed[current_speaker].append(' '.join(current_text))
                current_speaker = 'physician'
                text_part = line.split(':', 1)[1].strip() if ':' in line else ''
                current_text = [text_part] if text_part else []
            elif line.lower().startswith('patient:'):
                if current_speaker and current_text:
                    parsed[current_speaker].append(' '.join(current_text))
                current_speaker = 'patient'
                text_part = line.split(':', 1)[1].strip() if ':' in line else ''
                current_text = [text_part] if text_part else []
            else:
                if current_speaker:
                    current_text.append(line)
        
        if current_speaker and current_text:
            parsed[current_speaker].append(' '.join(current_text))
        
        return parsed
    
    def _extract_abstractive_summary(self, text: str, max_length: int = 150) -> str:
        """Generate abstractive summary using transformers."""
        if not self.summarizer:
            return ""
        
        try:
            # Truncate text if too long
            if len(text) > 1024:
                text = text[:1024]
            
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Summarization error: {e}")
            return ""
    
    def summarize(self, text: str, entities: Optional[Dict] = None) -> MedicalSummary:
        """
        Generate a structured medical summary from transcript.
        
        Args:
            text: The raw transcript text
            entities: Optional pre-extracted entities from NER
            
        Returns:
            MedicalSummary object
        """
        # If entities not provided, use basic extraction
        if entities is None:
            from .ner_extractor import MedicalNERExtractor
            extractor = MedicalNERExtractor(use_spacy=False)
            entities = extractor.extract_entities(text).to_dict()
        
        # Create summary from entities
        summary = MedicalSummary(
            patient_name=entities.get('Patient_Name', 'Unknown'),
            symptoms=entities.get('Symptoms', []),
            diagnosis=entities.get('Diagnosis', 'Not specified'),
            treatment=entities.get('Treatment', []),
            current_status=entities.get('Current_Status', 'Not specified'),
            prognosis=entities.get('Prognosis', 'Not specified')
        )
        
        return summary
    
    def generate_narrative_summary(self, text: str) -> str:
        """Generate a narrative (prose) summary of the transcript."""
        if self.use_transformers and self.summarizer:
            return self._extract_abstractive_summary(text)
        
        # Fallback to template-based narrative
        parsed = self._parse_transcript(text)
        patient_statements = ' '.join(parsed.get('patient', []))
        
        # Simple extractive summary
        sentences = patient_statements.split('.')
        key_sentences = [s.strip() for s in sentences[:5] if s.strip()]
        
        return '. '.join(key_sentences) + '.' if key_sentences else "No summary available."
    
    def generate_report(self, text: str, include_narrative: bool = False) -> Dict:
        """
        Generate a complete medical report.
        
        Args:
            text: The transcript text
            include_narrative: Whether to include narrative summary
            
        Returns:
            Dictionary containing structured summary and optional narrative
        """
        summary = self.summarize(text)
        report = summary.to_dict()
        
        if include_narrative:
            report['Narrative_Summary'] = self.generate_narrative_summary(text)
        
        return report


def main():
    """Test the summarizer with sample text."""
    sample_text = """
    Doctor: How are you feeling today?
    Patient: I had a car accident. My neck and back hurt a lot for four weeks.
    Doctor: Did you receive treatment?
    Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
    """
    
    summarizer = MedicalSummarizer(use_transformers=False)
    report = summarizer.generate_report(sample_text, include_narrative=True)
    
    print("Medical Summary Report:")
    print("-" * 40)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
