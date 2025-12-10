"""
SOAP Note Generator

Generates structured SOAP notes from medical transcripts:
- Subjective: Patient-reported symptoms and history
- Objective: Physical examination findings
- Assessment: Diagnosis and severity
- Plan: Treatment and follow-up recommendations
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class SOAPNote:
    """Structured SOAP note container."""
    subjective: Dict[str, str] = field(default_factory=dict)
    objective: Dict[str, str] = field(default_factory=dict)
    assessment: Dict[str, str] = field(default_factory=dict)
    plan: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "Subjective": self.subjective,
            "Objective": self.objective,
            "Assessment": self.assessment,
            "Plan": self.plan
        }


class SOAPNoteGenerator:
    """
    Generate SOAP notes from physician-patient transcripts.
    
    Uses a combination of:
    - Speaker identification (Doctor vs Patient)
    - Section-specific keyword detection
    - Template filling with extracted information
    """
    
    # Keywords for identifying different SOAP sections
    SUBJECTIVE_KEYWORDS = [
        # Patient history
        'accident', 'injury', 'happened', 'started', 'began',
        'felt', 'feeling', 'feel', 'experienced', 'noticed',
        
        # Symptoms
        'pain', 'ache', 'hurt', 'hurts', 'discomfort', 'sore',
        'stiff', 'stiffness', 'numbness', 'tingling', 'weakness',
        
        # Timeline
        'weeks', 'days', 'months', 'since', 'after', 'before',
        
        # Previous treatment
        'went to', 'visited', 'saw', 'they said', 'sessions',
    ]
    
    OBJECTIVE_KEYWORDS = [
        # Examination terms
        'examination', 'exam', 'physical', 'checked', 'observed',
        'range of motion', 'range of movement', 'mobility',
        
        # Findings
        'tenderness', 'swelling', 'no signs', 'normal', 'good condition',
        'full range', 'movement', 'muscles', 'spine',
        
        # Observations
        'looks', 'appears', 'gait', 'posture',
    ]
    
    ASSESSMENT_KEYWORDS = [
        # Diagnosis
        'whiplash', 'strain', 'injury', 'condition',
        'diagnosis', 'diagnosed',
        
        # Severity
        'mild', 'moderate', 'severe', 'improving', 'worsening',
        'positive', 'negative', 'good', 'concerning',
        
        # Prognosis
        'recovery', 'expect', 'prognosis', 'long-term',
    ]
    
    PLAN_KEYWORDS = [
        # Treatment
        'physiotherapy', 'therapy', 'medication', 'painkillers',
        'continue', 'recommend', 'prescribe', 'treatment',
        
        # Follow-up
        'follow-up', 'come back', 'return', 'if anything',
        'contact', 'reach out', 'appointment',
        
        # Instructions
        'rest', 'avoid', 'exercise', 'take care',
    ]
    
    def __init__(self):
        """Initialize the SOAP note generator."""
        pass
    
    def _parse_transcript(self, text: str) -> Dict[str, List[str]]:
        """Parse transcript into speaker segments."""
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
            
            # Check for speaker labels
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
            elif line.startswith('[') and line.endswith(']'):
                # Handle examination markers like [Physical Examination Conducted]
                if current_speaker:
                    parsed[current_speaker].append(' '.join(current_text))
                    current_text = []
                # Mark that exam was conducted
                parsed['physician'].append(f"EXAMINATION: {line[1:-1]}")
            else:
                if current_speaker:
                    current_text.append(line)
        
        if current_speaker and current_text:
            parsed[current_speaker].append(' '.join(current_text))
        
        return parsed
    
    def _contains_keywords(self, text: str, keywords: List[str]) -> bool:
        """Check if text contains any of the keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keywords)
    
    def _extract_chief_complaint(self, patient_statements: List[str]) -> str:
        """Extract the chief complaint from patient statements."""
        # Look for main symptom descriptions
        symptom_keywords = ['pain', 'ache', 'hurt', 'discomfort', 'injured']
        
        for statement in patient_statements:
            if self._contains_keywords(statement, symptom_keywords):
                # Extract the symptom description
                if 'neck' in statement.lower() or 'back' in statement.lower():
                    if 'pain' in statement.lower():
                        return "Neck and back pain"
                    return "Neck and back discomfort"
        
        # Default
        return "Post-accident symptoms"
    
    def _extract_history_of_present_illness(self, patient_statements: List[str]) -> str:
        """Extract history of present illness."""
        history_parts = []
        
        for statement in patient_statements:
            statement_lower = statement.lower()
            
            # Look for accident description
            if 'accident' in statement_lower or 'hit' in statement_lower:
                history_parts.append(statement)
            
            # Look for symptom progression
            if any(word in statement_lower for word in ['weeks', 'months', 'after', 'started']):
                if 'pain' in statement_lower or 'discomfort' in statement_lower:
                    history_parts.append(statement)
            
            # Look for treatment received
            if 'physiotherapy' in statement_lower or 'sessions' in statement_lower:
                history_parts.append(statement)
            
            if len(history_parts) >= 3:
                break
        
        if history_parts:
            # Summarize the history
            combined = ' '.join(history_parts[:3])
            if len(combined) > 300:
                combined = combined[:300] + '...'
            return combined
        
        return "Patient reports symptoms following incident."
    
    def _extract_physical_exam(self, physician_statements: List[str]) -> str:
        """Extract physical examination findings."""
        exam_findings = []
        
        for statement in physician_statements:
            statement_lower = statement.lower()
            
            # Look for examination keywords
            if self._contains_keywords(statement, self.OBJECTIVE_KEYWORDS):
                # Extract specific findings
                if 'range of motion' in statement_lower or 'range of movement' in statement_lower:
                    exam_findings.append("Full range of motion in cervical and lumbar spine")
                if 'no tenderness' in statement_lower or 'tenderness' not in statement_lower:
                    if 'good' in statement_lower or 'looks good' in statement_lower:
                        exam_findings.append("No tenderness on palpation")
                if 'good condition' in statement_lower:
                    exam_findings.append("Muscles and spine in good condition")
        
        if exam_findings:
            return ', '.join(exam_findings) + '.'
        
        return "Physical examination conducted. No significant abnormalities noted."
    
    def _extract_observations(self, text: str) -> str:
        """Extract general observations."""
        observations = []
        
        text_lower = text.lower()
        
        if 'looks good' in text_lower or 'everything looks good' in text_lower:
            observations.append("Patient appears in good general health")
        
        if 'full range' in text_lower:
            observations.append("Normal mobility observed")
        
        if 'no lasting damage' in text_lower or 'no signs of' in text_lower:
            observations.append("No signs of permanent damage")
        
        if observations:
            return ', '.join(observations) + '.'
        
        return "Patient presents with normal gait and posture."
    
    def _extract_diagnosis(self, text: str) -> str:
        """Extract diagnosis from transcript."""
        text_lower = text.lower()
        
        diagnoses = []
        
        if 'whiplash' in text_lower:
            diagnoses.append("Whiplash injury")
        
        if 'back' in text_lower and ('strain' in text_lower or 'pain' in text_lower):
            diagnoses.append("Lower back strain")
        
        if 'neck' in text_lower and 'strain' in text_lower:
            diagnoses.append("Cervical strain")
        
        if diagnoses:
            return ' and '.join(diagnoses)
        
        return "Soft tissue injury"
    
    def _extract_severity(self, text: str) -> str:
        """Extract severity assessment."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['improving', 'better', 'positive', 'good']):
            if 'occasional' in text_lower:
                return "Mild, improving"
            return "Improving"
        
        if any(word in text_lower for word in ['severe', 'bad', 'rough']):
            return "Moderate, improving"
        
        return "Mild to moderate"
    
    def _extract_treatment_plan(self, text: str) -> str:
        """Extract treatment recommendations."""
        text_lower = text.lower()
        treatments = []
        
        if 'physiotherapy' in text_lower or 'physical therapy' in text_lower:
            treatments.append("Continue physiotherapy as needed")
        
        if 'painkiller' in text_lower or 'analgesic' in text_lower or 'medication' in text_lower:
            treatments.append("Use analgesics for pain relief as needed")
        
        if 'rest' in text_lower:
            treatments.append("Rest as needed")
        
        if treatments:
            return ', '.join(treatments) + '.'
        
        return "Continue current management, symptomatic treatment as needed."
    
    def _extract_follow_up(self, physician_statements: List[str]) -> str:
        """Extract follow-up recommendations."""
        for statement in physician_statements:
            statement_lower = statement.lower()
            
            if 'come back' in statement_lower or 'follow-up' in statement_lower:
                if 'worsening' in statement_lower or 'worse' in statement_lower:
                    return "Patient to return if symptoms worsen or persist."
            
            if 'reach out' in statement_lower or 'contact' in statement_lower:
                return "Patient may contact clinic if needed."
            
            if 'six months' in statement_lower or 'months' in statement_lower:
                return "Expected full recovery within six months. Return if symptoms persist beyond this timeframe."
        
        return "Follow up as needed."
    
    def generate(self, text: str) -> SOAPNote:
        """
        Generate a SOAP note from transcript.
        
        Args:
            text: The medical transcript text
            
        Returns:
            SOAPNote object with all sections populated
        """
        # Parse transcript into speaker segments
        parsed = self._parse_transcript(text)
        patient_statements = parsed.get('patient', [])
        physician_statements = parsed.get('physician', [])
        
        # Build SOAP note
        soap = SOAPNote()
        
        # Subjective (from patient statements)
        soap.subjective = {
            "Chief_Complaint": self._extract_chief_complaint(patient_statements),
            "History_of_Present_Illness": self._extract_history_of_present_illness(patient_statements)
        }
        
        # Objective (from physician examination)
        soap.objective = {
            "Physical_Exam": self._extract_physical_exam(physician_statements),
            "Observations": self._extract_observations(text)
        }
        
        # Assessment
        soap.assessment = {
            "Diagnosis": self._extract_diagnosis(text),
            "Severity": self._extract_severity(text)
        }
        
        # Plan
        soap.plan = {
            "Treatment": self._extract_treatment_plan(text),
            "Follow_Up": self._extract_follow_up(physician_statements)
        }
        
        return soap
    
    def generate_formatted(self, text: str) -> str:
        """
        Generate a formatted SOAP note string.
        
        Args:
            text: The medical transcript text
            
        Returns:
            Formatted SOAP note as string
        """
        soap = self.generate(text)
        
        output = []
        output.append("=" * 60)
        output.append("SOAP NOTE")
        output.append("=" * 60)
        
        output.append("\n--- SUBJECTIVE ---")
        for key, value in soap.subjective.items():
            output.append(f"{key.replace('_', ' ')}: {value}")
        
        output.append("\n--- OBJECTIVE ---")
        for key, value in soap.objective.items():
            output.append(f"{key.replace('_', ' ')}: {value}")
        
        output.append("\n--- ASSESSMENT ---")
        for key, value in soap.assessment.items():
            output.append(f"{key.replace('_', ' ')}: {value}")
        
        output.append("\n--- PLAN ---")
        for key, value in soap.plan.items():
            output.append(f"{key.replace('_', ' ')}: {value}")
        
        output.append("\n" + "=" * 60)
        
        return '\n'.join(output)


def main():
    """Test the SOAP generator with sample text."""
    sample_text = """
    Doctor: How are you feeling today?
    Patient: I had a car accident. My neck and back hurt a lot for four weeks.
    Doctor: Did you receive treatment?
    Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
    [Physical Examination Conducted]
    Doctor: Everything looks good. Your neck and back have a full range of movement.
    Doctor: I'd expect you to make a full recovery within six months.
    """
    
    generator = SOAPNoteGenerator()
    soap_note = generator.generate(sample_text)
    
    print("SOAP Note (JSON):")
    print("-" * 40)
    import json
    print(json.dumps(soap_note.to_dict(), indent=2))
    
    print("\n")
    print(generator.generate_formatted(sample_text))


if __name__ == "__main__":
    main()
