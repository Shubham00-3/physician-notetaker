"""
Intent Detector for Medical Transcripts

Identifies patient intent categories:
- Seeking reassurance
- Reporting symptoms
- Expressing concern
- Asking questions
- Providing information
- Acknowledging
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class PatientIntent(Enum):
    """Patient intent categories."""
    SEEKING_REASSURANCE = "Seeking reassurance"
    REPORTING_SYMPTOMS = "Reporting symptoms"
    EXPRESSING_CONCERN = "Expressing concern"
    ASKING_QUESTION = "Asking question"
    PROVIDING_INFORMATION = "Providing information"
    ACKNOWLEDGING = "Acknowledging"
    EXPRESSING_GRATITUDE = "Expressing gratitude"


@dataclass
class IntentResult:
    """Container for intent detection result."""
    primary_intent: str
    secondary_intents: List[str]
    confidence: float
    evidence: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "Intent": self.primary_intent,
            "Secondary_Intents": self.secondary_intents,
            "Confidence": round(self.confidence, 3),
            "Evidence": self.evidence
        }


class IntentDetector:
    """
    Detect patient intent from medical transcript statements.
    
    Uses a hybrid approach:
    - Pattern matching for question/statement detection
    - Keyword matching for intent classification
    - Semantic role analysis
    """
    
    # Intent patterns and keywords
    INTENT_PATTERNS = {
        PatientIntent.SEEKING_REASSURANCE: {
            'patterns': [
                r'\b(will|does|is)\s+(this|it)\s+(affect|impact|get|be)',
                r'\bdon\'t.*need to worry\b',
                r'\b(will|should)\s+i\s+be\s+(okay|fine|alright)',
                r'\b(is|are)\s+(that|this|there)\s+(normal|okay|fine)',
                r'\bhope\s+(it|this|that|things)',
            ],
            'keywords': [
                'will this', 'is this', 'should i worry', 'need to worry',
                'affect me', 'in the future', 'what if', 'hoping',
                'get better', 'be okay', 'is that normal'
            ]
        },
        
        PatientIntent.REPORTING_SYMPTOMS: {
            'patterns': [
                r'\bi\s+(have|had|feel|felt|experience|experienced)\s+\w*\s*(pain|ache|discomfort)',
                r'\bmy\s+\w+\s+(hurts?|aches?|is\s+sore)',
                r'\bi\s+(can\'t|couldn\'t|have\s+trouble)',
                r'\bthe\s+(pain|ache|discomfort)\s+(is|was|started)',
            ],
            'keywords': [
                'pain', 'ache', 'hurts', 'hurt', 'discomfort', 'sore', 'stiff',
                'can\'t sleep', 'trouble sleeping', 'have trouble',
                'felt', 'feeling', 'experiencing'
            ]
        },
        
        PatientIntent.EXPRESSING_CONCERN: {
            'patterns': [
                r'\bi\'m\s+(worried|concerned|afraid|scared|nervous)',
                r'\bi\s+(worry|fear|dread)',
                r'\bwhat\s+if\s+(it|this|things)',
                r'\bcould\s+(this|it)\s+be\s+(serious|bad|dangerous)',
            ],
            'keywords': [
                'worried', 'concerned', 'afraid', 'scared', 'nervous',
                'fear', 'anxious', 'bothers me', 'concerning'
            ]
        },
        
        PatientIntent.ASKING_QUESTION: {
            'patterns': [
                r'^(what|when|where|why|how|is|are|do|does|can|could|will|would|should)',
                r'\?$',
                r'\bcan\s+you\s+tell\s+me\b',
                r'\bwhat\s+does\s+this\s+mean\b',
            ],
            'keywords': [
                'what', 'when', 'where', 'why', 'how', 'which',
                'is it', 'are there', 'do i', 'should i', 'can i'
            ]
        },
        
        PatientIntent.PROVIDING_INFORMATION: {
            'patterns': [
                r'\bi\s+(went|visited|saw|had|took|did)',
                r'\byes,?\s+i\b',
                r'\bit\s+was\s+(on|at|in)\b',
                r'\bthey\s+(said|told|gave|checked)',
            ],
            'keywords': [
                'i went', 'i had', 'i took', 'they said', 'they gave',
                'it was', 'i always', 'i did', 'i was'
            ]
        },
        
        PatientIntent.ACKNOWLEDGING: {
            'patterns': [
                r'^(yes|no|okay|alright|i see|i understand)\b',
                r'\b(got it|understood|makes sense)\b',
            ],
            'keywords': [
                'yes', 'no', 'okay', 'alright', 'i see', 'i understand',
                'got it', 'makes sense', 'of course', 'sure'
            ]
        },
        
        PatientIntent.EXPRESSING_GRATITUDE: {
            'patterns': [
                r'\bthank\s*(you|s)\b',
                r'\bi\s+appreciate\b',
                r'\bso\s+(glad|happy|relieved)\b',
                r'\bthat\'s\s+(great|wonderful|a relief)\b',
            ],
            'keywords': [
                'thank you', 'thanks', 'appreciate', 'grateful',
                'that\'s great', 'that\'s a relief', 'glad', 'relieved'
            ]
        }
    }
    
    def __init__(self):
        """Initialize the intent detector."""
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for intent, data in self.INTENT_PATTERNS.items():
            self.compiled_patterns[intent] = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in data['patterns']
            ]
    
    def _score_intent(self, text: str, intent: PatientIntent) -> Tuple[float, List[str]]:
        """
        Score how well text matches an intent.
        
        Returns:
            Tuple of (score, evidence list)
        """
        text_lower = text.lower()
        score = 0.0
        evidence = []
        
        # Check compiled patterns
        for pattern in self.compiled_patterns[intent]:
            match = pattern.search(text_lower)
            if match:
                score += 2.0
                matched_text = match.group(0)
                if matched_text not in evidence:
                    evidence.append(matched_text)
        
        # Check keywords
        keywords = self.INTENT_PATTERNS[intent]['keywords']
        for keyword in keywords:
            if keyword in text_lower:
                score += 1.0
                if keyword not in evidence:
                    evidence.append(keyword)
        
        # Bonus for question marks (for questions)
        if intent == PatientIntent.ASKING_QUESTION and '?' in text:
            score += 1.5
            if '?' not in evidence:
                evidence.append('question mark')
        
        return score, evidence
    
    def detect(self, text: str) -> IntentResult:
        """
        Detect intent from a single utterance.
        
        Args:
            text: Patient statement text
            
        Returns:
            IntentResult with primary intent, secondary intents, and evidence
        """
        if not text or not text.strip():
            return IntentResult(
                primary_intent=PatientIntent.ACKNOWLEDGING.value,
                secondary_intents=[],
                confidence=0.0,
                evidence=["Empty text"]
            )
        
        # Score all intents
        scores = {}
        all_evidence = {}
        
        for intent in PatientIntent:
            score, evidence = self._score_intent(text, intent)
            scores[intent] = score
            all_evidence[intent] = evidence
        
        # Sort intents by score
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get primary and secondary intents
        primary = sorted_intents[0][0]
        primary_score = sorted_intents[0][1]
        
        # Secondary intents have score > 0 and are not primary
        secondary = [
            intent.value for intent, score in sorted_intents[1:]
            if score > 0 and score >= primary_score * 0.5
        ]
        
        # Calculate confidence
        total_score = sum(scores.values()) + 0.01
        confidence = primary_score / total_score if primary_score > 0 else 0.1
        
        # If no strong signal, default to providing information
        if primary_score < 0.5:
            primary = PatientIntent.PROVIDING_INFORMATION
            confidence = 0.3
        
        return IntentResult(
            primary_intent=primary.value,
            secondary_intents=secondary[:2],
            confidence=min(confidence, 1.0),
            evidence=all_evidence[primary][:3]
        )
    
    def detect_from_transcript(self, text: str) -> Dict[str, IntentResult]:
        """
        Detect intents for all patient statements in a transcript.
        
        Args:
            text: Full transcript text
            
        Returns:
            Dictionary mapping statements to their intent results
        """
        results = {}
        
        # Extract patient statements
        lines = text.split('\n')
        patient_statements = []
        current_statement = []
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('patient:'):
                if current_statement:
                    patient_statements.append(' '.join(current_statement))
                statement = line.split(':', 1)[1].strip() if ':' in line else ''
                current_statement = [statement] if statement else []
            elif current_statement and not line.lower().startswith(('physician:', 'doctor:')):
                current_statement.append(line)
        
        if current_statement:
            patient_statements.append(' '.join(current_statement))
        
        # Analyze each statement
        for statement in patient_statements:
            if statement.strip():
                key = statement[:50] + '...' if len(statement) > 50 else statement
                results[key] = self.detect(statement)
        
        return results
    
    def get_intent_summary(self, text: str) -> Dict:
        """
        Get summary of intents across all patient statements.
        
        Args:
            text: Full transcript text
            
        Returns:
            Summary dictionary with intent distribution
        """
        results = self.detect_from_transcript(text)
        
        # Count intent occurrences
        intent_counts = {}
        for result in results.values():
            intent = result.primary_intent
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Sort by frequency
        sorted_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'primary_intents': [intent for intent, _ in sorted_intents],
            'intent_distribution': dict(sorted_intents),
            'total_statements': len(results)
        }


def main():
    """Test the intent detector with sample text."""
    sample_texts = [
        "I'm a bit worried about my back pain, but I hope it gets better soon.",
        "So, I don't need to worry about this affecting me in the future?",
        "Yes, I went to Moss Bank Accident and Emergency.",
        "My neck and back hurt a lot for four weeks.",
        "Thank you, doctor. I appreciate it.",
        "That's a relief!"
    ]
    
    detector = IntentDetector()
    
    print("Intent Detection Results:")
    print("-" * 50)
    
    for text in sample_texts:
        result = detector.detect(text)
        print(f"Text: {text[:60]}...")
        print(f"  Intent: {result.primary_intent}")
        print(f"  Secondary: {result.secondary_intents}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Evidence: {result.evidence}")
        print()


if __name__ == "__main__":
    main()
