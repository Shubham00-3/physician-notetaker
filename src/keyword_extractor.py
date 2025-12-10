"""
Medical Keyword Extractor

Extracts important medical phrases and keywords from transcripts using:
- YAKE (Yet Another Keyword Extractor)
- TF-IDF based extraction
- Medical vocabulary filtering
"""

import re
from typing import List, Tuple, Dict
from collections import Counter


class MedicalKeywordExtractor:
    """
    Extract medical keywords and phrases from clinical transcripts.
    
    Uses multiple extraction methods:
    - Statistical keyword extraction (YAKE/TF-IDF)
    - Medical vocabulary matching
    - N-gram extraction
    """
    
    # Medical vocabulary for filtering and boosting
    MEDICAL_TERMS = {
        # Symptoms
        'pain', 'ache', 'discomfort', 'stiffness', 'swelling', 'numbness',
        'tingling', 'weakness', 'fatigue', 'nausea', 'dizziness', 'headache',
        
        # Body parts
        'neck', 'back', 'spine', 'cervical', 'lumbar', 'thoracic', 'shoulder',
        'head', 'muscle', 'joint', 'vertebra', 'disc',
        
        # Conditions/Diagnosis
        'whiplash', 'injury', 'strain', 'sprain', 'fracture', 'concussion',
        'inflammation', 'trauma', 'accident', 'impact',
        
        # Treatments
        'physiotherapy', 'therapy', 'physical therapy', 'medication', 'painkillers',
        'treatment', 'surgery', 'injection', 'exercise', 'rehabilitation',
        
        # Medical procedures
        'x-ray', 'xray', 'mri', 'scan', 'examination', 'assessment',
        
        # Prognosis terms
        'recovery', 'prognosis', 'improvement', 'healing', 'chronic', 'acute',
        
        # Time-related
        'weeks', 'months', 'sessions', 'follow-up', 'appointment',
    }
    
    # Multi-word medical phrases
    MEDICAL_PHRASES = [
        'whiplash injury',
        'physical examination',
        'full recovery',
        'long-term damage',
        'range of motion',
        'range of movement',
        'back pain',
        'neck pain',
        'lower back',
        'upper back',
        'car accident',
        'physiotherapy sessions',
        'physical therapy',
        'full range',
        'no tenderness',
        'emergency room',
        'accident and emergency',
        'muscle strain',
        'soft tissue',
        'pain relief',
        'six months',
        'four weeks',
    ]
    
    def __init__(self, use_yake: bool = True):
        """
        Initialize the keyword extractor.
        
        Args:
            use_yake: Whether to use YAKE for extraction
        """
        self.use_yake = use_yake
        self.yake_extractor = None
        
        if use_yake:
            try:
                import yake
                self.yake_extractor = yake.KeywordExtractor(
                    lan="en",
                    n=3,  # max n-gram size
                    dedupLim=0.7,
                    dedupFunc='seqm',
                    windowsSize=2,
                    top=20
                )
            except ImportError:
                print("Warning: YAKE not installed. Using fallback extraction.")
                self.use_yake = False
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove speaker labels
        text = re.sub(r'^(Physician|Doctor|Patient):\s*', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep hyphens for medical terms
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip().lower()
    
    def _extract_ngrams(self, text: str, n: int) -> List[str]:
        """Extract n-grams from text."""
        words = text.split()
        ngrams = []
        
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def _score_medical_relevance(self, keyword: str) -> float:
        """Score a keyword based on medical relevance."""
        keyword_lower = keyword.lower()
        score = 0.0
        
        # Check if it's a known medical phrase
        if keyword_lower in [p.lower() for p in self.MEDICAL_PHRASES]:
            score += 2.0
        
        # Check if it contains medical terms
        words = keyword_lower.split()
        for word in words:
            if word in self.MEDICAL_TERMS:
                score += 1.0
        
        return score
    
    def extract_keywords_yake(self, text: str) -> List[Tuple[str, float]]:
        """Extract keywords using YAKE."""
        if not self.yake_extractor:
            return []
        
        cleaned_text = self._preprocess_text(text)
        keywords = self.yake_extractor.extract_keywords(cleaned_text)
        
        # Add medical relevance scoring
        scored_keywords = []
        for kw, score in keywords:
            medical_score = self._score_medical_relevance(kw)
            # Lower YAKE score is better, so we invert and add medical relevance
            combined_score = (1 / (score + 0.01)) + medical_score
            scored_keywords.append((kw, combined_score))
        
        # Sort by combined score (descending)
        scored_keywords.sort(key=lambda x: x[1], reverse=True)
        
        return scored_keywords
    
    def extract_keywords_tfidf(self, text: str, top_n: int = 15) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF-like scoring."""
        cleaned_text = self._preprocess_text(text)
        
        # Extract 1-grams, 2-grams, and 3-grams
        all_ngrams = []
        for n in range(1, 4):
            all_ngrams.extend(self._extract_ngrams(cleaned_text, n))
        
        # Count frequencies
        ngram_counts = Counter(all_ngrams)
        
        # Score ngrams
        scored = []
        for ngram, count in ngram_counts.items():
            # Base score from frequency
            base_score = count
            
            # Add medical relevance
            medical_score = self._score_medical_relevance(ngram)
            
            # Boost multi-word phrases
            length_bonus = len(ngram.split()) * 0.5
            
            total_score = base_score + medical_score + length_bonus
            
            # Filter out very short or common words
            if len(ngram) > 3 and medical_score > 0:
                scored.append((ngram, total_score))
        
        # Sort and return top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]
    
    def extract_medical_phrases(self, text: str) -> List[str]:
        """Extract known medical phrases from text."""
        text_lower = text.lower()
        found_phrases = []
        
        for phrase in self.MEDICAL_PHRASES:
            if phrase.lower() in text_lower:
                found_phrases.append(phrase.title())
        
        return found_phrases
    
    def extract_keywords(self, text: str, top_n: int = 15) -> Dict:
        """
        Extract all keywords from text.
        
        Args:
            text: The transcript text
            top_n: Number of top keywords to return
            
        Returns:
            Dictionary with different keyword categories
        """
        result = {
            'medical_phrases': self.extract_medical_phrases(text),
            'keywords': [],
            'scores': {}
        }
        
        # Use YAKE if available
        if self.use_yake and self.yake_extractor:
            yake_keywords = self.extract_keywords_yake(text)
            keywords = [(kw, score) for kw, score in yake_keywords[:top_n]]
        else:
            keywords = self.extract_keywords_tfidf(text, top_n)
        
        result['keywords'] = [kw for kw, _ in keywords]
        result['scores'] = {kw: round(score, 3) for kw, score in keywords}
        
        # Combine and deduplicate
        all_keywords = set(result['medical_phrases'])
        all_keywords.update(result['keywords'])
        result['all_keywords'] = list(all_keywords)
        
        return result


def main():
    """Test the keyword extractor with sample text."""
    sample_text = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
    Physician: I understand you were in a car accident last September.
    Patient: Yes, it was on September 1st. Out of nowhere, another car hit me from behind.
    Physician: What did you feel immediately after the accident?
    Patient: I had hit my head on the steering wheel, and I could feel pain in my neck and back.
    Physician: Did you seek medical attention?
    Patient: Yes, I went to Accident and Emergency. They said it was a whiplash injury.
    Patient: I had to go through ten sessions of physiotherapy.
    Physician: Your neck and back have a full range of movement.
    Physician: I'd expect you to make a full recovery within six months.
    """
    
    extractor = MedicalKeywordExtractor(use_yake=False)
    keywords = extractor.extract_keywords(sample_text)
    
    print("Extracted Keywords:")
    print("-" * 40)
    print(f"Medical Phrases: {keywords['medical_phrases']}")
    print(f"Top Keywords: {keywords['keywords'][:10]}")
    print(f"All Keywords: {keywords['all_keywords']}")


if __name__ == "__main__":
    main()
