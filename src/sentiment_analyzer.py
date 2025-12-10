"""
Sentiment Analyzer for Medical Transcripts

Classifies patient sentiment as:
- Anxious
- Neutral
- Reassured

Uses transformer-based models or rule-based fallback.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Sentiment(Enum):
    """Sentiment categories for patient statements."""
    ANXIOUS = "Anxious"
    NEUTRAL = "Neutral"
    REASSURED = "Reassured"


@dataclass
class SentimentResult:
    """Container for sentiment analysis result."""
    sentiment: str
    confidence: float
    indicators: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "Sentiment": self.sentiment,
            "Confidence": round(self.confidence, 3),
            "Indicators": self.indicators
        }


class SentimentAnalyzer:
    """
    Sentiment analyzer for patient statements in medical transcripts.
    
    Uses:
    - Transformer models (DistilBERT) when available
    - Rule-based lexicon matching as fallback
    """
    
    # Lexicons for rule-based analysis
    ANXIOUS_INDICATORS = [
        # Worry expressions
        'worried', 'worry', 'worrying', 'concern', 'concerned', 'concerning',
        'afraid', 'fear', 'scared', 'nervous', 'anxious', 'anxiety',
        'hope', 'hoping', 'hopefully',  # Often indicates underlying concern
        
        # Uncertainty
        'not sure', 'uncertain', 'wondering', 'what if',
        
        # Negative expectations
        'might get worse', 'could be serious', 'something wrong',
        
        # Pain/symptom focus
        'really bad', 'terrible', 'awful', 'unbearable', 'severe',
        'a lot of pain', 'hurts so much', 'can\'t take',
        
        # Questions about future
        'will this', 'is this going to', 'affect me in the future',
    ]
    
    REASSURED_INDICATORS = [
        # Relief expressions
        'relief', 'relieved', 'glad', 'happy', 'pleased',
        'thank', 'thanks', 'appreciate', 'grateful',
        
        # Positive acknowledgment
        'great to hear', 'good to know', 'that\'s good', 'that\'s great',
        'wonderful', 'excellent', 'fantastic',
        
        # Improvement acknowledgment
        'doing better', 'feeling better', 'improving', 'improved',
        'getting better', 'much better', 'a lot better',
        
        # Acceptance
        'understand', 'makes sense', 'okay', 'i see',
        'don\'t worry', 'not worried', 'no concerns',
    ]
    
    NEUTRAL_INDICATORS = [
        # Factual statements
        'yes', 'no', 'okay', 'alright',
        
        # Simple descriptions
        'i had', 'i went', 'i took', 'i did',
        'it was', 'it is', 'there was',
        
        # Neutral acknowledgment
        'i see', 'understood', 'got it',
    ]
    
    # Sentiment modifiers
    NEGATION_WORDS = ['not', 'no', 'never', 'don\'t', 'doesn\'t', 'didn\'t', 'won\'t', 'can\'t', 'haven\'t']
    INTENSIFIERS = ['very', 'really', 'extremely', 'so', 'quite', 'absolutely']
    
    def __init__(self, use_transformers: bool = True):
        """
        Initialize the sentiment analyzer.
        
        Args:
            use_transformers: Whether to use transformer models
        """
        self.use_transformers = use_transformers
        self.transformer_pipeline = None
        
        if use_transformers:
            try:
                from transformers import pipeline
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1
                )
            except Exception as e:
                print(f"Warning: Could not load transformer model: {e}")
                print("Falling back to rule-based sentiment analysis.")
                self.use_transformers = False
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\'\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _count_indicators(self, text: str, indicators: List[str]) -> Tuple[int, List[str]]:
        """Count how many indicators are present in text."""
        text_lower = text.lower()
        found = []
        
        for indicator in indicators:
            if indicator in text_lower:
                found.append(indicator)
        
        return len(found), found
    
    def _check_negation(self, text: str, indicator: str) -> bool:
        """Check if an indicator is negated."""
        text_lower = text.lower()
        indicator_pos = text_lower.find(indicator)
        
        if indicator_pos == -1:
            return False
        
        # Check for negation words before the indicator
        words_before = text_lower[:indicator_pos].split()
        for neg in self.NEGATION_WORDS:
            if neg in words_before[-3:]:  # Check last 3 words before indicator
                return True
        
        return False
    
    def _rule_based_analysis(self, text: str) -> SentimentResult:
        """Perform rule-based sentiment analysis."""
        text_processed = self._preprocess_text(text)
        
        # Count indicators for each sentiment
        anxious_count, anxious_found = self._count_indicators(text_processed, self.ANXIOUS_INDICATORS)
        reassured_count, reassured_found = self._count_indicators(text_processed, self.REASSURED_INDICATORS)
        neutral_count, neutral_found = self._count_indicators(text_processed, self.NEUTRAL_INDICATORS)
        
        # Check for negations that might flip sentiment
        for indicator in anxious_found[:]:
            if self._check_negation(text_processed, indicator):
                anxious_count -= 1
                anxious_found.remove(indicator)
                reassured_count += 0.5
        
        # Check for intensifiers
        has_intensifier = any(word in text_processed for word in self.INTENSIFIERS)
        if has_intensifier:
            if anxious_count > 0:
                anxious_count *= 1.5
            if reassured_count > 0:
                reassured_count *= 1.5
        
        # Determine sentiment
        scores = {
            Sentiment.ANXIOUS: anxious_count,
            Sentiment.NEUTRAL: neutral_count + 0.5,  # Slight bias toward neutral
            Sentiment.REASSURED: reassured_count
        }
        
        max_sentiment = max(scores, key=scores.get)
        max_score = scores[max_sentiment]
        total_score = sum(scores.values()) + 0.01  # Avoid division by zero
        
        confidence = max_score / total_score
        
        # Collect indicators
        if max_sentiment == Sentiment.ANXIOUS:
            indicators = anxious_found
        elif max_sentiment == Sentiment.REASSURED:
            indicators = reassured_found
        else:
            indicators = neutral_found if neutral_found else ["No strong indicators"]
        
        return SentimentResult(
            sentiment=max_sentiment.value,
            confidence=min(confidence, 1.0),
            indicators=indicators[:5]  # Limit to top 5
        )
    
    def _transformer_analysis(self, text: str) -> SentimentResult:
        """Perform transformer-based sentiment analysis."""
        if not self.transformer_pipeline:
            return self._rule_based_analysis(text)
        
        try:
            result = self.transformer_pipeline(text[:512])[0]  # Truncate for model limit
            
            # Map SST-2 labels to our categories
            label = result['label']
            score = result['score']
            
            # Check for anxiety-specific patterns
            text_lower = text.lower()
            has_anxiety_words = any(word in text_lower for word in ['worried', 'concerned', 'hope', 'afraid'])
            has_relief_words = any(word in text_lower for word in ['relief', 'thank', 'great', 'glad'])
            
            if label == 'NEGATIVE' or has_anxiety_words:
                sentiment = Sentiment.ANXIOUS.value
            elif label == 'POSITIVE' or has_relief_words:
                if score > 0.8 and has_relief_words:
                    sentiment = Sentiment.REASSURED.value
                else:
                    sentiment = Sentiment.NEUTRAL.value
            else:
                sentiment = Sentiment.NEUTRAL.value
            
            # Get rule-based indicators for explanation
            _, anxious_ind = self._count_indicators(text.lower(), self.ANXIOUS_INDICATORS)
            _, reassured_ind = self._count_indicators(text.lower(), self.REASSURED_INDICATORS)
            
            if sentiment == Sentiment.ANXIOUS.value:
                indicators = anxious_ind[:3] if anxious_ind else [label.lower()]
            elif sentiment == Sentiment.REASSURED.value:
                indicators = reassured_ind[:3] if reassured_ind else [label.lower()]
            else:
                indicators = ["neutral tone"]
            
            return SentimentResult(
                sentiment=sentiment,
                confidence=score,
                indicators=indicators
            )
            
        except Exception as e:
            print(f"Transformer analysis failed: {e}")
            return self._rule_based_analysis(text)
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of text.
        
        Args:
            text: The text to analyze (typically patient statement)
            
        Returns:
            SentimentResult with sentiment, confidence, and indicators
        """
        if not text or not text.strip():
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL.value,
                confidence=0.0,
                indicators=["Empty text"]
            )
        
        if self.use_transformers and self.transformer_pipeline:
            return self._transformer_analysis(text)
        else:
            return self._rule_based_analysis(text)
    
    def analyze_transcript(self, text: str) -> Dict[str, SentimentResult]:
        """
        Analyze sentiment of all patient statements in a transcript.
        
        Args:
            text: Full transcript text
            
        Returns:
            Dictionary mapping statement excerpts to their sentiment
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
                results[key] = self.analyze(statement)
        
        return results
    
    def get_overall_sentiment(self, text: str) -> SentimentResult:
        """
        Get overall sentiment across all patient statements.
        
        Args:
            text: Full transcript text
            
        Returns:
            Aggregated sentiment result
        """
        results = self.analyze_transcript(text)
        
        if not results:
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL.value,
                confidence=0.0,
                indicators=["No patient statements found"]
            )
        
        # Count sentiments
        sentiment_counts = {s.value: 0 for s in Sentiment}
        all_indicators = []
        total_confidence = 0.0
        
        for result in results.values():
            sentiment_counts[result.sentiment] += 1
            all_indicators.extend(result.indicators)
            total_confidence += result.confidence
        
        # Determine overall sentiment
        max_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        avg_confidence = total_confidence / len(results)
        
        return SentimentResult(
            sentiment=max_sentiment,
            confidence=avg_confidence,
            indicators=list(set(all_indicators))[:5]
        )


def main():
    """Test the sentiment analyzer with sample text."""
    sample_texts = [
        "I'm a bit worried about my back pain, but I hope it gets better soon.",
        "That's a relief! I'm so glad to hear that.",
        "Yes, I had ten physiotherapy sessions.",
        "I'm really concerned this might affect my work in the future.",
        "Thank you, doctor. I appreciate it."
    ]
    
    analyzer = SentimentAnalyzer(use_transformers=False)
    
    print("Sentiment Analysis Results:")
    print("-" * 50)
    
    for text in sample_texts:
        result = analyzer.analyze(text)
        print(f"Text: {text[:60]}...")
        print(f"  Sentiment: {result.sentiment}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Indicators: {result.indicators}")
        print()


if __name__ == "__main__":
    main()
