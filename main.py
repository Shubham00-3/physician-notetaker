"""
Physician Notetaker - Main Pipeline

A comprehensive NLP pipeline for medical transcription analysis:
- Named Entity Recognition (NER)
- Text Summarization
- Keyword Extraction
- Sentiment Analysis
- Intent Detection
- SOAP Note Generation
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.ner_extractor import MedicalNERExtractor
from src.summarizer import MedicalSummarizer
from src.keyword_extractor import MedicalKeywordExtractor
from src.sentiment_analyzer import SentimentAnalyzer
from src.intent_detector import IntentDetector
from src.soap_generator import SOAPNoteGenerator


class PhysicianNotetaker:
    """
    Main pipeline for medical transcript analysis.
    
    Combines all NLP components:
    - NER for entity extraction
    - Summarization for structured reports
    - Keyword extraction for important phrases
    - Sentiment analysis for patient emotional state
    - Intent detection for understanding patient goals
    - SOAP note generation for clinical documentation
    """
    
    def __init__(self, use_transformers: bool = False, use_spacy: bool = False):
        """
        Initialize the pipeline.
        
        Args:
            use_transformers: Whether to use transformer models (requires PyTorch)
            use_spacy: Whether to use spaCy models (requires spaCy installation)
        """
        print("Initializing Physician Notetaker Pipeline...")
        
        self.ner_extractor = MedicalNERExtractor(use_spacy=use_spacy)
        self.summarizer = MedicalSummarizer(use_transformers=use_transformers)
        self.keyword_extractor = MedicalKeywordExtractor(use_yake=False)  # YAKE optional
        self.sentiment_analyzer = SentimentAnalyzer(use_transformers=use_transformers)
        self.intent_detector = IntentDetector()
        self.soap_generator = SOAPNoteGenerator()
        
        print("Pipeline initialized successfully!")
    
    def analyze(self, transcript: str) -> Dict:
        """
        Perform complete analysis on a medical transcript.
        
        Args:
            transcript: The raw transcript text
            
        Returns:
            Dictionary with all analysis results
        """
        results = {}
        
        # 1. Named Entity Recognition
        print("\n[1/6] Extracting medical entities...")
        entities = self.ner_extractor.extract_entities(transcript)
        results['medical_entities'] = entities.to_dict()
        
        # 2. Text Summarization
        print("[2/6] Generating structured summary...")
        summary = self.summarizer.summarize(transcript, entities.to_dict())
        results['structured_summary'] = summary.to_dict()
        
        # 3. Keyword Extraction
        print("[3/6] Extracting medical keywords...")
        keywords = self.keyword_extractor.extract_keywords(transcript)
        results['keywords'] = {
            'medical_phrases': keywords['medical_phrases'],
            'top_keywords': keywords['keywords'][:10]
        }
        
        # 4. Sentiment Analysis
        print("[4/6] Analyzing patient sentiment...")
        sentiment = self.sentiment_analyzer.get_overall_sentiment(transcript)
        results['sentiment_analysis'] = sentiment.to_dict()
        
        # 5. Intent Detection
        print("[5/6] Detecting patient intents...")
        intent_summary = self.intent_detector.get_intent_summary(transcript)
        
        # Get a representative sample of intents
        sample_results = self.intent_detector.detect_from_transcript(transcript)
        sample_intents = []
        for text, result in list(sample_results.items())[:3]:
            sample_intents.append({
                'statement': text,
                'intent': result.primary_intent,
                'confidence': round(result.confidence, 2)
            })
        
        results['intent_analysis'] = {
            'primary_intents': intent_summary['primary_intents'][:3],
            'sample_detections': sample_intents
        }
        
        # 6. SOAP Note Generation
        print("[6/6] Generating SOAP note...")
        soap_note = self.soap_generator.generate(transcript)
        results['soap_note'] = soap_note.to_dict()
        
        return results
    
    def analyze_sentiment_only(self, text: str) -> Dict:
        """
        Analyze just the sentiment and intent of a text.
        
        Args:
            text: Patient statement text
            
        Returns:
            Dictionary with sentiment and intent
        """
        sentiment = self.sentiment_analyzer.analyze(text)
        intent = self.intent_detector.detect(text)
        
        return {
            'Sentiment': sentiment.sentiment,
            'Intent': intent.primary_intent,
            'Sentiment_Confidence': round(sentiment.confidence, 2),
            'Intent_Confidence': round(intent.confidence, 2)
        }
    
    def generate_summary(self, transcript: str) -> Dict:
        """
        Generate just the structured summary.
        
        Args:
            transcript: The medical transcript
            
        Returns:
            Structured summary dictionary
        """
        entities = self.ner_extractor.extract_entities(transcript)
        return entities.to_dict()
    
    def generate_soap(self, transcript: str) -> Dict:
        """
        Generate just the SOAP note.
        
        Args:
            transcript: The medical transcript
            
        Returns:
            SOAP note dictionary
        """
        return self.soap_generator.generate(transcript).to_dict()


def load_transcript(filepath: str) -> str:
    """Load transcript from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def print_section(title: str, data: Dict, indent: int = 2):
    """Pretty print a section of results."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)
    print(json.dumps(data, indent=indent))


def main():
    """Run the complete analysis pipeline."""
    # Get transcript path
    script_dir = Path(__file__).parent
    default_transcript = script_dir / "sample_transcript.txt"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        transcript_path = sys.argv[1]
    else:
        transcript_path = str(default_transcript)
    
    # Load transcript
    print(f"Loading transcript from: {transcript_path}")
    
    if not os.path.exists(transcript_path):
        print(f"Error: Transcript file not found: {transcript_path}")
        print("\nUsing sample transcript for demonstration...")
        
        transcript = """
Physician: Good morning, Ms. Jones. How are you feeling today?

Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.

Physician: I understand you were in a car accident last September. Can you walk me through what happened?

Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.

Physician: That sounds like a strong impact. Were you wearing your seatbelt?

Patient: Yes, I always do.

Physician: What did you feel immediately after the accident?

Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.

Physician: Did you seek medical attention at that time?

Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn't do any X-rays. They just gave me some advice and sent me home.

Physician: How did things progress after that?

Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.

Physician: That makes sense. Are you still experiencing pain now?

Patient: It's not constant, but I do get occasional backaches. It's nothing like before, though.

Physician: That's good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?

Patient: No, nothing like that. I don't feel nervous driving, and I haven't had any emotional issues from the accident.

Physician: And how has this impacted your daily life? Work, hobbies, anything like that?

Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn't really stopped me from doing anything.

Physician: That's encouraging. Let's go ahead and do a physical examination to check your mobility and any lingering pain.

[Physical Examination Conducted]

Physician: Everything looks good. Your neck and back have a full range of movement, and there's no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.

Patient: That's a relief!

Physician: Yes, your recovery so far has been quite positive. Given your progress, I'd expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.

Patient: That's great to hear. So, I don't need to worry about this affecting me in the future?

Physician: That's right. I don't foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you're on track for a full recovery.

Patient: Thank you, doctor. I appreciate it.

Physician: You're very welcome, Ms. Jones. Take care, and don't hesitate to reach out if you need anything.
"""
    else:
        transcript = load_transcript(transcript_path)
    
    # Initialize pipeline (without transformer/spacy for faster demo)
    pipeline = PhysicianNotetaker(use_transformers=False, use_spacy=False)
    
    # Run analysis
    print("\n" + "=" * 60)
    print(" RUNNING COMPLETE ANALYSIS")
    print("=" * 60)
    
    results = pipeline.analyze(transcript)
    
    # Print results
    print_section("1. MEDICAL ENTITIES (NER)", results['medical_entities'])
    print_section("2. STRUCTURED SUMMARY", results['structured_summary'])
    print_section("3. KEYWORDS", results['keywords'])
    print_section("4. SENTIMENT ANALYSIS", results['sentiment_analysis'])
    print_section("5. INTENT DETECTION", results['intent_analysis'])
    print_section("6. SOAP NOTE", results['soap_note'])
    
    # Save results to file
    output_path = script_dir / "analysis_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")
    
    # Demo: Single statement sentiment/intent analysis
    print("\n" + "=" * 60)
    print(" DEMO: SINGLE STATEMENT ANALYSIS")
    print("=" * 60)
    
    sample_statement = "I'm a bit worried about my back pain, but I hope it gets better soon."
    print(f"\nInput: \"{sample_statement}\"")
    
    single_result = pipeline.analyze_sentiment_only(sample_statement)
    print(f"\nOutput:")
    print(json.dumps(single_result, indent=2))
    
    print("\n" + "=" * 60)
    print(" ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
