"""
NLP utilities for text processing and entity extraction.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Some features may be limited.")

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Some features may be limited.")

from .config import get_config


@dataclass
class Entity:
    """Represents a named entity with metadata."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence
        }


class NLPProcessor:
    """Main NLP processor class using spaCy as primary backend."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize NLP processor.
        
        Args:
            model_name: spaCy model name to use.
        """
        self.config = get_config("model")
        self.model_name = model_name or self.config.spacy_model
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """Load the spaCy model."""
        if not SPACY_AVAILABLE:
            logging.error("spaCy not available. Please install spaCy.")
            return
        
        try:
            self.nlp = spacy.load(self.model_name)
            logging.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logging.warning(f"Model {self.model_name} not found. Falling back to basic model.")
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logging.info("Loaded fallback model: en_core_web_sm")
            except OSError:
                logging.error("No spaCy models available. Please install a model.")
    
    def is_available(self) -> bool:
        """Check if NLP processor is ready to use."""
        return self.nlp is not None


def extract_entities(text: str, processor: NLPProcessor = None) -> List[Entity]:
    """
    Extract named entities from text using spaCy.
    
    Args:
        text: Input text to process.
        processor: Optional NLP processor instance.
    
    Returns:
        List of Entity objects with metadata.
    """
    if processor is None:
        processor = NLPProcessor()
    
    if not processor.is_available():
        return _extract_entities_nltk_fallback(text)
    
    # Process text with spaCy
    doc = processor.nlp(text)
    
    entities = []
    config = get_config("model")
    
    for ent in doc.ents:
        # Filter by configured entity types
        if ent.label_ in config.entity_types:
            entity = Entity(
                text=ent.text.strip(),
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0  # spaCy doesn't provide confidence scores by default
            )
            entities.append(entity)
    
    return entities


def _extract_entities_nltk_fallback(text: str) -> List[Entity]:
    """
    Fallback entity extraction using NLTK.
    
    Args:
        text: Input text to process.
    
    Returns:
        List of Entity objects.
    """
    if not NLTK_AVAILABLE:
        logging.warning("Neither spaCy nor NLTK available for entity extraction.")
        return []
    
    try:
        # Download required NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
        
        try:
            nltk.data.find('chunkers/maxent_ne_chunker')
        except LookupError:
            nltk.download('maxent_ne_chunker', quiet=True)
        
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words', quiet=True)
        
        # Tokenize and tag
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        
        # Named entity chunking
        entities = []
        chunks = ne_chunk(tagged)
        
        current_pos = 0
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                # This is a named entity
                entity_text = ' '.join([token for token, pos in chunk])
                start_pos = text.find(entity_text, current_pos)
                if start_pos != -1:
                    entity = Entity(
                        text=entity_text,
                        label=chunk.label(),
                        start=start_pos,
                        end=start_pos + len(entity_text),
                        confidence=0.8  # Lower confidence for NLTK
                    )
                    entities.append(entity)
                    current_pos = start_pos + len(entity_text)
        
        return entities
    
    except Exception as e:
        logging.error(f"Error in NLTK entity extraction: {e}")
        return []


def split_sentences(text: str, processor: NLPProcessor = None) -> List[str]:
    """
    Split text into sentences using spaCy or NLTK.
    
    Args:
        text: Input text to split.
        processor: Optional NLP processor instance.
    
    Returns:
        List of sentence strings.
    """
    if processor is None:
        processor = NLPProcessor()
    
    config = get_config("model")
    
    if processor.is_available():
        # Use spaCy for sentence splitting
        doc = processor.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
    else:
        # Fallback to NLTK
        sentences = _split_sentences_nltk(text)
    
    # Filter sentences by length
    filtered_sentences = []
    for sentence in sentences:
        if (config.min_sentence_length <= len(sentence) <= config.max_sentence_length 
            and sentence.strip()):
            filtered_sentences.append(sentence.strip())
    
    return filtered_sentences


def _split_sentences_nltk(text: str) -> List[str]:
    """
    Fallback sentence splitting using NLTK.
    
    Args:
        text: Input text to split.
    
    Returns:
        List of sentence strings.
    """
    if not NLTK_AVAILABLE:
        # Basic sentence splitting using regex
        return _split_sentences_regex(text)
    
    try:
        # Ensure punkt tokenizer is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    
    except Exception as e:
        logging.error(f"Error in NLTK sentence splitting: {e}")
        return _split_sentences_regex(text)


def _split_sentences_regex(text: str) -> List[str]:
    """
    Basic sentence splitting using regex patterns.
    
    Args:
        text: Input text to split.
    
    Returns:
        List of sentence strings.
    """
    # Simple regex-based sentence splitting
    sentence_endings = r'[.!?]+(?:\s+|$)'
    sentences = re.split(sentence_endings, text)
    
    # Clean up and filter empty sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 3:  # Minimum viable sentence length
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def preprocess_text(text: str) -> str:
    """
    Preprocess text for better NLP processing.
    
    Args:
        text: Raw input text.
    
    Returns:
        Preprocessed text.
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Fix common encoding issues
    replacements = {
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '–': '-',
        '—': '-',
        '…': '...',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text.strip()


def extract_key_phrases(text: str, processor: NLPProcessor = None, max_phrases: int = 10) -> List[str]:
    """
    Extract key phrases from text.
    
    Args:
        text: Input text to analyze.
        processor: Optional NLP processor instance.
        max_phrases: Maximum number of phrases to return.
    
    Returns:
        List of key phrases.
    """
    if processor is None:
        processor = NLPProcessor()
    
    if not processor.is_available():
        return []
    
    doc = processor.nlp(text)
    
    # Extract noun phrases
    noun_phrases = []
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        if len(phrase.split()) >= 2 and len(phrase) > 5:  # Multi-word phrases only
            noun_phrases.append(phrase)
    
    # Remove duplicates and sort by frequency/importance
    unique_phrases = list(set(noun_phrases))
    
    # Simple scoring based on length and position
    scored_phrases = []
    for phrase in unique_phrases:
        score = len(phrase.split()) * text.lower().count(phrase)
        scored_phrases.append((phrase, score))
    
    # Sort by score and return top phrases
    scored_phrases.sort(key=lambda x: x[1], reverse=True)
    return [phrase for phrase, _ in scored_phrases[:max_phrases]]


# Convenience functions for easy access
def get_nlp_processor() -> NLPProcessor:
    """Get a configured NLP processor instance."""
    return NLPProcessor()


def quick_entity_extraction(text: str) -> List[Dict[str, Any]]:
    """
    Quick entity extraction with simple output format.
    
    Args:
        text: Input text.
    
    Returns:
        List of entity dictionaries.
    """
    entities = extract_entities(text)
    return [entity.to_dict() for entity in entities] 