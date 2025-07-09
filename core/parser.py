"""
Event parsing module for extracting structured triples from text.
"""

import re
import logging
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from .nlp_utils import NLPProcessor, Entity, extract_entities
from .config import get_config


class TripleType(Enum):
    """Types of semantic triples."""
    ENTITY_RELATION = "entity_relation"
    EVENT_PARTICIPATION = "event_participation" 
    TEMPORAL_RELATION = "temporal_relation"
    CAUSAL_RELATION = "causal_relation"
    SPATIAL_RELATION = "spatial_relation"


@dataclass
class Triple:
    """Represents a semantic triple (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    triple_type: TripleType = TripleType.ENTITY_RELATION
    source_sentence: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert triple to dictionary representation."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "type": self.triple_type.value,
            "source_sentence": self.source_sentence,
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.object})"


class EventParser:
    """Parser for extracting structured triples from sentences."""
    
    def __init__(self, nlp_processor: NLPProcessor = None):
        """
        Initialize event parser.
        
        Args:
            nlp_processor: Optional NLP processor instance.
        """
        self.nlp_processor = nlp_processor or NLPProcessor()
        self.config = get_config("model")
        
        # Predefined relation patterns
        self._relation_patterns = self._initialize_relation_patterns()
        
        # Common verbs that indicate relationships
        self._relation_verbs = {
            "is", "was", "are", "were", "becomes", "became", "remains", "stayed",
            "works", "worked", "lives", "lived", "leads", "led", "manages", "managed",
            "owns", "owned", "founded", "created", "established", "built", "developed",
            "announced", "said", "stated", "reported", "claimed", "revealed",
            "causes", "caused", "results", "resulted", "leads", "led", "influences",
            "affects", "impacts", "triggers", "triggered", "follows", "followed",
            "happens", "happened", "occurs", "occurred", "takes", "took", "gives", "gave"
        }
    
    def _initialize_relation_patterns(self) -> List[Dict[str, Any]]:
        """Initialize regex patterns for relation extraction."""
        patterns = [
            # Simple copula patterns (X is Y)
            {
                "pattern": r"(.+?)\s+(is|was|are|were)\s+(.+)",
                "type": TripleType.ENTITY_RELATION,
                "predicate": "IS_A"
            },
            
            # Verb-based patterns (X verbs Y)
            {
                "pattern": r"(.+?)\s+(works for|works at|leads|manages|owns|founded|created)\s+(.+)",
                "type": TripleType.ENTITY_RELATION,
                "predicate": "VERB_RELATION"
            },
            
            # Location patterns (X in/at Y)
            {
                "pattern": r"(.+?)\s+(in|at|from|to)\s+(.+)",
                "type": TripleType.SPATIAL_RELATION,
                "predicate": "LOCATED_IN"
            },
            
            # Temporal patterns (X during/after/before Y)
            {
                "pattern": r"(.+?)\s+(during|after|before|since|until)\s+(.+)",
                "type": TripleType.TEMPORAL_RELATION,
                "predicate": "TEMPORAL_RELATION"
            },
            
            # Causal patterns (X causes/results in Y)
            {
                "pattern": r"(.+?)\s+(causes|results in|leads to|triggers|due to)\s+(.+)",
                "type": TripleType.CAUSAL_RELATION,
                "predicate": "CAUSES"
            },
            
            # Participation patterns (X participates in Y)
            {
                "pattern": r"(.+?)\s+(participates in|joins|attends|takes part in)\s+(.+)",
                "type": TripleType.EVENT_PARTICIPATION,
                "predicate": "PARTICIPATES_IN"
            }
        ]
        
        return patterns
    
    def parse_sentence(self, sentence: str) -> List[Tuple[str, str, str]]:
        """
        Parse a sentence and extract subject-predicate-object triples.
        
        Args:
            sentence: Input sentence to parse.
        
        Returns:
            List of (subject, predicate, object) tuples.
        """
        triples = self.extract_triples(sentence)
        return [(t.subject, t.predicate, t.object) for t in triples]
    
    def extract_triples(self, sentence: str) -> List[Triple]:
        """
        Extract detailed Triple objects from a sentence.
        
        Args:
            sentence: Input sentence to parse.
        
        Returns:
            List of Triple objects with metadata.
        """
        if not sentence or not sentence.strip():
            return []
        
        sentence = sentence.strip()
        triples = []
        
        # Method 1: Pattern-based extraction
        pattern_triples = self._extract_with_patterns(sentence)
        triples.extend(pattern_triples)
        
        # Method 2: Dependency parsing (if spaCy available)
        if self.nlp_processor.is_available():
            dep_triples = self._extract_with_dependencies(sentence)
            triples.extend(dep_triples)
        
        # Method 3: Entity-relation extraction
        entity_triples = self._extract_entity_relations(sentence)
        triples.extend(entity_triples)
        
        # Remove duplicates and filter by confidence
        unique_triples = self._deduplicate_triples(triples)
        filtered_triples = self._filter_by_confidence(unique_triples)
        
        return filtered_triples
    
    def _extract_with_patterns(self, sentence: str) -> List[Triple]:
        """Extract triples using regex patterns."""
        triples = []
        
        for pattern_config in self._relation_patterns:
            pattern = pattern_config["pattern"]
            triple_type = pattern_config["type"]
            predicate = pattern_config["predicate"]
            
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            
            for match in matches:
                try:
                    groups = match.groups()
                    if len(groups) >= 3:
                        subject = self._clean_entity(groups[0])
                        predicate_text = groups[1] if predicate == "VERB_RELATION" else predicate
                        obj = self._clean_entity(groups[2])
                        
                        if subject and obj and predicate_text:
                            triple = Triple(
                                subject=subject,
                                predicate=predicate_text.upper().replace(" ", "_"),
                                object=obj,
                                confidence=0.8,
                                triple_type=triple_type,
                                source_sentence=sentence,
                                metadata={"extraction_method": "pattern", "pattern": pattern}
                            )
                            triples.append(triple)
                except Exception as e:
                    logging.debug(f"Error extracting with pattern {pattern}: {e}")
                    continue
        
        return triples
    
    def _extract_with_dependencies(self, sentence: str) -> List[Triple]:
        """Extract triples using dependency parsing."""
        if not self.nlp_processor.is_available():
            return []
        
        triples = []
        
        try:
            doc = self.nlp_processor.nlp(sentence)
            
            # Find verb-subject-object relationships
            for token in doc:
                if token.pos_ == "VERB" and token.lemma_ in self._relation_verbs:
                    # Find subject
                    subjects = [child for child in token.children if child.dep_ in ["nsubj", "nsubjpass"]]
                    # Find objects
                    objects = [child for child in token.children if child.dep_ in ["dobj", "pobj", "attr", "prep"]]
                    
                    for subject in subjects:
                        for obj in objects:
                            # Expand to include modifiers
                            subject_text = self._expand_entity(subject, doc)
                            object_text = self._expand_entity(obj, doc)
                            predicate_text = token.lemma_.upper()
                            
                            if subject_text and object_text and len(subject_text) > 1 and len(object_text) > 1:
                                triple = Triple(
                                    subject=subject_text,
                                    predicate=predicate_text,
                                    object=object_text,
                                    confidence=0.9,
                                    triple_type=TripleType.ENTITY_RELATION,
                                    source_sentence=sentence,
                                    metadata={"extraction_method": "dependency", "verb_pos": token.i}
                                )
                                triples.append(triple)
        
        except Exception as e:
            logging.debug(f"Error in dependency parsing: {e}")
        
        return triples
    
    def _extract_entity_relations(self, sentence: str) -> List[Triple]:
        """Extract relations between named entities."""
        triples = []
        
        # Extract entities from the sentence
        entities = extract_entities(sentence, self.nlp_processor)
        
        if len(entities) < 2:
            return triples
        
        # Find relationships between entity pairs
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Extract text between entities
                start_pos = min(entity1.end, entity2.end)
                end_pos = max(entity1.start, entity2.start)
                
                if start_pos < end_pos:
                    between_text = sentence[start_pos:end_pos].strip()
                    
                    # Look for relation indicators
                    relation = self._extract_relation_from_text(between_text)
                    
                    if relation:
                        # Determine subject/object order based on position
                        if entity1.start < entity2.start:
                            subject, obj = entity1.text, entity2.text
                        else:
                            subject, obj = entity2.text, entity1.text
                        
                        triple = Triple(
                            subject=subject,
                            predicate=relation,
                            object=obj,
                            confidence=0.7,
                            triple_type=TripleType.ENTITY_RELATION,
                            source_sentence=sentence,
                            metadata={
                                "extraction_method": "entity_relation",
                                "entity1_type": entity1.label,
                                "entity2_type": entity2.label
                            }
                        )
                        triples.append(triple)
        
        return triples
    
    def _extract_relation_from_text(self, text: str) -> Optional[str]:
        """Extract relation from text between entities."""
        if not text:
            return None
        
        # Clean and normalize text
        text = re.sub(r'[^\w\s]', ' ', text.lower()).strip()
        words = text.split()
        
        # Look for known relation verbs
        for word in words:
            if word in self._relation_verbs:
                return word.upper()
        
        # Look for prepositions that indicate relationships
        prepositions = {"in", "at", "of", "for", "with", "by", "from", "to"}
        for word in words:
            if word in prepositions:
                return f"RELATED_BY_{word.upper()}"
        
        # Default generic relation if text exists
        if words:
            return "RELATED_TO"
        
        return None
    
    def _expand_entity(self, token, doc) -> str:
        """Expand entity to include modifiers and compounds."""
        # Start with the token itself
        words = [token.text]
        
        # Add compound modifiers (left)
        for child in token.children:
            if child.dep_ in ["compound", "amod", "det"] and child.i < token.i:
                words.insert(0, child.text)
        
        # Add compound modifiers (right)
        for child in token.children:
            if child.dep_ in ["compound", "amod"] and child.i > token.i:
                words.append(child.text)
        
        return " ".join(words).strip()
    
    def _clean_entity(self, entity: str) -> str:
        """Clean and normalize entity text."""
        if not entity:
            return ""
        
        # Remove extra whitespace and punctuation
        entity = re.sub(r'[^\w\s]', ' ', entity)
        entity = re.sub(r'\s+', ' ', entity).strip()
        
        # Remove articles
        entity = re.sub(r'^(the|a|an)\s+', '', entity, flags=re.IGNORECASE)
        
        return entity.strip()
    
    def _deduplicate_triples(self, triples: List[Triple]) -> List[Triple]:
        """Remove duplicate triples, keeping higher confidence ones."""
        seen = {}
        
        for triple in triples:
            key = (triple.subject.lower(), triple.predicate.lower(), triple.object.lower())
            
            if key not in seen or triple.confidence > seen[key].confidence:
                seen[key] = triple
        
        return list(seen.values())
    
    def _filter_by_confidence(self, triples: List[Triple]) -> List[Triple]:
        """Filter triples by minimum confidence threshold."""
        threshold = self.config.min_triple_score
        return [t for t in triples if t.confidence >= threshold]
    
    def batch_parse(self, sentences: List[str]) -> List[List[Triple]]:
        """
        Parse multiple sentences in batch.
        
        Args:
            sentences: List of sentences to parse.
        
        Returns:
            List of triple lists, one for each sentence.
        """
        results = []
        
        for sentence in sentences:
            try:
                triples = self.extract_triples(sentence)
                results.append(triples)
            except Exception as e:
                logging.error(f"Error parsing sentence '{sentence}': {e}")
                results.append([])
        
        return results
    
    def get_statistics(self, triples: List[Triple]) -> Dict[str, Any]:
        """
        Get statistics about extracted triples.
        
        Args:
            triples: List of triples to analyze.
        
        Returns:
            Dictionary with statistics.
        """
        if not triples:
            return {"total": 0}
        
        stats = {
            "total": len(triples),
            "avg_confidence": sum(t.confidence for t in triples) / len(triples),
            "types": {},
            "predicates": {},
            "subjects": set(),
            "objects": set()
        }
        
        # Count by type
        for triple in triples:
            triple_type = triple.triple_type.value
            stats["types"][triple_type] = stats["types"].get(triple_type, 0) + 1
            
            predicate = triple.predicate
            stats["predicates"][predicate] = stats["predicates"].get(predicate, 0) + 1
            
            stats["subjects"].add(triple.subject)
            stats["objects"].add(triple.object)
        
        stats["unique_subjects"] = len(stats["subjects"])
        stats["unique_objects"] = len(stats["objects"])
        stats["unique_entities"] = len(stats["subjects"] | stats["objects"])
        
        # Convert sets to lists for JSON serialization
        stats["subjects"] = list(stats["subjects"])
        stats["objects"] = list(stats["objects"])
        
        return stats 