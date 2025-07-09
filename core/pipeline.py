"""
Main pipeline for the neuro-symbolic reasoning agent.
"""

import logging
import time
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .nlp_utils import NLPProcessor, preprocess_text, split_sentences, extract_entities
from .parser import EventParser, Triple
from .graph_builder import ReasoningGraph
from .config import get_config


@dataclass
class ProcessingResult:
    """Result of processing an article through the reasoning pipeline."""
    
    # Input data
    original_text: str
    processed_text: str
    
    # Intermediate results
    sentences: List[str]
    entities: List[Dict[str, Any]]
    triples: List[Dict[str, Any]]
    
    # Final graph
    graph: Dict[str, Any]
    
    # Metadata
    processing_time: float
    statistics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "input": {
                "original_text": self.original_text,
                "processed_text": self.processed_text
            },
            "intermediate": {
                "sentences": self.sentences,
                "entities": self.entities,
                "triples": self.triples
            },
            "graph": self.graph,
            "metadata": {
                "processing_time": self.processing_time,
                "statistics": self.statistics
            }
        }


class ReasoningAgent:
    """
    Main reasoning agent that processes articles and extracts structured knowledge.
    
    This agent coordinates the entire pipeline:
    1. Preprocessing of input text
    2. Named Entity Recognition (NER)
    3. Extraction of structured triples
    4. Building of reasoning graph
    """
    
    def __init__(self, nlp_model: str = None):
        """
        Initialize the reasoning agent.
        
        Args:
            nlp_model: Optional spaCy model name to use.
        """
        self.config = get_config()
        
        # Initialize components
        self.nlp_processor = NLPProcessor(nlp_model)
        self.parser = EventParser(self.nlp_processor)
        self.graph_builder = ReasoningGraph()
        
        # Setup logging
        self._setup_logging()
        
        logging.info("ReasoningAgent initialized successfully")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        pipeline_config = self.config["pipeline"]
        
        logging.basicConfig(
            level=getattr(logging, pipeline_config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(pipeline_config.log_file),
                logging.StreamHandler()
            ]
        )
    
    def process_article(self, text: str) -> ProcessingResult:
        """
        Process a news article and extract structured knowledge.
        
        Args:
            text: Raw article text to process.
        
        Returns:
            ProcessingResult containing all extracted information and graph.
        """
        start_time = time.time()
        
        try:
            logging.info("Starting article processing")
            
            # Step 1: Preprocess input text
            processed_text = self._preprocess_input(text)
            logging.info(f"Text preprocessing completed. Length: {len(processed_text)} chars")
            
            # Step 2: Split into sentences
            sentences = self._split_into_sentences(processed_text)
            logging.info(f"Sentence splitting completed. Found {len(sentences)} sentences")
            
            # Step 3: Perform Named Entity Recognition
            entities = self._perform_ner(processed_text)
            logging.info(f"NER completed. Found {len(entities)} entities")
            
            # Step 4: Extract structured triples
            triples = self._extract_triples(sentences)
            logging.info(f"Triple extraction completed. Found {len(triples)} triples")
            
            # Step 5: Build reasoning graph
            graph = self._build_reasoning_graph(triples)
            logging.info(f"Graph building completed. "
                        f"Nodes: {graph['metadata']['total_nodes']}, "
                        f"Edges: {graph['metadata']['total_edges']}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Compile statistics
            statistics = self._compile_statistics(
                processed_text, sentences, entities, triples, graph, processing_time
            )
            
            # Create result object
            result = ProcessingResult(
                original_text=text,
                processed_text=processed_text,
                sentences=sentences,
                entities=[entity.to_dict() if hasattr(entity, 'to_dict') else entity for entity in entities],
                triples=[triple.to_dict() if hasattr(triple, 'to_dict') else triple for triple in triples],
                graph=graph,
                processing_time=processing_time,
                statistics=statistics
            )
            
            logging.info(f"Article processing completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logging.error(f"Error processing article: {e}")
            # Return partial result with error information
            processing_time = time.time() - start_time
            return ProcessingResult(
                original_text=text,
                processed_text="",
                sentences=[],
                entities=[],
                triples=[],
                graph={"error": str(e)},
                processing_time=processing_time,
                statistics={"error": str(e)}
            )
    
    def _preprocess_input(self, text: str) -> str:
        """
        Preprocess the input text.
        
        Args:
            text: Raw input text.
        
        Returns:
            Preprocessed text.
        """
        if not text or not text.strip():
            raise ValueError("Input text is empty or None")
        
        # Use the preprocessing function from nlp_utils
        processed = preprocess_text(text)
        
        # Additional preprocessing specific to news articles
        processed = self._clean_news_text(processed)
        
        return processed
    
    def _clean_news_text(self, text: str) -> str:
        """
        Clean news-specific formatting and artifacts.
        
        Args:
            text: Text to clean.
        
        Returns:
            Cleaned text.
        """
        # Remove bylines and datelines
        text = re.sub(r'^.*?-- ', '', text)
        text = re.sub(r'\(.*?AP\).*?--', '', text)
        text = re.sub(r'\(.*?Reuters\).*?--', '', text)
        
        # Remove image captions and photo credits
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'Photo by.*?\.', '', text)
        text = re.sub(r'Image:.*?\.', '', text)
        
        # Remove advertisement markers
        text = re.sub(r'ADVERTISEMENT.*?CONTINUE READING', '', text, flags=re.DOTALL)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Preserve paragraph breaks
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
        
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split.
        
        Returns:
            List of sentences.
        """
        sentences = split_sentences(text, self.nlp_processor)
        
        # Filter out very short or malformed sentences
        filtered_sentences = []
        for sentence in sentences:
            # Remove sentences that are too short or just punctuation
            if len(sentence.split()) >= 3 and not re.match(r'^[^\w]*$', sentence):
                filtered_sentences.append(sentence)
        
        return filtered_sentences
    
    def _perform_ner(self, text: str) -> List[Any]:
        """
        Perform Named Entity Recognition on the text.
        
        Args:
            text: Text to analyze.
        
        Returns:
            List of entities.
        """
        entities = extract_entities(text, self.nlp_processor)
        
        # Filter entities by confidence if available
        model_config = self.config["model"]
        filtered_entities = []
        
        for entity in entities:
            if entity.confidence >= model_config.confidence_threshold:
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def _extract_triples(self, sentences: List[str]) -> List[Triple]:
        """
        Extract structured triples from sentences.
        
        Args:
            sentences: List of sentences to process.
        
        Returns:
            List of extracted triples.
        """
        all_triples = []
        
        # Process sentences in batches for efficiency
        pipeline_config = self.config["pipeline"]
        batch_size = pipeline_config.batch_size
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_triples = self.parser.batch_parse(batch)
            
            # Flatten the batch results
            for triple_list in batch_triples:
                all_triples.extend(triple_list)
        
        return all_triples
    
    def _build_reasoning_graph(self, triples: List[Triple]) -> Dict[str, Any]:
        """
        Build a reasoning graph from extracted triples.
        
        Args:
            triples: List of triples to convert to graph.
        
        Returns:
            Graph structure suitable for Neo4j ingestion.
        """
        # Apply limits to prevent excessive graph size
        graph_config = self.config["graph"]
        
        if len(triples) > graph_config.max_edges_per_article:
            # Sort by confidence and take top triples
            sorted_triples = sorted(triples, key=lambda t: t.confidence, reverse=True)
            triples = sorted_triples[:graph_config.max_edges_per_article]
            logging.warning(f"Limited triples to {len(triples)} due to max_edges_per_article setting")
        
        graph = self.graph_builder.build_graph(triples)
        
        # Apply node limits
        if graph["metadata"]["total_nodes"] > graph_config.max_nodes_per_article:
            logging.warning(f"Graph has {graph['metadata']['total_nodes']} nodes, "
                          f"exceeding limit of {graph_config.max_nodes_per_article}")
        
        return graph
    
    def _compile_statistics(self, text: str, sentences: List[str], entities: List[Any], 
                          triples: List[Triple], graph: Dict[str, Any], 
                          processing_time: float) -> Dict[str, Any]:
        """
        Compile comprehensive statistics about the processing.
        
        Returns:
            Dictionary with statistics.
        """
        # Basic text statistics
        word_count = len(text.split())
        char_count = len(text)
        
        # Entity statistics
        entity_types = {}
        for entity in entities:
            entity_type = entity.label if hasattr(entity, 'label') else 'Unknown'
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Triple statistics
        triple_stats = self.parser.get_statistics(triples)
        
        # Graph statistics
        graph_stats = graph.get("statistics", {})
        
        # Performance statistics
        sentences_per_second = len(sentences) / processing_time if processing_time > 0 else 0
        triples_per_sentence = len(triples) / len(sentences) if sentences else 0
        
        return {
            "text": {
                "character_count": char_count,
                "word_count": word_count,
                "sentence_count": len(sentences),
                "avg_sentence_length": word_count / len(sentences) if sentences else 0
            },
            "entities": {
                "total": len(entities),
                "types": entity_types,
                "entities_per_sentence": len(entities) / len(sentences) if sentences else 0
            },
            "triples": triple_stats,
            "graph": graph_stats,
            "performance": {
                "processing_time": processing_time,
                "sentences_per_second": sentences_per_second,
                "triples_per_sentence": triples_per_sentence
            }
        }
    
    def process_batch(self, articles: List[str]) -> List[ProcessingResult]:
        """
        Process multiple articles in batch.
        
        Args:
            articles: List of article texts to process.
        
        Returns:
            List of processing results.
        """
        results = []
        
        for i, article in enumerate(articles):
            logging.info(f"Processing article {i+1}/{len(articles)}")
            result = self.process_article(article)
            results.append(result)
        
        return results
    
    def export_graph_cypher(self, result: ProcessingResult) -> str:
        """
        Export the reasoning graph as Neo4j Cypher statements.
        
        Args:
            result: Processing result containing the graph.
        
        Returns:
            Cypher statements as string.
        """
        return self.graph_builder.export_to_neo4j_cypher(result.graph)
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the agent's components.
        
        Returns:
            Dictionary with component statistics.
        """
        return {
            "nlp_processor": {
                "available": self.nlp_processor.is_available(),
                "model": self.nlp_processor.model_name
            },
            "graph_builder": self.graph_builder.get_graph_statistics(),
            "configuration": {
                "model": self.config["model"].__dict__,
                "graph": self.config["graph"].__dict__,
                "pipeline": self.config["pipeline"].__dict__
            }
        }
    
    def update_configuration(self, section: str, **kwargs) -> None:
        """
        Update agent configuration.
        
        Args:
            section: Configuration section to update.
            **kwargs: Configuration parameters to update.
        """
        from .config import update_config
        update_config(section, **kwargs)
        
        # Reload configuration
        self.config = get_config()
        logging.info(f"Updated configuration for section: {section}")


# Convenience function for quick processing
def process_news_article(text: str, nlp_model: str = None) -> ProcessingResult:
    """
    Convenience function to quickly process a single news article.
    
    Args:
        text: Article text to process.
        nlp_model: Optional spaCy model name.
    
    Returns:
        Processing result.
    """
    agent = ReasoningAgent(nlp_model)
    return agent.process_article(text) 