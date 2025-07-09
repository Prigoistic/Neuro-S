"""
Neuro-Symbolic Reasoning Agent for News Article Knowledge Extraction.

This package provides a complete pipeline for extracting structured knowledge 
from news articles and building reasoning graphs.

Main Components:
- ReasoningAgent: Main pipeline orchestrator
- EventParser: Extracts semantic triples from text
- ReasoningGraph: Builds knowledge graphs from triples
- NLPProcessor: Handles NLP operations with spaCy/NLTK

Example Usage:
    from core import ReasoningAgent
    
    agent = ReasoningAgent()
    result = agent.process_article("Your news article text here...")
    print(result.graph)
"""

# Import main classes for easy access
from .pipeline import ReasoningAgent, ProcessingResult, process_news_article
from .parser import EventParser, Triple, TripleType
from .graph_builder import ReasoningGraph, GraphNode, GraphEdge
from .nlp_utils import NLPProcessor, Entity, extract_entities, split_sentences, preprocess_text
from .config import get_config, update_config, ModelConfig, GraphConfig, PipelineConfig

# Version information
__version__ = "1.0.0"
__author__ = "Neuro-Symbolic AI Team"

# Define public API
__all__ = [
    # Main classes
    'ReasoningAgent',
    'EventParser', 
    'ReasoningGraph',
    'NLPProcessor',
    
    # Data classes
    'ProcessingResult',
    'Triple',
    'GraphNode',
    'GraphEdge', 
    'Entity',
    
    # Enums
    'TripleType',
    
    # Configuration
    'ModelConfig',
    'GraphConfig', 
    'PipelineConfig',
    
    # Utility functions
    'process_news_article',
    'extract_entities',
    'split_sentences',
    'preprocess_text',
    'get_config',
    'update_config',
    
    # Package info
    '__version__',
    '__author__'
]

# Convenience imports for common usage patterns
def create_agent(nlp_model: str = None) -> ReasoningAgent:
    """
    Create a new reasoning agent with optional model specification.
    
    Args:
        nlp_model: Optional spaCy model name to use.
    
    Returns:
        Configured ReasoningAgent instance.
    """
    return ReasoningAgent(nlp_model)

def quick_process(text: str) -> ProcessingResult:
    """
    Quick processing function for single articles.
    
    Args:
        text: Article text to process.
    
    Returns:
        Processing result with extracted knowledge.
    """
    return process_news_article(text)

# Add convenience functions to public API
__all__.extend(['create_agent', 'quick_process']) 