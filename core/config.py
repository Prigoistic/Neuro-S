"""
Configuration settings for the neuro-symbolic reasoning agent.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class ModelConfig:
    """Configuration for NLP models and processing."""
    
    # SpaCy model configuration
    spacy_model: str = "en_core_web_sm"
    
    # Entity extraction settings
    entity_types: List[str] = None
    
    # Sentence splitting settings
    min_sentence_length: int = 10
    max_sentence_length: int = 500
    
    # Parsing thresholds
    confidence_threshold: float = 0.7
    min_triple_score: float = 0.5
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = [
                "PERSON", "ORG", "GPE", "EVENT", "DATE", 
                "MONEY", "PRODUCT", "LAW", "LANGUAGE"
            ]


@dataclass
class GraphConfig:
    """Configuration for knowledge graph building."""
    
    # Node types and properties
    node_types: List[str] = None
    edge_types: List[str] = None
    
    # Graph building parameters
    merge_similar_nodes: bool = True
    similarity_threshold: float = 0.8
    max_nodes_per_article: int = 1000
    max_edges_per_article: int = 2000
    
    def __post_init__(self):
        if self.node_types is None:
            self.node_types = ["Entity", "Event", "Concept", "Location", "Person", "Organization"]
        if self.edge_types is None:
            self.edge_types = ["RELATES_TO", "PARTICIPATES_IN", "LOCATED_IN", "CAUSED_BY", "LEADS_TO"]


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    
    # Processing settings
    batch_size: int = 32
    max_workers: int = 4
    enable_caching: bool = True
    cache_dir: str = ".cache"
    
    # Output settings
    output_format: str = "json"  # json, neo4j, rdf
    include_metadata: bool = True
    include_confidence_scores: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "reasoning_agent.log"


# Global configuration instance
CONFIG = {
    "model": ModelConfig(),
    "graph": GraphConfig(),
    "pipeline": PipelineConfig()
}


def get_config(section: str = None) -> Dict[str, Any]:
    """
    Get configuration settings.
    
    Args:
        section: Specific configuration section to retrieve.
                If None, returns all configurations.
    
    Returns:
        Configuration dictionary or specific section.
    """
    if section:
        return CONFIG.get(section, {})
    return CONFIG


def update_config(section: str, **kwargs) -> None:
    """
    Update configuration settings.
    
    Args:
        section: Configuration section to update.
        **kwargs: Key-value pairs to update.
    """
    if section in CONFIG:
        config_obj = CONFIG[section]
        for key, value in kwargs.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)


# Environment-based configuration overrides
def load_env_config():
    """Load configuration from environment variables."""
    
    # Model configuration from environment
    if os.getenv("SPACY_MODEL"):
        CONFIG["model"].spacy_model = os.getenv("SPACY_MODEL")
    
    if os.getenv("CONFIDENCE_THRESHOLD"):
        CONFIG["model"].confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD"))
    
    # Pipeline configuration from environment
    if os.getenv("BATCH_SIZE"):
        CONFIG["pipeline"].batch_size = int(os.getenv("BATCH_SIZE"))
    
    if os.getenv("MAX_WORKERS"):
        CONFIG["pipeline"].max_workers = int(os.getenv("MAX_WORKERS"))
    
    if os.getenv("LOG_LEVEL"):
        CONFIG["pipeline"].log_level = os.getenv("LOG_LEVEL")


# Load environment configuration on import
load_env_config() 