# Neuro-Symbolic Reasoning Agent

A sophisticated neuro-symbolic reasoning agent that extracts structured knowledge from news articles and builds comprehensive reasoning graphs. This system combines natural language processing, semantic triple extraction, and knowledge graph construction to transform unstructured text into structured, queryable knowledge representations.

## Features

- **Multi-layered NLP Processing**: Leverages spaCy and NLTK for robust text preprocessing and entity recognition
- **Semantic Triple Extraction**: Advanced pattern-based and dependency parsing methods to extract subject-predicate-object relationships
- **Knowledge Graph Construction**: Builds reasoning graphs with automatic node merging and relationship inference
- **Neo4j Integration**: Direct export capabilities for graph database ingestion
- **Configurable Pipeline**: Fully customizable processing parameters and thresholds
- **Comprehensive Statistics**: Detailed processing metrics and graph analytics

## Architecture

The system consists of five modular components:

### Core Components

1. **`pipeline.py`** - `ReasoningAgent` class

   - Main orchestrator that coordinates the entire processing pipeline
   - Implements `process_article(text: str)` method for end-to-end processing
   - Handles preprocessing, NER, triple extraction, and graph building

2. **`parser.py`** - `EventParser` class

   - Extracts structured triples from sentences using multiple methods:
     - Pattern-based extraction with regex
     - Dependency parsing with spaCy
     - Entity-relationship detection
   - Implements `parse_sentence(sentence: str)` returning (subject, predicate, object) tuples

3. **`graph_builder.py`** - `ReasoningGraph` class

   - Constructs knowledge graphs from extracted triples
   - Implements `build_graph(triples: list)` returning Neo4j-compatible structure
   - Features automatic node merging and relationship consolidation

4. **`nlp_utils.py`** - NLP utility functions

   - `extract_entities()` for named entity recognition
   - `split_sentences()` for text segmentation
   - `preprocess_text()` for text cleaning and normalization

5. **`config.py`** - Configuration management
   - Centralized settings for models, thresholds, and processing parameters
   - Environment variable support for deployment flexibility

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### Step 3: Optional NLTK Data Download

If using NLTK fallback features:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

## Quick Start

### Basic Usage

```python
from core import ReasoningAgent

# Initialize the agent
agent = ReasoningAgent()

# Process a news article
article_text = """
Apple Inc. announced today that CEO Tim Cook will lead
a new AI initiative in Silicon Valley. The project will
be headquartered in Palo Alto, California.
"""

result = agent.process_article(article_text)

# Access extracted information
print(f"Found {len(result.entities)} entities")
print(f"Extracted {len(result.triples)} semantic triples")
print(f"Built graph with {result.graph['metadata']['total_nodes']} nodes")
```

### Processing Results

The `ProcessingResult` object contains:

- **Input data**: Original and processed text
- **Intermediate results**: Sentences, entities, and triples
- **Knowledge graph**: Nodes, edges, and Neo4j-compatible format
- **Statistics**: Processing metrics and performance data

```python
# Access specific components
entities = result.entities
triples = result.triples
graph = result.graph

# Get processing statistics
stats = result.statistics
print(f"Processing time: {stats['performance']['processing_time']:.2f}s")
print(f"Entities per sentence: {stats['entities']['entities_per_sentence']:.2f}")
```

### Batch Processing

```python
articles = [
    "First news article text...",
    "Second news article text...",
    "Third news article text..."
]

results = agent.process_batch(articles)
for i, result in enumerate(results):
    print(f"Article {i+1}: {len(result.triples)} triples extracted")
```

## Configuration

### Environment Variables

```bash
export SPACY_MODEL="en_core_web_lg"
export CONFIDENCE_THRESHOLD=0.8
export BATCH_SIZE=16
export LOG_LEVEL="DEBUG"
```

### Programmatic Configuration

```python
from core import update_config

# Update model configuration
update_config("model",
    confidence_threshold=0.9,
    min_triple_score=0.6
)

# Update graph configuration
update_config("graph",
    merge_similar_nodes=True,
    similarity_threshold=0.85
)
```

## Output Formats

### Semantic Triples

```python
# Access extracted triples
for triple in result.triples:
    print(f"({triple['subject']}, {triple['predicate']}, {triple['object']})")
    print(f"Confidence: {triple['confidence']}")
```

### Knowledge Graph

```python
# Standard format
graph = result.graph
nodes = graph['nodes']
edges = graph['edges']

# Neo4j format
neo4j_data = graph['neo4j']
cypher_script = agent.export_graph_cypher(result)
```

### Neo4j Integration

```python
# Export Cypher statements for Neo4j
cypher_statements = agent.export_graph_cypher(result)
print(cypher_statements)

# Output example:
# CREATE (n:Organization {name: "Apple", id: "node_abc123"});
# CREATE (n:Person {name: "Tim Cook", id: "node_def456"});
# MATCH (a), (b) WHERE a.id = "node_abc123" AND b.id = "node_def456"
# CREATE (a)-[:LEADS]->(b);
```

## Examples

See the `examples/` folder for:

- **`example_input.txt`**: Sample news article demonstrating various entity types and relationships
- **`example_output.json`**: Complete processing result showing the full pipeline output structure

### Running the Example

```python
# Load example input
with open('examples/example_input.txt', 'r') as f:
    sample_article = f.read()

# Process the example
result = agent.process_article(sample_article)

# Compare with expected output
import json
with open('examples/example_output.json', 'r') as f:
    expected_output = json.load(f)
```

## Advanced Features

### Custom Entity Types

```python
from core import update_config

# Add custom entity types
update_config("model",
    entity_types=["PERSON", "ORG", "GPE", "PRODUCT", "TECHNOLOGY"]
)
```

### Graph Visualization

```python
import networkx as nx
import matplotlib.pyplot as plt

# Convert to NetworkX graph
G = nx.Graph()
for node in result.graph['nodes']:
    G.add_node(node['id'], label=node['label'], type=node['type'])

for edge in result.graph['edges']:
    G.add_edge(edge['source'], edge['target'],
               relationship=edge['relationship'])

# Visualize
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue')
plt.show()
```

### Performance Monitoring

```python
# Get detailed statistics
stats = agent.get_agent_statistics()
print(f"NLP Processor available: {stats['nlp_processor']['available']}")
print(f"Current model: {stats['nlp_processor']['model']}")

# Monitor processing performance
result = agent.process_article(text)
perf = result.statistics['performance']
print(f"Sentences/second: {perf['sentences_per_second']:.2f}")
print(f"Triples/sentence: {perf['triples_per_sentence']:.2f}")
```

## API Reference

### Main Classes

- **`ReasoningAgent`**: Main pipeline orchestrator
- **`EventParser`**: Semantic triple extraction
- **`ReasoningGraph`**: Knowledge graph construction
- **`NLPProcessor`**: NLP operations wrapper

### Data Classes

- **`ProcessingResult`**: Complete processing output
- **`Triple`**: Semantic triple with metadata
- **`Entity`**: Named entity with position and confidence
- **`GraphNode`**: Knowledge graph node
- **`GraphEdge`**: Knowledge graph edge

### Utility Functions

- **`process_news_article()`**: Quick processing function
- **`extract_entities()`**: Entity extraction utility
- **`split_sentences()`**: Sentence segmentation
- **`preprocess_text()`**: Text cleaning and normalization

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=core
```

## Performance Considerations

- **Memory Usage**: Large articles may require significant memory for graph construction
- **Processing Speed**: ~2-5 sentences per second depending on complexity
- **Scalability**: Batch processing recommended for multiple articles
- **Model Size**: Larger spaCy models provide better accuracy but slower processing

## Troubleshooting

### Common Issues

1. **spaCy model not found**:

   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **NLTK data missing**:

   ```python
   import nltk
   nltk.download('punkt')
   ```

3. **Memory issues with large texts**:
   - Reduce `max_nodes_per_article` and `max_edges_per_article` in configuration
   - Process articles in smaller batches

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{neuro_symbolic_reasoning_agent,
  title={Neuro-Symbolic Reasoning Agent for News Article Knowledge Extraction},
  author={Neuro-Symbolic AI Team},
  year={2024},
  version={1.0.0}
}
```
