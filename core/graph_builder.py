"""
Graph building module for constructing knowledge graphs from triples.
"""

import logging
import hashlib
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json

from .parser import Triple, TripleType
from .config import get_config


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    node_type: str = "Entity"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "id": self.id,
            "label": self.label,
            "properties": self.properties,
            "type": self.node_type
        }
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert node to Neo4j-compatible format."""
        return {
            "id": self.id,
            "labels": [self.node_type],
            "properties": {
                "name": self.label,
                **self.properties
            }
        }


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    id: str
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {
            "id": self.id,
            "source": self.source_id,
            "target": self.target_id,
            "relationship": self.relationship_type,
            "properties": self.properties
        }
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert edge to Neo4j-compatible format."""
        return {
            "id": self.id,
            "startNode": self.source_id,
            "endNode": self.target_id,
            "type": self.relationship_type,
            "properties": self.properties
        }


class ReasoningGraph:
    """Builder for knowledge graphs from semantic triples."""
    
    def __init__(self):
        """Initialize the reasoning graph builder."""
        self.config = get_config("graph")
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.entity_index: Dict[str, str] = {}  # Maps entity text to node ID
        self.statistics = {
            "nodes_created": 0,
            "edges_created": 0,
            "nodes_merged": 0,
            "triples_processed": 0
        }
    
    def build_graph(self, triples: List[Triple]) -> Dict[str, Any]:
        """
        Build a knowledge graph from a list of triples.
        
        Args:
            triples: List of Triple objects to convert to graph.
        
        Returns:
            Dictionary with nodes and edges suitable for Neo4j ingestion.
        """
        # Reset graph state
        self.nodes.clear()
        self.edges.clear()
        self.entity_index.clear()
        self.statistics = {
            "nodes_created": 0,
            "edges_created": 0,
            "nodes_merged": 0,
            "triples_processed": 0
        }
        
        # Process each triple
        for triple in triples:
            self._add_triple_to_graph(triple)
        
        # Apply post-processing
        if self.config.merge_similar_nodes:
            self._merge_similar_nodes()
        
        # Build final graph structure
        graph_data = self._build_graph_structure()
        
        # Add statistics
        graph_data["statistics"] = self.statistics
        
        return graph_data
    
    def _add_triple_to_graph(self, triple: Triple) -> None:
        """Add a single triple to the graph."""
        try:
            # Create or get subject node
            subject_node = self._create_or_get_node(
                triple.subject, 
                self._determine_node_type(triple.subject, triple.metadata)
            )
            
            # Create or get object node  
            object_node = self._create_or_get_node(
                triple.object,
                self._determine_node_type(triple.object, triple.metadata)
            )
            
            # Create edge between nodes
            edge = self._create_edge(
                subject_node.id,
                object_node.id,
                triple.predicate,
                triple
            )
            
            self.statistics["triples_processed"] += 1
            
        except Exception as e:
            logging.error(f"Error adding triple to graph: {triple} - {e}")
    
    def _create_or_get_node(self, entity_text: str, node_type: str) -> GraphNode:
        """Create a new node or return existing one for the entity."""
        # Normalize entity text for lookup
        normalized_text = self._normalize_entity_text(entity_text)
        
        # Check if node already exists
        if normalized_text in self.entity_index:
            node_id = self.entity_index[normalized_text]
            return self.nodes[node_id]
        
        # Create new node
        node_id = self._generate_node_id(entity_text)
        node = GraphNode(
            id=node_id,
            label=entity_text,
            node_type=node_type,
            properties={
                "normalized_text": normalized_text,
                "original_text": entity_text,
                "created_from": "triple_extraction"
            }
        )
        
        # Store node and update index
        self.nodes[node_id] = node
        self.entity_index[normalized_text] = node_id
        self.statistics["nodes_created"] += 1
        
        return node
    
    def _create_edge(self, source_id: str, target_id: str, relationship: str, triple: Triple) -> GraphEdge:
        """Create an edge between two nodes."""
        edge_id = self._generate_edge_id(source_id, target_id, relationship)
        
        # Check if edge already exists
        if edge_id in self.edges:
            # Update existing edge properties
            existing_edge = self.edges[edge_id]
            existing_edge.properties.setdefault("sources", []).append({
                "sentence": triple.source_sentence,
                "confidence": triple.confidence,
                "metadata": triple.metadata
            })
            return existing_edge
        
        # Create new edge
        edge = GraphEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship,
            properties={
                "confidence": triple.confidence,
                "triple_type": triple.triple_type.value,
                "sources": [{
                    "sentence": triple.source_sentence,
                    "confidence": triple.confidence,
                    "metadata": triple.metadata
                }]
            }
        )
        
        self.edges[edge_id] = edge
        self.statistics["edges_created"] += 1
        
        return edge
    
    def _determine_node_type(self, entity_text: str, metadata: Dict[str, Any]) -> str:
        """Determine the type of node based on entity text and metadata."""
        # Use entity type from metadata if available
        if metadata:
            if "entity1_type" in metadata:
                return self._map_entity_type_to_node_type(metadata["entity1_type"])
            if "entity2_type" in metadata:
                return self._map_entity_type_to_node_type(metadata["entity2_type"])
        
        # Simple heuristics based on text
        entity_lower = entity_text.lower()
        
        # Check for person indicators
        if any(indicator in entity_lower for indicator in ["mr.", "mrs.", "dr.", "prof."]):
            return "Person"
        
        # Check for organization indicators  
        if any(indicator in entity_lower for indicator in ["inc.", "corp.", "ltd.", "company", "organization"]):
            return "Organization"
        
        # Check for location indicators
        if any(indicator in entity_lower for indicator in ["city", "country", "state", "street", "avenue"]):
            return "Location"
        
        # Check for event indicators
        if any(indicator in entity_lower for indicator in ["conference", "meeting", "summit", "event"]):
            return "Event"
        
        # Default to Entity
        return "Entity"
    
    def _map_entity_type_to_node_type(self, entity_type: str) -> str:
        """Map spaCy entity types to graph node types."""
        mapping = {
            "PERSON": "Person",
            "ORG": "Organization", 
            "GPE": "Location",  # Geopolitical entity
            "EVENT": "Event",
            "DATE": "Temporal",
            "MONEY": "Financial",
            "PRODUCT": "Product",
            "LAW": "Legal",
            "LANGUAGE": "Language"
        }
        
        return mapping.get(entity_type, "Entity")
    
    def _normalize_entity_text(self, text: str) -> str:
        """Normalize entity text for consistent matching."""
        # Convert to lowercase and remove extra whitespace
        normalized = text.lower().strip()
        
        # Remove common articles and punctuation
        import re
        normalized = re.sub(r'^(the|a|an)\s+', '', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def _generate_node_id(self, entity_text: str) -> str:
        """Generate a unique ID for a node."""
        # Use hash of normalized text for consistent IDs
        normalized = self._normalize_entity_text(entity_text)
        hash_object = hashlib.md5(normalized.encode())
        return f"node_{hash_object.hexdigest()[:8]}"
    
    def _generate_edge_id(self, source_id: str, target_id: str, relationship: str) -> str:
        """Generate a unique ID for an edge."""
        edge_string = f"{source_id}_{relationship}_{target_id}"
        hash_object = hashlib.md5(edge_string.encode())
        return f"edge_{hash_object.hexdigest()[:8]}"
    
    def _merge_similar_nodes(self) -> None:
        """Merge nodes that represent the same entity."""
        if not self.config.merge_similar_nodes:
            return
        
        # Group nodes by similarity
        similarity_groups = self._find_similar_nodes()
        
        # Merge each group
        for group in similarity_groups:
            if len(group) > 1:
                self._merge_node_group(group)
    
    def _find_similar_nodes(self) -> List[List[str]]:
        """Find groups of similar nodes that should be merged."""
        # Simple implementation using text similarity
        from difflib import SequenceMatcher
        
        node_ids = list(self.nodes.keys())
        groups = []
        processed = set()
        
        for i, node_id1 in enumerate(node_ids):
            if node_id1 in processed:
                continue
                
            group = [node_id1]
            node1 = self.nodes[node_id1]
            
            for node_id2 in node_ids[i+1:]:
                if node_id2 in processed:
                    continue
                    
                node2 = self.nodes[node_id2]
                
                # Check similarity
                similarity = SequenceMatcher(
                    None, 
                    node1.properties.get("normalized_text", ""),
                    node2.properties.get("normalized_text", "")
                ).ratio()
                
                if similarity >= self.config.similarity_threshold:
                    group.append(node_id2)
            
            if len(group) > 1:
                groups.append(group)
                processed.update(group)
        
        return groups
    
    def _merge_node_group(self, node_ids: List[str]) -> None:
        """Merge a group of similar nodes into one."""
        # Use the first node as the primary node
        primary_id = node_ids[0]
        primary_node = self.nodes[primary_id]
        
        # Merge properties from other nodes
        for node_id in node_ids[1:]:
            if node_id not in self.nodes:
                continue
                
            node = self.nodes[node_id]
            
            # Merge properties
            for key, value in node.properties.items():
                if key not in primary_node.properties:
                    primary_node.properties[key] = value
                elif isinstance(value, list):
                    if isinstance(primary_node.properties[key], list):
                        primary_node.properties[key].extend(value)
                    else:
                        primary_node.properties[key] = [primary_node.properties[key]] + value
            
            # Update edges to point to primary node
            self._redirect_edges(node_id, primary_id)
            
            # Remove the merged node
            del self.nodes[node_id]
            self.statistics["nodes_merged"] += 1
    
    def _redirect_edges(self, old_node_id: str, new_node_id: str) -> None:
        """Redirect edges from old node to new node."""
        edges_to_update = []
        
        for edge_id, edge in self.edges.items():
            if edge.source_id == old_node_id:
                edge.source_id = new_node_id
                edges_to_update.append((edge_id, edge))
            elif edge.target_id == old_node_id:
                edge.target_id = new_node_id
                edges_to_update.append((edge_id, edge))
        
        # Update edge IDs if necessary
        for old_edge_id, edge in edges_to_update:
            new_edge_id = self._generate_edge_id(edge.source_id, edge.target_id, edge.relationship_type)
            
            if new_edge_id != old_edge_id:
                # Check if new edge already exists
                if new_edge_id in self.edges:
                    # Merge edge properties
                    existing_edge = self.edges[new_edge_id]
                    existing_edge.properties.setdefault("sources", []).extend(
                        edge.properties.get("sources", [])
                    )
                else:
                    # Update edge ID
                    edge.id = new_edge_id
                    self.edges[new_edge_id] = edge
                
                # Remove old edge
                if old_edge_id in self.edges:
                    del self.edges[old_edge_id]
    
    def _build_graph_structure(self) -> Dict[str, Any]:
        """Build the final graph structure for output."""
        # Convert nodes and edges to dictionaries
        nodes_list = [node.to_dict() for node in self.nodes.values()]
        edges_list = [edge.to_dict() for edge in self.edges.values()]
        
        # Create Neo4j-compatible format
        neo4j_nodes = [node.to_neo4j_dict() for node in self.nodes.values()]
        neo4j_edges = [edge.to_neo4j_dict() for edge in self.edges.values()]
        
        # Build graph summary
        summary = self._build_graph_summary()
        
        return {
            "nodes": nodes_list,
            "edges": edges_list,
            "neo4j": {
                "nodes": neo4j_nodes,
                "relationships": neo4j_edges
            },
            "summary": summary,
            "metadata": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "node_types": self._count_node_types(),
                "relationship_types": self._count_relationship_types()
            }
        }
    
    def _build_graph_summary(self) -> Dict[str, Any]:
        """Build a summary of the graph structure."""
        # Calculate graph metrics
        node_degrees = defaultdict(int)
        for edge in self.edges.values():
            node_degrees[edge.source_id] += 1
            node_degrees[edge.target_id] += 1
        
        # Find central nodes (highest degree)
        central_nodes = sorted(
            [(node_id, degree) for node_id, degree in node_degrees.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Get central node details
        central_node_details = []
        for node_id, degree in central_nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                central_node_details.append({
                    "id": node_id,
                    "label": node.label,
                    "type": node.node_type,
                    "degree": degree
                })
        
        return {
            "central_nodes": central_node_details,
            "avg_node_degree": sum(node_degrees.values()) / len(node_degrees) if node_degrees else 0,
            "max_degree": max(node_degrees.values()) if node_degrees else 0,
            "connected_components": self._count_connected_components()
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type."""
        type_counts = defaultdict(int)
        for node in self.nodes.values():
            type_counts[node.node_type] += 1
        return dict(type_counts)
    
    def _count_relationship_types(self) -> Dict[str, int]:
        """Count edges by relationship type."""
        type_counts = defaultdict(int)
        for edge in self.edges.values():
            type_counts[edge.relationship_type] += 1
        return dict(type_counts)
    
    def _count_connected_components(self) -> int:
        """Count the number of connected components in the graph."""
        # Simple DFS-based connected component counting
        visited = set()
        components = 0
        
        def dfs(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            
            # Visit connected nodes
            for edge in self.edges.values():
                if edge.source_id == node_id and edge.target_id not in visited:
                    dfs(edge.target_id)
                elif edge.target_id == node_id and edge.source_id not in visited:
                    dfs(edge.source_id)
        
        for node_id in self.nodes.keys():
            if node_id not in visited:
                dfs(node_id)
                components += 1
        
        return components
    
    def export_to_neo4j_cypher(self, graph_data: Dict[str, Any]) -> str:
        """Export graph data as Neo4j Cypher statements."""
        cypher_statements = []
        
        # Create nodes
        for node in graph_data["neo4j"]["nodes"]:
            labels = ":".join(node["labels"])
            properties = ", ".join([f"{k}: {json.dumps(v)}" for k, v in node["properties"].items()])
            cypher = f"CREATE (n:{labels} {{{properties}}});"
            cypher_statements.append(cypher)
        
        # Create relationships
        for rel in graph_data["neo4j"]["relationships"]:
            properties = ", ".join([f"{k}: {json.dumps(v)}" for k, v in rel["properties"].items()])
            props_str = f" {{{properties}}}" if properties else ""
            cypher = (f"MATCH (a), (b) WHERE a.id = {json.dumps(rel['startNode'])} "
                     f"AND b.id = {json.dumps(rel['endNode'])} "
                     f"CREATE (a)-[:{rel['type']}{props_str}]->(b);")
            cypher_statements.append(cypher)
        
        return "\n".join(cypher_statements)
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get current graph statistics."""
        return {
            **self.statistics,
            "current_nodes": len(self.nodes),
            "current_edges": len(self.edges)
        } 