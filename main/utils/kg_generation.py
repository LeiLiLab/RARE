import pandas as pd
import os
import networkx as nx
import pandas as pd
from typing import Dict, List, Any
from abc import ABC, abstractmethod
from tqdm import tqdm
from collections import defaultdict
import argparse
import logging
import json
logging.basicConfig(filename='logs/kg_generation.log', filemode='w', level=logging.INFO)
class BaseKGBuilder(ABC):
    """Abstract base class for building knowledge graphs across domains"""
    
    def __init__(self, output_dir: str, triplets_file: str = None, chunks_dir: str = None, model_name: str = None):
        """
        Initialize the knowledge graph builder
        
        Args:
            output_dir: Directory to save output files  
            triplets_file: Path to JSON file with triplets
            chunks_dir: Directory containing chunked documents
            model_name: Name of the LLM for generating connections (optional)
        """
        self.output_dir = output_dir
        self.model_name = model_name
        self.triplets_file = triplets_file
        self.chunks_dir = chunks_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_document_index(self) -> List[Dict]:
        """Load the document index containing references to all chunked documents"""
        index_file = os.path.join(self.chunks_dir, "document_index.json")
        if not os.path.exists(index_file):
            logging.info(f"Warning: Document index file not found at {index_file}")
            return []
            
        try:
            with open(index_file, 'r') as f:
                document_index = json.load(f)
            logging.info(f"Loaded index for {len(document_index)} documents")
            return document_index
        except Exception as e:
            logging.info(f"Error loading document index: {e}")
            return []
    
    def build_individual_graph(self, triplets: List[Dict]) -> nx.DiGraph:
        """
        Build a knowledge graph from a set of triplets
        
        Args:
            triplets: List of triplet dictionaries
            
        Returns:
            NetworkX directed graph
        """
        graph = nx.DiGraph()
        
        for triplet in triplets:
            entity_1 = triplet["entity_1"]
            relation = triplet["relation"]
            entity_2 = triplet["entity_2"]
            
            # Add nodes if they don't exist
            if not graph.has_node(entity_1):
                graph.add_node(entity_1, 
                              metadata=triplet.get("metadata", {}))
            
            if not graph.has_node(entity_2):
                graph.add_node(entity_2, 
                              answer_chunk_ids=triplet.get("answer_chunk_ids", []),
                              source_sentences=triplet.get("source_sentences", []),
                              metadata=triplet.get("metadata", {}))
            
            # Add edge with metadata
            graph.add_edge(entity_1, entity_2, 
                          label=relation,
                          triplet_id=triplet.get("triplet_id", ""),
                          answer_chunk_ids=triplet.get("answer_chunk_ids", []),
                          source_sentences=triplet.get("source_sentences", []),
                          metadata=triplet.get("metadata", {}))
        
        return graph
    
    def find_root_nodes(self, graph: nx.DiGraph, top_n: int = 1) -> List[str]:
        """
        Find potential root nodes for a graph based on out-degree
        
        Args:
            graph: NetworkX directed graph
            top_n: Number of top nodes to return
            
        Returns:
            List of node names sorted by out-degree
        """
        out_degrees = dict(graph.out_degree())
        sorted_nodes = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:top_n]]
    
    def merge_graphs(self, graphs: List[nx.DiGraph]) -> nx.DiGraph:
        """
        Merge multiple knowledge graphs into one
        
        Args:
            graphs: List of NetworkX directed graphs
            
        Returns:
            Merged NetworkX directed graph
        """
        # Just combines graphs without connecting them
        merged_graph = nx.DiGraph()
        
        for graph in graphs:
            merged_graph.add_nodes_from(graph.nodes(data=True))
            merged_graph.add_edges_from(graph.edges(data=True))
        
        return merged_graph
    
    def save_graph(self, graph: nx.DiGraph, filename: str) -> None:
        """
        Save knowledge graph to GraphML file
        
        Args:
            graph: NetworkX directed graph to save
            filename: Name of the output file
        """
        # Convert list and dict values to strings before saving to GraphML
        # GraphML doesn't support list or dictionary data types
        processed_graph = nx.DiGraph()
        
        # Copy nodes with converted attributes
        for node, data in graph.nodes(data=True):
            node_attrs = {}
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    node_attrs[key] = pd.Series([value]).to_json(orient='values').strip('[]')
                else:
                    node_attrs[key] = value
            processed_graph.add_node(node, **node_attrs)
            
        # Copy edges with converted attributes
        for u, v, data in graph.edges(data=True):
            edge_attrs = {}
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    edge_attrs[key] = pd.Series([value]).to_json(orient='values').strip('[]')
                else:
                    edge_attrs[key] = value
            processed_graph.add_edge(u, v, **edge_attrs)
        
        output_path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nx.write_graphml(processed_graph, output_path)
        logging.info(f"Knowledge graph saved to {output_path}")
        
        # Also save node and edge counts for reference
        logging.info(f"Graph statistics: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    
    @abstractmethod
    def process_triplets(self, triplets_file: str, **kwargs) -> None:
        """
        Process triplets to build and save knowledge graphs
        
        Args:
            triplets_file: Path to JSON file with triplets
            **kwargs: Additional arguments specific to domain implementation
        """
        pass


class FinanceKGBuilder(BaseKGBuilder):
    """Knowledge graph builder for finance domain"""
    
    def group_companies_by_sector(self, company_metadata: Dict) -> Dict[str, List[str]]:
        """
        Group companies by sector
        
        Args:
            company_metadata: Dictionary mapping CIK to company metadata
            
        Returns:
            Dictionary mapping sector to list of CIKs
        """
        sector_companies = defaultdict(list)
        
        for cik, metadata in company_metadata.items():
            sector = metadata.get('GICS_Sector', 'Unknown')
            sector_companies[sector].append(cik)
        
        return sector_companies
    
    def process_triplets(self, triplets_file: str) -> None:
        """
        Process triplets to build and save knowledge graphs
        
        Args:
            triplets_file: Path to JSON file with triplets
        """
        # Load triplets
        triplets_df = pd.read_json(triplets_file)
        triplets = triplets_df.to_dict(orient='records')
        logging.info(f"Loaded {len(triplets)} triplets")
        
        # Load company metadata from document index and chunks
        company_metadata = {}
        document_index = self.load_document_index()
        
        for doc_info in document_index:
            document_id = doc_info.get("document_id", "")
            domain = doc_info.get("domain", "")
            
            if domain == "finance" and document_id:
                # Load first chunk file to extract metadata
                chunk_file = doc_info.get("chunk_file")
                if chunk_file:
                    try:
                        chunk_file_path = os.path.join(self.chunks_dir, chunk_file)
                        chunks_df = pd.read_json(chunk_file_path)
                        chunks_data = chunks_df.to_dict(orient='records')
                        
                        if chunks_data:
                            # Extract metadata from first chunk
                            metadata = chunks_data[0].get("metadata", {})
                            cik = metadata.get("cik", "")
                            if cik and cik not in company_metadata:
                                company_metadata[cik] = metadata
                    except Exception as e:
                        logging.error(f"Error loading chunk file {chunk_file}: {e}")
        
        logging.info(f"Loaded metadata for {len(company_metadata)} companies")
        
        # Group triplets by CIK
        cik_triplets = {}
        for triplet in triplets:
            cik = triplet.get("metadata", {}).get("cik", "")
            if cik:
                if cik not in cik_triplets:
                    cik_triplets[cik] = []
                cik_triplets[cik].append(triplet)
        
        # Build individual company graphs
        company_graphs = {}
        for cik, triplets_list in tqdm(cik_triplets.items(), desc="Building company graphs"):
            graph = self.build_individual_graph(triplets_list)
            filename = f"company_{cik}.graphml"
            self.save_graph(graph, filename)
            company_graphs[cik] = graph
        
        # Step 1: Group companies by sector
        sector_companies = self.group_companies_by_sector(company_metadata)
        processed_ciks = set()
        
        # Merge graphs by sector (max 4 companies per file)
        logging.info("Merging graphs by sector")
        for sector, ciks in tqdm(sector_companies.items(), desc="Building sector graphs"):
            if sector == "Unknown":
                continue
                
            # Filter to include only CIKs that exist in company_graphs
            valid_ciks = [cik for cik in ciks if cik in company_graphs]
            
            # Skip if no valid CIKs
            if not valid_ciks:
                continue
                
            # Split companies into groups of at most 4
            for batch_idx in range(0, len(valid_ciks), 4):
                batch_ciks = valid_ciks[batch_idx:batch_idx+4]
                sector_graphs = [company_graphs[cik] for cik in batch_ciks]
                
                if sector_graphs:
                    merged_graph = self.merge_graphs(sector_graphs)
                    # Append batch number if there's more than one batch
                    batch_suffix = f"_batch_{batch_idx//4+1}" if len(valid_ciks) > 4 else ""
                    filename = f"finance_{sector.replace(' ', '_')}{batch_suffix}.graphml"
                    self.save_graph(merged_graph, filename)
                    
                    # Mark these CIKs as processed
                    processed_ciks.update(batch_ciks)

class EconomicsKGBuilder(BaseKGBuilder):
    """Knowledge graph builder for economy domain"""
    def merge_graphs(self, graphs: List[nx.DiGraph]) -> nx.DiGraph:
        # Start with base implementation to combine all graphs
        merged_graph = super().merge_graphs(graphs)
        
        # Create a central node for the group
        central_node = 'OECD Countries'
        merged_graph.add_node(central_node, type='general')
        
        # Connect all country nodes to the central node
        for idx, graph in enumerate(graphs):
            if len(graph.nodes) == 0:
                continue
            
            # Find root nodes
            root_nodes = self.find_root_nodes(graph, top_n=1)
            if not root_nodes:
                continue
            
            # Connect the root nodes to the central node
            for root_node in root_nodes:
                merged_graph.add_edge(
                    root_node,
                    central_node,
                    label="OECD member country",
                    type=f"general_connection"
                )
        
        return merged_graph
    
    
    def process_triplets(self, triplets_file: str, **kwargs) -> None:
        """
        Process triplets to build and save knowledge graphs for economy domain
        
        Args:
            triplets_file: Path to JSON file with triplets
            **kwargs: Additional arguments specific to economics domain implementation
        """
        # Get a list of triplets file paths
        triplets_files = os.listdir(triplets_file)
        triplets_files = [os.path.join(triplets_file, f) for f in triplets_files if f.endswith('.json')]
        logging.info(f"Found {len(triplets_files)} triplet files")
        
        # Load triplets and group by country
        country_triplets = defaultdict(list)
        for file in triplets_files:
            file_path = os.path.join(triplets_file, file)
            triplets = pd.read_json(file_path)
            triplets = triplets.to_dict(orient='records')    
            logging.info(f"Loaded {len(triplets)} triplets from {file}")
            
            # Group triplets by country
            if triplets:
                metadata = triplets[0].get('metadata', {})
                country = metadata.get('file_country', 'unknown')
                country_triplets[country].extend(triplets)
        logging.info(f"Grouped triplets for {len(country_triplets)} countries")
        
        # Build individual country graphs
        country_graphs = {}
        for country, triplets in country_triplets.items():
            graph = self.build_individual_graph(triplets)
            filename = f"{country.replace(' ', '_')}.graphml"
            self.save_graph(graph, os.path.join('graphs', filename))
            country_graphs[country] = graph
            logging.info(f"Graph for {country} saved")
        logging.info(f"Built graphs for {len(country_graphs)} countries")
        
        # Group all country graphs into a single graph
        merged_graph = self.merge_graphs(list(country_graphs.values()))
        filename = "OECD_graph.graphml"
        self.save_graph(merged_graph, filename)
        logging.info(f"Final merged graph saved as {filename}")



class PolicyKGBuilder(BaseKGBuilder):
    """Knowledge graph builder for policy domain - placeholder for future implementation"""
    def merge_graphs(self, graphs: List[nx.DiGraph]) -> nx.DiGraph:
        # Start with base implementation to combine all graphs
        merged_graph = super().merge_graphs(graphs)
        
        # Create a central node for the group
        central_node = 'federal housing and community development funds'
        merged_graph.add_node(central_node, type='general')
        
        # Connect all grantee nodes to the central node
        for idx, graph in enumerate(graphs):
            if len(graph.nodes) == 0:
                continue
            
            # Find root nodes
            root_nodes = self.find_root_nodes(graph, top_n=1)
            if not root_nodes:
                continue
            
            # Connect the root nodes to the central node
            for root_node in root_nodes:
                merged_graph.add_edge(
                    root_node,
                    central_node,
                    label="grantees that receive funds",
                    type=f"general_connection"
                )
        
        return merged_graph
    
    
    def process_triplets(self, triplets_file: str, **kwargs) -> None:
        """
        Process triplets to build and save knowledge graphs for policy domain
        
        Args:
            triplets_file: Path to JSON file with triplets
            **kwargs: Additional arguments specific to policy implementation
        """
        # Get a list of triplets file paths
        triplets_files = os.listdir(triplets_file)
        triplets_files = [os.path.join(triplets_file, f) for f in triplets_files if f.endswith('.json')]
        logging.info(f"Found {len(triplets_files)} triplet files")
        
        # Load triplets
        grantee_triplets = defaultdict(list)
        for file in triplets_files:
            file_path = os.path.join(triplets_file, file)
            triplets = pd.read_json(file_path).to_dict(orient='records')    
            print(f"Loaded {len(triplets)} triplets from {file}")
            
            # Group triplets by grantee
            if triplets:
                metadata = triplets[0].get('metadata', {})
                grantee = metadata.get('file_grantee', 'unknown')
                grantee_triplets[grantee].extend(triplets)
        print(f"Grouped triplets for {len(grantee_triplets)} grantees")
        
        # Build individual grantee graphs
        grantee_graphs = {}
        for grantee, triplets in grantee_triplets.items():
            graph = self.build_individual_graph(triplets)
            filename = f"{grantee.replace(' ', '_')}.graphml"
            self.save_graph(graph, os.path.join('graphs', filename))
            grantee_graphs[grantee] = graph
            print(f"Graph for {grantee} saved")
        print(f"Built graphs for {len(grantee_graphs)} grantees")
        
        # Group all grantee graphs into a single graph
        merged_graph = self.merge_graphs(list(grantee_graphs.values()))
        filename = "grantee_graph.graphml"
        self.save_graph(merged_graph, filename)
        logging.info(f"Final merged graph saved as {filename}")


def get_knowledge_graph_builder(domain: str, output_dir: str, triplets_file: str, chunks_dir: str, **kwargs) -> BaseKGBuilder:
    """Factory function to get the appropriate knowledge graph builder for a domain"""
    if domain == "finance":
        return FinanceKGBuilder(output_dir, triplets_file, chunks_dir, **kwargs)
    elif domain == "economics":
        return EconomicsKGBuilder(output_dir, **kwargs)
    elif domain == "policy":
        return PolicyKGBuilder(output_dir, **kwargs)
    else:
        raise ValueError(f"Unknown domain: {domain}")

def main():
    """Main function to directly build knowledge graphs from triplets"""
    parser = argparse.ArgumentParser(description="Knowledge Graph Builder")
    parser.add_argument("--domain", choices=["finance", "economics", "policy"], required=True,
                      help="Domain for the knowledge graph")
    parser.add_argument("--triplets_file", required=True,
                      help="Path to JSON file containing triplets")
    parser.add_argument("--chunks_dir", required=True,
                      help="Directory containing chunked documents")
    parser.add_argument("--output_dir", required=True,
                      help="Directory to save generated graphs")
    parser.add_argument("--model_name", default=None,
                      help="LLM model name for generating connections (optional)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get appropriate knowledge graph builder
    kg_builder = get_knowledge_graph_builder(
        args.domain, 
        output_dir=args.output_dir,
        triplets_file=args.triplets_file,
        chunks_dir=args.chunks_dir,
        model_name=args.model_name
    )
    
    kg_builder.process_triplets()
    
    logging.info(f"Knowledge graph generation completed. Graphs saved to {args.output_dir}")


if __name__ == "__main__":
    main()