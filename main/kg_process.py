import os
import random
from typing import Dict, List, Optional
import logging
import json
import argparse
from tqdm import tqdm
from abc import ABC, abstractmethod
import pandas as pd

from utils.api_keys import OPENAI_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME
from utils.kg_triplet_extraction import get_triplet_extractor
from utils.kg_generation import get_knowledge_graph_builder

# Set up logger
logging.basicConfig(level=logging.INFO, filename='logs/kg_process.log', filemode='w')

# os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
# os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
# os.environ["AWS_REGION_NAME"] = AWS_REGION_NAME

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class BaseKGProcessor(ABC):
    """Base class for knowledge graph processing across domains"""
    
    def __init__(self, 
                 output_dir: str,
                 model_name: str,
                 chunks_dir: str = None,
                 relations_file: str = None,
                 triplets_file: str = None):
        """Initialize the processor with required files and configuration"""
        self.output_dir = output_dir
        self.model_name = model_name
        self.chunks_dir = chunks_dir
        self.relations_file = relations_file
        self.triplets_file = triplets_file
        
        os.makedirs(output_dir, exist_ok=True)
        self.triplet_extractor = get_triplet_extractor(self.domain, model_name)
        self.knowledge_graph_builder = get_knowledge_graph_builder(
            self.domain, 
            output_dir,
            triplets_file,
            chunks_dir,
            model_name=model_name
        )
        
        # Document index tracks all available documents
        self.document_index = []
        # Dictionary to store chunk data by document ID
        self.document_chunks = {}
        # For finance domain, we need to load company metadata
        self.company_metadata = {}
    
    @property
    @abstractmethod
    def domain(self) -> str:
        """Return the domain name"""
        pass
    
    def load_document_index(self) -> None:
        """Load the document index file that contains information about all chunked documents"""
        index_file = os.path.join(self.chunks_dir, "document_index.json")
        if not os.path.exists(index_file):
            logging.info(f"Warning: Document index file not found at {index_file}")
            return
            
        try:
            with open(index_file, 'r') as f:
                self.document_index = json.load(f)
            logging.info(f"Loaded index for {len(self.document_index)} documents")
        except Exception as e:
            logging.info(f"Error loading document index: {e}")
            self.document_index = []
    
    def load_document_chunks(self, document_info: Dict) -> Dict[str, Dict]:
        """Load chunks for a specific document"""
        document_id = document_info.get("document_id")
        chunk_file = document_info.get("chunk_file")
        
        if not chunk_file:
            logging.info(f"Warning: No chunk file found for document {document_id}")
            return {}
            
        try:
            chunk_file_path = os.path.join(self.chunks_dir, chunk_file)
            chunks_df = pd.read_json(chunk_file_path)
            chunks_data = chunks_df.to_dict(orient='records')
            
            # Extract company metadata for finance domain
            if self.domain == "finance":
                for chunk in chunks_data:
                    metadata = chunk.get("metadata", {})
                    cik = metadata.get("cik", "")
                    if cik and cik not in self.company_metadata:
                        self.company_metadata[cik] = metadata
            
            # Create chunks dictionary for this document
            chunks_dict = {}
            for chunk in chunks_data:
                if "chunk_id" in chunk:
                    chunks_dict[chunk["chunk_id"]] = {
                        "text": chunk.get("text", ""),
                        "metadata": chunk.get("metadata", {})
                    }
            
            logging.info(f"Loaded {len(chunks_dict)} chunks for document {document_id}")
            return chunks_dict
        except Exception as e:
            logging.info(f"Error loading chunks for document {document_id}: {e}")
            return {}
    
    @abstractmethod
    def process_chunks(self, chunks_dict: Dict[str, Dict]) -> List[Dict]:
        """Process chunks into triplets"""
        pass
    
    @abstractmethod
    def build_knowledge_graphs(self) -> None:
        """Build knowledge graphs from triplets"""
        pass
    
    @abstractmethod
    def extract_doc_triplets(self, sample_size: Optional[int] = None, sectors: Optional[List[str]] = None) -> List[Dict]:
        """Extract triplets from documents with optional filtering by sector or sample size"""
        pass
    
    def process(self, mode: str = "full", sample_size: Optional[int] = None, sector: Optional[str] = None) -> None:
        """Process data for knowledge graph generation"""
        if mode in ["full", "extract_only"]:
            self.extract_doc_triplets(sample_size, sector)
            if mode == "extract_only":
                return
        
        if mode in ["full", "graph_only"]:
            self.build_knowledge_graphs()


class FinanceKGProcessor(BaseKGProcessor):
    """Processor for finance domain knowledge graphs"""
    
    def __init__(self, output_dir: str, model_name: str, **kwargs):
        # Set default paths if not provided
        chunks_dir = kwargs.get('chunks_dir', output_dir)
        relations_file = kwargs.get('relations_file', os.path.join(output_dir, "relations.json"))
        triplets_file = kwargs.get('triplets_file', os.path.join(output_dir, "triplets.json"))
        
        super().__init__(
            output_dir=output_dir,
            model_name=model_name,
            chunks_dir=chunks_dir,
            relations_file=relations_file,
            triplets_file=triplets_file
        )
    
    @property
    def domain(self) -> str:
        return "finance"
    
    def extract_doc_triplets(self, sample_size: Optional[int] = None, sectors: Optional[List[str]] = None) -> List[Dict]:
        """Extract triplets from documents with optional filtering by sector or sample size"""
        # Load relations and document index
        if os.path.exists(self.relations_file):
            self.triplet_extractor.load_relations(self.relations_file)
        self.load_document_index()
        
        # Filter documents by sector if requested (finance domain only)
        filtered_documents = []
        if self.domain == "finance" and sectors:
            for doc_info in self.document_index:
                document_id = doc_info.get("document_id")
                # Load chunks to check sector
                doc_chunks = self.load_document_chunks(doc_info)
                if doc_chunks:
                    # Get sector from first chunk's metadata
                    first_chunk_id = next(iter(doc_chunks))
                    metadata = doc_chunks[first_chunk_id].get("metadata", {})
                    sector = metadata.get("GICS_Sector")
                    if sector in sectors:
                        filtered_documents.append(doc_info)
            logging.info(f"kg_process: Filtered {len(filtered_documents)} documents with sectors '{sectors}'")
        else:
            filtered_documents = self.document_index
        
        # Sample documents if requested
        if sample_size and sample_size < len(filtered_documents):
            sampled_documents = random.sample(filtered_documents, sample_size)
            logging.info(f"Sampled {len(sampled_documents)} documents from {len(filtered_documents)} filtered documents")
        else:
            sampled_documents = filtered_documents
            logging.info(f"Processing all {len(sampled_documents)} documents")
        
        # Process each document
        all_triplets = []
        for doc_info in tqdm(sampled_documents, desc="Processing documents"):
            document_id = doc_info.get("document_id")
            chunks_dict = self.load_document_chunks(doc_info)
            if chunks_dict:
                document_triplets = self.process_chunks(chunks_dict)
                all_triplets.extend(document_triplets)
                logging.info(f"Processed document {document_id}: {len(document_triplets)} triplets")
        
        self.triplet_extractor.save_relations(self.relations_file)
        
        # Save all triplets
        pd.DataFrame(all_triplets).to_json(self.triplets_file, orient='records', indent=2)
        logging.info(f"Saved {len(all_triplets)} triplets to {self.triplets_file}")
        
        return all_triplets
    
    def process_chunks(self, chunks_dict: Dict[str, Dict]) -> List[Dict]:
        """Process finance chunks into triplets"""
        logging.info(f"kg_process: Processing {len(chunks_dict)} chunks")
        triplets = self.triplet_extractor.extract_triplets(chunks_dict)
        processed = self.triplet_extractor.process_extracted_triplets(
            triplets, chunks_dict, domain='finance'
        )
        logging.info(f"kg_process: Extracted {len(processed)} triplets")
        return processed
    
    def build_knowledge_graphs(self) -> None:
        """Build finance domain knowledge graphs"""
        self.knowledge_graph_builder.process_triplets(self.triplets_file)


class EconomicsKGProcessor(BaseKGProcessor):
    """Processor for economics domain knowledge graphs"""
    
    def __init__(self, output_dir: str, model_name: str, **kwargs):
        chunks_dir = kwargs.get('chunks_dir')
        relations_file = kwargs.get('relations_file')
        triplets_file = kwargs.get('triplets_file')
        
        super().__init__(
            output_dir=output_dir,
            model_name=model_name,
            chunks_dir=chunks_dir,
            relations_file=relations_file,
            triplets_file=triplets_file
        )
    
    @property
    def domain(self) -> str:
        return "economics"
    
    def extract_doc_triplets(self, sample_size: Optional[int] = None, sectors: Optional[List[str]] = None) -> List[Dict]:
        """Extract triplets from documents with optional filtering by sector or sample size"""
        # Load relations and document index
        if os.path.exists(self.relations_file):
            self.triplet_extractor.load_relations(self.relations_file)
    
        doc_list = os.listdir(self.chunks_dir)
        # Sample documents if requested
        if sample_size and sample_size < len(doc_list):
            doc_list = random.sample(doc_list, sample_size)
            logging.info(f"Sampled {len(doc_list)} documents from {len(doc_list)} filtered documents")
        else:
            logging.info(f"Processing all {len(doc_list)} documents")
        
        # Process each document
        all_triplets = []
        for doc_file in tqdm(doc_list, desc="Processing documents"):
            # Load chunks for the document
            file_path = os.path.join(self.chunks_dir, doc_file)
            save_path = os.path.join(self.output_dir, 'triplets', doc_file)
            if os.path.exists(save_path):
                print(f"kg_process: Triplets already exist for {doc_file}, skipping...")
                continue
            
            try:
                chunks = pd.read_json(file_path).to_dict(orient='records')
            except:
                chunks = pd.read_json(file_path, lines=True).to_dict(orient='records')
            print(f"kg_process: Loaded {len(chunks)} chunks from {file_path}")
            
            chunks_dict = {chk['chunk_id']: chk for chk in chunks}
            
            # Process chunks into triplets
            doc_triplets = self.process_chunks(chunks_dict)
            
            # Save triplets to file
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(doc_triplets, f, indent=2)
            print(f"kg_process: Generated {len(doc_triplets)} triplets for {file_path}")
            print(f"kg_process: Saved triplets to {save_path}")
            
            all_triplets.extend(doc_triplets)
        
        # Save relations
        self.triplet_extractor.save_relations(self.relations_file)
        
        # Save all triplets
        triplet_file = os.path.join(self.output_dir, 'triplets.json')
        os.makedirs(os.path.dirname(triplet_file), exist_ok=True)
        with open(triplet_file, 'w') as f:
            json.dump(all_triplets, f, indent=2)
        logging.info(f"kg_process: Saved {len(all_triplets)} triplets to {triplet_file}")
        
        return all_triplets
    
    
    def process_chunks(self, chunks_dict: Dict[str, Dict]) -> List[Dict]:
        """Process economics chunks into triplets"""
        triplets = self.triplet_extractor.extract_triplets(chunks_dict)
        logging.info(f"kg_process: Extracted {len(triplets)} triplets")
        processed = self.triplet_extractor.process_extracted_triplets(
            triplets, chunks_dict, domain='economics'
        )
        return processed

    
    def build_knowledge_graphs(self) -> None:
        """Build economics domain knowledge graphs"""
        self.knowledge_graph_builder.process_triplets(os.path.join(self.output_dir, "triplets"))


class PolicyKGProcessor(BaseKGProcessor):
    '''Domain processor for policy documents'''
    def __init__(self, output_dir: str, model_name: str, **kwargs):
        chunks_dir = kwargs.get('chunks_dir')
        relations_file = kwargs.get('relations_file')
        triplets_file = kwargs.get('triplets_file')
        
        super().__init__(
            output_dir=output_dir,
            model_name=model_name,
            chunks_dir=chunks_dir,
            relations_file=relations_file,
            triplets_file=triplets_file
        )
    
    @property
    def domain(self) -> str:
        return "policy"
    
    def extract_doc_triplets(self, sample_size: Optional[int] = None, sectors: Optional[List[str]] = None) -> List[Dict]:
        """Extract triplets from documents with optional filtering by sector or sample size"""
        # Load relations and document index
        if os.path.exists(self.relations_file):
            self.triplet_extractor.load_relations(self.relations_file)
    
        doc_list = os.listdir(self.chunks_dir)
        # Sample documents if requested
        if sample_size and sample_size < len(doc_list):
            doc_list = random.sample(doc_list, sample_size)
            logging.info(f"Sampled {len(doc_list)} documents from {len(doc_list)} filtered documents")
        else:
            logging.info(f"Processing all {len(doc_list)} documents")
        
        # Process each document
        all_triplets = []
        for doc_file in tqdm(doc_list, desc="Processing documents"):
            # Load chunks for the document
            file_path = os.path.join(self.chunks_dir, doc_file)
            save_path = os.path.join(self.output_dir, 'triplets', doc_file)
            if os.path.exists(save_path):
                print(f"kg_process: Triplets already exist for {doc_file}, skipping...")
                continue
            
            try:
                chunks = pd.read_json(file_path).to_dict(orient='records')
            except:
                chunks = pd.read_json(file_path, lines=True).to_dict(orient='records')
            print(f"kg_process: Loaded {len(chunks)} chunks from {file_path}")
            
            chunks_dict = {chk['chunk_id']: chk for chk in chunks}
            
            # Process chunks into triplets
            doc_triplets = self.process_chunks(chunks_dict)
            
            # Save triplets to file
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(doc_triplets, f, indent=2)
            print(f"kg_process: Generated {len(doc_triplets)} triplets for {file_path}")
            print(f"kg_process: Saved triplets to {save_path}")
            
            all_triplets.extend(doc_triplets)
        
        # Save relations
        self.triplet_extractor.save_relations(self.relations_file)
        
        # Save all triplets
        triplet_file = os.path.join(self.output_dir, 'triplets.json')
        os.makedirs(os.path.dirname(triplet_file), exist_ok=True)
        with open(triplet_file, 'w') as f:
            json.dump(all_triplets, f, indent=2)
        logging.info(f"kg_process: Saved {len(all_triplets)} triplets to {triplet_file}")
        
        return all_triplets
    
    def process_chunks(self, chunks_dict: Dict[str, Dict]) -> List[Dict]:
        """Process policy chunks into triplets"""
        triplets = self.triplet_extractor.extract_triplets(chunks_dict)
        logging.info(f"kg_process: Extracted {len(triplets)} triplets")
        processed = self.triplet_extractor.process_extracted_triplets(
            triplets, chunks_dict, domain='policy'
        )
        return processed
    
    def build_knowledge_graphs(self) -> None:
        """Build policy domain knowledge graphs"""
        self.knowledge_graph_builder.process_triplets(os.path.join(self.output_dir, "triplets"))


def get_domain_processor(domain: str, **kwargs) -> BaseKGProcessor:
    """Factory function to get the appropriate domain processor"""
    processors = {
        "finance": FinanceKGProcessor,
        "economics": EconomicsKGProcessor,
        "policy": PolicyKGProcessor
    }
    
    if domain == "finance":
        return FinanceKGProcessor(**kwargs)
    elif domain == "economics":
        return EconomicsKGProcessor(**kwargs)
    elif domain == "policy":
        return PolicyKGProcessor(**kwargs)
    else:
        raise ValueError(f"Unknown domain: {domain}")


def main():
    parser = argparse.ArgumentParser(description='Knowledge Graph Construction')
    parser.add_argument('--output_dir', required=True, help='Directory to save output knowledge graph')
    parser.add_argument('--domain', choices=['finance', 'economics', 'policy'], default='finance',
                       help='Domain to process')
    parser.add_argument('--model_name', default='openai/gpt-4.1',
                       help='LLM model name')
    parser.add_argument('--mode', choices=['full', 'extract_only', 'graph_only'], default='full',
                       help='Processing mode')
    parser.add_argument('--chunks_dir', help='Directory containing document chunk files')
    parser.add_argument('--relations_file', help='Path to JSON file with relations')
    parser.add_argument('--triplets_file', help='Path to JSON file to save triplets')
    parser.add_argument('--sample_size', type=int, help='Number of documents to process')
    # First filter by sector for finance domain, then sample if requested
    parser.add_argument('--sectors', nargs='+', help='GICS Sectors to filter by (finance domain only), for more details see https://en.wikipedia.org/wiki/List_of_S%26P_500_companies. If the sector is not provided, all chunks will be processed.')
    
    args = parser.parse_args()
    
    processor = get_domain_processor(
        domain=args.domain,
        output_dir=args.output_dir,
        model_name=args.model_name,
        chunks_dir=args.chunks_dir,
        relations_file=args.relations_file,
        triplets_file=args.triplets_file
    )
    
    processor.process(args.mode, args.sample_size, args.sectors)


if __name__ == "__main__":
    main()