import os
import json
import ast
import logging
from typing import Dict, List, Any
from io import StringIO

import pandas as pd
import numpy as np
from pydantic import BaseModel
from abc import ABC, abstractmethod
import litellm
import torch
from sentence_transformers import SentenceTransformer
import traceback

from utils.prompts import (
    econ_triplets_prompt_system, 
    econ_triplets_prompt_user, 
    econ_relation_similarity_prompt,
    finance_triplets_prompt_system, 
    finance_triplets_prompt_user,
    finance_relation_similarity_prompt,
    policy_triplets_prompt_system,
    policy_triplets_prompt_user,
    policy_relation_similarity_prompt
)
from utils.api_keys import OPENAI_API_KEY

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

logging.basicConfig(filename='logs/kg_triplet_extraction.log', filemode='w', level=logging.INFO)
litellm.global_disable_no_log_param = True

class Triplet(BaseModel): 
    entity_1: str
    relation: str
    entity_2: str
    answer_chunk_id: str
    source_sentence: str

class TripletList(BaseModel):
    triplets: list[Triplet]

class BaseTripletExtractor(ABC):
    """Abstract base class for triplet extraction across different domains"""
    
    def __init__(self, model_name: str, embedding_model_name: str = 'intfloat/e5-mistral-7b-instruct'):
        """
        Initialize triplet extractor with specified models
        
        Args:
            model_name: Name of the LLM to use for extraction
            embedding_model_name: Name of the embedding model for similarity matching
        """
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        
        # Initialize the embedding model and tokenizer
        logging.info(f"kg_triplet_extraction: Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, model_kwargs={"torch_dtype": torch.bfloat16})
        self.embedding_model.max_seq_length = 4096
        self.relation_data = {"relations": []}
        # It store the embeddings of all relations
        self.relation_embeddings = None
        
    def load_relations(self, relations_file: str) -> None:
        """Load existing relations from file"""
        if os.path.exists(relations_file):
            try:
                df = pd.read_json(relations_file)
                if 'relations' in df.columns:
                    self.relation_data = {"relations": df['relations'].tolist()}
                    if self.relation_data["relations"]:
                        self.relation_embeddings = self.embedding_model.encode(self.relation_data["relations"])
                        logging.info(f"kg_triplet_extraction: Loaded {len(self.relation_data['relations'])} existing relations")
            except Exception as e:
                logging.error(f"kg_triplet_extraction: Error loading relations file: {e}")
                self.relation_data = {"relations": []}
    
    def save_relations(self, relations_file: str) -> None:
        """Save relations to file"""
        try:
            df = pd.DataFrame(self.relation_data)
            df.to_json(relations_file, orient='columns', indent=2)
            logging.info(f"kg_triplet_extraction: Relations database updated with {len(self.relation_data['relations'])} unique relations")
        except Exception as e:
            logging.error(f"kg_triplet_extraction: Error saving relations file: {e}")
            
    def normalize_text(self, text: str) -> str:
        return text.lower().strip().replace("\n\n", " ")
    
    def normalize_relation(self, relation: str, similarity_threshold: float = 0.9, domain: str = 'finance') -> str:
        """
        Normalize a relation by finding similar existing relations or adding it as new
        
        Args:
            relation: The relation to normalize
            similarity_threshold: Threshold for considering relations as similar
            domain: The domain of the relation, it will be used to select the correct prompt for e5-mistral-7b-instruct
            
        Returns:
            Normalized relation string
        """
        normalized_relation = self.normalize_text(relation)
        
        # Check for exact match
        for existing_relation in self.relation_data["relations"]:
            if normalized_relation == self.normalize_text(existing_relation):
                # logging.info(f"kg_triplet_extraction: Found exact match for relation: {existing_relation}")
                return existing_relation
        if domain == "finance":
            prompt = finance_relation_similarity_prompt
        elif domain == "economics":
            prompt = econ_relation_similarity_prompt
        elif domain == "policy":
            prompt = policy_relation_similarity_prompt
        else:
            raise ValueError(f"Unknown domain: {domain}")
        # Check similarity with existing relations
        if self.relation_embeddings is not None and len(self.relation_embeddings) > 0:
            # Calculate similarity
            similarities = self.embedding_model.encode(relation, prompt=prompt) @ self.relation_embeddings.T
            # logging.info(f"kg_triplet_extraction: Similarities: {similarities}, relation: {relation}, closest relation: {self.relation_data['relations'][np.argmax(similarities)]}")
            if max(similarities) > similarity_threshold:
                best_idx = np.argmax(similarities)
                logging.info(f"kg_triplet_extraction: Found similar relation: {self.relation_data['relations'][best_idx]}")
                return self.relation_data["relations"][best_idx]
        
        # Add as new relation
        self.relation_data["relations"].append(normalized_relation)
        
        # Update embeddings
        if self.relation_embeddings is not None and self.relation_embeddings.size > 0:
            new_embedding = self.embedding_model.encode([normalized_relation])
            self.relation_embeddings = np.vstack([self.relation_embeddings, new_embedding])
        else:
            # Initial embedding
            self.relation_embeddings = self.embedding_model.encode([normalized_relation])
        
        return normalized_relation
    
    def normalize_entity(self, entity: str) -> str:
        """
        Normalize an entity while preserving critical information
        
        Args:
            entity: The entity to normalize
            
        Returns:
            Normalized entity string
        """
        normalized = self.normalize_text(entity)
        
        # Remove common company suffixes
        company_suffixes = [" inc", " inc.", " corporation", " corp", " corp.", 
                           " ltd", " ltd.", " llc", " limited", " company", 
                           " co", " co.", " group", " holdings", " holding"]
        
        for suffix in company_suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break
        
        # Remove trailing commas and periods
        normalized = normalized.rstrip('.,')
        
        # Remove multiple spaces
        normalized = ' '.join(normalized.split())
        
        return normalized.strip()
    
    @abstractmethod
    def extract_triplets(self, chunks_dict: Dict = None) -> List[Dict]:
        """
        Extract triplets from text
        
        Args:
            chunks_dict: Dictionary of chunks keyed by chunk_id (optional)
            
        Returns:
            List of triplet dictionaries
        """
        pass
    
    def locate_source_chunks(self, sentence: str, chunks_dict: Dict[str, Dict]) -> str:
        """
        Find chunk IDs that contain the source sentence
        
        Args:
            sentence: Source sentence to locate
            chunks_dict: Dictionary of chunks keyed by chunk_id
            
        Returns:
            List of chunk IDs containing the sentence
        """
        normalized_sentence = self.normalize_text(sentence)
        
        for chunk_id, chunk_data in chunks_dict.items():
            chunk_text = self.normalize_text(chunk_data.get("text", ""))
            if normalized_sentence in chunk_text:
                return chunk_id
        logging.info(f"kg_triplet_extraction (locate_source_chunks): No chunk found for sentence: {normalized_sentence}")
        logging.info(f"kg_triplet_extraction (locate_source_chunks): Chunk text: {chunk_text}")
        return "INVALID_CHUNK_ID"
    
    def process_extracted_triplets(self, triplets: List[Dict], chunks_dict: Dict = None, domain: str = None) -> List[Dict]:
        """
        Process extracted triplets: normalize entities and relations,
        locate answer chunk id and add metadata
        
        Args:
            triplets: List of extracted triplet dictionaries
            chunks_dict: Dictionary of chunks keyed by chunk_id
            
        Returns:
            List of processed triplet dictionaries
        """
        processed_triplets = []
        for idx, triplet in enumerate(triplets):
            entity_1 = self.normalize_entity(triplet["entity_1"])
            relation = self.normalize_relation(triplet["relation"], domain=domain)
            entity_2 = self.normalize_entity(triplet["entity_2"])
            source_sentence = triplet.get("source_sentence", "")
            
            # Verify and potentially correct the answer_chunk_id if chunks_dict is provided
            if chunks_dict and source_sentence:
                # Check if the source_sentence is actually in the answer_chunk_id
                correct_chunk_id = self.locate_source_chunks(source_sentence, chunks_dict)
                if correct_chunk_id != "INVALID_CHUNK_ID":
                    answer_chunk_id = correct_chunk_id
                    # logging.info(f"kg_triplet_extraction: Corrected answer_chunk_id for triplet {idx}: {answer_chunk_id}")
                    if domain == "finance":
                        triplet_id = f"{triplet.get('metadata', {}).get('domain', '')}_{triplet.get('metadata', {}).get('cik', '')}_{triplet.get('metadata', {}).get('filing_date', '')}_triplet_{idx}"
                    elif domain == "economics":
                        triplet_id = f"{triplet.get('metadata', {}).get('domain', '')}_{triplet.get('metadata', {}).get('file_name', '')}_triplet_{idx}"
                    elif domain == "policy":
                        triplet_id = f"{triplet.get('metadata', {}).get('domain', '')}_{triplet.get('metadata', {}).get('file_name', '')}_triplet_{idx}"
                    
                    processed_triplet = {
                        "triplet_id": triplet_id,
                        "entity_1": entity_1,
                        "relation": relation,
                        "entity_2": entity_2,
                        "answer_chunk_id": answer_chunk_id,
                        "source_sentence": source_sentence,
                        "metadata": triplet.get("metadata", {})
                    }
                    processed_triplets.append(processed_triplet)
                else:
                    logging.info(f"kg_triplet_extraction: No chunk found for sentence: {source_sentence}")
                    continue
            
        return processed_triplets

class FinanceTripletExtractor(BaseTripletExtractor):
    """Triplet extractor for finance domain"""
    
    def extract_triplets(self, chunks_dict: Dict = None) -> List[Dict]:
        """
        Extract triplets from finance text using LLM
        
        Args:
            chunks_dict: Dictionary of chunks keyed by chunk_id
            
        Returns:
            List of triplet dictionaries
        """
        
        try:
            # Filter chunks for this company/CIK
            company_chunks = []
            for chunk_id, chunk_data in chunks_dict.items():
                chunk_metadata = chunk_data.get('metadata', {})
                cik = chunk_metadata.get('cik', '')
                filing_date = chunk_metadata.get('filing_date', '')
                if (cik and filing_date):
                    # Add chunk_id to the chunk data for later reference
                    chunk_with_id = chunk_data.copy()
                    chunk_with_id['chunk_id'] = chunk_id
                    company_chunks.append(chunk_with_id)
            
            # Process chunks in groups of 5
            all_triplets = []
            chunk_group_size = 5
            target_triplet_count = 150            

            logging.info(f"kg_triplet_extraction: Number of chunks: {len(company_chunks)}")
            for i in range(0, len(company_chunks), chunk_group_size):
                # Get 5 consecutive chunks
                chunk_group = company_chunks[i:i+chunk_group_size]
                chunk_texts = [chunk['text'] for chunk in chunk_group]
                chunk_ids = [chunk['chunk_id'] for chunk in chunk_group]
                # Add chunk_id into the text
                combined_text = "\n\n".join([f"Chunk ID: {chunk_id}\n{chunk_text}" for chunk_id, chunk_text in zip(chunk_ids, chunk_texts)])
                # Since the triplets are extracted from the same company, we can use the metadata of the first chunk
                metadata = chunk_group[0]['metadata']
                # Extract triplets from this chunk group
                response = litellm.completion(
                    model=self.model_name,
                    messages=[
                        {   "role": "system", 
                            "content": finance_triplets_prompt_system
                        },
                        {
                            "role": "user",
                            "content": finance_triplets_prompt_user.format(chunk_text=combined_text, metadata=metadata, number_of_triplets_per_run=target_triplet_count // chunk_group_size)
                        }],
                    temperature=0.2,
                    max_tokens=8000,
                    response_format=TripletList
                )
                
                content = response.choices[0].message.content
                # logging.info(f"kg_triplet_extraction: Content: {content}")
                # logging.info(f"kg_triplet_extraction: Content type: {type(content)}")
                group_triplets = ast.literal_eval(content)
                group_triplets = [triplet for triplet in group_triplets['triplets']]
                
                # Add metadata and source chunk IDs to each triplet
                for triplet in group_triplets:
                    triplet['metadata'] = metadata
                all_triplets.extend(group_triplets)
                
                # If we have enough triplets, stop processing more chunks
                if len(all_triplets) >= target_triplet_count:
                    break
            
            # Process the triplets (normalize entities/relations, verify chunk IDs)
            processed_triplets = self.process_extracted_triplets(all_triplets, chunks_dict, domain='finance')
            
            return processed_triplets
                
        except Exception as e:
            logging.error(f"kg_triplet_extraction: Error extracting entities and relations from chunks: {traceback.format_exc()}")
            return []

class EconomicsTripletExtractor(BaseTripletExtractor):
    """Triplet extractor for economics domain - placeholder for future implementation"""
    def extract_triplets(self, chunk_dict: Dict[str, Dict]) -> List[Dict]:
        """
        Extract triplets from economy-related text
        
        Args:
            chunk_dict: Dictionary of chunks keyed by chunk_id
        
        Returns:
            List of triplet dictionaries
        """
        # Extract the document metadata
        chunk_dict = list(chunk_dict.values())
        metadata = chunk_dict[0]['metadata']
        country_name = metadata.get('file_country', '')
        survey_year = metadata.get('file_year', '')
        logging.info(f"kg_triplet_extraction: country_name: {country_name}, survey_year: {survey_year}, total chunks: {len(chunk_dict)}")
        
        # Process chunks in groups of 5
        all_triplets = []
        chunk_group_size = 5
        target_triplet_count = 300
        for i in range(0, len(chunk_dict), chunk_group_size):
            # Get 5 consecutive chunks
            chunk_group = chunk_dict[i:i+chunk_group_size]
            chunk_texts = [chunk['text'] for chunk in chunk_group]
            chunk_ids = [chunk['chunk_id'] for chunk in chunk_group]
            # Add chunk_id into the text
            combined_text = "\n\n".join([f"Chunk ID: {chunk_id}\n{chunk_text}" for chunk_id, chunk_text in zip(chunk_ids, chunk_texts)])
            # Extract triplets from this chunk group
            response = litellm.completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": econ_triplets_prompt_system},
                    {"role": "user", "content": econ_triplets_prompt_user.format(chunk_text=combined_text, country_name=country_name, survey_year=survey_year)}
                ],
                temperature=0.2,
                max_tokens=8000,
                response_format=TripletList
            )
            
            content = response.choices[0].message.content.strip()
            try:
                df = pd.read_json(StringIO(content))
                group_triplets = df['triplets'].tolist() if 'triplets' in df.columns else []
                
                # Add metadata and source chunk IDs to each triplet
                for triplet in group_triplets:
                    triplet["metadata"] = {
                        'domain': metadata.get('domain', ''),
                        'file_name': metadata.get('file_name', ''),
                        'file_type': metadata.get('file_type', ''),
                        'file_country': metadata.get('file_country', ''),
                        'file_year': metadata.get('file_year', ''),
                    }
                
                all_triplets.extend(group_triplets)
                
                # If we have enough triplets, stop processing more chunks
                if len(all_triplets) >= target_triplet_count:
                    break
            except Exception as e:
                logging.error(f"kg_triplet_extraction: Error parsing JSON: {e}")
                return []
        
        logging.info(f"kg_triplet_extraction: Total triplets extracted: {len(all_triplets)}")
        return all_triplets

class PolicyTripletExtractor(BaseTripletExtractor):
    """Triplet extractor for policy domain - placeholder for future implementation"""
    def extract_triplets(self, chunk_dict: Dict[str, Dict]) -> List[Dict]:
        """
        Extract triplets from policy-related text
        
        Args:
            chunk_dict: Dictionary of chunks keyed by chunk_id
        
        Returns:
            List of triplet dictionaries
        """
        # Extract the document metadata
        chunk_dict = list(chunk_dict.values())
        metadata = chunk_dict[0]['metadata']
        grantee_name = metadata.get('file_grantee', '')
        year = metadata.get('file_year', '')
        logging.info(f"kg_triplet_extraction: grantee_name: {grantee_name}, year: {year}, total chunks: {len(chunk_dict)}")
        
        # Process chunks in groups of 5
        all_triplets = []
        chunk_group_size = 5
        target_triplet_count = 400
        for i in range(0, len(chunk_dict), chunk_group_size):
            # Get 5 consecutive chunks
            chunk_group = chunk_dict[i:i+chunk_group_size]
            chunk_texts = [chunk['text'] for chunk in chunk_group]
            chunk_ids = [chunk['chunk_id'] for chunk in chunk_group]
            combined_text = "\n\n".join([f"Chunk ID: {chunk_id}\n{chunk_text}" for chunk_id, chunk_text in zip(chunk_ids, chunk_texts)])
            
            # Extract triplets from this chunk group
            response = litellm.completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": policy_triplets_prompt_system},
                    {"role": "user", "content": policy_triplets_prompt_user.format(chunk_text=combined_text, grantee_name=grantee_name, year=year)}
                ],
                temperature=0.2,
                max_tokens=8000,
                response_format=TripletList
            )
            
            content = response.choices[0].message.content.strip()
            try:
                df = pd.read_json(StringIO(content))
                group_triplets = df['triplets'].tolist() if 'triplets' in df.columns else []
                
                # Add metadata and source chunk IDs to each triplet
                for triplet in group_triplets:
                    triplet["metadata"] = {
                        'domain': metadata.get('domain', ''),
                        'file_name': metadata.get('file_name', ''),
                        'file_type': metadata.get('file_type', ''),
                        'file_grantee': metadata.get('file_grantee', ''),
                        'file_state': metadata.get('file_state', ''),
                        'file_year': metadata.get('file_year', ''),
                    }
                
                all_triplets.extend(group_triplets)
                
                # If we have enough triplets, stop processing more chunks
                if len(all_triplets) >= target_triplet_count:
                    break
            except Exception as e:
                logging.error(f"kg_triplet_extraction: Error parsing JSON: {e}")
                return []
        
        logging.info(f"kg_triplet_extraction: Total triplets extracted: {len(all_triplets)}")
        return all_triplets


    # def process_extracted_triplets(self, triplets: List[Dict]) -> List[Dict]:
    #     """
    #     Process extracted triplets: normalize entities and relations,
    #     locate answer chunk id and add metadata
        
    #     Args:
    #         triplets: List of extracted triplet dictionaries
            
    #     Returns:
    #         List of processed triplet dictionaries
    #     """
    #     processed_triplets = []
    #     for idx, triplet in enumerate(triplets):
    #         entity_1 = self.normalize_entity(triplet["entity_1"])
    #         relation = self.normalize_relation(triplet["relation"])
    #         entity_2 = self.normalize_entity(triplet["entity_2"])
                
    #         answer_chunk_id = triplet.get("answer_chunk_id", "")
            
    #         metadata = triplet.get("metadata", {})
    #         triplet_id = f"{metadata.get('domain', '')}_{metadata.get('file_name', '')}_triplet_{idx}"
    #         processed_triplet = {
    #             "triplet_id": triplet_id,
    #             "entity_1": entity_1,
    #             "relation": relation,
    #             "entity_2": entity_2,
    #             "answer_chunk_id": answer_chunk_id,
    #             "metadata": metadata
    #         }
            
    #         processed_triplets.append(processed_triplet)
            
    #     return processed_triplets
    

def get_triplet_extractor(domain: str, model_name: str, **kwargs) -> BaseTripletExtractor:
    """Factory function to get the appropriate triplet extractor for a domain"""
    if domain == "finance":
        return FinanceTripletExtractor(model_name, **kwargs)
    elif domain == "economics":
        return EconomicsTripletExtractor(model_name, **kwargs)
    elif domain == "policy":
        return PolicyTripletExtractor(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown domain: {domain}")