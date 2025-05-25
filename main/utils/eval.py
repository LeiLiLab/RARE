import os
import json
import logging
from typing import List, Dict, Any, Tuple
from utils.prompts import rag_prompt_system, rag_prompt_user
from utils.vllm_server_location import MODEL_SERVER_MAPPING
from openai import OpenAI
import numpy as np

# Global client for vLLM embedding model
vllm_client = None
EMBEDDING_MODEL_NAME = 'intfloat/e5-mistral-7b-instruct'

def get_vllm_client():
    """
    Initialize and return the vLLM client for the embedding model.
    """
    global vllm_client
    if vllm_client is None:
        base_url = MODEL_SERVER_MAPPING.get(EMBEDDING_MODEL_NAME)
        if not base_url:
            logging.error(f"Model {EMBEDDING_MODEL_NAME} not found in MODEL_SERVER_MAPPING")
            return None
        
        # Initialize OpenAI client for vLLM server
        vllm_client = OpenAI(
            api_key="EMPTY",
            base_url=base_url
        )
    return vllm_client

def get_embedding_from_vllm(prediction, truth):
    """
    Get embedding from vLLM-deployed model.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector
    """
    client = get_vllm_client()
    if client is None:
        return None
    
    try:
        # Format the prompt for e5-mistral-7b-instruct model
        formatted_prediction = f"Text: {prediction}\nInstruct: Retrieve semantically similar text.\nQuery: "
        
        # Get embedding using the vLLM API via OpenAI client
        response = client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=[
                formatted_prediction,
                truth
            ],
            encoding_format="float"
        )
        
        # Extract embedding from response
        embedding_prediction = response.data[0].embedding
        embedding_truth = response.data[1].embedding
        return embedding_prediction, embedding_truth
    except Exception as e:
        logging.error(f"Error getting embedding from vLLM: {e}")
        return None

def answer_judger(pred: str, truth: str, threshold=0.9) -> bool:
    """
    Judge if the prediction matches the ground truth answer.
    Returns True if the prediction is correct, False otherwise.
    """
    norm_p = pred.lower().strip()
    norm_t = truth.lower().strip()
    
    # Direct string match
    if (norm_t in norm_p or norm_p in norm_t) or (norm_t == norm_p):
        return True
    
    try:
        # Try to get embeddings from vLLM
        embed_p, embed_t = get_embedding_from_vllm(norm_p, norm_t)
        
        if embed_p is not None and embed_t is not None:
            # Compute cosine similarity
            similarity = np.dot(embed_p, embed_t)
            if similarity > threshold:
                # print(f"similarity: {similarity}")
                return True
            
            return False
    except Exception as e:
        logging.warning(f"Error using vLLM embeddings: {e}")


def retrieval_judger(retrieval_docs: List[str], answer_chunk_ids: List[str]) -> bool:
    """
    Judge if the retrieval contains the ground truth answer chunks.
    Returns True if any of the ground truth answer chunks are in the retrieval, False otherwise.
    
    Args:
        retrieval_docs: List of retrieved documents
        answer_chunk_ids: List of chunk IDs that contain the answer
    """
    # As long as one of the answer chunk IDs is in the retrieval, return True
    return bool(set(retrieval_docs) & set(answer_chunk_ids))

def robust_judger(
    prediction: str,
    truth: str,
    can_answer_without_retrieval: bool,
    doc_type: str,
    retrieval_contains_answer: bool = False
) -> bool:
    """
    Judge if the model's behavior is robust according to the definition:
    - When generator can answer the query correctly with no retrieval, it should always answer it correctly
      regardless of the retrieval (whether the retrieval is correct, incorrect or irrelevant)
    - When generator cannot answer the query correctly with no retrieval, it should be able to answer it
      correctly when given the correct retrieval, or return "no such info" when given incorrect retrieval.
    
    This strictly follows the definition in the metrics section.
    
    Args:
        prediction: Model's prediction
        truth: Ground truth answer
        can_answer_without_retrieval: Whether the model can answer correctly without retrieval
        doc_type: Type of document (ground_truth_docs, semantic_diff_with_answer_docs, etc.)
        retrieval_contains_answer: Whether the retrieval contains the ground truth answer chunks
    """
    # Check if the prediction is correct
    # print("\n--------------------------------")
    # print(f"prediction: {prediction}, truth: {truth}")
    # print(f"can_answer_without_retrieval: {can_answer_without_retrieval}")
    # print(f"doc_type: {doc_type}")
    # print(f"retrieval_contains_answer: {retrieval_contains_answer}")
    # print("--------------------------------\n")

    is_correct = answer_judger(prediction, truth)
    
    # Apply robustness criteria based on whether the model can answer without retrieval
    if can_answer_without_retrieval:
        # If the model can answer without retrieval, it should answer correctly regardless of retrieval
        # print(f"If the model can answer without retrieval, it should answer correctly regardless of retrieval (judge result: {is_correct})")
        return is_correct
    else:
        # If the model cannot answer without retrieval, its behavior depends on the document type
        if doc_type in ['ground-truth-docs', 'lexical-diff-with-answer-docs']:
            # Should answer correctly with ground truth or docs with answer
            # print(f"answer correctly with ground truth or docs with answer (judge result: {is_correct})")
            return is_correct
        elif doc_type == 'lexical-similar-no-answer-docs':
            # Should return "No such info" for docs without answer
            # print(f"Should return 'no such info' for docs without answer (judge result: {'no such info' in prediction.lower()})")
            return "no such info" in prediction.lower()
        elif doc_type == 'real-world-docs':
            # For real-world docs, check if retrieval contains the answer
            if retrieval_contains_answer:
                # If retrieval contains the answer, it should answer correctly
                # print(f"If retrieval contains the answer, it should answer correctly (judge result: {is_correct})")
                return is_correct
            else:
                # If retrieval doesn't contain the answer, it should return "no such info"
                # print(f"If retrieval doesn't contain the answer, it should return 'no such info' (judge result: {'no such info' in prediction.lower()})")
                return "no such info" in prediction.lower()

# Prepare a batch of prompts for generation
def prepare_batch_prompts(
    queries: List[str],
    docs_list: List[Any],
    domain: str
) -> List[Tuple[str, str]]:
    """
    Prepare a batch of prompts for generation.
    Returns list of (system_prompt, user_prompt) tuples.
    """
    batch_messages = []
    
    for query, docs in zip(queries, docs_list):
        # Check if docs is a list of documents (for complete retrieval sets)
        is_nested_list = isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], list)
        
        # Handle empty docs case
        if not docs:
            context = "[No documents provided]"
        else:
            # Build context from document texts
            context_parts = []
            
            if is_nested_list:
                # Handle nested list of documents (complete retrieval set)
                for i, doc in enumerate(docs):
                    if isinstance(doc, dict) and 'text' in doc:
                        # Dictionary with 'text' key
                        context_parts.append(f"[Doc {i+1}] {doc['text']}")
                    elif isinstance(doc, str):
                        # Plain string
                        context_parts.append(f"[Doc {i+1}] {doc}")
                    elif isinstance(doc, dict) and 'chunk_id' in doc:
                        # Document with chunk_id but no text (might be a reference)
                        context_parts.append(f"[Doc {i+1}] Document with ID: {doc['chunk_id']}")
                    else:
                        # Fallback for other formats
                        context_parts.append(f"[Doc {i+1}] {str(doc)}")
            else:
                # Normalize single-doc dict into list
                if not isinstance(docs, list):
                    docs = [docs]
                
                # Handle list of documents
                for i, d in enumerate(docs):
                    if isinstance(d, dict) and 'text' in d:
                        # Dictionary with 'text' key
                        context_parts.append(f"[Doc {i+1}] {d['text']}")
                    elif isinstance(d, str):
                        # Plain string
                        context_parts.append(f"[Doc {i+1}] {d}")
                    elif isinstance(d, dict) and 'chunk_id' in d:
                        # Document with chunk_id but no text (might be a reference)
                        context_parts.append(f"[Doc {i+1}] Document with ID: {d['chunk_id']}")
                    else:
                        # Fallback for other formats
                        context_parts.append(f"[Doc {i+1}] {str(d)}")
                        
            context = '\n\n'.join(context_parts)
        
        system = rag_prompt_system.format(domain=domain)
        user = rag_prompt_user.format(question=query, context=context)
        # Add instructions for structured output to the system prompt
        system_with_format = system + """\n
You MUST follow the JSON format below as your output:
{
    "cot_answer": Your detailed chain-of-thought reasoning in python str,
    "answer": Your concise final answer in python str
}
IMPORTANT: Make sure your 'answer' value is properly enclosed in quotes as a valid JSON string.

Example of correct format:
Example #1:
{
  "cot_answer": "The question asks for the net revenues generated by the Consumer Goods, Retail & Travel vertical in EPAM's Europe segment for the year ended December 31, 2024. According to the context, the Europe segment revenues by industry vertical for 2024 show that the Consumer Goods, Retail & Travel vertical generated $562,976,000 in revenues. This vertical remained the largest in the Europe segment but experienced a 5.7% decline compared to 2023. The amount $562,976,000 represents the net revenues for this vertical in the Europe segment for 2024.",
  "answer": "$562,976,000"
}

Example #2:
{
  "cot_answer": "The question asks for KEYCORP's prescribed tax-equivalent effective tax rate for fiscal year 2025. According to the context, the tax-equivalent effective rate for FY2025 is expected to be approximately between 23% and 24%. This is explicitly stated in the business outlook section under the category 'Tax-equivalent Effective Rate' for FY2025.",
  "answer": "~23% to 24%"
}
"""
        batch_messages.append((system_with_format, user))
    
    return batch_messages

# Function to save intermediate results after each batch
def save_intermediate_results(domain, model_results, output_dir, batch_num):
    """Save intermediate results after each batch to prevent data loss."""
    try:
        # Calculate current scores based on data collected so far
        intermediate_results = {}
        
        for model, scores in model_results.items():
            # Calculate metrics according to the defined formulas
            
            # 1. Overall Robustness (using flat list of all scores)
            overall_robustness = 0.0
            all_scores = scores.get('all_robustness_scores', [])
            if all_scores:
                overall_robustness = sum(all_scores) / len(all_scores)
            
            # 2. Robustness on Query
            query_robustness = 0.0
            if scores.get('query_scores', []):
                query_robustness = sum(scores.get('query_scores', [])) / len(scores.get('query_scores', []))
            
            # 3. Robustness on Document
            doc_robustness = 0.0
            if scores.get('doc_scores', []):
                doc_robustness = sum(scores.get('doc_scores', [])) / len(scores.get('doc_scores', []))
            
            # Add to intermediate results
            intermediate_results[model] = {
                'overall_robustness': overall_robustness,  # Using flat list approach
                'query_robustness': query_robustness,
                'doc_robustness': doc_robustness,
            }
            
            # 4. Robustness on Real-World Retrieval
            for retriever_name, retriever_scores in scores.get('realret_scores', {}).items():
                if retriever_scores:
                    intermediate_results[model][f'real_retrieval_robustness_{retriever_name}'] = sum(retriever_scores) / len(retriever_scores)
                else:
                    intermediate_results[model][f'real_retrieval_robustness_{retriever_name}'] = 0.0
            
            # 5. Add variant-specific scores
            for variant_name, variant_scores in scores.get('variant_scores', {}).items():
                if variant_scores:
                    intermediate_results[model][f'variant_{variant_name}_robustness'] = sum(variant_scores) / len(variant_scores)
                else:
                    intermediate_results[model][f'variant_{variant_name}_robustness'] = 0.0
            
            # 6. Add variant-doc-specific scores
            for variant_doc_key, variant_doc_scores in scores.get('variant_doc_scores', {}).items():
                if variant_doc_scores:
                    intermediate_results[model][f'variant_doc_{variant_doc_key}_robustness'] = sum(variant_doc_scores) / len(variant_doc_scores)
                else:
                    intermediate_results[model][f'variant_doc_{variant_doc_key}_robustness'] = 0.0
            
            # 7. Calculate combined retrieval robustness
            all_retriever_scores = []
            for retriever_scores in scores.get('realret_scores', {}).values():
                all_retriever_scores.extend(retriever_scores)
            
            if all_retriever_scores:
                intermediate_results[model]['real_retrieval_robustness_combined'] = sum(all_retriever_scores) / len(all_retriever_scores)
            else:
                intermediate_results[model]['real_retrieval_robustness_combined'] = 0.0
        
        # Save to file
        interim_file = os.path.join(output_dir, f"{domain}_interim_results_batch_{batch_num}.json")
        with open(interim_file, 'w', encoding='utf-8') as wf:
            json.dump(intermediate_results, wf, indent=2)
            
        # Print interim results
        print(f"\n--- INTERIM RESULTS (Batch {batch_num}) for {domain} ---")
        for model, scores in intermediate_results.items():
            model_short = model.split('/')[-1]
            print(f"\n{model_short}:")
            print(f"  overall_robustness: {scores['overall_robustness']:.4f}")
            print(f"  query_robustness: {scores['query_robustness']:.4f}")
            print(f"  doc_robustness: {scores['doc_robustness']:.4f}")
            
            # Print retriever scores
            for key in sorted(scores.keys()):
                if key.startswith('real_retrieval_robustness_'):
                    print(f"  {key}: {scores[key]:.4f}")
            
            # Print variant scores
            for key in sorted(scores.keys()):
                if key.startswith('variant_') and not key.startswith('variant_doc_'):
                    print(f"  {key}: {scores[key]:.4f}")
            
            # Print variant-doc scores
            for key in sorted(scores.keys()):
                if key.startswith('variant_doc_'):
                    print(f"  {key}: {scores[key]:.4f}")
        
        logging.info(f"Saved interim results after batch {batch_num} for {domain}")
    except Exception as e:
        logging.error(f"Error saving interim results: {e}")