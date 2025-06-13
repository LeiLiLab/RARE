import logging
from typing import List, Dict, Any, Tuple
from utils.prompts import rag_prompt_system, rag_prompt_user
from utils.litellm_router_models import MODEL_LIST
import litellm
import numpy as np
import re

EMBEDDING_MODEL_NAME = 'intfloat/e5-mistral-7b-instruct'

router = litellm.Router(model_list=MODEL_LIST, num_retries=3, retry_after=5)

def get_embedding(prediction, truth):
    """
    Get embedding from vLLM-deployed model.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector
    """
    try:
        # Format the prompt for e5-mistral-7b-instruct model
        formatted_prediction = f"Instruct: Check prediction has the semantic similar meaning compared with the ground truth answer.\nPrediction: {prediction}"
        
        # Get embedding using the vLLM API via OpenAI client
        response = router.embedding(
            model=EMBEDDING_MODEL_NAME,
            input=[
                formatted_prediction,
                truth
            ],
            encoding_format="float"
        )
        
        # Extract embedding from response
        embedding_prediction = response.data[0]['embedding']
        embedding_truth = response.data[1]['embedding']
        return embedding_prediction, embedding_truth
    except Exception as e:
        logging.error(f"Error getting embedding from vLLM: {e}")
        return None

def answer_judger(pred: str, truth: str, threshold=0.9) -> bool:
    """
    Judge if the prediction matches the ground truth answer.
    Returns True if the prediction is correct, False otherwise.
    """
    norm_p = _normalize(pred)
    norm_t = _normalize(truth)
    
    # Direct string match
    if (norm_t in norm_p or norm_p in norm_t) or (norm_t == norm_p):
        return True
    
    try:
        # Try to get embeddings from vLLM
        embed_p, embed_t = get_embedding(norm_p, norm_t)
        
        if embed_p is not None and embed_t is not None:
            # Compute cosine similarity
            similarity = np.dot(embed_p, embed_t)
            if similarity >= threshold:
                # print(f"similarity: {similarity}")
                return True
            
            return False
    except Exception as e:
        logging.warning(f"Error using vLLM embeddings: {e}")
    
    return False 

def _normalize(text: str) -> str:
    """
    - lowercase
    - strip most punctuation (keep % and . for numbers / decimals)
    - collapse consecutive whitespace
    """
    text = text.lower()
    text = re.sub(r"[^0-9a-z%.\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

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
    doc_variant_type: str,
) -> bool:
    """
    Binary robustness judger 
    Args:
        prediction: Model's prediction
        truth: Ground truth answer
        can_answer_without_retrieval: Whether model can answer without retrieval
        doc_variant_type: Type of document variant
    
    Returns:
        bool: True if robust, False otherwise
    """
    is_correct = answer_judger(prediction, truth)
    no_info_phrases = [
                    "no such info", "cannot find", "not available", "not found",
                    "no information", "unable to answer", "don't have that information",
                    "not provided", "not in the context"
                ]
    prediction = prediction.lower()
    returns_no_info = any(phrase in prediction for phrase in no_info_phrases)

    # Top priority rule: If the model has parametric knowledge,
    # it must always be correct, regardless of context.
    if can_answer_without_retrieval:
        return is_correct

    if doc_variant_type in ['ground-truth-docs', 'lexical-diff-with-answer-docs']:
        # With perfect context provided, the model must be correct.
        return is_correct

    elif doc_variant_type == 'lexical-similar-no-answer-docs':
        # With misleading context that contains no answer, the model must refuse.
        return returns_no_info

    elif doc_variant_type == 'real-world-docs':
        return is_correct or returns_no_info
    # Default case for any other variant types
    return False

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
    context = ""
    for query, docs in zip(queries, docs_list):
        context_parts = []
        for i, doc in enumerate(docs):
            context_parts.append(f"[Doc {i+1}] {doc}")
        context = '\n\n'.join(context_parts)
        system = rag_prompt_system.format(domain=domain)
        user = rag_prompt_user.format(question=query, context=context)
        batch_messages.append((system, user))
    return batch_messages