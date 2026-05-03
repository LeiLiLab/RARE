import logging
import asyncio
from typing import List, Dict, Any, Tuple, Union
from utils.prompts import rag_prompt_system, rag_prompt_user
from utils.litellm_router_models import MODEL_LIST
import litellm
import numpy as np
import re
import random
import os
# litellm._turn_on_debug()

LLM_MODEL_NAME = "bedrock/us.anthropic.claude-3-haiku-20240307-v1:0"

random.seed(42)

# Single shared event loop for ALL LiteLLM async calls (precedes router creation)
EVENT_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(EVENT_LOOP)

def get_shared_loop():
    """Expose the single shared event loop to other modules."""
    return EVENT_LOOP

router = litellm.Router(model_list=MODEL_LIST, num_retries=3, retry_after=5)

def _normalize(text: str) -> str:
    """
        - lowercase
        - strip most punctuation (keep % and . for numbers / decimals)
        - collapse consecutive whitespace
    """
    # 1. lowercase
    text = text.lower()
    # 2. drop a leading $ that immediately precedes digits (currency symbol)
    text = re.sub(r"\$(?=\d)", "", text)
    # 3. drop commas NOT between digits – keeps numeric grouping commas only
    text = re.sub(r"(?<!\d),(?!\d)", " ", text)
    # 4. strip unwanted punctuation but keep numerically‐useful symbols
    text = re.sub(r"[^0-9a-z%.,\s]", " ", text)
    # 5. collapse consecutive whitespace and trim
    return re.sub(r"\s+", " ", text).strip()

def answer_judger(pred: Union[str, List[str]], truth: Union[str, List[str]]):
    """
        Return True/False if pred semantically matches ground truth.
        Now supports single input OR batched inputs.
        - If `pred` and `truth` are strings -> returns bool.
        - If both are lists of equal length -> returns List[bool].

        Internally uses `router.abatch_completion` for LLM judging on a shared event loop.
    """
    system_prompt = """
You are a fair and strict judger, given prediction and ground truth, your task is to determine if the prediction has the same/highly similar meaning as the ground truth answer. 
Return true if:
- The prediction and ground truth are semantically identical or highly similar.
- The prediction provides the same information as the ground truth.
- If ground truth is included in the prediction consider it a match.
    -  Which means if prediction not only contains the ground truth, but also contains other information, it should be considered a match.
    -  Example: prediction: "The company's revenue was $50 million in 2023" and ground truth: "$50 million" are considered the same.
Return false if:
- The prediction does not match the ground truth in meaning.
- The prediction is a refusal or does not provide an answer.
- The ground truth has more specific information than the prediction.
- If the prediction is a numeric value, it should match the ground truth numerically
    - Example #1: 120,000,000 and 120000000 are considered the same.
    - Example #2: 120,000,000 and 120 billion are considered the same.

For your output, you should only answer 'true' or 'false', no extra text.
Examples:
1. Prediction: "The company's revenue was $50 million in 2023", Ground Truth: "$50 million", Output: true
2. Prediction: "Apple Inc.", Ground Truth: "Apple", Output: true
3. Prediction: "I cannot find that information", Ground Truth: "25%", Output: false
4. Prediction: "The answer is 42", Ground Truth: "42", Output: true
5. Prediction: "The population is around 1 million", Ground Truth: "1,000,000", Output: true
6. Prediction: "Tesla", Ground Truth: "General Motors", Output: false
"""
    # Normalize and shape inputs
    is_batch = isinstance(pred, list) and isinstance(truth, list)
    preds = pred if isinstance(pred, list) else [pred]
    truths = truth if isinstance(truth, list) else [truth]

    if len(preds) != len(truths):
        raise ValueError("For batched judging, `pred` and `truth` lists must be the same length.")

    norm_ps = [_normalize(p) for p in preds]
    norm_ts = [_normalize(t) for t in truths]

    # Quick-path heuristics & refusal detection
    results: List[Union[bool, None]] = [None] * len(norm_ps)
    to_judge_indices: List[int] = []
    messages: List[List[Dict[str, str]]] = []

    for i, (norm_p, norm_t) in enumerate(zip(norm_ps, norm_ts)):
        if is_refusal(norm_p):
            results[i] = False
            continue

        if (norm_t in norm_p or norm_p in norm_t) or (norm_t == norm_p):
            results[i] = True
            continue

        # Needs LLM judgment
        to_judge_indices.append(i)
        messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Prediction: {norm_p}\nGround Truth: {norm_t}\nOutput:"},
        ])

    # If nothing needs LLM judging, return early
    if not messages:
        return results[0] if not is_batch else results  # type: ignore[return-value]

    # LLM batch call via abatch_completion on the shared loop
    try:
        async def _run_batch():
            return await router.abatch_completion(
                models=[LLM_MODEL_NAME],
                messages=messages,
                temperature=0.0,
                max_tokens=50,
            )

        raw_responses = EVENT_LOOP.run_until_complete(_run_batch())

        # raw_responses: list over requests; each item is a list over models
        for req_idx, req_row in enumerate(raw_responses):
            try:
                response = req_row[0]  # single model used
                msg = response.choices[0].message
                content_str = getattr(msg, "content", "")
                # Prefer explicit boolean token; fall back to parsed object; else False
                judged_true = False
                text = str(content_str).strip().lower()
                if "true" in text and "false" not in text:
                    judged_true = True
                elif "false" in text and "true" not in text:
                    judged_true = False
                else:
                    parsed_obj = getattr(msg, "parsed", None)
                    if isinstance(parsed_obj, dict) and "result" in parsed_obj:
                        judged_true = bool(parsed_obj["result"])

                idx = to_judge_indices[req_idx]
                results[idx] = judged_true

                # logging.error(
                #     f"LLM judger result: {judged_true}, prediction: {norm_ps[idx]}, truth: {norm_ts[idx]}"
                # )
            except Exception as inner_e:
                idx = to_judge_indices[req_idx]
                results[idx] = False
                logging.warning(f"Error parsing LLM judger response at index {idx}: {inner_e}")

    except Exception as e:
        logging.warning(f"Error using LLM batch judging: {e}")
        # Any unresolved items default to False
        for idx in to_judge_indices:
            if results[idx] is None:
                results[idx] = False

    # Return shape consistent with input
    return results[0] if not is_batch else results  # type: ignore[return-value]

def is_refusal(pred: str) -> bool:
    """
    Check if the prediction is a refusal.
    A refusal is defined as a prediction that does not provide an answer.
    """
    refusal_phrases = [
        "no such info", "cannot find", "not available", "not found",
        "no information", "unable to answer", "don't have that information",
        "not provided", "not in the context"
    ]
    pred_lower = _normalize(pred)
    return any(phrase in pred_lower for phrase in refusal_phrases)

def robust_judger(
    prediction: Union[str, List[str]],
    truth: Union[str, List[str]],
    can_answer_without_retrieval: Union[bool, List[bool]],
    doc_variant_type: Union[str, List[str]],
) -> Union[bool, List[bool]]:
    """
    Binary robustness judger (now supports batch).
    Inputs can be scalars or lists of equal length.
    Returns:
        bool or List[bool]: robust flags
    """
    # Harmonize to lists
    preds = prediction if isinstance(prediction, list) else [prediction]
    truths = truth if isinstance(truth, list) else [truth]
    cans  = can_answer_without_retrieval if isinstance(can_answer_without_retrieval, list) else [can_answer_without_retrieval] * len(preds)
    docts = doc_variant_type if isinstance(doc_variant_type, list) else [doc_variant_type] * len(preds)

    if not (len(preds) == len(truths) == len(cans) == len(docts)):
        raise ValueError("Batched robust_judger inputs must have equal lengths.")

    # Correctness via (batched) answer_judger
    is_correct_list = answer_judger(preds, truths)
    if not isinstance(is_correct_list, list):
        is_correct_list = [is_correct_list]

    results: List[bool] = []
    for p, is_corr, can_ans, dvt in zip(preds, is_correct_list, cans, docts):
        returns_no_info = (not is_corr) and is_refusal(str(p))
        # Robustness rules by doc variant type
        if dvt in ['ground-truth-docs', 'lexical-diff-with-answer-docs']:
            # Must be correct when docs contain the answer
            results.append(bool(is_corr))
        elif dvt == 'lexical-similar-no-answer-docs':
            results.append(bool(returns_no_info or (can_ans and is_corr)))
        elif dvt == 'real-world-docs':
            # Accept either correct answer or safe abstention
            results.append(bool(is_corr or returns_no_info))
        else:
            results.append(False)

    # Return scalar if scalar input
    if not isinstance(prediction, list) and not isinstance(truth, list):
        return results[0]
    return results

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