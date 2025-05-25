import os
import json
import argparse
import logging
import asyncio
import concurrent.futures
import time
from typing import List, Dict, Any
from tqdm.auto import tqdm as tqdm_auto
from rag_pipeline import load_chunks
from utils.vllm_inference import vllm_inference
from utils.api_keys import OPENAI_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME, OPENROUTER_API_KEY
from utils.vllm_server_location import MODEL_SERVER_MAPPING
from utils.eval import answer_judger, robust_judger, retrieval_judger, prepare_batch_prompts, save_intermediate_results
from openai import AsyncOpenAI
import sys
import litellm
import re
import httpx

# Set environment variables for API keys
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
os.environ['AWS_REGION_NAME'] = AWS_REGION_NAME
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

DEFAULT_INFERENCE_ENGINE = "auto"  # Can be "vllm", "litellm", or "auto"

# Only used when inference_engine is "auto"
def auto_detect_model_type(model_name: str) -> str:
    """
    Automatically detect the appropriate inference engine for a model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        String indicating which inference engine to use ("vllm" or "litellm")
    """
    if model_name in MODEL_SERVER_MAPPING or model_name.startswith('openai/'):
        return "vllm"
    elif model_name.startswith('anthropic/') \
        or model_name.startswith('bedrock/') \
        or model_name.startswith('openrouter/'):
        return "litellm"
    # Default to vLLM for open source models
    elif '/' in model_name:
        return "vllm"
    else:
        logging.warning(f"Could not automatically determine inference engine for {model_name}. " 
                        f"Defaulting to litellm. Specify --inference_engine explicitly if needed.")
        return "litellm"

def is_vllm_model(model_name: str, inference_engine: str = None) -> bool:
    """
    Determine if a model should use vLLM inference.
    
    Args:
        model_name: Name of the model
        inference_engine: Override the global inference engine setting
    
    Returns:
        True if vLLM inference should be used, False otherwise
    """
    engine = inference_engine or DEFAULT_INFERENCE_ENGINE
    
    if engine == "vllm":
        return True
    elif engine == "litellm":
        return False
    else:  # "auto"
        return auto_detect_model_type(model_name) == "vllm"

def is_api_model(model_name: str, inference_engine: str = None) -> bool:
    """
    Determine if a model should use API-based inference.
    
    Args:
        model_name: Name of the model
        inference_engine: Override the global inference engine setting
        
    Returns:
        True if API inference should be used, False otherwise
    """
    engine = inference_engine or DEFAULT_INFERENCE_ENGINE
    
    if engine == "litellm":
        return True
    elif engine == "vllm":
        return False
    else:  # "auto"
        return auto_detect_model_type(model_name) == "litellm"

# Run batch generation using litellm (for API models)
def run_batch_generation_litellm(
    queries: List[str],
    docs_list: List[Any],
    model: str,
    domain: str
) -> List[str]:
    """
    Run generation in batch for a single API model using litellm.
    
    Args:
        queries: List of queries to process
        docs_list: List of document sets matching the queries
        model: Model to use
        domain: Domain for prompt formatting
    Returns:
        List of generated answers
    """
    # Prepare batch prompts
    batch_messages = prepare_batch_prompts(queries, docs_list, domain)
    batch_messages = [[{"role": "system", "content": system_with_format}, {"role": "user", "content": user}] for system_with_format, user in batch_messages]
    
    try:
        logging.info(f"Starting batch inference for API model: {model}")
        
        # Run batch completion
        responses = litellm.batch_completion(
            model=model,
            messages=batch_messages,
            temperature=0.0,
            max_tokens=1024,
            top_p=1.0,
        )
        
        logging.info(f"Completed batch inference for API model: {model}")
        
        # Process responses
        processed_responses = []
        for resp in responses:
            try:
                # Extract the content from the response
                content = resp.choices[0].message.content
                # Try to parse as JSON
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "answer" in parsed:
                        processed_responses.append(parsed["answer"])
                    else:
                        processed_responses.append(content)
                except json.JSONDecodeError:
                    # Extract the "answer" content from the response
                    logging.error(f"JSON Decode Error")
                    answer_pattern = r'"answer"\s*:\s*(("[^"]*")|([^,}\s][^,}]*))'
                    match = re.search(answer_pattern, content)
                    logging.error(f'Match result: {match.group(1).strip()}')
                    if match:
                        answer_value = match.group(1).strip()
                        if answer_value.startswith('"') and answer_value.endswith('"'):
                            answer_value = answer_value[1:-1]
                        logging.error(f'Final answer: {answer_value}')
                        processed_responses.append(answer_value)
                    else:
                        processed_responses.append(content)
            except Exception as e:
                logging.error(f"Error processing response: {e}")
                # Fallback to the raw content
                processed_responses.append(str(resp))
        
        return processed_responses
        
    except Exception as e:
        logging.error(f"Error in batch generation with litellm: {e}")
        # Return empty results for error handling
        return ["Error in generation"] * len(queries)

# Run parallel generation across multiple vLLM models
async def run_parallel_generation_vllm(
    queries: List[str],
    docs_list: List[Any],
    models: List[str],
    domain: str
) -> Dict[str, List[str]]:
    """
    Run generation in parallel for multiple vLLM models.
    
    Args:
        queries: List of queries to process
        docs_list: List of document sets matching the queries
        models: List of models to use
        domain: Domain for prompt formatting
    Returns:
        Dictionary mapping model names to lists of generated answers
    """
    # Prepare batch prompts once
    batch_messages = prepare_batch_prompts(queries, docs_list, domain)
    
    # Dictionary to store results for each model
    all_results = {}
    
    # Create clients for all models
    clients = {}
    try:
        for model in models:
            if "openai/" in model:
                http_client = httpx.AsyncClient(
                    limits=httpx.Limits(max_keepalive_connections=100, max_connections=100),
                    timeout=httpx.Timeout(timeout=300.0),
                    http2=True
                )
                clients[model] = AsyncOpenAI(
                    api_key=OPENAI_API_KEY,
                    base_url="https://api.openai.com/v1",
                    http_client=http_client
                )
            else:
                clients[model] = AsyncOpenAI(
                    api_key='EMPTY', 
                    base_url=MODEL_SERVER_MAPPING[model]
                )

        # Function to run a single model's inference
        def run_model_inference(model):
            try:
                logging.info(f"Starting inference for vLLM model: {model}")
                client = clients.get(model)
                if not client:
                    logging.error(f"No client available for model {model}")
                    return model, ["Error: No client available"] * len(queries)
                if model.startswith("openai/"):
                    model_name = model.split("/")[1]
                    requests_per_minute = 5000
                else:
                    model_name = model
                    requests_per_minute = 200
                responses = vllm_inference(
                    client=client,
                    prompts=batch_messages,
                    model=model_name,
                    temperature=0.0,
                    max_tokens=1024,
                    top_p=1.0,
                    requests_per_minute=requests_per_minute,
                    num_responses_per_prompt=1
                )
                
                logging.info(f"Completed inference for vLLM model: {model}")
                
                # Process responses
                processed_responses = []
                for resp in responses:
                    try:
                        if isinstance(resp, dict) and "answer" in resp:
                            processed_responses.append(resp["answer"])
                    except Exception as e:
                        logging.error(f"Error processing response: {e}")
                        # Extract the "answer" content from the response
                        answer_pattern = r'"answer"\s*:\s*(("[^"]*")|([^,}\s][^,}]*))'
                        match = re.search(answer_pattern, resp)
                        try:
                            logging.error(f"Error Matched result: {match.group(1).strip()}")
                            if match:
                                answer_value = match.group(1).strip()
                                if answer_value.startswith('"') and answer_value.endswith('"'):
                                    answer_value = answer_value[1:-1]
                                processed_responses.append(str(answer_value))
                        except Exception as e:
                            logging.error(f"Error: regex does not matched")
                            logging.error(f"Return original response: {resp}")
                            processed_responses.append(str(resp))
                
                return model, processed_responses
            except Exception as e:
                logging.error(f"Error running inference for vLLM model {model}: {e}")
                return model, ["Error during inference"] * len(queries)
        
        # Run all models in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = [executor.submit(run_model_inference, model) for model in models]
            for future in concurrent.futures.as_completed(futures):
                model, responses = future.result()
                all_results[model] = responses
    except Exception as e:
        logging.error(f"Error in parallel generation: {e}")
        # Return empty results for error handling
        for model in models:
            all_results[model] = ["Error in generation"] * len(queries)
    # finally:
    #     # Clean up clients to avoid event loop is closed errors
    #     for client in clients.values():
    #         try:
    #             await client.close()
    #         except Exception as e:
    #             logging.error(f"Error closing client: {e}")

    return all_results

# Unified function to run generation for both vLLM and API models
async def run_unified_generation(
    queries: List[str],
    docs_list: List[Any],
    models: List[str],
    domain: str,
    inference_engine: str = None
) -> Dict[str, List[str]]:
    """
    Run generation for both vLLM and API models.
    
    Args:
        queries: List of queries to process
        docs_list: List of document sets matching the queries
        models: List of models to use
        domain: Domain for prompt formatting
        inference_engine: Override the inference engine setting for all models
    Returns:
        Dictionary mapping model names to lists of generated answers
    """
    all_results = {}
    
    # Separate models into vLLM and API models based on inference_engine setting
    vllm_models = [model for model in models if is_vllm_model(model, inference_engine)]
    api_models = [model for model in models if is_api_model(model, inference_engine)]
    
    # Log which models are using which inference method
    if vllm_models:
        logging.info(f"Using vLLM inference for: {', '.join(vllm_models)}")
    if api_models:
        logging.info(f"Using API inference for: {', '.join(api_models)}")
    
    # Run vLLM models in parallel if any exist
    if vllm_models:
        vllm_results = await run_parallel_generation_vllm(
            queries=queries,
            docs_list=docs_list,
            models=vllm_models,
            domain=domain
        )
        all_results.update(vllm_results)
    
    # Run API models one by one
    for model in api_models:
        try:
            model_responses = run_batch_generation_litellm(
                queries=queries,
                docs_list=docs_list,
                model=model,
                domain=domain
            )
            # print(f"\nmodel_responses: {model_responses}")
            all_results[model] = model_responses
        except Exception as e:
            logging.error(f"Error running API model {model}: {e}")
            all_results[model] = ["Error during API inference"] * len(queries)
    
    return all_results

def evaluate_robustness(
    records: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    retrieval_results_file: str,
    generation_models: List[str],
    output_dir: str,
    domain: str,
    batch_size: int = 5,
    inference_engine: str = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute robustness metrics over the dataset using the robust_judger.
    Load retrieval results from file instead of running retrieval during evaluation.
    
    Args:
        records: List of records to evaluate
        chunks: List of chunks to use for evaluation
        retrieval_results_file: Path to retrieval results file
        generation_models: List of models to evaluate
        output_dir: Directory to save results
        domain: Domain for prompt formatting
        batch_size: Batch size for processing
        inference_engine: Inference engine to use ("vllm", "litellm", or "auto")
    
    Returns:
        Dictionary mapping model names to their robustness scores.
    """
    # Map chunk_id to chunk text for fast lookup
    id2chunk = {c['chunk_id']: c for c in chunks}
    
    # Load retrieval results
    with open(retrieval_results_file, 'r', encoding='utf-8') as f:
        retrieval_results = json.load(f)
    
    # Results for each model
    model_results = {}
    
    # Initialize model results structure for all models
    for model in generation_models:
        model_results[model] = {
            'all_robustness_scores': [],      # Store all f(phi(q',d'),a) values in a flat list
            'query_scores': [],               # Query robustness
            'doc_scores': [],                 # Document robustness
            'realret_scores': {},             # Real-world retrieval robustness by retriever
            'variant_scores': {},             # Store scores by query variant name
            'variant_doc_scores': {}          # Store scores by variant and document type
        }
    
    # Main progress bar for overall batch processing
    total_batches = (len(records) + batch_size - 1) // batch_size
    main_pbar = tqdm_auto(total=total_batches, desc="Processing batches", position=0)
    
    for batch_start in range(0, len(records), batch_size):
        batch_records = records[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size
        
        # Collect all query-doc combinations for this batch
        batch_queries = []
        batch_docs = []
        batch_metadata = []  # Store query_id, doc_type, etc. for result processing
        
        # Progress bar for record preparation within the batch
        prep_pbar = tqdm_auto(total=len(batch_records), desc="Preparing queries", position=1, leave=False)
        
        for rec in batch_records:
            q = rec['question']
            ans = rec['answer']
            gt_ids = rec.get('answer_chunk_ids', [])
            rec_domain = rec['metadata']['domain']
            query_id = rec.get('id', hash(q))
            
            # Skip if query not in retrieval results
            if str(query_id) not in retrieval_results:
                logging.warning(f"Query ID {query_id} not found in retrieval results, skipping")
                continue
            
            # Get result data for this query
            query_results = retrieval_results[str(query_id)]
            
            # Ground-truth docs
            gt_docs = [id2chunk[c] for c in gt_ids]
            
            # Query Perturbation
            q_variants = [q]  # Include original query
            variant_names = ["original"]  # Name for original query
            
            for variant_name, variant_text in query_results["variants"].items():
                q_variants.append(variant_text)
                variant_names.append(variant_name)
            
            # Document Perturbation
            lexical_similar_no_answer_docs = rec.get('document_perturbations', {}).get('regex_delete', [])
            
            lexical_diff_with_answer_docs = rec.get('document_perturbations', {}).get('back_translation', [])
            
            # Get real retrieval results for each retriever
            real_docs_by_retriever = {}
            for retriever_name, retriever_data in query_results["retrievers"].items():
                real_docs_by_retriever[retriever_name] = retriever_data["original"]
                
                # Initialize accumulator for this retriever in all models
                for model in generation_models:
                    if retriever_name not in model_results[model]['realret_scores']:
                        model_results[model]['realret_scores'][retriever_name] = []
            
            # Prepare empty query test (to check if model can answer without retrieval)
            batch_idx = len(batch_queries)  # Current index for reference
            batch_queries.append(q)
            batch_docs.append([])
            batch_metadata.append({
                'record_id': query_id,
                'answer': ans,
                'doc_type': 'empty',
                'query_variant': 0,
                'variant_name': 'original',
                'domain': rec_domain,
                'question': q,  # Add the original question for reference
                'gt_ids': gt_ids,  # Add ground truth chunk IDs
                'batch_idx': batch_idx,  # Store the index for reference
                'is_for_metric': False  # This is just to check if model can answer without retrieval, not for metric
            })
            
            # Generate all combinations of query variants and document types
            # Empty list to track all doc variants (for all combinations)
            all_doc_variants = []
            
            # 1. Ground truth docs
            all_doc_variants.append(('ground-truth-docs', gt_docs))
            
            # 2. lexical similar no answer docs
            all_doc_variants.append(('lexical-similar-no-answer-docs', lexical_similar_no_answer_docs))
            
            # 3. lexical different with answer docs
            all_doc_variants.append(('lexical-diff-with-answer-docs', lexical_diff_with_answer_docs))
            
            # 4. Real-world retrieval docs
            for retriever_name, real_docs in real_docs_by_retriever.items():
                # Check if the complete retrieval set contains any answer
                complete_retrieval_contains_answer = retrieval_judger(real_docs, gt_ids)
                doc_type = f'real-world-docs_{retriever_name}'
                
                # Add this as a distinct document type with the complete set
                all_doc_variants.append((doc_type, real_docs, {
                    'retriever': retriever_name,
                    'retrieval_contains_answer': complete_retrieval_contains_answer
                }))
            
            # Process all combinations of query variants and document types
            for q_idx, (qv, variant_name) in enumerate(zip(q_variants, variant_names)):
                for doc_info in all_doc_variants:
                    if len(doc_info) == 2:
                        doc_type, doc = doc_info
                        extra_metadata = {}
                    else:
                        doc_type, doc, extra_metadata = doc_info
                    
                    # Add this query-document combination
                    batch_idx = len(batch_queries)
                    batch_queries.append(qv)
                    batch_docs.append(doc)
                    
                    metadata = {
                        'record_id': query_id,
                        'answer': ans,
                        'doc_type': doc_type.split('_')[0] if '_' in doc_type else doc_type,  # Extract base doc type
                        'query_variant': q_idx,
                        'variant_name': variant_name,
                        'domain': rec_domain,
                        'question': q,
                        'gt_ids': gt_ids,
                        'batch_idx': batch_idx,
                        'is_for_metric': True  # This combination is used for the metric calculation
                    }
                    
                    # Add any extra metadata
                    metadata.update(extra_metadata)
                    
                    batch_metadata.append(metadata)
            
            # Update preparation progress
            prep_pbar.update(1)
            
        # Close preparation progress bar
        prep_pbar.close()
        
        # Skip batch if empty
        if not batch_queries:
            main_pbar.update(1)
            continue
        
        # Get domain from the first metadata (safely)
        batch_domain = batch_metadata[0]['domain'] if batch_metadata else domain  # Fallback to provided domain
        
        # Progress bar for model evaluation
        model_pbar = tqdm_auto(total=1, desc="Preparing to evaluate models", position=1, leave=False)
        
        # Run unified generation for this batch across all models
        try:
            # Create async event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run unified generation
            all_model_responses = loop.run_until_complete(
                run_unified_generation(
                    queries=batch_queries,
                    docs_list=batch_docs,
                    models=generation_models,
                    domain=batch_domain,
                    inference_engine=inference_engine
                )
            )
            
            # Clean up event loop
            loop.close()
            
            # Update progress
            model_pbar.update(1)
            model_pbar.close()
            
            # Process results for each model
            model_process_pbar = tqdm_auto(total=len(generation_models), desc="Processing model results", position=1, leave=False)
            
            for model in generation_models:
                # Update progress description to show current model
                model_process_pbar.set_description(f"Processing model: {model.split('/')[-1]}")
                model_responses = all_model_responses.get(model, [""] * len(batch_queries))
                
                # Group results by record_id
                results_by_record = {}
                for i, meta in enumerate(batch_metadata):
                    record_id = meta['record_id']
                    if record_id not in results_by_record:
                        results_by_record[record_id] = []
                    
                    # Get response (safely)
                    response = model_responses[i] if i < len(model_responses) else ""
                    
                    # Add to results
                    results_by_record[record_id].append({
                        'meta': meta,
                        'response': response
                    })
                
                # Process results by record with progress tracking
                record_pbar = tqdm_auto(total=len(results_by_record), desc=f"Processing records for {model.split('/')[-1]}", position=2, leave=False)
                
                for record_id, results in results_by_record.items():
                    # Find the empty docs result to check if model can answer without retrieval
                    empty_results = [r for r in results if r['meta']['doc_type'] == 'empty']
                    if empty_results:
                        empty_result = empty_results[0]
                        empty_response = empty_result['response']
                        answer = empty_result['meta']['answer']
                        empty_response = str(empty_response).strip()
                        answer = str(answer).strip()
                        can_answer_without_retrieval = answer_judger(empty_response, answer)
                    else:
                        can_answer_without_retrieval = False
                    
                    # Track traditional metrics (for backward compatibility)
                    query_robust_vals = []  # For query_robustness metric
                    doc_robust_vals = []    # For doc_robustness metric
                    real_ret_vals = {}      # For real_retrieval_robustness metric
                    
                    # Process all results for all combinations
                    for result in results:
                        meta = result['meta']
                        
                        # Skip entries not used for metric calculation (like empty query test)
                        if not meta.get('is_for_metric', True):
                            continue
                            
                        response = str(result['response']).strip()
                        answer = str(meta['answer']).strip()
                        query_variant = meta['query_variant']
                        variant_name = meta.get('variant_name', 'original')
                        doc_type = meta['doc_type']
                        
                        # Unified evaluation for all document types
                        rob_result = robust_judger(
                            prediction=response,
                            truth=answer,
                            can_answer_without_retrieval=can_answer_without_retrieval,
                            doc_type=doc_type,
                            retrieval_contains_answer=meta.get('retrieval_contains_answer', False)
                        )
                        
                        # Add to the main flat list for overall robustness
                        model_results[model]['all_robustness_scores'].append(rob_result)
                        
                        # Track scores by variant name
                        if variant_name not in model_results[model]['variant_scores']:
                            model_results[model]['variant_scores'][variant_name] = []
                        model_results[model]['variant_scores'][variant_name].append(rob_result)
                        
                        # Track scores by variant name and document type
                        variant_doc_key = f"{variant_name}_{doc_type}"
                        if variant_doc_key not in model_results[model]['variant_doc_scores']:
                            model_results[model]['variant_doc_scores'][variant_doc_key] = []
                        model_results[model]['variant_doc_scores'][variant_doc_key].append(rob_result)
                        
                        # Track for backward compatibility metrics
                        if doc_type == 'real-world-docs':
                            retriever = meta.get('retriever', '')
                            if query_variant == 0:  # Original query
                                if retriever not in real_ret_vals:
                                    real_ret_vals[retriever] = []
                                real_ret_vals[retriever].append(rob_result)
                        elif doc_type == 'ground-truth-docs':
                            if query_variant > 0:  # Query variant with ground truth docs
                                query_robust_vals.append(rob_result)
                        elif query_variant == 0:  # Original query with perturbed docs
                            doc_robust_vals.append(rob_result)
                    
                    # Store backward compatibility metrics
                    if query_robust_vals:
                        query_score = sum(query_robust_vals) / len(query_robust_vals)
                        model_results[model]['query_scores'].append(query_score)
                    
                    if doc_robust_vals:
                        doc_score = sum(doc_robust_vals) / len(doc_robust_vals)
                        model_results[model]['doc_scores'].append(doc_score)
                    
                    for retriever, vals in real_ret_vals.items():
                        if vals:
                            retriever_score = sum(vals) / len(vals)
                            if retriever not in model_results[model]['realret_scores']:
                                model_results[model]['realret_scores'][retriever] = []
                            model_results[model]['realret_scores'][retriever].append(retriever_score)
                    
                    record_pbar.update(1)
                
                record_pbar.close()
                
                model_process_pbar.update(1)
            
            model_process_pbar.close()
            
        except Exception as e:
            logging.error(f"Error running generation: {e}")
            import traceback
            logging.error(traceback.format_exc())
            if 'model_pbar' in locals():
                model_pbar.close()
            if 'model_process_pbar' in locals():
                model_process_pbar.close()
        
        # Save intermediate results after each batch
        try:
            save_intermediate_results(domain, model_results, output_dir, batch_num)
        except Exception as e:
            logging.error(f"Failed to save interim results: {e}")

        main_pbar.update(1)
    
    # Calculate final aggregate scores for each model
    final_results = {}
    score_pbar = tqdm_auto(total=len(model_results), desc="Calculating final scores")
    
    for model, scores in model_results.items():
        # Update progress description to show current model
        score_pbar.set_description(f"Calculating scores for {model.split('/')[-1]}")
        
        # Calculate overall robustness as the simple mean of all evaluations (flat list)
        all_scores = scores.get('all_robustness_scores', [])
        if all_scores:
            overall_robustness = sum(all_scores) / len(all_scores)
        else:
            overall_robustness = 0.0
        
        # For backward compatibility, also calculate the other metrics
        query = scores.get('query_scores', [])
        doc = scores.get('doc_scores', [])
        realret = scores.get('realret_scores', {})
        print(f"real_retrieval results: {realret}")
        # Store final results
        final_results[model] = {
            'overall_robustness': overall_robustness,
            'query_robustness': sum(query) / max(len(query), 1), 
            'doc_robustness': sum(doc) / max(len(doc), 1),
        }
        
        # Include real retrieval scores (backward compatibility)
        for retriever_name, retriever_scores in realret.items():
            if retriever_scores:
                final_results[model][f'real_retrieval_robustness_{retriever_name}'] = sum(retriever_scores) / len(retriever_scores)
            else:
                final_results[model][f'real_retrieval_robustness_{retriever_name}'] = 0.0
        
        # Calculate combined real retrieval robustness
        all_retriever_scores = []
        for retriever_scores in realret.values():
            all_retriever_scores.extend(retriever_scores)
        
        if all_retriever_scores:
            final_results[model]['real_retrieval_robustness_combined'] = sum(all_retriever_scores) / len(all_retriever_scores)
        else:
            final_results[model]['real_retrieval_robustness_combined'] = 0.0
            
        # Calculate per-variant scores
        for variant_name, variant_scores in scores.get('variant_scores', {}).items():
            if variant_scores:
                final_results[model][f'variant_{variant_name}_robustness'] = sum(variant_scores) / len(variant_scores)
            else:
                final_results[model][f'variant_{variant_name}_robustness'] = 0.0
        
        # Calculate per-variant-per-doc-type scores
        for variant_doc_key, variant_doc_scores in scores.get('variant_doc_scores', {}).items():
            if variant_doc_scores:
                final_results[model][f'variant_doc_{variant_doc_key}_robustness'] = sum(variant_doc_scores) / len(variant_doc_scores)
            else:
                final_results[model][f'variant_doc_{variant_doc_key}_robustness'] = 0.0
        
        score_pbar.update(1)
    
    score_pbar.close()
    
    main_pbar.close()
    
    return final_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate RAG robustness (generation phase)')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing dataset JSON files')
    parser.add_argument('--chunks_file', type=str, required=True, help='Path to chunks JSON')
    parser.add_argument('--retrieval_results_dir', type=str, required=True, help='Directory containing retrieval results JSON files')
    parser.add_argument('--models', type=str, nargs='+', default=['meta-llama/Llama-3.1-8B-Instruct', 'openai/gpt-4.1'], 
                      help='List of model names to evaluate (can include both vLLM and API models)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save result files')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing (higher uses more memory but is faster)')
    parser.add_argument('--log_level', type=str, default='ERROR', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                      help='Logging level')
    parser.add_argument('--inference_engine', type=str, default='auto', choices=['auto', 'vllm', 'litellm'],
                      help='Inference engine to use for all models (auto=detect based on model name)')
    args = parser.parse_args()
    
    # Set the global inference engine setting
    DEFAULT_INFERENCE_ENGINE = args.inference_engine
    
    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Add file handler for logging
    log_file = os.path.join(args.output_dir, f"robustness_eval_{int(time.time())}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"Starting robustness evaluation with arguments: {args}")
    
    # Use the models directly from args
    generation_models = args.models
    logging.info(f"Using models: {generation_models}")
    
    # Load chunks
    print(f"Loading chunks from {args.chunks_file}...")
    logging.info(f"Loading chunks from {args.chunks_file}")
    _, chunks = load_chunks(args.chunks_file)
    print(f"Loaded {len(chunks)} chunks")
    logging.info(f"Loaded {len(chunks)} chunks")
    
    # Get all domain-specific dataset files
    dataset_files = []
    for filename in os.listdir(args.dataset_dir):
        if filename.endswith('.json'):
            dataset_files.append(os.path.join(args.dataset_dir, filename))
    
    if not dataset_files:
        print(f"No dataset files found in {args.dataset_dir}")
        logging.error(f"No dataset files found in {args.dataset_dir}")
        sys.exit(1)
    
    # Process each dataset file with progress bar
    print(f"Found {len(dataset_files)} dataset files")
    logging.info(f"Found {len(dataset_files)} dataset files")
    
    for dataset_file in tqdm_auto(dataset_files, desc="Processing datasets"):
        filename = os.path.basename(dataset_file)
        domain = os.path.splitext(filename)[0]  # Extract domain from filename
        
        print(f"\n{'='*80}")
        print(f"Processing dataset: {domain}")
        print(f"{'='*80}")
        logging.info(f"Processing dataset: {domain}")
        
        # Load dataset
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            logging.info(f"Loaded {len(records)} records from {domain}")
        except Exception as e:
            print(f"Error loading dataset {domain}: {e}")
            logging.error(f"Error loading dataset {domain}: {e}")
            continue
        
        # Check for corresponding retrieval results
        retrieval_results_file = os.path.join(args.retrieval_results_dir, f"{domain}_retrieval_results.json")
        if not os.path.exists(retrieval_results_file):
            print(f"Retrieval results not found for {domain}, skipping generation")
            logging.warning(f"Retrieval results not found for {domain}, skipping generation")
            continue
        
        print(f"Evaluating robustness for {domain} with {len(records)} records...")
        print(f"Using models: {', '.join([m.split('/')[-1] for m in generation_models])}")
        logging.info(f"Evaluating robustness for {domain} with {len(records)} records...")
        logging.info(f"Using models: {', '.join([m.split('/')[-1] for m in generation_models])}")
        
        # Group models by type for reporting based on inference engine setting
        if args.inference_engine == "auto":
            vllm_models = [m for m in generation_models if is_vllm_model(m)]
            api_models = [m for m in generation_models if is_api_model(m)]
            
            if vllm_models:
                print(f"Models using vLLM: {', '.join([m.split('/')[-1] for m in vllm_models])}")
                logging.info(f"Models using vLLM: {', '.join([m.split('/')[-1] for m in vllm_models])}")
            
            if api_models:
                print(f"Models using API: {', '.join([m.split('/')[-1] for m in api_models])}")
                logging.info(f"Models using API: {', '.join([m.split('/')[-1] for m in api_models])}")
        else:
            print(f"Using {args.inference_engine} inference for all models")
            logging.info(f"Using {args.inference_engine} inference for all models")
        
        # Run evaluation using the retrieval results with batch processing
        try:
            results = evaluate_robustness(
                records,
                chunks,
                retrieval_results_file,
                generation_models,
                args.output_dir,
                domain,
                args.batch_size,
                args.inference_engine
            )
            
            # Save results
            output_file = os.path.join(args.output_dir, f"{domain}_robustness_results.json")
            with open(output_file, 'w', encoding='utf-8') as wf:
                json.dump(results, wf, indent=2)
            
            logging.info(f"Results saved to {output_file}")
            
            # Print table-like results for easy reading
            print(f"\n{'='*80}")
            print(f"FINAL ROBUSTNESS SCORES FOR {domain.upper()}")
            print(f"{'='*80}")
            
            # Get all metrics and model names
            if results:
                metrics = list(next(iter(results.values())).keys())
                model_names = [m.split('/')[-1] for m in results.keys()]
                
                # Print header
                print(f"\n{'Metric':<40} | " + " | ".join(f"{m:<15}" for m in model_names))
                print("-" * 40 + "+" + "+".join(["-" * 17] * len(model_names)))
                
                # Print each metric row
                for metric in metrics:
                    print(f"{metric:<40} | " + " | ".join(f"{results[model][metric]:.4f}" for model in results.keys()))
                
                print(f"\nResults saved to {output_file}")
                
        except Exception as e:
            print(f"Error evaluating {domain}: {e}")
            logging.error(f"Error evaluating {domain}: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    print("\nEvaluation complete!")
    logging.info("Evaluation complete!")