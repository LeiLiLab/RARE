import os
import json
import argparse
import logging
import asyncio
import time
from typing import List, Dict, Any, Tuple
from tqdm.auto import tqdm as tqdm_auto
from rag_pipeline import load_chunks
from utils.eval import answer_judger, robust_judger, prepare_batch_prompts, is_refusal, get_shared_loop
from utils.litellm_router_models import MODEL_LIST
import sys
import litellm
import re
from collections import defaultdict
from pathlib import Path
from pydantic import BaseModel
import ast

os.environ["BEDROCK_QWEN3_32B_MODEL_ID"] = "qwen.qwen3-32b-v1:0"
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION"] = "us-east-1"

class CoTResponse(BaseModel):
    cot_answer: str
    answer: str
router = litellm.Router(model_list=MODEL_LIST, num_retries=3, retry_after=5)

async def run_generation(
    queries: List[str],
    docs_list: List[Any],
    models: List[str],
    domain: str,
) -> Dict[str, List[str]]:
    messages = [
        [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_p}
        ]
        for sys_p, user_p in prepare_batch_prompts(queries, docs_list, domain)
    ]

    # Minimal branching: use Bedrock via boto3 only for Qwen3-32B; otherwise keep original path
    parsed: Dict[str, List[str]] = {m: [] for m in models}

    qwen_bedrock_models = [
        m for m in models
        if "qwen" in m.lower() and ("qwen3-32b" in m.lower() or "qwen/qwen3-32b" in m.lower())
    ]
    other_models = [m for m in models if m not in qwen_bedrock_models]

    # 1) Handle non-Qwen models via existing litellm router (no changes to behavior)
    if other_models:
        raw_responses = await router.abatch_completion(
            models=other_models,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            tool_choice="none",
            response_format=CoTResponse
        )
        for req_row in raw_responses:
            for model_idx, response in enumerate(req_row):
                model = other_models[model_idx]
                try:
                    if not isinstance(response, litellm.ContextWindowExceededError):
                        ans = response.choices[0].message.content
                        ans = ast.literal_eval(ans)
                        if isinstance(ans, dict) and "answer" in ans:
                            parsed[model].append(ans["answer"])
                    else:
                        logging.error(f"Context window exceeded for model {model} with response: {response}")
                        parsed[model].append("Context window exceeded")
                except Exception as e:
                    matched = re.search(r'"?answer"?\s*:\s*(.*)', ans)  # type: ignore[name-defined]
                    try:
                        if matched:
                            ans = matched.group(1).replace('"', '').replace("\\", "").strip()
                            parsed[model].append(ans)
                        else:
                            parsed[model].append(ans)  # type: ignore[arg-type]
                    except Exception:
                        parsed[model].append(ans)  # type: ignore[arg-type]

    # 2) Handle Qwen3-32B via Amazon Bedrock using boto3 in concurrent mini-batches
    if qwen_bedrock_models:
        try:
            import boto3  # local import to avoid dependency for non-Qwen paths
            from botocore.config import Config
        except Exception as e:
            logging.error(f"Failed to import boto3/Config for Bedrock: {e}")
            for m in qwen_bedrock_models:
                parsed[m].extend(["Bedrock client import error"] * len(messages))
        else:
            # Resolve modelId for Bedrock; allow env override
            def _resolve_bedrock_qwen_model_id(model_name: str) -> str:
                env_override = os.getenv("BEDROCK_QWEN3_32B_MODEL_ID", "").strip()
                if env_override:
                    return env_override
                lower = model_name.lower()
                if ":" in model_name:
                    return model_name  # looks like a direct Bedrock modelId/ARN
                if "qwen3-32b" in lower:
                    return "qwen.qwen3-32b-instruct-v1:0"  # sensible default
                return model_name

            # Build Bedrock client with retries; honor AWS env for region/creds
            region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or os.getenv("aws_region_name")
            br_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=region if region else "us-east-1",
                config=Config(retries={"total_max_attempts": 10, "mode": "standard"})
            )

            # Compose prompts using Qwen chat template compatible with imported models
            chat_messages_per_item: List[List[Dict[str, Any]]] = []
            for pair in messages:
                sys_p = pair[0]["content"]
                user_p = pair[1]["content"]
                chat_messages_per_item.append([
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": f"{user_p} /no_think"},
                ])

            # Concurrency limiter for Bedrock requests
            concurrency = int(os.getenv("BEDROCK_QWEN_CONCURRENCY", "8"))
            sem = asyncio.Semaphore(max(1, concurrency))

            async def _invoke_one(model_id: str, msg_list: List[Dict[str, Any]]) -> str:
                async with sem:
                    def _call_sync() -> str:
                        try:
                            resp = br_client.invoke_model(
                                modelId=model_id,
                                body=json.dumps({
                                    "messages": msg_list,
                                    "temperature": 0.0,
                                    "max_tokens": 1024,
                                    "top_p": 1.0
                                }),
                                accept="application/json",
                                contentType="application/json"
                            )
                            body_bytes = resp["body"].read()
                            payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
                            # Try common fields across providers/imports
                            if isinstance(payload, dict):
                                # OpenAI-style chat completions
                                if "choices" in payload and isinstance(payload["choices"], list) and payload["choices"]:
                                    first = payload["choices"][0]
                                    if isinstance(first, dict):
                                        if "message" in first and isinstance(first["message"], dict):
                                            msg = first["message"]
                                            content = msg.get("content")
                                            if isinstance(content, str):
                                                return content
                                            if isinstance(content, list):
                                                texts = [b.get("text", "") for b in content if isinstance(b, dict)]
                                                if any(texts):
                                                    return "".join(texts)
                                        for ck in ("text", "output_text"):
                                            if ck in first and isinstance(first[ck], str):
                                                return first[ck]
                                # Generic fields
                                for key in ("output_text", "text", "answer", "generated_text"):
                                    if key in payload and isinstance(payload[key], str):
                                        return payload[key]
                            # If payload is a string-like dict, mirror existing regex/ast parse to extract answer
                            if isinstance(payload, str):
                                raw = payload
                            else:
                                raw = json.dumps(payload)
                            try:
                                parsed_obj = ast.literal_eval(raw)
                                if isinstance(parsed_obj, dict) and "answer" in parsed_obj and isinstance(parsed_obj["answer"], str):
                                    return parsed_obj["answer"]
                            except Exception:
                                pass
                            matched = re.search(r'"?answer"?\s*:\s*(.*)', raw)
                            if matched:
                                ans = matched.group(1).replace('"', '').replace("\\", "").strip()
                                return ans
                            return str(payload)
                        except Exception as e_inner:
                            logging.error(f"Bedrock invoke_model error for {model_id}: {e_inner}")
                            return ""
                    return await asyncio.to_thread(_call_sync)

            # For each Qwen model, invoke across all prompts
            for model_name in qwen_bedrock_models:
                model_id = _resolve_bedrock_qwen_model_id(model_name)
                tasks = [_invoke_one(model_id, m) for m in chat_messages_per_item]
                results = await asyncio.gather(*tasks, return_exceptions=False)
                parsed[model_name].extend(results)

    return parsed

def evaluate_robustness(
    records: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    retrieval_results_file: os.PathLike | str,
    generation_models: List[str],
    output_dir: os.PathLike | str,
    domain: str,
    batch_size: int = 32,
    resume_from_checkpoint: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compute robustness metrics and print cumulative results after each batch."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoint
    checkpoint_path = output_dir / f"{domain}_checkpoint.json"
    start_batch = 1  # Start from batch 1 instead of 0
    loaded_checkpoint = False
    
    # Initialize model store
    model_store: Dict[str, Dict[str, Any]] = {m: {
        "all": [],
        "query": [],
        "doc": [],
        "real": defaultdict(list),
        "variant": defaultdict(list),
        "variant_doc": defaultdict(list),
        "response_details": [],
        "response_by_doc_type": defaultdict(list),
        "can_answer_without": defaultdict(bool),
    } for m in generation_models}
    
    can_answer_without: Dict[str, Dict[str, bool]] = {m: defaultdict(bool) for m in generation_models}
    
    # Load checkpoint if exists and resume is enabled
    if resume_from_checkpoint and checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            if (set(checkpoint['models']) == set(generation_models) and 
                checkpoint['domain'] == domain):
                
                start_batch = checkpoint.get('last_completed_batch', 0) + 1
                loaded_checkpoint = True
                
                for model in generation_models:
                    if model in checkpoint['model_store']:
                        store_data = checkpoint['model_store'][model]
                        model_store[model]['all'] = store_data['all']
                        model_store[model]['query'] = store_data['query']
                        model_store[model]['doc'] = store_data['doc']
                        model_store[model]['real'] = defaultdict(list, store_data['real'])
                        model_store[model]['variant'] = defaultdict(list, store_data['variant'])
                        model_store[model]['variant_doc'] = defaultdict(list, store_data['variant_doc'])
                        model_store[model]['response_details'] = store_data['response_details']
                        model_store[model]['response_by_doc_type'] = defaultdict(list, store_data['response_by_doc_type'])
                        model_store[model]['can_answer_without'] = defaultdict(bool, store_data['can_answer_without'])
                
                for model, mappings in checkpoint['can_answer_without'].items():
                    can_answer_without[model] = defaultdict(bool, mappings)
                
                logging.info(f"Resumed from checkpoint at batch {start_batch}")
                print(f"\n{'='*80}")
                print(f"RESUMING from batch {start_batch} (already completed {start_batch-1} batches)")
                print(f"{'='*80}\n")
            else:
                logging.warning("Checkpoint found but models or domain don't match, starting fresh")
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}, starting fresh")
    
    def _analyze_responses(model_store: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        analysis = {}
        
        for model, store in model_store.items():
            doc_type_stats = defaultdict(lambda: {
                "total": 0, "robust": 0, "correct": 0, 
                "no_info": 0, "incorrect": 0, "can_answer_without": 0
            })
            for detail in store["response_details"]:
                doc_type = detail["doc_variant_type"]
                stats = doc_type_stats[doc_type]
                stats["total"] += 1
                if detail["robust"]:
                    stats["robust"] += 1
                if detail["is_correct"]:
                    stats["correct"] += 1
                if detail["returns_no_info"]:
                    stats["no_info"] += 1
                if not detail["is_correct"] and not detail["returns_no_info"]:
                    stats["incorrect"] += 1
                if detail["can_answer_without"]:
                    stats["can_answer_without"] += 1
            for doc_type, stats in doc_type_stats.items():
                if stats["total"] > 0:
                    stats["robustness_rate"] = stats["robust"] / stats["total"]
                    stats["correct_rate"] = stats["correct"] / stats["total"]
                    stats["no_info_rate"] = stats["no_info"] / stats["total"]
                    stats["incorrect_rate"] = stats["incorrect"] / stats["total"]
            analysis[model] = {
                "total_responses": len(store["response_details"]),
                "overall_robustness": sum(store["all"]) / max(len(store["all"]), 1),
                "by_doc_type": dict(doc_type_stats),
                "parametric_knowledge_count": sum(store["can_answer_without"].values()),
            }
        
        return analysis
    
    def _save_checkpoint(batch_no: int, work_items_total: int):
        checkpoint_data = {
            "domain": domain,
            "models": generation_models,
            "last_completed_batch": batch_no,
            "total_work_items": work_items_total,
            "batch_size": batch_size,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_store": {},
            "can_answer_without": {}
        }
        for model, store in model_store.items():
            checkpoint_data["model_store"][model] = {
                "all": store["all"],
                "query": store["query"],
                "doc": store["doc"],
                "real": dict(store["real"]),
                "variant": dict(store["variant"]),
                "variant_doc": dict(store["variant_doc"]),
                "response_details": store["response_details"],
                "response_by_doc_type": dict(store["response_by_doc_type"]),
                "can_answer_without": dict(store["can_answer_without"])
            }
        for model, mappings in can_answer_without.items():
            checkpoint_data["can_answer_without"][model] = dict(mappings)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        logging.info(f"Checkpoint saved at batch {batch_no}")
    
    def _dump_interim(batch_no: int, total_work_items: int):
        intermediate: Dict[str, Dict[str, float]] = {}
        for model, store in model_store.items():
            res: Dict[str, float] = {}
            res["overall_robustness"] = sum(store["all"]) / max(len(store["all"]), 1)
            res["query_robustness"] = sum(store["query"]) / max(len(store["query"]), 1)
            res["doc_robustness"] = sum(store["doc"]) / max(len(store["doc"]), 1)
            combined: List[float] = []
            for retr, scores in store["real"].items():
                res[f"real_retrieval_robustness_{retr}"] = sum(scores) / len(scores)
                combined.extend(scores)
            res["real_retrieval_robustness_combined"] = (
                sum(combined) / len(combined) if combined else 0.0
            )
            for vname, scores in store["variant"].items():
                res[f"variant_{vname}_robustness"] = sum(scores) / len(scores)
            for key, scores in store["variant_doc"].items():
                res[f"variant_doc_{key}_robustness"] = sum(scores) / len(scores)
            intermediate[model] = res

        if batch_no % 1 == 0 or (batch_no - 1) * batch_size >= total_work_items:
            response_analysis = _analyze_responses(model_store)
            interim_data = {
                "metrics": intermediate,
                "analysis": response_analysis,
                "batch_info": {
                    "current_batch": batch_no,
                    "batch_size": batch_size,
                    "total_work_items": total_work_items,
                    "progress": min((batch_no - 1) * batch_size + batch_size, total_work_items) / total_work_items
                }
            }
            interim_path = output_dir / f"{domain}_interim_results_batch_{batch_no}.json"
            with open(interim_path, "w", encoding="utf-8") as fp:
                json.dump(interim_data, fp, indent=2)
            logging.info("Interim results with analysis saved to %s", interim_path)
            _save_checkpoint(batch_no, total_work_items)
            print(f"\n--- RESPONSE ANALYSIS (Batch {batch_no}) for {domain} ---")
            print(f"Progress: {min((batch_no - 1) * batch_size + batch_size, total_work_items)}/{total_work_items} items processed")
            for model, analysis in response_analysis.items():
                ms = model.split("/")[-1]
                print(f"\n{ms}:")
                print(f"  Parametric knowledge instances: {analysis['parametric_knowledge_count']}")
                print(f"  Response breakdown by document type:")
                for doc_type, stats in analysis['by_doc_type'].items():
                    if stats['total'] > 0:
                        print(f"    {doc_type}:")
                        print(f"      Robustness: {stats.get('robustness_rate', 0):.2%} ({stats['robust']}/{stats['total']})")
                        print(f"      Correct: {stats.get('correct_rate', 0):.2%}, No info: {stats.get('no_info_rate', 0):.2%}, Incorrect: {stats.get('incorrect_rate', 0):.2%}")

        print(f"\n--- INTERIM RESULTS (Batch {batch_no}) for {domain} ---")
        for model, scores in intermediate.items():
            ms = model.split("/")[-1]
            print(f"\n{ms}:")
            print(f"  overall_robustness: {scores['overall_robustness']:.4f}")
            print(f"  query_robustness: {scores['query_robustness']:.4f}")
            print(f"  doc_robustness: {scores['doc_robustness']:.4f}")
            for key in sorted(scores):
                if key.startswith("real_retrieval_robustness_"):
                    print(f"  {key}: {scores[key]:.4f}")
            for key in sorted(scores):
                if key.startswith("variant_") and not key.startswith("variant_doc_"):
                    print(f"  {key}: {scores[key]:.4f}")
            for key in sorted(scores):
                if key.startswith("variant_doc_"):
                    print(f"  {key}: {scores[key]:.4f}")

    # 0. Pre-processing & look-ups
    id2text = {c["chunk_id"]: c.get("text", "") for c in chunks}
    with open(retrieval_results_file, "r", encoding="utf-8") as fp:
        retrieval_results: Dict[str, Any] = json.load(fp)

    # 1. Build work items once – each (query variant x doc variant) pair
    work_items: List[Tuple[str, List[str], Dict[str, Any]]] = []
    
    def _id_list_to_texts(ids: List[str], id2text: Dict[str, str]):
        return [id2text[cid] for cid in ids if cid in id2text]

    for rec in records:
        rid = str(rec.get("id", ""))
        if rid not in retrieval_results:
            logging.warning("No retrieval result for record %s - skipped", rid)
            continue

        answer = rec["answer"].strip()
        retrieval_result = retrieval_results[rid]

        # Query variants
        query_variants: List[Tuple[str, str]] = [("original", rec["question"])]
        query_variants += [(n, v["perturbed_query"]) for n, v in rec["query_perturbations"].items()]

        # Document variants
        gt_docs = _id_list_to_texts(rec["answer_chunk_ids"], id2text)
        regex_docs = [d["text"] for d in rec["document_perturbations"]["regex_delete"]]
        back_docs = [d["text"] for d in rec["document_perturbations"]["back_translation"]]
        # Real-world retrievals
        real_docs: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
        for retriever, retrieval_map in retrieval_result["retrievers"].items():
            for query_variant, doc_ids in retrieval_map.items():
                real_docs[retriever][query_variant] = _id_list_to_texts(doc_ids, id2text)

        # Test whether the model can answer with no docs
        work_items.append((rec["question"], [], {
            "record_id": rid,
            "answer": answer,
            "doc_variant_type": "empty",
            "query_variant_type": "original",
            "retriever": "",
            "for_metric": False,
        }))

        # All other combinations
        for query_variant_name, query_variant_text in query_variants:
            # Ground-truth docs
            work_items.append((query_variant_text, gt_docs, {
                "record_id": rid,
                "answer": answer,
                "doc_variant_type": "ground-truth-docs",
                "query_variant_type": query_variant_name,
                "retriever": "",
                "for_metric": True,
            }))
            # Lexical similar but no answer
            if regex_docs:
                work_items.append((query_variant_text, regex_docs, {
                    "record_id": rid,
                    "answer": answer,
                    "doc_variant_type": "lexical-similar-no-answer-docs",
                    "query_variant_type": query_variant_name,
                    "retriever": "",
                    "for_metric": True,
                }))
            # Lexical diff with answer
            if back_docs:
                work_items.append((query_variant_text, back_docs, {
                    "record_id": rid,
                    "answer": answer,
                    "doc_variant_type": "lexical-diff-with-answer-docs",
                    "query_variant_type": query_variant_name,
                    "retriever": "",
                    "for_metric": True,
                }))
            # Real-world retrievals
            for retriever, query_variant_map in real_docs.items():
                docs_here = query_variant_map[query_variant_name]
                work_items.append((query_variant_text, docs_here, {
                    "record_id": rid,
                    "answer": answer,
                    "doc_variant_type": "real-world-docs",
                    "query_variant_type": query_variant_name,
                    "retriever": retriever,
                    "for_metric": True,
                }))

    # 3. Batched generation & metric accumulation with enhanced tracking
    total_batches = (len(work_items) + batch_size - 1) // batch_size
    
    shared_loop = get_shared_loop()  # <-- use the same loop everywhere

    for batch_idx in tqdm_auto(range(start_batch, total_batches + 1), 
                               initial=start_batch - 1, 
                               total=total_batches, 
                               desc="Batches"):
        batch_start = (batch_idx - 1) * batch_size
        batch = work_items[batch_start : batch_start + batch_size]
        if not batch:
            break
            
        queries = [w[0] for w in batch]
        docs_list = [w[1] for w in batch]
        meta = [w[2] for w in batch]

        # Run generation on the shared loop (no asyncio.run)
        model2resp = shared_loop.run_until_complete(run_generation(queries, docs_list, generation_models, domain))

        # Post-process responses with enhanced tracking (BATCHED judgers)
        for model, responses in model2resp.items():
            assert len(responses) == len(meta)

            # 1) Handle "empty" doc_variant in batch for can_answer_without
            empty_items = [(i, str(responses[i]).strip(), meta[i])
                           for i in range(len(meta)) if meta[i]["doc_variant_type"] == "empty"]
            if empty_items:
                preds_empty = [p for _, p, _ in empty_items]
                truths_empty = [m["answer"] for _, _, m in empty_items]
                can_list = answer_judger(preds_empty, truths_empty)  # batched
                if not isinstance(can_list, list):
                    can_list = [can_list]
                for (idx, _, m), can_flag in zip(empty_items, can_list):
                    rid = m["record_id"]
                    if not can_answer_without[model][rid]:
                        can_answer_without[model][rid] = bool(can_flag)
                    model_store[model]["can_answer_without"][rid] = can_answer_without[model][rid]

            # 2) Non-empty items — batch compute correctness and robustness
            non_items = [(i, str(responses[i]).strip(), meta[i])
                         for i in range(len(meta)) if meta[i]["doc_variant_type"] != "empty"]
            if not non_items:
                _dump_interim(batch_idx, len(work_items))
                continue

            preds = [p for _, p, _ in non_items]
            truths = [m["answer"] for _, _, m in non_items]
            rids = [m["record_id"] for _, _, m in non_items]
            doc_types = [m["doc_variant_type"] for _, _, m in non_items]
            query_variants = [m["query_variant_type"] for _, _, m in non_items]
            retrievers = [m["retriever"] for _, _, m in non_items]
            can_flags = [can_answer_without[model][rid] for rid in rids]

            is_correct_list = answer_judger(preds, truths)
            if not isinstance(is_correct_list, list):
                is_correct_list = [is_correct_list]

            robust_list = robust_judger(preds, truths, can_flags, doc_types)
            if not isinstance(robust_list, list):
                robust_list = [robust_list]

            returns_no_info_list = [(not corr) and is_refusal(pred) for corr, pred in zip(is_correct_list, preds)]

            for resp, rid, qv, dvt, retr, corr, rni, rob in zip(
                preds, rids, query_variants, doc_types, retrievers, is_correct_list, returns_no_info_list, robust_list
            ):
                response_detail = {
                    "record_id": rid,
                    "query_variant": qv,
                    "doc_variant_type": dvt,
                    "robust": bool(rob),
                    "is_correct": bool(corr),
                    "returns_no_info": bool(rni),
                    "can_answer_without": can_answer_without[model][rid],
                    "response_preview": resp[:100] + "..." if len(resp) > 100 else resp
                }
                
                st = model_store[model]
                st["response_details"].append(response_detail)
                st["response_by_doc_type"][dvt].append(response_detail)
                
                st["all"].append(bool(rob))
                st["variant"][qv].append(bool(rob))
                st["variant_doc"][f"{qv}_{dvt}"].append(bool(rob))

                if qv != "original" and dvt == "ground-truth-docs":
                    st["query"].append(bool(rob))
                if qv == "original" and dvt in ["lexical-similar-no-answer-docs", "lexical-diff-with-answer-docs"]:
                    st["doc"].append(bool(rob))
                if qv == "original" and dvt == "real-world-docs":
                    st["real"][retr].append(bool(rob))
                
                if batch_idx == start_batch and len(st["response_details"]) <= 10:
                    logging.info(f"Model: {model.split('/')[-1]}, Doc type: {dvt}, "
                                 f"Query variant: {qv}, "
                                 f"Correct: {corr}, No info: {rni}, "
                                 f"Can answer without: {can_answer_without[model][rid]}, "
                                 f"Robust: {rob}")

        _dump_interim(batch_idx, len(work_items))

    # 4. Final aggregation with complete analysis
    final_batch = batch_idx
    _dump_interim(final_batch, len(work_items))
    
    final_analysis = _analyze_responses(model_store)
    
    final_report = {
        "metrics": {},
        "analysis": final_analysis,
        "summary": {
            "domain": domain,
            "total_records": len(records),
            "total_work_items": len(work_items),
            "models_evaluated": list(generation_models),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "resumed_from_checkpoint": loaded_checkpoint,
            "final_batch": final_batch
        }
    }
    
    final_path = output_dir / f"{domain}_interim_results_batch_{final_batch}.json"
    with open(final_path, "r", encoding="utf-8") as fp:
        final_data = json.load(fp)
        if isinstance(final_data, dict) and "metrics" in final_data:
            final_report["metrics"] = final_data["metrics"]
        else:
            final_report["metrics"] = final_data
    
    report_path = output_dir / f"{domain}_final_report.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(final_report, fp, indent=2)
    
    checkpoint_path = output_dir / f"{domain}_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logging.info("Checkpoint file removed after successful completion")
    
    return final_report["metrics"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate RAG robustness (generation phase)')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing dataset JSON files')
    parser.add_argument('--chunks_file', type=str, required=True, help='Path to chunks JSON')
    parser.add_argument('--retrieval_results_dir', type=str, required=True, help='Directory containing retrieval results JSON files')
    parser.add_argument('--models', type=str, nargs='+', default=['openai/gpt-4.1'], 
                      help='List of model names to evaluate')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save result files')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing (higher uses more memory but is faster)')
    parser.add_argument('--log_level', type=str, default='ERROR', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                      help='Logging level')
    parser.add_argument('--no_resume', action='store_true', 
                      help='Do not resume from checkpoint, start fresh even if checkpoint exists')
    args = parser.parse_args()
    
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_file = os.path.join(args.output_dir, f"robustness_eval_{int(time.time())}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"Starting robustness evaluation with arguments: {args}")
    
    generation_models = args.models
    logging.info(f"Using models: {generation_models}")
    
    print(f"Loading chunks from {args.chunks_file}...")
    logging.info(f"Loading chunks from {args.chunks_file}")
    _, chunks = load_chunks(args.chunks_file)
    print(f"Loaded {len(chunks)} chunks")
    logging.info(f"Loaded {len(chunks)} chunks")
    
    dataset_files = [os.path.join(args.dataset_dir, f) for f in os.listdir(args.dataset_dir) if f.endswith('.json')]
    if not dataset_files:
        print(f"No dataset files found in {args.dataset_dir}")
        logging.error(f"No dataset files found in {args.dataset_dir}")
        sys.exit(1)
    
    print(f"Found {len(dataset_files)} dataset files")
    logging.info(f"Found {len(dataset_files)} dataset files")
    
    for dataset_file in tqdm_auto(dataset_files, desc="Processing datasets"):
        filename = os.path.basename(dataset_file)
        domain = os.path.splitext(filename)[0]
        
        print(f"\n{'='*80}")
        print(f"Processing dataset: {domain}")
        print(f"{'='*80}")
        logging.info(f"Processing dataset: {domain}")
        
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
            logging.info(f"Loaded {len(records)} records from {domain}")
        except Exception as e:
            print(f"Error loading dataset {domain}: {e}")
            logging.error(f"Error loading dataset {domain}: {e}")
            continue
        
        retrieval_results_file = os.path.join(args.output_dir if args.retrieval_results_dir == "" else args.retrieval_results_dir,
                                              f"{domain}_retrieval_results.json")
        if not os.path.exists(retrieval_results_file):
            print(f"Retrieval results not found for {domain}, skipping generation")
            logging.warning(f"Retrieval results not found for {domain}, skipping generation")
            continue
        
        print(f"Evaluating robustness for {domain} with {len(records)} records...")
        print(f"Using models: {', '.join([m.split('/')[-1] for m in generation_models])}")
        logging.info(f"Evaluating robustness for {domain} with {len(records)} records...")
        logging.info(f"Using models: {', '.join([m.split('/')[-1] for m in generation_models])}")
        
        try:
            results = evaluate_robustness(
                records,
                chunks,
                retrieval_results_file,
                generation_models,
                args.output_dir,
                domain,
                args.batch_size,
                resume_from_checkpoint=not args.no_resume
            )
            
            print(f"\n{'='*80}")
            print(f"FINAL ROBUSTNESS SCORES FOR {domain.upper()}")
            print(f"{'='*80}")
            if results:
                metrics = list(next(iter(results.values())).keys())
                model_names = [m.split('/')[-1] for m in results.keys()]
                print(f"\n{'Metric':<40} | " + " | ".join(f"{m:<15}" for m in model_names))
                print("-" * 40 + "+" + "+".join(["-" * 17] * len(model_names)))
                for metric in metrics:
                    print(f"{metric:<40} | " + " | ".join(f"{results[model][metric]:.4f}" for model in results.keys()))
        except Exception as e:
            print(f"Error evaluating {domain}: {e}")
            logging.error(f"Error evaluating {domain}: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    print("\nEvaluation complete!")
    logging.info("Evaluation complete!")
