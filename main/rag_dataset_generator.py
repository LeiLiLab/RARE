import os
import json
import random
from pathlib import Path

import argparse
import pandas as pd
import networkx as nx
import litellm
from tqdm import tqdm
import ast
from pydantic import BaseModel

from utils.prompts import (
    finance_onehop_query_prompt_system,
    finance_onehop_query_prompt_user,
    finance_multihop_query_prompt_system,
    finance_multihop_query_prompt_user,
    
    econ_onehop_query_prompt_system,
    econ_onehop_query_prompt_user,
    econ_multihop_query_prompt_system,
    econ_multihop_query_prompt_user,
    
    policy_onehop_query_prompt_system,
    policy_onehop_query_prompt_user,
    policy_multihop_query_prompt_system,
    policy_multihop_query_prompt_user
)
from utils.api_keys import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Pydantic models to enforce structured LLM output
class QAPair(BaseModel):
    question: str
    answer: str

# Knowledge Graph Traversal
class KGTraverser:
    """
    Traverses Directed GraphML KGs in patterns:
      - single-hop
      - chained multi-hop (depth up to 4)
      - star-shaped multi-hop (depth=2)
      - inverted_star-shaped triplets
      - cycle-path triplets
    Ensures no duplicate paths within each pattern.
    """
    def __init__(self, triplets_file: str):
        triplets = json.loads(Path(triplets_file).read_text())
        self.index = { (t['entity_1'], t['relation'], t['entity_2']): t for t in triplets }
        self.visited = set()
        self.valid_succ = {}

    def load_graph(self, path: str) -> nx.DiGraph:
        """Read a GraphML file and return a directed graph."""
        return nx.read_graphml(path)

    def _lookup(self, e1, rel, e2):
        """Fast triplet lookup by (entity1, relation, entity2)."""
        return self.index.get((e1, rel, e2))

    def _prepare(self, G: nx.DiGraph):
        # Precompute valid successors using triplets index
        self.valid_succ.clear()
        for n in G.nodes():
            self.valid_succ[n] = []
            for nbr in G.successors(n):
                rel = G.edges[n, nbr].get('label', '')
                if self._lookup(n, rel, nbr):
                    self.valid_succ[n].append((nbr, rel))

    def single_hop(self, G: nx.DiGraph) -> list[dict]:
        """
        Extract every edge u->v where:
        1. v is a leaf (out-degree=0)
        2. v has in-degree=1 (only reachable from u)
        3. There's only one path from u to v (single relation)
        4. The triplet has sufficient information for a standalone question
        """
        results = []
        for u, v, data in G.edges(data=True):
            # Check leaf node with single incoming edge
            if G.out_degree(v) == 0 and G.in_degree(v) == 1:
                rel = data.get('label', '')
                triplet = self._lookup(u, rel, v)
                
                if triplet:
                    # Count paths between u and v to ensure uniqueness
                    path_count = sum(1 for _ in nx.all_simple_paths(G, u, v))
                    
                    # Only proceed if there's exactly one path
                    if path_count == 1:
                        # Extract source sentence if available or use empty string
                        source_sentence = triplet.get('source_sentence', '')
                        single_hop_triplet = {
                            'entity_1': u,
                            'relation': rel,
                            'entity_2': v,
                            'triplet_id': triplet['triplet_id'],
                            'answer_chunk_id': triplet['answer_chunk_id'],
                            'metadata': triplet['metadata'],
                            'source_sentence': source_sentence
                        }
                        results.append(single_hop_triplet)
                        print(f"Find {len(results)} single-hop triplets")
        return results

    def chained_multi_hop(self, G: nx.DiGraph, max_depth=3) -> list[list[dict]]:
        """
        Perform iterative deepening DFS up to max_depth:
          - Each path must have at least 2 hops
          - Deduplicate using SHA1 hash of triplet IDs
        """
        all_chains = []
        self.visited.clear()

        def dfs(path, node, depth):
            # if a valid chain end
            if len(path) >= 2 and (len(path) == depth or G.out_degree(node) == 0):
                ids = ','.join(p['triplet_id'] for p in path)
                h = __import__('hashlib').sha1(ids.encode()).hexdigest()
                if h not in self.visited:
                    self.visited.add(h)
                    all_chains.append(path.copy())
                return
            for entity_2, rel in self.valid_succ.get(node, []):
                triplet = self._lookup(node, rel, entity_2)
                if triplet:
                    dfs(path + [{
                        'entity_1': node,
                        'relation': rel,
                        'entity_2': entity_2,
                        'triplet_id': triplet['triplet_id'],
                        'answer_chunk_id': triplet['answer_chunk_id'],
                        'metadata': triplet['metadata'],
                        'source_sentence': triplet.get('source_sentence', '')
                    }], entity_2, depth)

        # start DFS from every node
        for depth in range(2, max_depth + 1):
            for start in G.nodes():
                dfs([], start, depth)
        return all_chains

    def star_shaped(self, G: nx.DiGraph) -> list[list[dict]]:
        results = []
        self.visited.clear()
        
        # Iterate through all nodes in the graph as potential roots
        for root in G.nodes():
            # Get all valid outgoing edges from this root node
            outgoing_edges = []
            for nbr in G.successors(root):
                rel = G.edges[root, nbr].get('label', '')
                triplet = self._lookup(root, rel, nbr)
                if triplet:
                    outgoing_edges.append((nbr, rel, triplet))
            
            # Skip if fewer than 2 outgoing edges
            if len(outgoing_edges) < 2:
                continue
                
            # Limit to max 4 outgoing edges if we have more
            outgoing_edges = outgoing_edges[:4]
            
            # Generate all valid combinations of size 2, 3, and 4
            for size in range(2, min(5, len(outgoing_edges) + 1)):
                for combo in __import__('itertools').combinations(outgoing_edges, size):
                    chain = []
                    for nbr, rel, t in combo:
                        chain.append({
                            'entity_1': root, 
                            'relation': rel, 
                            'entity_2': nbr,
                            'triplet_id': t['triplet_id'], 
                            'answer_chunk_id': t['answer_chunk_id'], 
                            'metadata': t['metadata'], 
                            'source_sentence': t.get('source_sentence', '')
                        })
                    
                    # Deduplicate using hash of triplet IDs
                    ids = ','.join(c['triplet_id'] for c in chain)
                    h = __import__('hashlib').sha1(ids.encode()).hexdigest()
                    if h not in self.visited:
                        self.visited.add(h)
                        results.append(chain)
        
        return results

    def inverted_star_shaped(self, G: nx.DiGraph) -> list[list[dict]]:
        results = []
        self.visited.clear()
        # map each attribute to all (entity, relation, triplet)
        attr_map = {}
        for e1, attr, data in G.edges(data=True):
            rel = data.get('label', '')
            t = self._lookup(e1, rel, attr)
            if t:
                attr_map.setdefault(attr, []).append((e1, rel, t))
        for attr, entries in attr_map.items():
            if len(entries) < 2:
                continue
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    e1, rel1, t1 = entries[i]
                    e3, rel2, t2 = entries[j]
                    pair = [
                        {'entity_1': e1, 'relation': rel1, 'entity_2': attr,
                         'triplet_id': t1['triplet_id'], 'answer_chunk_id': t1['answer_chunk_id'], 'metadata': t1['metadata'],
                         'source_sentence': t1.get('source_sentence', '')},
                        {'entity_1': e3, 'relation': rel2, 'entity_2': attr,
                         'triplet_id': t2['triplet_id'], 'answer_chunk_id': t2['answer_chunk_id'], 'metadata': t2['metadata'],
                         'source_sentence': t2.get('source_sentence', '')}
                    ]
                    ids = ','.join(p['triplet_id'] for p in pair)
                    h = __import__('hashlib').sha1(ids.encode()).hexdigest()
                    if h not in self.visited:
                        self.visited.add(h)
                        results.append(pair)
        return results

    def cycle_path(self, G: nx.DiGraph) -> list[list[dict]]:
        results = []
        self.visited.clear()
        # use networkx to find simple cycles
        for cycle in nx.simple_cycles(G):
            if len(cycle) < 3:
                continue
            triplets = []
            valid_cycle = True
            for idx in range(len(cycle)):
                src = cycle[idx]
                dst = cycle[(idx + 1) % len(cycle)]
                rel = G.edges[src, dst].get('label', '')
                t = self._lookup(src, rel, dst)
                if not t:
                    valid_cycle = False
                    break
                triplets.append({
                    'entity_1': src, 'relation': rel, 'entity_2': dst,
                    'triplet_id': t['triplet_id'], 'answer_chunk_id': t['answer_chunk_id'], 'metadata': t['metadata']
                })
            if not valid_cycle:
                continue
            ids = ','.join(t_['triplet_id'] for t_ in triplets)
            h = __import__('hashlib').sha1(ids.encode()).hexdigest()
            if h not in self.visited:
                self.visited.add(h)
                results.append(triplets)
        return results

    def traverse(self, kg_dir: str) -> dict:
        all_sets = {
            'single_hop': [],
            'chained_multi_hop': [],
            'star_shaped': [],
            'inverted_star_shaped': [],
        }
        for graph_file in Path(kg_dir).glob("*.graphml"):
            print(f"Processing {graph_file}")
            G = self.load_graph(str(graph_file))
            self._prepare(G)
            all_sets['single_hop'] += self.single_hop(G)
            all_sets['chained_multi_hop'] += self.chained_multi_hop(G)
            all_sets['star_shaped'] += self.star_shaped(G)
            all_sets['inverted_star_shaped'] += self.inverted_star_shaped(G)
        return all_sets


# QA Generation via LLM
class QAGenerator:
    """
    Wraps LLM calls to generate QA pairs from triplets using domain-specific prompts.
    """
    def __init__(self, model_name: str, domain: str = "finance"):
        self.model = model_name
        self.domain = domain
        # load prompt templates by domain
        self.prompts = {
            "finance": {
                "single": [finance_onehop_query_prompt_system, finance_onehop_query_prompt_user],
                "multi": [finance_multihop_query_prompt_system, finance_multihop_query_prompt_user]
            },
            "economics": {
                "single": [econ_onehop_query_prompt_system, econ_onehop_query_prompt_user],
                "multi": [econ_multihop_query_prompt_system, econ_multihop_query_prompt_user]
            },
            "policy": {
                "single": [policy_onehop_query_prompt_system, policy_onehop_query_prompt_user],
                "multi": [policy_multihop_query_prompt_system, policy_multihop_query_prompt_user]
            }
        }

    def _call_llm(self, system_prompt: str, user_prompt: str, fmt):
        """Execute an LLM completion with a system prompt and structured output."""
        return litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=512,
            response_format=fmt
        )

    def generate_single_hop(self, triplet: dict, chunk: dict) -> dict | None:
        """
        Build and run single-hop prompt; return a QA dict or None if skipped.
        """
        if self.domain == "finance":
            # combine chunk texts and metadata for prompt context
            chunk_text = chunk["text"]
            metadata = chunk['metadata']
            prompt_template = self.prompts[self.domain]["single"]
            system_prompt = prompt_template[0]
            user_prompt = prompt_template[1].format(
                entity_1 = triplet['entity_1'],
                relation = triplet['relation'],
                entity_2 = triplet['entity_2'],
                chunk_text = chunk_text,
                metadata=metadata
            )
        elif self.domain == "economics":
            chunk_text = chunk["text"]
            metadata = chunk.get("metadata", {})
            file_type = metadata.get("file_type", "")
            file_country = metadata.get("file_country", "")
            file_year = metadata.get("file_year", "")
            
            prompt_template = self.prompts[self.domain]["single"]
            system_prompt = prompt_template[0].format(
                file_country = file_country
            )
            user_prompt = prompt_template[1].format(
                triplet = '{"entity_1": "' + triplet['entity_1'] + '", "relation": "' + triplet['relation'] + '", "entity_2": "' + triplet['entity_2'] + '"}',
                chunk_text = chunk_text,
                file_type = file_type,
                file_country = file_country,
                file_year = file_year
            )
        
        elif self.domain == 'policy':
            prompt_template = self.prompts[self.domain]["single"]
            
            chunk_text = chunk["text"]
            
            metadata = chunk.get("metadata", {})
            # change the file_year to string
            metadata["file_year"] = str(metadata.get("file_year", ""))
            file_type = metadata.get("file_type", "")
            file_grantee = metadata.get("file_grantee", "")
            file_year = metadata.get("file_year", "")
            
            system_prompt = prompt_template[0]
            user_prompt = prompt_template[1].format(
                triplet = '{"entity_1": "' + triplet['entity_1'] + '", "relation": "' + triplet['relation'] + '", "entity_2": "' + triplet['entity_2'] + '"}',
                chunk_text = chunk_text,
                file_type = file_type,
                file_grantee = file_grantee,
                file_year = file_year
            )
        
        qa: QAPair = self._call_llm(system_prompt, user_prompt, QAPair)
        return qa.model_dump() if qa else None

    def generate_multi_hop(self, triplets: list[dict], chunks: list[dict], multi_hop_type: str) -> list[dict]:
        """
        Build and run multi-hop prompt; return list of QA dicts.
        """
        # triplets with corresponding triplet id
        triplet_ids = [triplet['triplet_id'] for triplet in triplets]
        triplets_str = "\n".join(f"Triplet {triplet_ids[i]}, Answer Chunk ID: {triplet['answer_chunk_id']}:\n Entity 1: {triplet['entity_1']}\n Relation: {triplet['relation']}\n Entity 2: {triplet['entity_2']}" for i, triplet in enumerate(triplets))
        chunk_text = "\n\n".join(f"Answer Chunk ID: {chunk['chunk_id']}:\n{chunk['text']}" for i, chunk in enumerate(chunks)) # chunk text with corresponding triplet id
        # metadata with corresponding triplet id
        if self.domain == "finance":
            metadata_str = "\n".join(f"Metadata for triplet {triplet_ids[i]}:\n{json.dumps(chunk.get('metadata', {}))}" for i, chunk in enumerate(chunks))
        elif self.domain == "economics":
            metadata_str = f"Metadata for the triplet extraction file:\n{json.dumps(triplets[0]['metadata'])}"
        elif self.domain == "policy":
            metadata_str = f"Metadata for the triplet extraction file:\n{json.dumps(triplets[0]['metadata'])}"
        
        print(f"Triplet IDs: {triplet_ids}")
        print(f"Triplet String: {triplets_str}")
        print("-" * 50)
        
        prompt_template = self.prompts[self.domain]["multi"]
        system_prompt = prompt_template[0]
        user_prompt = prompt_template[1].format(
            triplets=triplets_str,
            chunk_text=chunk_text,
            metadata=metadata_str,
            multi_hop_type=multi_hop_type
        )
        qa: QAPair = self._call_llm(system_prompt, user_prompt, QAPair)
        return qa.model_dump() if qa else None


# --- Main Entrypoint ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QA dataset from GraphML KGs")
    parser.add_argument("--domain", choices=["finance","economics","policy"], default="finance")
    parser.add_argument("--kg_dir", required=True, help="Path to GraphML files")
    parser.add_argument("--triplets_file", default="/mnt/storage/rag-robust-eval/data/finance/benchmark/triplets.json", help="Path to triplets.json")
    parser.add_argument("--chunks_dir", required=True, help="Path to chunks directory")
    parser.add_argument("--output_dir", required=True, help="Output QA dataset directory")
    parser.add_argument("--model", default="openai/gpt-4.1", help="LLM model name")
    args = parser.parse_args()

    # 1) Extract and save all triplets for review
    traversed_sets_path = os.path.join(args.output_dir, 'kg_traverse_triplets.json')
    if os.path.exists(traversed_sets_path):
        print(f"Load traversed triplets from kg_traverse_triplets.json")
        with open(traversed_sets_path, 'r') as f:
            sets = json.load(f)
    else:
        print(f"Extracting triplets from knowledge graph...")
        kt = KGTraverser(args.triplets_file)
        sets = kt.traverse(args.kg_dir)
        
        # Check if the multi-hop triplets are extracted from different chunks
        filtered_sets = {
            'single_hop': sets['single_hop'],
            'chained_multi_hop': [],
            'star_shaped': [],
            'inverted_star_shaped': [],
        }
        for multi_hop_type in ['chained_multi_hop', 'star_shaped', 'inverted_star_shaped']:
            for group in sets[multi_hop_type]:
                ids = {triplet['answer_chunk_id'] for triplet in group}
                if len(ids) == len(group):
                    filtered_sets[multi_hop_type].append(group)
        sets = filtered_sets
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(traversed_sets_path, 'w') as f:
            json.dump(sets, f, indent=2)
        print(f"Extracted triplets saved to {traversed_sets_path}")
    
    print(f"Extracted {len(sets['single_hop'])} single-hop triplets, {len(sets['chained_multi_hop'])} chained multi-hop triplets, {len(sets['star_shaped'])} star-shaped triplets, {len(sets['inverted_star_shaped'])} inverted_star-shaped triplets")
    
    # 2) Load chunks into a dict
    # Merge chunks from different files first
    chunks_data = []
    for file in Path(args.chunks_dir).glob("*.json"):
        try:
            chunks = pd.read_json(file).to_dict(orient='records')
        except:
            chunks = pd.read_json(file, lines=True).to_dict(orient='records')
        chunks_data.extend(chunks)
    chunks_map = {c['chunk_id']: c for c in chunks_data}

    # 3) Generate QA pairs
    generator = QAGenerator(args.model, domain=args.domain)
    output_qa = {'single_hop': [], 'multi_hop': []}

    # Single-hop QA
    for triplet in tqdm(sets['single_hop'], desc="Single-hop QA"):
        if triplet['answer_chunk_id'] in chunks_map:
            chunk = chunks_map[triplet['answer_chunk_id']]
        else:
            print(f"Chunk {triplet['answer_chunk_id']} not found in chunks_map")
            continue
        qa = generator.generate_single_hop(triplet, chunk)
        qa = qa['choices'][0]['message']['content']
        qa = ast.literal_eval(qa)
        if qa and qa['question'] and qa['answer']:
            qa.update({
                'answer_chunk_ids': [triplet['answer_chunk_id']], 
                'metadata': triplet['metadata'],
                'entity_1': [triplet['entity_1']],
                'relation': [triplet['relation']],
                'entity_2': [triplet['entity_2']],
                'source_sentences': [triplet['source_sentence']]
                })
            output_qa['single_hop'].append(qa)
            
            # write qa to file
            with open(os.path.join(args.output_dir, 'updated_rag_single_hop_dataset.json'), 'a') as f:
                json.dump(qa, f, indent=2)
                f.write('\n')
    print(f"Generated {len(output_qa['single_hop'])} single-hop QA pairs from {len(sets['single_hop'])} triplets")

    # Multi-hop QA (chained, star, inverted_star)
    for multi_hop_type in ['chained_multi_hop', 'star_shaped', 'inverted_star_shaped']:
        random.seed(42)
        sample_set = random.sample(sets[multi_hop_type], min(30000, len(sets[multi_hop_type])))
        
        for group in tqdm(sample_set, desc=f"{multi_hop_type} QA"):
            ids = {triplet['answer_chunk_id'] for triplet in group}
            chunk_list = [chunks_map[cid] for cid in ids if cid in chunks_map]
            qa = generator.generate_multi_hop(group, chunk_list, multi_hop_type)
            qa = qa['choices'][0]['message']['content']
            qa = ast.literal_eval(qa)
            
            # Get the pivot entity for true multi-hop checking
            if multi_hop_type == 'chained_multi_hop':
                pivot_entity = [triplet['entity_2'] for triplet in group[:-1]]
            elif multi_hop_type == 'star_shaped':
                pivot_entity = [group[0]['entity_1']]
            elif multi_hop_type == 'inverted_star_shaped':
                pivot_entity = [group[0]['entity_2']]
            
            if qa:
                if not qa['answer'] or not qa['question']:
                    print(f"QA pair is invalid: {qa}, missing question or answer")
                    continue
                if qa['answer'] in qa['question']:
                    print(f"QA pair is invalid: {qa}, answer is in the question")
                    continue
                if any(pe in qa['question'] for pe in pivot_entity):
                    print(f"QA pair is invalid: {qa}, pivot entity is in the question")
                    continue
            
                qa.update({
                    'answer_chunk_ids': list(ids),
                    'metadata': group[0]['metadata'] | group[-1]['metadata'],
                    'entity_1': [triplet['entity_1'] for triplet in group],
                    'relation': [triplet['relation'] for triplet in group],
                    'entity_2': [triplet['entity_2'] for triplet in group],
                    'source_sentences': [triplet['source_sentence'] for triplet in group],
                    'multi_hop_type': multi_hop_type,
                })
                output_qa['multi_hop'].append(qa)
                # write qa to file
                with open(os.path.join(args.output_dir, 'updated_rag_multi_hop_dataset.json'), 'a') as f:
                    json.dump(qa, f, indent=2)
                    f.write('\n')
        print(f"Generated {len(output_qa['multi_hop'])} multi-hop QA pairs from {len(sets[multi_hop_type])} {multi_hop_type} triplets")
        print('-'* 50)
    print(f"Generated {len(output_qa['multi_hop'])} multi-hop QA pairs from {len(sets[multi_hop_type])} triplets")

    # 4) Write QA dataset
    with open(os.path.join(args.output_dir, 'rag_single_hop_dataset.json'), 'w') as f:
        json.dump(output_qa['single_hop'], f, indent=2)
    with open(os.path.join(args.output_dir, 'rag_multi_hop_dataset.json'), 'w') as f:
        json.dump(output_qa['multi_hop'], f, indent=2)
        
    print(f"Wrote {len(output_qa['single_hop'])} single-hop QA pairs to {args.output_dir}/rag_single_hop_dataset.json")
    print(f"Wrote {len(output_qa['multi_hop'])} multi-hop QA pairs to {args.output_dir}/rag_multi_hop_dataset.json")