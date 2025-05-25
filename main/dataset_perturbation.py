import os
import re
import argparse
import json
from tqdm import tqdm

# TextAttack imports for perturbations
from textattack.transformations import (
    CompositeTransformation,
    WordSwapQWERTY,
    WordSwapRandomCharacterDeletion,
    WordSwapNeighboringCharacterSwap,
    WordSwapHomoglyphSwap,
    WordSwapEmbedding
)
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification
)
from textattack.augmentation import Augmenter
from sentence_transformers import SentenceTransformer
import litellm
import torch
from utils.prompts import (
    grammar_perturbation_prompt_system,
    grammar_perturbation_prompt_user,
    irrelevant_info_prompt_system,
    irrelevant_info_prompt_user
)

# Set your API key
from utils.api_keys import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Constants
SIMILARITY_MODEL = "intfloat/e5-mistral-7b-instruct"
SIMILARITY_THRESHOLD = 0.85
DEFAULT_LLM = "openai/gpt-4.1"

class SimilarityChecker:
    def __init__(self, model_name: str = SIMILARITY_MODEL):
        self.model = SentenceTransformer(model_name, model_kwargs={"torch_dtype": torch.bfloat16})

    def compute(self, original: str, perturbed: str) -> float:
        prompt = (
            "Instruction: Check the intrinsic similarity (not lexical similarity) "
            "between the following two sentences (or paragraphs).\nQuery: "
        )
        emb1 = self.model.encode(original, prompt=prompt)
        emb2 = self.model.encode(perturbed)
        return float(emb1 @ emb2.T)

class QueryPerturbator:
    def __init__(self, sim_checker: SimilarityChecker):
        self.sim = sim_checker
        char_transforms = CompositeTransformation([
            WordSwapQWERTY(),
            WordSwapRandomCharacterDeletion(),
            WordSwapNeighboringCharacterSwap(),
            WordSwapHomoglyphSwap()
        ])
        char_constraints = [RepeatModification(), StopwordModification()]
        self.char_aug = Augmenter(
            transformation=char_transforms,
            constraints=char_constraints,
            pct_words_to_swap=0.1,
            transformations_per_example=1
        )
        word_transforms = WordSwapEmbedding(max_candidates=50)
        word_constraints = [RepeatModification(), StopwordModification()]
        self.word_aug = Augmenter(
            transformation=word_transforms,
            constraints=word_constraints,
            pct_words_to_swap=0.1,
            transformations_per_example=1
        )

    def char_level(self, query: str):
        candidates = []
        for _ in range(5):
            out = self.char_aug.augment(query)
            if out:
                cand = out[0]
                sim = self.sim.compute(query, cand)
                candidates.append((cand, sim))
                if sim >= SIMILARITY_THRESHOLD:
                    return cand, sim
        if candidates:
            return max(candidates, key=lambda x: x[1])

    def word_level(self, query: str):
        candidates = []
        for _ in range(5):
            out = self.word_aug.augment(query)
            if out:
                cand = out[0]
                sim = self.sim.compute(query, cand)
                candidates.append((cand, sim))
                if sim >= SIMILARITY_THRESHOLD:
                    return cand, sim
        if candidates:
            return max(candidates, key=lambda x: x[1])

    def irrelevant_info(self, query: str):
        candidates = []

        resp = litellm.batch_completion(
            model=DEFAULT_LLM,
            messages=[
                [
                    {"role":"system","content":irrelevant_info_prompt_system},
                    {"role":"user","content":irrelevant_info_prompt_user.format(query=query)}
                ],
                [
                    {"role":"system","content":irrelevant_info_prompt_system},
                    {"role":"user","content":irrelevant_info_prompt_user.format(query=query)}
                ],
                [
                    {"role":"system","content":irrelevant_info_prompt_system},
                    {"role":"user","content":irrelevant_info_prompt_user.format(query=query)}
                ],
            ],
            temperature=0.7
        )
        for i in range(3):
            cand = resp[i].choices[0].message.content.strip()
            sim = self.sim.compute(query, cand)
            candidates.append((cand, sim))
        if candidates:
            return max(candidates, key=lambda x: x[1])

    def grammar_perturbation(self, query: str):
        candidates = []
        resp = litellm.batch_completion(
            model=DEFAULT_LLM,
            messages=[
                [
                    {"role":"system","content":grammar_perturbation_prompt_system},
                    {"role":"user","content":grammar_perturbation_prompt_user.format(query=query)}
                ],
                [
                    {"role":"system","content":grammar_perturbation_prompt_system},
                    {"role":"user","content":grammar_perturbation_prompt_user.format(query=query)}
                ],
                [
                    {"role":"system","content":grammar_perturbation_prompt_system},
                    {"role":"user","content":grammar_perturbation_prompt_user.format(query=query)}
                ],
            ],
            temperature=0.7
        )
        for i in range(3):
            cand = resp[i].choices[0].message.content.strip()
            sim = self.sim.compute(query, cand)
            candidates.append((cand, sim))
        if candidates:
            return max(candidates, key=lambda x: x[1])

    def generate(self, query: str) -> dict:
        return {
            'character_level': self.char_level(query),
            'word_level': self.word_level(query),
            'irrelevant_info': self.irrelevant_info(query),
            'grammar_perturbation': self.grammar_perturbation(query)
        }

class DocumentPerturbator:
    def __init__(self, chunks_list: list, sim_checker: SimilarityChecker):
        # Create mapping from chunk_id to text
        self.chunks = {chunk['chunk_id']: chunk['text'] for chunk in chunks_list}
        self.sim = sim_checker

    def regex_delete(self, chunk_ids, sentences):
        results = []
        for cid in tqdm(chunk_ids, desc="Generating regex perturbations"):
            orig = self.chunks.get(cid, '')
            txt = orig
            for sent in sentences:
                txt = re.sub(re.escape(sent), '', txt, flags=re.IGNORECASE)
            sim = self.sim.compute(orig, txt)
            results.append({
                'chunk_id': cid,
                'text': txt,
                'similarity': sim,
            })
        return results

    def back_translate(self, chunk_ids):
        results = []
        # Collect all original chunks
        originals = {cid: self.chunks.get(cid, '') for cid in chunk_ids}
        
        # Create batch messages for French translation
        french_messages = []
        for cid, text in originals.items():
            french_messages.append([{"role": "user", "content": f"Translate to French:\n{text}"}])
        
        # Batch translate to French
        french_responses = litellm.batch_completion(
            model=DEFAULT_LLM,
            messages=french_messages,
            temperature=0
        )
        
        # Extract French translations
        french_translations = {}
        for i, cid in enumerate(originals.keys()):
            french_translations[cid] = french_responses[i].choices[0].message.content.strip()
        
        # Create batch messages for English translation
        english_messages = []
        for cid, text in french_translations.items():
            english_messages.append([{"role": "user", "content": f"Translate to English:\n{text}"}])
        
        # Batch translate back to English
        english_responses = litellm.batch_completion(
            model=DEFAULT_LLM,
            messages=english_messages,
            temperature=0
        )
        
        # Process results
        for i, cid in enumerate(originals.keys()):
            orig = originals[cid]
            en = english_responses[i].choices[0].message.content.strip()
            sim = self.sim.compute(orig, en)
            results.append({
                'chunk_id': cid,
                'text': en,
                'similarity': sim,
            })
        
        return results


def main(args):
    # Load original RG dataset JSON
    with open(args.input_file, 'r') as f:
        records = json.load(f)

    # Load chunks JSON
    with open(args.chunks_file, 'r') as f:
        chunks_list = json.load(f)

    # Collect all unique answer_chunk_ids and regex patterns
    all_chunk_ids = set()
    regex_patterns = set()
    for rec in tqdm(records, desc="Collecting unique answer_chunk_ids and regex patterns"):
        answer_chunk_ids = rec.get('answer_chunk_ids', [])
        source_sentences = rec.get('source_sentences', [])
        all_chunk_ids.update(answer_chunk_ids)
        if answer_chunk_ids and source_sentences:
            regex_patterns.add((tuple(answer_chunk_ids), tuple(source_sentences)))

    sim_checker = SimilarityChecker()
    query_perturbator = QueryPerturbator(sim_checker)
    doc_perturbator = DocumentPerturbator(chunks_list, sim_checker)

    # Pre-generate document perturbations
    back_trans_map = {
        cid: pt for cid, pt in zip(
            all_chunk_ids,
            doc_perturbator.back_translate(list(all_chunk_ids))
        )
    }
    regex_map = {
        patterns: doc_perturbator.regex_delete(list(patterns[0]), list(patterns[1]))
        for patterns in regex_patterns
    }

    # Append perturbations to each record
    for rec in tqdm(records):
        rec['query_perturbations'] = {
            key: {
                'perturbed_query': qp_,
                'similarity': sim,
            }
            for key, (qp_, sim) in query_perturbator.generate(rec['question']).items()
        }
        answer_chunk_ids = rec.get('answer_chunk_ids', [])
        source_sentences = rec.get('source_sentences', [])
        rec['document_perturbations'] = {}
        if answer_chunk_ids:
            rec['document_perturbations']['back_translation'] = [back_trans_map[cid] for cid in answer_chunk_ids]
            rec['document_perturbations']['regex_delete'] = regex_map.get((tuple(answer_chunk_ids), tuple(source_sentences)), [])

    # Save augmented JSON
    with open(args.output_file, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"Augmented {len(records)} records and saved to {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='Original rag_dataset_generator JSON')
    parser.add_argument('--output_file', required=True, help='Path for augmented JSON')
    parser.add_argument('--chunks_file', required=True, help='Chunks file path')
    args = parser.parse_args()
    main(args)
    
'''
python dataset_perturbation.py \
    --input_file /home/yixiaozeng/rag-robust-eval/main/finance_multi_hop.json \
    --output_file /home/yixiaozeng/rag-robust-eval/main/finance_multi_hop_perturbed.json\
    --chunks_file /mnt/storage/rag-robust-eval/data/benchmark/chunks.json

'''
