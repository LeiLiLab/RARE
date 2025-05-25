import os
import ast
import json
from tqdm import tqdm

import litellm
import argparse
import pandas as pd
from pydantic import BaseModel
from utils.api_keys import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME

os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["AWS_REGION_NAME"] = AWS_REGION_NAME

qa_eval_system_prompt = {
    'multi_hop': '''
# Multi-Hop Query Quality Evaluator

You are an expert evaluator of multi-hop queries. Assess each query's quality across three dimensions on a 1-5 scale.

## Assessment Criteria

### 1. Reasonableness & Multi-hop Need (1-5)
- **5**: Meaningful question requiring all hops; each hop justified
- **4**: Reasonable but one hop weakly motivated or could be merged
- **3**: Sensible but answerable by single chunk with assumptions
- **2**: Forced/trivial question; multi-hop structure unnecessary
- **1**: Nonsensical/irrelevant; multi-hop structure meaningless

### 2. Clarity (Question & Answer) (1-5)
- **5**: Concise, unambiguous wording; answer mirrors clarity
- **4**: Minor wording issue but still unambiguous
- **3**: Some vagueness but meaning recoverable
- **2**: Ambiguities/redundancies hinder understanding
- **1**: Unclear or contradictory wording

### 3. Correctness (vs. Ground-Truth) (1-5)
- **5**: Answer matches all facts in chunks; nothing missing
- **4**: Correct but one minor fact omitted/loosely paraphrased
- **3**: At least half of facts correct; one factual slip
- **2**: Major fact missing/misstated/unsupported
- **1**: Contradicts or ignores ground truth

## Evaluation Process
1. Identify distinct reasoning hops and assess necessity
2. Check alignment between hops and provided chunks
3. Evaluate clarity of question and answer
4. Verify factual correctness against ground-truth chunks

## Input
- query: The multi-hop question
- answer: The provided answer
- text chunks: Source text chunks

## Output
{{
  "score": <average_of_dimension_scores>,
  "dimension_scores": {
    "reasonableness": <1-5>,
    "clarity": <1-5>,
    "correctness": <1-5>
  }
}}''',
    'single_hop': '''
# Single-Hop Query Quality Evaluator

You are an expert evaluator of single-hop queries. Assess each query's quality across two dimensions on a 1-5 scale.

## Assessment Criteria

### 1. Clarity (Question & Answer) (1-5)
- **5**: Concise, unambiguous wording; answer mirrors clarity
- **4**: Minor wording issue but still unambiguous
- **3**: Some vagueness but meaning recoverable
- **2**: Ambiguities/redundancies hinder understanding
- **1**: Unclear or contradictory wording

### 2. Correctness (vs. Ground-Truth) (1-5)
- **5**: Answer matches all facts in chunks; nothing missing
- **4**: Correct but one minor fact omitted/loosely paraphrased
- **3**: At least half of facts correct; one factual slip
- **2**: Major fact missing/misstated/unsupported
- **1**: Contradicts or ignores ground truth

## Evaluation Process
1. Identify reasoning process
2. Assess alignment between query and provided text chunk
3. Evaluate clarity of question and answer
4. Verify factual correctness against ground-truth chunk

## Input
- query: The single-hop question
- answer: The provided answer
- text chunk: Source text chunk

## Output
{{
  "score": <average_of_dimension_scores>,
  "dimension_scores": {
    "clarity": <1-5>,
    "correctness": <1-5>
  }
}}'''
}

qa_eval_user_prompt_multi_hop = '''
query: {query}

answer: {answer}

text chunks: {text_chunks}
'''

qa_eval_user_prompt_single_hop = '''
query: {query}

answer: {answer}

text chunk: {text_chunk}
'''


class DimensionScores_MultiHop(BaseModel):
    reasonableness_multi_hop_need: int
    clarity: int
    correctness: int

class DimensionScores_SingleHop(BaseModel):
    clarity: int
    correctness: int

class QAScore_MultiHop(BaseModel):
    score: float
    dimension_scores: DimensionScores_MultiHop

class QAScore_SingleHop(BaseModel):
    score: float
    dimension_scores: DimensionScores_SingleHop

def _call_llm(system_prompt: str, user_prompt: str, fmt):
    """Execute an LLM completion with a system prompt and structured output."""
    return litellm.completion(
        model='bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.8,
        max_tokens=512,
        response_format=fmt
    )

def evaluate_query(qa_dict: dict, chunks_map: dict, query_type: str):
    answer_chunk_ids = qa_dict['answer_chunk_ids']
    text_chunks = [chunk_id + '\n' + chunks_map[chunk_id] for chunk_id in answer_chunk_ids]
    
    if query_type == 'multi_hop':    
        system_prompt = qa_eval_system_prompt[query_type]
        user_prompt = qa_eval_user_prompt_multi_hop.format(
            query=qa_dict['question'],
            answer=qa_dict['answer'],
            text_chunks='\n\n'.join(text_chunks)
        )
        qa_score: QAScore_MultiHop = _call_llm(system_prompt, user_prompt, QAScore_MultiHop)
    elif query_type == 'single_hop':
        system_prompt = qa_eval_system_prompt[query_type]
        user_prompt = qa_eval_user_prompt_single_hop.format(
            query=qa_dict['question'],
            answer=qa_dict['answer'],
            text_chunk=text_chunks[0]
        )
        print(system_prompt)
        print(user_prompt)
        qa_score: QAScore_SingleHop = _call_llm(system_prompt, user_prompt, QAScore_SingleHop)
        
    return qa_score.model_dump() if qa_score else None


def main():
    parser = argparse.ArgumentParser(description="QA Quality Check")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/data/group_data/rag-robust-eval/data/economics/qa_dataset/rag_multi_hop_dataset.json",
        help="Path to the input QA file",
    )
    parser.add_argument(
        "--chunks_file",
        type=str,
        default="/data/group_data/rag-robust-eval/data/economics/chunks.json",
        help="Path to the input chunks file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/data/group_data/rag-robust-eval/data/economics/qa_dataset/multi_hop_quality_checked.json",
        help="Path to the output QA file",
    )
    parser.add_argument(
        "--query_type",
        type=str,
        default="multi_hop",
        help="Type of query to check (multi_hop or single_hop)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="economics",
        help="Domain of the dataset (finance, economics, policy)",
    )
    parser.add_argument(
        "--target_true_questions",
        type=int,
        default=1000,
        help="Target number of true questions to filter",
    )
    args = parser.parse_args()
    
    df = pd.read_json(args.input_file)
    print(f"Loaded {len(df)} records from {args.input_file}")
    
    if args.query_type == "multi_hop":
        print("="*20, "Filtering with multiple answer chunks", "="*20)
        df = df[df['answer_chunk_ids'].apply(len) > 1]
        print(f"Filtered to {len(df)} records with multiple answer chunks")
        
        multi_hop_types = df['multi_hop_type'].unique()
        for multi_hop_type in multi_hop_types:
            type_df = df[df['multi_hop_type'] == multi_hop_type]
            print(f"Filtered to {len(type_df)} records of type {multi_hop_type}")
    
    print("="*20, "Filtering with Claude-3.5", "="*20)
    # Load the chunks map
    chunks = pd.read_json(args.chunks_file).to_dict(orient='records')
    chunks_map = {chunk['chunk_id']: chunk['text'] for chunk in chunks}
    print(f"Loaded {len(chunks_map)} chunks from {args.chunks_file}")
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True).copy()
    
    df['flag'] = False
    df['qa_score'] = None
    true_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        qa_dict = row.to_dict()
        qa_score = evaluate_query(qa_dict, chunks_map, args.query_type)
        qa_score = qa_score['choices'][0]['message']['content']
        qa_score = ast.literal_eval(qa_score)
        
        if qa_score:
            print(qa_score)
            df.at[idx, 'qa_score'] = qa_score
            # check if all dimension scores are larger than 3
            try:
                dimension_scores = list(qa_score['dimension_scores'].values())
            except:
                print(f"Error parsing dimension scores for index {idx}")
                continue
            if all([score > 2 for score in dimension_scores]):
                df.at[idx, 'flag'] = True
                true_count += 1
                # write the true row to a new file
                with open(args.output_file.replace('.json', '_realtime.json'), 'a') as f:
                    json.dump(df.loc[idx].to_dict(), f, indent=2)
                    f.write('\n')
                if true_count == args.target_true_questions:
                    print(f"Reached {args.target_true_questions} records with all dimension scores larger than 3")
                    break
        
    # Filtering out the flagged records
    df_true = df[df['flag'] == True].copy()
    
    # Set final query ids
    df_true['id'] = df_true.index
    df_true['id'] = df_true['id'].astype(str).apply(lambda x: args.domain + "_" + x.zfill(6))
    # move the id column to the front
    cols = df_true.columns.tolist()
    cols.insert(0, cols.pop(cols.index('id')))
    df_true = df_true[cols]
    
    # Save the filtered records to a new file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df_true.to_json(args.output_file, orient='records')
    print(f"Filtered to {len(df_true)} records with all dimension scores larger than 3")
    
    
if __name__ == "__main__":
    main()