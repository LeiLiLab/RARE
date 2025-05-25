import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from rag_pipeline import Retriever, load_chunks
from utils.api_keys import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME

# Configure environment variables
os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
os.environ['AWS_REGION_NAME'] = AWS_REGION_NAME

# Define retrieval models to test
RETRIEVAL_MODELS = [
    {
        'embedding_model': 'intfloat/multilingual-e5-large-instruct',
        'reranker_model': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
        'name': 'e5-large'
    },
    {
        'embedding_model': 'jinaai/jina-embeddings-v3',
        'reranker_model': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
        'name': 'jina-embeddings-v3'
    },
    {
        'embedding_model': 'NovaSearch/stella_en_1.5B_v5',
        'reranker_model': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
        'name': 'stella-en-1.5B-v5'
    }
]
def run_and_store_retrievals(
    records: List[Dict[str, Any]],
    retrievers: Dict[str, Any],
    output_file: str
) -> None:
    """
    Run retrievers on all queries and save results to a JSON file.
    This separates the retrieval phase from the generation phase.
    """
    retrieval_results = {}
    
    for retrieve_config in retrievers:
        retriever = Retriever(
            embedding_model=retrieve_config['embedding_model'],
            reranker_model=retrieve_config['reranker_model'],
            index_dir="./faiss_index_" + retrieve_config['name']
        )
        
        print(f"Running retrieval batch with {retrieve_config['name']}...")
        retriever.load_index()
        
        # Prepare query data structure
        for rec in records:
            q = rec['question']
            query_id = rec.get('id', hash(q))
            
            # Initialize the result structure if not exists
            if query_id not in retrieval_results:
                retrieval_results[query_id] = {
                    "original_query": q,
                    "answer": rec['answer'],
                    "variants": {},
                    "retrievers": {}
                }
                
                # Store query variants
                for k, v in rec.get('query_perturbations', {}).items():
                    retrieval_results[query_id]["variants"][k] = v['perturbed_query'] 
            
            # Initialize retriever results if not exists
            if retrieve_config['name'] not in retrieval_results[query_id]["retrievers"]:
                retrieval_results[query_id]["retrievers"][retrieve_config['name']] = {}
        
        # Perform retrieval on all queries and their perturbations
        for rec in tqdm(records, desc=f"Processing with {retrieve_config['name']}"):
            query_id = rec.get('id', hash(rec['question']))
            retriever_name = retrieve_config['name']
            
            # Retrieve for original query
            original_query = rec['question']
            original_results = retriever.retrieve(original_query, k_retrank=3)
            original_chunk_ids = [doc.metadata.get('chunk_id') for doc in original_results]
            
            # Store original query results
            retrieval_results[query_id]["retrievers"][retriever_name]["original"] = original_chunk_ids
            
            # Process each query perturbation
            for perturb_type, perturb_data in rec.get('query_perturbations', {}).items():
                perturbed_query = perturb_data['perturbed_query']
                perturbed_results = retriever.retrieve(perturbed_query, k_retrank=3)
                perturbed_chunk_ids = [doc.metadata.get('chunk_id') for doc in perturbed_results]
                
                # Store perturbation results
                retrieval_results[query_id]["retrievers"][retriever_name][perturb_type] = perturbed_chunk_ids

        retriever.unload()
    
    # Save results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(retrieval_results, f, indent=2)
    
    print(f"Saved retrieval results to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run retrievals for RAG robustness evaluation')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing dataset JSON files')
    parser.add_argument('--chunks_file', type=str, required=True, help='Path to chunks JSON')
    parser.add_argument('--index_dir', type=str, default='./faiss_index', help='Base index directory')
    parser.add_argument('--output_dir', type=str, default='./retrieval_results', help='Directory to save retrieval results')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load chunks for index creation if needed
    langchain_docs, chunks = load_chunks(args.chunks_file)
    print(f"Loaded {len(chunks)} chunks")

    # Initialize retrievers with different embedding models
    for retriever_config in RETRIEVAL_MODELS:
        retriever = Retriever(
            embedding_model=retriever_config['embedding_model'],
            reranker_model=retriever_config['reranker_model'],
            index_dir=f"{args.index_dir}_{retriever_config['name']}"
        )
        retriever.name = retriever_config['name']
        
        # Check if index exists, otherwise create it
        if os.path.exists(f"{args.index_dir}_{retriever_config['name']}"):
            print(f"Loading index for {retriever_config['name']}...")
            retriever.load_index()
        else:
            print(f"Creating index for {retriever_config['name']}...")
            retriever.create_index(langchain_docs)
        retriever.unload()

    # Get all domain-specific dataset files
    dataset_files = []
    for filename in os.listdir(args.dataset_dir):
        if filename.endswith('.json'):
            dataset_files.append(os.path.join(args.dataset_dir, filename))
    
    if not dataset_files:
        print(f"No dataset files found in {args.dataset_dir}")
        exit(1)
    
    # Process each dataset file
    for dataset_file in dataset_files:
        filename = os.path.basename(dataset_file)
        domain = os.path.splitext(filename)[0]  # Extract domain from filename
        print(f"Processing dataset: {domain}")
        
        # Load dataset
        with open(dataset_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        # Run retrievals and save results
        retrieval_output = os.path.join(args.output_dir, f"{domain}_retrieval_results.json")
        run_and_store_retrievals(records, RETRIEVAL_MODELS, retrieval_output)
    
    print("All retrievals complete!") 