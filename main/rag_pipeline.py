import os
import json
from tqdm import tqdm
from typing import List, Optional, Tuple, Dict, Union
import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder


def load_chunks(chunks_path: str) -> Tuple[List[Document], List[Dict]]:
    """
    Load chunked documents from a JSON file and convert to LangChain Document objects.
    Note: This function returns a tuple of two lists:
    - The first list contains the LangChain Document objects.
    - The second list contains the original chunks as dictionaries.
    """
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    docs = []
    for chunk in chunks:
        docs.append(
            Document(
                page_content=chunk['text'],
                metadata={**chunk.get('metadata', {}), 'chunk_id': chunk.get('chunk_id')}
            )
        )
    return docs, chunks

class Retriever:
    def __init__(
        self,
        embedding_model: str = "intfloat/multilingual-e5-large-instruct",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        index_dir: Optional[str] = None
    ):
        """
        RAG pipeline retriever with FAISS + HuggingFace embeddings and a cross-encoder reranker.
        """
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.index_dir = index_dir

        self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_model, 
                                               encode_kwargs={'prompt': "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"},
                                               model_kwargs={"trust_remote_code": True})
        self.reranker = CrossEncoder(self.reranker_model)

        self.index: Optional[FAISS] = None
        self.retriever = None

    def create_index(self, docs: List[Document], **faiss_kwargs) -> None:
        """
        Build a FAISS index from provided documents and save locally if index_dir is set.
        """
        self.index = FAISS.from_documents(docs, self.embedding, **faiss_kwargs)
        if self.index_dir:
            os.makedirs(self.index_dir, exist_ok=True)
            self.index.save_local(self.index_dir)
        self.retriever = self.index.as_retriever()

    def load_index(self) -> None:
        """
        Load an existing FAISS index from index_dir.
        """
        if not self.index_dir:
            raise ValueError("index_dir must be set to load an index")
        self.index = FAISS.load_local(self.index_dir, self.embedding, allow_dangerous_deserialization=True)
        self.retriever = self.index.as_retriever(search_kwargs={"k": 5})

    def retrieve(self, query: str, k_retrank: int = 3) -> List[Document]:
        """
        Retrieve top documents by embedding search, then rerank with cross-encoder.
        """
        docs = self.retriever.invoke(query)
        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:k_retrank]]
    
    def batch_retrieve(self, queries: List[str], k_retrank: int = 3, batch_size: int = 100) -> Dict[str, List[Document]]:
        """
        Retrieve top documents for multiple queries, processing in batches for efficiency.
        
        Args:
            queries: List of query strings
            k_retrank: Number of documents to return after reranking
            batch_size: Size of batches for processing
            
        Returns:
            Dictionary mapping each query to its list of retrieved documents
        """
        results = {}
        
        all_docs = {}
        for query in tqdm(queries):
            all_docs[query] = self.retriever.invoke(query)
        
        all_pairs = []
        query_to_pair_indices = {}
        
        for query, docs in all_docs.items():
            start_idx = len(all_pairs)
            pairs = [(query, d.page_content) for d in docs]
            all_pairs.extend(pairs)
            query_to_pair_indices[query] = (start_idx, start_idx + len(pairs), docs)
        
        all_scores = []
        for i in range(0, len(all_pairs), batch_size):
            batch_pairs = all_pairs[i:i+batch_size]
            batch_scores = self.reranker.predict(batch_pairs, show_progress_bar=False)
            all_scores.extend(batch_scores)
        
        for query, (start_idx, end_idx, docs) in query_to_pair_indices.items():
            query_scores = all_scores[start_idx:end_idx]
            ranked = sorted(zip(query_scores, docs), key=lambda x: x[0], reverse=True)
            results[query] = [doc for _, doc in ranked[:k_retrank]]
        
        return results
    
    def unload(self):
        if hasattr(self, "embedding") and self.embedding is not None:
            if hasattr(self.embedding, "client"):
                try:
                    del self.embedding.client.model
                    del self.embedding.client.tokenizer
                except AttributeError:
                    pass
            del self.embedding

        if hasattr(self, "reranker") and self.reranker is not None:
            try:
                del self.reranker.model
                del self.reranker.tokenizer
            except AttributeError:
                pass
            del self.reranker

        if hasattr(self, "index") and self.index is not None:
            del self.index

        self.retriever = None

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG retrieval pipeline")
    parser.add_argument("--chunks_file", type=str, required=True, help="Path to chunks.json file")
    parser.add_argument("--index_dir", type=str, default="./faiss_index_e5", help="Directory to save/load FAISS index")
    parser.add_argument("--mode", choices=["create", "load", "query"], default="create")
    parser.add_argument("--query", type=str, help="Query string for retrieval")
    parser.add_argument("--embed_model", type=str, default="intfloat/multilingual-e5-large-instruct")
    parser.add_argument("--rerank_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    args = parser.parse_args()

    retriever = Retriever(
        embedding_model=args.embed_model,
        reranker_model=args.rerank_model,
        index_dir=args.index_dir
    )
    if args.mode == "create":
        docs, _ = load_chunks(args.chunks_file)
        retriever.create_index(docs)
        print("Index created and saved to", args.index_dir)
    elif args.mode == "load":
        retriever.load_index()
        print("Index loaded from", args.index_dir)
    else:
        retriever.load_index()
        query = "What do transaction-related and non-recurring items primarily consist of for Blackstone Inc.?"
        out = retriever.retrieve(query, k_retrank=3)
        for doc in out:
            print(f"ChunkID: {doc.metadata['chunk_id']}\n{doc.page_content[:200]}...")