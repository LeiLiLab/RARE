import os
import re
import tiktoken
import pandas as pd
from typing import Dict, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse
import json

tokenizer = tiktoken.get_encoding("cl100k_base")

class BaseChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=lambda text: len(tokenizer.encode(text))
        )
    
    def load_document(self, file_path: str) -> Dict:
        raise NotImplementedError("Subclasses must implement load_document")
    
    def chunk_document(self, file_path: str) -> List[Dict]:
        raise NotImplementedError("Subclasses must implement chunk_document")
    
    def save_chunks(self, chunks: List[Dict], output_dir: str, domain: str, document_id: str):
        """
        Save chunks for a single document to its own JSON file within the output directory.
        
        Args:
            chunks: List of chunk dictionaries to save
            output_dir: Path to the output directory
            domain: Domain of the current chunks (finance, economy, politics)
            document_id: Unique identifier for the document (e.g., CIK for finance)
        """
        os.makedirs(output_dir, exist_ok=True)
        document_file_path = os.path.join(output_dir, f"{domain}_{document_id}_chunks.json")
        
        pd.DataFrame(chunks).to_json(document_file_path, orient='records', indent=2)
        
        print(f"Saved {len(chunks)} {domain} chunks for document {document_id} to {document_file_path}")
        
        index_file = os.path.join(output_dir, "document_index.json")
        document_index = []
        
        if os.path.exists(index_file):
            try:
                with open(index_file, 'r') as f:
                    document_index = json.load(f)
            except Exception:
                print(f"Error loading {index_file}, creating new index")
                document_index = []
        
        document_info = {
            "document_id": document_id,
            "domain": domain,
            "chunk_file": os.path.basename(document_file_path),
            "chunk_count": len(chunks)
        }
        
        exists = False
        for i, doc in enumerate(document_index):
            if doc.get("document_id") == document_id and doc.get("domain") == domain:
                document_index[i] = document_info
                exists = True
                break
        
        if not exists:
            document_index.append(document_info)
        
        with open(index_file, 'w') as f:
            json.dump(document_index, f, indent=2)
        
        print(f"Updated document index with {len(document_index)} documents")

class FinanceChunker(BaseChunker):
    def load_document(self, file_path: str) -> Dict:
        document_df = pd.read_json(file_path, typ='series')
        return document_df.to_dict()
    
    def _is_section_title(self, text: str) -> bool:
        """
        Improved section title detection for SEC 10-K reports.
        Recognizes standard section headers like "ITEM X." and other common patterns.
        """
        patterns = [
            r'^ITEM\s+\d+[A-Z]?\.',
            r'^[A-Z][A-Za-z\s]+$',
            r'^\d+\.\s+[A-Z][A-Za-z\s]+$',
            r'^[A-Z][A-Za-z\s]+ and [A-Z][A-Za-z\s]+$',
            r'^[A-Z][A-Za-z\s]+ of [A-Z][A-Za-z\s]+$',
            r'^[A-Z][A-Za-z\s]+:[A-Za-z\s]+$'
        ]
        
        if len(text.split()) <= 10:
            for pattern in patterns:
                if re.match(pattern, text.strip()):
                    return True
        return False

    def _contains_table(self, text: str) -> bool:
        """
        Improved table detection that looks for markdown table patterns.
        """
        return ('|' in text and 
                (('-|-' in text) or 
                 ('--' in text and '|' in text) or
                 (text.count('|') >= 3)))

    def _preprocess_document(self, item_7_text: str) -> List[str]:
        """
        Ensure titles are associated with content and prevent standalone title chunks.
        """
        paragraphs = item_7_text.split("\n")
        segments = []
        current_segment = ""
        table_buffer = []
        in_table = False
        
        i = 0
        while i < len(paragraphs):
            para = paragraphs[i].strip()
            
            if not para:
                i += 1
                continue
            
            is_title = self._is_section_title(para)
            
            if is_title and not in_table:
                if current_segment.strip():
                    segments.append(current_segment.strip())
                
                current_segment = para + "\n\n"
                i += 1
                
                continue
            
            if self._contains_table(para) and not in_table:
                in_table = True
                table_buffer = []
                
                table_context_start = max(0, i-2)
                for j in range(table_context_start, i):
                    if paragraphs[j].strip() and not self._is_section_title(paragraphs[j].strip()):
                        table_buffer.append(paragraphs[j])
                
                table_buffer.append(para)
                i += 1
                continue
            
            if in_table:
                table_buffer.append(para)
                
                end_table = False
                if len(para) > 0:
                    if '|' not in para:
                        end_table = True
                    elif not self._contains_table(para) and i+1 < len(paragraphs) and '|' not in paragraphs[i+1]:
                        end_table = True
                
                if end_table:
                    in_table = False
                    table_content = "\n".join(table_buffer)
                    
                    if len(current_segment.strip().split()) > 30:
                        segments.append(current_segment.strip())
                        current_segment = table_content + "\n\n"
                    else:
                        current_segment += table_content + "\n\n"
                    
                    table_buffer = []
                i += 1
                continue
            
            current_segment += para + "\n\n"
            i += 1
        
        if current_segment.strip():
            segments.append(current_segment.strip())
        
        final_segments = []
        for idx, segment in enumerate(segments):
            lines = segment.split('\n')
            first_line = lines[0].strip() if lines else ""
            
            if self._is_section_title(first_line) and len(segment.split()) < 30:
                if idx < len(segments) - 1:
                    segments[idx + 1] = segment + "\n\n" + segments[idx + 1]
                elif final_segments:
                    final_segments[-1] += "\n\n" + segment
                else:
                    final_segments.append(segment)
            else:
                final_segments.append(segment)
        
        return final_segments
    
    def chunk_document(self, file_path: str) -> List[Dict]:
        document = self.load_document(file_path)
        chunks = []
        
        if "item_7" not in document:
            print(f"Warning: item_7 not found in document for CIK {document.get('cik', 'unknown')}")
            return []
        
        sp500_info = pd.read_csv('/mnt/storage/rag-robust-eval/data/finance/sp500_companies.csv')
        
        doc_cik = document.get("cik", "")
        
        company_info = sp500_info[sp500_info['CIK'] == int(doc_cik)] if doc_cik.isdigit() else None
        
        gics_sector = ""
        gics_subindustry = ""
        if company_info is not None and not company_info.empty:
            gics_sector = company_info['GICS_Sector'].values[0]
            gics_subindustry = company_info['GICS_SubIndustry'].values[0]
        
        metadata = {
            "cik": doc_cik,
            "company": document.get("company", ""),
            "filing_type": document.get("filing_type", ""),
            "filing_date": document.get("filing_date", ""),
            "period_of_report": document.get("period_of_report", ""),
            "GICS_Sector": gics_sector,
            "GICS_SubIndustry": gics_subindustry,
            "domain": "finance"
        }
        
        segments = self._preprocess_document(document["item_7"])
        
        chunk_index = 0
        for segment in segments:
            lines = segment.split('\n')
            first_line = lines[0].strip() if lines else ""
            
            is_section_title = self._is_section_title(first_line)
            section_title = first_line if is_section_title else "Untitled Section"
            
            contains_table = self._contains_table(segment)
            chunk_id = f"{metadata['domain']}_{metadata['cik']}_{metadata['filing_date']}_chunk_{chunk_index}"
            
            if contains_table:
                chunk_metadata = metadata.copy()
                chunk_metadata["contains_table"] = True
                
                chunk_data = {
                    "chunk_id": chunk_id,
                    "text": segment,
                    "metadata": chunk_metadata
                }
                chunks.append(chunk_data)
                chunk_index += 1
            else:
                recursive_chunks = self.text_splitter.split_text(segment)
                
                processed_chunks = []
                current_chunk = ""
                
                for chunk_text in recursive_chunks:
                    if not chunk_text.strip():
                        continue
                        
                    if len(chunk_text.split()) < 30:
                        if current_chunk:
                            current_chunk += "\n\n" + chunk_text
                        else:
                            current_chunk = chunk_text
                    else:
                        if current_chunk:
                            processed_chunks.append(current_chunk)
                            current_chunk = ""
                        processed_chunks.append(chunk_text)
                
                if current_chunk:
                    processed_chunks.append(current_chunk)
                
                for i, chunk_text in enumerate(processed_chunks):
                    if chunk_text.strip():
                        if i == 0 and is_section_title and first_line not in chunk_text:
                            chunk_text = first_line + "\n\n" + chunk_text
                        chunk_metadata = metadata.copy()
                        chunk_metadata["contains_table"] = False
                        
                        chunk_data = {
                            "chunk_id": chunk_id,
                            "text": chunk_text.strip(),
                            "metadata": chunk_metadata
                        }
                        chunks.append(chunk_data)
                        chunk_index += 1
        
        return chunks


class EconomicsChunker(BaseChunker):
    def load_document(self, file_path: str) -> Dict:
        raise NotImplementedError("Economy chunker not yet implemented")
    
    def chunk_document(self, file_path: str) -> List[Dict]:
        raise NotImplementedError("Economy chunker not yet implemented")


class PoliticsChunker(BaseChunker):
    def load_document(self, file_path: str) -> Dict:
        raise NotImplementedError("Politics chunker not yet implemented")
    
    def chunk_document(self, file_path: str) -> List[Dict]:
        raise NotImplementedError("Politics chunker not yet implemented")

class CorpusChunker(BaseChunker):
    def load_document(self, file_path: str) -> Dict:
        raise NotImplementedError("Corpus chunker not yet implemented")
    
    def chunk_document(self, file_path: str) -> List[Dict]:
        raise NotImplementedError("Corpus chunker not yet implemented")

def get_chunker(domain: str, **kwargs) -> BaseChunker:
    if domain == "finance":
        return FinanceChunker(**kwargs)
    elif domain == "economy":
        return EconomicsChunker(**kwargs)
    elif domain == "politics":
        return PoliticsChunker(**kwargs)
    elif domain == "corpus":
        return CorpusChunker(**kwargs)
    else:
        raise ValueError(f"Unknown domain: {domain}")


def main():
    parser = argparse.ArgumentParser(description='Ground truth chunks for RAG evaluation dataset')
    parser.add_argument('--input_dir', required=True, help='Directory containing input files')
    parser.add_argument('--output_dir', required=True, help='Directory to save output files')
    parser.add_argument('--domain', choices=['finance', 'economy', 'politics', 'corpus'], default='finance')
    parser.add_argument('--chunk_size', type=int, default=800)
    parser.add_argument('--chunk_overlap', type=int, default=100)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    chunker = get_chunker(
        args.domain,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.json'):
            print(f"Processing {filename}")
            file_path = os.path.join(args.input_dir, filename)
            try:
                document_id = os.path.splitext(filename)[0]
                
                if args.domain == "finance":
                    document = chunker.load_document(file_path)
                    document_id = document.get("cik", document_id)
                
                file_chunks = chunker.chunk_document(file_path)
                
                chunker.save_chunks(file_chunks, args.output_dir, args.domain, document_id)
                
                print(f"Processed {filename}: {len(file_chunks)} chunks")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()