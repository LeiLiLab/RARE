import os
import re
import argparse
import json
from typing import Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# Shared tokenizer for token-aware splitting where needed
_tokenizer = tiktoken.get_encoding("cl100k_base")

class BaseChunker:
    def __init__(self, domain: str, chunk_size: int = 800, chunk_overlap: int = 100):
        self.domain = domain
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=lambda text: len(_tokenizer.encode(text)),
            is_separator_regex=False,
        )

    def chunk_document(self, *args, **kwargs) -> List[Dict]:
        raise NotImplementedError("Subclasses must implement chunk_document")

class FinanceChunker(BaseChunker):
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        super().__init__(domain="finance", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _is_section_title(self, text: str) -> bool:
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
        return ('|' in text and (('-|-' in text) or ('--' in text and '|' in text) or (text.count('|') >= 3)))

    def _preprocess_document(self, item_7_text: str) -> List[str]:
        paragraphs = item_7_text.split("\n")
        segments: List[str] = []
        current_segment = ""
        table_buffer: List[str] = []
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
                table_context_start = max(0, i - 2)
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
                    elif not self._contains_table(para) and i + 1 < len(paragraphs) and '|' not in paragraphs[i + 1]:
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
        final_segments: List[str] = []
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

    def chunk_document(self, file_path: str, gics_path: Optional[str] = None) -> List[Dict]:
        document_series = pd.read_json(file_path, typ='series')
        document = document_series.to_dict()
        chunks: List[Dict] = []
        if "item_7" not in document:
            print(f"Warning: item_7 not found in document for CIK {document.get('cik', 'unknown')} ({os.path.basename(file_path)})")
            return []
        gics_sector = ""
        gics_subindustry = ""
        doc_cik = document.get("cik", "")
        if gics_path and os.path.exists(gics_path):
            try:
                sp500_info = pd.read_csv(gics_path)
                company_info = sp500_info[sp500_info['CIK'] == int(doc_cik)] if doc_cik.isdigit() else None
                if company_info is not None and not company_info.empty:
                    gics_sector = company_info['GICS_Sector'].values[0]
                    gics_subindustry = company_info['GICS_SubIndustry'].values[0]
            except Exception as e:
                print(f"GICS lookup failed for CIK {doc_cik}: {e}")
        metadata = {
            "cik": doc_cik,
            "company": document.get("company", ""),
            "filing_type": document.get("filing_type", ""),
            "filing_date": document.get("filing_date", ""),
            "period_of_report": document.get("period_of_report", ""),
            "GICS_Sector": gics_sector,
            "GICS_SubIndustry": gics_subindustry,
            "domain": self.domain,
        }
        segments = self._preprocess_document(document["item_7"])
        chunk_index = 0
        for segment in segments:
            lines = segment.split('\n')
            first_line = lines[0].strip() if lines else ""
            is_section_title = self._is_section_title(first_line)
            contains_table = self._contains_table(segment)
            chunk_id = f"{metadata['domain']}_{metadata.get('cik', 'unknown')}_{metadata.get('filing_date', 'unknown')}_chunk_{chunk_index}"
            if contains_table:
                chunk_metadata = metadata.copy()
                chunk_metadata["contains_table"] = True
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": segment,
                    "metadata": chunk_metadata,
                })
                chunk_index += 1
            else:
                recursive_chunks = self.text_splitter.split_text(segment)
                processed_chunks: List[str] = []
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
                    if not chunk_text.strip():
                        continue
                    if i == 0 and is_section_title and first_line not in chunk_text:
                        chunk_text = first_line + "\n\n" + chunk_text
                    chunk_metadata = metadata.copy()
                    chunk_metadata["contains_table"] = False
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text.strip(),
                        "metadata": chunk_metadata,
                    })
                    chunk_index += 1
        return chunks

def _html_tab_to_md(table: BeautifulSoup) -> str:
    markdown_table: List[str] = []
    rows = table.find_all('tr')
    if not rows:
        return ""
    header_cells = rows[0].find_all(['th'])
    if not header_cells:
        header_cells = rows[0].find_all('td')
    if not header_cells:
        return ""
    header = "| " + " | ".join([cell.get_text(strip=True) or " " for cell in header_cells]) + " |"
    markdown_table.append(header)
    separator = "| " + " | ".join(["---" for _ in range(len(header_cells))]) + " |"
    markdown_table.append(separator)
    start_row = 1 if header_cells else 0
    for row in rows[start_row:]:
        cells = row.find_all(['td', 'th'])
        if cells:
            data_row: List[str] = []
            for cell in cells:
                cell_text = cell.get_text(strip=True) or " "
                cell_text = cell_text.replace("|", "\\|")
                data_row.append(cell_text)
            row_text = "| " + " | ".join(data_row) + " |"
            markdown_table.append(row_text)
    return "\n".join(markdown_table)


def _convert_html_table_to_markdown(table_html: str) -> Optional[str]:
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    if table:
        return _html_tab_to_md(table)
    else:
        print("Row does not contain a valid table.")
        return None


def _econ_table_caption_detect(cur_table_chunk: str, row: pd.Series, idx: int, content: pd.DataFrame) -> str:
    if len(row.get('table_caption', [])) > 0 and 'Table' in row['table_caption'][0]:
        cur_table_chunk = row['table_caption'][0] + '\n' + cur_table_chunk
        if len(row.get('table_footnote', [])) > 0:
            cur_table_chunk += '\n'.join(row['table_footnote'])
    else:
        prev_r_idx = idx - 1
        while prev_r_idx >= 0:
            r_prev = content.iloc[prev_r_idx]
            if r_prev['type'] != 'text' or (r_prev['type'] == 'text' and len(r_prev.get('text', '')) > 100):
                break
            if r_prev['type'] == 'text':
                if 'Table' not in r_prev.get('text', ''):
                    prev_r_idx -= 1
                    continue
                else:
                    combined_caption = '\n'.join(content.iloc[prev_r_idx:idx]['text'].tolist())
                    cur_table_chunk = combined_caption + '\n' + cur_table_chunk
                    if len(row.get('table_footnote', [])) > 0:
                        cur_table_chunk += '\n'.join(row['table_footnote'])
                    break
    return cur_table_chunk


class EconomicsChunker(BaseChunker):
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        super().__init__(domain="economics", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk_document(self, doc_root_dir: str, file_id: str) -> List[Dict]:
        file_name = os.path.join(doc_root_dir, file_id, 'auto', f'{file_id}_content_list.json')
        if not os.path.exists(file_name):
            print(f"Missing economics content file: {file_name}")
            return []
        content = pd.read_json(file_name)
        first_valid_row = content[(content['type'] == 'text') & (content['text'].str.contains('OECD Economic Surveys:', na=False))]
        if not first_valid_row.empty:
            idx_found = first_valid_row.index[0]
            meta_text = content.iloc[idx_found]['text'].split(':')[1].strip()
            year_match = re.search(r'\d{4}', meta_text)
            year = year_match.group(0) if year_match else ""
            country = meta_text.replace(year, '').strip()
        else:
            print(f"No 'OECD Economic Surveys:' row for economics doc {file_id}")
            return []
        metadata_dict = {
            'domain': 'economics',
            'file_name': file_id,
            'file_type': 'OECD Economic Survey',
            'file_country': country,
            'file_year': year,
            'chunk_type': None,
            'chunk_page_idx': 0,
        }
        try:
            table_idx = int(content[content['type'] == 'table'].index[0])
        except Exception:
            table_idx = 0
        content = content.iloc[max(0, table_idx - 1):].reset_index(drop=True).copy()
        chunks: List[Dict] = []
        cur_text_chunk = ''
        for idx, row in content.iterrows():
            if row['type'] == 'text':
                if len((cur_text_chunk + row.get('text', '')).split()) < 600:
                    cur_text_chunk += row.get('text', '') + '\n'
                else:
                    chunk_id = f"economics_{metadata_dict['file_name'].replace(' ', '_')}_chunk_{len(chunks)}"
                    metadata_dict['chunk_type'] = 'text'
                    metadata_dict['chunk_page_idx'] = row.get('page_idx', 0)
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": cur_text_chunk.strip(),
                        "metadata": metadata_dict.copy(),
                    })
                    cur_text_chunk = row.get('text', '') + '\n'
            elif row['type'] == 'table':
                if pd.isna(row.get('table_body', None)):
                    continue
                cur_table_chunk = _convert_html_table_to_markdown(row['table_body'])
                if cur_table_chunk is None:
                    continue
                cur_table_chunk = _econ_table_caption_detect(cur_table_chunk, row, idx, content)
                chunk_id = f"economics_{metadata_dict['file_name'].replace(' ', '_')}_chunk_{len(chunks)}"
                metadata_dict['chunk_type'] = 'table'
                metadata_dict['chunk_page_idx'] = row.get('page_idx', 0)
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": cur_table_chunk.strip(),
                    "metadata": metadata_dict.copy(),
                })
            else:
                continue
        if cur_text_chunk.strip():
            chunk_id = f"economics_{metadata_dict['file_name'].replace(' ', '_')}_chunk_{len(chunks)}"
            metadata_dict['chunk_type'] = 'text'
            chunks.append({
                "chunk_id": chunk_id,
                "text": cur_text_chunk.strip(),
                "metadata": metadata_dict.copy(),
            })
        return chunks

class PolicyChunker(BaseChunker):
    def __init__(self, metadata_df: Optional[pd.DataFrame] = None, chunk_size: int = 600, chunk_overlap: int = 100):
        super().__init__(domain="policy", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.metadata_df = metadata_df
        if self.metadata_df is not None:
            self.metadata_df['id'] = self.metadata_df['id'].astype(str)

    def chunk_document(self, doc_root_dir: str, file_id: str) -> List[Dict]:
        file_name = os.path.join(doc_root_dir, file_id, 'auto', f'{file_id}_content_list.json')
        if not os.path.exists(file_name):
            print(f"Missing policy content file: {file_name}")
            return []
        if self.metadata_df is None or self.metadata_df.empty:
            print("Policy metadata is missing or empty; skipping.")
            return []
        content = pd.read_json(file_name)
        metadata_row = self.metadata_df[self.metadata_df['id'] == file_id]
        if metadata_row.empty:
            print(f"No metadata for policy doc {file_id}")
            return []
        metadata_row = metadata_row.iloc[0]
        metadata_dict = {
            'domain': 'policy',
            'file_name': metadata_row['id'],
            'file_type': metadata_row.get('planType', ''),
            'file_grantee': metadata_row.get('grantee', {}).get('granteeName', '') if isinstance(metadata_row.get('grantee', {}), dict) else '',
            'file_state': metadata_row.get('grantee', {}).get('state', {}).get('name', '') if isinstance(metadata_row.get('grantee', {}), dict) else '',
            'file_year': metadata_row.get('startYear', ''),
            'chunk_type': None,
            'chunk_page_idx': 0,
        }
        attachment_idx_list = content[content['type'] == 'text'].index[content.loc[content['type'] == 'text', 'text'].str.startswith('Attachment', na=False)].tolist()
        if attachment_idx_list:
            content = content.iloc[:attachment_idx_list[0]].reset_index(drop=True).copy()
        chunks: List[Dict] = []
        cur_text_chunk = ''
        for idx, row in content.iterrows():
            if row['type'] == 'text':
                if len((cur_text_chunk + row.get('text', '')).split()) < 600:
                    cur_text_chunk += row.get('text', '') + '\n'
                else:
                    chunk_id = f"policy_{metadata_dict['file_name'].replace(' ', '_')}_chunk_{len(chunks)}"
                    metadata_dict['chunk_type'] = 'text'
                    metadata_dict['chunk_page_idx'] = row.get('page_idx', 0)
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": cur_text_chunk.strip(),
                        "metadata": metadata_dict.copy(),
                    })
                    cur_text_chunk = row.get('text', '') + '\n'
            elif row['type'] == 'table':
                if pd.isna(row.get('table_body', None)):
                    continue
                cur_table_chunk = _convert_html_table_to_markdown(row['table_body'])
                if cur_table_chunk is None:
                    continue
                if len(row.get('table_caption', [])) > 0:
                    cur_table_chunk = '\n'.join(row['table_caption']) + '\n' + cur_table_chunk
                    if len(row.get('table_footnote', [])) > 0:
                        cur_table_chunk += '\n'.join(row['table_footnote'])
                chunk_id = f"policy_{metadata_dict['file_name'].replace(' ', '_')}_chunk_{len(chunks)}"
                metadata_dict['chunk_type'] = 'table'
                metadata_dict['chunk_page_idx'] = row.get('page_idx', 0)
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": cur_table_chunk.strip(),
                    "metadata": metadata_dict.copy(),
                })
            else:
                continue
        if cur_text_chunk.strip():
            chunk_id = f"policy_{metadata_dict['file_name'].replace(' ', '_')}_chunk_{len(chunks)}"
            metadata_dict['chunk_type'] = 'text'
            chunks.append({
                "chunk_id": chunk_id,
                "text": cur_text_chunk.strip(),
                "metadata": metadata_dict.copy(),
            })
        return chunks


def _load_policy_metadata(metadata_path: Optional[str]) -> Optional[pd.DataFrame]:
    if not metadata_path:
        return None
    if not os.path.exists(metadata_path):
        print(f"Policy metadata not found at {metadata_path}")
        return None
    try:
        return pd.read_json(metadata_path)
    except Exception as e:
        print(f"Failed to read policy metadata: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Unified chunking across finance, economics, and policy domains.')
    parser.add_argument('--finance_input_dir', type=str, default=None, help='Directory containing finance JSON filings')
    parser.add_argument('--finance_gics_path', type=str, default=None, help='Path to GICS CSV for finance')
    parser.add_argument('--economics_input_dir', type=str, default=None, help='Root directory containing economics docs (subdir/auto/{id}_content_list.json)')
    parser.add_argument('--policy_input_dir', type=str, default=None, help='Root directory containing policy docs (subdir/auto/{id}_content_list.json)')
    parser.add_argument('--policy_metadata_path', type=str, default=None, help='Path to policy metadata.json')
    parser.add_argument('--chunk_size', type=int, default=800, help='Chunk size for token-aware splitter (finance)')
    parser.add_argument('--chunk_overlap', type=int, default=100, help='Chunk overlap for token-aware splitter (finance)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output single JSON (chunking.json)')
    args = parser.parse_args()

    all_chunks: List[Dict] = []

    # Finance
    if args.finance_input_dir and os.path.isdir(args.finance_input_dir):
        finance_chunker = FinanceChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        for filename in os.listdir(args.finance_input_dir):
            if not filename.endswith('.json'):
                continue
            file_path = os.path.join(args.finance_input_dir, filename)
            print(f"[finance] Processing {filename}")
            try:
                file_chunks = finance_chunker.chunk_document(file_path=file_path, gics_path=args.finance_gics_path)
                all_chunks.extend(file_chunks)
                print(f"[finance] {filename}: +{len(file_chunks)} chunks (total={len(all_chunks)})")
            except Exception as e:
                print(f"[finance] Error processing {filename}: {e}")
    else:
        if args.finance_input_dir:
            print(f"[finance] Skipped: directory not found {args.finance_input_dir}")

    # Economics
    if args.economics_input_dir and os.path.isdir(args.economics_input_dir):
        econ_chunker = EconomicsChunker(chunk_size=600, chunk_overlap=100)
        for file_id in os.listdir(args.economics_input_dir):
            doc_dir = os.path.join(args.economics_input_dir, file_id)
            if not os.path.isdir(doc_dir):
                continue
            print(f"[economics] Processing {file_id}")
            try:
                file_chunks = econ_chunker.chunk_document(doc_root_dir=args.economics_input_dir, file_id=file_id)
                all_chunks.extend(file_chunks)
                print(f"[economics] {file_id}: +{len(file_chunks)} chunks (total={len(all_chunks)})")
            except Exception as e:
                print(f"[economics] Error processing {file_id}: {e}")
    else:
        if args.economics_input_dir:
            print(f"[economics] Skipped: directory not found {args.economics_input_dir}")

    # Policy
    policy_metadata_df = _load_policy_metadata(args.policy_metadata_path)
    if args.policy_input_dir and os.path.isdir(args.policy_input_dir):
        policy_chunker = PolicyChunker(metadata_df=policy_metadata_df, chunk_size=600, chunk_overlap=100)
        for file_id in os.listdir(args.policy_input_dir):
            doc_dir = os.path.join(args.policy_input_dir, file_id)
            if not os.path.isdir(doc_dir):
                continue
            print(f"[policy] Processing {file_id}")
            try:
                file_chunks = policy_chunker.chunk_document(doc_root_dir=args.policy_input_dir, file_id=file_id)
                all_chunks.extend(file_chunks)
                print(f"[policy] {file_id}: +{len(file_chunks)} chunks (total={len(all_chunks)})")
            except Exception as e:
                print(f"[policy] Error processing {file_id}: {e}")
    else:
        if args.policy_input_dir:
            print(f"[policy] Skipped: directory not found {args.policy_input_dir}")

    # Write single output
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    pd.DataFrame(all_chunks).to_json(args.output_file, orient='records')
    print(f"Wrote {len(all_chunks)} chunks to {args.output_file}")


if __name__ == "__main__":
    main() 